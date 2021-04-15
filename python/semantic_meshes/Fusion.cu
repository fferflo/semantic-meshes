#include <template_tensors/TemplateTensors.h>
#include <semantic_meshes/fusion/Mesh.h>
#include "Common.h"

template <typename TAnnotation, typename TAllocator>
class ModelRenderer
{
public:
  ModelRenderer(semantic_meshes::ModelRenderer<TAnnotation, TAllocator>&& renderer)
    : m_renderer(util::move(renderer))
  {
  }

  struct Render
  {
    ModelRenderer<TAnnotation, TAllocator>& self;
    boost::python::object rendered;

    template <typename TPrimitiveImage>
    void operator()(TPrimitiveImage&& primitive_image)
    {
      tt::AllocMatrixT<TAnnotation, mem::alloc::host_heap, tt::RowMajor> image(primitive_image.dims());
      self.m_renderer.render(image, util::forward<TPrimitiveImage>(primitive_image), TAnnotation(0));
      rendered = tt::boost::python::toNumpy(tt::total<2>(image));
    }
  };

  boost::python::object render(boost::python::object primitive_image)
  {
    tt::boost::python::without_gil guard;
    Render functor{*this};
    auto result = dispatch::all(FromPrimitiveImage(primitive_image))(functor);
    if (!result)
    {
      throw std::invalid_argument(result.error());
    }
    return functor.rendered;
  }

private:
  semantic_meshes::ModelRenderer<TAnnotation, TAllocator> m_renderer;
};

template <typename TAggregator, typename TAllocator>
class ModelAggregator
{
private:
  semantic_meshes::ModelAggregator<TAggregator, TAllocator> m_aggregator;

  using Annotation = typename semantic_meshes::ModelAggregator<TAggregator, TAllocator>::Annotation;

public:
  ModelAggregator(size_t num)
    : m_aggregator(num)
  {
  }

  ModelAggregator(size_t num, float images_equal_weight)
    : m_aggregator(num, images_equal_weight)
  {
  }

  struct Add
  {
    ModelAggregator<TAggregator, TAllocator>& self;

    template <typename TPrimitiveImage, typename TProbsImage, typename TWeightsImage>
    void operator()(TPrimitiveImage&& primitive_image, TProbsImage&& probs_image, TWeightsImage&& weights_image)
    {
      self.m_aggregator.add(
        util::forward<TPrimitiveImage>(primitive_image),
        tt::partial<2>(util::forward<TProbsImage>(probs_image)),
        util::forward<TWeightsImage>(weights_image)
      );
    }

    template <typename TPrimitiveImage, typename TProbsImage>
    void operator()(TPrimitiveImage&& primitive_image, TProbsImage&& probs_image)
    {
      self.m_aggregator.add(
        util::forward<TPrimitiveImage>(primitive_image),
        tt::partial<2>(util::forward<TProbsImage>(probs_image))
      );
    }
  };

  void add1(boost::python::object primitive_image, boost::python::object probs_image, boost::python::object weights_image)
  {
    tt::boost::python::without_gil guard;
    auto result = dispatch::all(FromPrimitiveImage(primitive_image), FromProbsImage(probs_image), FromWeightsImage(weights_image))(Add{*this});
    if (!result)
    {
      throw std::invalid_argument(result.error());
    }
  }

  void add2(boost::python::object primitive_image, boost::python::object probs_image)
  {
    tt::boost::python::without_gil guard;
    auto result = dispatch::all(FromPrimitiveImage(primitive_image), FromProbsImage(probs_image))(Add{*this});
    if (!result)
    {
      throw std::invalid_argument(result.error());
    }
  }

  void reset()
  {
    tt::boost::python::without_gil guard;
    m_aggregator.reset();
  }

  ModelRenderer<Annotation, TAllocator> renderer()
  {
    tt::boost::python::without_gil guard;
    return ModelRenderer<Annotation, TAllocator>(m_aggregator.renderer());
  }

  ::boost::python::object get()
  {
    tt::boost::python::without_gil guard;
    return tt::boost::python::toNumpy(tt::total<1>(m_aggregator.get()));
  }
};



std::map<std::string, std::function<boost::python::object(size_t, float)>> constructors;

template <typename TAllocator, typename TAggregator>
void registerMeshAggregator(std::string name, TAggregator aggregator)
{
  using Aggregator = ModelAggregator<
    typename std::decay<TAggregator>::type,
    TAllocator
  >;

  name = std::string("MeshAggregator") + name;

  boost::python::class_<Aggregator>(name.c_str(), boost::python::init<size_t>())
    .def(boost::python::init<size_t, float>())
    .def("add", &Aggregator::add1)
    .def("add", &Aggregator::add2)
    .def("renderer", &Aggregator::renderer)
    .def("reset", &Aggregator::reset)
    .def("get", &Aggregator::get)
  ;

  constructors[name] = [](size_t primitives_num, float images_equal_weight){
    return boost::python::object(Aggregator(primitives_num, images_equal_weight));
  };
}

struct nan_and_inf_to_zero
{
  template <typename T, typename TDecay = typename std::decay<T>::type>
  TDecay operator()(T&& t) const volatile
  {
    return (math::isnan(t) || math::isinf(t)) ? static_cast<TDecay>(0) : static_cast<TDecay>(util::forward<T>(t));
  }
};

struct nan_and_inf_to_zero_elwise
{
  template <typename T, size_t TRows = tt::rows_v<T>::value, typename TElementType = tt::decay_elementtype_t<T>>
  tt::VectorXT<TElementType, TRows> operator()(T&& t) const volatile
  {
    return tt::elwise(nan_and_inf_to_zero(), util::forward<T>(t));
  }
};

struct logprob_normalize
{
  template <typename T, size_t TRows = tt::rows_v<T>::value>
  tt::VectorXT<float, TRows> operator()(T&& p) const volatile
  {
    return tt::VectorXT<float, TRows>(tt::static_cast_to<float>(p / tt::max_el(p)));
  }
};


template <size_t... TClassesNums>
struct RegisterClasses;

template <>
struct RegisterClasses<>
{
  void operator()()
  {
  }
};

template <size_t TClassesNum, size_t... TClassesNumRest>
struct RegisterClasses<TClassesNum, TClassesNumRest...>
{
  void operator()()
  {
    using Renderer = ModelRenderer<tt::VectorXT<float, TClassesNum>, mem::alloc::host_heap>;
    boost::python::class_<Renderer>((std::string("MeshRenderer") + util::to_string(TClassesNum)).c_str(), boost::python::no_init)
      .def("render", &Renderer::render)
    ;

    // TODO: differentiate on parallel and sequential here

    registerMeshAggregator<mem::alloc::host_heap>(std::string("Sum") + util::to_string(TClassesNum),
      aggregator::map_output(nan_and_inf_to_zero_elwise(),
        aggregator::map_output(tt::functor::normalize<tt::functor::l1_norm>(),
          aggregator::map_output(atomic::functor::load(),
            aggregator::weighted::sum<
              atomic::Variable<tt::VectorXT<float, TClassesNum>, atomic::op::Lock<std::mutex>>
            >()
          )
        )
      )
    );

    registerMeshAggregator<mem::alloc::host_heap>(std::string("Mul") + util::to_string(TClassesNum),
      aggregator::map_output(nan_and_inf_to_zero_elwise(),
        aggregator::map_output(tt::functor::normalize<tt::functor::l1_norm>(),
          aggregator::map_output(logprob_normalize(),//tt::functor::static_cast_to<float>(),
            aggregator::map_output(atomic::functor::load(),
              aggregator::map_input(tt::functor::pow(),
                aggregator::prod<
                  atomic::Variable<tt::VectorXT<numeric::LogProb<float>, TClassesNum>, atomic::op::Lock<std::mutex>>
                >()
              )
            )
          )
        )
      )
    );

    RegisterClasses<TClassesNumRest...>()();
  }
};

boost::python::object construct1(size_t primitives_num, size_t classes_num, std::string aggregator, float images_equal_weight)
{
  aggregator[0] = std::toupper(aggregator[0]);
  return constructors[(std::string("MeshAggregator") + aggregator + util::to_string(classes_num)).c_str()](primitives_num, images_equal_weight);
}

boost::python::object construct2(size_t primitives_num, size_t classes_num, std::string aggregator)
{
  return construct1(primitives_num, classes_num, aggregator, 0.5);
}

boost::python::object construct3(size_t primitives_num, size_t classes_num)
{
  return construct2(primitives_num, classes_num, "sum");
}

BOOST_PYTHON_MODULE(fusion)
{
  Py_Initialize();
  boost::python::numpy::initialize();

  // CLASSES_NUMS is passed as boost preprocessor sequence from cmake to c++ since cmake compile-definitions does not allow ',' character => unpack sequence here
  RegisterClasses<BOOST_PP_SEQ_ENUM(CLASSES_NUMS)>()();

  boost::python::def("MeshAggregator", construct1, boost::python::args("primitives", "classes", "aggregator", "images_equal_weight"));
  boost::python::def("MeshAggregator", construct2, boost::python::args("primitives", "classes", "aggregator"));
  boost::python::def("MeshAggregator", construct3, boost::python::args("primitives", "classes"));
};
