#include <Fusion.h>

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
    .def("reset", &Aggregator::reset)
    .def("get", &Aggregator::get)
  ;

  constructors[name] = [](size_t primitives_num, float images_equal_weight){
    return boost::python::object(Aggregator(primitives_num, images_equal_weight));
  };
}

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
