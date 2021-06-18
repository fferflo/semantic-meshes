#pragma once

#include <template_tensors/TemplateTensors.h>
#include <semantic_meshes/fusion/Mesh.h>
#include "Common.h"

template <typename TAnnotation, typename TAllocator>
class ModelRenderer
{
public:
  ModelRenderer(semantic_meshes::ModelRenderer<TAnnotation, TAllocator>&& renderer)
    : m_renderer(std::move(renderer))
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
      self.m_renderer.render(image, std::forward<TPrimitiveImage>(primitive_image), TAnnotation(0));
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
        std::forward<TPrimitiveImage>(primitive_image),
        tt::partial<2>(std::forward<TProbsImage>(probs_image)),
        std::forward<TWeightsImage>(weights_image)
      );
    }

    template <typename TPrimitiveImage, typename TProbsImage>
    void operator()(TPrimitiveImage&& primitive_image, TProbsImage&& probs_image)
    {
      self.m_aggregator.add(
        std::forward<TPrimitiveImage>(primitive_image),
        tt::partial<2>(std::forward<TProbsImage>(probs_image))
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

struct nan_and_inf_to_zero
{
  template <typename T, typename TDecay = typename std::decay<T>::type>
  TDecay operator()(T&& t) const volatile
  {
    return (math::isnan(t) || math::isinf(t)) ? static_cast<TDecay>(0) : static_cast<TDecay>(std::forward<T>(t));
  }
};

struct nan_and_inf_to_zero_elwise
{
  template <typename T, size_t TRows = tt::rows_v<T>::value, typename TElementType = tt::decay_elementtype_t<T>>
  tt::VectorXT<TElementType, TRows> operator()(T&& t) const volatile
  {
    return tt::elwise(nan_and_inf_to_zero(), std::forward<T>(t));
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
