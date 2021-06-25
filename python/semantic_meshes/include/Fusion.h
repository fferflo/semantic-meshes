#pragma once

#include <template_tensors/TemplateTensors.h>
#include <semantic_meshes/fusion/Mesh.h>
#include "Common.h"

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

  template <typename TDestTensor>
  struct TensorConstructor
  {
    template <typename TSrcTensor>
    TDestTensor operator()(TSrcTensor&& src)
    {
      return TDestTensor(src.dims()) = std::forward<TSrcTensor>(src);
    }
  };

  void add1(boost::python::object primitive_image_py, boost::python::object probs_image_py, boost::python::object weights_image_py)
  {
    tt::boost::python::without_gil guard;
    auto primitive_image = dispatch::get_result(TensorConstructor<tt::AllocTensorT<uint32_t, mem::alloc::host_heap, tt::RowMajor, 2>>(), FromPrimitiveImage(primitive_image_py));
    auto probs_image = dispatch::get_result(TensorConstructor<tt::AllocTensorT<float, mem::alloc::host_heap, tt::RowMajor, 3>>(), FromProbsImage(probs_image_py));
    auto weights_image = dispatch::get_result(TensorConstructor<tt::AllocTensorT<float, mem::alloc::host_heap, tt::RowMajor, 2>>(), FromWeightsImage(weights_image_py));
    m_aggregator.add(
      std::move(primitive_image),
      tt::partial<2>(std::move(probs_image)),
      std::move(weights_image)
    );
  }

  void add2(boost::python::object primitive_image_py, boost::python::object probs_image_py)
  {
    tt::boost::python::without_gil guard;
    auto primitive_image = dispatch::get_result(TensorConstructor<tt::AllocTensorT<uint32_t, mem::alloc::host_heap, tt::RowMajor, 2>>(), FromPrimitiveImage(primitive_image_py));
    auto probs_image = dispatch::get_result(TensorConstructor<tt::AllocTensorT<float, mem::alloc::host_heap, tt::RowMajor, 3>>(), FromProbsImage(probs_image_py));
    m_aggregator.add(
      std::move(primitive_image),
      tt::partial<2>(std::move(probs_image))
    );
  }

  void reset()
  {
    tt::boost::python::without_gil guard;
    m_aggregator.reset();
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
