#pragma once

#include <template_tensors/TemplateTensors.h>

namespace semantic_meshes {

template <typename TAggregator, typename TAllocator>
class ModelAggregator;

template <typename TAnnotation, typename TAllocator>
class ModelRenderer
{
private:
  tt::AllocVectorT<TAnnotation, TAllocator> m_annotations;

  ModelRenderer(size_t num)
    : m_annotations(num)
  {
  }

  template <typename TAggregator2, typename TAllocator2>
  friend class ModelAggregator;

public:
  template <typename TDestImage, typename TPrimitiveImage, typename TPixel>
  void render(TDestImage&& dest_image, TPrimitiveImage&& primitives_image_in, TPixel background)
  {
    static_assert(mem::isOnHost<mem::memorytype_v<TDestImage>::value>(), "Destination image must be on host");
    DECLTYPE_AUTO(primitives_image, mem::toHost(std::forward<TPrimitiveImage>(primitives_image_in)));
    using DestPixel = decltype(dest_image());
    tt::op::LocalArrayForEach<openmp::ForEach>::for_each<2>([&](tt::Vector2s pos, DestPixel p){
      size_t primitive_index = primitives_image(pos);
      if (primitive_index < m_annotations.rows())
      {
        p = m_annotations(primitive_index);
      }
      else
      {
        p = background;
      }
    }, dest_image);
  }
};

template <typename TAggregator, typename TAllocator>
class ModelAggregator
{
private:
  tt::AllocVectorT<TAggregator, TAllocator> m_annotations;
  TAggregator m_zero;
  float m_images_equal_weight;

public:
  static const size_t CLASSES_NUM = tt::rows_v<decltype(m_zero.get())>::value;
  using Annotation = tt::VectorXT<float, CLASSES_NUM>;

  ModelAggregator(size_t num, float images_equal_weight = 0.5, TAggregator aggregator = TAggregator())
    : m_annotations(num)
    , m_zero(aggregator)
    , m_images_equal_weight(images_equal_weight)
  {
    tt::fill(m_annotations, m_zero);
  }

  template <typename TPrimitiveImage, typename TProbsImage, typename TWeightsImage>
  void add(TPrimitiveImage&& primitives_image_in, TProbsImage&& probs_image_in, TWeightsImage&& weights_image_in)
  {
    if (!tt::areSameDimensions(primitives_image_in.dims(), probs_image_in.dims()) || !tt::areSameDimensions(primitives_image_in.dims(), weights_image_in.dims()))
    {
      throw std::invalid_argument("Primitive image " + util::to_string(primitives_image_in.dims())
        + ", probs image " + util::to_string(probs_image_in.dims())
        + " and weights image " + util::to_string(weights_image_in.dims())
        + " must have the same width and height");
    }
    DECLTYPE_AUTO(primitives_image,
      tt::eval<tt::RowMajor, mem::alloc::host_heap>(
        mem::toHost(std::forward<TPrimitiveImage>(primitives_image_in))
      )
    );
    DECLTYPE_AUTO(probs_image,
      //tt::eval<tt::RowMajor, mem::alloc::host_heap>(
        tt::static_cast_to<tt::VectorXT<float, CLASSES_NUM>>(mem::toHost(std::forward<TProbsImage>(probs_image_in)))
      //)
    );
    DECLTYPE_AUTO(weights_image,
      tt::eval<tt::RowMajor, mem::alloc::host_heap>(
        mem::toHost(std::forward<TWeightsImage>(weights_image_in))
      )
    );

    std::map<size_t, size_t> pixels_per_face;
    tt::op::LocalForEach::for_each([&](size_t primitive_index){
      pixels_per_face.insert({primitive_index, 0}).first->second += 1;
    }, primitives_image); // TODO: this could be precomputed as a pixel weight map per frame
    tt::op::LocalArrayForEach<openmp::ForEach>::for_each<2>([&](tt::Vector2s pos, size_t primitive_index, float weight){
      if (primitive_index < m_annotations.rows())
      {
        auto next = probs_image(pos);
        if (tt::sum(next) > 0.5) // Not the don't-care class
        {
          float image_weight = 1.0f / ((float) pixels_per_face[primitive_index]);
          float pixel_weight = 1.0f;
          float image_pixel_weight = m_images_equal_weight * image_weight + (1 - m_images_equal_weight) * pixel_weight;
          m_annotations(primitive_index)(next, image_pixel_weight * weight);
        }
      }
    }, primitives_image, weights_image);
  }

  template <typename TPrimitiveImage, typename TProbsImage>
  void add(TPrimitiveImage&& primitives_image_in, TProbsImage&& probs_image_in)
  {
    add(
      std::forward<TPrimitiveImage>(primitives_image_in),
      std::forward<TProbsImage>(probs_image_in),
      tt::broadcast<tt::dimseq_t<TPrimitiveImage>>(tt::singleton(1.0f), primitives_image_in.dims())
    );
  }

  void reset()
  {
    tt::fill(m_annotations, m_zero);
  }

  ModelRenderer<Annotation, TAllocator> renderer()
  {
    ModelRenderer<Annotation, TAllocator> result(m_annotations.rows());
    result.m_annotations = tt::elwise(util::functor::get(), m_annotations);
    return result;
  }

  auto get() const
  RETURN_AUTO(tt::elwise(util::functor::get(), m_annotations))
};

} // end of ns semantic_meshes
