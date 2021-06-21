#pragma once

#include <template_tensors/TemplateTensors.h>
#include <semantic_meshes/data/Ply.h>
#include <semantic_meshes/render/Camera.h>

namespace semantic_meshes {

namespace render {

class TriangleRenderer
{
public:
  struct TriangleLambda
  {
    tt::Vector3f* vertices_d_ptr;

    TriangleLambda(tt::Vector3f* vertices_d_ptr)
      : vertices_d_ptr(vertices_d_ptr)
    {
    }

    __device__
    void operator()(tt::geometry::render::VertexIndexTriangle<float, int32_t>& triangle, const tt::Vector3i& face)
    {
      triangle = tt::geometry::render::VertexIndexTriangle<float, int32_t>(vertices_d_ptr, face);
    }
  };

  TriangleRenderer(std::shared_ptr<data::Ply> ply)
    : m_renderer(0)
    , m_vertices_d(ply->getTinyplyVertices().size())
    , m_triangles_d(ply->getTinyplyFaces().size())
  {
    // Construct model data
    tt::fromThrust(m_vertices_d) = ply->getTinyplyVertices().data();
    tt::for_each(TriangleLambda(thrust::raw_pointer_cast(m_vertices_d.data())),
      tt::fromThrust(m_triangles_d), mem::toDevice(ply->getTinyplyFaces().data()));
  }

  size_t getPrimitivesNum() const
  {
    return m_triangles_d.size();
  }

  struct Shader
  {
    tt::geometry::render::VertexIndexTriangle<float, int32_t>* begin;

    Shader(tt::geometry::render::VertexIndexTriangle<float, int32_t>* begin)
      : begin(begin)
    {
    }

    template <typename TPixel, typename TIntersect>
    __host__ __device__
    void operator()(TPixel&& pixel, const tt::geometry::render::VertexIndexTriangle<float, int32_t>& primitive, TIntersect&& intersect) const
    {
      pixel.primitive_index = &primitive - begin;
    }
  };

  template <typename TImage>
  void render(TImage&& image_d, Camera camera)
  {
    using Pixel = decltype(image_d());

    size_t required_size = tt::prod(camera.resolution);
    if (required_size > m_renderer.getMutexMemory()->size())
    {
      m_renderer = decltype(m_renderer)(required_size);
    }

    // Reset z buffer
    tt::for_each([]__device__(Pixel& p){
      p.z = math::consts<float>::INF;
      p.primitive_index = static_cast<decltype(p.primitive_index)>(-1);
    }, image_d);

    // Render
    dispatch::all(
      dispatch::id(image_d),
      dispatch::id(m_triangles_d.begin()),
      dispatch::id(m_triangles_d.end()),
      dispatch::id(Shader(thrust::raw_pointer_cast(m_triangles_d.data()))),
      dispatch::id(camera.extr),
      camera.intr
    )(m_renderer);
  }

private:
  tt::geometry::render::DeviceMutexRasterizer<16 * 16> m_renderer;
  thrust::device_vector<tt::Vector3f> m_vertices_d;
  thrust::device_vector<tt::geometry::render::VertexIndexTriangle<float, int32_t>> m_triangles_d;
};

} // end of ns render

} // end of ns semantic_meshes
