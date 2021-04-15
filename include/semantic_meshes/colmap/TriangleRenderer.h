#pragma once

#include <template_tensors/TemplateTensors.h>

namespace semantic_meshes {

namespace colmap {

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

  TriangleRenderer(std::shared_ptr<Workspace> workspace)
    : m_workspace(workspace)
    , m_renderer(0)
    , m_vertices_d(workspace->getTinyplyVertices().size())
    , m_triangles_d(workspace->getTinyplyFaces().size())
  {
    // Construct model data
    tt::fromThrust(m_vertices_d) = workspace->getTinyplyVertices().data();
    tt::for_each(TriangleLambda(thrust::raw_pointer_cast(m_vertices_d.data())),
      tt::fromThrust(m_triangles_d), mem::toDevice(workspace->getTinyplyFaces().data()));

    // Construct renderer
    size_t max_pixels = 0;
    for (auto it = m_workspace->getCameras().begin(); it != m_workspace->getCameras().end(); ++it)
    {
      max_pixels = math::max(max_pixels, tt::prod(it->second.resolution));
    }
    m_renderer = decltype(m_renderer)(max_pixels);
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

  template <typename TImageId>
  tt::Vector2s getResolution(TImageId image_id) const
  {
    auto image_meta_data = m_workspace->getImageMetaData(image_id);
    return m_workspace->getCamera(image_meta_data.camera_id).resolution;
  }

  template <typename TImage, typename TImageId>
  void render(TImage&& image_d, TImageId image_id)
  {
    using Pixel = decltype(image_d());
    auto image_meta_data = m_workspace->getImageMetaData(image_id);
    auto& intr = m_workspace->getCamera(image_meta_data.camera_id).projection;
    tt::geometry::transform::Rigid<float, 3> extr = image_meta_data.transform;

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
      dispatch::id(extr),
      intr
    )(m_renderer);
  }

private:
  std::shared_ptr<Workspace> m_workspace;
  tt::geometry::render::DeviceMutexRasterizer<16 * 16> m_renderer;
  thrust::device_vector<tt::Vector3f> m_vertices_d;
  thrust::device_vector<tt::geometry::render::VertexIndexTriangle<float, int32_t>> m_triangles_d;
};

} // end of ns colmap

} // end of ns semantic_meshes
