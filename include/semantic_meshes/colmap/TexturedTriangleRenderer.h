#pragma once

#include <template_tensors/TemplateTensors.h>

namespace semantic_meshes {

namespace colmap {

template <typename TScalar, typename TIndexType, typename TResolutionType>
class TexturedTriangle : public tt::geometry::render::VertexIndexTriangle<TScalar, TIndexType>
{
public:
  __host__ __device__
  TexturedTriangle(tt::VectorXT<TScalar, 3>* vertices, tt::VectorXT<TIndexType, 3> indices, tt::VectorXT<TResolutionType, 2> resolution, TIndexType first_texel_index)
    : tt::geometry::render::VertexIndexTriangle<TScalar, TIndexType>(vertices, indices)
    , resolution(resolution)
    , first_texel_index(first_texel_index)
  {
  }

  __host__ __device__
  TexturedTriangle()
    : tt::geometry::render::VertexIndexTriangle<TScalar, TIndexType>()
    , resolution(0)
    , first_texel_index(0)
  {
  }

  __host__ __device__
  size_t getTexelIndex(tt::Vector3f barycentric_coords) const
  {
    tt::Vector2f uv =
        barycentric_coords(0) * tt::Vector2f(0, 0)
      + barycentric_coords(1) * tt::Vector2f(1, 0)
      + barycentric_coords(2) * tt::Vector2f(1, 1);
    tt::VectorXT<TResolutionType, 2> texel_coords = (uv - 1e-6) * resolution;
    TIndexType relative_texel_index = tt::SymmetricMatrixLowerTriangleRowMajor().toIndex(resolution, texel_coords);
    return first_texel_index + relative_texel_index;
  }

  __host__ __device__
  static size_t getTexelNum(tt::VectorXT<TResolutionType, 2> resolution)
  {
    return tt::SymmetricMatrixLowerTriangleRowMajor().getSize(resolution);
  }

private:
  tt::VectorXT<TResolutionType, 2> resolution;
  TIndexType first_texel_index;
};

class TexturedTriangleRenderer
{
public:
  using Triangle = TexturedTriangle<float, int32_t, int32_t>;

  struct TriangleLambda
  {
    tt::Vector3f* vertices_d_ptr;

    TriangleLambda(tt::Vector3f* vertices_d_ptr)
      : vertices_d_ptr(vertices_d_ptr)
    {
    }

    __device__
    void operator()(Triangle& triangle, const tt::Vector3i& face, const tt::Vector2i& resolution, uint32_t first_texel_index)
    {
      triangle = Triangle(vertices_d_ptr, face, resolution, first_texel_index);
    }
  };

  struct Project
  {
    tt::Vector3f point;

    template <typename TProjection>
    __host__ __device__
    tt::Vector2f operator()(TProjection&& projection) const
    {
      return projection.project(point);
    }
  };

  TexturedTriangleRenderer(std::shared_ptr<Workspace> workspace, float texels_per_pixel = 0.1)
    : m_workspace(workspace)
    , m_renderer(0)
    , m_vertices_d(workspace->getTinyplyVertices().size())
    , m_triangles_d(workspace->getTinyplyFaces().size())
  {
    thrust::host_vector<tt::Vector2ui> triangles_resolution(workspace->getTinyplyFaces().size());
    openmp::ForEach::for_each(iterator::counting_iterator<size_t>(0), iterator::counting_iterator<size_t>(workspace->getTinyplyFaces().size()),
      [&](size_t triangle_index){
        auto num_pixels_agg = aggregator::max<float>(0);
        auto& face = workspace->getTinyplyFaces()[triangle_index];
        auto get_vertex = [&](size_t vertex_index){
          return workspace->getTinyplyVertices()[face(vertex_index % 3)];
        };
        for (size_t image_index = 0; image_index < m_workspace->getImageNum(); image_index++)
        {
          // Project triangle to screen
          auto& image = m_workspace->getImageMetaData(image_index);
          auto& camera = workspace->getCamera(image.camera_id);
          tt::VectorXT<tt::Vector2f, 3> projected;
          bool in_front_of_camera = false;
          for (size_t vertex_index = 0; vertex_index < 3; vertex_index++)
          {
            tt::Vector3f vertex_c = image.transform.transformPoint(get_vertex(vertex_index));
            in_front_of_camera |= vertex_c(2) > 0;
            projected(vertex_index) = dispatch::get_result(Project{vertex_c}, camera.projection);
          }

          // Aggregate
          tt::Vector2i resolution = camera.resolution;
          float border = 0.5;
          if (in_front_of_camera
            && tt::all(-border * resolution <= projected(0) && projected(0) < (1 + border) * resolution)
            && tt::all(-border * resolution <= projected(1) && projected(1) < (1 + border) * resolution)
            && tt::all(-border * resolution <= projected(2) && projected(2) < (1 + border) * resolution)
          )
          {
            float area = 0.5 * math::abs(projected(0)(0) * (projected(1)(1) - projected(2)(1))
                                       + projected(1)(0) * (projected(2)(1) - projected(0)(1))
                                       + projected(2)(0) * (projected(0)(1) - projected(1)(1)));
            num_pixels_agg(area);
          }
        }
        triangles_resolution[triangle_index] = math::ceil(texels_per_pixel * math::sqrt(num_pixels_agg.get()));

        // Optimally order face indices
        float best_diff = math::consts<float>::INF;
        size_t best_00_vertex_index;
        for (size_t vertex_index = 0; vertex_index < 3; vertex_index++)
        {
          float angle = tt::angle(get_vertex(vertex_index + 1) - get_vertex(vertex_index), get_vertex(vertex_index + 2) - get_vertex(vertex_index));
          float diff = math::abs(angle - math::to_rad(90.0));
          if (diff < best_diff)
          {
            best_diff = diff;
            best_00_vertex_index = vertex_index;
          }
        }
        if (best_00_vertex_index != 0)
        {
          util::swap(face(0), face(best_00_vertex_index));
        }
    });

    // Prefix sum
    m_primitive_num = 0;
    size_t num_without_texels = 0;
    thrust::host_vector<uint32_t> triangles_first_texel_index(workspace->getTinyplyFaces().size());
    for (size_t triangle_index = 0; triangle_index < workspace->getTinyplyFaces().size(); triangle_index++)
    {
      triangles_first_texel_index[triangle_index] = m_primitive_num;
      size_t texel_num = Triangle::getTexelNum(triangles_resolution[triangle_index]);
      if (texel_num == 0)
      {
        num_without_texels++;
      }
      m_primitive_num += texel_num;
    }
    std::cout << "Got " << triangles_first_texel_index.size() << " triangles, " << m_primitive_num << " texels and " << num_without_texels << " triangles without texels" << std::endl;

    // Construct model data
    tt::fromThrust(m_vertices_d) = workspace->getTinyplyVertices().data();
    tt::for_each(
      TriangleLambda(thrust::raw_pointer_cast(m_vertices_d.data())),
      tt::fromThrust(m_triangles_d),
      mem::toDevice(workspace->getTinyplyFaces().data()),
      mem::toDevice(tt::fromThrust(triangles_resolution)),
      mem::toDevice(tt::fromThrust(triangles_first_texel_index))
    );

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
    return m_primitive_num;
  }

  struct Shader
  {
    template <typename TPixel, typename TIntersect>
    __host__ __device__
    void operator()(TPixel&& pixel, const Triangle& triangle, TIntersect&& intersect) const
    {
      pixel.primitive_index = triangle.getTexelIndex(intersect.barycentric_coords);
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
      dispatch::id(Shader()),
      dispatch::id(extr),
      intr
    )(m_renderer);
  }

private:
  std::shared_ptr<Workspace> m_workspace;
  tt::geometry::render::DeviceMutexRasterizer<16 * 16> m_renderer;
  thrust::device_vector<tt::Vector3f> m_vertices_d;
  thrust::device_vector<Triangle> m_triangles_d;
  size_t m_primitive_num;
};

} // end of ns colmap

} // end of ns semantic_meshes
