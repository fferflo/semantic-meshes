#include <template_tensors/TemplateTensors.h>
#include <semantic_meshes/colmap/Workspace.h>
#include <semantic_meshes/colmap/TriangleRenderer.h>
#include <semantic_meshes/colmap/TexturedTriangleRenderer.h>
#include "Common.h"

template <typename TThisType>
struct ColmapMesh
{
  ColmapMesh(boost::filesystem::path workspace_path, boost::filesystem::path ply_file)
    : workspace(std::make_shared<semantic_meshes::colmap::Workspace>(workspace_path, ply_file))
  {
  }

  size_t getPrimitivesNum() const
  {
    return static_cast<const TThisType*>(this)->renderer.getPrimitivesNum();
  }

  struct Pixel
  {
    float z;
    uint32_t primitive_index;
  };

  boost::python::object render(std::string image_path)
  {
    tt::boost::python::without_gil guard;
    size_t image_index = workspace->getImageIndex(image_path);
    tt::AllocMatrixT<Pixel, mem::alloc::device, tt::RowMajor> image_d(static_cast<TThisType*>(this)->renderer.getResolution(image_index));
    static_cast<TThisType*>(this)->renderer.render(image_d, image_index);

    tt::AllocMatrixT<uint32_t, mem::alloc::device, tt::RowMajor> indices_d(image_d.dims());
    indices_d = TENSOR_ELWISE_MEMBER(image_d, primitive_index);
    tt::AllocMatrixT<float, mem::alloc::device, tt::RowMajor> depth_d(image_d.dims());
    depth_d = TENSOR_ELWISE_MEMBER(image_d, z);

    boost::python::object indices = tt::boost::python::fromDlPack(tt::toDlPack(indices_d), "dltensor");
    boost::python::object depth = tt::boost::python::fromDlPack(tt::toDlPack(depth_d), "dltensor");
    {
      tt::boost::python::with_gil guard;
      return ::boost::python::make_tuple(indices, depth);
    }
  }

  std::shared_ptr<semantic_meshes::colmap::Workspace> workspace;
};

struct ColmapTexelMesh : ColmapMesh<ColmapTexelMesh>
{
  ColmapTexelMesh(std::string workspace_path, std::string ply_file, float texels_per_pixel)
    : ColmapMesh(workspace_path, ply_file)
    , renderer(this->workspace, texels_per_pixel)
  {
  }

  ColmapTexelMesh(std::string workspace_path, std::string ply_file)
    : ColmapMesh(workspace_path, ply_file)
    , renderer(this->workspace)
  {
  }

  semantic_meshes::colmap::TexturedTriangleRenderer renderer;
};

struct ColmapTriangleMesh : ColmapMesh<ColmapTriangleMesh>
{
  ColmapTriangleMesh(std::string workspace_path, std::string ply_file)
    : ColmapMesh(workspace_path, ply_file)
    , renderer(this->workspace)
  {
  }

  struct Save
  {
    ColmapMesh& self;

    template <typename TAnnotationColors>
    void operator()(std::string path, TAnnotationColors&& annotation_colors_in, bool bin)
    {
      auto annotation_colors = tt::eval<tt::ColMajor, mem::alloc::host_heap>
        (tt::static_cast_to<tt::VectorXT<uint8_t, 3>>(tt::partial<1>(util::forward<TAnnotationColors>(annotation_colors_in))));

      auto colors_red = mem::toHost(tt::eval(tt::elwise([](tt::VectorXT<uint8_t, 3> color){return color(0);}, annotation_colors)));
      auto colors_green = mem::toHost(tt::eval(tt::elwise([](tt::VectorXT<uint8_t, 3> color){return color(1);}, annotation_colors)));
      auto colors_blue = mem::toHost(tt::eval(tt::elwise([](tt::VectorXT<uint8_t, 3> color){return color(2);}, annotation_colors)));

      tt::tinyply::WriteProperty<uint8_t, 1> tinyply_faces_red("face", std::vector<std::string>{"red"}, colors_red);
      tt::tinyply::WriteProperty<uint8_t, 1> tinyply_faces_green("face", std::vector<std::string>{"green"}, colors_green);
      tt::tinyply::WriteProperty<uint8_t, 1> tinyply_faces_blue("face", std::vector<std::string>{"blue"}, colors_blue);
      tt::tinyply::write(path, bin, self.workspace->getTinyplyVertices(), self.workspace->getTinyplyFaces(), tinyply_faces_red, tinyply_faces_green, tinyply_faces_blue);
    }
  };

  void save1(std::string path, boost::python::object annotation_colors, bool bin)
  {
    tt::boost::python::without_gil guard;
    auto result = dispatch::all(dispatch::id(path), FromClassColors<2>(annotation_colors), dispatch::id(bin))(Save{*this});
    if (!result)
    {
      throw std::invalid_argument(result.error());
    }
  }

  void save2(std::string path, boost::python::object annotation_colors)
  {
    save1(path, annotation_colors, true);
  }

  semantic_meshes::colmap::TriangleRenderer renderer;
};

BOOST_PYTHON_MODULE(colmap)
{
  Py_Initialize();
  boost::python::numpy::initialize();

  boost::python::class_<ColmapTexelMesh>("ColmapTexelMesh", boost::python::init<std::string, std::string>())
    .def(boost::python::init<std::string, std::string, float>())
    .def("getPrimitivesNum", &ColmapTexelMesh::getPrimitivesNum)
    .def("render", &ColmapTexelMesh::render)
  ;

  boost::python::class_<ColmapTriangleMesh>("ColmapTriangleMesh", boost::python::init<std::string, std::string>())
    .def("getPrimitivesNum", &ColmapTriangleMesh::getPrimitivesNum)
    .def("render", &ColmapTriangleMesh::render)
    .def("save", &ColmapTriangleMesh::save1)
    .def("save", &ColmapTriangleMesh::save2)
  ;
};
