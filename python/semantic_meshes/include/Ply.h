#pragma once

#include <semantic_meshes/data/Ply.h>
#include <Colmap.h>
#include "Common.h"
#include <Renderer.h>
#include <semantic_meshes/render/TriangleRenderer.h>
#include <semantic_meshes/render/TexturedTriangleRenderer.h>

struct Ply
{
  Ply(std::string ply_file)
    : ply(std::make_shared<semantic_meshes::data::Ply>(ply_file))
  {
  }

  struct Save
  {
    Ply& self;

    template <typename TAnnotationColors>
    void operator()(std::string path, TAnnotationColors&& annotation_colors_in, bool bin)
    {
      auto annotation_colors = tt::eval<tt::ColMajor, mem::alloc::host_heap>
        (tt::static_cast_to<tt::VectorXT<uint8_t, 3>>(tt::partial<1>(std::forward<TAnnotationColors>(annotation_colors_in))));

      auto colors_red = mem::toHost(tt::eval(tt::elwise([](tt::VectorXT<uint8_t, 3> color){return color(0);}, annotation_colors)));
      auto colors_green = mem::toHost(tt::eval(tt::elwise([](tt::VectorXT<uint8_t, 3> color){return color(1);}, annotation_colors)));
      auto colors_blue = mem::toHost(tt::eval(tt::elwise([](tt::VectorXT<uint8_t, 3> color){return color(2);}, annotation_colors)));

      tt::tinyply::WriteProperty<uint8_t, 1> tinyply_faces_red("face", std::vector<std::string>{"red"}, colors_red);
      tt::tinyply::WriteProperty<uint8_t, 1> tinyply_faces_green("face", std::vector<std::string>{"green"}, colors_green);
      tt::tinyply::WriteProperty<uint8_t, 1> tinyply_faces_blue("face", std::vector<std::string>{"blue"}, colors_blue);
      tt::tinyply::write(path, bin, self.ply->getTinyplyVertices(), self.ply->getTinyplyFaces(), tinyply_faces_red, tinyply_faces_green, tinyply_faces_blue);
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

  std::shared_ptr<semantic_meshes::data::Ply> ply;
};

struct PlyRendererTexels : Renderer<PlyRendererTexels>
{
  PlyRendererTexels(Ply ply, Colmap colmap, float texels_per_pixel)
    : renderer(std::make_shared<semantic_meshes::render::TexturedTriangleRenderer>(ply.ply, colmap.colmap->getCameras(), texels_per_pixel))
  {
  }

  PlyRendererTexels(Ply ply, Colmap colmap)
    : renderer(std::make_shared<semantic_meshes::render::TexturedTriangleRenderer>(ply.ply, colmap.colmap->getCameras()))
  {
  }

  std::shared_ptr<semantic_meshes::render::TexturedTriangleRenderer> renderer;
};

struct PlyRendererTriangles : Renderer<PlyRendererTriangles>
{
  PlyRendererTriangles(Ply ply, Colmap colmap)
    : renderer(std::make_shared<semantic_meshes::render::TriangleRenderer>(ply.ply, colmap.colmap->getCameras()))
  {
  }

  std::shared_ptr<semantic_meshes::render::TriangleRenderer> renderer;
};

PlyRendererTexels renderer_texels_ply1(Ply ply, Colmap colmap)
{
  return PlyRendererTexels(ply, colmap);
}

PlyRendererTexels renderer_texels_ply2(Ply ply, Colmap colmap, float texels_per_pixel)
{
  return PlyRendererTexels(ply, colmap, texels_per_pixel);
}

PlyRendererTriangles renderer_triangles_ply(Ply ply, Colmap colmap)
{
  return PlyRendererTriangles(ply, colmap);
}
