#pragma once

#include <template_tensors/TemplateTensors.h>
#include <Camera.h>
#include "Common.h"
#include <semantic_meshes/data/Colmap.h>
#include <Ply.h>
#include <semantic_meshes/render/TriangleRenderer.h>
#include <semantic_meshes/render/TexturedTriangleRenderer.h>

template <typename TThisType>
struct Renderer
{
  size_t getPrimitivesNum() const
  {
    return static_cast<const TThisType*>(this)->renderer->getPrimitivesNum();
  }

  struct Pixel
  {
    float z;
    uint32_t primitive_index;
  };

  boost::python::object render(Camera camera)
  {
    tt::boost::python::without_gil guard;

    tt::AllocMatrixT<Pixel, mem::alloc::device, tt::RowMajor> image_d(camera.camera.resolution);
    static_cast<TThisType*>(this)->renderer->render(image_d, camera.camera);

    tt::AllocMatrixT<uint32_t, mem::alloc::device, tt::RowMajor> indices_d(image_d.dims());
    indices_d = TT_ELWISE_MEMBER(image_d, primitive_index);
    tt::AllocMatrixT<float, mem::alloc::device, tt::RowMajor> depth_d(image_d.dims());
    depth_d = TT_ELWISE_MEMBER(image_d, z);

    boost::python::object indices = tt::boost::python::fromDlPack(tt::toDlPack(indices_d), "dltensor");
    boost::python::object depth = tt::boost::python::fromDlPack(tt::toDlPack(depth_d), "dltensor");
    {
      tt::boost::python::with_gil guard;
      return ::boost::python::make_tuple(indices, depth);
    }
  }
};
