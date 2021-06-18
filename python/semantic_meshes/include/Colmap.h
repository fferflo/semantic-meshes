#pragma once

#include <template_tensors/TemplateTensors.h>
#include <semantic_meshes/data/Colmap.h>
#include <Camera.h>
#include "Common.h"

struct Colmap
{
  Colmap(std::string workspace_path)
    : colmap(std::make_shared<semantic_meshes::data::Colmap>(workspace_path))
  {
  }

  Camera getCamera1(size_t index) const
  {
    return Camera{colmap->getCamera(index)};
  }

  Camera getCamera2(std::string path) const
  {
    return Camera{colmap->getCamera(path)};
  }

  std::shared_ptr<semantic_meshes::data::Colmap> colmap;
};
