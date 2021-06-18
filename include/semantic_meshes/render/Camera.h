#pragma once

#include <template_tensors/TemplateTensors.h>

namespace semantic_meshes {

struct Camera
{
  dispatch::Union<
    template_tensors::geometry::projection::PinholeFC<double, template_tensors::Vector2d>,
    template_tensors::geometry::projection::PinholeFC<template_tensors::Vector2d, template_tensors::Vector2d>
  > intr; // TODO: float vs double here
  tt::geometry::transform::Rigid<float, 3> extr;
  tt::Vector2s resolution;
};

} // end of ns semantic_meshes
