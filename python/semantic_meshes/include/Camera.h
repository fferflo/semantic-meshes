#pragma once

#include <template_tensors/TemplateTensors.h>
#include <semantic_meshes/render/Camera.h>
#include "Common.h"

struct Camera
{
  semantic_meshes::Camera camera;

  Camera(semantic_meshes::Camera camera)
    : camera(camera)
  {
  }

  Camera(boost::python::object rotation_py, boost::python::object translation_py, boost::python::object resolution_py, boost::python::object focal_lengths_py,
    boost::python::object principal_point_py)
  {
    tt::Matrix3f rotation;
    tt::boost::python::dispatch::FromTensor<
      metal::list<float, double>,
      metal::numbers<2>,
      metal::numbers<mem::HOST>
    >(rotation_py)(util::functor::assign_to(rotation));

    tt::Vector3f translation;
    tt::boost::python::dispatch::FromTensor<
      metal::list<float, double>,
      metal::numbers<1>,
      metal::numbers<mem::HOST>
    >(translation_py)(util::functor::assign_to(translation));

    tt::Vector2s resolution;
    tt::boost::python::dispatch::FromTensor<
      metal::list<int32_t, uint32_t, int64_t, uint64_t>,
      metal::numbers<1>,
      metal::numbers<mem::HOST>
    >(resolution_py)(util::functor::assign_to(resolution));

    tt::Vector2f focal_lengths;
    tt::boost::python::dispatch::FromTensor<
      metal::list<float, double>,
      metal::numbers<1>,
      metal::numbers<mem::HOST>
    >(focal_lengths_py)(util::functor::assign_to(focal_lengths));

    tt::Vector2f principal_point;
    tt::boost::python::dispatch::FromTensor<
      metal::list<float, double>,
      metal::numbers<1>,
      metal::numbers<mem::HOST>
    >(principal_point_py)(util::functor::assign_to(principal_point));

    camera.intr = tt::geometry::projection::PinholeFC<template_tensors::Vector2d, template_tensors::Vector2d>(focal_lengths, principal_point);
    camera.extr = tt::geometry::transform::Rigid<float, 3>(rotation, translation);
    camera.resolution = resolution;
  }
};
