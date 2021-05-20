#pragma once

#include <template_tensors/TemplateTensors.h>

auto FromPrimitiveImage(::boost::python::object object)
RETURN_AUTO(
  tt::boost::python::dispatch::FromTensor<
    metal::list<uint32_t, int32_t, uint64_t, int64_t>,
    metal::numbers<2>,
    metal::numbers<mem::HOST, mem::DEVICE>
  >(object)
)

auto FromProbsImage(::boost::python::object object)
RETURN_AUTO(
  tt::boost::python::dispatch::FromTensor<
    metal::list<float>,
    metal::numbers<3>,
    metal::numbers<mem::HOST, mem::DEVICE>
  >(object)
)

auto FromWeightsImage(::boost::python::object object)
RETURN_AUTO(
  tt::boost::python::dispatch::FromTensor<
    metal::list<float>,
    metal::numbers<2>,
    metal::numbers<mem::HOST, mem::DEVICE>
  >(object)
)

template <size_t TRank>
auto FromClassColors(::boost::python::object object)
RETURN_AUTO(
  tt::boost::python::dispatch::FromTensor<
    metal::list<uint8_t>,
    metal::numbers<TRank>,
    metal::numbers<mem::HOST, mem::DEVICE>
  >(object)
)
