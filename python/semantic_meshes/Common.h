#include <template_tensors/TemplateTensors.h>

auto FromPrimitiveImage(boost::python::object object)
RETURN_AUTO(
  tt::boost::python::dispatch::FromTensor<
    tmp::ts::Sequence<uint32_t, int32_t, uint64_t, int64_t>,
    tmp::vs::Sequence<size_t, 2>,
    tmp::vs::Sequence<mem::MemoryType, mem::HOST, mem::DEVICE>
  >(object)
)

auto FromProbsImage(boost::python::object object)
RETURN_AUTO(
  tt::boost::python::dispatch::FromTensor<
    tmp::ts::Sequence<float>,
    tmp::vs::Sequence<size_t, 3>,
    tmp::vs::Sequence<mem::MemoryType, mem::HOST, mem::DEVICE>
  >(object)
)

auto FromWeightsImage(boost::python::object object)
RETURN_AUTO(
  tt::boost::python::dispatch::FromTensor<
    tmp::ts::Sequence<float>,
    tmp::vs::Sequence<size_t, 2>,
    tmp::vs::Sequence<mem::MemoryType, mem::HOST, mem::DEVICE>
  >(object)
)

template <size_t TRank>
auto FromClassColors(::boost::python::object object)
RETURN_AUTO(
  tt::boost::python::dispatch::FromTensor<
    tmp::ts::Sequence<uint8_t>,
    tmp::vs::Sequence<size_t, TRank>,
    tmp::vs::Sequence<mem::MemoryType, mem::HOST, mem::DEVICE>
  >(object)
)
