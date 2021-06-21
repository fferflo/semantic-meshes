#include <Ply.h>
#include <Colmap.h>
#include <Camera.h>

BOOST_PYTHON_MODULE(data)
{
  Py_Initialize();
  boost::python::numpy::initialize();

  boost::python::class_<Colmap>("Colmap", boost::python::init<std::string>())
    .def("getCamera", &Colmap::getCamera1)
    .def("getCamera", &Colmap::getCamera2)
  ;

  boost::python::class_<Ply>("Ply", boost::python::init<std::string>())
    .def("save", &Ply::save1)
    .def("save", &Ply::save2)
  ;
  boost::python::class_<Camera>("Camera", boost::python::init<boost::python::object, boost::python::object, boost::python::object, boost::python::object, boost::python::object>());
};
