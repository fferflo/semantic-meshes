#include <Ply.h>

template <typename TRenderer>
void registerRenderer(std::string name)
{
  boost::python::class_<TRenderer>(name.c_str(), boost::python::no_init)
    .def("getPrimitivesNum", &TRenderer::getPrimitivesNum)
    .def("render", &TRenderer::render)
  ;
}

BOOST_PYTHON_MODULE(render)
{
  Py_Initialize();
  boost::python::numpy::initialize();

  registerRenderer<PlyRendererTexels>("PlyRendererTexels");
  registerRenderer<PlyRendererTriangles>("PlyRendererTriangles");

  boost::python::def("texels", renderer_texels_ply1);
  boost::python::def("texels", renderer_texels_ply2);
  boost::python::def("texels", renderer_texels_ply3);
  boost::python::def("texels", renderer_texels_ply4);
  boost::python::def("triangles", renderer_triangles_ply);
};
