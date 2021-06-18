#pragma once

#include <template_tensors/TemplateTensors.h>

namespace semantic_meshes {

namespace data {

class Ply
{
public:
  Ply(boost::filesystem::path ply_file);

  tt::tinyply::ReadProperty<float, 3>& getTinyplyVertices();
  tt::tinyply::ReadProperty<int32_t, 3, uint8_t>& getTinyplyFaces();

private:
  tt::tinyply::ReadProperty<float, 3> m_tinyply_vertices;
  tt::tinyply::ReadProperty<int32_t, 3, uint8_t> m_tinyply_faces;
};

} // end of ns data

} // end of ns semantic_meshes
