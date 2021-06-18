#include <semantic_meshes/data/Ply.h>

#include <boost/algorithm/string/predicate.hpp>

namespace semantic_meshes {

namespace data {

Ply::Ply(boost::filesystem::path ply_file)
  : m_tinyply_vertices("vertex", {"x", "y", "z"})
  , m_tinyply_faces("face")
{
  tt::tinyply::read(ply_file, m_tinyply_vertices, m_tinyply_faces);
  m_tinyply_faces.setPropertyKey("vertex_indices");
}

tt::tinyply::ReadProperty<float, 3>& Ply::getTinyplyVertices()
{
  return m_tinyply_vertices;
}

tt::tinyply::ReadProperty<int32_t, 3, uint8_t>& Ply::getTinyplyFaces()
{
  return m_tinyply_faces;
}

} // end of ns data

} // end of ns semantic_meshes
