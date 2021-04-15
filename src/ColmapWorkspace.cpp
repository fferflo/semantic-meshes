#include <semantic_meshes/colmap/Workspace.h>

#include <boost/algorithm/string/predicate.hpp>

namespace semantic_meshes {

namespace colmap {

Workspace::Workspace(boost::filesystem::path colmap_path, boost::filesystem::path ply_file)
  : m_tinyply_vertices("vertex", {"x", "y", "z"})
  , m_tinyply_faces("face")
  , m_cameras(tt::colmap::readCameras(colmap_path / "cameras.bin"))
{
  std::map<uint32_t, tt::colmap::ImageMetaData> image_meta_data = tt::colmap::readImageMetaData(colmap_path / "images.bin");

  m_image_meta_data_sorted.resize(image_meta_data.size());
  {
    size_t i = 0;
    for (decltype(image_meta_data)::iterator it = image_meta_data.begin(); it != image_meta_data.end(); it++)
    {
      m_image_meta_data_sorted[i++] = it->second;
    }
    std::sort(m_image_meta_data_sorted.begin(), m_image_meta_data_sorted.end(), [](tt::colmap::ImageMetaData& image1, tt::colmap::ImageMetaData& image2){
      return image1.name < image2.name;
    });
  }

  tt::tinyply::read(ply_file, m_tinyply_vertices, m_tinyply_faces);
  m_tinyply_faces.setPropertyKey("vertex_indices");
}

tt::tinyply::ReadProperty<float, 3>& Workspace::getTinyplyVertices()
{
  return m_tinyply_vertices;
}

tt::tinyply::ReadProperty<int32_t, 3, uint8_t>& Workspace::getTinyplyFaces()
{
  return m_tinyply_faces;
}

const tt::colmap::Camera& Workspace::getCamera(uint32_t id) const
{
  return m_cameras.at(id);
}

const std::map<uint32_t, tt::colmap::Camera>& Workspace::getCameras() const
{
  return m_cameras;
}

size_t Workspace::getImageNum() const
{
  return m_image_meta_data_sorted.size();
}

const tt::colmap::ImageMetaData& Workspace::getImageMetaData(size_t index) const
{
  return m_image_meta_data_sorted[index];
}

const tt::colmap::ImageMetaData& Workspace::getImageMetaData(boost::filesystem::path path) const
{
  return getImageMetaData(getImageIndex(path));
}

size_t Workspace::getImageIndex(boost::filesystem::path path) const
{
  std::string filename = path.remove_trailing_separator().filename().string();
  for (size_t i = 0; i < m_image_meta_data_sorted.size(); i++)
  {
    if (m_image_meta_data_sorted[i].name == filename)
    {
      return i;
    }
  }
  std::cout << "Image with name " << filename << " not found in colmap workspace" << std::endl;
  exit(-1); // TODO: proper exception handling
}

} // end of ns colmap

} // end of ns semantic_meshes
