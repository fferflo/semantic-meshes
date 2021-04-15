#pragma once

#include <template_tensors/TemplateTensors.h>

namespace semantic_meshes {

namespace colmap {

class Workspace
{
public:
  Workspace(boost::filesystem::path workspace_path, boost::filesystem::path ply_file);

  tt::tinyply::ReadProperty<float, 3>& getTinyplyVertices();
  tt::tinyply::ReadProperty<int32_t, 3, uint8_t>& getTinyplyFaces();

  const tt::colmap::Camera& getCamera(uint32_t id) const;
  const std::map<uint32_t, tt::colmap::Camera>& getCameras() const;

  size_t getImageNum() const;
  const tt::colmap::ImageMetaData& getImageMetaData(size_t index) const;
  const tt::colmap::ImageMetaData& getImageMetaData(boost::filesystem::path path) const;
  const boost::filesystem::path& getImagePath(size_t index) const;
  size_t getImageIndex(boost::filesystem::path path) const;

private:
  tt::tinyply::ReadProperty<float, 3> m_tinyply_vertices;
  tt::tinyply::ReadProperty<int32_t, 3, uint8_t> m_tinyply_faces;

  std::map<uint32_t, tt::colmap::Camera> m_cameras;
  std::vector<tt::colmap::ImageMetaData> m_image_meta_data_sorted;
};

} // end of ns colmap

} // end of ns semantic_meshes
