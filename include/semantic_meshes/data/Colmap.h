#pragma once

#include <template_tensors/TemplateTensors.h>
#include <semantic_meshes/render/Camera.h>

namespace semantic_meshes {

namespace data {

class Colmap
{
public:
  Colmap(boost::filesystem::path workspace_path);

  size_t getImageNum() const;
  const boost::filesystem::path& getImagePath(size_t index) const;
  size_t getImageIndex(boost::filesystem::path path) const;

  template <typename TImageId>
  Camera getCamera(TImageId image_id) const
  {
    auto image_meta_data = this->getImageMetaData(image_id);
    auto& camera = m_cameras.at(image_meta_data.camera_id);
    return Camera{camera.projection, image_meta_data.transform, camera.resolution};
  }

  std::vector<Camera> getCameras() const;

private:
  std::map<uint32_t, tt::colmap::Camera> m_cameras;
  std::vector<tt::colmap::ImageMetaData> m_image_meta_data_sorted;

  const tt::colmap::ImageMetaData& getImageMetaData(size_t index) const;
  const tt::colmap::ImageMetaData& getImageMetaData(boost::filesystem::path path) const;
};

} // end of ns colmap

} // end of ns semantic_meshes
