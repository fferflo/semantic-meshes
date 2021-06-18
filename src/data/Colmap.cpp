#include <semantic_meshes/data/Colmap.h>

namespace semantic_meshes {

namespace data {

Colmap::Colmap(boost::filesystem::path colmap_path)
  : m_cameras(tt::colmap::readCameras(colmap_path / "cameras.*"))
{
  std::map<uint32_t, tt::colmap::ImageMetaData> image_meta_data = tt::colmap::readImageMetaData(colmap_path / "images.*");

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
}

std::vector<Camera> Colmap::getCameras() const
{
  std::vector<Camera> result;
  for (size_t index = 0; index < m_image_meta_data_sorted.size(); index++)
  {
    result.push_back(this->getCamera(index));
  }
  return result;
}

size_t Colmap::getImageNum() const
{
  return m_image_meta_data_sorted.size();
}

const tt::colmap::ImageMetaData& Colmap::getImageMetaData(size_t index) const
{
  return m_image_meta_data_sorted[index];
}

const tt::colmap::ImageMetaData& Colmap::getImageMetaData(boost::filesystem::path path) const
{
  return getImageMetaData(getImageIndex(path));
}

size_t Colmap::getImageIndex(boost::filesystem::path path) const
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

} // end of ns data

} // end of ns semantic_meshes
