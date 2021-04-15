find_package(TemplateTensors REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_library(semantic_meshes SHARED ColmapWorkspace.cpp)
target_include_directories(semantic_meshes
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(semantic_meshes PUBLIC cxx_std_11)
target_link_libraries(semantic_meshes PUBLIC
  TemplateTensors::base
  TemplateTensors::interface.openmp
  TemplateTensors::interface.tinyply
  Boost::filesystem
)
install(
  TARGETS semantic_meshes
  EXPORT install_targets
)
