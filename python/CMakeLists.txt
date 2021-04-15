add_subdirectory(semantic_meshes)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/semantic_meshes/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/semantic_meshes/__init__.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/colorize_cityscapes_mesh.py ${CMAKE_CURRENT_BINARY_DIR}/scripts/colorize_cityscapes_mesh.py @ONLY)
