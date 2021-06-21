add_subdirectory(semantic_meshes)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/colorize_cityscapes_mesh.py ${CMAKE_CURRENT_BINARY_DIR}/scripts/colorize_cityscapes_mesh.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/colorize_mesh.py ${CMAKE_CURRENT_BINARY_DIR}/scripts/colorize_mesh.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/semantic_meshes/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/semantic_meshes/__init__.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/semantic_meshes/data2/scannet.py ${CMAKE_CURRENT_BINARY_DIR}/semantic_meshes/data2/scannet.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/semantic_meshes/data2/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/semantic_meshes/data2/__init__.py COPYONLY)
