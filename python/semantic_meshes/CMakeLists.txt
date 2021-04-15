set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(TemplateTensors REQUIRED)

if(NOT TARGET TemplateTensors::interface.boost.python)
  message(FATAL_ERROR "Could not find boost python library")
endif()
if(NOT TARGET TemplateTensors::interface.boost.numpy)
  message(FATAL_ERROR "Could not find boost numpy library")
endif()

set(CLASSES_NUMS "19" CACHE STRING "List of number of classes for which the project should be compiled")

Python3_add_library(fusion Fusion.cu)
target_link_libraries(fusion PRIVATE
  semantic_meshes
  TemplateTensors::interface.boost.python
  TemplateTensors::interface.boost.numpy
  TemplateTensors::interface.dlpack
)

set(CLASSES_NUMS_BOOST_SEQ "")
foreach(C ${CLASSES_NUMS})
  set(CLASSES_NUMS_BOOST_SEQ "${CLASSES_NUMS_BOOST_SEQ}(${C})")
endforeach()
target_compile_definitions(fusion PRIVATE CLASSES_NUMS=${CLASSES_NUMS_BOOST_SEQ})

Python3_add_library(colmap Colmap.cu)
target_link_libraries(colmap PRIVATE
  semantic_meshes
  TemplateTensors::interface.boost.python
  TemplateTensors::interface.boost.numpy
  TemplateTensors::interface.dlpack
)
