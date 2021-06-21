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

Python3_add_library(fusion
  src/Fusion.cu
)
target_link_libraries(fusion PRIVATE
  semantic_meshes
  TemplateTensors::interface.boost.python
  TemplateTensors::interface.boost.numpy
  TemplateTensors::interface.dlpack
)
target_include_directories(fusion PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set(CLASSES_NUMS_BOOST_SEQ "")
foreach(C ${CLASSES_NUMS})
  set(CLASSES_NUMS_BOOST_SEQ "${CLASSES_NUMS_BOOST_SEQ}(${C})")
endforeach()
target_compile_definitions(fusion PRIVATE CLASSES_NUMS=${CLASSES_NUMS_BOOST_SEQ})

Python3_add_library(data
  src/Data.cu
)
target_link_libraries(data PRIVATE
  semantic_meshes
  TemplateTensors::interface.boost.python
  TemplateTensors::interface.boost.numpy
  TemplateTensors::interface.dlpack
)
target_include_directories(data PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

Python3_add_library(render
  src/Render.cu
)
target_link_libraries(render PRIVATE
  semantic_meshes
  TemplateTensors::interface.boost.python
  TemplateTensors::interface.boost.numpy
  TemplateTensors::interface.dlpack
)
target_include_directories(render PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
