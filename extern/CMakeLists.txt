option(BUILD_DEPENDENCIES_IF_NECESSARY "Download and build some external dependencies if they cannot be found" ON)

if(BUILD_DEPENDENCIES_IF_NECESSARY)
  function(build_dependency NAME FIND_NAME CMAKE_ARGS)
    # Check if dependency can already be found
    find_package(${FIND_NAME} QUIET)
    if(NOT ${FIND_NAME}_FOUND)
      # Git submodule update
      find_package(Git QUIET)
      if(NOT GIT_FOUND)
        message(FATAL_ERROR "Cannot build external dependencies without git")
      endif()
      if(NOT EXISTS "${PROJECT_SOURCE_DIR}/.git")
        message(FATAL_ERROR "Project is not a git directory")
      endif()
      execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init ${NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND_ERROR_IS_FATAL ANY
      )

      # Build
      file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${NAME}-extern)
      set(CMAKE_PREFIX_PATH_STR ${CMAKE_PREFIX_PATH})
      string(REPLACE ";" "\;" CMAKE_PREFIX_PATH_STR "${CMAKE_PREFIX_PATH_STR}")
      execute_process(
        COMMAND ${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/${NAME} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH_STR} ${CMAKE_ARGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${NAME}-extern
        COMMAND_ERROR_IS_FATAL ANY
      )
      execute_process(
        COMMAND ${CMAKE_COMMAND} --build . --target install
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${NAME}-extern
        COMMAND_ERROR_IS_FATAL ANY
      )
    endif()
  endfunction()

  build_dependency(tinyply tinyply "-DSHARED_LIB=ON")
  build_dependency(dlpack dlpack "")
  build_dependency(template-tensors TemplateTensors "")

  set(CMAKE_PREFIX_PATH "${CMAKE_INSTALL_PREFIX};${CMAKE_PREFIX_PATH}")
endif()
