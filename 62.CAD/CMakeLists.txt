
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(EXAMPLE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/curves.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/Hatch.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/curves.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/Hatch.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/Polyline.h"
)
set(EXAMPLE_INCLUDES
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/boost/superproject/libs/math/include")
nbl_create_executable_project("${EXAMPLE_SOURCES}" "" "${EXAMPLE_INCLUDES}" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")