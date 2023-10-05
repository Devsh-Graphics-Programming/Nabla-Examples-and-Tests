
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(EXAMPLE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/curves.h"
)
nbl_create_executable_project("${EXAMPLE_SOURCES}" "" "" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")