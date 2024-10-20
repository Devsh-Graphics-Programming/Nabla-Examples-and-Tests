include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in '${NBL_ROOT_PATH}/cmake' directory")
endif()

nbl_create_executable_project("" "" "" "" "")

set(NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET "${EXECUTABLE_NAME}" CACHE INTERNAL "")
get_target_property(NBL_NABLA_TARGET_SOURCE_DIR Nabla SOURCE_DIR)
set_target_properties("${EXECUTABLE_NAME}" PROPERTIES DISABLE_PRECOMPILE_HEADERS OFF)
target_precompile_headers("${EXECUTABLE_NAME}" PUBLIC 
	"${CMAKE_CURRENT_SOURCE_DIR}/PCH.hpp" # Common PCH for examples
	"${NBL_NABLA_TARGET_SOURCE_DIR}/pch.h" # Nabla's PCH
)
unset(NBL_NABLA_TARGET_SOURCE_DIR)