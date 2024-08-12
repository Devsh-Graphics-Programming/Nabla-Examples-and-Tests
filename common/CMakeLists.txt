###########################################
# TODO: the way it should work is following (remove the comment once all done!)
# - one top PCH which includes <nabla.h> -> currently not done
# - sources used only within examples splitted into "common libraries" (optional -> with options to toggle if include them to build tree), each common library should reuse the above top PCH
# - examples_tests CMake loop over example targets and hook the interface library with NBL_HOOK_COMMON_API [done]
# - each common library should declare ONLY interface and never expose source definition into headers nor any 3rdparty stuff!
##

# interface libraries don't have build rules (except custom commands however it doesn't matter here) but properties
add_library(nblCommonAPI INTERFACE)
set(NBL_COMMON_API_INCLUDE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(nblCommonAPI INTERFACE "${NBL_COMMON_API_INCLUDE_DIRECTORY}")

add_subdirectory(src EXCLUDE_FROM_ALL)

########## <-
# TODO: disable this CommonPCH thing! + DEPRICATED!
nbl_create_ext_library_project(CommonAPI "${CMAKE_CURRENT_SOURCE_DIR}/CommonAPI.h" "${CMAKE_CURRENT_SOURCE_DIR}/CommonAPI.cpp" "" "" "")
set(NBL_EXECUTABLE_COMMON_API_TARGET "${LIB_NAME}" CACHE INTERNAL "")

add_subdirectory(CommonPCH EXCLUDE_FROM_ALL)

#target_precompile_headers("${NBL_EXECUTABLE_COMMON_API_TARGET}" REUSE_FROM "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")
########## <-

set(NBL_COMMON_API_TARGETS nblCommonAPI ${NBL_COMMON_API_TARGETS} ${NBL_EXECUTABLE_COMMON_API_TARGET} PARENT_SCOPE)