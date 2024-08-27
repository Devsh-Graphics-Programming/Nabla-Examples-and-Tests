nbl_create_ext_library_project(CommonAPI "${CMAKE_CURRENT_SOURCE_DIR}/CAssetConverter.h" "${CMAKE_CURRENT_SOURCE_DIR}/CAssetConverter.cpp" "" "" "")
set(NBL_EXECUTABLE_COMMON_API_TARGET "${LIB_NAME}" CACHE INTERNAL "")

add_subdirectory(CommonPCH EXCLUDE_FROM_ALL)

# @Arek what is this for?
#target_precompile_headers("${NBL_EXECUTABLE_COMMON_API_TARGET}" REUSE_FROM "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")

# @Arek do I still need this ? old comment : tmp for testing
# target_link_libraries(CommonAPI PUBLIC blake3)