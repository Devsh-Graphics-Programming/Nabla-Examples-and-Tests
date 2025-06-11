nbl_create_ext_library_project(ExamplesAPI "" "${CMAKE_CURRENT_SOURCE_DIR}/src/nbl/examples/pch.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/include" "" "")

set_target_properties(${LIB_NAME} PROPERTIES DISABLE_PRECOMPILE_HEADERS OFF)
target_precompile_headers(${LIB_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/nbl/examples/PCH.hpp")

add_subdirectory("src/nbl/examples" EXCLUDE_FROM_ALL)
set(NBL_EXAMPLES_API_TARGET ${LIB_NAME} PARENT_SCOPE)