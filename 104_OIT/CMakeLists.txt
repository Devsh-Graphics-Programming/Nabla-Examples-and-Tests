# Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h
include(common)

set(NBL_INCLUDE_SEARCH_DIRECTORIES
	"${CMAKE_CURRENT_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
)

nbl_create_executable_project(
	""
	""
	"${NBL_INCLUDE_SEARCH_DIRECTORIES}"
	""
	"${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}"
)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE Nabla::ext::FullScreenTriangle)
set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(SM 6_8)

set(JSON [=[
[
    {
        "INPUT": "app_resources/triangle.vert.hlsl",
        "KEY": "triangle_vert",
    },
    {
        "INPUT": "app_resources/triangle.frag.hlsl",
        "KEY": "triangle_frag",
    },
	{
    "INPUT": "app_resources/mlab.frag.hlsl",
    "KEY": "mlab_frag",
	}
]
]=])
string(CONFIGURE "${JSON}" JSON)

set(COMPILE_OPTIONS
	-I "${CMAKE_CURRENT_SOURCE_DIR}"
	-T lib_${SM}
)

NBL_CREATE_NSC_COMPILE_RULES(
	TARGET ${EXECUTABLE_NAME}SPIRV
	LINK_TO ${EXECUTABLE_NAME}
	BINARY_DIR ${OUTPUT_DIRECTORY}
	MOUNT_POINT_DEFINE NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT
	COMMON_OPTIONS ${COMPILE_OPTIONS}
	OUTPUT_VAR KEYS
	INCLUDE nbl/this_example/builtin/build/spirv/keys.hpp
	NAMESPACE nbl::this_example::builtin::build
	INPUTS ${JSON}
)

NBL_CREATE_RESOURCE_ARCHIVE(
	NAMESPACE nbl::this_example::builtin::build
	TARGET ${EXECUTABLE_NAME}_builtinsBuild
	LINK_TO ${EXECUTABLE_NAME}
	BIND ${OUTPUT_DIRECTORY}
	BUILTINS ${KEYS}
)
