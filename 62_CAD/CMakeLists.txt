include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(EXAMPLE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/Curves.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/Curves.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/Hatch.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/Hatch.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/Polyline.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/Polyline.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/DrawResourcesFiller.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/DrawResourcesFiller.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/SingleLineText.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/SingleLineText.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/Images.cpp"
)
set(EXAMPLE_INCLUDES
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/boost/superproject/libs/math/include")
nbl_create_executable_project("${EXAMPLE_SOURCES}" "" "${EXAMPLE_INCLUDES}" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")
target_link_libraries(${EXECUTABLE_NAME} PRIVATE Nabla::ext::FullScreenTriangle)

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")

set(SM 6_8)
set(REQUIRED_CAPS [=[
{
  "kind": "features",
  "name": "fragmentShaderPixelInterlock",
  "type": "bool",
  "values": [1]
}
]=])

set(JSON [=[
[
	{
		"INPUT": "shaders/main_pipeline/vertex_shader.hlsl",
		"KEY": "main_pipeline_vertex_shader",
		"CAPS": [${REQUIRED_CAPS}]
    },
    {
		"INPUT": "shaders/main_pipeline/all_fragment_shaders.hlsl",
		"KEY": "main_pipeline_fragment_shader",
		"CAPS": [${REQUIRED_CAPS}]
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
