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
  "../../src/nbl/ext/TextRendering/TextRendering.cpp" # TODO: this one will be a part of dedicated Nabla ext called "TextRendering" later on which uses MSDF + Freetype
)
set(EXAMPLE_INCLUDES
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/boost/superproject/libs/math/include")
nbl_create_executable_project("${EXAMPLE_SOURCES}" "" "${EXAMPLE_INCLUDES}" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")
target_link_libraries(${EXECUTABLE_NAME} PRIVATE Nabla::ext::FullScreenTriangle)

# if enabled then try use Nabla "Text Rendering" extension 
# with an implemented interface using the 3rdparty deps 

set(NBL_CAD_EX_USE_TEXT_RENDERING_EXT OFF) # do not enable, for future usage when the extension is written

if(NBL_BUILD_TEXT_RENDERING AND NBL_CAD_EX_USE_TEXT_RENDERING_EXT)
	add_dependencies(${EXECUTABLE_NAME} ${NBL_EXT_TEXT_RENDERING_TARGET})
	target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${NBL_EXT_TEXT_RENDERING_TARGET})
	target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:${NBL_EXT_TEXT_RENDERING_TARGET},INCLUDE_DIRECTORIES>)
else()
	# Freetype
	add_dependencies(${EXECUTABLE_NAME} freetype)
	target_link_libraries(${EXECUTABLE_NAME} PRIVATE freetype)
	target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:freetype,INCLUDE_DIRECTORIES>)

	# msdfgen
	add_dependencies(${EXECUTABLE_NAME} ${NBL_MSDFGEN_TARGETS})
	target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${NBL_MSDFGEN_TARGETS})
	foreach(NBL_TARGET IN LISTS NBL_MSDFGEN_TARGETS)
		target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:${NBL_TARGET},INCLUDE_DIRECTORIES>)
	endforeach()
endif()

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
