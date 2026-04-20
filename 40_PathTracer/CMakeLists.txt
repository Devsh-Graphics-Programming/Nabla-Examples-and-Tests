include(common)

set(NBL_INCLUDE_SERACH_DIRECTORIES
	"${NBL_EXT_MITSUBA_LOADER_INCLUDE_DIRS}"
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
	"${CMAKE_CURRENT_SOURCE_DIR}/app_resources"
)
set(NBL_LIBRARIES
	"${NBL_EXT_MITSUBA_LOADER_LIB}"
	Nabla::ext::FullScreenTriangle
    imguizmo
	"${NBL_EXT_IMGUI_UI_LIB}"
	argparse
)

set(PT_RUNTIME_CONFIG_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/bin/config/pt.$<LOWER_CASE:$<CONFIG>>.json")
set_source_files_properties("${PT_RUNTIME_CONFIG_OUTPUT}" PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
source_group("config" FILES "${PT_RUNTIME_CONFIG_OUTPUT}")
set(PT_RUNTIME_CONFIG_JSON [=[
{
  "cli": {
    "--output-dir": "../out/$<LOWER_CASE:$<CONFIG>>"
  }
}
]=])
file(GENERATE OUTPUT "${PT_RUNTIME_CONFIG_OUTPUT}" CONTENT "${PT_RUNTIME_CONFIG_JSON}")

set(NBL_EXAMPLE_SOURCES
    "${PT_RUNTIME_CONFIG_OUTPUT}"
    "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/resolve/rwmc.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/io/RuntimeConfig.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/CSceneLoader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/RuntimeConfig.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CSession.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CScene.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CRenderer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/resolve/CBasicRWMCResolver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/present/CWindowPresenter.cpp"
)
nbl_create_executable_project("${NBL_EXAMPLE_SOURCES}" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}")

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")

set(SM 6_9)
set(JSON [=[
[
    {
        "INPUT": "app_resources/pathtrace/previs.hlsl",
        "KEY": "pathtrace_previs",
    },
    {
        "INPUT": "app_resources/pathtrace/beauty.hlsl",
        "KEY": "pathtrace_beauty",
    },
    {
        "INPUT": "app_resources/pathtrace/debug.hlsl",
        "KEY": "pathtrace_debug",
    },
	{
        "INPUT": "app_resources/present/default.hlsl",
        "KEY": "present_default",
    },
	{
        "INPUT": "app_resources/resolve/rwmc.hlsl",
        "KEY": "resolve_rwmc",
    }
]
]=])
string(CONFIGURE "${JSON}" JSON)

set(COMPILE_OPTIONS
    -I "${CMAKE_CURRENT_SOURCE_DIR}/../common/include"
    -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
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


