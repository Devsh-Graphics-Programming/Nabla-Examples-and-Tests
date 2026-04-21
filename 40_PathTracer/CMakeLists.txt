include(common)

set(NBL_INCLUDE_SERACH_DIRECTORIES
	"include"
	"src"
	"app_resources"
)
set(NBL_LIBRARIES
	"${NBL_EXT_MITSUBA_LOADER_LIB}"
	Nabla::ext::FullScreenTriangle
    imguizmo
	"${NBL_EXT_IMGUI_UI_LIB}"
	argparse
)

set(PT_RUNTIME_CONFIG_OUTPUT "bin/config/pt.$<LOWER_CASE:$<CONFIG>>.json")
file(GLOB_RECURSE NBL_EXAMPLE_SOURCES CONFIGURE_DEPENDS "app_resources/*.hlsl" "include/*.h" "include/*.hpp" "src/*.cpp" "src/*.h" "src/*.hpp")
file(GLOB_RECURSE PT_REPORT_SOURCES CONFIGURE_DEPENDS "report/*.css" "report/*.html" "report/*.js" "report/*.py")
set(PT_PROJECT_SOURCES ${NBL_EXAMPLE_SOURCES} ${PT_REPORT_SOURCES} "${PT_RUNTIME_CONFIG_OUTPUT}")
set_source_files_properties("${PT_RUNTIME_CONFIG_OUTPUT}" PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
set_source_files_properties(${PT_REPORT_SOURCES} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("config" FILES "${PT_RUNTIME_CONFIG_OUTPUT}")
source_group("CI Report" FILES ${PT_REPORT_SOURCES})
set(PT_RUNTIME_CONFIG_JSON [=[
{
  "cli": {
    "--output-dir": "../out/$<LOWER_CASE:$<CONFIG>>/renders",
    "--report-dir": "../out/$<LOWER_CASE:$<CONFIG>>"
  }
}
]=])
file(GENERATE OUTPUT "${PT_RUNTIME_CONFIG_OUTPUT}" CONTENT "${PT_RUNTIME_CONFIG_JSON}")

nbl_create_executable_project("${PT_PROJECT_SOURCES}" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}")
target_compile_definitions(${EXECUTABLE_NAME} PRIVATE NBL_THIS_EXAMPLE_BUILD_CONFIG=\"$<CONFIG>\")

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


