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
)
set(NBL_EXAMPLE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/CSceneLoader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CSession.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CScene.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CRenderer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/resolve/CBasicRWMCResolver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/present/CWindowPresenter.cpp"
)
nbl_create_executable_project("${NBL_EXAMPLE_SOURCES}" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}")

add_dependencies(${EXECUTABLE_NAME} argparse)
target_include_directories(${EXECUTABLE_NAME} PRIVATE $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PATH_TRACER_RUNTIME_CONFIG_FILENAME="pt.$<LOWER_CASE:$<CONFIG>>.json")

if(NOT TARGET Nabla::examples::denoiserTonemapper)
	message(FATAL_ERROR "40_PathTracer requires the Nabla::examples::denoiserTonemapper target to be available.")
endif()

set(PT_RUNTIME_CONFIG_JSON [=[
{
  "denoiserStubMode": $<IF:$<BOOL:${NBL_DENOISER_TONEMAPPER_EXE_STUB_MODE}>,true,false>,
  "cli": {
    "--denoiser-exe": "$<PATH:RELATIVE_PATH,$<TARGET_FILE:Nabla::examples::denoiserTonemapper>,$<PATH:APPEND,$<TARGET_FILE_DIR:@EXECUTABLE_NAME@>,config>>"
  }
}
]=])
string(CONFIGURE "${PT_RUNTIME_CONFIG_JSON}" PT_RUNTIME_CONFIG_JSON @ONLY)

set(PT_RUNTIME_CONFIG_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/bin/config/pt.$<LOWER_CASE:$<CONFIG>>.json")
file(GENERATE OUTPUT "${PT_RUNTIME_CONFIG_OUTPUT}" CONTENT "${PT_RUNTIME_CONFIG_JSON}")
set_source_files_properties("${PT_RUNTIME_CONFIG_OUTPUT}" PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
target_sources(${EXECUTABLE_NAME} PRIVATE "${PT_RUNTIME_CONFIG_OUTPUT}")
source_group("config" REGULAR_EXPRESSION [[.*[/\\]bin[/\\]config[/\\].*\.json$]])

get_target_property(_PT_RUNTIME_EXE_DIR ${EXECUTABLE_NAME} NBL_PACKAGE_RUNTIME_EXE_DIR_PATH)
install(FILES "${PT_RUNTIME_CONFIG_OUTPUT}" DESTINATION "${_PT_RUNTIME_EXE_DIR}/config" COMPONENT Executables)

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


