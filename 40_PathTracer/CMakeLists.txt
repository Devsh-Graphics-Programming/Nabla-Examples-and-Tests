include(common RESULT_VARIABLE RES)
if(NOT RES)
        message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()


set(NBL_INCLUDE_SERACH_DIRECTORIES
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
)

list(APPEND NBL_LIBRARIES 
	imtestengine
	"${NBL_EXT_IMGUI_UI_LIB}"
)

nbl_create_executable_project("" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")

if(NBL_EMBED_BUILTIN_RESOURCES)
	set(_BR_TARGET_ ${EXECUTABLE_NAME}_builtinResourceData)
	set(RESOURCE_DIR "app_resources")

	get_filename_component(_SEARCH_DIRECTORIES_ "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
	get_filename_component(_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/src" ABSOLUTE)
	get_filename_component(_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)

	file(GLOB_RECURSE BUILTIN_RESOURCE_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_DIR}/*")
	foreach(RES_FILE ${BUILTIN_RESOURCE_FILES})
		LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "${RES_FILE}")
	endforeach()

	ADD_CUSTOM_BUILTIN_RESOURCES(${_BR_TARGET_} RESOURCES_TO_EMBED "${_SEARCH_DIRECTORIES_}" "${RESOURCE_DIR}" "nbl::this_example::builtin" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}")

	LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} ${_BR_TARGET_})
endif()

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
# TODO: why not use GLOB_RECURSE from above ?
set(DEPENDS
    app_resources/common.hlsl
    app_resources/light_directional.rcall.hlsl
	app_resources/light_point.rcall.hlsl
	app_resources/light_spot.rcall.hlsl
	app_resources/present.frag.hlsl
	app_resources/raytrace.rahit.hlsl
	app_resources/raytrace.rchit.hlsl
	app_resources/raytrace.rgen.hlsl
	app_resources/raytrace.rint.hlsl
	app_resources/raytrace.rmiss.hlsl
	app_resources/raytrace_procedural.rchit.hlsl
	app_resources/raytrace_shadow.rahit.hlsl
	app_resources/raytrace_shadow.rmiss.hlsl
)
target_sources(${EXECUTABLE_NAME} PRIVATE ${DEPENDS})
set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)

set(SM 6_8)
set(JSON [=[
[
    {
        "INPUT": "app_resources/raytrace.rgen.hlsl",
        "KEY": "raytrace_rgen",
    },
	{
        "INPUT": "app_resources/raytrace.rchit.hlsl",
        "KEY": "raytrace_rchit",
    },
	{
        "INPUT": "app_resources/raytrace_procedural.rchit.hlsl",
        "KEY": "raytrace_procedural_rchit",
    },
	{
        "INPUT": "app_resources/raytrace.rint.hlsl",
        "KEY": "raytrace_rint",
    },
	{
        "INPUT": "app_resources/raytrace.rahit.hlsl",
        "KEY": "raytrace_rahit",
    },
	{
        "INPUT": "app_resources/raytrace_shadow.rahit.hlsl",
        "KEY": "raytrace_shadow_rahit",
    },
	{
        "INPUT": "app_resources/raytrace.rmiss.hlsl",
        "KEY": "raytrace_rmiss",
    },
	{
        "INPUT": "app_resources/raytrace_shadow.rmiss.hlsl",
        "KEY": "raytrace_shadow_rmiss",
    },
	{
        "INPUT": "app_resources/light_directional.rcall.hlsl",
        "KEY": "light_directional_rcall",
    },
	{
        "INPUT": "app_resources/light_point.rcall.hlsl",
        "KEY": "light_point_rcall",
    },
	{
        "INPUT": "app_resources/light_spot.rcall.hlsl",
        "KEY": "light_spot_rcall",
    },
	{
        "INPUT": "app_resources/present.frag.hlsl",
        "KEY": "present_frag",
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
    DEPENDS ${DEPENDS}
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


