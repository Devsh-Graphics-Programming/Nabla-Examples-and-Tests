include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

nbl_create_executable_project("" "" "" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")

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

set(SM 6_8)
set(REQUIRED_CAPS [=[
	{
	  "kind": "limits",
	  "name": "maxComputeWorkGroupInvocations",
	  "type": "uint32_t",
	  "values": [256,512,1024]
	},
	{
	  "kind": "limits",
	  "name": "maxComputeSharedMemorySize",
	  "type": "uint32_t",
	  "values": [16384, 32768, 65536]
	},
	{
	  "kind": "limits",
	  "name": "maxSubgroupSize",
	  "type": "uint32_t",
	  "values": [4, 8, 16, 32, 64, 128]
	}
]=])

set(JSON [=[
[
	{
		"INPUT": "app_resources/prefix_sum_shader.comp.hlsl",
		"KEY": "prefix_sum_shader",
		"CAPS": [${REQUIRED_CAPS}]
    },
    {
		"INPUT": "app_resources/scatter_shader.comp.hlsl",
		"KEY": "scatter_shader",
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
