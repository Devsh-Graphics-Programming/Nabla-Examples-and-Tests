if(NBL_BUILD_IMGUI)
	set(NBL_EXTRA_SOURCES
		"${CMAKE_CURRENT_SOURCE_DIR}/src/transform.cpp"
	)

	set(NBL_INCLUDE_SERACH_DIRECTORIES
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
	)

	list(APPEND NBL_LIBRARIES 
		imtestengine
		imguizmo
		"${NBL_EXT_IMGUI_UI_LIB}"
	)

	nbl_create_executable_project("${NBL_EXTRA_SOURCES}" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}")

	if(NBL_EMBED_BUILTIN_RESOURCES)
		set(_BR_TARGET_ ${EXECUTABLE_NAME}_builtinResourceData)
		set(RESOURCE_DIR "app_resources")

		get_filename_component(_SEARCH_DIRECTORIES_ "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
		get_filename_component(_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/src" ABSOLUTE)
		get_filename_component(_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)
		
		file(GLOB_RECURSE BUILTIN_RESOURCE_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_DIR}" CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_DIR}/*")
		foreach(RES_FILE ${BUILTIN_RESOURCE_FILES})
			LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "${RES_FILE}")
		endforeach()

		ADD_CUSTOM_BUILTIN_RESOURCES(${_BR_TARGET_} RESOURCES_TO_EMBED "${_SEARCH_DIRECTORIES_}" "${RESOURCE_DIR}" "nbl::this_example::builtin" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}")

		LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} ${_BR_TARGET_})
	endif()

	set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")

	set(JSON [=[
[
	{
		"INPUT": "app_resources/imgui_vertex.hlsl",
		"KEY": "imgui_vertex",
		"COMPILE_OPTIONS": ["-T", "vs_6_7", "-E", "VSMain", "-O3"]
	},
	{
		"INPUT": "app_resources/imgui_fragment.hlsl",
		"KEY": "imgui_fragment",
		"COMPILE_OPTIONS": ["-T", "ps_6_7", "-E", "PSMain", "-O3"]
	}
]
]=])
	string(CONFIGURE "${JSON}" JSON)

	set(COMPILE_OPTIONS
		-I "${CMAKE_CURRENT_SOURCE_DIR}"
		-I "${NBL_ROOT_PATH}/include"
		-I "${NBL_ROOT_PATH}/include/nbl/ext/ImGui/builtin/hlsl"
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

	add_dependencies(${EXECUTABLE_NAME} argparse)
	target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
