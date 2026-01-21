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

	# TODO; Arek I removed `NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET` from the last parameter here, doesn't this macro have 4 arguments anyway !?
	nbl_create_executable_project("${NBL_EXTRA_SOURCES}" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}")

	# TODO: Arek temporarily disabled cause I haven't figured out how to make this target yet
	# LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} nblExamplesGeometrySpirvBRD)
	set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
	set(DEPENDS
		app_resources/hlsl/common.hlsl
		app_resources/hlsl/gpu_common.hlsl
		app_resources/hlsl/Drawing.hlsl
		app_resources/hlsl/Sampling.hlsl
		app_resources/hlsl/silhouette.hlsl
		app_resources/hlsl/utils.hlsl
		app_resources/hlsl/parallelogram_sampling.hlsl

		# app_resources/hlsl/test.comp.hlsl
		app_resources/hlsl/benchmark/benchmark.comp.hlsl
		app_resources/hlsl/benchmark/common.hlsl
	)
	target_sources(${EXECUTABLE_NAME} PRIVATE ${DEPENDS})
	set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)

	set(SM 6_8)
	set(JSON [=[
	[

		{
			"INPUT": "app_resources/hlsl/benchmark/benchmark.comp.hlsl",
			"KEY": "benchmark",
		},
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
endif()