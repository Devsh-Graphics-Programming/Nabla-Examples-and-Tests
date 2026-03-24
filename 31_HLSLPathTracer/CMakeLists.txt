include(common RESULT_VARIABLE RES)

if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

if(NBL_BUILD_IMGUI)
	# PATH_TRACER_BUILD_MODE controls how EX31 packages triangle polygon-method variants.
	# WALLTIME_OPTIMIZED keeps triangle polygon-method selection inside one shared shader module.
	# This preserves the same runtime feature surface while keeping triangle methods on one heavy backend compile path.
	# SPECIALIZED bakes Area, SolidAngle and ProjectedSolidAngle into separate triangle entrypoints.
	# That does not add only thin entrypoints. It multiplies the heavy triangle-side path tracing instantiations and pushes more work into the DXC/SPIR-V backend.
	# Honest CLI rebuild measurement on AMD Ryzen 5 5600G with Radeon Graphics (6C/12T) using:
	#   cmake -E time cmake --build build/dynamic --config RelWithDebInfo --target examples_tests/31_HLSLPathTracer/31_hlslpathtracerSPIRV -- /m:12 /t:Rebuild /p:BuildProjectReferences=false
	# gave:
	#   WALLTIME_OPTIMIZED = 20.6093 s
	#   SPECIALIZED = 30.1792 s
	# SPECIALIZED is +9.5699 s slower which is +46.43%.
	# Therefore the default stays WALLTIME_OPTIMIZED.
	set(PATH_TRACER_BUILD_MODE "WALLTIME_OPTIMIZED" CACHE STRING
		"Choose the EX31 precompiled shader layout. WALLTIME_OPTIMIZED keeps polygon-method selection inside the shared triangle shader module to reduce compile wall time. SPECIALIZED bakes triangle polygon methods into distinct entrypoints, which increases backend work and rebuild time but keeps those variants as separate precompiled entrypoints."
	)
	set_property(CACHE PATH_TRACER_BUILD_MODE PROPERTY STRINGS WALLTIME_OPTIMIZED SPECIALIZED)
	set(_PATH_TRACER_BUILD_MODE_VALUES WALLTIME_OPTIMIZED SPECIALIZED)
	if(NOT PATH_TRACER_BUILD_MODE IN_LIST _PATH_TRACER_BUILD_MODE_VALUES)
		message(FATAL_ERROR "Unsupported PATH_TRACER_BUILD_MODE='${PATH_TRACER_BUILD_MODE}'. Expected one of: ${_PATH_TRACER_BUILD_MODE_VALUES}")
	endif()

	set(NBL_INCLUDE_SERACH_DIRECTORIES
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
	)

	list(APPEND NBL_LIBRARIES
		imguizmo
		"${NBL_EXT_IMGUI_UI_LIB}"
		Nabla::ext::FullScreenTriangle
	)

	nbl_create_executable_project("" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")
	if(PATH_TRACER_BUILD_MODE STREQUAL "SPECIALIZED")
		target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PATH_TRACER_BUILD_MODE_SPECIALIZED=1)
	else()
		target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PATH_TRACER_BUILD_MODE_WALLTIME_OPTIMIZED=1)
	endif()

	set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen-flat")
	get_filename_component(OUTPUT_DIRECTORY_ABSOLUTE "${OUTPUT_DIRECTORY}" ABSOLUTE)
	target_compile_definitions(${EXECUTABLE_NAME} PRIVATE NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT="${OUTPUT_DIRECTORY_ABSOLUTE}")

	file(GLOB_RECURSE EXAMPLE_HLSL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/hlsl/*.hlsl")
	target_sources(${EXECUTABLE_NAME} PRIVATE ${EXAMPLE_HLSL_SOURCES})
	set_source_files_properties(${EXAMPLE_HLSL_SOURCES} PROPERTIES HEADER_FILE_ONLY ON)

	set(SM 6_8)
	set(COMPILE_OPTIONS
		-I "${CMAKE_CURRENT_SOURCE_DIR}/include"
		-I "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/hlsl"
		-I "${OUTPUT_DIRECTORY_ABSOLUTE}/$<CONFIG>"
		-T "lib_${SM}"
		-Wno-conversion
		-Wno-sign-conversion
		-Wno-float-conversion
		-Wno-shorten-64-to-32
		-Wno-shadow
		-Wno-literal-range
	)

	set(SPIRV_FOLDER "SPIRV")
	set(SPIRV_INTERNAL_FOLDER "${SPIRV_FOLDER}/Internal")
	if(PATH_TRACER_BUILD_MODE STREQUAL "SPECIALIZED")
		set(SPIRV_RULES_JSON_TEMPLATE [=[
[
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.sphere.proxy.hlsl",
		"KEY": "pt.compute.sphere"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.sphere.rwmc.proxy.hlsl",
		"KEY": "pt.compute.sphere.rwmc"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.triangle.linear.proxy.hlsl",
		"KEY": "pt.compute.triangle.linear"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.triangle.persistent.proxy.hlsl",
		"KEY": "pt.compute.triangle.persistent"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.triangle.rwmc.linear.proxy.hlsl",
		"KEY": "pt.compute.triangle.rwmc.linear"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.triangle.rwmc.persistent.proxy.hlsl",
		"KEY": "pt.compute.triangle.rwmc.persistent"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.rectangle.proxy.hlsl",
		"KEY": "pt.compute.rectangle"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.rectangle.rwmc.linear.proxy.hlsl",
		"KEY": "pt.compute.rectangle.rwmc.linear"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.rectangle.rwmc.persistent.proxy.hlsl",
		"KEY": "pt.compute.rectangle.rwmc.persistent"
	},
	{
		"INPUT": "app_resources/hlsl/resolve.comp.hlsl",
		"KEY": "pt.compute.resolve"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.misc.proxy.hlsl",
		"KEY": "pt.misc"
	}
]
]=])
	else()
		set(SPIRV_RULES_JSON_TEMPLATE [=[
[
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.sphere.proxy.hlsl",
		"KEY": "pt.compute.sphere"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.sphere.rwmc.proxy.hlsl",
		"KEY": "pt.compute.sphere.rwmc"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.triangle.proxy.hlsl",
		"KEY": "pt.compute.triangle"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.triangle.rwmc.proxy.hlsl",
		"KEY": "pt.compute.triangle.rwmc"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.rectangle.proxy.hlsl",
		"KEY": "pt.compute.rectangle"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.compute.rectangle.rwmc.proxy.hlsl",
		"KEY": "pt.compute.rectangle.rwmc"
	},
	{
		"INPUT": "app_resources/hlsl/resolve.comp.hlsl",
		"KEY": "pt.compute.resolve"
	},
	{
		"INPUT": "app_resources/hlsl/spirv/pt.misc.proxy.hlsl",
		"KEY": "pt.misc"
	}
]
]=])
	endif()
	string(CONFIGURE "${SPIRV_RULES_JSON_TEMPLATE}" SPIRV_RULES_JSON @ONLY)
	set(ALL_SPIRV_KEYS "")
	NBL_CREATE_NSC_COMPILE_RULES(
		DISCARD_DEFAULT_GLOB
		TARGET ${EXECUTABLE_NAME}SPIRV
		BINARY_DIR ${OUTPUT_DIRECTORY}
		MOUNT_POINT_DEFINE NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT
		COMMON_OPTIONS ${COMPILE_OPTIONS}
		OUTPUT_VAR ALL_SPIRV_KEYS
		INCLUDE "nbl/this_example/builtin/build/spirv/generated/PathTracerKeys.hpp"
		NAMESPACE nbl::this_example::builtin::build
		INPUTS ${SPIRV_RULES_JSON}
	)
	set_target_properties(${EXECUTABLE_NAME}SPIRV PROPERTIES FOLDER "${SPIRV_FOLDER}")
	set(SPIRV_KEYS ${ALL_SPIRV_KEYS})
	list(FILTER SPIRV_KEYS INCLUDE REGEX "\\.spv$")

	NBL_CREATE_RESOURCE_ARCHIVE(
		NAMESPACE nbl::this_example::builtin::build
		TARGET ${EXECUTABLE_NAME}_builtinsBuild
		LINK_TO ${EXECUTABLE_NAME}
		BIND ${OUTPUT_DIRECTORY}
		BUILTINS ${SPIRV_KEYS}
	)
	set_target_properties(${EXECUTABLE_NAME}_builtinsBuild PROPERTIES FOLDER "${SPIRV_INTERNAL_FOLDER}")
	add_dependencies(${EXECUTABLE_NAME}_builtinsBuild ${EXECUTABLE_NAME}SPIRV)

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
		set_target_properties(${_BR_TARGET_} PROPERTIES FOLDER "${SPIRV_FOLDER}")
	endif()
endif()


