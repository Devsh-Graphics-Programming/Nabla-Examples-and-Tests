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
	# On AMD Ryzen 5 5600G with Radeon Graphics (6C/12T),
	# a Visual Studio Debug x64 full rebuild of the SPIR-V project completed in:
	#   WALLTIME_OPTIMIZED = 12.785 s
	#   SPECIALIZED = 18.314 s
	# SPECIALIZED is +5.529 s slower which is +43.25%.
	# Equivalently, WALLTIME_OPTIMIZED is 30.19% faster than SPECIALIZED.
	# This comes from multiplying the heavy triangle-side path tracing instantiations
	# and pushing more work into the DXC/SPIR-V backend.
	# Therefore the default stays WALLTIME_OPTIMIZED.
	set(PATH_TRACER_BUILD_MODE "WALLTIME_OPTIMIZED" CACHE STRING
		"Choose the EX31 precompiled shader layout. WALLTIME_OPTIMIZED keeps polygon-method selection inside the shared triangle shader module to reduce compile wall time. SPECIALIZED bakes triangle polygon methods into distinct entrypoints, which increases backend work and rebuild time but keeps those variants as separate precompiled entrypoints."
	)
	set(PATH_TRACER_CACHE_ROOT "pipeline/cache" CACHE STRING
		"Relative cache root written to path_tracer.runtime.json in the common bin directory. The runtime resolves this path relative to the JSON file location. Empty disables the generated dev-mode JSON and falls back to --pipeline-cache-dir or LocalAppData."
	)
	set_property(CACHE PATH_TRACER_BUILD_MODE PROPERTY STRINGS WALLTIME_OPTIMIZED SPECIALIZED)
	set(_PATH_TRACER_BUILD_MODE_VALUES WALLTIME_OPTIMIZED SPECIALIZED)
	if(NOT PATH_TRACER_BUILD_MODE IN_LIST _PATH_TRACER_BUILD_MODE_VALUES)
		message(FATAL_ERROR "Unsupported PATH_TRACER_BUILD_MODE='${PATH_TRACER_BUILD_MODE}'. Expected one of: ${_PATH_TRACER_BUILD_MODE_VALUES}")
	endif()
	if(IS_ABSOLUTE "${PATH_TRACER_CACHE_ROOT}")
		message(FATAL_ERROR "PATH_TRACER_CACHE_ROOT must stay relative because the runtime resolves it against path_tracer.runtime.json")
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
	target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PATH_TRACER_BUILD_CONFIG_NAME=\"$<CONFIG>\")
	if(PATH_TRACER_BUILD_MODE STREQUAL "SPECIALIZED")
		target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PATH_TRACER_BUILD_MODE_SPECIALIZED=1)
	else()
		target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PATH_TRACER_BUILD_MODE_WALLTIME_OPTIMIZED=1)
	endif()
	if(NOT PATH_TRACER_CACHE_ROOT STREQUAL "")
		string(REPLACE "\\" "/" PATH_TRACER_CACHE_ROOT_JSON "${PATH_TRACER_CACHE_ROOT}")
		file(GENERATE
			OUTPUT "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>/path_tracer.runtime.json"
			CONTENT "{\n  \"cache_root\": \"${PATH_TRACER_CACHE_ROOT_JSON}\"\n}\n"
		)
		unset(PATH_TRACER_CACHE_ROOT_JSON)
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
	macro(PATH_TRACER_APPEND_SPIRV_RULE)
		set(options)
		set(oneValueArgs INPUT KEY)
		set(multiValueArgs COMPILE_OPTIONS)
		cmake_parse_arguments(PATH_TRACER_RULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
		if(PATH_TRACER_RULE_INPUT STREQUAL "" OR PATH_TRACER_RULE_KEY STREQUAL "")
			message(FATAL_ERROR "PATH_TRACER_APPEND_SPIRV_RULE requires INPUT and KEY")
		endif()
		set(PATH_TRACER_RULE_JSON "{\"INPUT\":\"${PATH_TRACER_RULE_INPUT}\",\"KEY\":\"${PATH_TRACER_RULE_KEY}\"")
		if(PATH_TRACER_RULE_COMPILE_OPTIONS)
			set(PATH_TRACER_RULE_COMPILE_OPTIONS_JSON "")
			foreach(PATH_TRACER_RULE_COMPILE_OPTION IN LISTS PATH_TRACER_RULE_COMPILE_OPTIONS)
				string(APPEND PATH_TRACER_RULE_COMPILE_OPTIONS_JSON "\"${PATH_TRACER_RULE_COMPILE_OPTION}\",")
			endforeach()
			string(REGEX REPLACE ",$" "" PATH_TRACER_RULE_COMPILE_OPTIONS_JSON "${PATH_TRACER_RULE_COMPILE_OPTIONS_JSON}")
			string(APPEND PATH_TRACER_RULE_JSON ",\"COMPILE_OPTIONS\":[${PATH_TRACER_RULE_COMPILE_OPTIONS_JSON}]")
		endif()
		string(APPEND PATH_TRACER_RULE_JSON "}")
		list(APPEND SPIRV_RULE_OBJECTS "${PATH_TRACER_RULE_JSON}")
		unset(PATH_TRACER_RULE_JSON)
		unset(PATH_TRACER_RULE_COMPILE_OPTIONS_JSON)
	endmacro()

	set(SPIRV_RULE_OBJECTS "")
	set(PATH_TRACER_SCENE_SPHERE 0)
	set(PATH_TRACER_SCENE_TRIANGLE 1)
	set(PATH_TRACER_SCENE_RECTANGLE 2)
	set(PATH_TRACER_ENTRYPOINT_RUNTIME_UNIFORM 0)
	set(PATH_TRACER_ENTRYPOINT_LINEAR 1)
	set(PATH_TRACER_ENTRYPOINT_PERSISTENT 2)
	# Keep the payload flat and explicit here. Once Nabla PR #988 lands, these per-rule compile axes should move to first-class packaged-variant support there.
	PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.sphere.proxy.hlsl" KEY "pt.compute.sphere" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_SCENE_KIND=${PATH_TRACER_SCENE_SPHERE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=0")
	PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.sphere.rwmc.proxy.hlsl" KEY "pt.compute.sphere.rwmc" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_SCENE_KIND=${PATH_TRACER_SCENE_SPHERE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=0")
	if(PATH_TRACER_BUILD_MODE STREQUAL "SPECIALIZED")
		PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.triangle.linear.proxy.hlsl" KEY "pt.compute.triangle.linear" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_LINEAR}")
		PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.triangle.persistent.proxy.hlsl" KEY "pt.compute.triangle.persistent" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_PERSISTENT}")
		PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.triangle.rwmc.linear.proxy.hlsl" KEY "pt.compute.triangle.rwmc.linear" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_LINEAR}")
		PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.triangle.rwmc.persistent.proxy.hlsl" KEY "pt.compute.triangle.rwmc.persistent" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_PERSISTENT}")
		PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.rwmc.linear.proxy.hlsl" KEY "pt.compute.rectangle.rwmc.linear" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_SCENE_KIND=${PATH_TRACER_SCENE_RECTANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_LINEAR}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
		PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.rwmc.persistent.proxy.hlsl" KEY "pt.compute.rectangle.rwmc.persistent" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_SCENE_KIND=${PATH_TRACER_SCENE_RECTANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_PERSISTENT}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
	else()
		PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.triangle.proxy.hlsl" KEY "pt.compute.triangle" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_SCENE_KIND=${PATH_TRACER_SCENE_TRIANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
		PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.triangle.rwmc.proxy.hlsl" KEY "pt.compute.triangle.rwmc" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_SCENE_KIND=${PATH_TRACER_SCENE_TRIANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
		PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.rwmc.proxy.hlsl" KEY "pt.compute.rectangle.rwmc" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_SCENE_KIND=${PATH_TRACER_SCENE_RECTANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
	endif()
	PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.proxy.hlsl" KEY "pt.compute.rectangle" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_SCENE_KIND=${PATH_TRACER_SCENE_RECTANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PATH_TRACER_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
	PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/resolve.comp.hlsl" KEY "pt.compute.resolve")
	PATH_TRACER_APPEND_SPIRV_RULE(INPUT "app_resources/hlsl/spirv/pt.misc.proxy.hlsl" KEY "pt.misc")
	string(JOIN ",\n" PATH_TRACER_RULES_BODY ${SPIRV_RULE_OBJECTS})
	set(SPIRV_RULES_JSON_TEMPLATE [=[
[
@PATH_TRACER_RULES_BODY@
]
]=])
	string(CONFIGURE "${SPIRV_RULES_JSON_TEMPLATE}" SPIRV_RULES_JSON @ONLY)
	unset(SPIRV_RULE_OBJECTS)
	unset(PATH_TRACER_RULES_BODY)
	set(ALL_SPIRV_KEYS "")
	NBL_CREATE_NSC_COMPILE_RULES(
		DISCARD_DEFAULT_GLOB
		TARGET ${EXECUTABLE_NAME}SPIRV
		LINK_TO ${EXECUTABLE_NAME}
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


