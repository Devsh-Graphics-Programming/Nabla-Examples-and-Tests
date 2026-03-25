include(common)
include("${CMAKE_CURRENT_SOURCE_DIR}/pt.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/pt.variant_ids.cmake")

if(NBL_BUILD_IMGUI)
	# PT_BUILD_MODE controls how EX31 packages triangle polygon-method variants.
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
	set(PT_BUILD_MODE "WALLTIME_OPTIMIZED" CACHE STRING
		"Choose the EX31 precompiled shader layout. WALLTIME_OPTIMIZED keeps polygon-method selection inside the shared triangle shader module to reduce compile wall time. SPECIALIZED bakes triangle polygon methods into distinct entrypoints, which increases backend work and rebuild time but keeps those variants as separate precompiled entrypoints."
	)
	set(PT_CACHE_ROOT "pipeline/cache" CACHE STRING
		"Relative cache root written to path_tracer.runtime.json in the common bin directory. The runtime resolves this path relative to the JSON file location. Empty disables the generated dev-mode JSON and falls back to --pipeline-cache-dir or LocalAppData."
	)
	set_property(CACHE PT_BUILD_MODE PROPERTY STRINGS WALLTIME_OPTIMIZED SPECIALIZED)
	set(_PT_BUILD_MODE_VALUES WALLTIME_OPTIMIZED SPECIALIZED)
	if(NOT PT_BUILD_MODE IN_LIST _PT_BUILD_MODE_VALUES)
		message(FATAL_ERROR "Unsupported PT_BUILD_MODE='${PT_BUILD_MODE}'. Expected one of: ${_PT_BUILD_MODE_VALUES}")
	endif()
	if(IS_ABSOLUTE "${PT_CACHE_ROOT}")
		message(FATAL_ERROR "PT_CACHE_ROOT must stay relative because the runtime resolves it against path_tracer.runtime.json")
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
	add_dependencies(${EXECUTABLE_NAME} argparse)
	target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)
	target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PT_BUILD_CONFIG_NAME=\"$<CONFIG>\")
	if(PT_BUILD_MODE STREQUAL "SPECIALIZED")
		target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PT_BUILD_MODE_SPECIALIZED=1)
	else()
		target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PT_BUILD_MODE_WALLTIME_OPTIMIZED=1)
	endif()
	if(NOT PT_CACHE_ROOT STREQUAL "")
		string(REPLACE "\\" "/" PT_CACHE_ROOT_JSON "${PT_CACHE_ROOT}")
		configure_file(
			"${CMAKE_CURRENT_SOURCE_DIR}/path_tracer.runtime.json.in"
			"${CMAKE_CURRENT_BINARY_DIR}/path_tracer.runtime.json"
			@ONLY
		)
		file(GENERATE
			OUTPUT "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>/path_tracer.runtime.json"
			INPUT "${CMAKE_CURRENT_BINARY_DIR}/path_tracer.runtime.json"
		)
	endif()

	set(SM 6_8)
	set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
	set(COMPILE_OPTIONS
		-I "${CMAKE_CURRENT_SOURCE_DIR}/include"
		-I "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/hlsl"
		-I "${NBL_ROOT_PATH}/include" # workaround, the same thing like in IES I will address this issue later
		-T "lib_${SM}"
		-Wno-conversion
		-Wno-sign-conversion
		-Wno-float-conversion
		-Wno-shorten-64-to-32
		-Wno-shadow
		-Wno-literal-range
	)

	# Keep the payload flat and explicit here. Once Nabla PR #988 lands, these per-rule compile axes should move to first-class packaged-variant support there.
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.sphere.proxy.hlsl" KEY "pt.compute.sphere" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_SCENE_KIND=${PT_SCENE_SPHERE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=0")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.sphere.rwmc.proxy.hlsl" KEY "pt.compute.sphere.rwmc" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_SCENE_KIND=${PT_SCENE_SPHERE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=0")
	if(PT_BUILD_MODE STREQUAL "SPECIALIZED")
		PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.triangle.linear.proxy.hlsl" KEY "pt.compute.triangle.linear" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_LINEAR}")
		PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.triangle.persistent.proxy.hlsl" KEY "pt.compute.triangle.persistent" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_PERSISTENT}")
		PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.triangle.rwmc.linear.proxy.hlsl" KEY "pt.compute.triangle.rwmc.linear" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_LINEAR}")
		PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.triangle.rwmc.persistent.proxy.hlsl" KEY "pt.compute.triangle.rwmc.persistent" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_PERSISTENT}")
		PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.rwmc.linear.proxy.hlsl" KEY "pt.compute.rectangle.rwmc.linear" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_SCENE_KIND=${PT_SCENE_RECTANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_LINEAR}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
		PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.rwmc.persistent.proxy.hlsl" KEY "pt.compute.rectangle.rwmc.persistent" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_SCENE_KIND=${PT_SCENE_RECTANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_PERSISTENT}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
	else()
		PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.triangle.proxy.hlsl" KEY "pt.compute.triangle" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_SCENE_KIND=${PT_SCENE_TRIANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
		PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.triangle.rwmc.proxy.hlsl" KEY "pt.compute.triangle.rwmc" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_SCENE_KIND=${PT_SCENE_TRIANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
		PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.rwmc.proxy.hlsl" KEY "pt.compute.rectangle.rwmc" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_SCENE_KIND=${PT_SCENE_RECTANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
	endif()
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.proxy.hlsl" KEY "pt.compute.rectangle" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_SCENE_KIND=${PT_SCENE_RECTANGLE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_RUNTIME_UNIFORM}" "-DPT_VARIANT_RUNTIME_POLYGON_METHOD=1")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/resolve.comp.hlsl" KEY "pt.compute.resolve")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.misc.proxy.hlsl" KEY "pt.misc")
	PT_FINALIZE_JSON_PAYLOAD(INOUT JSON)

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
