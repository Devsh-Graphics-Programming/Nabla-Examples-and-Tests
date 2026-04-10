include(common)
include("${CMAKE_CURRENT_SOURCE_DIR}/pt.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/pt.variant_ids.cmake")

if(NBL_BUILD_IMGUI)
	# EX31 keeps triangle polygon-method variants as separate precompiled entrypoints.
	# This keeps polygon-method choice compile-time and avoids runtime shader switching on this axis.
	# On AMD Ryzen 5 5600G with Radeon Graphics (6C/12T),
	# a Visual Studio Debug x64 full rebuild of the SPIR-V project completed in about 19.789 s.
	set(PT_CACHE_ROOT "pipeline/cache" CACHE STRING
		"Relative cache root written to path_tracer.runtime.json in the common bin directory. The runtime resolves this path relative to the JSON file location. Empty disables the generated dev-mode JSON and falls back to --pipeline-cache-dir or LocalAppData."
	)
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
	target_compile_definitions(${EXECUTABLE_NAME} PRIVATE PATH_TRACER_BUILD_CONFIG_NAME=\"$<CONFIG>\")
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
		-isystem "${NBL_ROOT_PATH}/include" # workaround, the same thing like in IES I will address this issue later
		$<$<CONFIG:Debug>:-O1experimental>
		$<$<CONFIG:RelWithDebInfo>:-O1experimental>
		-T "lib_${SM}"
		-Wno-conversion
		-Wno-sign-conversion
		-Wno-float-conversion
		-Wno-shorten-64-to-32
		-Wno-shadow
		-Wno-literal-range
	)

	# Keep the payload flat and explicit here. Once Nabla PR #988 lands, these per-rule compile axes should move to first-class packaged-variant support there.
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.sphere.proxy.hlsl" KEY "pt.compute.sphere" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_SCENE_KIND=${PT_SCENE_SPHERE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_COMBINED}")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.sphere.rwmc.proxy.hlsl" KEY "pt.compute.sphere.rwmc" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_SCENE_KIND=${PT_SCENE_SPHERE}" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_COMBINED}")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.triangle.linear.proxy.hlsl" KEY "pt.compute.triangle.linear" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_LINEAR}")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.triangle.persistent.proxy.hlsl" KEY "pt.compute.triangle.persistent" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_PERSISTENT}")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.triangle.rwmc.linear.proxy.hlsl" KEY "pt.compute.triangle.rwmc.linear" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_LINEAR}")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.triangle.rwmc.persistent.proxy.hlsl" KEY "pt.compute.triangle.rwmc.persistent" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_PERSISTENT}")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.linear.proxy.hlsl" KEY "pt.compute.rectangle.linear" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_LINEAR}")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.persistent.proxy.hlsl" KEY "pt.compute.rectangle.persistent" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=0" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_PERSISTENT}")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.rwmc.linear.proxy.hlsl" KEY "pt.compute.rectangle.rwmc.linear" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_LINEAR}")
	PT_APPEND_SPIRV_RULE(VAR JSON INPUT "app_resources/hlsl/spirv/pt.compute.rectangle.rwmc.persistent.proxy.hlsl" KEY "pt.compute.rectangle.rwmc.persistent" COMPILE_OPTIONS "-DPT_VARIANT_USE_RWMC=1" "-DPT_VARIANT_ENTRYPOINT_KIND=${PT_ENTRYPOINT_PERSISTENT}")
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
