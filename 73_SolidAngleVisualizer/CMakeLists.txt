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
		Nabla::ext::FullScreenTriangle
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
		app_resources/hlsl/debug_vis.hlsl
		app_resources/hlsl/drawing.hlsl
		app_resources/hlsl/silhouette.hlsl
		app_resources/hlsl/utils.hlsl
		app_resources/hlsl/triangle_sampling.hlsl
		app_resources/hlsl/parallelogram_sampling.hlsl
		app_resources/hlsl/pyramid_sampling.hlsl

		app_resources/hlsl/pyramid_sampling/bilinear.hlsl
		app_resources/hlsl/pyramid_sampling/biquadratic.hlsl


		app_resources/hlsl/solid_angle_vis.frag.hlsl
		app_resources/hlsl/ray_vis.frag.hlsl

		app_resources/hlsl/benchmark/benchmark.comp.hlsl
		app_resources/hlsl/benchmark/common.hlsl
	)
	target_sources(${EXECUTABLE_NAME} PRIVATE ${DEPENDS})
	set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)

	set(SM 6_8)
	set(SA_VIS "app_resources/hlsl/solid_angle_vis.frag.hlsl")
	set(RAY_VIS "app_resources/hlsl/ray_vis.frag.hlsl")
	set(BENCH "app_resources/hlsl/benchmark/benchmark.comp.hlsl")

	set(JSON [=[
	[
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_tri_sa",      "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=0", "-DDEBUG_DATA=0", "-DVISUALIZE_SAMPLES=0"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_tri_sa_dbg",  "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=0", "-DDEBUG_DATA=1", "-DVISUALIZE_SAMPLES=1"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_tri_psa",     "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=1", "-DDEBUG_DATA=0", "-DVISUALIZE_SAMPLES=0"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_tri_psa_dbg", "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=1", "-DDEBUG_DATA=1", "-DVISUALIZE_SAMPLES=1"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_para",        "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=2", "-DDEBUG_DATA=0", "-DVISUALIZE_SAMPLES=0"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_para_dbg",    "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=2", "-DDEBUG_DATA=1", "-DVISUALIZE_SAMPLES=1"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_rectangle",     "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=3", "-DDEBUG_DATA=0", "-DVISUALIZE_SAMPLES=0"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_rectangle_dbg", "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=3", "-DDEBUG_DATA=1", "-DVISUALIZE_SAMPLES=1"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_biquad",      "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=4", "-DDEBUG_DATA=0", "-DVISUALIZE_SAMPLES=0"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_biquad_dbg",  "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=4", "-DDEBUG_DATA=1", "-DVISUALIZE_SAMPLES=1"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_bilinear",     "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=5", "-DDEBUG_DATA=0", "-DVISUALIZE_SAMPLES=0"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_bilinear_dbg", "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=5", "-DDEBUG_DATA=1", "-DVISUALIZE_SAMPLES=1"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_proj_rectangle",     "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=6", "-DDEBUG_DATA=0", "-DVISUALIZE_SAMPLES=0"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_proj_rectangle_dbg", "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=6", "-DDEBUG_DATA=1", "-DVISUALIZE_SAMPLES=1"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_silhouette",     "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=7", "-DDEBUG_DATA=0", "-DVISUALIZE_SAMPLES=0"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_silhouette_dbg", "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=7", "-DDEBUG_DATA=1", "-DVISUALIZE_SAMPLES=1"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_pyramid",        "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=8", "-DDEBUG_DATA=0", "-DVISUALIZE_SAMPLES=0"]},
		{"INPUT": "${SA_VIS}", "KEY": "sa_vis_pyramid_dbg",    "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-O3","-DSAMPLING_MODE_CONST=8", "-DDEBUG_DATA=1", "-DVISUALIZE_SAMPLES=1"]},

		{"INPUT": "${RAY_VIS}", "KEY": "ray_vis",     "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-DDEBUG_DATA=0", "-DVISUALIZE_SAMPLES=0"]},
		{"INPUT": "${RAY_VIS}", "KEY": "ray_vis_dbg", "COMPILE_OPTIONS": ["-T", "ps_${SM}", "-DDEBUG_DATA=1", "-DVISUALIZE_SAMPLES=1"]},

		{"INPUT": "${BENCH}", "KEY": "benchmark_tri_sa",  "COMPILE_OPTIONS": ["-T", "cs_${SM}", "-DSAMPLING_MODE_CONST=0"]},
		{"INPUT": "${BENCH}", "KEY": "benchmark_tri_psa", "COMPILE_OPTIONS": ["-T", "cs_${SM}", "-DSAMPLING_MODE_CONST=1"]},
		{"INPUT": "${BENCH}", "KEY": "benchmark_para",    "COMPILE_OPTIONS": ["-T", "cs_${SM}", "-DSAMPLING_MODE_CONST=2"]},
		{"INPUT": "${BENCH}", "KEY": "benchmark_rectangle", "COMPILE_OPTIONS": ["-T", "cs_${SM}", "-DSAMPLING_MODE_CONST=3"]},
		{"INPUT": "${BENCH}", "KEY": "benchmark_biquad",    "COMPILE_OPTIONS": ["-T", "cs_${SM}", "-DSAMPLING_MODE_CONST=4"]},
		{"INPUT": "${BENCH}", "KEY": "benchmark_bilinear", "COMPILE_OPTIONS": ["-T", "cs_${SM}", "-DSAMPLING_MODE_CONST=5"]},
		{"INPUT": "${BENCH}", "KEY": "benchmark_proj_rectangle", "COMPILE_OPTIONS": ["-T", "cs_${SM}", "-DSAMPLING_MODE_CONST=6"]},
		{"INPUT": "${BENCH}", "KEY": "benchmark_silhouette", "COMPILE_OPTIONS": ["-T", "cs_${SM}", "-DSAMPLING_MODE_CONST=7"]},
		{"INPUT": "${BENCH}", "KEY": "benchmark_pyramid_creation", "COMPILE_OPTIONS": ["-T", "cs_${SM}", "-DSAMPLING_MODE_CONST=8"]},
	]
	]=])
	string(CONFIGURE "${JSON}" JSON)

	set(COMPILE_OPTIONS
		-I "${CMAKE_CURRENT_SOURCE_DIR}"
		-Zi -Qembed_debug
		-fspv-debug=file
		-fspv-debug=source
		-fspv-debug=line
		-enable-16bit-types
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