if(TARGET Nabla::ext::FullScreenTriangle AND TARGET Nabla::ext::Cameras AND TARGET Nabla::ext::ImGUI)
	include("${NBL_ROOT_PATH}/cmake/nam/nam.cmake")

	file(GLOB_RECURSE NBL_EXTRA_SOURCES CONFIGURE_DEPENDS
		"${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.inl"
		"${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/include/*.inl"
		"${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/src/*.inl"
	)

	set(NBL_INCLUDE_SERACH_DIRECTORIES
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
	)

	list(APPEND NBL_LIBRARIES
		argparse
		Nabla::ext::ImGUI
		Nabla::ext::FullScreenTriangle
		Nabla::ext::Cameras
	)

	nbl_create_executable_project("${NBL_EXTRA_SOURCES}" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}")

	nam_add_channel_target(
		TARGET ${EXECUTABLE_NAME}_envmaps
		CHANNEL envmaps
		MANIFEST_ROOT "${NBL_ROOT_PATH}/examples_tests/manifests"
		REPO Devsh-Graphics-Programming/Nabla-Asset-Manifests
		TAG envmaps
		DESTINATION_ROOT "${NBL_ROOT_PATH}/examples_tests/runtime"
		FLAT_RELEASE_ASSET_NAMES
		ITEMS
			space_spheremaps/LICENSE.txt
			space_spheremaps/rich_blue_nebulae_1_8k.rgba16f.envblob
	)
	add_dependencies(${EXECUTABLE_NAME} ${EXECUTABLE_NAME}_envmaps)

	set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
	file(GLOB_RECURSE NBL_HLSL_SOURCES CONFIGURE_DEPENDS
		"${CMAKE_CURRENT_SOURCE_DIR}/app_resources/*.hlsl"
	)
	target_sources(${EXECUTABLE_NAME} PRIVATE ${NBL_HLSL_SOURCES})
	set_source_files_properties(${NBL_HLSL_SOURCES} PROPERTIES HEADER_FILE_ONLY ON)

	set(SM 6_7)

	set(JSON [=[
[
	{
		"INPUT": "app_resources/imgui.unified.hlsl",
		"KEY": "imgui.unified"
	},
	{
		"INPUT": "app_resources/sky_env_fragment.hlsl",
		"KEY": "sky_env_fragment"
	}
]
]=])
	string(CONFIGURE "${JSON}" JSON)

	set(COMPILE_OPTIONS
		-isystem "${NBL_ROOT_PATH}/include"
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

	enable_testing()

	set(CAMERA_CONTINUITY_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/cameraz_continuity.json")

	add_test(NAME NBL_61_UI_CAMERA_SMOKE
		COMMAND "$<TARGET_FILE:${EXECUTABLE_NAME}>" --headless-camera-smoke --file "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/cameras.json"
		WORKING_DIRECTORY "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>"
		COMMAND_EXPAND_LISTS
	)
	add_test(NAME NBL_61_UI_CAMERA_CONTINUITY
		COMMAND "$<TARGET_FILE:${EXECUTABLE_NAME}>" --ci --script "${CAMERA_CONTINUITY_SCRIPT}" --script-visual-debug --no-screenshots
		WORKING_DIRECTORY "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>"
		COMMAND_EXPAND_LISTS
	)

	set_tests_properties(
		NBL_61_UI_CAMERA_SMOKE
		NBL_61_UI_CAMERA_CONTINUITY
		PROPERTIES
		RUN_SERIAL TRUE
		TIMEOUT 300
	)
endif()
