
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

nbl_create_executable_project("" "" "" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")

if(NBL_EMBED_BUILTIN_RESOURCES)
	set(_BR_TARGET_ ${EXECUTABLE_NAME}_builtinResourceData)
	
	# builtin resources with custom namespace test
	get_filename_component(_BUNDLE_SEARCH_DIRECTORY_ "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
	get_filename_component(_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/src" ABSOLUTE)
	get_filename_component(_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)

	# look below that the built-in archive will have a mount point of "example_data" so we need to take into account these files be be relative to that dir 
	LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "shader.vert")
	LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "shader.frag")

	# always embed into the EXE
	ADD_CUSTOM_BUILTIN_RESOURCES(${_BR_TARGET_} RESOURCES_TO_EMBED "${_BUNDLE_SEARCH_DIRECTORY_}" "example_data" "example_data::builtin" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}")

	LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} ${_BR_TARGET_})
endif()