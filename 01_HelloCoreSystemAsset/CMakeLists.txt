include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

nbl_create_executable_project("" "" "" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")

if(NBL_EMBED_BUILTIN_RESOURCES)
	set(_BR_TARGET_ ${EXECUTABLE_NAME}_builtinResourceData)

	# builtin resources with custom namespace test
	get_filename_component(_BUNDLE_SEARCH_DIRECTORY_ "${CMAKE_CURRENT_SOURCE_DIR}/builtin/include" ABSOLUTE)
	get_filename_component(_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/src" ABSOLUTE)
	get_filename_component(_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)

	LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "data/test.txt" "aliasTest1" "aliasTest2")

	if(NBL_STATIC_BUILD)
		ADD_CUSTOM_BUILTIN_RESOURCES(${_BR_TARGET_} RESOURCES_TO_EMBED "${_BUNDLE_SEARCH_DIRECTORY_}" "dir" "yourNamespace::builtin" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}")
	else()
		ADD_CUSTOM_BUILTIN_RESOURCES(${_BR_TARGET_} RESOURCES_TO_EMBED "${_BUNDLE_SEARCH_DIRECTORY_}" "dir" "yourNamespace::builtin" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}" "SHARED")
		target_compile_definitions(${EXECUTABLE_NAME} PUBLIC _BR_DLL_DIRECTORY_="$<TARGET_FILE_DIR:${_BR_TARGET_}>")
		target_compile_definitions(${EXECUTABLE_NAME} PUBLIC _BR_DLL_NAME_="$<TARGET_FILE_NAME:${_BR_TARGET_}>")
		target_link_options(${EXECUTABLE_NAME} PUBLIC "/DELAYLOAD:$<TARGET_FILE_NAME:${_BR_TARGET_}>")
	endif()

	LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} ${_BR_TARGET_})
endif()