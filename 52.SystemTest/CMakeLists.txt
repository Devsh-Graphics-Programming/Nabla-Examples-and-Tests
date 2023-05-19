
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

nbl_create_executable_project("" "" "" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")

# builtin resources with custom namespace test
get_filename_component(_BUNDLE_SEARCH_DIRECTORY_ "${CMAKE_CURRENT_SOURCE_DIR}/builtin/include" ABSOLUTE)
get_filename_component(_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/src" ABSOLUTE)
get_filename_component(_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)

LIST_BUILTIN_RESOURCE(SYSTEM_TEST_RESOURCES_TO_EMBED "dir/data/test.txt" "aliasTest1" "aliasTest2")
ADD_CUSTOM_BUILTIN_RESOURCES(systemTestBuiltinResourceData SYSTEM_TEST_RESOURCES_TO_EMBED "${_BUNDLE_SEARCH_DIRECTORY_}" "yourNamespace::builtin" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}")
LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} systemTestBuiltinResourceData)