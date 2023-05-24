
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

nbl_create_executable_project("" "" "" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")

# builtin resources with custom namespace test
get_filename_component(_BUNDLE_SEARCH_DIRECTORY_ "${CMAKE_CURRENT_SOURCE_DIR}/builtin/include" ABSOLUTE)
get_filename_component(_OUTPUT_DIRECTORY_ "${CMAKE_CURRENT_BINARY_DIR}/include/yourNamespace/builtin" ABSOLUTE)

LIST_BUILTIN_RESOURCE(SYSTEM_TEST_RESOURCES_TO_EMBED "yourNamespace/data/test.txt")
ADD_CUSTOM_BUILTIN_RESOURCES(systemTestBuiltinResourceData SYSTEM_TEST_RESOURCES_TO_EMBED "${_BUNDLE_SEARCH_DIRECTORY_}" "yourNamespace::builtin" "${_OUTPUT_DIRECTORY_}")

unset(_BUNDLE_SEARCH_DIRECTORY_)
unset(_OUTPUT_DIRECTORY_)

add_dependencies(${EXECUTABLE_NAME} systemTestBuiltinResourceData)
target_link_libraries(${EXECUTABLE_NAME} PUBLIC systemTestBuiltinResourceData)
target_include_directories(${EXECUTABLE_NAME} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include")