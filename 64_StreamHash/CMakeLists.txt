nbl_create_executable_project("" "" "" "")
target_link_libraries("${EXECUTABLE_NAME}" PUBLIC blake3)

add_dependencies(${EXECUTABLE_NAME} argparse)
target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)

enable_testing()

add_test(NAME NBL_STREAM_HASH_RUN_TEST
	COMMAND "$<TARGET_FILE:${EXECUTABLE_NAME}>" --test
	WORKING_DIRECTORY "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>/.."
	COMMAND_EXPAND_LISTS
)