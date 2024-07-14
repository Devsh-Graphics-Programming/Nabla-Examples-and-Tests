nbl_create_executable_project("" "" "" "")
target_link_libraries("${EXECUTABLE_NAME}" PUBLIC blake3)

add_dependencies(${EXECUTABLE_NAME} argparse)
target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)