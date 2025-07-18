include(common)

nbl_create_executable_project("" "" "" "")

get_filename_component(MOUNT_POINT "${CMAKE_CURRENT_SOURCE_DIR}/app_resources" ABSOLUTE)
file(GLOB_RECURSE KEYS RELATIVE ${MOUNT_POINT} CONFIGURE_DEPENDS app_resources/*.hlsl)
NBL_CREATE_RESOURCE_ARCHIVE(
	TARGET ${EXECUTABLE_NAME}_builtins
	LINK_TO ${EXECUTABLE_NAME}
	BIND ${MOUNT_POINT}
	BUILTINS ${KEYS}
	NAMESPACE nbl::this_example::builtin
)