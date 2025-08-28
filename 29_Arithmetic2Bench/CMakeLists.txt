include(common)

nbl_create_executable_project("" "" "" "")

NBL_CREATE_RESOURCE_ARCHIVE(
	NAMESPACE nbl::this_example::builtin
	TARGET ${EXECUTABLE_NAME}_builtins
	LINK_TO ${EXECUTABLE_NAME}
	BIND app_resources
	BUILTINS
		benchmarkSubgroup.comp.hlsl
		benchmarkWorkgroup.comp.hlsl
		common.hlsl
		shaderCommon.hlsl
)