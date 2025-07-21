include(common)

nbl_create_executable_project("" "" "" "")

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
file(GLOB_RECURSE DEPENDS app_resources/*.hlsl)
list(FILTER DEPENDS EXCLUDE REGEX "preprocessed\\.hlsl$")

set(JSON [=[
[
    {
		"INPUT": "app_resources/shader.comp.hlsl",
		"KEY": "shader",
        "COMPILE_OPTIONS": ["-T", "cs_6_8"],
		"DEPENDS": [],
		"CAPS": []
    }
]
]=])

NBL_CREATE_NSC_COMPILE_RULES(
	TARGET ${EXECUTABLE_NAME}SPIRV
	LINK_TO ${EXECUTABLE_NAME}
	DEPENDS ${DEPENDS}
	BINARY_DIR ${OUTPUT_DIRECTORY}
	MOUNT_POINT_DEFINE NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT
	COMMON_OPTIONS -I "${CMAKE_CURRENT_SOURCE_DIR}"
	OUTPUT_VAR KEYS
	INCLUDE nbl/this_example/builtin/build/spirv/keys.hpp
	NAMESPACE nbl::this_example::builtin::build
	INPUTS ${JSON}
)

NBL_CREATE_RESOURCE_ARCHIVE(
	TARGET ${EXECUTABLE_NAME}_builtinsBuild
	LINK_TO ${EXECUTABLE_NAME}
	BIND "${OUTPUT_DIRECTORY}"
	BUILTINS ${KEYS}
	NAMESPACE nbl::this_example::builtin::build
)