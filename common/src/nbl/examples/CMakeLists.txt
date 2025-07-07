set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(ARGS 
	TARGET NblExtExamplesAPISPIRV
	BINARY_DIR ${OUTPUT_DIRECTORY}
	COMMON_OPTIONS -I "${COMMON_INCLUDE_DIRECTORY}"
	OUTPUT_VAR KEYS
)

# note json is array of objects, you can register all rules at once
set(JSON [=[
[
    {
		"INPUT": "shaders/geometry/unified.hlsl",
		"KEY": "shaders/geometry/unified.spv",
        "COMPILE_OPTIONS": ["-T", "lib_6_6"],
		"DEPENDS": [],
		"CAPS": []
    }
]
]=])

NBL_CREATE_NSC_COMPILE_RULES(${ARGS} INPUTS ${JSON})

set(JSON [=[
[
    {
		"INPUT": "shaders/geometry/unified.hlsl",
		"KEY": "shaders/geometry/unified.spv",
        "COMPILE_OPTIONS": ["-T", "lib_6_6"],
		"DEPENDS": [],
		"CAPS": [
            {
                "name": "shaderFloat64",
				"type": "bool",
                "values": ["true", "false"]
            },
            {
                "name": "subgroupSize",
				"type": "uint16_t",
                "values": ["32", "64"]
            }
        ]
    }
]
]=])

# but it also supports incremental rule updates, uncomment to add rules with permutation caps
# NBL_CREATE_NSC_COMPILE_RULES(${ARGS} INPUTS ${JSON})

# note we can add more inputs from build dir which keys can be part of the same archive/mount point,
# ex. one could auto generate bc texture or whatever and add here like
# file(WRITE "${OUTPUT_DIRECTORY}/dummy.txt" "dummy, test")
# list(APPEND KEYS dummy.txt)

NBL_CREATE_RESOURCE_ARCHIVE(
	TARGET NblExtExamplesAPIBuiltinsBuild
	BIND "${OUTPUT_DIRECTORY}"
	NAMESPACE nbl::builtin::examples::build
	MOUNT_POINT_DEFINE NBL_EXAMPLES_BUILD_MOUNT_POINT
	BUILTINS ${KEYS}
)

if(NBL_EMBED_BUILTIN_RESOURCES)
	INTERFACE_TO_BUILTINS(NblExtExamplesAPIBuiltinsBuild)
endif()

target_link_libraries(NblExtExamplesAPIBuiltinsBuild PUBLIC NblExtExamplesAPISPIRV)
