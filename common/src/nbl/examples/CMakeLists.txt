set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(ARGS
	# meta INTERFACE target with NSC compilation rules
	TARGET NblExtExamplesAPISPIRV

	# build directory for its SPIRV outputs
	BINARY_DIR ${OUTPUT_DIRECTORY}

	# preprocessor #define for BINARY_DIR bind point
	MOUNT_POINT_DEFINE NBL_EXAMPLES_BUILD_MOUNT_POINT

	# extra NSC compile options
	COMMON_OPTIONS -I "${COMMON_INCLUDE_DIRECTORY}"

	# out variable to which SPIRV access keys are appended to (including permutations), relative to BINARY_DIR
	OUTPUT_VAR KEYS

	# include file with inline template key getters, use with #include directive on downstream targets
	INCLUDE nbl/examples/common/build/spirv/keys.hpp

	# namespace for key getters in the include file
	NAMESPACE nbl::builtin::examples::build
)

# note json is array of objects, you can register all rules at once
set(JSON [=[
[
    {
		"INPUT": "shaders/geometry/unified.hlsl",
		"KEY": "shaders/geometry/unified",
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
		"KEY": "shaders/geometry/unified-caps",
        "COMPILE_OPTIONS": ["-T", "lib_6_6"],
		"DEPENDS": [],
		"CAPS": [
            {
                "name": "shaderFloat64",
				"type": "bool",
                "values": [1, 0]
            },
            {
                "name": "subgroupSize",
				"type": "uint16_t",
                "values": [32, 64]
            }
        ]
    },
	{
		"INPUT": "shaders/geometry/unified.hlsl",
		"KEY": "shaders/geometry/unified-caps-2",
        "COMPILE_OPTIONS": ["-T", "lib_6_6"],
		"DEPENDS": [],
		"CAPS": [
            {
                "name": "shaderFloat64",
				"type": "bool",
                "values": [1, 0]
            }
        ]
    }
]
]=])

# it also supports incremental rule updates, uncomment to add rules with permutation caps (testing purposes, remove after review)
# NBL_CREATE_NSC_COMPILE_RULES(${ARGS} INPUTS ${JSON})

# note we can add more inputs from build dir which keys can be part of the same archive/mount point,
# ex. one could auto generate bc texture or whatever and add here like
# file(WRITE "${OUTPUT_DIRECTORY}/dummy.txt" "dummy, test")
# list(APPEND KEYS dummy.txt)

NBL_CREATE_RESOURCE_ARCHIVE(
	TARGET NblExtExamplesAPIBuiltinsBuild
	BIND "${OUTPUT_DIRECTORY}"
	BUILTINS ${KEYS}
	NAMESPACE nbl::builtin::examples::build
)
