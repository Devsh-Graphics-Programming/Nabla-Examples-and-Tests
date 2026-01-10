include(common)

nbl_create_executable_project("" "" "" "")

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
file(MAKE_DIRECTORY "${OUTPUT_DIRECTORY}")

enable_testing()

add_test(NAME ${EXECUTABLE_NAME}
	COMMAND "$<TARGET_FILE:${EXECUTABLE_NAME}>"
	WORKING_DIRECTORY "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>"
	COMMAND_EXPAND_LISTS
)

set(HLSL_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/input.hlsl")
file(TO_CMAKE_PATH "${HLSL_INPUT}" HLSL_INPUT)

set(JSON_TEMPLATE [=[
[
  {
    "INPUT": "@HLSL_INPUT@",
    "KEY": "k_plain",
    "COMPILE_OPTIONS": ["-T", "cs_6_8"]
  },
  {
    "INPUT": "@HLSL_INPUT@",
    "KEY": "k_ints",
    "COMPILE_OPTIONS": ["-T", "cs_6_8", "-DNBL_TEST_HAS_I"],
    "CAPS": [
      {
        "kind": "custom",
        "struct": "i",
        "members": [
          { "name": "b1", "type": "bool", "values": [1] },
          { "name": "b0", "type": "bool", "values": [0] },
          { "name": "u16", "type": "uint16_t", "values": [65535] },
          { "name": "u32", "type": "uint32_t", "values": [123456789] },
          { "name": "u64", "type": "uint64_t", "values": [1234567890123] },
          { "name": "s16", "type": "int16_t", "values": [-32768] },
          { "name": "s32", "type": "int32_t", "values": [-123456789] },
          { "name": "s64", "type": "int64_t", "values": [-1234567890123] }
        ]
      }
    ]
  },
  {
    "INPUT": "@HLSL_INPUT@",
    "KEY": "k_two",
    "COMPILE_OPTIONS": ["-T", "cs_6_8", "-DNBL_TEST_HAS_T"],
    "CAPS": [
      {
        "kind": "custom",
        "struct": "t",
        "members": [
          { "name": "sel", "type": "uint32_t", "values": [0, 1] }
        ]
      }
    ]
  },
  {
    "INPUT": "@HLSL_INPUT@",
    "KEY": "k_f",
    "COMPILE_OPTIONS": ["-T", "cs_6_8", "-DNBL_TEST_HAS_F"],
    "CAPS": [
      {
        "kind": "custom",
        "struct": "f",
        "members": [
          { "name": "min", "type": "float", "values": [1.17549435e-38] },
          { "name": "max", "type": "float", "values": [3.40282347e+38] },
          { "name": "neg", "type": "float", "values": [-1] },
          { "name": "exp", "type": "float", "values": ["1.25e-1"] }
        ]
      }
    ]
  },
  {
    "INPUT": "@HLSL_INPUT@",
    "KEY": "k_d",
    "COMPILE_OPTIONS": ["-T", "cs_6_8", "-DNBL_TEST_HAS_D"],
    "CAPS": [
      {
        "kind": "custom",
        "struct": "d",
        "members": [
          { "name": "min", "type": "double", "values": ["2.2250738585072010e-308"] },
          { "name": "max", "type": "double", "values": ["1.7976931348623165e+308"] },
          { "name": "neg", "type": "double", "values": [-1.0] },
          { "name": "exp", "type": "double", "values": ["1.25e-1"] }
        ]
      }
    ]
  },
  {
    "INPUT": "@HLSL_INPUT@",
    "KEY": "k_mix",
    "COMPILE_OPTIONS": ["-T", "cs_6_8", "-DNBL_TEST_HAS_M", "-DNBL_TEST_HAS_LIMITS", "-DNBL_TEST_HAS_FEATURES"],
    "CAPS": [
      {
        "kind": "custom",
        "struct": "m",
        "members": [
          { "name": "md", "type": "uint32_t", "values": [7] },
          { "name": "en", "type": "bool", "values": [1] },
          { "name": "sc", "type": "float", "values": [1] }
        ]
      },
      {
        "kind": "limits",
        "members": [
          { "name": "maxImageDimension2D", "type": "uint32_t", "values": [16384] }
        ]
      },
      {
        "kind": "features",
        "members": [
          { "name": "shaderCullDistance", "type": "bool", "values": [1] }
        ]
      }
    ]
  }
]
]=])
string(CONFIGURE "${JSON_TEMPLATE}" JSON @ONLY)

NBL_CREATE_NSC_COMPILE_RULES(
  TARGET ${EXECUTABLE_NAME}SPIRV
  LINK_TO ${EXECUTABLE_NAME}
  BINARY_DIR ${OUTPUT_DIRECTORY}
  MOUNT_POINT_DEFINE NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT
  COMMON_OPTIONS -I ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VAR KEYS
  INCLUDE nbl/this_example/builtin/build/spirv/keys.hpp
  NAMESPACE nbl::this_example::builtin::build
  INPUTS ${JSON}
  DISCARD_DEFAULT_GLOB
)
