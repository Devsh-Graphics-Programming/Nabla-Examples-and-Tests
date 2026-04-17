set(NBL_INCLUDE_SEARCH_DIRECTORIES
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

include(common RESULT_VARIABLE RES)

if(NOT RES)
  message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

nbl_create_executable_project("" "" "${NBL_INCLUDE_SEARCH_DIRECTORIES}" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")

set(DEPENDS
  app_resources/shaders/test_compile.comp.hlsl
  app_resources/shaders/linear_test.comp.hlsl
  app_resources/shaders/uniform_hemisphere_test.comp.hlsl
  app_resources/shaders/uniform_sphere_test.comp.hlsl
  app_resources/shaders/projected_hemisphere_test.comp.hlsl
  app_resources/shaders/projected_sphere_test.comp.hlsl
  app_resources/shaders/spherical_triangle.comp.hlsl
  app_resources/shaders/concentric_mapping_test.comp.hlsl
  app_resources/shaders/polar_mapping_test.comp.hlsl
  app_resources/shaders/bilinear_test.comp.hlsl
  app_resources/shaders/box_muller_transform_test.comp.hlsl
  app_resources/shaders/projected_spherical_triangle_test.comp.hlsl
  app_resources/shaders/projected_spherical_rectangle_test.comp.hlsl
  app_resources/shaders/spherical_rectangle_test.comp.hlsl
  app_resources/shaders/alias_table_test.comp.hlsl
  app_resources/shaders/cumulative_probability_test.comp.hlsl
  app_resources/common/linear.hlsl
  app_resources/common/uniform_hemisphere.hlsl
  app_resources/common/uniform_sphere.hlsl
  app_resources/common/projected_hemisphere.hlsl
  app_resources/common/projected_sphere.hlsl
  app_resources/common/spherical_triangle.hlsl
  app_resources/common/bilinear.hlsl
  app_resources/common/box_muller_transform.hlsl
  app_resources/common/projected_spherical_triangle.hlsl
  app_resources/common/projected_spherical_rectangle.hlsl
  app_resources/common/spherical_rectangle.hlsl
  app_resources/common/concentric_mapping.hlsl
  app_resources/common/polar_mapping.hlsl
  app_resources/common/discrete_sampler_bench.hlsl
  app_resources/common/alias_table.hlsl
  app_resources/common/cumulative_probability.hlsl
)
list(TRANSFORM DEPENDS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

target_sources(${EXECUTABLE_NAME} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CLinearTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CUniformHemisphereTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CUniformSphereTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CProjectedHemisphereTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CProjectedSphereTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CSphericalTriangleTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CConcentricMappingTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CPolarMappingTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CBilinearTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CBoxMullerTransformTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CProjectedSphericalTriangleTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CProjectedSphericalRectangleTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CSphericalRectangleTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CDiscreteTableTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CAliasTableGPUTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/CCumulativeProbabilityGPUTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/SamplerTestHelpers.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/property/CSamplerPropertyTester.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CSamplerBenchmark.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CDiscreteSamplerBenchmark.h"
  ${DEPENDS}
)

if(NBL_EMBED_BUILTIN_RESOURCES)
  set(_BR_TARGET_ ${EXECUTABLE_NAME}_builtinResourceData)
  set(RESOURCE_DIR "app_resources")

  get_filename_component(_SEARCH_DIRECTORIES_ "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
  get_filename_component(_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/src" ABSOLUTE)
  get_filename_component(_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)

  file(GLOB_RECURSE BUILTIN_RESOURCE_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_DIR}" CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_DIR}/*")
  foreach(RES_FILE ${BUILTIN_RESOURCE_FILES})
    LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "${RES_FILE}")
  endforeach()

  ADD_CUSTOM_BUILTIN_RESOURCES(${_BR_TARGET_} RESOURCES_TO_EMBED "${_SEARCH_DIRECTORIES_}" "${RESOURCE_DIR}" "nbl::this_example::builtin" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}")

  LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} ${_BR_TARGET_})
endif()

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")

set(BENCH_ITERS 2048)
set(WORKGROUP_SIZE 64)

target_compile_definitions(${EXECUTABLE_NAME} PRIVATE
  BENCH_ITERS=${BENCH_ITERS}
  WORKGROUP_SIZE=${WORKGROUP_SIZE}
)

set(BENCH_OPTS "\"-DBENCH_ITERS=${BENCH_ITERS}\", \"-DWORKGROUP_SIZE=${WORKGROUP_SIZE}\"")

set(JSON "
[
  {
    \"INPUT\": \"app_resources/shaders/test_compile.comp.hlsl\",
    \"KEY\": \"shader\"
  },
  {
    \"INPUT\": \"app_resources/shaders/linear_test.comp.hlsl\",
    \"KEY\": \"linear_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/linear_test.comp.hlsl\",
    \"KEY\": \"linear_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/uniform_hemisphere_test.comp.hlsl\",
    \"KEY\": \"uniform_hemisphere_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/uniform_hemisphere_test.comp.hlsl\",
    \"KEY\": \"uniform_hemisphere_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/uniform_sphere_test.comp.hlsl\",
    \"KEY\": \"uniform_sphere_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/uniform_sphere_test.comp.hlsl\",
    \"KEY\": \"uniform_sphere_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/projected_hemisphere_test.comp.hlsl\",
    \"KEY\": \"projected_hemisphere_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/projected_hemisphere_test.comp.hlsl\",
    \"KEY\": \"projected_hemisphere_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/projected_sphere_test.comp.hlsl\",
    \"KEY\": \"projected_sphere_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/projected_sphere_test.comp.hlsl\",
    \"KEY\": \"projected_sphere_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/spherical_triangle.comp.hlsl\",
    \"KEY\": \"spherical_triangle\"
  },
  {
    \"INPUT\": \"app_resources/shaders/spherical_triangle.comp.hlsl\",
    \"KEY\": \"spherical_triangle_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/concentric_mapping_test.comp.hlsl\",
    \"KEY\": \"concentric_mapping_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/concentric_mapping_test.comp.hlsl\",
    \"KEY\": \"concentric_mapping_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/polar_mapping_test.comp.hlsl\",
    \"KEY\": \"polar_mapping_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/polar_mapping_test.comp.hlsl\",
    \"KEY\": \"polar_mapping_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/bilinear_test.comp.hlsl\",
    \"KEY\": \"bilinear_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/bilinear_test.comp.hlsl\",
    \"KEY\": \"bilinear_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/box_muller_transform_test.comp.hlsl\",
    \"KEY\": \"box_muller_transform_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/box_muller_transform_test.comp.hlsl\",
    \"KEY\": \"box_muller_transform_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/projected_spherical_triangle_test.comp.hlsl\",
    \"KEY\": \"projected_spherical_triangle_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/projected_spherical_triangle_test.comp.hlsl\",
    \"KEY\": \"projected_spherical_triangle_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/projected_spherical_rectangle_test.comp.hlsl\",
    \"KEY\": \"projected_spherical_rectangle_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/projected_spherical_rectangle_test.comp.hlsl\",
    \"KEY\": \"projected_spherical_rectangle_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/spherical_rectangle_test.comp.hlsl\",
    \"KEY\": \"spherical_rectangle_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/spherical_rectangle_test.comp.hlsl\",
    \"KEY\": \"spherical_rectangle_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/alias_table_test.comp.hlsl\",
    \"KEY\": \"alias_table_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/alias_table_test.comp.hlsl\",
    \"KEY\": \"alias_table_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  },
  {
    \"INPUT\": \"app_resources/shaders/cumulative_probability_test.comp.hlsl\",
    \"KEY\": \"cumulative_probability_test\"
  },
  {
    \"INPUT\": \"app_resources/shaders/cumulative_probability_test.comp.hlsl\",
    \"KEY\": \"cumulative_probability_bench\",
    \"COMPILE_OPTIONS\": [${BENCH_OPTS}]
  }
]
")

NBL_CREATE_NSC_COMPILE_RULES(
  TARGET ${EXECUTABLE_NAME}SPIRV
  LINK_TO ${EXECUTABLE_NAME}
  BINARY_DIR ${OUTPUT_DIRECTORY}
  MOUNT_POINT_DEFINE NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT
  COMMON_OPTIONS -I ${CMAKE_CURRENT_SOURCE_DIR} -T cs_6_8
  OUTPUT_VAR KEYS
  INCLUDE nbl/this_example/builtin/build/spirv/keys.hpp
  NAMESPACE nbl::this_example::builtin::build
  INPUTS ${JSON}
)

NBL_CREATE_RESOURCE_ARCHIVE(
  NAMESPACE nbl::this_example::builtin::build
  TARGET ${EXECUTABLE_NAME}_builtinsBuild
  LINK_TO ${EXECUTABLE_NAME}
  BIND ${OUTPUT_DIRECTORY}
  BUILTINS ${KEYS}
)
