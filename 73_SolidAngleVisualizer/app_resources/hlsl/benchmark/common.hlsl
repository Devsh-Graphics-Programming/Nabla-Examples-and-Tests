//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_X = 64u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_Y = 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_Z = 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_COUNT = 1920u * 1080u / BENCHMARK_WORKGROUP_DIMENSION_SIZE_X;

enum SAMPLING_BENCHMARK_MODE
{
	TRIANGLE_SOLID_ANGLE,
	TRIANGLE_PROJECTED_SOLID_ANGLE,
};

struct BenchmarkPushConstants
{
	float32_t3x4 modelMatrix;
	uint32_t samplingMode;
	SAMPLING_BENCHMARK_MODE benchmarkMode;
};