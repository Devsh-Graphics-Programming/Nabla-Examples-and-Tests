//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_X = 128u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_Y = 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_Z = 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_COUNT = 1024u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_SAMPLE_PER_THREAD = 1000u;

enum BENCHMARK_MODE
{
    BM_EXACT,
    BM_ORDER1,
    BM_ORDER2,
};

struct BenchmarkPushConstants
{
    BENCHMARK_MODE benchmarkMode;
};
