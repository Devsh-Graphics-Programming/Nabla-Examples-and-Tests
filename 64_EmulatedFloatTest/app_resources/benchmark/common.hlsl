//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_X = 128u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_Y = 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_Z = 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_COUNT = 1024u;

enum EF64_BENCHMARK_MODE
{
    NATIVE,
    EF64_FAST_MATH_ENABLED,
    EF64_FAST_MATH_DISABLED,
    SUBGROUP_DIVIDED_WORK,
    INTERLEAVED
};

struct BenchmarkPushConstants
{
    EF64_BENCHMARK_MODE benchmarkMode;
};