//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

NBL_CONSTEXPR uint32_t BENCHMARK_WORKGROUP_SIZE = 1024;

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