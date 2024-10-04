//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

NBL_CONSTEXPR uint32_t BENCHMARK_WORKGROUP_SIZE = 1024;

struct BenchmarkPushConstants
{
    uint32_t rawBufferAddress;
    int testEmulatedFloat64;
};