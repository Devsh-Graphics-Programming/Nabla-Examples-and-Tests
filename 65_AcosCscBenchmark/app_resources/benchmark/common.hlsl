//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/math/fast_acos.hlsl>

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_X = 128u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_Y = 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_Z = 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_COUNT = 3280;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_SAMPLE_PER_THREAD = 10000;

using namespace nbl;
using namespace nbl::hlsl;

using real_t = float;

enum BENCHMARK_MODE
{
    BM_SETUP,
    BM_EXACT,
    BM_ORDER1,
    BM_ORDER2,
    BM_ORDER3,
    BM_SIGN_FLIP,
};

struct BenchmarkPushConstants
{
    BENCHMARK_MODE benchmarkMode;
};

template <typename T, int order>
T fast_acos_csc_call(const T val)
{
  return nbl::hlsl::math::fast_acos_csc<T, order>::__call(val);
}

template <typename T>
T fast_acos_csc_directed_call(const T val, bool overestimate)
{
  return nbl::hlsl::math::fast_acos_csc_directed<T>::__call(val, overestimate);
}
