//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_X = 128u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_Y = 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_DIMENSION_SIZE_Z = 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_WORKGROUP_COUNT = 1024u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t BENCHMARK_SAMPLE_PER_THREAD = 1000u;

using namespace nbl;
using namespace nbl::hlsl;

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

template<typename T, int order=2>
T acos_csc_approx(const T arg)
{
    const T u = hlsl::log2(_static_cast<T>(1)+arg);
    // See https://www.desmos.com/calculator/sdptomhbju
    // Furthermore we could clip the polynomial calc to `Cu+D or `(Bu+C)u+D` for small arguments
    T poly;
    // TODO: actually optimize these constants in real world scenarios (renders)
    if (order==1)
        poly = (_static_cast<T>(1)-u)*_static_cast<T>(0.6);
    else if (order==2)
    {
        const T a = -0.637;
        const T b = -0.0115;
        const T c = -(a + b);
        poly = hlsl::fma(u, hlsl::fma(u, a, b), c);

    }
    else if (order==3)
    {
        const T a = 0.6494;
        const T b = -0.6311;
        const T c = -0.0122;
        const T d = -0.00039;
        poly = hlsl::fma(u, hlsl::fma(u, hlsl::fma(u, d, c), b), a);
    }
    return hlsl::exp2<T>(poly);
}

float acos_csc_approx_sign_flip(const float arg, bool isPositive)
{
    // u = log2(1 + cosTheta)
    float u = log2(1.0 + arg);

    float a1 = 0.646153;
    float a2 = 0.656153;
    float b1 = -0.63452;
    float b2 = -0.5;

    float c1 = -0.01163;
    float c2 = -0.00609;

    // select directly between the two folded literals instead of computing at runtime
    float a = isPositive ? a1 : a2;
    float c = isPositive ? c1 : c2;
    float b = isPositive ? b1 : b2;
    float poly = hlsl::fma(u, hlsl::fma(u, a, b), c);
    return hlsl::exp2<float>(poly);
}

