//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/benchmark/common.hlsl"
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/math/fast_acos.hlsl>
#include <nbl/builtin/hlsl/math/angle_adding.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>

using namespace nbl::hlsl;
using namespace nbl;

[[vk::binding(0, 0)]] RWByteAddressBuffer outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

float acos_csc_approx_polynomial0(const float arg, bool isPositive)
{
    // u = log2(1 + cosTheta)
    float u = log2(1.0 + arg);

    float a = 0.646153;
    float b = -0.63452;

    float c1 = -0.01163;
    float c2 = -0.00609;

    // select directly between the two folded literals instead of computing at runtime
    float c = isPositive ? c2 : c1;
    float poly = hlsl::fma(u, hlsl::fma(u, c, b), a);
    return hlsl::exp2<float>(poly);
}

float acos_csc_approx_polynomial1(const float arg, float sign)
{
    // u = log2(1 + cosTheta)
    float u = log2(1.0 + arg);

    // Order 2 polynomial curve-fit matching, coefficients from UI:
    // ((1-u) * alpha) + ((1 - u^2) * beta)
    float a1 = 0.646153;
    float b1 = -0.63452;
    float c1 = -0.01163;

    float a2 = 0.66353;
    float b2 = -0.6172;
    float c2 = -0.00609;

    float a = hlsl::fma(sign, (a2 - a1), (a1 + a2));
    float b = hlsl::fma(sign, (b2 - b1), (b1 + b2));
    float c = hlsl::fma(sign, (c2 - c1), (c1 + c2));

    float poly = hlsl::fma(u, hlsl::fma(u, c, b), a) * 0.5;
    return hlsl::exp2<float>(poly);
}

float acos_csc_approx_polynomial_order3(const float arg)
{
    // u = log2(1 + cosTheta)
    float u = log2(1.0 + arg);
    float a = 0.6494;
    float b = -0.6311;
    float c = -0.0122;
    float d = -0.00039;
    float poly = hlsl::fma(u, hlsl::fma(u, hlsl::fma(u, d, c), b), a);
    return exp2(poly);
}

template<typename T = float32_t>
T calcAcosCsc(uint32_t seed, BENCHMARK_MODE benchmarkMode)
{
	static const uint32_t iteration = BENCHMARK_SAMPLE_PER_THREAD;
	T result = T(0);
	Xoroshiro64Star rng = Xoroshiro64Star::construct(uint32_t2(seed, 0xdeadbeefu));
	for (uint32_t i = 0; i < iteration; i++)
	{
		// Map uniform uint32 to [-1, 1)
		T cosTheta = T(rng()) / T(0x80000000u) - T(1);
		if (benchmarkMode == BM_SETUP)
		{
      result += cosTheta;
		}
    else if (benchmarkMode == BM_EXACT)
    {
			float theta = nbl::hlsl::acos(cosTheta);
			result += (theta * nbl::hlsl::rsqrt(1 - (cosTheta * cosTheta)));
    }
    else if (benchmarkMode == BM_ORDER1)
    {
      result += nbl::hlsl::shapes::acos_csc_approx<T, 1>(cosTheta);
    } 
	  else if (benchmarkMode == BM_ORDER2)
    {
      result += nbl::hlsl::shapes::acos_csc_approx<T, 2>(cosTheta);
    }
    else if (benchmarkMode == BM_POLYNOMIAL_ORDER3)
    {
      result += acos_csc_approx_polynomial_order3(cosTheta);
    }
    else
		{
      result += cosTheta;
		} 
	}
	return result;
}


[numthreads(BENCHMARK_WORKGROUP_DIMENSION_SIZE_X, 1, 1)]
[shader("compute")]
void main(uint3 invocationID : SV_DispatchThreadID)
{
	uint64_t output = 0ull;

  uint32_t seed = invocationID.x;
  float result = calcAcosCsc(seed, pc.benchmarkMode);
	// Xoroshiro64Star rng = Xoroshiro64Star::construct(uint32_t2(seed, 0xdeadbeefu));
	// vector<float, 3> v1, v2;
	// static const uint32_t iteration = BENCHMARK_SAMPLE_PER_THREAD;
 //  float result = 0;
 //  float cos_theta_base = (rng()) / float(0x80000000u) - 1.0;
 //  float cross_result_base = (rng()) / float(0x80000000u) - 1.0;
	// for (uint32_t i = 0; i < iteration; i++)
	// {
 //    float wobble = (rng() / float(0x80000000u) - 1.0);
 //
 //    float cos_theta = cos_theta_base + wobble;
 //    float cross_result = cross_result_base + wobble;
 //
 //    cos_theta = clamp(cos_theta, -0.9999, 0.9999);
 //
 //    if (pc.benchmarkMode==BM_SETUP)
 //    {
 //      result += cross_result * cos_theta;
 //    }
 //    else if (pc.benchmarkMode==BM_EXACT)
 //    {
	// 		float theta = nbl::hlsl::acos(cos_theta);
	// 		result += cross_result * (theta * nbl::hlsl::rsqrt(1 - (cos_theta * cos_theta)));
 //    }
 //    else if (pc.benchmarkMode==BM_ORDER1)
 //    {
 //      result += cross_result * nbl::hlsl::shapes::acos_csc_approx<float, 1>(cos_theta);
 //    }
 //    else if (pc.benchmarkMode == BM_ORDER2)
 //    {
 //      result += cross_result * nbl::hlsl::shapes::acos_csc_approx<float, 2>(cos_theta);
 //    }
 //    else if (pc.benchmarkMode==BM_POLYNOMIAL0)
 //    {
 //      result += cross_result * acos_csc_approx_polynomial0(cos_theta, cross_result > 0);
 //    }	
 //    else if (pc.benchmarkMode==BM_POLYNOMIAL1)
 //    {
 //      result += cross_result * acos_csc_approx_polynomial1(cos_theta, hlsl::sign(cross_result));
 //    }
 //    else if (pc.benchmarkMode==BM_POLYNOMIAL_ORDER3)
 //    {
 //      result += cross_result * acos_csc_approx_polynomial_order3(cos_theta);
 //    }
	// }

  output = (uint64_t)result;
	const uint32_t offset = sizeof(uint64_t) * invocationID.x;
	outputBuffer.Store<uint64_t>(offset, output);
}
