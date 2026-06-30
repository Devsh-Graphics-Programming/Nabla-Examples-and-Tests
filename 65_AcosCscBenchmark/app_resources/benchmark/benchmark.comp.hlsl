//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/benchmark/common.hlsl"
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>

using namespace nbl::hlsl;

[[vk::binding(0, 0)]] RWByteAddressBuffer outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

template<typename T = float32_t, int order=2>
T calcAcosCsc(uint32_t seed)
{
	static const uint32_t iteration = BENCHMARK_SAMPLE_PER_THREAD;
	T result = T(0);
	Xoroshiro64Star rng = Xoroshiro64Star::construct(uint32_t2(seed, 0xdeadbeefu));
	for (uint32_t i = 0; i < iteration; i++)
	{
		// Map uniform uint32 to [-1, 1)
		T cosTheta = T(rng()) / T(0x80000000u) - T(1);
		if (order == 0)
		{
			T theta = acos(cosTheta);
			result += (theta * nbl::hlsl::rsqrt(1 - (cosTheta * cosTheta)));
		} else
		{
      result += nbl::hlsl::shapes::acos_csc_approx<T, order>(cosTheta);
		}
	}
	return result;
}

[numthreads(BENCHMARK_WORKGROUP_DIMENSION_SIZE_X, 1, 1)]
[shader("compute")]
void main(uint3 invocationID : SV_DispatchThreadID)
{
	uint64_t output = 0ull;

	switch (pc.benchmarkMode)
	{
	case BM_EXACT:
	  output = calcAcosCsc<float32_t, 0>(invocationID.x);
		break;
	case BM_ORDER1:
	  output = calcAcosCsc<float32_t, 1>(invocationID.x);
		break;
	case BM_ORDER2:
	  output = calcAcosCsc<float32_t, 2>(invocationID.x);
		break;
	}

	const uint32_t offset = sizeof(uint64_t) * invocationID.x;
	outputBuffer.Store<uint64_t>(offset, output);
}
