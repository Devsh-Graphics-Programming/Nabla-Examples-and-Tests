//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/benchmark/common.hlsl"
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>
#include <nbl/builtin/hlsl/math/fast_acos.hlsl>

using namespace nbl::hlsl;
using namespace nbl;

[[vk::binding(0, 0)]] RWByteAddressBuffer outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

template<typename T = float32_t, int benchmarkMode>
T calcAcosCsc(uint32_t seed)
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
    if (benchmarkMode == BM_EXACT)
    {
			T theta = nbl::hlsl::acos<T>(cosTheta);
			result += (theta * nbl::hlsl::rsqrt<T>(T(1.0) - (cosTheta * cosTheta)));
    }
    else if (benchmarkMode == BM_ORDER1)
    {
      result += fast_acos_csc_call<T, 1>(cosTheta);
    } 
	  else if (benchmarkMode == BM_ORDER2)
    {
      result += fast_acos_csc_call<T, 2>(cosTheta);
    }
    else if (benchmarkMode == BM_ORDER3)
    {
      result += fast_acos_csc_call<T, 3>(cosTheta);
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

  switch (pc.benchmarkMode)
  {
  case BM_SETUP:    
    output = calcAcosCsc<real_t, BM_SETUP>(invocationID.x);
    break;
  case BM_EXACT:
    output = calcAcosCsc<real_t, BM_EXACT>(invocationID.x);
    break;
  case BM_ORDER1:
    output = calcAcosCsc<real_t, BM_ORDER1>(invocationID.x);
    break;
  case BM_ORDER2:
    output = calcAcosCsc<real_t, BM_ORDER2>(invocationID.x);
    break;
  case BM_ORDER3:
    output = calcAcosCsc<real_t, BM_ORDER3>(invocationID.x);
    break;
  }

	const uint32_t offset = sizeof(uint64_t) * invocationID.x;
	outputBuffer.Store<uint64_t>(offset, output);
}

