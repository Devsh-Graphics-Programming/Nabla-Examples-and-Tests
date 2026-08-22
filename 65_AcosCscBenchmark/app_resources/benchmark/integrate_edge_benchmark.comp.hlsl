//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/benchmark/common.hlsl"
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>
#include <nbl/builtin/hlsl/math/fast_acos.hlsl>

using namespace nbl::hlsl;
using namespace nbl::hlsl::math;
using namespace nbl;

[[vk::binding(0, 0)]] RWByteAddressBuffer outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

template <typename T>
void gen_rand_vecs(inout Xoroshiro64Star rng, out vector<T, 3> v1, out vector<T, 3> v2)
{
    const T INV_UINT32_MAX_PLUS1 = 1.0 / 4294967296.0;
    const T TWO_PI = 6.28318530717958647692;

    // First vector - compute directly without intermediate storage
    T z0 = (T)rng() * INV_UINT32_MAX_PLUS1 * 2.0 - 1.0;
    T phi0 = (T)rng() * INV_UINT32_MAX_PLUS1 * TWO_PI;
    T rxy0 = nbl::hlsl::sqrt(max(0.0, 1.0 - z0 * z0));
    v1 = vector<T, 3>(rxy0 * hlsl::cos(phi0), rxy0 * hlsl::sin(phi0), z0);

    // Second vector
    T z1 = (T)rng() * INV_UINT32_MAX_PLUS1 * 2.0 - 1.0;
    T phi1 = (T)rng() * INV_UINT32_MAX_PLUS1 * TWO_PI;
    T rxy1 = nbl::hlsl::sqrt(max(0.0, 1.0 - z1 * z1));
    v2 = vector<T, 3>(rxy1 * hlsl::cos(phi1), rxy1 * hlsl::sin(phi1), z1);
}

template<typename T, int mode>
T integrate_edge(uint32_t seed)
{
  Xoroshiro64Star rng = Xoroshiro64Star::construct(uint32_t2(seed, 0xdeadbeefu));
  static const uint32_t iteration = BENCHMARK_SAMPLE_PER_THREAD;
  T result = 0;
  vector<T, 3> v1, v2;
  for (uint32_t i = 0; i < iteration; i++)
  {
    gen_rand_vecs<T>(rng, v1, v2);
    const T cos_theta = clamp(dot(v1, v2), T(-0.9999), T(0.9999));
    const T cross_z = v1.x * v2.y - v1.y * v2.x;
    if (mode==BM_SETUP)
    {
      result += cross_z * cos_theta;
    }
    else if (mode==BM_EXACT)
    {
      T theta = nbl::hlsl::acos<T>(cos_theta);
      result += cross_z * (theta * nbl::hlsl::rsqrt<T>(T(1.0) - (cos_theta * cos_theta)));
    }
    else if (mode==BM_ORDER1)
    {
      result += cross_z * fast_acos_csc_call<T, 1>(cos_theta);
    }
    else if (mode == BM_ORDER2)
    {
      result += cross_z * fast_acos_csc_call<T, 2>(cos_theta);
    }
    else if (mode==BM_ORDER3)
    {
      result += cross_z * fast_acos_csc_call<T, 3>(cos_theta);
    }
    else if (mode==BM_SIGN_FLIP)
    {
      result += cross_z * fast_acos_csc_directed_call<T>(cos_theta, cross_z > 0);
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
    output = integrate_edge<real_t, BM_SETUP>(invocationID.x);
    break;
  case BM_EXACT:
    output = integrate_edge<real_t, BM_EXACT>(invocationID.x);
    break;
  case BM_ORDER1:
    output = integrate_edge<real_t, BM_ORDER1>(invocationID.x);
    break;
  case BM_ORDER2:
    output = integrate_edge<real_t, BM_ORDER2>(invocationID.x);
    break;
  case BM_ORDER3:
    output = integrate_edge<real_t, BM_ORDER3>(invocationID.x);
    break;
  case BM_SIGN_FLIP:
    output = integrate_edge<real_t, BM_SIGN_FLIP>(invocationID.x);
    break;
  }

	const uint32_t offset = sizeof(uint64_t) * invocationID.x;
	outputBuffer.Store<uint64_t>(offset, output);
}
