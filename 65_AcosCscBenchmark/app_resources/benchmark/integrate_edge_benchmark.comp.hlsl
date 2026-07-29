//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/benchmark/common.hlsl"
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

using namespace nbl::hlsl;
using namespace nbl;

[[vk::binding(0, 0)]] RWByteAddressBuffer outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

void gen_rand_vecs(inout Xoroshiro64Star rng, out vector<float, 3> v1, out vector<float, 3> v2)
{
    const float INV_UINT32_MAX_PLUS1 = 1.0 / 4294967296.0; // 1 / 2^32
    const float PI = 3.14159265358979323846;

    // sample first direction
    uint r0 = rng();
    uint r1 = rng();
    float u0x = (float)r0 * INV_UINT32_MAX_PLUS1; // in [0,1)
    float u0y = (float)r1 * INV_UINT32_MAX_PLUS1; // in [0,1)

    float z0 = u0x * 2.0 - 1.0;
    float phi0 = u0y * (2.0 * PI);
    float rxy0 = hlsl::sqrt(max(0.0, 1.0 - z0 * z0));
    v1 = vector<float, 3>(rxy0 * hlsl::cos(phi0), rxy0 * hlsl::sin(phi0), z0);

    // sample second direction
    uint r2 = rng();
    uint r3 = rng();
    float u1x = (float)r2 * INV_UINT32_MAX_PLUS1;
    float u1y = (float)r3 * INV_UINT32_MAX_PLUS1;

    float z1 = u1x * 2.0 - 1.0;
    float phi1 = u1y * (2.0 * PI);
    float rxy1 = hlsl::sqrt(max(0.0, 1.0 - z1 * z1));
    v2 = vector<float, 3>(rxy1 * hlsl::cos(phi1), rxy1 * hlsl::sin(phi1), z1);
}

template<int mode>
float integrate_edge(uint32_t seed)
{
  Xoroshiro64Star rng = Xoroshiro64Star::construct(uint32_t2(seed, 0xdeadbeefu));
  vector<float, 3> v1, v2;
  static const uint32_t iteration = BENCHMARK_SAMPLE_PER_THREAD;
  float result = 0;
  for (uint32_t i = 0; i < iteration; i++)
  {
    gen_rand_vecs(rng, v1, v2);
    float cos_theta = dot(v1, v2);
    cos_theta = clamp(cos_theta, -0.9999, 0.9999);
    float cross_result = cross(v1, v2).z;

    if (mode==BM_SETUP)
    {
      result += cross_result * cos_theta;
    }
    else if (mode==BM_EXACT)
    {
      float theta = nbl::hlsl::acos(cos_theta);
      result += cross_result * (theta * nbl::hlsl::rsqrt(1 - (cos_theta * cos_theta)));
    }
    else if (mode==BM_ORDER1)
    {
      result += cross_result * acos_csc_approx<float, 1>(cos_theta);
    }
    else if (mode == BM_ORDER2)
    {
      result += cross_result * acos_csc_approx<float, 2>(cos_theta);
    }
    else if (mode==BM_ORDER3)
    {
      result += cross_result * acos_csc_approx<float, 3>(cos_theta);
    }
    else if (mode==BM_SIGN_FLIP)
    {
      result += cross_result * acos_csc_approx_sign_flip(cos_theta, cross_result > 0);
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
    output = integrate_edge<BM_SETUP>(invocationID.x);
    break;
  case BM_EXACT:
    output = integrate_edge<BM_EXACT>(invocationID.x);
    break;
  case BM_ORDER1:
    output = integrate_edge<BM_ORDER1>(invocationID.x);
    break;
  case BM_ORDER2:
    output = integrate_edge<BM_ORDER2>(invocationID.x);
    break;
  case BM_ORDER3:
    output = integrate_edge<BM_ORDER3>(invocationID.x);
    break;
  case BM_SIGN_FLIP:
    output = integrate_edge<BM_SIGN_FLIP>(invocationID.x);
    break;
  }

	const uint32_t offset = sizeof(uint64_t) * invocationID.x;
	outputBuffer.Store<uint64_t>(offset, output);
}
