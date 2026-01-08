//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)
#define DEBUG_DATA 0
#include "app_resources/hlsl/benchmark/common.hlsl"
#include "app_resources/hlsl/silhouette.hlsl"
#include "app_resources/hlsl/Sampling.hlsl"

using namespace nbl::hlsl;

[[vk::binding(0, 0)]] RWByteAddressBuffer outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

[numthreads(BENCHMARK_WORKGROUP_DIMENSION_SIZE_X, 1, 1)]
	[shader("compute")] void
	main(uint3 invocationID : SV_DispatchThreadID)
{
	uint32_t3 region;
	uint32_t configIndex;
	uint32_t vertexCount;
	uint32_t sil = computeRegionAndConfig(pc.modelMatrix, region, configIndex, vertexCount);

	ClippedSilhouette silhouette;
	computeSilhouette(pc.modelMatrix, vertexCount, sil, silhouette);

	SamplingData samplingData;
	samplingData = buildSamplingDataFromSilhouette(silhouette, pc.samplingMode);

	nbl::hlsl::random::PCG32 seedGen = nbl::hlsl::random::PCG32::construct(65536u + invocationID.x);
	const uint32_t2 seeds = uint32_t2(seedGen(), seedGen());

	float32_t pdf;
	uint32_t triIdx;
	float32_t3 sampleDir = float32_t3(0.0, 0.0, 0.0);
	for (uint32_t i = 0; i < 64; i++)
	{
		nbl::hlsl::Xoroshiro64StarStar rnd = nbl::hlsl::Xoroshiro64StarStar::construct(seeds);
		float32_t2 xi = nextRandomUnorm2(rnd);
		sampleDir += sampleFromData(samplingData, silhouette, xi, pdf, triIdx);
	}

	const uint32_t offset = sizeof(uint32_t) * invocationID.x;
	outputBuffer.Store<float32_t>(offset, pdf + triIdx + asuint(sampleDir.x) + asuint(sampleDir.y) + asuint(sampleDir.z));
}
