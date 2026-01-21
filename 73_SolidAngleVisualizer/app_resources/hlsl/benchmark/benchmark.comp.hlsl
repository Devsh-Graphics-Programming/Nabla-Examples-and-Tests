//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/hlsl/common.hlsl"
// doesn't change Z coordinate
float32_t3 sphereToCircle(float32_t3 spherePoint)
{
	if (spherePoint.z >= 0.0f)
	{
		return float32_t3(spherePoint.xy, spherePoint.z);
	}
	else
	{
		float32_t r2 = (1.0f - spherePoint.z) / (1.0f + spherePoint.z);
		float32_t uv2Plus1 = r2 + 1.0f;
		return float32_t3((spherePoint.xy * uv2Plus1 / 2.0f), spherePoint.z);
	}
}

#undef DEBUG_DATA // Avoid conflict with DebugDataBuffer in this file
#undef VISUALIZE_SAMPLES

#include "app_resources/hlsl/benchmark/common.hlsl"
#include "app_resources/hlsl/silhouette.hlsl"
#include "app_resources/hlsl/Sampling.hlsl"
#include "app_resources/hlsl/parallelogram_sampling.hlsl"

using namespace nbl::hlsl;

[[vk::binding(0, 0)]] RWByteAddressBuffer outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

[numthreads(BENCHMARK_WORKGROUP_DIMENSION_SIZE_X, 1, 1)]
	[shader("compute")] void
	main(uint32_t3 invocationID : SV_DispatchThreadID)
{
	// Perturb model matrix slightly per sample group
	float32_t3x4 perturbedMatrix = pc.modelMatrix;
	perturbedMatrix[0][3] += float32_t(invocationID.x) * 1e-6f;

	uint32_t3 region;
	uint32_t configIndex;
	uint32_t vertexCount;
	uint32_t sil = computeRegionAndConfig(perturbedMatrix, region, configIndex, vertexCount);

	ClippedSilhouette silhouette;
	computeSilhouette(perturbedMatrix, vertexCount, sil, silhouette);
	float32_t pdf;
	uint32_t triIdx;
	float32_t3 sampleDir = float32_t3(0.0, 0.0, 0.0);
	if (pc.benchmarkMode == SAMPLING_MODE::TRIANGLE_SOLID_ANGLE ||
		pc.benchmarkMode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
	{
		SamplingData samplingData;
		samplingData = buildSamplingDataFromSilhouette(silhouette, pc.benchmarkMode);

		for (uint32_t i = 0; i < 64; i++)
		{
			float32_t2 xi = float32_t2(
				(float32_t(i & 7u) + 0.5f) / 8.0f,
				(float32_t(i >> 3u) + 0.5f) / 8.0f);

			sampleDir += sampleFromData(samplingData, silhouette, xi, pdf, triIdx);
		}
	}
	else if (pc.benchmarkMode == SAMPLING_MODE::PROJECTED_PARALLELOGRAM_SOLID_ANGLE)
	{
		// Precompute parallelogram for sampling
		ParallelogramSilhouette paraSilhouette = buildParallelogram(silhouette);
		for (uint32_t i = 0; i < 64; i++)
		{
			float32_t2 xi = float32_t2(
				(float32_t(i & 7u) + 0.5f) / 8.0f,
				(float32_t(i >> 3u) + 0.5f) / 8.0f);

			bool valid;
			sampleDir += sampleFromParallelogram(paraSilhouette, xi, pdf, valid);
		}
	}

	const uint32_t offset = sizeof(uint32_t) * invocationID.x;
	outputBuffer.Store<float32_t>(offset, pdf + triIdx + asuint(sampleDir.x) + asuint(sampleDir.y) + asuint(sampleDir.z));
}
