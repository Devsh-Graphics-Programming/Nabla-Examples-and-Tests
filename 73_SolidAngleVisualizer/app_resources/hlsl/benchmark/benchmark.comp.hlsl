//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/hlsl/common.hlsl"
#include "app_resources/hlsl/benchmark/common.hlsl"
#include "app_resources/hlsl/silhouette.hlsl"
#include "app_resources/hlsl/parallelogram_sampling.hlsl"
#include "app_resources/hlsl/pyramid_sampling.hlsl"
#include "app_resources/hlsl/triangle_sampling.hlsl"

using namespace nbl::hlsl;

[[vk::binding(0, 0)]] RWByteAddressBuffer outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

static const SAMPLING_MODE benchmarkMode = (SAMPLING_MODE)SAMPLING_MODE_CONST;

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
	uint32_t sil = ClippedSilhouette::computeRegionAndConfig(perturbedMatrix, region, configIndex, vertexCount);

	ClippedSilhouette silhouette = (ClippedSilhouette)0;
	silhouette.compute(perturbedMatrix, vertexCount, sil);

	float32_t pdf;
	uint32_t triIdx;
	uint32_t validSampleCount = 0;
	float32_t3 sampleDir = float32_t3(0.0, 0.0, 0.0);

	bool sampleValid;
	if (benchmarkMode == SAMPLING_MODE::TRIANGLE_SOLID_ANGLE ||
		benchmarkMode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
	{
		TriangleFanSampler samplingData;
		samplingData = TriangleFanSampler::create(silhouette, benchmarkMode);

		for (uint32_t i = 0; i < pc.sampleCount; i++)
		{
			float32_t2 xi = float32_t2(
				(float32_t(i & 7u) + 0.5f) / 8.0f,
				(float32_t(i >> 3u) + 0.5f) / 8.0f);

			sampleDir += samplingData.sample(silhouette, xi, pdf, triIdx);
			validSampleCount++;
		}
	}
	else if (benchmarkMode == SAMPLING_MODE::PROJECTED_PARALLELOGRAM_SOLID_ANGLE)
	{
		// Precompute parallelogram for sampling
		silhouette.normalize();
		SilEdgeNormals silEdgeNormals;
		Parallelogram parallelogram = Parallelogram::create(silhouette, silEdgeNormals);
		for (uint32_t i = 0; i < pc.sampleCount; i++)
		{
			float32_t2 xi = float32_t2(
				(float32_t(i & 7u) + 0.5f) / 8.0f,
				(float32_t(i >> 3u) + 0.5f) / 8.0f);

			sampleDir += parallelogram.sample(silEdgeNormals, xi, pdf, sampleValid);
			validSampleCount += sampleValid ? 1u : 0u;
		}
	}
	else if (benchmarkMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_RECTANGLE)
	{
		// Precompute spherical pyramid and Urena sampler once (edge normals fused)
		SilEdgeNormals silEdgeNormals;
		SphericalPyramid pyramid = SphericalPyramid::create(silhouette, silEdgeNormals);
		UrenaSampler urena = UrenaSampler::create(pyramid);

		for (uint32_t i = 0; i < pc.sampleCount; i++)
		{
			float32_t2 xi = float32_t2(
				(float32_t(i & 7u) + 0.5f) / 8.0f,
				(float32_t(i >> 3u) + 0.5f) / 8.0f);

			sampleDir += urena.sample(pyramid, silEdgeNormals, xi, pdf, sampleValid);
			validSampleCount += sampleValid ? 1u : 0u;
		}
	}
	else if (benchmarkMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BIQUADRATIC)
	{
		// Precompute spherical pyramid and biquadratic sampler once (edge normals fused)
		SilEdgeNormals silEdgeNormals;
		SphericalPyramid pyramid = SphericalPyramid::create(silhouette, silEdgeNormals);
		BiquadraticSampler biquad = BiquadraticSampler::create(pyramid);

		for (uint32_t i = 0; i < pc.sampleCount; i++)
		{
			float32_t2 xi = float32_t2(
				(float32_t(i & 7u) + 0.5f) / 8.0f,
				(float32_t(i >> 3u) + 0.5f) / 8.0f);

			sampleDir += biquad.sample(pyramid, silEdgeNormals, xi, pdf, sampleValid);
			validSampleCount += sampleValid ? 1u : 0u;
		}
	}
	else if (benchmarkMode == SAMPLING_MODE::SYMMETRIC_PYRAMID_SOLID_ANGLE_BILINEAR)
	{
		// Precompute spherical pyramid and bilinear sampler once (edge normals fused)
		SilEdgeNormals silEdgeNormals;
		SphericalPyramid pyramid = SphericalPyramid::create(silhouette, silEdgeNormals);
		BilinearSampler bilin = BilinearSampler::create(pyramid);

		for (uint32_t i = 0; i < pc.sampleCount; i++)
		{
			float32_t2 xi = float32_t2(
				(float32_t(i & 7u) + 0.5f) / 8.0f,
				(float32_t(i >> 3u) + 0.5f) / 8.0f);

			sampleDir += bilin.sample(pyramid, silEdgeNormals, xi, pdf, sampleValid);
			validSampleCount += sampleValid ? 1u : 0u;
		}
	}

	const uint32_t offset = sizeof(uint32_t) * invocationID.x;
	outputBuffer.Store<float32_t>(offset, pdf + validSampleCount + triIdx + asuint(sampleDir.x) + asuint(sampleDir.y) + asuint(sampleDir.z));
}
