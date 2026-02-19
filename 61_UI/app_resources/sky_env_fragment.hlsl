// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma wave shader_stage(fragment)

#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

struct PushConstants
{
	float32_t4x4 invProj;
	float32_t4x4 invViewRot;
	uint32_t orthoMode;
	uint32_t pad0;
	uint32_t pad1;
	uint32_t pad2;
};

[[vk::push_constant]] PushConstants pc;

[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2D envMap;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState envSampler;

float32_t3 safeNormalize(float32_t3 v)
{
	const float32_t len2 = max(dot(v, v), 1e-12f);
	return v * rsqrt(len2);
}

float32_t3 safeHomogeneousDivide(float32_t4 v)
{
	float32_t w = v.w;
	if (abs(w) < 1e-6f)
		w = (w < 0.0f) ? -1e-6f : 1e-6f;
	return v.xyz / w;
}

float32_t3 acesToneMap(float32_t3 x)
{
	const float32_t a = 2.51f;
	const float32_t b = 0.03f;
	const float32_t c = 2.43f;
	const float32_t d = 0.59f;
	const float32_t e = 0.14f;
	return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[[vk::location(0)]] float32_t4 main(SVertexAttributes vxAttr) : SV_Target0
{
	const float32_t2 ndc = vxAttr.uv * 2.0f - float32_t2(1.0f, 1.0f);
	float32_t3 dirVS;
	if (pc.orthoMode != 0u)
	{
		const float32_t4 centerNearVS_H = mul(pc.invProj, float32_t4(0.0f, 0.0f, 0.0f, 1.0f));
		const float32_t4 centerFarVS_H = mul(pc.invProj, float32_t4(0.0f, 0.0f, 1.0f, 1.0f));
		const float32_t3 centerNearVS = safeHomogeneousDivide(centerNearVS_H);
		const float32_t3 centerFarVS = safeHomogeneousDivide(centerFarVS_H);
		const float32_t3 orthoForward = safeNormalize(centerFarVS - centerNearVS);

		const float32_t4 leftNearVS_H = mul(pc.invProj, float32_t4(-1.0f, 0.0f, 0.0f, 1.0f));
		const float32_t4 rightNearVS_H = mul(pc.invProj, float32_t4(1.0f, 0.0f, 0.0f, 1.0f));
		const float32_t4 downNearVS_H = mul(pc.invProj, float32_t4(0.0f, -1.0f, 0.0f, 1.0f));
		const float32_t4 upNearVS_H = mul(pc.invProj, float32_t4(0.0f, 1.0f, 0.0f, 1.0f));

		const float32_t3 leftNearVS = safeHomogeneousDivide(leftNearVS_H);
		const float32_t3 rightNearVS = safeHomogeneousDivide(rightNearVS_H);
		const float32_t3 downNearVS = safeHomogeneousDivide(downNearVS_H);
		const float32_t3 upNearVS = safeHomogeneousDivide(upNearVS_H);

		const float32_t3 orthoRight = safeNormalize(rightNearVS - leftNearVS);
		const float32_t3 orthoUp = safeNormalize(upNearVS - downNearVS);
		const float32_t tanHalfFov = 0.7673269879789604f; // tan(37.5 deg)
		dirVS = safeNormalize(orthoForward + orthoRight * ndc.x * tanHalfFov + orthoUp * ndc.y * tanHalfFov);
	}
	else
	{
		const float32_t4 clip = float32_t4(ndc, 1.0f, 1.0f);
		const float32_t4 viewH = mul(pc.invProj, clip);
		dirVS = safeNormalize(safeHomogeneousDivide(viewH));
	}
	const float32_t3 dir = safeNormalize(mul(pc.invViewRot, float32_t4(dirVS, 0.0f)).xyz);

	const float32_t invPi = 0.31830988618379067f;
	const float32_t invTwoPi = 0.15915494309189535f;
	float32_t2 envUv;
	envUv.x = atan2(dir.z, dir.x) * invTwoPi + 0.5f;
	envUv.y = acos(clamp(dir.y, -1.0f, 1.0f)) * invPi;

	float32_t3 color = max(envMap.SampleLevel(envSampler, envUv, 0.0f).rgb - 0.0010f, 0.0f);
	color = acesToneMap(color * 0.45f);
	return float32_t4(color, 1.0f);
}
