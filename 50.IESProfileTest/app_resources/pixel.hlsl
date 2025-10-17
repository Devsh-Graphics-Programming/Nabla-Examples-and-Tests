// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hlsl"
#include "nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl"
using namespace nbl::hlsl::ext::FullScreenTriangle;

[[vk::binding(0, 0)]] Texture2D inIESCandelaImage[MAX_IES_IMAGES];
[[vk::binding(1, 0)]] Texture2D inSphericalCoordinatesImage[MAX_IES_IMAGES];
[[vk::binding(2, 0)]] Texture2D inOUVProjectionDirectionImage[MAX_IES_IMAGES];
[[vk::binding(3, 0)]] Texture2D inPassTMaskImage[MAX_IES_IMAGES];
[[vk::binding(10, 0)]] SamplerState generalSampler;

[[vk::push_constant]] struct PushConstants pc;

float32_t2 iesDirToUv(float32_t3 dir) 
{
	float32_t sum = dot(float32_t3(1.0f, 1.0f, 1.0f), abs(dir));
	float32_t3 s = dir / sum;

	if (s.z < 0.0f)
		s.xy = sign(s.xy) * (1.0f - abs(s.yx));

	return s.xy * 0.5f + 0.5f;
}

float32_t plot(float32_t cand, float32_t pct, float32_t bold)
{
	return smoothstep(pct-0.005*bold, pct, cand) - smoothstep( pct, pct+0.005*bold, cand);
}

// vertical cut of IES (i.e. cut by plane x = 0)
float32_t f(float32_t2 uv) 
{
	return inIESCandelaImage[pc.texIx].Sample(generalSampler, iesDirToUv(normalize(float32_t3(uv.x, 0.001, uv.y)))).x;
}

[shader("pixel")]
float32_t4 PSMain(SVertexAttributes input) : SV_Target0
{
	switch (pc.mode)
	{
		case 0:
		{
			float32_t2 ndc = input.uv * 2.f - 1.f;
			float32_t dist = length(ndc) * 1.015625f;
			float32_t p = plot(dist, 1.0f, 0.75f);
			float32_t3 col = float32_t3(p, p, p);

			float32_t normalizedStrength = f(ndc);
			if (dist < normalizedStrength)
				col += float32_t3(1.0f, 0.0f, 0.0f);

			return float32_t4(col, 1.0f);
		}
		case 1:
			return float32_t4(inIESCandelaImage[pc.texIx].Sample(generalSampler, input.uv).x, 0.f, 0.f, 1.f);
		case 2:
			return float32_t4(inSphericalCoordinatesImage[pc.texIx].Sample(generalSampler, input.uv).xy, 0.f, 1.f);
		case 3:
			return float32_t4(inOUVProjectionDirectionImage[pc.texIx].Sample(generalSampler, input.uv).xyz, 1.f);
		default:
			return float32_t4(inPassTMaskImage[pc.texIx].Sample(generalSampler, input.uv).xy, 0.f, 1.f);
	}
}
