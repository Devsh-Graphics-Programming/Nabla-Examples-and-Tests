// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hlsl"
#include "PSInput.hlsl"

[[vk::combinedImageSampler]] [[vk::binding(0, 3)]] Texture2D<float32_t> inIESCandelaImage;
[[vk::combinedImageSampler]] [[vk::binding(1, 3)]] Texture2D<float32_t2> inSphericalCoordinatesImage;
[[vk::combinedImageSampler]] [[vk::binding(2, 3)]] Texture2D<float32_t3> inOUVProjectionDirectionImage;
[[vk::combinedImageSampler]] [[vk::binding(3, 3)]] Texture2D<unorm float2> inPassTMaskImage;

[[vk::combinedImageSampler]] [[vk::binding(0, 3)]] SamplerState inIESCandelaSampler;
[[vk::combinedImageSampler]] [[vk::binding(1, 3)]] SamplerState inSphericalCoordinatesSampler;
[[vk::combinedImageSampler]] [[vk::binding(2, 3)]] SamplerState inOUVProjectionDirectionSampler;
[[vk::combinedImageSampler]] [[vk::binding(3, 3)]] SamplerState inPassTMaskSampler;

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
	return inIESCandelaImage.Sample(inIESCandelaSampler, iesDirToUv(normalize(float32_t3(uv.x, 0.001, uv.y)))).x;
}

[shader("pixel")]
float32_t4 main(PSInput input) : SV_Target0
{
    float32_t2 ndc = input.position.xy;
	float32_t2 uv = (ndc + 1) / 2;
	
	switch (pc.mode)
	{
		case 0:
		{
			float32_t dist = length(ndc) * 1.015625f;
			float32_t p = plot(dist, 1.0f, 0.75f);
			float32_t3 col = float32_t3(p, p, p);

			float32_t normalizedStrength = f(ndc);
			if (dist < normalizedStrength)
				col += float32_t3(1.0f, 0.0f, 0.0f);

			return float32_t4(col, 1.0f);
		}
		case 1:
			return float32_t4(inIESCandelaImage.Sample(inIESCandelaSampler, uv).x, 0.f, 0.f, 1.f);
		case 2:
			return float32_t4(inSphericalCoordinatesImage.Sample(inSphericalCoordinatesSampler, uv).xy, 0.f, 1.f);
		case 3:
			return float32_t4(inOUVProjectionDirectionImage.Sample(inOUVProjectionDirectionSampler, uv).xyz, 1.f);
		default:
			return float32_t4(inPassTMaskImage.Sample(inPassTMaskSampler, uv).xy, 0.f, 1.f);
	}
}
