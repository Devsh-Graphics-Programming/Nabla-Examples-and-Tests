#ifndef _THIS_EXAMPLE_GRID_COMMON_HLSL_
#define _THIS_EXAMPLE_GRID_COMMON_HLSL_

#ifdef __HLSL_VERSION
    struct VSInput
	{
		[[vk::location(0)]] float3 position : POSITION;
		[[vk::location(1)]] float4 color : COLOR;
		[[vk::location(2)]] float2 uv : TEXCOORD;
		[[vk::location(3)]] float3 normal : NORMAL;
	};

    struct PSInput
    {
        float4 position : SV_Position;
        float2 uv : TEXCOORD0;
    };

    float gridTextureGradBox(float2 p, float2 ddx, float2 ddy)
    {
        float N = 30.0; // grid ratio
        float2 w = max(abs(ddx), abs(ddy)) + 0.01; // filter kernel

        // analytic (box) filtering
        float2 a = p + 0.5 * w;
        float2 b = p - 0.5 * w;
        float2 i = (floor(a) + min(frac(a) * N, 1.0) - floor(b) - min(frac(b) * N, 1.0)) / (N * w);

        // pattern
        return (1.0 - i.x) * (1.0 - i.y);
    }
#endif // __HLSL_VERSION

#include "common/SBasicViewParameters.hlsl"

#endif // _THIS_EXAMPLE_GRID_COMMON_HLSL_

/*
    do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/