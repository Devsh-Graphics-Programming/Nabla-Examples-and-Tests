#ifndef _THIS_EXAMPLE_CUBE_COMMON_HLSL_
#define _THIS_EXAMPLE_CUBE_COMMON_HLSL_

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
		float4 color : COLOR0;
	};
#endif // __HLSL_VERSION

#include "common.hlsl"

#endif // _THIS_EXAMPLE_CUBE_COMMON_HLSL_

/*
	do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/