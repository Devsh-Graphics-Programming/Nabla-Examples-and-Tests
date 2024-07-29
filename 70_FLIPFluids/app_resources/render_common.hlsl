#ifndef _FLIP_EXAMPLE_RENDER_COMMON_HLSL
#define _FLIP_EXAMPLE_RENDER_COMMON_HLSL

#ifdef __HLSL_VERSION
struct PSInput
{
	float4 position : SV_Position;
	float4 color : COLOR0;
};
#endif

#endif