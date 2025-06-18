#ifndef _NBL_EXAMPLES_GC_COMMON_HLSL_
#define _NBL_EXAMPLES_GC_COMMON_HLSL_


#include "common/SBasicViewParameters.hlsl"

#ifdef __HLSL_VERSION
[[vk::push_constant]] SBasicViewParameters params;

struct PSInput
{
	float4 position : SV_Position;
	float3 color : COLOR0;
};
#endif // __HLSL_VERSION


#endif // _NBL_EXAMPLES_GC_COMMON_HLSL_

/*
	do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/