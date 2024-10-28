#ifndef _THIS_EXAMPLE_GC_COMMON_HLSL_
#define _THIS_EXAMPLE_GC_COMMON_HLSL_

#ifdef __HLSL_VERSION
	struct PSInput
	{
		float4 position : SV_Position;
		float4 color : COLOR0;
	};
#endif // __HLSL_VERSION

#include "SBasicViewParameters.hlsl"

#endif // _THIS_EXAMPLE_GC_COMMON_HLSL_

/*
	do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/