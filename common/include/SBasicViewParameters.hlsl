#ifndef _S_BASIC_VIEW_PARAMETERS_COMMON_HLSL_
#define _S_BASIC_VIEW_PARAMETERS_COMMON_HLSL_

#ifdef __HLSL_VERSION
struct SBasicViewParameters //! matches CPU version size & alignment (160, 4)
{
	float4x4 MVP;
	float3x4 MV;
	float3x3 normalMat;
};
#endif // _S_BASIC_VIEW_PARAMETERS_COMMON_HLSL_

#endif

/*
	do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/