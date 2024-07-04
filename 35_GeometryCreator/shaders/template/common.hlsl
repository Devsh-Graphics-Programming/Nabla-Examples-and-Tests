#ifndef _THIS_EXAMPLE_COMMON_HLSL_
#define _THIS_EXAMPLE_COMMON_HLSL_

struct PushConstants
{
	bool withGizmo;

	bool padding[3]; //! size of PushConstants must be multiple of 4
};

#ifdef __HLSL_VERSION
struct SBasicViewParameters //! matches CPU version size & alignment (160, 4)
{
	float4x4 MVP;
	float3x4 MV;
	float3x3 normalMat;
};
#endif // __HLSL_VERSION

#endif

/*
	do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/