#ifndef _NBL_THIS_EXAMPLE_PSINPUT_HLSL_
#define _NBL_THIS_EXAMPLE_PSINPUT_HLSL_

#ifdef __HLSL_VERSION
struct PSInput
{
	float32_t4 position : SV_Position;
};
#endif // __HLSL_VERSION
#endif // _NBL_THIS_EXAMPLE_PSINPUT_HLSL_
