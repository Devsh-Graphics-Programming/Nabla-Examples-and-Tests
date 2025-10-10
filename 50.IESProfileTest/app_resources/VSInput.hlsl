#ifndef _NBL_THIS_EXAMPLE_VSINPUT_HLSL_
#define _NBL_THIS_EXAMPLE_VSINPUT_HLSL_

#ifdef __HLSL_VERSION
struct VSInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(3)]] float3 normal : NORMAL;
};
#endif // __HLSL_VERSION
#endif // _NBL_THIS_EXAMPLE_VSINPUT_HLSL_
