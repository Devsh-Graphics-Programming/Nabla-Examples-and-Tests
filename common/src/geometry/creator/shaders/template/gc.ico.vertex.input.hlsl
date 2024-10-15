#ifndef _THIS_EXAMPLE_GC_ICO_VERTEX_INPUT_HLSL_
#define _THIS_EXAMPLE_GC_ICO_VERTEX_INPUT_HLSL_

struct VSInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD;
};

#endif // _THIS_EXAMPLE_GC_ICO_VERTEX_INPUT_HLSL_

/*
    do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/
