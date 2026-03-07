#ifndef _CAD_EXAMPLE_MAIN_PIPELINE_COMMON_HLSL_INCLUDED_
#define _CAD_EXAMPLE_MAIN_PIPELINE_COMMON_HLSL_INCLUDED_

#include "../globals.hlsl"

struct PSInput
{
    [[vk::location(0)]] float4 position : SV_Position;
};

// Set 0 - Scene Data and Globals, buffer bindings don't change the buffers only get updated

// [[vk::binding(0, 0)]] ConstantBuffer<Globals> globals; ---> moved to globals.hlsl

[[vk::push_constant]] PushConstants pc;

#endif
