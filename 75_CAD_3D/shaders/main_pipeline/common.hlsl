#ifndef _CAD_3D_EXAMPLE_MAIN_PIPELINE_COMMON_HLSL_INCLUDED_
#define _CAD_3D_EXAMPLE_MAIN_PIPELINE_COMMON_HLSL_INCLUDED_

#include "../globals.hlsl"

struct PSInput
{
    [[vk::location(0)]] float4 position : SV_Position;

    [[vk::location(1)]] nointerpolation float4 data1 : COLOR1;
    [[vk::location(2)]] float4 interpolatedData1 : COLOR2;

    // TODO: do we even need vertexScreenSpacePos?
#ifndef FRAGMENT_SHADER_INPUT // vertex shader
    [[vk::location(3)]] float3 vertexScreenSpacePos : COLOR3;
#else
    [[vk::location(3)]] [[vk::ext_decorate(/*spv::DecoratePerVertexKHR*/5285)]] float3 vertexScreenSpacePos[3] : COLOR3;
#endif

    void setNormal(NBL_CONST_REF_ARG(float3) normal) { data1.xyz = normal; }
    float3 getNormal() { return data1.xyz; }

    void setHeight(float height) { interpolatedData1.x = height; }
    float getHeight() { return interpolatedData1.x; }

#ifndef FRAGMENT_SHADER_INPUT // vertex shader
    void setScreenSpaceVertexAttribs(float3 pos) { vertexScreenSpacePos = pos; }
#else // fragment shader
    float3 getScreenSpaceVertexAttribs(uint32_t vertexIndex) { return vertexScreenSpacePos[vertexIndex]; }
#endif
};

// [[vk::binding(0, 0)]] ConstantBuffer<Globals> globals; ---> moved to globals.hlsl

[[vk::push_constant]] PushConstants pc;

#endif
