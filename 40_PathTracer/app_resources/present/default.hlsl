// Copyright (C) 2024-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "renderer/shaders/session.hlsl"
#include "renderer/shaders/present/push_constants.hlsl"
// vertex shader is provided by the fullScreenTriangle extension
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>

using namespace nbl::hlsl;
using namespace nbl::this_example;
using namespace ext::FullScreenTriangle;


[[vk::push_constant]] DefaultResolvePushConstants pc;

[shader("pixel")]
float32_t4 present_default(SVertexAttributes vxAttr) : SV_Target0
{
    float32_t3 tint = promote<float32_t3>(1);
    float32_t3 uv;
    if (pc.isCubemap)
    {
        const float32_t4 ndc = float32_t4(vxAttr.uv*2.f-float32_t2(1,1),1.f,1.f);
        float32_t4 tmp = mul(pc.operator SDefaultResolvePushConstants::Cubemap().invProjView,ndc);
        float32_t3 dir = tmp.xyz/tmp.www;
        // TODO: convert dir to cubemap face, and the UV coord
        tint = float32_t3(1,0,1); // right now go magenta error colour
    }
    else
    {
        const SDefaultResolvePushConstants::Regular regular = pc.operator SDefaultResolvePushConstants::Regular();
        uv.xy = vxAttr.uv;
        if (regular.scale<0.f)
            uv.y *= -regular.scale;
        else
            uv.y *= regular.scale;
        uv.z = pc.layer;
    }
    return float32_t4(gSensorTextures[pc.imageIndex].SampleLevel(gSensorSamplers[0],uv,0.f).rgb*tint,1.0f);
}
