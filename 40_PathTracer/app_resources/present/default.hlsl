// Copyright (C) 2024-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "renderer/shaders/present/push_constants.hlsl"
// vertex shader is provided by the fullScreenTriangle extension
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t SessionDSIndex = 0;
#include "renderer/shaders/session.hlsl"

using namespace nbl::hlsl;
using namespace nbl::this_example;
using namespace ext::FullScreenTriangle;


[[vk::push_constant]] SDefaultResolvePushConstants pc;

[shader("pixel")]
float32_t4 present_default(SVertexAttributes vxAttr) : SV_Target0
{
    float32_t3 tint = promote<float32_t3>(1.f);
    float32_t3 uv;
    if (pc.isCubemap)
    {
        const float32_t4 ndc = float32_t4(vxAttr.uv*2.f-float32_t2(1,1),1.f,1.f);
        float32_t4 tmp = mul(pc.cubemap().invProjView,ndc);
        float32_t3 dir = tmp.xyz/tmp.www;
        // TODO: convert dir to cubemap face, and the UV coord
        tint = float32_t3(1,0,1); // right now go magenta error colour
    }
    else
    {
        const SDefaultResolvePushConstants::Regular regular = pc.regular();
        uv.xy = vxAttr.uv*regular.scale;
        if (any(uv.xy>float32_t2(1,1)))
            return promote<float32_t4>(0.f);
        uv.z = pc.layer;
        if (any(regular._min>uv.xy) || any(regular._max<uv.xy))
            tint *= 0.33f;
    }
    return float32_t4(gSensorTextures[pc.imageIndex].SampleLevel(gSensorSamplers[0],uv,0.f).rgb*tint,1.0f);
}
