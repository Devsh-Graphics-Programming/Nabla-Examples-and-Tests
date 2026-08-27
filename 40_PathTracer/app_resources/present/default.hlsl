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
        // TODO: currently unused
        // const float32_t4 ndc = float32_t4(vxAttr.uv*2.f-float32_t2(1,1),1.f,1.f);
        // float32_t4 tmp = mul(pc.cubemap().invProjView,ndc);
        // float32_t3 dir = tmp.xyz/tmp.www;

        const uint32_t x = uint32_t(floor(vxAttr.uv.x * 4.0));
        const uint32_t y = uint32_t(floor(vxAttr.uv.y * 3.0));
        const float32_t one_third = 1.0/3.0;
        if (y == 1)
        {
            float32_t2 coord = float32_t2(vxAttr.uv.x * 4.0, (vxAttr.uv.y - one_third) * 3.0);
            uv.xy = float32_t2(coord.x - float32_t(x) * 1.0, coord.y);
            switch (x) // tile index
            {
                case 0:	// -X
                    uv.z = 1.f;
                    break;
                case 1: // +Z
                    uv.z = 4.f;
                    break;
                case 2: // +X
                    uv.z = 0.f;
                    break;
                case 3: // -Z
                    uv.z = 5.f;
                    break;
            }
        }
        else if (x == 1)
        {
            uv.xy = float32_t2((vxAttr.uv.x - 0.25) * 4.0, (vxAttr.uv.y - float32_t(y) * one_third) * 3.0);
            switch (y)
            {
                case 0: // +Y
                    uv.z = 2.f;
                    break;
                case 2: // -Y
                    uv.z = 3.f;
                    break;
            }
        }
        else
            return float32_t4(0,0,0,1);
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
