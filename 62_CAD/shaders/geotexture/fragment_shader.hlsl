#pragma shader_stage(fragment)

#include "common.hlsl"

float4 main(PSInput input) : SV_TARGET
{
    const float2 uv = input.uv;
    return geoTexture.Sample(geoTextureSampler, uv);
}