#pragma shader_stage(fragment)

#include "render_common.hlsl"

float4 main(PSInput input) : SV_TARGET
{
    return input.color;
}