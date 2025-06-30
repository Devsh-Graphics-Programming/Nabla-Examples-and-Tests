#pragma shader_stage(fragment)

#include "simple_common.hlsl"

[shader("pixel")]
float32_t4 main(PSInput input) : SV_TARGET
{
    float32_t4 outColor = input.color;

    return outColor;
}