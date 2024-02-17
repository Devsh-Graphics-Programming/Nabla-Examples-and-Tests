#pragma shader_stage(fragment)

#include "common_test1.hlsl"

struct PSOutput
{
    float4 color : SV_TARGET;
}

PSOutput main(PSInput input)
{
    PSOutput color;
    color = float4(1.0f,2.0f,3.0f,4.0f);
}