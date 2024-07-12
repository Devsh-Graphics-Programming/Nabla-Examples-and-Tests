#pragma shader_stage(fragment)

#include "common_test1.hlsl"

float4 main(PSInput input) : COLOR
{
    return float4(input.data1.x,input.data2.y,float(outputBuff.Load<uint32_t>(0)),0.0f);
}