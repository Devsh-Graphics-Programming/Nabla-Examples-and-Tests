#include "common.hlsl"

float4 PSMain(PSInput input) : SV_TARGET
{
    float2 start = input.start_end.xy;
    float2 end =  input.start_end.zw;
    float2 position = input.position.xy;
    return input.color;
}