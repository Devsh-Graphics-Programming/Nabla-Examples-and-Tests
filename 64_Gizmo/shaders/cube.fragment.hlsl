#include "cube.common.hlsl"

float4 PSMain(PSInput input) : SV_Target0
{
    return input.color;
}