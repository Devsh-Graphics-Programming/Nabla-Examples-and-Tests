#include "gc.common.hlsl"

PSInput VSMain()
{
    PSInput output;

    output.position = mul(params.MVP, float4(input.position, 1.0));
    output.color = float4(input.normal * 0.5 + 0.5, 1.0);

    return output;
}

/*
    do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/
