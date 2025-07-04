#include "template/grid.common.hlsl"


PSInput VSMain(VSInput input)
{
    PSInput output;
    output.position = mul(params.MVP, float4(input.position, 1.0));
    output.uv = (input.uv - float2(0.5, 0.5)) * abs(input.position.xy);
    
    return output;
}