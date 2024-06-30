#include "grid.common.hlsl"

float4 PSMain(PSInput input) : SV_Target0
{
    float2 uv = (input.uv - float2(0.5, 0.5)) + 0.5 / 30.0;
    float grid = gridTextureGradBox(uv, ddx(input.uv), ddy(input.uv));
    float4 fragColor = float4(1.0 - grid, 1.0 - grid, 1.0 - grid, 1.0);
    fragColor *= 0.25;
    fragColor *= 0.3 + 0.6 * smoothstep(0.0, 0.1, 1.0 - length(input.uv) / 5.5);
    
    return fragColor;
}