#pragma shader_stage(fragment)

#include "common.hlsl"
#include "render_common.hlsl"

[[vk::binding(0, 1)]]
cbuffer CameraData // TODO: BDA instead of UBO, one less thing in DSLayout
{
    SMVPParams camParams;
};

[shader("pixel")]
float4 main(PSInput input, out float depthTest : SV_DEPTHGREATEREQUAL) : SV_TARGET
{
    float3 N;
    N.xy = input.uv * 2.0 - 1.0;
    float r2 = dot(N.xy, N.xy);
    if (r2 > 1.0)
        discard;
    N.z = -sqrt(1.0 - r2);

    float4 pixelPos = float4(input.vsSpherePos + N * input.radius, 1.0);
    float4 clipSpacePos = mul(camParams.P, pixelPos);

    // invert, because reverse z-buffer
    depthTest = 1.0 - clipSpacePos.z / clipSpacePos.w;

    float4 outColor = input.color;
   
    const float3 lightDir = float3(1, 0.5, 0.5);
    float diffuse = max(0.0, dot(N, lightDir));
    outColor += float4(1, 1, 1, 1) * diffuse;

    return outColor;
}