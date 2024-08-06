#pragma shader_stage(fragment)

#include "common.hlsl"
#include "render_common.hlsl"

[[vk::binding(0, 1)]]
cbuffer CameraData
{
    SMVPParams camParams;
};

float4 main(PSInput input, out float depthTest : SV_DEPTHGREATEREQUAL) : SV_TARGET
{
    float3 m = normalize(input.vsPos);
    float3 a = -input.vsSpherePos;
    float MdotA = dot(m, a);
    float aLen = length(a);
    float r = input.radius;

    float4 outColor = input.color;

    float r2 = MdotA * MdotA - (aLen * aLen - r * r);
    if (r2 < 0)
        discard;

    float3 vsPos = (-MdotA - sqrt(r2)) * m;

    float3 vsNormal = normalize(vsPos - input.vsSpherePos);
    float3 vsViewDir = normalize(-vsPos);

    float d = vsPos.z;
    if (-d > 0.1f)
        discard;
    depthTest = (camParams.P._m22 * d + camParams.P._m23) / (camParams.P._m32 * d + camParams.P._m33);

    const float fresnelFactor = 0.3;
    float VdotN = dot(vsViewDir, vsNormal);
    float F = 1 + (1.0f - fresnelFactor) * pow(1.0f - VdotN, 5.0) / fresnelFactor;
    outColor.rgb *= F;

    return outColor;
}