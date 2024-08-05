#pragma shader_stage(fragment)

#include "common.hlsl"
#include "render_common.hlsl"

float4 main(PSInput input) : SV_TARGET
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
    float depth = vsPos.z;

    return outColor;
}