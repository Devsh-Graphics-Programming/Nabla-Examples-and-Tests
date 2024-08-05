#pragma shader_stage(geometry)

#include "common.hlsl"
#include "render_common.hlsl"

[[vk::binding(0, 1)]]
cbuffer CameraData
{
    SMVPParams camParams;
};

[[vk::binding(0, 2)]]
cbuffer ParticleParams
{
    SParticleRenderParams pParams;
};

static const float4 quadVertices[4] = {
    float4(-1, -1, 0, 1),
    float4(1, -1, 0, 1),
    float4(-1, 1, 0, 1),
    float4(1, 1, 0, 1)
};

[maxvertexcount(4)]
void main(point GSInput input[1], inout TriangleStream<PSInput> outStream)
{
    float3 wsSpherePos = input[0].particle.xyz;
    float radius = pParams.radius;

    float3 wsCamPos = camParams.camPos.xyz;
    float3 viewDir = wsCamPos - wsSpherePos;
    float dist = length(viewDir);

    float3 z = normalize(-viewDir);
    float3 x = normalize(cross(camParams.V._m10_m11_m12, z));
    float3 y = normalize(cross(z, x));

    float scaledRadius = dist * radius / sqrt(dist * dist - radius * radius);

    float4x4 mat = 0;
    mat._m00_m10_m20 = x;
    mat._m01_m11_m21 = y;
    mat._m02_m12_m22 = z;
    mat._m00_m10_m20 *= scaledRadius;
    mat._m01_m11_m21 *= scaledRadius;
    mat._m03_m13_m23 = wsSpherePos;
    mat._m33 = 1;

    float vertScale = (dist - radius) / dist;
    float4x4 vertMat = mat;
    vertMat._m00_m10_m20 *= vertScale;
    vertMat._m01_m11_m21 *= vertScale;
    vertMat._m03_m13_m23 += viewDir * radius / dist;

    for (uint i = 0; i < 4; i++)
    {
        PSInput output;

        output.radius = radius;

        float3 pos = mul(mat, quadVertices[i]).xyz;
        output.vsPos = mul(camParams.V, float4(pos, 1)).xyz;
        output.vsSpherePos = mul(camParams.V, float4(wsSpherePos, 1)).xyz;

        output.position = mul(camParams.MVP, float4(mul(vertMat, quadVertices[i]).xyz, 1));

        output.color = float4(1, 1, 1, 1);

        outStream.Append(output);
    }
    
    outStream.RestartStrip();
}