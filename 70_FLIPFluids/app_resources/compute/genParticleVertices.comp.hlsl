#pragma shader_stage(compute)

#include "../common.hlsl"
#include "../render_common.hlsl"

[[vk::binding(0, 1)]]
cbuffer CameraData
{
    SMVPParams camParams;
};

[[vk::binding(1, 1)]]
cbuffer ParticleParams
{
    SParticleRenderParams pParams;
};

[[vk::binding(2, 1)]] RWStructuredBuffer<Particle> particleBuffer;
[[vk::binding(0, 2)]] RWStructuredBuffer<VertexInfo> particleVertexBuffer;

static const uint vertexOrder[6] = {0, 1, 2, 2, 1, 3};

static const float4 quadVertices[4] = {
    float4(-1, -1, 0, 1),
    float4(1, -1, 0, 1),
    float4(-1, 1, 0, 1),
    float4(1, 1, 0, 1)
};

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t pid = ID.x;
    Particle p = particleBuffer[pid];

    uint32_t quadBeginIdx = pid * 6;

    float3 wsSpherePos = p.position.xyz;
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

    for (uint i = 0; i < 6; i++)
    {
        VertexInfo vertex;

        vertex.radius = radius;

        float3 pos = mul(mat, quadVertices[vertexOrder[i]]).xyz;
        vertex.vsPos = float4(mul(camParams.V, float4(pos, 1)).xyz, 1);
        vertex.vsSpherePos = float4(mul(camParams.V, float4(wsSpherePos, 1)).xyz, 1);

        vertex.position = mul(camParams.MVP, float4(mul(vertMat, quadVertices[vertexOrder[i]]).xyz, 1));

        vertex.color = float4(0.1, 0.1, 0.8, 1);

        particleVertexBuffer[quadBeginIdx + i] = vertex;
    }
}
