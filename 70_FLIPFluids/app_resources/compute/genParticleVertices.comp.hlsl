#pragma shader_stage(compute)

#include "../common.hlsl"
#include "../render_common.hlsl"
#include "../descriptor_bindings.hlsl"


// TODO: this whole shader needs to disappear!
// One can just compute `offset` on line 61 from `gl_InstanceIndex` and `i` from `gl_VertexIndex&0b11u`
// Sure it leads to redundant computation of everything before the `for` loop but no bandwidth consumed and no extra compute dispatch
struct SPushConstants
{
    uint64_t particlePosAddress;
    uint64_t particleVelAddress;
    uint64_t particleVerticesAddress;
};

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(b_gpvCamData, s_gpv)]]
cbuffer CameraData
{
    SMVPParams camParams;
};

[[vk::binding(b_gpvPParams, s_gpv)]]
cbuffer ParticleParams
{
    SParticleRenderParams pParams;
};


static const uint vertexOrder[6] = {0, 1, 2, 2, 1, 3};

// static const float4 quadVertices[4] = {
//     float4(-1, -1, 0, 1),
//     float4(1, -1, 0, 1),
//     float4(-1, 1, 0, 1),
//     float4(1, 1, 0, 1)
// };

static const float4 quadVertices[4] = {
    float4(-0.5f, -0.5f, 0.0f, 1.0f),
    float4(0.5f, -0.5f, 0.0f, 1.0f),
    float4(-0.5f, 0.5f, 0.0f, 1.0f),
    float4(0.5f, 0.5f, 0.0f, 1.0f),
};

static const float2 quadUVs[4] = {
    float2(0, 0),
    float2(1, 0),
    float2(0, 1),
    float2(1, 1)
};

#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
using namespace nbl::hlsl;

[numthreads(WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t pid = ID.x;
    Particle p;

    int offset = sizeof(float32_t3) * pid;
    p.position = vk::RawBufferLoad<float32_t3>(pc.particlePosAddress + offset);
    p.velocity = vk::RawBufferLoad<float32_t3>(pc.particleVelAddress + offset);

    uint32_t quadBeginIdx = pid * 6;

    float3 wsSpherePos = p.position;
    float radius = pParams.radius;

    float3 wsCamPos = camParams.camPos.xyz;
    float3 viewDir = wsCamPos - wsSpherePos;
    float dist = length(viewDir);

    float3 z = normalize(-viewDir);
    float3 x = normalize(cross(camParams.V._m10_m11_m12, z));
    float3 y = normalize(cross(z, x));

    float scaledRadius = dist * radius / sqrt(dist * dist - radius * radius);

    const float4 color1 = float4(0, 0.35, 0.75, 1);
    const float4 color2 = float4(0.85, 0.2, 0, 1);
    float speed = length(p.velocity);
    float factor = saturate((speed - 0.5) / 10.0);
    float4 color = lerp(color1, color2, factor);

    float4x4 mat = 0;
    mat._m00_m10_m20 = x;
    mat._m01_m11_m21 = y;
    mat._m02_m12_m22 = z;
    mat._m00_m10_m20 *= scaledRadius;
    mat._m01_m11_m21 *= scaledRadius;
    mat._m03_m13_m23 = wsSpherePos;
    mat._m33 = 1;

    float vertScale = (dist - radius) / dist;
    mat._m00_m10_m20 *= vertScale;
    mat._m01_m11_m21 *= vertScale;
    mat._m03_m13_m23 += viewDir * radius / dist;

    for (uint i = 0; i < 6; i++)
    {
        VertexInfo vertex;

        vertex.radius = radius;
        
        vertex.vsSpherePos = float4(mul(camParams.V, float4(wsSpherePos, 1)).xyz, 1);

        vertex.position = mul(camParams.MVP, float4(mul(mat, quadVertices[vertexOrder[i]]).xyz, 1));

        vertex.color = color;

        vertex.uv = quadUVs[vertexOrder[i]];

        (bda::__ptr<VertexInfo>::create(pc.particleVerticesAddress)+(quadBeginIdx+i)).deref_restrict().store(vertex);
    }
}
