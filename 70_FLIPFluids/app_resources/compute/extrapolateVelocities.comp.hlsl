#include "../common.hlsl"
#include "../gridSampling.hlsl"

[[vk::binding(0, 1)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<Particle> particleBuffer;
[[vk::binding(2, 1)]] RWStructuredBuffer<float4> velocityFieldBuffer;
[[vk::binding(3, 1)]] RWStructuredBuffer<float4> prevVelocityFieldBuffer;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t pid = ID.x;

    Particle p = particleBuffer[pid];

    float3 gridPrevVel = sampleVelocityAt(p.position.xyz, prevVelocityFieldBuffer, gridData);
    float3 gridVel = sampleVelocityAt(p.position.xyz, velocityFieldBuffer, gridData);

    float3 picVel = gridVel;
    float3 flipVel = p.velocity.xyz + gridVel - gridPrevVel;

    p.velocity.xyz = lerp(picVel, flipVel, ratioFLIPPIC);

    particleBuffer[pid] = p;
}