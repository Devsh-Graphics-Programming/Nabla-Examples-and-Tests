#include "../common.hlsl"
#include "../gridSampling.hlsl"
#include "../descriptor_bindings.hlsl"

[[vk::binding(b_evGridData, s_ev)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(b_evPBuffer, s_ev)]] RWStructuredBuffer<Particle> particleBuffer;
[[vk::binding(b_evVelFieldBuffer, s_ev)]] RWStructuredBuffer<float4> velocityFieldBuffer;
[[vk::binding(b_evPrevVelFieldBuffer, s_ev)]] RWStructuredBuffer<float4> prevVelocityFieldBuffer;

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