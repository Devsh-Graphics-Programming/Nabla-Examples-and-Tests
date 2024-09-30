#include "../common.hlsl"
#include "../gridSampling.hlsl"
#include "../descriptor_bindings.hlsl"

[[vk::binding(b_apGridData, s_ap)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(b_apPBuffer, s_ap)]] RWStructuredBuffer<Particle> particleBuffer;
[[vk::binding(b_apVelFieldBuffer, s_ap)]] Texture3D<float4> velocityFieldBuffer;
[[vk::binding(b_apVelSampler, s_ap)]] SamplerState velocityFieldSampler;

// delta time push constant?

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t pid = ID.x;

    Particle p = particleBuffer[pid];

    // use RK4
    float3 k1 = sampleVelocityAt(p.position.xyz, velocityFieldBuffer, velocityFieldSampler, gridData);
    float3 k2 = sampleVelocityAt(p.position.xyz + k1 * 0.5f * deltaTime, velocityFieldBuffer, velocityFieldSampler, gridData);
    float3 k3 = sampleVelocityAt(p.position.xyz + k2 * 0.5f * deltaTime, velocityFieldBuffer, velocityFieldSampler, gridData);
    float3 k4 = sampleVelocityAt(p.position.xyz + k3 * deltaTime, velocityFieldBuffer, velocityFieldSampler, gridData);
    float3 velocity = (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;

    p.position.xyz += velocity * deltaTime;

    p.position = clampPosition(p.position, gridData.worldMin, gridData.worldMax);

    particleBuffer[pid] = p;
}