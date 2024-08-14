#include "../common.hlsl"
#include "../gridSampling.hlsl"

[[vk::binding(0, 1)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<Particle> particleBuffer;
[[vk::binding(2, 1)]] RWStructuredBuffer<float4> velocityFieldBuffer;

// delta time push constant?

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t pid = ID.x;

    Particle p = particleBuffer[pid];

    // use RK4
    float3 k1 = sampleVelocityAt(p.position.xyz, velocityFieldBuffer, gridData);
    float3 k2 = sampleVelocityAt(p.position.xyz + k1 * 0.5f * deltaTime, velocityFieldBuffer, gridData);
    float3 k3 = sampleVelocityAt(p.position.xyz + k2 * 0.5f * deltaTime, velocityFieldBuffer, gridData);
    float3 k4 = sampleVelocityAt(p.position.xyz + k3 * deltaTime, velocityFieldBuffer, gridData);
    float3 velocity = (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;

    p.position.xyz += velocity * deltaTime;

    p.position = clampPosition(p.position, gridData.worldMin, gridData.worldMax);

    particleBuffer[pid] = p;
}
