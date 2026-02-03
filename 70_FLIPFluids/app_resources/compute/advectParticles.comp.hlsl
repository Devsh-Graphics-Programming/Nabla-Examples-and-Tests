#include "../common.hlsl"
#include "../gridSampling.hlsl"
#include "../descriptor_bindings.hlsl"

struct SPushConstants
{
    uint64_t particlePosAddress;
    uint64_t particleVelAddress;
};

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(b_apGridData, s_ap)]]
cbuffer GridData
{
    SGridData gridData;
};


[[vk::binding(b_apVelField, s_ap)]] Texture3D<float> velocityField[3];
[[vk::binding(b_apPrevVelField, s_ap)]] Texture3D<float> prevVelocityField[3];
[[vk::binding(b_apVelSampler, s_ap)]] SamplerState velocityFieldSampler;

#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
using namespace nbl::hlsl;

// TODO: delta time push constant? (but then for CI need a commandline `-fixed-timestep=MS` and `-frames=N` option too)
[numthreads(WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t pid = ID.x;
    Particle p;

    // use a restrict reference for speed
    bda::__ref<float32_t3,4,true> rPosition = (bda::__ptr<float32_t3>::create(pc.particlePosAddress)+pid).deref<4,true>();
    bda::__ref<float32_t3,4,true> rVelocity = (bda::__ptr<float32_t3>::create(pc.particleVelAddress)+pid).deref<4,true>();
    p.position = rPosition.load();
    p.velocity = rVelocity.load();

    // advect velocity
    float3 gridPrevVel = sampleVelocityAt(p.position, prevVelocityField, velocityFieldSampler, gridData);
    float3 gridVel = sampleVelocityAt(p.position, velocityField, velocityFieldSampler, gridData);

    float3 picVel = gridVel;
    float3 flipVel = p.velocity + gridVel - gridPrevVel;

    p.velocity = lerp(picVel, flipVel, ratioFLIPPIC);

    // move particle, use RK4
    float3 k1 = gridVel;
    float3 k2 = sampleVelocityAt(p.position + k1 * 0.5f * deltaTime, velocityField, velocityFieldSampler, gridData);
    float3 k3 = sampleVelocityAt(p.position + k2 * 0.5f * deltaTime, velocityField, velocityFieldSampler, gridData);
    float3 k4 = sampleVelocityAt(p.position + k3 * deltaTime, velocityField, velocityFieldSampler, gridData);
    float3 velocity = (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;

    p.position += velocity * deltaTime;

    p.position = clampPosition(p.position, gridData.worldMin, gridData.worldMax);

    rPosition.store(p.position);
    rVelocity.store(p.velocity);
}
