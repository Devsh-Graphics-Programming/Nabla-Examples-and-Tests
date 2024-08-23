#include "../common.hlsl"
#include "../gridUtils.hlsl"

struct SPushConstants
{
    uint numElements;
    uint pad;
}

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 1)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<Particle> particleInBuffer;
[[vk::binding(2, 1)]] RWStructuredBuffer<Particle> particleOutBuffer;

[[vk::binding(3, 1)]] RWStructuredBuffer<uint2> particleCellPairBuffer;
[[vk::binding(4, 1)]] RWStructuredBuffer<uint2> gridParticleIDBuffer;

[numthreads(WorkgroupSize, 1, 1)]
void makeParticleCellPairs(uint32_t3 ID : SV_DispatchThreadID)
{
    uint p_id = ID.x;
    uint c_id = worldPosToFlatIdx(particleInBuffer[p_id].position.xyz, gridData);

    particleCellPairBuffer[p_id] = uint2(c_id, p_id);
}

[numthreads(WorkgroupSize, 1, 1)]
void setGridParticleID(uint32_t3 ID : SV_DispatchThreadID)
{
    uint currp_id = ID.x;

    uint prevp_id = currp_id == 0 ? pc.numElements - 1 : currp_id - 1;
    uint nextp_id = currp_id == pc.numElements - 1 ? 0 : currp_id + 1;
    uint currc_id = particleCellPairBuffer[currp_id].x;
    uint prevc_id = particleCellPairBuffer[prevp_id].x;
    uint nextc_id = particleCellPairBuffer[nextp_id].x;

    if (currc_id != prevc_id)
        gridParticleIDBuffer[currc_id].x = currp_id;
    if (currc_id != nextc_id)
        gridParticleIDBuffer[currc_id].y = currp_id + 1;
}

[numthreads(WorkgroupSize, 1, 1)]
void shuffleParticles(uint32_t3 ID : SV_DispatchThreadID)
{
    uint currp_id = ID.x;
    uint prevp_id = particleCellPairBuffer[p_id].y;

    particleOutBuffer[p_id] = particleInBuffer[prevp_id];
}
