#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../descriptor_bindings.hlsl"

struct SPushConstants
{
    uint numElements;
    uint pad;
};

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(b_pcuGridData, s_pcu)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(b_pcuPInBuffer, s_pcu)]] RWStructuredBuffer<Particle> particleInBuffer;
[[vk::binding(b_pcuPOutBuffer, s_pcu)]] RWStructuredBuffer<Particle> particleOutBuffer;

[[vk::binding(b_pcuPairBuffer, s_pcu)]] RWStructuredBuffer<uint2> particleCellPairBuffer;
[[vk::binding(b_pcuGridIDBuffer, s_pcu)]] RWStructuredBuffer<uint2> gridParticleIDBuffer;

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
    uint prevp_id = particleCellPairBuffer[currp_id].y;

    particleOutBuffer[currp_id] = particleInBuffer[prevp_id];
}
