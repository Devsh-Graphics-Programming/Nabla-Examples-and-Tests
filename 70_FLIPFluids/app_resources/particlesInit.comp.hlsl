#include "common.hlsl"
#include "gridUtils.hlsl"

[[vk::binding(0, 1)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<Particle> particleBuffer;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint pid = ID.x;

    Particle p = (Particle)0;

    p.id = pid;

    const int x = pid % (gridData.particleInitSize.x * 2);
    const int y = pid / (gridData.particleInitSize.x * 2) % (gridData.particleInitSize.y * 2);
    const int z = pid / ((gridData.particleInitSize.x * 2) * (gridData.particleInitSize.y * 2));
    p.position = gridPosToWorldPos(gridData.particleInitMin + 0.25f + float4(x, y, z, 1) * 0.5f, gridData);
    clampPosition(p.position, gridData.worldMin, gridData.worldMax);

    particleBuffer[pid] = p;
}
