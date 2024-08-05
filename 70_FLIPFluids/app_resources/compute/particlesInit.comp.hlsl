#include "../common.hlsl"
#include "../gridUtils.hlsl"

[[vk::binding(0, 1)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(1, 1)]] RWStructuredBuffer<Particle> particleBuffer;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t pid = ID.x;

    Particle p;

    int x = pid % (gridData.particleInitSize.x * 2);
    int y = pid / (gridData.particleInitSize.x * 2) % (gridData.particleInitSize.y * 2);
    int z = pid / ((gridData.particleInitSize.x * 2) * (gridData.particleInitSize.y * 2));
    float4 position = gridPosToWorldPos(gridData.particleInitMin + 0.25f + float4(x, y, z, 1) * 0.5f, gridData);
    clampPosition(position, gridData.worldMin, gridData.worldMax);

    // particleBuffer[pid].id = pid;
    // particleBuffer[pid].position = float4(1, 0, 0, 1);
    // particleBuffer[pid].velocity = position;

    p.id = 1;
    p.position = position;
    p.velocity = float4(0, 0, 0, 0);

    particleBuffer[pid] = p;
}
