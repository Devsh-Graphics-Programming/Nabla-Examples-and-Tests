#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../descriptor_bindings.hlsl"

struct SPushConstants
{
    uint64_t particleAddress;
};

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(b_piGridData, s_pi)]]
cbuffer GridData
{
    SGridData gridData;
};

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t pid = ID.x;

    Particle p;

    int x = pid % (gridData.particleInitSize.x * 2);
    int y = pid / (gridData.particleInitSize.x * 2) % (gridData.particleInitSize.y * 2);
    int z = pid / ((gridData.particleInitSize.x * 2) * (gridData.particleInitSize.y * 2));
    float4 position = gridPosToWorldPos(gridData.particleInitMin + 0.25f + float4(x, y, z, 1) * 0.5f, gridData);
    position = clampPosition(position, gridData.worldMin, gridData.worldMax);

    p.id = pid;
    p.position = position;
    p.velocity = float4(0, 0, 0, 1);

    vk::RawBufferStore<Particle>(pc.particleAddress + sizeof(Particle) * pid, p);
}
