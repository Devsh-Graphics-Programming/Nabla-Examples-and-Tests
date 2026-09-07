#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../descriptor_bindings.hlsl"

using namespace nbl::hlsl;

struct SPushConstants
{
    uint64_t particlePosAddress;
    uint64_t particleVelAddress;
};

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(b_piGridData, s_pi)]]
cbuffer GridData
{
    SGridData gridData;
};

[numthreads(WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t pid = ID.x;

    Particle p;

    const int pidSigned = _static_cast<int>(pid);
    int x = pidSigned % (gridData.particleInitSize.x * 2);
    int y = pidSigned / (gridData.particleInitSize.x * 2) % (gridData.particleInitSize.y * 2);
    int z = pidSigned / ((gridData.particleInitSize.x * 2) * (gridData.particleInitSize.y * 2));
    float3 position = gridPosToWorldPos(gridData.particleInitMin.xyz + 0.25f + float3(x, y, z) * 0.5f, gridData);
    position = clampPosition(position, gridData.worldMin, gridData.worldMax);

    p.position = position;
    p.velocity = (float3)0;

    uint32_t offset = sizeof(float32_t3) * pid;
    vk::RawBufferStore<float32_t3>(pc.particlePosAddress + offset, p.position);
    vk::RawBufferStore<float32_t3>(pc.particleVelAddress + offset, p.velocity);
}
