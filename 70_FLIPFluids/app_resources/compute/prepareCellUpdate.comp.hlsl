#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../kernel.hlsl"
#include "../descriptor_bindings.hlsl"

[[vk::binding(b_ufcGridData, s_ufc)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(b_ufcPBuffer, s_ufc)]]        RWStructuredBuffer<Particle> particleBuffer;
[[vk::binding(b_ufcGridPCountBuffer, s_ufc)]]   RWTexture3D<uint> gridParticleCountBuffer;

[[vk::binding(b_ufcVelBuffer, s_ufc)]]      RWTexture3D<uint> velocityFieldBuffer[3];
[[vk::binding(b_ufcPrevVelBuffer, s_ufc)]]  RWTexture3D<uint> prevVelocityFieldBuffer[3];

void casAdd(RWTexture3D<uint> grid, int3 idx, float value)
{
    uint actualValue = 0;
    uint expectedValue;
    do
    {
        expectedValue = actualValue;
        uint newValue = asuint(asfloat(actualValue) + value);
        InterlockedCompareExchange(grid[idx], expectedValue, newValue, actualValue);
    } while (actualValue != expectedValue);
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    uint pid = ID.x;
    Particle p = particleBuffer[pid];
    int3 cIdx = worldPosToCellIdx(p.position.xyz, gridData);

    for (int i = max(cIdx.x - 1, 0); i <= min(cIdx.x + 1, gridData.gridSize.x - 1); i++)
    {
        for (int j = max(cIdx.y - 1, 0); j <= min(cIdx.y + 1, gridData.gridSize.y - 1); j++)
        {
            for (int k = max(cIdx.z - 1, 0); k <= min(cIdx.z + 1, gridData.gridSize.z - 1); k++)
            {
                int3 cellIdx = int3(i, j, k);
                float3 position = cellIdxToWorldPos(cellIdx, gridData);
                float3 posvx = position + float3(-0.5f * gridData.gridCellSize, 0.0f, 0.0f);
                float3 posvy = position + float3(0.0f, -0.5f * gridData.gridCellSize, 0.0f);
                float3 posvz = position + float3(0.0f, 0.0f, -0.5f * gridData.gridCellSize);

                float3 weight;
                weight.x = getWeight(p.position.xyz, posvx, gridData.gridInvCellSize);
                weight.y = getWeight(p.position.xyz, posvy, gridData.gridInvCellSize);
                weight.z = getWeight(p.position.xyz, posvz, gridData.gridInvCellSize);

                float3 velocity = weight * p.velocity.xyz;

                // store weighted velocity in velocity buffer
                casAdd(velocityFieldBuffer[0], cellIdx, velocity.x);
                casAdd(velocityFieldBuffer[1], cellIdx, velocity.y);
                casAdd(velocityFieldBuffer[2], cellIdx, velocity.z);

                // store total weight in prev velocity buffer
                casAdd(prevVelocityFieldBuffer[0], cellIdx, weight.x);
                casAdd(prevVelocityFieldBuffer[1], cellIdx, weight.y);
                casAdd(prevVelocityFieldBuffer[2], cellIdx, weight.z);
            }
        }
    }

    InterlockedAdd(gridParticleCountBuffer[cIdx], 1);
}
