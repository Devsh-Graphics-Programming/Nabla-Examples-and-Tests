#pragma kernel updateFluidCells
#pragma kernel updateNeighborFluidCells
#pragma kernel addParticlesToCells

#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../kernel.hlsl"

[[vk::binding(0, 1)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(1, 1)]] StructuredBuffer<Particle> particleBuffer;
[[vk::binding(2, 1)]] StructuredBuffer<uint2> gridParticleIDBuffer;

[[vk::binding(3, 1)]] RWStructuredBuffer<uint> gridCellTypeBuffer;
[[vk::binding(4, 1)]] RWStructuredBuffer<float4> velocityFieldBuffer;
[[vk::binding(5, 1)]] RWStructuredBuffer<float4> prevVelocityFieldBuffer;

static const kernel[6] = { -1, 1, -1, 1, -1, 1 };

[numthreads(WorkgroupSize, 1, 1)]
void updateFluidCells(uint32_t3 ID : SV_DispatchThreadID)
{
    uint tid = ID.x;
    int3 cIdx = flatIdxToCellIdx(tid, gridData.gridSize);

    uint2 pid = gridParticleIDBuffer[tid];
    uint cType =
        isSolidCell(cellIdxToWorldPos(cIdx)) ? CM_SOLID :
        pid.y - pid.x > 0 ? CM_FLUID :
        CM_AIR;

    uint cellMaterial = 0;
    setCellMaterial(cellMaterial, cType);

    gridCellTypeBuffer[tid] = cellMaterial;
}

[numthreads(WorkgroupSize, 1, 1)]
void updateNeighborFluidCells(uint32_t3 ID : SV_DispatchThreadID)
{
    uint tid = ID.x;
    int3 cIdx = flatIdxToCellIdx(tid, gridData.gridSize);

    uint thisCellMaterial = getCellMaterial(gridCellTypeBuffer[tid]);
    uint cellMaterial = 0;
    setCellMaterial(cellMaterial, cType);

    uint xpCm = cIdx.x == 0 ? CM_SOLID : getCellMaterial(gridCellTypeBuffer[cellIdxToFlatIdx(cIdx + int3(-1, 0, 0))]);
    setXPrevMaterial(cellMaterial, xpCm);

    uint xnCm = cIdx.x == gridData.gridSize.x - 1 ? CM_SOLID : getCellMaterial(gridCellTypeBuffer[cellIdxToFlatIdx(cIdx + int3(1, 0, 0))]);
    setXNextMaterial(cellMaterial, xnCm);

    uint ypCm = cIdx.y == 0 ? CM_SOLID : getCellMaterial(gridCellTypeBuffer[cellIdxToFlatIdx(cIdx + int3(0, -1, 0))]);
    setYPrevMaterial(cellMaterial, ypCm);

    uint ynCm = cIdx.y == gridData.gridSize.y - 1 ? CM_SOLID : getCellMaterial(gridCellTypeBuffer[cellIdxToFlatIdx(cIdx + int3(0, 1, 0))]);
    setYNextMaterial(cellMaterial, ynCm);

    uint zpCm = cIdx.z == 0 ? CM_SOLID : getCellMaterial(gridCellTypeBuffer[cellIdxToFlatIdx(cIdx + int3(0, 0, -1))]);
    setZPrevMaterial(cellMaterial, zpCm);

    uint znCm = cIdx.z == gridData.gridSize.z - 1 ? CM_SOLID : getCellMaterial(gridCellTypeBuffer[cellIdxToFlatIdx(cIdx + int3(0, 0, 1))]);
    setZNextMaterial(cellMaterial, znCm);

    gridCellTypeBuffer[tid] = cellMaterial;
}

[numthreads(WorkgroupSize, 1, 1)]
void addParticlesToCells(uint32_t3 ID : SV_DispatchThreadID)
{
    uint tid = ID.x;
    int3 cIdx = flatIdxToCellIdx(tid, gridData.gridSize);

    float3 position = cellIdxToWorldPos(cIdx);
    float3 posvx = position + float3(-0.5f * gridData.gridCellSize, 0.0f, 0.0f);
    float3 posyx = position + float3(0.0f, -0.5f * gridData.gridCellSize, 0.0f);
    float3 poszx = position + float3(0.0f, 0.0f, -0.5f * gridData.gridCellSize);

    float3 totalWeight = 0;
    float3 totalVel = 0;

    LOOP_PARTICLE_NEIGHBOR_CELLS_BEGIN(cIdx, pid, gridParticleIDBuffer, kernel, gridData.gridCellSize)
    {
        const Particle p = particleBuffer[pid];

        float3 weight;
        weight.x = getWeight(p.position, posvx, gridData.gridInvCellSize);
        weight.y = getWeight(p.position, posvy, gridData.gridInvCellSize);
        weight.z = getWeight(p.position, posvz, gridData.gridInvCellSize);

        totalWeight += weight;
        totalVel += weight * p.velocity;
    }
    LOOP_PARTICLE_NEIGHBOR_CELLS_END

    float3 velocity = totalWeight > 0 ? totalVel / max(totalWeight, FLT_MIN) : 0.0f;

    enforceBoundaryCondition(velocity, gridCellTypeBuffer[tid]);

    velocityFieldBuffer[tid] = velocity;
    prevVelocityFieldBuffer[tid] = velocity;
}
