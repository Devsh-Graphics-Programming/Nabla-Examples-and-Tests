#pragma kernel updateFluidCells
#pragma kernel updateNeighborFluidCells
#pragma kernel addParticlesToCells

#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../kernel.hlsl"
#include "../descriptor_bindings.hlsl"

[[vk::binding(b_ufcGridData, s_ufc)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(b_ufcPBuffer, s_ufc)]]        RWStructuredBuffer<Particle> particleBuffer;
[[vk::binding(b_ufcGridIDBuffer, s_ufc)]]   RWStructuredBuffer<uint2> gridParticleIDBuffer;

[[vk::binding(b_ufcCMInBuffer, s_ufc)]]     RWStructuredBuffer<uint> cellMaterialInBuffer;
[[vk::binding(b_ufcCMOutBuffer, s_ufc)]]    RWStructuredBuffer<uint> cellMaterialOutBuffer;

[[vk::binding(b_ufcVelBuffer, s_ufc)]]      RWStructuredBuffer<float4> velocityFieldBuffer;
[[vk::binding(b_ufcPrevVelBuffer, s_ufc)]]  RWStructuredBuffer<float4> prevVelocityFieldBuffer;

static const int kernel[6] = { -1, 1, -1, 1, -1, 1 };

[numthreads(WorkgroupSize, 1, 1)]
void updateFluidCells(uint32_t3 ID : SV_DispatchThreadID)
{
    uint tid = ID.x;
    int3 cIdx = flatIdxToCellIdx(tid, gridData.gridSize);

    uint2 pid = gridParticleIDBuffer[tid];
    uint thisCellMaterial =
        isSolidCell(cellIdxToWorldPos(cIdx, gridData)) ? CM_SOLID :
        pid.y - pid.x > 0 ? CM_FLUID :
        CM_AIR;

    uint cellMaterial = 0;
    setCellMaterial(cellMaterial, thisCellMaterial);

    cellMaterialOutBuffer[tid] = cellMaterial;
}

[numthreads(WorkgroupSize, 1, 1)]
void updateNeighborFluidCells(uint32_t3 ID : SV_DispatchThreadID)
{
    uint tid = ID.x;
    int3 cIdx = flatIdxToCellIdx(tid, gridData.gridSize);

    uint thisCellMaterial = getCellMaterial(cellMaterialInBuffer[tid]);
    uint cellMaterial = 0;
    setCellMaterial(cellMaterial, thisCellMaterial);

    uint xpCm = cIdx.x == 0 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[cellIdxToFlatIdx(cIdx + int3(-1, 0, 0), gridData.gridSize)]);
    setXPrevMaterial(cellMaterial, xpCm);

    uint xnCm = cIdx.x == gridData.gridSize.x - 1 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[cellIdxToFlatIdx(cIdx + int3(1, 0, 0), gridData.gridSize)]);
    setXNextMaterial(cellMaterial, xnCm);

    uint ypCm = cIdx.y == 0 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[cellIdxToFlatIdx(cIdx + int3(0, -1, 0), gridData.gridSize)]);
    setYPrevMaterial(cellMaterial, ypCm);

    uint ynCm = cIdx.y == gridData.gridSize.y - 1 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[cellIdxToFlatIdx(cIdx + int3(0, 1, 0), gridData.gridSize)]);
    setYNextMaterial(cellMaterial, ynCm);

    uint zpCm = cIdx.z == 0 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[cellIdxToFlatIdx(cIdx + int3(0, 0, -1), gridData.gridSize)]);
    setZPrevMaterial(cellMaterial, zpCm);

    uint znCm = cIdx.z == gridData.gridSize.z - 1 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[cellIdxToFlatIdx(cIdx + int3(0, 0, 1), gridData.gridSize)]);
    setZNextMaterial(cellMaterial, znCm);

    cellMaterialOutBuffer[tid] = cellMaterial;
}

[numthreads(WorkgroupSize, 1, 1)]
void addParticlesToCells(uint32_t3 ID : SV_DispatchThreadID)
{
    uint tid = ID.x;
    int3 cIdx = flatIdxToCellIdx(tid, gridData.gridSize);

    float3 position = cellIdxToWorldPos(cIdx, gridData);
    float3 posvx = position + float3(-0.5f * gridData.gridCellSize, 0.0f, 0.0f);
    float3 posvy = position + float3(0.0f, -0.5f * gridData.gridCellSize, 0.0f);
    float3 posvz = position + float3(0.0f, 0.0f, -0.5f * gridData.gridCellSize);

    float3 totalWeight = 0;
    float3 totalVel = 0;

    LOOP_PARTICLE_NEIGHBOR_CELLS_BEGIN(cIdx, pid, gridParticleIDBuffer, kernel, gridData.gridSize)
    {
        const Particle p = particleBuffer[pid];

        float3 weight;
        weight.x = getWeight(p.position.xyz, posvx, gridData.gridInvCellSize);
        weight.y = getWeight(p.position.xyz, posvy, gridData.gridInvCellSize);
        weight.z = getWeight(p.position.xyz, posvz, gridData.gridInvCellSize);

        totalWeight += weight;
        totalVel += weight * p.velocity.xyz;
    }
    LOOP_PARTICLE_NEIGHBOR_CELLS_END

    float3 velocity = select(totalWeight > 0, totalVel / max(totalWeight, FLT_MIN), 0.0f);

    enforceBoundaryCondition(velocity, cellMaterialInBuffer[tid]);

    velocityFieldBuffer[tid].xyz = velocity;
    prevVelocityFieldBuffer[tid].xyz = velocity;
}
