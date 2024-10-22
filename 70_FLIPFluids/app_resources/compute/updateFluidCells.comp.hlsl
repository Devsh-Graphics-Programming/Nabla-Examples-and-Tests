#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../descriptor_bindings.hlsl"

[[vk::binding(b_ufcGridData, s_ufc)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(b_ufcPBuffer, s_ufc)]]        RWStructuredBuffer<Particle> particleBuffer;
[[vk::binding(b_ufcGridPCountBuffer, s_ufc)]]   RWTexture3D<uint> gridParticleCountBuffer;

[[vk::binding(b_ufcCMInBuffer, s_ufc)]]     RWTexture3D<uint> cellMaterialInBuffer;
[[vk::binding(b_ufcCMOutBuffer, s_ufc)]]    RWTexture3D<uint> cellMaterialOutBuffer;

[[vk::binding(b_ufcVelBuffer, s_ufc)]]      RWTexture3D<float> velocityFieldBuffer[3];
[[vk::binding(b_ufcPrevVelBuffer, s_ufc)]]  RWTexture3D<float> prevVelocityFieldBuffer[3];

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void updateFluidCells(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cIdx = ID;

    uint count = gridParticleCountBuffer[cIdx];
    uint thisCellMaterial =
        isSolidCell(cellIdxToWorldPos(cIdx, gridData)) ? CM_SOLID :
        count > 0 ? CM_FLUID :
        CM_AIR;

    uint cellMaterial = 0;
    setCellMaterial(cellMaterial, thisCellMaterial);

    cellMaterialOutBuffer[cIdx] = cellMaterial;
}

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void updateNeighborFluidCells(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cIdx = ID;

    uint thisCellMaterial = getCellMaterial(cellMaterialInBuffer[cIdx]);
    uint cellMaterial = 0;
    setCellMaterial(cellMaterial, thisCellMaterial);

    uint xpCm = cIdx.x == 0 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[clampToGrid(cIdx + int3(-1, 0, 0), gridData.gridSize)]);
    setXPrevMaterial(cellMaterial, xpCm);

    uint xnCm = cIdx.x == gridData.gridSize.x - 1 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[clampToGrid(cIdx + int3(1, 0, 0), gridData.gridSize)]);
    setXNextMaterial(cellMaterial, xnCm);

    uint ypCm = cIdx.y == 0 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[clampToGrid(cIdx + int3(0, -1, 0), gridData.gridSize)]);
    setYPrevMaterial(cellMaterial, ypCm);

    uint ynCm = cIdx.y == gridData.gridSize.y - 1 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[clampToGrid(cIdx + int3(0, 1, 0), gridData.gridSize)]);
    setYNextMaterial(cellMaterial, ynCm);

    uint zpCm = cIdx.z == 0 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[clampToGrid(cIdx + int3(0, 0, -1), gridData.gridSize)]);
    setZPrevMaterial(cellMaterial, zpCm);

    uint znCm = cIdx.z == gridData.gridSize.z - 1 ? CM_SOLID : getCellMaterial(cellMaterialInBuffer[clampToGrid(cIdx + int3(0, 0, 1), gridData.gridSize)]);
    setZNextMaterial(cellMaterial, znCm);

    cellMaterialOutBuffer[cIdx] = cellMaterial;

    GroupMemoryBarrierWithGroupSync();

    // do final velocity weight here, after sync
    float3 totalVelocity;
    totalVelocity.x = velocityFieldBuffer[0][cIdx];
    totalVelocity.y = velocityFieldBuffer[1][cIdx];
    totalVelocity.z = velocityFieldBuffer[2][cIdx];

    float3 totalWeight;
    totalWeight.x = prevVelocityFieldBuffer[0][cIdx];
    totalWeight.y = prevVelocityFieldBuffer[1][cIdx];
    totalWeight.z = prevVelocityFieldBuffer[2][cIdx];

    float3 velocity = select(totalWeight > 0, totalVelocity / max(totalWeight, FLT_MIN), 0.0f);
    enforceBoundaryCondition(velocity, cellMaterial);

    velocityFieldBuffer[0][cIdx] = velocity.x;
    velocityFieldBuffer[1][cIdx] = velocity.y;
    velocityFieldBuffer[2][cIdx] = velocity.z;
    prevVelocityFieldBuffer[0][cIdx] = velocity.x;
    prevVelocityFieldBuffer[1][cIdx] = velocity.y;
    prevVelocityFieldBuffer[2][cIdx] = velocity.z;
}
