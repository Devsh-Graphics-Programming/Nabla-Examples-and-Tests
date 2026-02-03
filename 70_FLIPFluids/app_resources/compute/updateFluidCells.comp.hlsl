#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../descriptor_bindings.hlsl"

#include "nbl/builtin/hlsl/limits.hlsl"

[[vk::binding(b_ufcGridData, s_ufc)]]
cbuffer GridData
{
    SGridData gridData;
};

// TODO: image shouldn't exist, atomics should be performed directly on `cellMaterial`
[[vk::binding(b_ufcGridPCount, s_ufc)]]   RWTexture3D<uint> gridParticleCount;

// TODO: If 0 is AIR, and >=2 is SOLID, we can perform Atomic OR 0b01 to set FLUID, this means we only need to reset the `cellMaterial` with a bitwise-AND and only need one image copy
[[vk::binding(b_ufcCMIn, s_ufc)]]     RWTexture3D<uint> cellMaterialIn;
[[vk::binding(b_ufcCMOut, s_ufc)]]    RWTexture3D<uint> cellMaterialOut;

[[vk::binding(b_ufcVel, s_ufc)]]      RWTexture3D<float> velocityField[3];
[[vk::binding(b_ufcPrevVel, s_ufc)]]  RWTexture3D<float> prevVelocityField[3];

// TODO: f 0 is AIR, and >=2 is SOLID, we can perform Atomic OR 0b01 to have a particle set the cell to FLUID, and this dispatch looping over all grid cells is not needed!
[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
[shader("compute")]
void updateFluidCells(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cIdx = ID;

    uint count = gridParticleCount[cIdx];
    uint thisCellMaterial =
        isSolidCell(cellIdxToWorldPos(cIdx, gridData)) ? CM_SOLID :
        count > 0 ? CM_FLUID :
        CM_AIR;

    uint cellMaterial = 0;
    setCellMaterial(cellMaterial, thisCellMaterial);

    cellMaterialOut[cIdx] = cellMaterial;
}

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
[shader("compute")]
void updateNeighborFluidCells(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cIdx = ID;

    uint thisCellMaterial = getCellMaterial(cellMaterialIn[cIdx]);
    uint cellMaterial = 0;
    setCellMaterial(cellMaterial, thisCellMaterial);

    uint xpCm = cIdx.x == 0 ? CM_SOLID : getCellMaterial(cellMaterialIn[clampToGrid(cIdx + int3(-1, 0, 0), gridData.gridSize)]);
    setXPrevMaterial(cellMaterial, xpCm);

    uint xnCm = cIdx.x == gridData.gridSize.x - 1 ? CM_SOLID : getCellMaterial(cellMaterialIn[clampToGrid(cIdx + int3(1, 0, 0), gridData.gridSize)]);
    setXNextMaterial(cellMaterial, xnCm);

    uint ypCm = cIdx.y == 0 ? CM_SOLID : getCellMaterial(cellMaterialIn[clampToGrid(cIdx + int3(0, -1, 0), gridData.gridSize)]);
    setYPrevMaterial(cellMaterial, ypCm);

    uint ynCm = cIdx.y == gridData.gridSize.y - 1 ? CM_SOLID : getCellMaterial(cellMaterialIn[clampToGrid(cIdx + int3(0, 1, 0), gridData.gridSize)]);
    setYNextMaterial(cellMaterial, ynCm);

    uint zpCm = cIdx.z == 0 ? CM_SOLID : getCellMaterial(cellMaterialIn[clampToGrid(cIdx + int3(0, 0, -1), gridData.gridSize)]);
    setZPrevMaterial(cellMaterial, zpCm);

    uint znCm = cIdx.z == gridData.gridSize.z - 1 ? CM_SOLID : getCellMaterial(cellMaterialIn[clampToGrid(cIdx + int3(0, 0, 1), gridData.gridSize)]);
    setZNextMaterial(cellMaterial, znCm);

    cellMaterialOut[cIdx] = cellMaterial;

    GroupMemoryBarrierWithGroupSync();

    // do final velocity weight here, after sync
    float3 totalVelocity;
    totalVelocity.x = velocityField[0][cIdx];
    totalVelocity.y = velocityField[1][cIdx];
    totalVelocity.z = velocityField[2][cIdx];

    float3 totalWeight;
    totalWeight.x = prevVelocityField[0][cIdx];
    totalWeight.y = prevVelocityField[1][cIdx];
    totalWeight.z = prevVelocityField[2][cIdx];

    float3 velocity = select(totalWeight > 0, totalVelocity / max(totalWeight, nbl::hlsl::numeric_limits<float32_t>::min), 0.0f);
    enforceBoundaryCondition(velocity, cellMaterial);

    velocityField[0][cIdx] = velocity.x;
    velocityField[1][cIdx] = velocity.y;
    velocityField[2][cIdx] = velocity.z;
    prevVelocityField[0][cIdx] = velocity.x;
    prevVelocityField[1][cIdx] = velocity.y;
    prevVelocityField[2][cIdx] = velocity.z;
}
