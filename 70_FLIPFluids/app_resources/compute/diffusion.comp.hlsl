#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../descriptor_bindings.hlsl"

struct SPushConstants
{
    float4 diffusionParameters;
};

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(b_dGridData, s_d)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(b_dCMBuffer, s_d)]] RWTexture3D<uint> cellMaterialBuffer;
[[vk::binding(b_dVelBuffer, s_d)]] RWTexture3D<float> velocityFieldBuffer[3];
[[vk::binding(b_dAxisInBuffer, s_d)]] RWTexture3D<uint4> axisCellMaterialInBuffer;
[[vk::binding(b_dAxisOutBuffer, s_d)]] RWTexture3D<uint4> axisCellMaterialOutBuffer;
[[vk::binding(b_dDiffInBuffer, s_d)]] RWTexture3D<float4> gridDiffusionInBuffer;
[[vk::binding(b_dDiffOutBuffer, s_d)]] RWTexture3D<float4> gridDiffusionOutBuffer;

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void setAxisCellMaterial(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cellIdx = ID;

    uint cellMaterial = cellMaterialBuffer[cellIdx];

    uint this_cm = getCellMaterial(cellMaterial);
    uint xp_cm = getXPrevMaterial(cellMaterial);
    uint yp_cm = getYPrevMaterial(cellMaterial);
    uint zp_cm = getZPrevMaterial(cellMaterial);

    uint3 cellAxisType;
    cellAxisType.x = 
        isSolidCell(this_cm) || isSolidCell(xp_cm) ? CM_SOLID :
        isFluidCell(this_cm) || isFluidCell(xp_cm) ? CM_FLUID :
        CM_AIR;
    cellAxisType.y = 
        isSolidCell(this_cm) || isSolidCell(yp_cm) ? CM_SOLID :
        isFluidCell(this_cm) || isFluidCell(yp_cm) ? CM_FLUID :
        CM_AIR;
    cellAxisType.z = 
        isSolidCell(this_cm) || isSolidCell(zp_cm) ? CM_SOLID :
        isFluidCell(this_cm) || isFluidCell(zp_cm) ? CM_FLUID :
        CM_AIR;

    uint3 cmAxisTypes = 0;
    setCellMaterial(cmAxisTypes, cellAxisType);

    axisCellMaterialOutBuffer[cellIdx].xyz = cmAxisTypes;
}

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void setNeighborAxisCellMaterial(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cellIdx = ID;

    uint3 axisCm = (uint3)0;
    uint3 this_axiscm = getCellMaterial(axisCellMaterialInBuffer[cellIdx].xyz);
    setCellMaterial(axisCm, this_axiscm);

    uint3 xp_axiscm = cellIdx.x == 0 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialInBuffer[clampToGrid(cellIdx + int3(-1, 0, 0), gridData.gridSize)].xyz);
    setXPrevMaterial(axisCm, xp_axiscm);

    uint3 xn_axiscm = cellIdx.x == gridData.gridSize.x - 1 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialInBuffer[clampToGrid(cellIdx + int3(1, 0, 0), gridData.gridSize)].xyz);
    setXNextMaterial(axisCm, xn_axiscm);

    uint3 yp_axiscm = cellIdx.y == 0 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialInBuffer[clampToGrid(cellIdx + int3(0, -1, 0), gridData.gridSize)].xyz);
    setYPrevMaterial(axisCm, yp_axiscm);

    uint3 yn_axiscm = cellIdx.y == gridData.gridSize.y - 1 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialInBuffer[clampToGrid(cellIdx + int3(0, 1, 0), gridData.gridSize)].xyz);
    setYNextMaterial(axisCm, yn_axiscm);

    uint3 zp_axiscm = cellIdx.z == 0 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialInBuffer[clampToGrid(cellIdx + int3(0, 0, -1), gridData.gridSize)].xyz);
    setZPrevMaterial(axisCm, zp_axiscm);

    uint3 zn_axiscm = cellIdx.z == gridData.gridSize.z - 1 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialInBuffer[clampToGrid(cellIdx + int3(0, 0, 1), gridData.gridSize)].xyz);
    setZNextMaterial(axisCm, zn_axiscm);

    axisCellMaterialOutBuffer[cellIdx].xyz = axisCm;
}

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void applyDiffusion(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cellIdx = ID;

    uint3 axisCm = axisCellMaterialInBuffer[cellIdx].xyz;
    float3 velocity = (float3)0;

    float3 diff = gridDiffusionInBuffer[cellIdx].xyz;
    float3 xp_diff = select(isFluidCell(getXPrevMaterial(axisCm)),
        gridDiffusionInBuffer[clampToGrid(cellIdx + int3(-1, 0, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.x * xp_diff;

    float3 xn_diff = select(isFluidCell(getXNextMaterial(axisCm)),
        gridDiffusionInBuffer[clampToGrid(cellIdx + int3(1, 0, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.x * xn_diff;

    float3 yp_diff = select(isFluidCell(getYPrevMaterial(axisCm)),
        gridDiffusionInBuffer[clampToGrid(cellIdx + int3(0, -1, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.y * yp_diff;

    float3 yn_diff = select(isFluidCell(getYNextMaterial(axisCm)),
        gridDiffusionInBuffer[clampToGrid(cellIdx + int3(0, 1, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.y * yn_diff;

    float3 zp_diff = select(isFluidCell(getZPrevMaterial(axisCm)),
        gridDiffusionInBuffer[clampToGrid(cellIdx + int3(0, 0, -1), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.z * zp_diff;

    float3 zn_diff = select(isFluidCell(getZNextMaterial(axisCm)),
        gridDiffusionInBuffer[clampToGrid(cellIdx + int3(0, 0, 1), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.z * zn_diff;

    float3 sampledVel;
    sampledVel.x = velocityFieldBuffer[0][cellIdx];
    sampledVel.y = velocityFieldBuffer[1][cellIdx];
    sampledVel.z = velocityFieldBuffer[2][cellIdx];
    velocity += pc.diffusionParameters.w * sampledVel;

    enforceBoundaryCondition(velocity, cellMaterialBuffer[cellIdx]);

    gridDiffusionOutBuffer[cellIdx].xyz = velocity;
}

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void updateVelocity(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cellIdx = ID;

    float3 velocity = gridDiffusionInBuffer[cellIdx].xyz;

    enforceBoundaryCondition(velocity, cellMaterialBuffer[cellIdx]);

    velocityFieldBuffer[0][cellIdx] = velocity.x;
    velocityFieldBuffer[1][cellIdx] = velocity.y;
    velocityFieldBuffer[2][cellIdx] = velocity.z;
}
