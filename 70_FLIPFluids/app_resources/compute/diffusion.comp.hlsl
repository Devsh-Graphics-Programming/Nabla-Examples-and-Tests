#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../descriptor_bindings.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

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
[[vk::binding(b_dDiffBuffer, s_d)]] RWTexture3D<float4> gridDiffusionBuffer;

groupshared uint16_t3 sAxisCellMat[14][14][14];
groupshared float16_t3 sDiffusion[14][14][14];

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

float3 calculateDiffusionVelStep(int3 idx, float3 sampledVelocity, uint cellMaterial)
{
    float3 velocity = (float3)0;
    uint3 axisCm = uint3(sAxisCellMat[idx.x][idx.y][idx.z]);

    float3 diff = float3(sDiffusion[idx.x][idx.y][idx.z]);
    float3 xp = select(isFluidCell(getXPrevMaterial(axisCm)), float3(sDiffusion[idx.x - 1][idx.y][idx.z]), diff);
    velocity += pc.diffusionParameters.x * xp;

    float3 xn = select(isFluidCell(getXNextMaterial(axisCm)), float3(sDiffusion[idx.x + 1][idx.y][idx.z]), diff);
    velocity += pc.diffusionParameters.x * xn;

    float3 yp = select(isFluidCell(getYPrevMaterial(axisCm)), float3(sDiffusion[idx.x][idx.y - 1][idx.z]), diff);
    velocity += pc.diffusionParameters.y * yp;

    float3 yn = select(isFluidCell(getYNextMaterial(axisCm)), float3(sDiffusion[idx.x][idx.y + 1][idx.z]), diff);
    velocity += pc.diffusionParameters.y * yn;

    float3 zp = select(isFluidCell(getZPrevMaterial(axisCm)), float3(sDiffusion[idx.x][idx.y][idx.z - 1]), diff);
    velocity += pc.diffusionParameters.z * zp;

    float3 zn = select(isFluidCell(getZNextMaterial(axisCm)), float3(sDiffusion[idx.x][idx.y][idx.z + 1]), diff);
    velocity += pc.diffusionParameters.z * zn;

    velocity += pc.diffusionParameters.w * sampledVelocity;
    enforceBoundaryCondition(velocity, cellMaterial);

    return velocity;
}

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void iterateDiffusion(uint32_t3 ID : SV_DispatchThreadID)
{
    uint3 gid = nbl::hlsl::glsl::gl_WorkGroupID();
    int3 cellIdx = ID;

    float3 sampledVel;
    sampledVel.x = velocityFieldBuffer[0][cellIdx];
    sampledVel.y = velocityFieldBuffer[1][cellIdx];
    sampledVel.z = velocityFieldBuffer[2][cellIdx];

    uint cellMaterial = cellMaterialBuffer[cellIdx];

    // load shared mem
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 14 * 14 * 14; virtualIdx += 8 * 8 * 8)
    {
        int3 lid = flatIdxToLocalGridID(virtualIdx, 14);
        
        int3 cellIdx = clampToGrid(lid + int3(-3, -3, -3) + gid * WorkgroupGridDim, gridData.gridSize);
        sAxisCellMat[lid.x][lid.y][lid.z] = uint16_t3(axisCellMaterialInBuffer[cellIdx].xyz);
        sDiffusion[lid.x][lid.y][lid.z] = float16_t3(gridDiffusionBuffer[cellIdx].xyz);
    }
    GroupMemoryBarrierWithGroupSync();

    // do 12x12x12 iteration
    float3 tmp[6];
    uint i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 12 * 12 * 12; virtualIdx += 8 * 8 * 8)
    {
        int3 lid = flatIdxToLocalGridID(virtualIdx, 12);
        lid += int3(1, 1, 1);
        
        float3 velocity = calculateDiffusionVelStep(lid, sampledVel, cellMaterial);
        tmp[i++] = velocity;
    }
    GroupMemoryBarrierWithGroupSync();

    i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 12 * 12 * 12; virtualIdx += 8 * 8 * 8)
    {
        int3 lid = flatIdxToLocalGridID(virtualIdx, 12);
        lid += int3(1, 1, 1);
        sDiffusion[lid.x][lid.y][lid.z] = float16_t3(tmp[i++]);
    }
    GroupMemoryBarrierWithGroupSync();

    // do 10x10x10 iteration
    i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 10 * 10 * 10; virtualIdx += 8 * 8 * 8)
    {
        int3 lid = flatIdxToLocalGridID(virtualIdx, 10);
        lid += int3(2, 2, 2);
        
        float3 velocity = calculateDiffusionVelStep(lid, sampledVel, cellMaterial);
        tmp[i++] = velocity;
    }
    GroupMemoryBarrierWithGroupSync();

    i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 10 * 10 * 10; virtualIdx += 8 * 8 * 8)
    {
        int3 lid = flatIdxToLocalGridID(virtualIdx, 10);
        lid += int3(2, 2, 2);
        sDiffusion[lid.x][lid.y][lid.z] = float16_t3(tmp[i++]);
    }
    GroupMemoryBarrierWithGroupSync();

    // do 8x8x8 iteration (final) and write
    int3 lid = nbl::hlsl::glsl::gl_LocalInvocationID();
    lid += int3(3, 3, 3);

    float3 velocity = calculateDiffusionVelStep(lid, sampledVel, cellMaterial);
    gridDiffusionBuffer[cellIdx].xyz = velocity;
}

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void applyDiffusion(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cellIdx = ID;

    uint3 axisCm = axisCellMaterialInBuffer[cellIdx].xyz;
    float3 velocity = (float3)0;

    float3 diff = gridDiffusionBuffer[cellIdx].xyz;
    float3 xp_diff = select(isFluidCell(getXPrevMaterial(axisCm)),
        gridDiffusionBuffer[clampToGrid(cellIdx + int3(-1, 0, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.x * xp_diff;

    float3 xn_diff = select(isFluidCell(getXNextMaterial(axisCm)),
        gridDiffusionBuffer[clampToGrid(cellIdx + int3(1, 0, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.x * xn_diff;

    float3 yp_diff = select(isFluidCell(getYPrevMaterial(axisCm)),
        gridDiffusionBuffer[clampToGrid(cellIdx + int3(0, -1, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.y * yp_diff;

    float3 yn_diff = select(isFluidCell(getYNextMaterial(axisCm)),
        gridDiffusionBuffer[clampToGrid(cellIdx + int3(0, 1, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.y * yn_diff;

    float3 zp_diff = select(isFluidCell(getZPrevMaterial(axisCm)),
        gridDiffusionBuffer[clampToGrid(cellIdx + int3(0, 0, -1), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.z * zp_diff;

    float3 zn_diff = select(isFluidCell(getZNextMaterial(axisCm)),
        gridDiffusionBuffer[clampToGrid(cellIdx + int3(0, 0, 1), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.z * zn_diff;

    float3 sampledVel;
    sampledVel.x = velocityFieldBuffer[0][cellIdx];
    sampledVel.y = velocityFieldBuffer[1][cellIdx];
    sampledVel.z = velocityFieldBuffer[2][cellIdx];
    velocity += pc.diffusionParameters.w * sampledVel;

    enforceBoundaryCondition(velocity, cellMaterialBuffer[cellIdx]);

    velocityFieldBuffer[0][cellIdx] = velocity.x;
    velocityFieldBuffer[1][cellIdx] = velocity.y;
    velocityFieldBuffer[2][cellIdx] = velocity.z;
}
