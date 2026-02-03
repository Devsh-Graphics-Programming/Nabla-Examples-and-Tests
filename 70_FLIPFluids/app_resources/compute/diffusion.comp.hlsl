#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../descriptor_bindings.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

struct SPushConstants // TODO: each Push Constant struct should be HLSL/C++ shared and called `S[DispatchName]PushConstants`
{
    // DOCS DOCS and SEMANTICALLY USEFUL NAMES
    float32_t4 diffusionParameters;
};

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(b_dGridData, s_d)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(b_dCM, s_d)]] RWTexture3D<uint> cellMaterialGrid;
[[vk::binding(b_dVel, s_d)]] RWTexture3D<float> velocityField[3];
// THESE ARE HORRIBLY INEFFICIENT DATA STORAGE (each axis cell is 14 bits, gives 42 for 3 axes, not 128)
// TODO: there's no need for these to be stored at all, can be worked out from `cellMaterial` entirely in shared memory! Data is only needed during diffusion step!
// TODO: investigate using a staggered grid, there's something fishy about each axis needing a full XYZ cell!
[[vk::binding(b_dAxisIn, s_d)]] RWTexture3D<uint4> axisCellMaterialIn;
[[vk::binding(b_dAxisOut, s_d)]] RWTexture3D<uint4> axisCellMaterialOut;
[[vk::binding(b_dDiff, s_d)]] RWTexture3D<float4> gridDiffusion;

// TODO: THOU SHALT NOT USE INTEGER DIVISION AND MODULO IN SHADERS! To stop using `flatIdxToLocalGridID`, the shared arrays need to be flat.
// TODO: `vector<>` types should not be used in `groupshared` because they cause bank conflicts, they should be laid out `SoA` instead (component has largest stride)
groupshared uint16_t3 sAxisCellMat[14][14][14]; // TODO: `uint16_t` per axis is too much
groupshared float16_t3 sDiffusion[14][14][14];

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
[shader("compute")]
void setAxisCellMaterial(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cellIdx = ID;

    uint cellMaterial = cellMaterialGrid[cellIdx];

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

    axisCellMaterialOut[cellIdx].xyz = cmAxisTypes;
}

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
[shader("compute")]
void setNeighborAxisCellMaterial(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cellIdx = ID;

    uint3 axisCm = (uint3)0;
    uint3 this_axiscm = getCellMaterial(axisCellMaterialIn[cellIdx].xyz);
    setCellMaterial(axisCm, this_axiscm);

    uint3 xp_axiscm = cellIdx.x == 0 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialIn[clampToGrid(cellIdx + int3(-1, 0, 0), gridData.gridSize)].xyz);
    setXPrevMaterial(axisCm, xp_axiscm);

    uint3 xn_axiscm = cellIdx.x == gridData.gridSize.x - 1 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialIn[clampToGrid(cellIdx + int3(1, 0, 0), gridData.gridSize)].xyz);
    setXNextMaterial(axisCm, xn_axiscm);

    uint3 yp_axiscm = cellIdx.y == 0 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialIn[clampToGrid(cellIdx + int3(0, -1, 0), gridData.gridSize)].xyz);
    setYPrevMaterial(axisCm, yp_axiscm);

    uint3 yn_axiscm = cellIdx.y == gridData.gridSize.y - 1 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialIn[clampToGrid(cellIdx + int3(0, 1, 0), gridData.gridSize)].xyz);
    setYNextMaterial(axisCm, yn_axiscm);

    uint3 zp_axiscm = cellIdx.z == 0 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialIn[clampToGrid(cellIdx + int3(0, 0, -1), gridData.gridSize)].xyz);
    setZPrevMaterial(axisCm, zp_axiscm);

    uint3 zn_axiscm = cellIdx.z == gridData.gridSize.z - 1 ? (uint3)CM_SOLID : getCellMaterial(axisCellMaterialIn[clampToGrid(cellIdx + int3(0, 0, 1), gridData.gridSize)].xyz);
    setZNextMaterial(axisCm, zn_axiscm);

    axisCellMaterialOut[cellIdx].xyz = axisCm;
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
[shader("compute")]
void iterateDiffusion(uint32_t3 ID : SV_DispatchThreadID)
{
    uint3 gid = nbl::hlsl::glsl::gl_WorkGroupID();
    int3 cellIdx = ID;

    float3 sampledVel;
    sampledVel.x = velocityField[0][cellIdx];
    sampledVel.y = velocityField[1][cellIdx];
    sampledVel.z = velocityField[2][cellIdx];

    uint cellMaterial = cellMaterialGrid[cellIdx];

    // load shared mem
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 14 * 14 * 14; virtualIdx += 8 * 8 * 8)
    {
        // TODO: THOU SHALT NOT USE INTEGER DIVISION AND MODULO IN SHADERS! STOP USING `flatIdxToLocalGridID`
        int3 lid = flatIdxToLocalGridID(virtualIdx, 14);
        
        int3 cellIdx = clampToGrid(lid + int3(-3, -3, -3) + gid * WorkgroupGridDim, gridData.gridSize);
        sAxisCellMat[lid.x][lid.y][lid.z] = uint16_t3(axisCellMaterialIn[cellIdx].xyz);
        sDiffusion[lid.x][lid.y][lid.z] = float16_t3(gridDiffusion[cellIdx].xyz);
    }
    GroupMemoryBarrierWithGroupSync();

    // TODO: undo the unroll, use two nested for loops

    // do 12x12x12 iteration
    float3 tmp[6];
    uint i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 12 * 12 * 12; virtualIdx += 8 * 8 * 8)
    {
        // TODO: THOU SHALT NOT USE INTEGER DIVISION AND MODULO IN SHADERS! STOP USING `flatIdxToLocalGridID`
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
    gridDiffusion[cellIdx].xyz = velocity;
}

// TODO: same as the pressure solver, this kernel/dispatch should be fused onto `iterateDiffusion` guarded by `isLastIteration` push constant
[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
[shader("compute")]
void applyDiffusion(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cellIdx = ID;

    uint3 axisCm = axisCellMaterialIn[cellIdx].xyz;
    float3 velocity = (float3)0;

    float3 diff = gridDiffusion[cellIdx].xyz;
    float3 xp_diff = select(isFluidCell(getXPrevMaterial(axisCm)),
        gridDiffusion[clampToGrid(cellIdx + int3(-1, 0, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.x * xp_diff;

    float3 xn_diff = select(isFluidCell(getXNextMaterial(axisCm)),
        gridDiffusion[clampToGrid(cellIdx + int3(1, 0, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.x * xn_diff;

    float3 yp_diff = select(isFluidCell(getYPrevMaterial(axisCm)),
        gridDiffusion[clampToGrid(cellIdx + int3(0, -1, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.y * yp_diff;

    float3 yn_diff = select(isFluidCell(getYNextMaterial(axisCm)),
        gridDiffusion[clampToGrid(cellIdx + int3(0, 1, 0), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.y * yn_diff;

    float3 zp_diff = select(isFluidCell(getZPrevMaterial(axisCm)),
        gridDiffusion[clampToGrid(cellIdx + int3(0, 0, -1), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.z * zp_diff;

    float3 zn_diff = select(isFluidCell(getZNextMaterial(axisCm)),
        gridDiffusion[clampToGrid(cellIdx + int3(0, 0, 1), gridData.gridSize)].xyz, diff);
    velocity += pc.diffusionParameters.z * zn_diff;

    float3 sampledVel;
    sampledVel.x = velocityField[0][cellIdx];
    sampledVel.y = velocityField[1][cellIdx];
    sampledVel.z = velocityField[2][cellIdx];
    velocity += pc.diffusionParameters.w * sampledVel;

    enforceBoundaryCondition(velocity, cellMaterialGrid[cellIdx]);

    velocityField[0][cellIdx] = velocity.x;
    velocityField[1][cellIdx] = velocity.y;
    velocityField[2][cellIdx] = velocity.z;
}
