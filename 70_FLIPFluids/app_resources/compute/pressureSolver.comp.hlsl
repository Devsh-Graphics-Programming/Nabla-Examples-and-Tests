#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../descriptor_bindings.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_arithmetic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_ballot.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

struct SPressureSolverParams
{
    float4 coeff1;
    float4 coeff2;
};

[[vk::binding(b_psGridData, s_ps)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(b_psParams, s_ps)]]
cbuffer PressureSolverParams
{
    SPressureSolverParams params;
};

[[vk::binding(b_psCMBuffer, s_ps)]] RWTexture3D<uint> cellMaterialBuffer;
[[vk::binding(b_psVelBuffer, s_ps)]] RWTexture3D<float> velocityFieldBuffer[3];
[[vk::binding(b_psDivBuffer, s_ps)]] RWTexture3D<float> divergenceBuffer;
[[vk::binding(b_psPresInBuffer, s_ps)]] RWTexture3D<float> pressureInBuffer;
[[vk::binding(b_psPresOutBuffer, s_ps)]] RWTexture3D<float> pressureOutBuffer;

groupshared uint sCellMat[14][14][14];
groupshared float sDivergence[14][14][14];
groupshared float sPressure[14][14][14];

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void calculateNegativeDivergence(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cellIdx = ID;

    float3 param = (float3)gridData.gridInvCellSize;
    float3 velocity;
    velocity.x = velocityFieldBuffer[0][cellIdx];
    velocity.y = velocityFieldBuffer[1][cellIdx];
    velocity.z = velocityFieldBuffer[2][cellIdx];

    float divergence = 0;
    if (isFluidCell(getCellMaterial(cellMaterialBuffer[cellIdx])))
    {
        int3 cell_xn = cellIdx + int3(1, 0, 0);
        divergence += param.x * ((cell_xn.x < gridData.gridSize.x ? velocityFieldBuffer[0][cell_xn] : 0.0f) - velocity.x);

        int3 cell_yn = cellIdx + int3(0, 1, 0);
        divergence += param.y * ((cell_yn.y < gridData.gridSize.y ? velocityFieldBuffer[1][cell_yn] : 0.0f) - velocity.y);

        int3 cell_zn = cellIdx + int3(0, 0, 1);
        divergence += param.z * ((cell_zn.z < gridData.gridSize.z ? velocityFieldBuffer[2][cell_zn] : 0.0f) - velocity.z);
    }

    divergenceBuffer[cellIdx] = divergence;
}

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void iteratePressureSystem(uint32_t3 ID : SV_DispatchThreadID)
{
    uint3 gid = nbl::hlsl::glsl::gl_WorkGroupID();

    // load shared mem
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 14 * 14 * 14; virtualIdx += 8 * 8 * 8)
    {
        uint a = 14 * 14;
        int3 lid;
        lid.z = virtualIdx / a;
        lid.x = virtualIdx - lid.z * a;
        lid.y = lid.x / 14;
        lid.x = lid.x - lid.y * 14;

        int3 cellIdx = clampToGrid(lid - int3(-3, -3, -3) + gid * WorkGroupGridDim, gridData.gridSize);
        sCellMat[lid.x][lid.y][lid.z] = cellMaterialBuffer[cellIdx];
        sDivergence[lid.x][lid.y][lid.z] = divergenceBuffer[cellIdx];
        sPressure[lid.x][lid.y][lid.z] = pressureInBuffer[cellIdx];
    }
    GroupMemoryBarrierWithGroupSync();

    // do 12x12x12 iteration
    float tmp[6];
    uint i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 12 * 12 * 12; virtualIdx += 8 * 8 * 8)
    {
        uint a = 12 * 12;
        int3 lid;
        lid.z = virtualIdx / a;
        lid.x = virtualIdx - lid.z * a;
        lid.y = lid.x / 12;
        lid.x = lid.x - lid.y * 12;
        lid += int3(1, 1, 1);

        float pressure = 0;
        uint cellMaterial = sCellMat[lid.x][lid.y][lid.z];

        if (isFluidCell(getCellMaterial(cellMaterial)))
        {
            int3 xp = isSolidCell(getXPrevMaterial(cellMaterial)) ? lid : lid + int3(-1, 0, 0);
            int3 xn = isSolidCell(getXNextMaterial(cellMaterial)) ? lid : lid + int3(1, 0, 0);
            pressure += params.coeff1.x * (sPressure[xp.x][xp.y][xp.z] + sPressure[xn.x][xn.y][xn.z]);

            int3 yp = isSolidCell(getYPrevMaterial(cellMaterial)) ? lid : lid + int3(0, -1, 0);
            int3 yn = isSolidCell(getYNextMaterial(cellMaterial)) ? lid : lid + int3(0, 1, 0);
            pressure += params.coeff1.y * (sPressure[yp.x][yp.y][yp.z] + sPressure[yn.x][yn.y][yn.z]);

            int3 zp = isSolidCell(getZPrevMaterial(cellMaterial)) ? lid : lid + int3(0, 0, -1);
            int3 zn = isSolidCell(getZNextMaterial(cellMaterial)) ? lid : lid + int3(0, 0, 1);
            pressure += params.coeff1.z * (sPressure[zp.x][zp.y][zp.z] + sPressure[zn.x][zn.y][zn.z]);

            pressure += params.coeff1.w * sDivergence[lid.x][lid.y][lid.z];
        }

        tmp[i++] = pressure;
    }
    GroupMemoryBarrierWithGroupSync();

    i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 12 * 12 * 12; virtualIdx += 8 * 8 * 8)
    {
        sPressure[lid.x][lid.y][lid.z] = tmp[i++];
    }
    GroupMemoryBarrierWithGroupSync();

    // do 10x10x10 iteration
    i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 10 * 10 * 10; virtualIdx += 8 * 8 * 8)
    {
        uint a = 10 * 10;
        int3 lid;
        lid.z = virtualIdx / a;
        lid.x = virtualIdx - lid.z * a;
        lid.y = lid.x / 10;
        lid.x = lid.x - lid.y * 10;
        lid += int3(2, 2, 2);

        float pressure = 0;
        uint cellMaterial = sCellMat[lid.x][lid.y][lid.z];

        if (isFluidCell(getCellMaterial(cellMaterial)))
        {
            int3 xp = isSolidCell(getXPrevMaterial(cellMaterial)) ? lid : lid + int3(-1, 0, 0);
            int3 xn = isSolidCell(getXNextMaterial(cellMaterial)) ? lid : lid + int3(1, 0, 0);
            pressure += params.coeff1.x * (sPressure[xp.x][xp.y][xp.z] + sPressure[xn.x][xn.y][xn.z]);

            int3 yp = isSolidCell(getYPrevMaterial(cellMaterial)) ? lid : lid + int3(0, -1, 0);
            int3 yn = isSolidCell(getYNextMaterial(cellMaterial)) ? lid : lid + int3(0, 1, 0);
            pressure += params.coeff1.y * (sPressure[yp.x][yp.y][yp.z] + sPressure[yn.x][yn.y][yn.z]);

            int3 zp = isSolidCell(getZPrevMaterial(cellMaterial)) ? lid : lid + int3(0, 0, -1);
            int3 zn = isSolidCell(getZNextMaterial(cellMaterial)) ? lid : lid + int3(0, 0, 1);
            pressure += params.coeff1.z * (sPressure[zp.x][zp.y][zp.z] + sPressure[zn.x][zn.y][zn.z]);

            pressure += params.coeff1.w * sDivergence[lid.x][lid.y][lid.z];
        }

        tmp[i++] = pressure;
    }
    GroupMemoryBarrierWithGroupSync();

    i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 10 * 10 * 10; virtualIdx += 8 * 8 * 8)
    {
        sPressure[lid.x][lid.y][lid.z] = tmp[i++];
    }
    GroupMemoryBarrierWithGroupSync();

    // do 8x8x8 iteration
    i = 0;
    {
        int3 lid = nbl::hlsl::glsl::gl_LocalInvocationID();
        lid += int3(3, 3, 3);

        float pressure = 0;
        uint cellMaterial = sCellMat[lid.x][lid.y][lid.z];

        if (isFluidCell(getCellMaterial(cellMaterial)))
        {
            int3 xp = isSolidCell(getXPrevMaterial(cellMaterial)) ? lid : lid + int3(-1, 0, 0);
            int3 xn = isSolidCell(getXNextMaterial(cellMaterial)) ? lid : lid + int3(1, 0, 0);
            pressure += params.coeff1.x * (sPressure[xp.x][xp.y][xp.z] + sPressure[xn.x][xn.y][xn.z]);

            int3 yp = isSolidCell(getYPrevMaterial(cellMaterial)) ? lid : lid + int3(0, -1, 0);
            int3 yn = isSolidCell(getYNextMaterial(cellMaterial)) ? lid : lid + int3(0, 1, 0);
            pressure += params.coeff1.y * (sPressure[yp.x][yp.y][yp.z] + sPressure[yn.x][yn.y][yn.z]);

            int3 zp = isSolidCell(getZPrevMaterial(cellMaterial)) ? lid : lid + int3(0, 0, -1);
            int3 zn = isSolidCell(getZNextMaterial(cellMaterial)) ? lid : lid + int3(0, 0, 1);
            pressure += params.coeff1.z * (sPressure[zp.x][zp.y][zp.z] + sPressure[zn.x][zn.y][zn.z]);

            pressure += params.coeff1.w * sDivergence[lid.x][lid.y][lid.z];
        }

        tmp[i++] = pressure;
    }
    GroupMemoryBarrierWithGroupSync();  // probably don't need to sync here

    // write to buffer
    int3 lid = nbl::hlsl::glsl::gl_LocalInvocationID();
    int3 cellIdx = clampToGrid(lid + gid * WorkGroupGridDim, gridData.gridSize);
    pressureInBuffer[cellIdx] = tmp[0];
}

// [numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
// void solvePressureSystem(uint32_t3 ID : SV_DispatchThreadID)
// {
//     int3 cellIdx = ID;

//     float pressure = 0;

//     uint cellMaterial = cellMaterialBuffer[cellIdx];

//     if (isFluidCell(getCellMaterial(cellMaterial)))
//     {
//         int3 cell_xp = clampToGrid(cellIdx + int3(-1, 0, 0), gridData.gridSize);
//         cell_xp = isSolidCell(getXPrevMaterial(cellMaterial)) ? cellIdx : cell_xp;
//         int3 cell_xn = clampToGrid(cellIdx + int3(1, 0, 0), gridData.gridSize);
//         cell_xn = isSolidCell(getXNextMaterial(cellMaterial)) ? cellIdx : cell_xn;
//         pressure += params.coeff1.x * (pressureInBuffer[cell_xp] + pressureInBuffer[cell_xn]);

//         int3 cell_yp = clampToGrid(cellIdx + int3(0, -1, 0), gridData.gridSize);
//         cell_yp = isSolidCell(getYPrevMaterial(cellMaterial)) ? cellIdx : cell_yp;
//         int3 cell_yn = clampToGrid(cellIdx + int3(0, 1, 0), gridData.gridSize);
//         cell_yn = isSolidCell(getYNextMaterial(cellMaterial)) ? cellIdx : cell_yn;
//         pressure += params.coeff1.y * (pressureInBuffer[cell_yp] + pressureInBuffer[cell_yn]);

//         int3 cell_zp = clampToGrid(cellIdx + int3(0, 0, -1), gridData.gridSize);
//         cell_zp = isSolidCell(getZPrevMaterial(cellMaterial)) ? cellIdx : cell_zp;
//         int3 cell_zn = clampToGrid(cellIdx + int3(0, 0, 1), gridData.gridSize);
//         cell_zn = isSolidCell(getZNextMaterial(cellMaterial)) ? cellIdx : cell_zn;
//         pressure += params.coeff1.z * (pressureInBuffer[cell_zp] + pressureInBuffer[cell_zn]);

//         pressure += params.coeff1.w * divergenceBuffer[cellIdx];
//     }

//     pressureOutBuffer[cellIdx] = pressure;
// }

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void updateVelocities(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cellIdx = ID;

    uint cellMaterial = cellMaterialBuffer[cellIdx];

    float3 velocity;
    velocity.x = velocityFieldBuffer[0][cellIdx];
    velocity.y = velocityFieldBuffer[1][cellIdx];
    velocity.z = velocityFieldBuffer[2][cellIdx];
    float pressure = pressureInBuffer[cellIdx];

    int3 cell_xp = clampToGrid(cellIdx + int3(-1, 0, 0), gridData.gridSize);
    cell_xp = isSolidCell(getXPrevMaterial(cellMaterial)) ? cellIdx : cell_xp;
    velocity.x -= params.coeff2.x * (pressure - pressureInBuffer[cell_xp]);

    int3 cell_yp = clampToGrid(cellIdx + int3(0, -1, 0), gridData.gridSize);
    cell_yp = isSolidCell(getYPrevMaterial(cellMaterial)) ? cellIdx : cell_yp;
    velocity.y -= params.coeff2.y * (pressure - pressureInBuffer[cell_yp]);

    int3 cell_zp = clampToGrid(cellIdx + int3(0, 0, -1), gridData.gridSize);
    cell_zp = isSolidCell(getZPrevMaterial(cellMaterial)) ? cellIdx : cell_zp;
    velocity.z -= params.coeff2.z * (pressure - pressureInBuffer[cell_zp]);

    enforceBoundaryCondition(velocity, cellMaterial);

    velocityFieldBuffer[0][cellIdx] = velocity.x;
    velocityFieldBuffer[1][cellIdx] = velocity.y;
    velocityFieldBuffer[2][cellIdx] = velocity.z;
}
