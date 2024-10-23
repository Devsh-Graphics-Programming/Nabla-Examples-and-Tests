#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../descriptor_bindings.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

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
[[vk::binding(b_psPresBuffer, s_ps)]] RWTexture3D<float> pressureBuffer;

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

int3 flatIdxToLocalGridID(uint idx, int size)
{
    uint a = 14 * 14;
    int3 b;
    b.z = idx / a;
    b.x = idx - b.z * a;
    b.y = b.x / 14;
    b.x = b.x - b.y * 14;
    return b;
}

float calculatePressureStep(int3 idx)
{
    float pressure = 0.0f;
    uint cellMaterial = sCellMat[idx.x][idx.y][idx.z];

    if (isFluidCell(getCellMaterial(cellMaterial)))
    {
        int3 xp = isSolidCell(getXPrevMaterial(cellMaterial)) ? idx : idx + int3(-1, 0, 0);
        int3 xn = isSolidCell(getXNextMaterial(cellMaterial)) ? idx : idx + int3(1, 0, 0);
        pressure += params.coeff1.x * (sPressure[xp.x][xp.y][xp.z] + sPressure[xn.x][xn.y][xn.z]);

        int3 yp = isSolidCell(getYPrevMaterial(cellMaterial)) ? idx : idx + int3(0, -1, 0);
        int3 yn = isSolidCell(getYNextMaterial(cellMaterial)) ? idx : idx + int3(0, 1, 0);
        pressure += params.coeff1.y * (sPressure[yp.x][yp.y][yp.z] + sPressure[yn.x][yn.y][yn.z]);

        int3 zp = isSolidCell(getZPrevMaterial(cellMaterial)) ? idx : idx + int3(0, 0, -1);
        int3 zn = isSolidCell(getZNextMaterial(cellMaterial)) ? idx : idx + int3(0, 0, 1);
        pressure += params.coeff1.z * (sPressure[zp.x][zp.y][zp.z] + sPressure[zn.x][zn.y][zn.z]);

        pressure += params.coeff1.w * sDivergence[idx.x][idx.y][idx.z];
    }

    return pressure;
}

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void iteratePressureSystem(uint32_t3 ID : SV_DispatchThreadID)
{
    uint3 gid = nbl::hlsl::glsl::gl_WorkGroupID();

    // load shared mem
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 14 * 14 * 14; virtualIdx += 8 * 8 * 8)
    {
        int3 lid = flatIdxToLocalGridID(virtualIdx, 14);
        
        int3 cellIdx = clampToGrid(lid + int3(-3, -3, -3) + gid * WorkgroupGridDim, gridData.gridSize);
        sCellMat[lid.x][lid.y][lid.z] = cellMaterialBuffer[cellIdx];
        sDivergence[lid.x][lid.y][lid.z] = divergenceBuffer[cellIdx];
        sPressure[lid.x][lid.y][lid.z] = pressureBuffer[cellIdx];
    }
    GroupMemoryBarrierWithGroupSync();

    // do 12x12x12 iteration
    float tmp[6];
    uint i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 12 * 12 * 12; virtualIdx += 8 * 8 * 8)
    {
        int3 lid = flatIdxToLocalGridID(virtualIdx, 12);
        lid += int3(1, 1, 1);

        float pressure = calculatePressureStep(lid);

        tmp[i++] = pressure;
    }
    GroupMemoryBarrierWithGroupSync();

    i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 12 * 12 * 12; virtualIdx += 8 * 8 * 8)
    {
        int3 lid = flatIdxToLocalGridID(virtualIdx, 12);
        lid += int3(1, 1, 1);
        sPressure[lid.x][lid.y][lid.z] = tmp[i++];
    }
    GroupMemoryBarrierWithGroupSync();

    // do 10x10x10 iteration
    i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 10 * 10 * 10; virtualIdx += 8 * 8 * 8)
    {
        int3 lid = flatIdxToLocalGridID(virtualIdx, 10);
        lid += int3(2, 2, 2);

        float pressure = calculatePressureStep(lid);

        tmp[i++] = pressure;
    }
    GroupMemoryBarrierWithGroupSync();

    i = 0;
    for (uint virtualIdx = nbl::hlsl::glsl::gl_LocalInvocationIndex();
        virtualIdx < 10 * 10 * 10; virtualIdx += 8 * 8 * 8)
    {
        int3 lid = flatIdxToLocalGridID(virtualIdx, 10);
        lid += int3(2, 2, 2);
        sPressure[lid.x][lid.y][lid.z] = tmp[i++];
    }
    GroupMemoryBarrierWithGroupSync();

    // do 8x8x8 iteration (final) and write
    int3 lid = nbl::hlsl::glsl::gl_LocalInvocationID();
    lid += int3(3, 3, 3);

    float pressure = calculatePressureStep(lid);
    pressureBuffer[ID] = pressure;
}

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void updateVelocities(uint32_t3 ID : SV_DispatchThreadID)
{
    int3 cellIdx = ID;

    uint cellMaterial = cellMaterialBuffer[cellIdx];

    float3 velocity;
    velocity.x = velocityFieldBuffer[0][cellIdx];
    velocity.y = velocityFieldBuffer[1][cellIdx];
    velocity.z = velocityFieldBuffer[2][cellIdx];
    float pressure = pressureBuffer[cellIdx];

    int3 cell_xp = clampToGrid(cellIdx + int3(-1, 0, 0), gridData.gridSize);
    cell_xp = isSolidCell(getXPrevMaterial(cellMaterial)) ? cellIdx : cell_xp;
    velocity.x -= params.coeff2.x * (pressure - pressureBuffer[cell_xp]);

    int3 cell_yp = clampToGrid(cellIdx + int3(0, -1, 0), gridData.gridSize);
    cell_yp = isSolidCell(getYPrevMaterial(cellMaterial)) ? cellIdx : cell_yp;
    velocity.y -= params.coeff2.y * (pressure - pressureBuffer[cell_yp]);

    int3 cell_zp = clampToGrid(cellIdx + int3(0, 0, -1), gridData.gridSize);
    cell_zp = isSolidCell(getZPrevMaterial(cellMaterial)) ? cellIdx : cell_zp;
    velocity.z -= params.coeff2.z * (pressure - pressureBuffer[cell_zp]);

    enforceBoundaryCondition(velocity, cellMaterial);

    velocityFieldBuffer[0][cellIdx] = velocity.x;
    velocityFieldBuffer[1][cellIdx] = velocity.y;
    velocityFieldBuffer[2][cellIdx] = velocity.z;
}
