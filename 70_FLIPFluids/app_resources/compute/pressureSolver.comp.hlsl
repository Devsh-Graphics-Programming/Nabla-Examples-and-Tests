#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../descriptor_bindings.hlsl"

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

[numthreads(WorkgroupSize, 1, 1)]
void calculateNegativeDivergence(uint32_t3 ID : SV_DispatchThreadID)
{
    uint cid = ID.x;
    int3 cellIdx = flatIdxToCellIdx(cid, gridData.gridSize);

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

[numthreads(WorkgroupSize, 1, 1)]
void solvePressureSystem(uint32_t3 ID : SV_DispatchThreadID)
{
    uint cid = ID.x;
    int3 cellIdx = flatIdxToCellIdx(cid, gridData.gridSize);

    float pressure = 0;

    uint cellMaterial = cellMaterialBuffer[cellIdx];

    if (isFluidCell(getCellMaterial(cellMaterial)))
    {
        int3 cell_xp = clampToGrid(cellIdx + int3(-1, 0, 0), gridData.gridSize);
        cell_xp = isSolidCell(getXPrevMaterial(cellMaterial)) ? cellIdx : cell_xp;
        int3 cell_xn = clampToGrid(cellIdx + int3(1, 0, 0), gridData.gridSize);
        cell_xn = isSolidCell(getXNextMaterial(cellMaterial)) ? cellIdx : cell_xn;
        pressure += params.coeff1.x * (pressureInBuffer[cell_xp] + pressureInBuffer[cell_xn]);

        int3 cell_yp = clampToGrid(cellIdx + int3(0, -1, 0), gridData.gridSize);
        cell_yp = isSolidCell(getYPrevMaterial(cellMaterial)) ? cellIdx : cell_yp;
        int3 cell_yn = clampToGrid(cellIdx + int3(0, 1, 0), gridData.gridSize);
        cell_yn = isSolidCell(getYNextMaterial(cellMaterial)) ? cellIdx : cell_yn;
        pressure += params.coeff1.y * (pressureInBuffer[cell_yp] + pressureInBuffer[cell_yn]);

        int3 cell_zp = clampToGrid(cellIdx + int3(0, 0, -1), gridData.gridSize);
        cell_zp = isSolidCell(getZPrevMaterial(cellMaterial)) ? cellIdx : cell_zp;
        int3 cell_zn = clampToGrid(cellIdx + int3(0, 0, 1), gridData.gridSize);
        cell_zn = isSolidCell(getZNextMaterial(cellMaterial)) ? cellIdx : cell_zn;
        pressure += params.coeff1.z * (pressureInBuffer[cell_zp] + pressureInBuffer[cell_zn]);

        pressure += params.coeff1.w * divergenceBuffer[cellIdx];
    }

    pressureOutBuffer[cellIdx] = pressure;
}

[numthreads(WorkgroupSize, 1, 1)]
void updateVelocities(uint32_t3 ID : SV_DispatchThreadID)
{
    uint cid = ID.x;
    int3 cellIdx = flatIdxToCellIdx(cid, gridData.gridSize);

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
