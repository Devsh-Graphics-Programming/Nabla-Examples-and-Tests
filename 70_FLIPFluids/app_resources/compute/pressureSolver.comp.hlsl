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

[[vk::binding(b_psCMBuffer, s_ps)]] RWStructuredBuffer<uint> cellMaterialBuffer;
[[vk::binding(b_psVelBuffer, s_ps)]] RWStructuredBuffer<float4> velocityFieldBuffer;
[[vk::binding(b_psDivBuffer, s_ps)]] RWStructuredBuffer<float> divergenceBuffer;
[[vk::binding(b_psPresInBuffer, s_ps)]] RWStructuredBuffer<float> pressureInBuffer;
[[vk::binding(b_psPresOutBuffer, s_ps)]] RWStructuredBuffer<float> pressureOutBuffer;

[numthreads(WorkgroupSize, 1, 1)]
void calculateNegativeDivergence(uint32_t3 ID : SV_DispatchThreadID)
{
    uint cid = ID.x;
    int3 cellIdx = flatIdxToCellIdx(cid, gridData.gridSize);

    float3 param = (float3)gridData.gridInvCellSize;
    float3 velocity = velocityFieldBuffer[cid].xyz;

    float divergence = 0;
    if (isFluidCell(getCellMaterial(cellMaterialBuffer[cid])))
    {
        int3 cell_xn = cellIdx + int3(1, 0, 0);
        uint cid_xn = cellIdxToFlatIdx(cell_xn, gridData.gridSize);
        divergence += param.x * ((cell_xn.x < gridData.gridSize.x ? velocityFieldBuffer[cid_xn].x : 0.0f) - velocity.x);

        int3 cell_yn = cellIdx + int3(0, 1, 0);
        uint cid_yn = cellIdxToFlatIdx(cell_yn, gridData.gridSize);
        divergence += param.y * ((cell_yn.y < gridData.gridSize.y ? velocityFieldBuffer[cid_yn].y : 0.0f) - velocity.y);

        int3 cell_zn = cellIdx + int3(0, 0, 1);
        uint cid_zn = cellIdxToFlatIdx(cell_zn, gridData.gridSize);
        divergence += param.z * ((cell_zn.z < gridData.gridSize.z ? velocityFieldBuffer[cid_zn].z : 0.0f) - velocity.z);
    }

    divergenceBuffer[cid] = divergence;
}

[numthreads(WorkgroupSize, 1, 1)]
void solvePressureSystem(uint32_t3 ID : SV_DispatchThreadID)
{
    uint cid = ID.x;
    int3 cellIdx = flatIdxToCellIdx(cid, gridData.gridSize);

    float pressure = 0;

    uint cellMaterial = cellMaterialBuffer[cid];

    if (isFluidCell(cellMaterial))
    {
        uint cid_xp = cellIdxToFlatIdx(cellIdx + int3(-1, 0, 0), gridData.gridSize);
        cid_xp = isSolidCell(getXPrevMaterial(cellMaterial)) ? cid : cid_xp;
        uint cid_xn = cellIdxToFlatIdx(cellIdx + int3(1, 0, 0), gridData.gridSize);
        cid_xn = isSolidCell(getXNextMaterial(cellMaterial)) ? cid : cid_xn;
        pressure += params.coeff1.x * (pressureInBuffer[cid_xp] + pressureInBuffer[cid_xn]);

        uint cid_yp = cellIdxToFlatIdx(cellIdx + int3(0, -1, 0), gridData.gridSize);
        cid_yp = isSolidCell(getYPrevMaterial(cellMaterial)) ? cid : cid_yp;
        uint cid_yn = cellIdxToFlatIdx(cellIdx + int3(0, 1, 0), gridData.gridSize);
        cid_yn = isSolidCell(getYNextMaterial(cellMaterial)) ? cid : cid_yn;
        pressure += params.coeff1.y * (pressureInBuffer[cid_yp] + pressureInBuffer[cid_yn]);

        uint cid_zp = cellIdxToFlatIdx(cellIdx + int3(0, 0, -1), gridData.gridSize);
        cid_zp = isSolidCell(getZPrevMaterial(cellMaterial)) ? cid : cid_zp;
        uint cid_zn = cellIdxToFlatIdx(cellIdx + int3(0, 0, 1), gridData.gridSize);
        cid_zn = isSolidCell(getZNextMaterial(cellMaterial)) ? cid : cid_zn;
        pressure += params.coeff1.z * (pressureInBuffer[cid_zp] + pressureInBuffer[cid_zn]);

        pressure += params.coeff1.w * divergenceBuffer[cid];
    }

    pressureOutBuffer[cid] = pressure;
}

[numthreads(WorkgroupSize, 1, 1)]
void updateVelocities(uint32_t3 ID : SV_DispatchThreadID)
{
    uint cid = ID.x;
    int3 cellIdx = flatIdxToCellIdx(cid, gridData.gridSize);

    uint cellMaterial = cellMaterialBuffer[cid];

    float3 velocity = velocityFieldBuffer[cid].xyz;
    float pressure = pressureInBuffer[cid];

    uint cid_xp = cellIdxToFlatIdx(cellIdx + int3(-1, 0, 0), gridData.gridSize);
    cid_xp = isSolidCell(getXPrevMaterial(cellMaterial)) ? cid : cid_xp;
    velocity.x -= params.coeff2.x * (pressure - pressureInBuffer[cid_xp]);

    uint cid_yp = cellIdxToFlatIdx(cellIdx + int3(0, -1, 0), gridData.gridSize);
    cid_yp = isSolidCell(getYPrevMaterial(cellMaterial)) ? cid : cid_yp;
    velocity.y -= params.coeff2.y * (pressure - pressureInBuffer[cid_yp]);

    uint cid_zp = cellIdxToFlatIdx(cellIdx + int3(0, 0, -1), gridData.gridSize);
    cid_zp = isSolidCell(getZPrevMaterial(cellMaterial)) ? cid : cid_zp;
    velocity.z -= params.coeff2.z * (pressure - pressureInBuffer[cid_zp]);

    enforceBoundaryCondition(velocity, cellMaterial);

    velocityFieldBuffer[cid].xyz = velocity;
}
