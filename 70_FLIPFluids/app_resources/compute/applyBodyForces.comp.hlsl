#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../descriptor_bindings.hlsl"

[[vk::binding(b_abfGridData, s_abf)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(b_abfVelFieldBuffer, s_abf)]] RWTexture3D<float> velocityFieldBuffer[3];
[[vk::binding(b_abfCMBuffer, s_abf)]] RWStructuredBuffer<uint> cellMaterialBuffer;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    // only gravity for now
    uint c_id = ID.x;
    int3 cIdx = flatIdxToCellIdx(c_id, gridData.gridSize);

    float3 velocity;
    velocity.x = velocityFieldBuffer[0][cIdx];
    velocity.y = velocityFieldBuffer[1][cIdx];
    velocity.z = velocityFieldBuffer[2][cIdx];

    velocity += float3(0, -1, 0) * gravity * deltaTime;

    enforceBoundaryCondition(velocity, cellMaterialBuffer[c_id]);

    velocityFieldBuffer[0][cIdx] = velocity.x;
    velocityFieldBuffer[1][cIdx] = velocity.y;
    velocityFieldBuffer[2][cIdx] = velocity.z;
}
