#include "../common.hlsl"
#include "../gridUtils.hlsl"
#include "../cellUtils.hlsl"
#include "../descriptor_bindings.hlsl"

[[vk::binding(b_abfGridData, s_abf)]]
cbuffer GridData
{
    SGridData gridData;
};

[[vk::binding(b_abfVelField, s_abf)]] RWTexture3D<float> velocityField[3];
[[vk::binding(b_abfCM, s_abf)]] RWTexture3D<uint> cellMaterialGrid;

// TODO: can this kernel be fused with any preceeding/succeeding it?
[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    // only gravity for now
    int3 cIdx = ID;

    float3 velocity;
    velocity.x = velocityField[0][cIdx];
    velocity.y = velocityField[1][cIdx];
    velocity.z = velocityField[2][cIdx];

    velocity += float3(0, -1, 0) * gravity * deltaTime;

    enforceBoundaryCondition(velocity, cellMaterialGrid[cIdx]);

    velocityField[0][cIdx] = velocity.x;
    velocityField[1][cIdx] = velocity.y;
    velocityField[2][cIdx] = velocity.z;
}
