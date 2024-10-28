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
[[vk::binding(b_abfCMBuffer, s_abf)]] RWTexture3D<uint> cellMaterialBuffer;

[numthreads(WorkgroupGridDim, WorkgroupGridDim, WorkgroupGridDim)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    // only gravity for now
    int3 cIdx = ID;

    float3 velocity;
    velocity.x = velocityFieldBuffer[0][cIdx];
    velocity.y = velocityFieldBuffer[1][cIdx];
    velocity.z = velocityFieldBuffer[2][cIdx];

    velocity += float3(0, -1, 0) * gravity * deltaTime;

    enforceBoundaryCondition(velocity, cellMaterialBuffer[cIdx]);

    velocityFieldBuffer[0][cIdx] = velocity.x;
    velocityFieldBuffer[1][cIdx] = velocity.y;
    velocityFieldBuffer[2][cIdx] = velocity.z;
}
