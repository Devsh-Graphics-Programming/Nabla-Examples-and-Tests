#include "../common.hlsl"
#include "../cellUtils.hlsl"
#include "../descriptor_bindings.hlsl"

[[vk::binding(b_abfVelFieldBuffer, s_abf)]] RWStructuredBuffer<float4> velocityFieldBuffer;
[[vk::binding(b_abfCMBuffer, s_abf)]] RWStructuredBuffer<uint> cellMaterialBuffer;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    // only gravity for now
    uint c_id = ID.x;

    float3 velocity = velocityFieldBuffer[c_id].xyz;
    velocity += float3(0, -1, 0) * gravity * deltaTime;

    enforceBoundaryCondition(velocity, cellMaterialBuffer[c_id]);

    velocityFieldBuffer[c_id].xyz = velocity;
}
