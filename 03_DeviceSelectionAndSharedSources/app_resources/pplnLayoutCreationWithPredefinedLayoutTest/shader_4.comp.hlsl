#pragma wave shader_stage(compute)
#include "common.hlsl"

[[vk::binding(0,1)]] RWByteAddressBuffer output;

struct SubsetPushConstants
{
    uint32_t a[4];
};

[[vk::push_constant]] 
SubsetPushConstants pc;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint32_t index = ID.x;
	const uint32_t byteOffset = sizeof(uint32_t)*index;

    output.Store<uint32_t>(byteOffset, pc.a[index % 4]);
}