#pragma wave shader_stage(compute)
#include "common.hlsl"

[[vk::binding(4,2)]] RWStructuredBuffer<SomeType> asdf;
[[vk::binding(6,3)]] RWByteAddressBuffer outputBuff;
[[vk::binding(2,0)]] RWByteAddressBuffer output2[];

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint32_t index = ID.x;
	const uint32_t byteOffset = sizeof(uint32_t)*index;

    outputBuff.Store<uint32_t>(byteOffset, asdf[index].a);
    output2[0].Store<uint32_t>(byteOffset, asdf[index].a);
}