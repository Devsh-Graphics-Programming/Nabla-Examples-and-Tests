#pragma wave shader_stage(compute)
#include "common.hlsl"

[[vk::binding(0,0)]] RWStructuredBuffer<SomeType> asdf;
[[vk::binding(0,1)]] RWByteAddressBuffer output;
[[vk::binding(0,2)]] RWByteAddressBuffer output2[2];
[[vk::binding(1,2)]] RWByteAddressBuffer output3[];

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint32_t index = ID.x;
	const uint32_t byteOffset = sizeof(uint32_t)*index;

    output.Store<uint32_t>(byteOffset, asdf[index].a);
    output2[1].Store<uint32_t>(byteOffset, asdf[index].b[1]);
    output3[5].Store<uint32_t>(byteOffset, asdf[index].c[2]);
}