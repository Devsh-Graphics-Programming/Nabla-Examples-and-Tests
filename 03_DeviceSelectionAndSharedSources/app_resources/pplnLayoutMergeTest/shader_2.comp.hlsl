#pragma wave shader_stage(compute)
#include "common.hlsl"

[[vk::binding(0,0)]] RWStructuredBuffer<SomeType> asdf;
[[vk::binding(1,0)]] RWStructuredBuffer<SomeType> fdsa;
[[vk::binding(2,0)]] RWByteAddressBuffer output;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint32_t index = ID.x;
	const uint32_t byteOffset = sizeof(uint32_t)*index;

    output.Store<uint32_t>(byteOffset, asdf[index].a + fdsa[index].b[2]);
}