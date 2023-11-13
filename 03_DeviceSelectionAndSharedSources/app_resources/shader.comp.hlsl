#include "common.hlsl"

// intentionally making my live difficult here, to showcase the power of reflection
[[vk::binding(2,1)]] ByteAddressBuffer inputs[2];
[[vk::binding(6,3)]] RWByteAddressBuffer output;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	const uint32_t byteOffset = sizeof(uint32_t)*ID.x;
	output.Store<uint32_t>(byteOffset,inputs[0].Load<uint32_t>(byteOffset)+inputs[1].Load<uint32_t>(byteOffset));
}