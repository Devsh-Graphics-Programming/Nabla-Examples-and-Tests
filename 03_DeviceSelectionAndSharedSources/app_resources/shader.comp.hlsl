#include "common.hlsl"

[[vk::binding(2,1)]] ByteAddressBuffer inputs[2];
[[vk::binding(6,3)]] RWByteAddressBuffer output;

struct PushConstants
{
	uint2 a;
	uint b;
};

[[vk::push_constant]]
PushConstants u_pushConstants;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	const uint32_t byteOffset = sizeof(uint32_t)*ID.x;
	output.Store<uint32_t>(byteOffset,inputs[0].Load<uint32_t>(byteOffset)+inputs[1].Load<uint32_t>(byteOffset)
		+ uint32_t(u_pushConstants.a.x) + uint32_t(u_pushConstants.a.y) + uint32_t(u_pushConstants.b));
}