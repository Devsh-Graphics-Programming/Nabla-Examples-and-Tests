#include "common.hlsl"

// just a small test
#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	if (ID.x>=pushConstants.dataElementCount)
		return;

	const uint32_t self = vk::RawBufferLoad<uint32_t>(pushConstants.inputAddress+sizeof(uint32_t)*ID.x);

	vk::RawBufferStore<uint32_t>(pushConstants.inputAddress+sizeof(uint32_t)*ID.x, 2 * self);
}
