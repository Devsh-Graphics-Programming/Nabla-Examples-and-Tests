#include "common.hlsl"

// just a small test
#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	if (ID.x>=pushConstants.dataElementCount)
		return;

	const input_t self = vk::RawBufferLoad<input_t>(pushConstants.inputAddress+sizeof(input_t)*ID.x);

	float32_t acc = nbl::hlsl::numeric_limits<float32_t>::max;
	vk::RawBufferStore<float32_t>(pushConstants.outputAddress+sizeof(float32_t)*ID.x,acc);
}
