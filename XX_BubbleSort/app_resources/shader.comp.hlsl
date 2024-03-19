#include "common.hlsl"

// just a small test
#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

// does absolutely nothing, a later example will show how it gets used
template<typename capability_traits=nbl::hlsl::jit::device_capabilities_traits>
void dummyTraitTest() {}

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	dummyTraitTest();
	if (ID.x>=pushConstants.dataElementCount)
		return;

	const input_t self = vk::RawBufferLoad<input_t>(pushConstants.inputAddress+sizeof(input_t)*ID.x);

	nbl::hlsl::Xoroshiro64StarStar rng = nbl::hlsl::Xoroshiro64StarStar::construct(uint32_t2(pushConstants.dataElementCount,ID.x)^0xdeadbeefu);

	float32_t acc = nbl::hlsl::numeric_limits<float32_t>::max;
	const static uint32_t OthersToTest = 15;
	[[unroll(OthersToTest)]]
	for (uint32_t i=0; i<OthersToTest; i++)
	{
		const uint32_t offset = rng() % pushConstants.dataElementCount;
		const input_t other = vk::RawBufferLoad<input_t>(pushConstants.inputAddress+sizeof(input_t)*offset);
		acc = min(length(other-self),acc);
	}
	vk::RawBufferStore<float32_t>(pushConstants.outputAddress+sizeof(float32_t)*ID.x,acc);
}