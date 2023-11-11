#include "common.hlsl"

// Yes we do have our own re-creation of C++'s STL in HLSL2021 !
#include "nbl/builtin/hlsl/limits.hlsl"

// Thanks to Scalar Block Layout being required there's no weird padding on the vector
[[vk::binding(DataBufferBinding,0)]] StructuredBuffer<input_t> data;
[[vk::binding(MetricBufferBinding,0)]] RWStructuredBuffer<output_t> metrics;

[[vk::push_constant]] PushConstantData pushConstants;

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	/*
	if (ID.x<pushConstants.dataElementCount)
		return;

	float32_t acc = nbl::hlsl::numeric_limits<float32_t>::lowest;

	const static int32_t NeighbourRange = 15;
	[[unroll(NeighbourRange)]]
	for (int32_t i=max(ID.x-,0); i<max(,pushConstants.dataElementCount); i++)
		acc = min(acc,length(data[ID.x]-pushConstants.centralPoint));
		*/
}