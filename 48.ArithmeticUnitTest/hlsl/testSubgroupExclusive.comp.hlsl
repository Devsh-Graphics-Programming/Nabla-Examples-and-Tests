static uint3 gl_GlobalInvocationID;
static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "../examples_tests/48.ArithmeticUnitTest/hlsl/shaderCommon.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"

#define exclusive_scan_t(Binop) nbl::hlsl::subgroup::exclusive_scan<uint, nbl::hlsl::Binop<uint> >

[numthreads(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1)]
void main(uint3 globalId : SV_DispatchThreadID,
          uint3 groupId : SV_GroupID,
          uint invIdx : SV_GroupIndex)
{
	gl_GlobalInvocationID = globalId;
	gl_WorkGroupID = groupId;
	gl_LocalInvocationIndex = invIdx;

	outand[0].subgroupSize = nbl::hlsl::glsl::gl_SubgroupSize();
	outxor[0].subgroupSize = nbl::hlsl::glsl::gl_SubgroupSize();
	outor[0].subgroupSize = nbl::hlsl::glsl::gl_SubgroupSize();
	outadd[0].subgroupSize = nbl::hlsl::glsl::gl_SubgroupSize();
	outmul[0].subgroupSize = nbl::hlsl::glsl::gl_SubgroupSize();
	outmin[0].subgroupSize = nbl::hlsl::glsl::gl_SubgroupSize();
	outmax[0].subgroupSize = nbl::hlsl::glsl::gl_SubgroupSize();

	const uint sourceVal = inputValue[gl_GlobalInvocationID.x];

	exclusive_scan_t(bit_and) exscan_and;
	outand[0].output[gl_GlobalInvocationID.x] = exscan_and(sourceVal);

	exclusive_scan_t(bit_xor) exscan_xor;
	outxor[0].output[gl_GlobalInvocationID.x] = exscan_xor(sourceVal);

	exclusive_scan_t(bit_or) exscan_or;
	outor[0].output[gl_GlobalInvocationID.x] = exscan_or(sourceVal);

	exclusive_scan_t(plus) exscan_add;
	outadd[0].output[gl_GlobalInvocationID.x] = exscan_add(sourceVal);

	exclusive_scan_t(multiplies) exscan_mul;
	outmul[0].output[gl_GlobalInvocationID.x] = exscan_mul(sourceVal);

	exclusive_scan_t(minimum) exscan_min;
	outmin[0].output[gl_GlobalInvocationID.x] = exscan_min(sourceVal);

	exclusive_scan_t(maximum) exscan_max;
	outmax[0].output[gl_GlobalInvocationID.x] = exscan_max(sourceVal);
}
