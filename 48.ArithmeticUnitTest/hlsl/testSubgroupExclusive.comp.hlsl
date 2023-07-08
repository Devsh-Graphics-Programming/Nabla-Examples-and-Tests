static uint3 gl_GlobalInvocationID;
static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "../examples_tests/48.ArithmeticUnitTest/hlsl/shaderCommon.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"
#include "nbl/builtin/hlsl/shared_memory_accessor.hlsl"

#define exclusive_scan_t(Binop) nbl::hlsl::subgroup::exclusive_scan<uint, nbl::hlsl::binops::Binop<uint>, nbl::hlsl::SharedMemoryAdaptor<nbl::hlsl::MemProxy> >

[numthreads(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1)]
void main(uint3 globalId : SV_DispatchThreadID, 
          uint3 groupId : SV_GroupID, 
          uint invIdx : SV_GroupIndex)
{
	gl_GlobalInvocationID = globalId;
	gl_WorkGroupID = groupId;
	gl_LocalInvocationIndex = invIdx;
	
	outand[0].subgroupSize = nbl::hlsl::subgroup::Size();
	outxor[0].subgroupSize = nbl::hlsl::subgroup::Size();
	outor[0].subgroupSize = nbl::hlsl::subgroup::Size();
	outadd[0].subgroupSize = nbl::hlsl::subgroup::Size();
	outmul[0].subgroupSize = nbl::hlsl::subgroup::Size();
	outmin[0].subgroupSize = nbl::hlsl::subgroup::Size();
	outmax[0].subgroupSize = nbl::hlsl::subgroup::Size();
	
	const uint sourceVal = inputValue[gl_GlobalInvocationID.x];
	
	exclusive_scan_t(bitwise_and) exscan_and;
	outand[0].output[gl_GlobalInvocationID.x] = exscan_and(sourceVal);
	
	exclusive_scan_t(bitwise_xor) exscan_xor;
	outxor[0].output[gl_GlobalInvocationID.x] = exscan_xor(sourceVal);
	
	exclusive_scan_t(bitwise_or) exscan_or;
	outor[0].output[gl_GlobalInvocationID.x] = exscan_or(sourceVal);
	
	exclusive_scan_t(add) exscan_add;
	outadd[0].output[gl_GlobalInvocationID.x] = exscan_add(sourceVal);
	
	exclusive_scan_t(mul) exscan_mul;
	outmul[0].output[gl_GlobalInvocationID.x] = exscan_mul(sourceVal);
	
	exclusive_scan_t(min) exscan_min;
	outmin[0].output[gl_GlobalInvocationID.x] = exscan_min(sourceVal);
	
	exclusive_scan_t(max) exscan_max;
	outmax[0].output[gl_GlobalInvocationID.x] = exscan_max(sourceVal);
}
