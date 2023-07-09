#include "../examples_tests/48.ArithmeticUnitTest/hlsl/wgsize.hlsl"
static uint3 gl_GlobalInvocationID;
static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "../examples_tests/48.ArithmeticUnitTest/hlsl/shaderCommon.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/shared_memory_accessor.hlsl"

#define inclusive_scan_t(Binop) nbl::hlsl::workgroup::inclusive_scan<uint, nbl::hlsl::binops::Binop<uint>, nbl::hlsl::SharedMemory>

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
	outbitcount[0].subgroupSize = nbl::hlsl::subgroup::Size();
	
	const uint sourceVal = inputValue[gl_GlobalInvocationID.x];
	
	inclusive_scan_t(bitwise_and) iscan_and;
	outand[0].output[gl_GlobalInvocationID.x] = iscan_and(sourceVal);
	
	inclusive_scan_t(bitwise_xor) iscan_xor;
	outxor[0].output[gl_GlobalInvocationID.x] = iscan_xor(sourceVal);
	
	inclusive_scan_t(bitwise_or) iscan_or;
	outor[0].output[gl_GlobalInvocationID.x] = iscan_or(sourceVal);
	
	inclusive_scan_t(add) iscan_add;
	outadd[0].output[gl_GlobalInvocationID.x] = iscan_add(sourceVal);
	
	inclusive_scan_t(mul) iscan_mul;
	outmul[0].output[gl_GlobalInvocationID.x] = iscan_mul(sourceVal);
	
	inclusive_scan_t(min) iscan_min;
	outmin[0].output[gl_GlobalInvocationID.x] = iscan_min(sourceVal);
	
	inclusive_scan_t(max) iscan_max;
	outmax[0].output[gl_GlobalInvocationID.x] = iscan_max(sourceVal);
	
	//nbl::hlsl::workgroup::ballot<nbl::hlsl::SharedMemoryAdaptor<nbl::hlsl::MemProxy>, false>((sourceVal & 0x1u) == 0x1u);
	//outbitcount[0].output[gl_GlobalInvocationID.x] = nbl::hlsl::workgroup::ballotInclusiveBitCount<nbl::hlsl::SharedMemoryAdaptor<nbl::hlsl::MemProxy> >();
}
