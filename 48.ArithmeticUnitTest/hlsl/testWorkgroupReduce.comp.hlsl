#include "../hlsl/wgsize.hlsl"
static uint3 gl_GlobalInvocationID;
static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "../hlsl/shaderCommon.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/shared_memory_accessor.hlsl"

#define reduction_t(Binop) nbl::hlsl::workgroup::reduction<uint, nbl::hlsl::binops::Binop<uint>, nbl::hlsl::SharedMemory>

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
	
	reduction_t(bitwise_and) r_and;
	outand[0].output[gl_GlobalInvocationID.x] = r_and(sourceVal);
	
	reduction_t(bitwise_xor) r_xor;
	outxor[0].output[gl_GlobalInvocationID.x] = r_xor(sourceVal);
	
	reduction_t(bitwise_or) r_or;
	outor[0].output[gl_GlobalInvocationID.x] = r_or(sourceVal);
	
	reduction_t(add) r_add;
	outadd[0].output[gl_GlobalInvocationID.x] = r_add(sourceVal);
	
	reduction_t(mul) r_mul;
	outmul[0].output[gl_GlobalInvocationID.x] = r_mul(sourceVal);
	
	reduction_t(min) r_min;
	outmin[0].output[gl_GlobalInvocationID.x] = r_min(sourceVal);
	
	reduction_t(max) r_max;
	outmax[0].output[gl_GlobalInvocationID.x] = r_max(sourceVal);
	
	nbl::hlsl::workgroup::ballot<nbl::hlsl::SharedMemory, false>((sourceVal & 0x1u) == 0x1u);
	outbitcount[0].output[gl_GlobalInvocationID.x] = nbl::hlsl::workgroup::ballotBitCount<nbl::hlsl::SharedMemory>();
}
