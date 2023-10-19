static uint3 gl_GlobalInvocationID;
static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "shaderCommon.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/shared_memory_accessor.hlsl"

#define exclusive_scan_t(Binop) nbl::hlsl::workgroup::exclusive_scan<uint, nbl::hlsl::binops::Binop<uint>, SharedMemory>

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
	outbitcount[0].subgroupSize = nbl::hlsl::glsl::gl_SubgroupSize();

    SharedMemory memoryAccessor;
	const uint sourceVal = inputValue[gl_GlobalInvocationID.x];
	
	outand[0].output[gl_GlobalInvocationID.x] = exclusive_scan_t(bitwise_and)(sourceVal, memoryAccessor);
	
	outxor[0].output[gl_GlobalInvocationID.x] = exclusive_scan_t(bitwise_xor)(sourceVal, memoryAccessor);
	
	outor[0].output[gl_GlobalInvocationID.x] = exclusive_scan_t(bitwise_or)(sourceVal, memoryAccessor);
	
	outadd[0].output[gl_GlobalInvocationID.x] = exclusive_scan_t(add)(sourceVal, memoryAccessor);
	
	outmul[0].output[gl_GlobalInvocationID.x] = exclusive_scan_t(mul)(sourceVal, memoryAccessor);
	
	outmin[0].output[gl_GlobalInvocationID.x] = exclusive_scan_t(min)(sourceVal, memoryAccessor);
	
	outmax[0].output[gl_GlobalInvocationID.x] = exclusive_scan_t(max)(sourceVal, memoryAccessor);
	
	nbl::hlsl::workgroup::ballot<SharedMemory>((sourceVal & 0x1u) == 0x1u, memoryAccessor);
    memoryAccessor.main.workgroupExecutionAndMemoryBarrier();
	outbitcount[0].output[gl_GlobalInvocationID.x] = nbl::hlsl::workgroup::ballotExclusiveBitCount<SharedMemory>(memoryAccessor);
}