static uint3 gl_GlobalInvocationID;
static uint3 gl_WorkGroupID;
static uint gl_LocalInvocationIndex;

#include "shaderCommon.hlsl"

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"

#define inclusive_scan_t(Binop) nbl::hlsl::workgroup::inclusive_scan<uint, nbl::hlsl::Binop<uint>, SharedMemory>

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
    
	outand[0].output[gl_GlobalInvocationID.x] = inclusive_scan_t(bit_and)(sourceVal, memoryAccessor);
	memoryAccessor.workgroupExecutionAndMemoryBarrier();
	outxor[0].output[gl_GlobalInvocationID.x] = inclusive_scan_t(bit_xor)(sourceVal, memoryAccessor);
	memoryAccessor.workgroupExecutionAndMemoryBarrier();
	outor[0].output[gl_GlobalInvocationID.x] = inclusive_scan_t(bit_or)(sourceVal, memoryAccessor);
	memoryAccessor.workgroupExecutionAndMemoryBarrier();
	outadd[0].output[gl_GlobalInvocationID.x] = inclusive_scan_t(plus)(sourceVal, memoryAccessor);
	memoryAccessor.workgroupExecutionAndMemoryBarrier();
	outmul[0].output[gl_GlobalInvocationID.x] = inclusive_scan_t(multiplies)(sourceVal, memoryAccessor);
	memoryAccessor.workgroupExecutionAndMemoryBarrier();
	outmin[0].output[gl_GlobalInvocationID.x] = inclusive_scan_t(minimum)(sourceVal, memoryAccessor);
	memoryAccessor.workgroupExecutionAndMemoryBarrier();
	outmax[0].output[gl_GlobalInvocationID.x] = inclusive_scan_t(maximum)(sourceVal, memoryAccessor);
    memoryAccessor.workgroupExecutionAndMemoryBarrier();
	nbl::hlsl::workgroup::ballot<SharedMemory>((sourceVal & 0x1u) == 0x1u, memoryAccessor);
    memoryAccessor.broadcast.workgroupExecutionAndMemoryBarrier();
	outbitcount[0].output[gl_GlobalInvocationID.x] = nbl::hlsl::workgroup::ballotInclusiveBitCount<SharedMemory>(memoryAccessor);
}
