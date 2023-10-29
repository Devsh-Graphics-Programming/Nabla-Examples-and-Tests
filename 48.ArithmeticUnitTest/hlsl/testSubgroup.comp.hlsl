#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "shaderCommon.hlsl"

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint32_t invIdx : SV_GroupIndex, uint32_t3 globalId : SV_DispatchThreadID)
{
	__gl_LocalInvocationIndex = invIdx;
	__gl_GlobalInvocationID = globalId;

	test();
}