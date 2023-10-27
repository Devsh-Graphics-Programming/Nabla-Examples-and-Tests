#include "shaderCommon.hlsl"

#ifndef OPERATION
#error "Define OPERATION!"
#endif

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint32_t invIdx : SV_GroupIndex, uint32_t3 globalId : SV_DispatchThreadID)
{
	__gl_LocalInvocationIndex = invIdx;
	__gl_GlobalInvocationID = globalId;
	test<nbl::hlsl::subgroup::OPERATION>::run();
}
