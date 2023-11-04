#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "shaderCommon.hlsl"

uint32_t globalIndex()
{
	return nbl::hlsl::glsl::gl_WorkGroupID().x*WORKGROUP_SIZE+nbl::hlsl::workgroup::SubgroupContiguousIndex();
}

bool canStore() {return true;}

[numthreads(WORKGROUP_SIZE,1,1)]
void main(uint32_t invIdx : SV_GroupIndex, uint32_t3 wgId : SV_GroupID)
{
	__gl_LocalInvocationIndex = invIdx;
	__gl_WorkGroupID = wgId;

	test();
}