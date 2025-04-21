#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "shaderCommon.hlsl"

uint32_t globalFirstItemIndex(uint32_t itemIdx)
{
    return nbl::hlsl::glsl::gl_WorkGroupID().x*WORKGROUP_SIZE*ITEMS_PER_INVOCATION+((nbl::hlsl::glsl::gl_SubgroupID()*ITEMS_PER_INVOCATION+itemIdx)<<SUBGROUP_SIZE_LOG2);
}

bool canStore() {return true;}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    test();
}
