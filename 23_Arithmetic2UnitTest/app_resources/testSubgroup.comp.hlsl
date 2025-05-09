#pragma shader_stage(compute)

#define operation_t nbl::hlsl::OPERATION

#include "shaderCommon.hlsl"

uint32_t globalIndex()
{
    return nbl::hlsl::glsl::gl_WorkGroupID().x*WORKGROUP_SIZE+nbl::hlsl::workgroup::SubgroupContiguousIndex();
}

bool canStore() {return true;}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    test();
}
