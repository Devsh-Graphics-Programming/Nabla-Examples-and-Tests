#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "nbl/builtin/hlsl/sort/counting.hlsl"

[[vk::push_constant]] nbl::hlsl::sort::CountingPushData pushData;

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    nbl::hlsl::sort::counting < bda::PtrAccessor<uint32_t>, bda::PtrAccessor<uint32_t>, bda::PtrAccessor<uint32_t> > counter;
    counter.scatter(
        bda::PtrAccessor<uint32_t>::createAccessor(pushData.inputKeyAddress),
        bda::PtrAccessor<uint32_t>::createAccessor(pushData.inputValueAddress),
        bda::PtrAccessor<uint32_t>::createAccessor(pushData.scratchAddress),
        bda::PtrAccessor<uint32_t>::createAccessor(pushData.outputKeyAddress),
        bda::PtrAccessor<uint32_t>::createAccessor(pushData.outputValueAddress),
        pushData
    );
}