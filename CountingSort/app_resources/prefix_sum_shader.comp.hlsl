#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "nbl/builtin/hlsl/sort/counting.hlsl"
#include "nbl/builtin/hlsl/sort/common.hlsl"

[[vk::push_constant]] nbl::hlsl::sort::CountingPushData pushData;

struct PtrAccessor
{
    static PtrAccessor createAccessor(uint64_t addr)
    {
        PtrAccessor ptr;
        ptr.addr = addr;
        return ptr;
    }

    uint32_t get(uint64_t index)
    {
        return bda::__ptr < uint32_t > (addr + sizeof(uint32_t) * index).template
        deref().load();
    }

    void set(uint64_t index, uint32_t value)
    {
        bda::__ptr < uint32_t > (addr + sizeof(uint32_t) * index).template
        deref().store(value);
    }

    uint32_t atomicAdd(uint64_t index, uint32_t value)
    {
        return bda::__ptr < uint32_t > (addr + sizeof(uint32_t) * index).template
        deref().atomicAdd(value);
    }

    uint64_t addr;
};

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    nbl::hlsl::sort::counting <PtrAccessor, PtrAccessor, PtrAccessor> counter;
    counter.histogram(
        PtrAccessor::createAccessor(pushData.inputKeyAddress),
        PtrAccessor::createAccessor(pushData.scratchAddress),
        pushData
    );
}