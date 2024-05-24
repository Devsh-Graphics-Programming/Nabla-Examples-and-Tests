#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "nbl/builtin/hlsl/sort/counting.hlsl"
#include "app_resources/common.hlsl"

[[vk::push_constant]] CountingPushData pushData;

struct PtrAccessor
{
    static PtrAccessor create(uint64_t addr)
    {
        PtrAccessor ptr;
        ptr.addr = addr;
        return ptr;
    }

    uint32_t get(uint64_t index)
    {
        return nbl::hlsl::bda::__ptr < uint32_t > (addr + sizeof(uint32_t) * index).template
        deref().load();
    }

    void set(uint64_t index, uint32_t value)
    {
        nbl::hlsl::bda::__ptr < uint32_t > (addr + sizeof(uint32_t) * index).template
        deref().store(value);
    }

    nbl::hlsl::bda::__spv_ptr_t<uint32_t> get_ptr(uint64_t index)
    {
        return nbl::hlsl::bda::__ptr < uint32_t > (addr + sizeof(uint32_t) * index).template
        deref().get_ptr();
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
    nbl::hlsl::sort::counting <uint32_t, PtrAccessor, PtrAccessor, PtrAccessor> counter;
    PtrAccessor input_accessor = PtrAccessor::create(pushData.inputKeyAddress);
    PtrAccessor scratch_accessor = PtrAccessor::create(pushData.scratchAddress);
    /*counter.histogram(
        input_accessor,
        scratch_accessor,
        pushData
    );*/
}