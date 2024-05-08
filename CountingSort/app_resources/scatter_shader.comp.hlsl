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

    bda::__spv_ptr_t<uint32_t> get_ptr(uint64_t index)
    {
        return bda::__ptr < uint32_t > (addr + sizeof(uint32_t) * index).template
        deref().get_ptr();
    }

    uint64_t addr;
};

struct DoublePtrAccessor
{
    static DoublePtrAccessor createAccessor(uint64_t in_addr, uint64_t out_addr)
    {
        DoublePtrAccessor ptr;
        ptr.in_addr = in_addr;
        ptr.out_addr = out_addr;
        return ptr;
    }

    uint32_t get(uint64_t index)
    {
        return bda::__ptr < uint32_t > (in_addr + sizeof(uint32_t) * index).template
        deref().load();
    }

    void set(uint64_t index, uint32_t value)
    {
        bda::__ptr < uint32_t > (out_addr + sizeof(uint32_t) * index).template
        deref().store(value);
    }

    uint64_t in_addr, out_addr;
};

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    nbl::hlsl::sort::counting <DoublePtrAccessor, DoublePtrAccessor, PtrAccessor> counter;
    DoublePtrAccessor key_accessor = DoublePtrAccessor::createAccessor(pushData.inputKeyAddress, pushData.outputKeyAddress);
    DoublePtrAccessor value_accessor = DoublePtrAccessor::createAccessor(pushData.inputValueAddress, pushData.outputValueAddress);
    PtrAccessor scratch_accessor = PtrAccessor::createAccessor(pushData.scratchAddress);
    counter.scatter(
        key_accessor,
        value_accessor,
        scratch_accessor,
        pushData
    );
}