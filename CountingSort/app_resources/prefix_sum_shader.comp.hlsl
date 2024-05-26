#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "nbl/builtin/hlsl/sort/counting.hlsl"
#include "app_resources/common.hlsl"

[[vk::push_constant]] CountingPushData pushData;

struct PtrAccessor
{
    static PtrAccessor create(const uint64_t addr)
    {
        PtrAccessor ptr;
        ptr.addr = addr;
        return ptr;
    }

    uint32_t get(const uint64_t index)
    {
        return nbl::hlsl::bda::__ptr < uint32_t > (addr + sizeof(uint32_t) * index).template
        deref().load();
    }

    void set(const uint64_t index, const uint32_t value)
    {
        nbl::hlsl::bda::__ptr < uint32_t > (addr + sizeof(uint32_t) * index).template
        deref().store(value);
    }

    uint32_t atomicAdd(const uint64_t index, const uint32_t value)
    {
        nbl::hlsl::bda::__spv_ptr_t < uint32_t > ptr = nbl::hlsl::bda::__ptr < uint32_t > (addr + sizeof(uint32_t) * index).template
        deref().get_ptr();

        return nbl::hlsl::glsl::atomicAdd(ptr, value);
    }

    uint32_t atomicSub(const uint64_t index, const uint32_t value)
    {
        return atomicAdd(index, (uint32_t) (-1 * value));
    }

    uint64_t addr;
};

groupshared uint32_t sdata[BucketCount];

struct SharedAccessor
{
    uint32_t get(const uint32_t index)
    {
        return sdata[index];
    }

    void set(const uint32_t index, const uint32_t value)
    {
        sdata[index] = value;
    }

    uint32_t atomicAdd(const uint32_t index, const uint32_t value)
    {
        return nbl::hlsl::glsl::atomicAdd(sdata[index], value);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
    }
};

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    nbl::hlsl::sort::CountingParameters < uint32_t > params;
    params.dataElementCount = pushData.dataElementCount;
    params.elementsPerWT = pushData.elementsPerWT;
    params.minimum = pushData.minimum;
    params.maximum = pushData.maximum;

    nbl::hlsl::sort::counting <WorkgroupSize, BucketCount, uint32_t, PtrAccessor, PtrAccessor, PtrAccessor, SharedAccessor> counter;
    PtrAccessor input_accessor = PtrAccessor::create(pushData.inputKeyAddress);
    PtrAccessor scratch_accessor = PtrAccessor::create(pushData.scratchAddress);
    SharedAccessor shared_accessor;
    counter.histogram(
        input_accessor,
        scratch_accessor,
        shared_accessor,
        params
    );
}