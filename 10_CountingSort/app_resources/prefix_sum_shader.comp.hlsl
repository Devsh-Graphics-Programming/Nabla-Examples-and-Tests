#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "nbl/builtin/hlsl/sort/counting.hlsl"
#include "app_resources/common.hlsl"

[[vk::push_constant]] CountingPushData pushData;

using Ptr = nbl::hlsl::bda::__ptr < uint32_t >;
using PtrAccessor = nbl::hlsl::BdaAccessor < uint32_t >;

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

    using Counter = nbl::hlsl::sort::counting < WorkgroupSize, BucketCount, PtrAccessor, PtrAccessor, PtrAccessor, SharedAccessor>;
    Counter counter = Counter::create(nbl::hlsl::glsl::gl_WorkGroupID().x);

    const Ptr input_ptr = Ptr::create(pushData.inputKeyAddress);
    const Ptr histogram_ptr = Ptr::create(pushData.histogramAddress);

    PtrAccessor input_accessor = PtrAccessor::create(input_ptr);
    PtrAccessor histogram_accessor = PtrAccessor::create(histogram_ptr);
    SharedAccessor shared_accessor;
    counter.histogram(
        input_accessor,
        histogram_accessor,
        shared_accessor,
        params
    );
}