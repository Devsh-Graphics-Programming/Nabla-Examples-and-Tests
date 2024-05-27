#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"
#include "nbl/builtin/hlsl/sort/counting.hlsl"
#include "app_resources/common.hlsl"

[[vk::push_constant]] CountingPushData pushData;

using PtrAccessor = nbl::hlsl::bda::BdaAccessor < uint32_t >;

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

struct DoublePtrAccessor
{
    static DoublePtrAccessor create(const PtrAccessor input, const PtrAccessor output)
    {
        DoublePtrAccessor accessor;
        accessor.input = input;
        accessor.output = output;
        return accessor;
    }

    uint32_t get(const uint64_t index)
    {
        return input.get(index);
    }

    void set(const uint64_t index, const uint32_t value)
    {
        output.set(index, value);
    }

    PtrAccessor input, output;
};

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    nbl::hlsl::sort::CountingParameters < uint32_t > params;
    params.dataElementCount = pushData.dataElementCount;
    params.elementsPerWT = pushData.elementsPerWT;
    params.minimum = pushData.minimum;
    params.maximum = pushData.maximum;

    nbl::hlsl::sort::counting <WorkgroupSize, BucketCount, uint32_t, DoublePtrAccessor, DoublePtrAccessor, PtrAccessor, SharedAccessor > counter;
    DoublePtrAccessor key_accessor = DoublePtrAccessor::create(
        PtrAccessor::create(pushData.inputKeyAddress),
        PtrAccessor::create(pushData.outputKeyAddress)
    );
    DoublePtrAccessor value_accessor = DoublePtrAccessor::create(
        PtrAccessor::create(pushData.inputValueAddress),
        PtrAccessor::create(pushData.outputValueAddress)
    );
    PtrAccessor histogram_accessor = PtrAccessor::create(pushData.histogramAddress);
    SharedAccessor shared_accessor;
    counter.scatter(
        key_accessor,
        value_accessor,
        histogram_accessor,
        shared_accessor,
        params
    );
}