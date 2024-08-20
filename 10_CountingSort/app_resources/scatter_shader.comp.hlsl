#include "nbl/builtin/hlsl/sort/counting.hlsl"
#include "app_resources/common.hlsl"

[[vk::push_constant]] CountingPushData pushData;

struct DoublePtrAccessor
{
    using type_t = uint32_t;

    static DoublePtrAccessor create(const PtrAccessor input, const PtrAccessor output)
    {
        DoublePtrAccessor accessor;
        accessor.input = input;
        accessor.output = output;
        return accessor;
    }

    void get(const uint64_t index, NBL_REF_ARG(uint32_t) value)
    {
        input.get(index,value);
    }

    void set(const uint64_t index, const uint32_t value)
    {
        output.set(index,value);
    }

    PtrAccessor input, output;
};

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    sort::CountingParameters<uint32_t> params;
    params.dataElementCount = pushData.dataElementCount;
    params.elementsPerWT = pushData.elementsPerWT;
    params.minimum = pushData.minimum;
    params.maximum = pushData.maximum;

    using Counter = sort::counting<WorkgroupSize, BucketCount, DoublePtrAccessor, DoublePtrAccessor, PtrAccessor, SharedAccessor, PtrAccessor::type_t>;
    Counter counter = Counter::create(glsl::gl_WorkGroupID().x);

    const Ptr input_key_ptr = Ptr::create(pushData.inputKeyAddress);
    const Ptr input_value_ptr = Ptr::create(pushData.inputValueAddress);
    const Ptr histogram_ptr = Ptr::create(pushData.histogramAddress);
    const Ptr output_key_ptr = Ptr::create(pushData.outputKeyAddress);
    const Ptr output_value_ptr = Ptr::create(pushData.outputValueAddress);

    DoublePtrAccessor key_accessor = DoublePtrAccessor::create(
        PtrAccessor::create(input_key_ptr),
        PtrAccessor::create(output_key_ptr)
    );
    DoublePtrAccessor value_accessor = DoublePtrAccessor::create(
        PtrAccessor::create(input_value_ptr),
        PtrAccessor::create(output_value_ptr)
    );
    PtrAccessor histogram_accessor = PtrAccessor::create(histogram_ptr);
    SharedAccessor shared_accessor;
    counter.scatter(
        key_accessor,
        value_accessor,
        histogram_accessor,
        shared_accessor,
        params
    );
}