#include "nbl/builtin/hlsl/sort/counting.hlsl"
#include "app_resources/common.hlsl"

[[vk::push_constant]] CountingPushData pushData;

using DoublePtrAccessor = DoubleBdaAccessor<uint32_t>;

[numthreads(WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    sort::CountingParameters<uint32_t> params;
    params.dataElementCount = pushData.dataElementCount;
    params.elementsPerWT = pushData.elementsPerWT;
    params.minimum = pushData.minimum;
    params.maximum = pushData.maximum;

    using Counter = sort::counting<WorkgroupSize, BucketCount, DoublePtrAccessor, DoublePtrAccessor, PtrAccessor, SharedAccessor, ArithmeticConfig, PtrAccessor::type_t>;
    Counter counter = Counter::create(glsl::gl_WorkGroupID().x);

    const Ptr input_key_ptr = Ptr::create(pushData.inputKeyAddress);
    const Ptr input_value_ptr = Ptr::create(pushData.inputValueAddress);
    const Ptr histogram_ptr = Ptr::create(pushData.histogramAddress);
    const Ptr output_key_ptr = Ptr::create(pushData.outputKeyAddress);
    const Ptr output_value_ptr = Ptr::create(pushData.outputValueAddress);

    DoublePtrAccessor key_accessor = DoublePtrAccessor::create(
        input_key_ptr,
        output_key_ptr
    );
    DoublePtrAccessor value_accessor = DoublePtrAccessor::create(
        input_value_ptr,
        output_value_ptr
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