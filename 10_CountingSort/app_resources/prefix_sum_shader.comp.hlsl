#include "nbl/builtin/hlsl/sort/counting.hlsl"
#include "app_resources/common.hlsl"

[[vk::push_constant]] CountingPushData pushData;

[numthreads(WorkgroupSize,1,1)]
[shader("compute")]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    sort::CountingParameters < uint32_t > params;
    params.dataElementCount = pushData.dataElementCount;
    params.elementsPerWT = pushData.elementsPerWT;
    params.minimum = pushData.minimum;
    params.maximum = pushData.maximum;

    using Counter = sort::counting<WorkgroupSize, BucketCount, PtrAccessor, PtrAccessor, PtrAccessor, SharedAccessor, PtrAccessor::type_t>;
    Counter counter = Counter::create(glsl::gl_WorkGroupID().x);

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