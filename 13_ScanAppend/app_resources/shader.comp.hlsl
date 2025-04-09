#include "common.hlsl"
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

// The AtomicCounterAccessor should have a storage_t and a function that takes a storage_t and returns a storage_t --> add concept
// OutputIndexBits should be less than sizeof(storage_t)*8

template<class AtomicCounterAccessor, uint32_t PrefixSumBits, typename T>
struct scanning_append
{
    using storage_t = typename AtomicCounterAccessor::storage_t;
    using result_t = packed_uint_pair<typename AtomicCounterAccessor::storage_t, PrefixSumBits>;

    result_t operator()(NBL_REF_ARG(AtomicCounterAccessor) accessor, T value)
    {
        storage_t add = value; // _static_cast value
        add &= ((1ull << PrefixSumBits) - 1ull);
        add |= 1ull << PrefixSumBits;
        const storage_t count_reduction = accessor.fetchIncr(add);
        result_t retval = result_t::construct(count_reduction);
        return retval;
    }
};

// TODO: make subgroup version

struct atomic_counter_accessor
{
    using storage_t = uint64_t;

    storage_t fetchIncr(storage_t val)
    {
        nbl::hlsl::bda::__ptr<uint64_t> ptr = nbl::hlsl::bda::__ptr<uint64_t>::create(pushConstants.atomicBDA);
        nbl::hlsl::BdaAccessor<uint64_t> bdaAccessor = nbl::hlsl::BdaAccessor<uint64_t>::create(ptr);
        return bdaAccessor.atomicAdd<uint64_t>(0ull, val);
    }
};

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    if (pushConstants.isAtomicClearDispatch)
    {
        if (ID.x == 0u)
        {
            nbl::hlsl::bda::__ptr<uint64_t> ptr = nbl::hlsl::bda::__ptr<uint64_t>::create(pushConstants.atomicBDA);
            nbl::hlsl::BdaAccessor<uint64_t> bdaAccessor = nbl::hlsl::BdaAccessor<uint64_t>::create(ptr);
            bdaAccessor.set(0ull, 0ull);
        }
        return;
    }

    if (ID.x>=pushConstants.dataElementCount)
        return;
        
    nbl::hlsl::bda::__ptr<uint32_t> inputPtr = nbl::hlsl::bda::__ptr<uint32_t>::create(pushConstants.inputAddress);
    nbl::hlsl::BdaAccessor<uint32_t> inputBDAAccessor = nbl::hlsl::BdaAccessor<uint32_t>::create(inputPtr);
    uint32_t inputVal = inputBDAAccessor.get(ID.x);
    
    atomic_counter_accessor accessor;

    using scan_append_t = scanning_append<atomic_counter_accessor, PrefixSumBits, uint32_t>;
    scan_append_t scan_append;
    typename scan_append_t::result_t ret = scan_append(accessor, inputVal);
    
    nbl::hlsl::bda::__ptr<uint64_t> outputPtr = nbl::hlsl::bda::__ptr<uint64_t>::create(pushConstants.outputAddress);
    nbl::hlsl::BdaAccessor<uint64_t> outputBDAAccessor = nbl::hlsl::BdaAccessor<uint64_t>::create(outputPtr);
    outputBDAAccessor.set(ID.x, ret.value_packed);
}