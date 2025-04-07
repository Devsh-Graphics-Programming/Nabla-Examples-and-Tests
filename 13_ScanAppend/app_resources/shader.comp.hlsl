#include "common.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

template<typename T, uint32_t FirstPartBits=(sizeof(T)*4)>
struct packed_uint_pair
{
    using first_t = nbl::hlsl::conditional_t<(FirstPartBits <= 32), uint32_t, uint64_t>;
    using second_t = nbl::hlsl::conditional_t<((sizeof(T) * 8u) - FirstPartBits <= 32), uint32_t, uint64_t>;

    static packed_uint_pair invalid()
    {
        packed_uint_pair retval;
        retval.count_reduction_packed = ~0; // static_cast 0
        return retval;
    }

    static packed_uint_pair construct(T packed_val)
    {
        packed_uint_pair retval;
        retval.value_packed = packed_val;
        return retval;
    }

    static packed_uint_pair construct(first_t index, second_t sum)
    {
        packed_uint_pair retval;
        // TODO: ::bitfield insert
        return retval;
    }

    first_t getFirst()
    {
        // ::bitfield extract
    }

    second_t getSecond()
    {
        // ::bitfield extract
    }

    T value_packed;
    
    // doens't produce correct SPIR-V in hlsl 
    //T outputIndex : OutputBits;   
    //T exclusivePrefixSum : (32-OutputBits);
};

// The AtomicCounterAccessor should have a storage_t and a function that takes a storage_t and returns a storage_t --> add concept
// OutputIndexBits should be less than sizeof(storage_t)*8

template<class AtomicCounterAccessor, uint32_t OutputIndexBits, typename T>
struct scanning_append
{
    using storage_t = typename AtomicCounterAccessor::storage_t;
    using result_t = packed_uint_pair<typename AtomicCounterAccessor::storage_t, OutputIndexBits>;

    result_t operator()(NBL_REF_ARG(AtomicCounterAccessor) accessor, T value)
    {
        storage_t add = value; // _static_cast value
        add &= ((1ull << OutputIndexBits) - 1ull);
        add |= 1ull << OutputIndexBits; 
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
        //TODO: Spirv atomicAddI
        return 0ull;
    }
};

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    if (ID.x>=pushConstants.dataElementCount)
        return;
        
    atomic_counter_accessor accessor;

    using scan_append_t = scanning_append<atomic_counter_accessor, 32u, uint32_t>;
    scan_append_t scan_append;
    typename scan_append_t::result_t ret = scan_append(accessor, 1u);
}