#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

// T should be integral
// FirstPartBits should be less than sizeof(T) * 8
template<typename T, uint32_t FirstPartBits=(sizeof(T)*4)>
struct packed_uint_pair
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SecondPartBits = (sizeof(T) * 8u) - FirstPartBits;
    using type_t = T;
    using first_t = nbl::hlsl::conditional_t<FirstPartBits <= 32, uint32_t, uint64_t>;
    using second_t = nbl::hlsl::conditional_t<SecondPartBits <= 32, uint32_t, uint64_t>;

    static packed_uint_pair invalid()
    {
        packed_uint_pair retval;
        retval.value_packed = ~0; // static_cast 0
        return retval;
    }

    static packed_uint_pair construct(T packed_val)
    {
        packed_uint_pair retval;
        retval.value_packed = packed_val;
        return retval;
    }

    static packed_uint_pair construct(first_t first, second_t second)
    {
        packed_uint_pair retval;
        retval.value_packed = nbl::hlsl::glsl::bitfieldInsert<T>(nbl::hlsl::_static_cast<T>(first), nbl::hlsl::_static_cast<T>(second), FirstPartBits, SecondPartBits);
        return retval;
    }

    first_t getFirst() NBL_CONST_MEMBER_FUNC
    {
        return nbl::hlsl::_static_cast<first_t>(value_packed & ((1u << FirstPartBits) - 1u));
        // TODO: DXC Bug?
        // return nbl::hlsl::_static_cast<first_t>(nbl::hlsl::glsl::bitfieldExtract<T>(value_packed, 0ull, FirstPartBits));
    }

    second_t getSecond() NBL_CONST_MEMBER_FUNC
    {
        return nbl::hlsl::_static_cast<second_t>(value_packed >> FirstPartBits);
        // TODO: DXC Bug?
        // return nbl::hlsl::_static_cast<second_t>(nbl::hlsl::glsl::bitfieldExtract<T>(value_packed, FirstPartBits, SecondPartBits));
    }

    T value_packed;
    
    // doesn't produce correct SPIR-V in hlsl 
    //T outputIndex : OutputBits;   
    //T exclusivePrefixSum : (32-OutputBits);
};

NBL_CONSTEXPR_STATIC uint32_t PrefixSumBits = 16u;
using packed_uints = packed_uint_pair<uint64_t, PrefixSumBits>;

//
using input_t = uint32_t;
using output_t = packed_uints;

struct PushConstantData
{
    uint64_t inputAddress;
    uint64_t outputAddress;
    uint64_t atomicBDA;
    uint32_t dataElementCount;
    uint32_t isAtomicClearDispatch;
};

NBL_CONSTEXPR uint32_t ElementCount = 1 << 4u;

NBL_CONSTEXPR uint32_t WorkgroupSize = 256;

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"

// Yes we do have our own re-creation of C++'s STL in HLSL2021 !
#include "nbl/builtin/hlsl/limits.hlsl"