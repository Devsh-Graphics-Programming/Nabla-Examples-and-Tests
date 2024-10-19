#ifndef _NBL_HLSL_FORMAT_SHARED_EXP_HLSL_
#define _NBL_HLSL_FORMAT_SHARED_EXP_HLSL_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"

namespace nbl
{
namespace hlsl
{

namespace format
{

template<typename IntT, uint16_t _Components, uint16_t _ExponentBits>
struct shared_exp
{
    using this_t = shared_exp<IntT,_Components,_ExponentBits>;
    using storage_t = typename make_unsigned<IntT>::type;
//    static_assert(_ExponentBits<16);

    //
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Components = _Components;
    //
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ExponentBits = _ExponentBits;

    // Not even going to consider fp16 and fp64 dependence on device traits
    using decode_t = float32_t;

    storage_t storage;

// private: // lots of fun things because DXC sucks with bugs
    NBL_CONSTEXPR_STATIC_INLINE bool __private_is_signed = is_signed_v<IntT>;
};

// all of this because DXC has bugs in partial template spec
namespace impl
{
template<typename IntT, uint16_t _Components, uint16_t _ExponentBits>
struct numeric_limits_shared_exp
{
    using type = format::shared_exp<IntT,_Components,_ExponentBits>;
    using value_type = typename type::decode_t;
    using __storage_t = typename type::storage_t;

    NBL_CONSTEXPR_STATIC_INLINE bool is_specialized = true;
    NBL_CONSTEXPR_STATIC_INLINE bool is_signed = is_signed_v<IntT>;
    NBL_CONSTEXPR_STATIC_INLINE bool is_integer = false;
    NBL_CONSTEXPR_STATIC_INLINE bool is_exact = false;
    // infinity and NaN are not representable in shared exponent formats
    NBL_CONSTEXPR_STATIC_INLINE bool has_infinity = false;
    NBL_CONSTEXPR_STATIC_INLINE bool has_quiet_NaN = false;
    NBL_CONSTEXPR_STATIC_INLINE bool has_signaling_NaN = false;
    // shared exponent formats have no leading 1 in the mantissa, therefore denormalized values aren't really a concept, although one can argue all values are denorm then?
    NBL_CONSTEXPR_STATIC_INLINE bool has_denorm = false;
    NBL_CONSTEXPR_STATIC_INLINE bool has_denorm_loss = false;
    // truncation
//    NBL_CONSTEXPR_STATIC_INLINE float_round_style round_style = round_to_nearest;
    NBL_CONSTEXPR_STATIC_INLINE bool is_iec559 = false;
    NBL_CONSTEXPR_STATIC_INLINE bool is_bounded = true;
    NBL_CONSTEXPR_STATIC_INLINE bool is_modulo = false;
#if 0
    NBL_CONSTEXPR_STATIC_INLINE int32_t digits = (sizeof(__storage_t)*8-(is_signed ? _Components:0)-_ExponentBits)/_Components;
    NBL_CONSTEXPR_STATIC_INLINE int32_t radix = 2;
    NBL_CONSTEXPR_STATIC_INLINE int32_t max_exponent = 1<<(_ExponentBits-1);
    NBL_CONSTEXPR_STATIC_INLINE int32_t min_exponent = 1-max_exponent;
    NBL_CONSTEXPR_STATIC_INLINE bool traps = false;
    
    // extras
    NBL_CONSTEXPR_STATIC_INLINE __storage_t MantissaMask = ((__storage_t(1))<<digits)-__storage_t(1);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ExponentBits = _ExponentBits;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ExponentMask = uint16_t((1<<_ExponentBits)-1);


 // TODO: functions done as vars
//    NBL_CONSTEXPR_STATIC_INLINE value_type min = base::min();
    NBL_CONSTEXPR_STATIC_INLINE value_type max = asfloat(
        0x477f8000u
//        ((max_exponent+numeric_limits<value_type>::min_exponent)<<(numeric_limits<value_type>::digits-1))|
//        (MantissaMask<<(numeric_limits<value_type>::digits-1-digits))
    );
    NBL_CONSTEXPR_STATIC_INLINE value_type lowest = is_signed ? (-max):value_type(0.f);
/*
    NBL_CONSTEXPR_STATIC_INLINE value_type epsilon = base::epsilon();
    NBL_CONSTEXPR_STATIC_INLINE value_type round_error = base::round_error();
*/
#endif
};
}

}

// specialize the limits
template<typename IntT, uint16_t _Components, uint16_t _ExponentBits>
struct numeric_limits<format::shared_exp<IntT,_Components,_ExponentBits> > : format::impl::numeric_limits_shared_exp<IntT,_Components,_ExponentBits>
{
};

namespace impl
{
// TODO: remove after the `emulated_float` merge
template<typename T, typename U>
struct _static_cast_helper;

// decode
template<typename IntT, uint16_t _Components, uint16_t _ExponentBits>
struct _static_cast_helper<
    vector<typename format::shared_exp<IntT,_Components,_ExponentBits>::decode_t,_Components>,
    format::shared_exp<IntT,_Components,_ExponentBits>
>
{
    using U = format::shared_exp<IntT,_Components,_ExponentBits>;
    using T = vector<typename U::decode_t,_Components>;

    T operator()(U val)
    {
#if 0
        using storage_t = typename U::storage_t;
        using decode_t = typename U::decode_t;
        using limits_t = numeric_limits<U>;

        T retval;
        for (uint16_t i=0; i<_Components; i++)
            retval[i] = decode_t((val.storage>>storage_t(limits_t::digits*i))&limits_t::MantissaMask);
        uint16_t exponent = val.storage>>storage_t(limits_t::digits*3);
        if (U::IsSigned)
        {
            for (uint16_t i=0; i<_Components; i++)
            if (exponent&(uint16_t(1)<<(_ExponentBits+i)))
                retval[i] = -retval[i];
            exponent &= limits_t::ExponentMask;
        }
        return retval*exp2(int32_t(exponent-limits_t::digits)+limits_t::min_exponent);
#endif
        T retval;
        return retval;
    }
};
// encode (WARNING DOES NOT CHECK THAT INPUT IS IN THE RANGE!)
template<typename IntT, uint16_t _Components, uint16_t _ExponentBits>
struct _static_cast_helper<
    format::shared_exp<IntT,_Components,_ExponentBits>,
    vector<typename format::shared_exp<IntT,_Components,_ExponentBits>::decode_t,_Components>
>
{
    using T = format::shared_exp<IntT,_Components,_ExponentBits>;
    using U = vector<typename T::decode_t,_Components>;
#if 0
    //private
    using decode_t = typename T::decode_t;
    using decode_bits_t = unsigned_integer_of_size<sizeof(decode_t)>;

    NBL_CONSTEXPR_STATIC_INLINE int32_t dec_MantissaStoredBits = numeric_limits<decode_t>::digits-1;

    uint16_t extract_biased_exponent(decode_t v)
    {
        if (T::IsSigned)
            v = abs(v);
        return uint16_t(bit_cast<decode_bits_t>(v)>>dec_MantissaStoredBits);
    }
#endif

    T operator()(U val)
    {
#if 0
        using storage_t = typename T::storage_t;
        //
        const decode_bits_t dec_MantissaMask = (decode_bits_t(1)<<dec_MantissaStoredBits)-1;

        // get exponents
        vector<uint16_t,_Components> exponentsDecBias;
        for (uint16_t i=0; i<_Components; i++)
            exponentsDecBias[i] = extract_biased_exponent(val[i]);

        // get the maximum exponent
        uint16_t sharedExponentDecBias = exponentsDecBias[0];
        for (uint16_t i=1; i<_Components; i++)
            sharedExponentDecBias = max(exponentsDecBias[i], sharedExponentDecBias);

        // NOTE: we don't consider clamping against `limits_t::max_exponent`, should be ensured by clamping the inputs against `limits_t::max` before casting!
        using limits_t = numeric_limits<T>;

        // we need to stop "shifting up" implicit leading 1. to farthest left position if the exponent too small
        uint16_t clampedSharedExponentDecBias;
        if (limits_t::float_min_exponent>numeric_limits<decode_t>::float_min_exponent) // if ofc its needed at all
            clampedSharedExponentDecBias = max(sharedExponentDecBias,limits_t::float_min_exponent-numeric_limits<decode_t>::float_min_exponent);
        else
            clampedSharedExponentDecBias = sharedExponentDecBias;

        // we always shift down, the question is how much
        vector<uint16_t,_Components> mantissaShifts;
        for (uint16_t i=0; i<_Components; i++)
            mantissaShifts[i] = max(clampedSharedExponentDecBias-exponentsDecBias[i],numeric_limits<decode_t>::digits);

        // finally lets re-bias our exponent (it will always be positive)
        const uint16_t sharedExponentEncBias = int16_t(clampedSharedExponentDecBias+uint16_t(-numeric_limits<decode_t>::float_min_exponent))+int16_t(limits_t::float_min_exponent);

        //
        T retval;
        retval.storage = storage_t(sharedExponentEncBias)<<(limits_t::digits*3);
        for (uint16_t i=0; i<_Components; i++)
        {
            decode_bits_t origBitPattern = bit_cast<decode_bits_t>(val[i])&dec_MantissaMask;
            // put the implicit 1 in (don't care about denormalized because its probably less than our `limits_t::min` (TODO: static assert it)
            origBitPattern |= decode_bits_t(1)<<dec_MantissaStoredBits;
            // shift and put in the right place
            retval.storage |= storage_t(origBitPattern>>mantissaShifts[i])<<(limits_t::digits*i);
        }
        if (T::IsSigned)
        {
            // doing ops on smaller integers is faster
            decode_bits_t SignMask = 0x1<<(sizeof(decode_t)*8-1);
            decode_bits_t signs = bit_cast<decode_bits_t>(val[0])&SignMask;
            for (uint16_t i=1; i<_Components; i++)
                signs |= (bit_cast<decode_bits_t>(val[i])&SignMask)>>i;
            retval.storage |= storage_t(signs)<<((sizeof(storage_t)-sizeof(decode_t))*8);
        }
#endif
        T retval;
        return retval;
    }
};
}

// TODO: remove after the `emulated_float` merge
namespace format
{
template<typename T, typename U>
T _static_cast(U val)
{
    nbl::hlsl::impl::_static_cast_helper<T,U> fn;
    return fn(val);
}
}

}
}
#endif