#ifndef _NBL_HLSL_EXT_RANDGEN_INCLUDED_
#define _NBL_HLSL_EXT_RANDGEN_INCLUDED_

namespace RandGen
{

template<typename RNG>
struct Uniform3D
{
    using rng_type = RNG;

    static Uniform3D<RNG> create(uint32_t2 seed)
    {
        Uniform3D<RNG> retval;
        retval.rng = rng_type::construct(seed);
        return retval;
    }

    uint32_t3 operator()()
    {
        return uint32_t3(rng(), rng(), rng());
    }

    rng_type rng;
};

}

#endif
