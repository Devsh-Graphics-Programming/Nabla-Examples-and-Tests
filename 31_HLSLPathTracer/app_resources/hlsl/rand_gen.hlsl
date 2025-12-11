#ifndef _NBL_HLSL_EXT_RANDGEN_INCLUDED_
#define _NBL_HLSL_EXT_RANDGEN_INCLUDED_

namespace RandGen
{

template<typename RNG>
struct Uniform1D
{
    using rng_type = RNG;
    using return_type = uint32_t;

    static Uniform1D<RNG> construct(uint32_t2 seed)
    {
        Uniform1D<RNG> retval;
        retval.rng = rng_type::construct(seed);
        return retval;
    }

    return_type operator()()
    {
        return rng();
    }

    rng_type rng;
};

template<typename RNG>
struct Uniform3D
{
    using rng_type = RNG;
    using return_type = uint32_t3;

    static Uniform3D<RNG> construct(uint32_t2 seed)
    {
        Uniform3D<RNG> retval;
        retval.rng = rng_type::construct(seed);
        return retval;
    }

    return_type operator()()
    {
        return return_type(rng(), rng(), rng());
    }

    rng_type rng;
};

}

#endif
