#ifndef _PATHTRACER_EXAMPLE_RANDGEN_INCLUDED_
#define _PATHTRACER_EXAMPLE_RANDGEN_INCLUDED_

namespace RandGen
{

template<typename RNG, uint16_t N>
struct UniformND
{
    using rng_type = RNG;
    using return_type = vector<uint32_t, N>;

    static UniformND<RNG,N> create(uint32_t2 seed)
    {
        UniformND<RNG,N> retval;
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
