#ifndef _NBL_HLSL_EXT_PATHTRACER_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACER_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace PathTracer
{

template<class RandGen, class RayGen, class Intersector, class MaterialSystem, /* class PathGuider, */ class NextEventEstimator>
struct Unidirectional
{
    using this_t = Unidirectional<RandGen, RayGen, Intersector, MaterialSystem, NextEventEstimator>;

    static this_t create(RandGen randGen,
                        RayGen rayGen,
                        Intersector intersector,
                        MaterialSystem materialSystem,
                        /* PathGuider pathGuider, */
                        NextEventEstimator nee)
    {}

    // closest hit

    // Li
    MaterialSystem::measure_type getMeasure()
    {
        // loop through bounces, do closest hit
        // return ray.payload.accumulation --> color
    }
};

}
}
}
}

#endif