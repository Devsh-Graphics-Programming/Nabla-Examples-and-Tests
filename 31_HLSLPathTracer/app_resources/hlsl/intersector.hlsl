#ifndef _NBL_HLSL_EXT_INTERSECTOR_INCLUDED_
#define _NBL_HLSL_EXT_INTERSECTOR_INCLUDED_

#include "common.hlsl"
#include <nbl/builtin/hlsl/limits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace Intersector
{

// ray query method
// ray query struct holds AS info
// pass in address to vertex/index buffers?

// ray tracing pipeline method

// does everything in traceray in ex 30
template<class Ray>
struct Procedural
{
    using scalar_type = typename Ray::scalar_type;
    using ray_type = Ray;

    static int traceRay(NBL_REF_ARG(ray_type) ray, IIntersection objects[32], int objCount)
    {
        const bool anyHit = ray.intersectionT != numeric_limits<scalar_type>::max;

        int objectID = -1;
        for (int i = 0; i < objCount; i++)
        {
            float t;
            if (objects[i].type == PIT_SPHERE)  // we don't know what type of intersection it is so cast, there has to be a better way to do this
            {
                Intersection<PIT_SPHERE> sphere = (Intersection<PIT_SPHERE>)objects[i];
                t = sphere.intersect(ray.origin, ray.direction);
            }
            // TODO: other types
            
            bool closerIntersection = t > 0.0 && t < ray.intersectionT;

            ray.intersectionT = closerIntersection ? t : ray.intersectionT;
            objectID = closerIntersection ? i : objectID;
            
            // allowing early out results in a performance regression, WTF!?
            //if (anyHit && closerIntersection)
            //break;
        }
        return objectID;
    }

    // TODO? traceray with vertex/index buffer
};

}
}
}
}

#endif