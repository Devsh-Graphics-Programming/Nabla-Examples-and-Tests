#ifndef _PATHTRACER_EXAMPLE_RAYGEN_INCLUDED_
#define _PATHTRACER_EXAMPLE_RAYGEN_INCLUDED_

#include <nbl/builtin/hlsl/sampling/box_muller_transform.hlsl>

#include "example_common.hlsl"

namespace RayGen
{

template<class Ray>
struct Basic
{
    using this_t = Basic<Ray>;
    using ray_type = Ray;
    using scalar_type = typename Ray::scalar_type;
    using vector3_type = typename Ray::vector3_type;
    
    using vector2_type = vector<scalar_type, 2>;
    using vector4_type = vector<scalar_type, 4>;
    using matrix4x4_type = matrix<scalar_type, 4, 4>;

    static this_t create(NBL_CONST_REF_ARG(vector2_type) pixOffsetParam, NBL_CONST_REF_ARG(vector3_type) camPos, NBL_CONST_REF_ARG(vector4_type) NDC, NBL_CONST_REF_ARG(matrix4x4_type) invMVP)
    {
        this_t retval;
        retval.pixOffsetParam = pixOffsetParam;
        retval.camPos = camPos;
        retval.NDC = NDC;
        retval.invMVP = invMVP;
        return retval;
    }

    ray_type generate(NBL_CONST_REF_ARG(vector3_type) randVec)
    {
        ray_type ray;
        ray.origin = camPos;

        vector4_type tmp = NDC;
        // apply stochastic reconstruction filter
        const float gaussianFilterCutoff = 2.5;
        const float truncation = nbl::hlsl::exp(-0.5 * gaussianFilterCutoff * gaussianFilterCutoff);
        vector2_type remappedRand = randVec.xy;
        remappedRand.x *= 1.0 - truncation;
        remappedRand.x += truncation;
        nbl::hlsl::sampling::BoxMullerTransform<scalar_type> boxMuller;
        boxMuller.stddev = 1.5;
        tmp.xy += pixOffsetParam * boxMuller(remappedRand);
        // for depth of field we could do another stochastic point-pick
        tmp = nbl::hlsl::mul(invMVP, tmp);
        ray.direction = nbl::hlsl::normalize(tmp.xyz / tmp.w - camPos);

        // #if POLYGON_METHOD==2
        //     ray._immutable.normalAtOrigin = vec3(0.0,0.0,0.0);
        //     ray._immutable.wasBSDFAtOrigin = false;
        // #endif

        ray.payload.accumulation = (vector3_type)0.0;
        ray.payload.otherTechniqueHeuristic = 0.0; // needed for direct eye-light paths
        ray.payload.throughput = (vector3_type)1.0;
        // #ifdef KILL_DIFFUSE_SPECULAR_PATHS
        // ray._payload.hasDiffuse = false;
        // #endif

        return ray;
    }

    vector2_type pixOffsetParam;
    vector3_type camPos;
    vector4_type NDC;
    matrix4x4_type invMVP;
};

}

#endif
