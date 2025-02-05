#ifndef _NBL_HLSL_EXT_PATHTRACING_COMMON_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACING_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ext
{

template<typename T>
struct Payload
{
    using this_t = Payload<T>;
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    vector3_type accumulation;
    scalar_type otherTechniqueHeuristic;
    vector3_type throughput;
    // #ifdef KILL_DIFFUSE_SPECULAR_PATHS
    // bool hasDiffuse;
    // #endif
};

template<typename T>
struct Ray
{
    using this_t = Ray<T>;
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    // immutable
    vector3_type origin;
    vector3_type direction;
    // TODO: polygon method == 2 stuff

    // mutable
    scalar_type intersectionT;
    uint32_t objectID;

    Payload<T> payload;
};

enum PTIntersectionType : uint16_t
{
    PIT_NONE = 0,
    PIT_SPHERE,
    PIT_TRIANGLE,
    PIT_RECTANGLE
};

// TODO: check if this works for ambiguous arrays of Intersection
// unsure if calling correct method
struct IIntersection
{
    PTIntersectionType type = PIT_NONE;
};

template<PTIntersectionType shape>
struct Intersection : IIntersection
{
    PTIntersectionType type = PIT_NONE;
};

template<>
struct Intersection<PIT_SPHERE> : IIntersection
{
    static Intersection<PIT_SPHERE> create(NBL_CONST_REF_ARG(float32_t3) position, float32_t radius, uint32_t bsdfID, uint32_t lightID)
    {
        Intersection<PIT_SPHERE> retval;
        retval.type = PIT_SPHERE;
        retval.position = position;
        retval.radius2 = radius * radius;
        retval.bsdfLightIDs = spirv::bitFieldInsert<uint32_t>(bsdfID, lightID, 16, 16);
        return retval;
    }

    // return intersection distance if found, nan otherwise
    float intersect(NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(float32_t3) direction)
    {
        float32_t3 relOrigin = origin - position;
        float relOriginLen2 = nbl::hlsl::dot(relOrigin, relOrigin);

        float dirDotRelOrigin = nbl::hlsl::dot(direction, relOrigin);
        float det = radius2 - relOriginLen2 + dirDotRelOrigin * dirDotRelOrigin;

        // do some speculative math here
        float detsqrt = nbl::hlsl::sqrt(det);
        return -dirDotRelOrigin + (relOriginLen2 > radius2 ? (-detsqrt) : detsqrt);
    }

    float32_t3 getNormal(NBL_CONST_REF_ARG(float32_t3) hitPosition)
    {
        const float radiusRcp = spirv::inverseSqrt<float32_t>(radius2);
        return (hitPosition - position) * radiusRcp;
    }

    float getSolidAngle(NBL_CONST_REF_ARG(float32_t3) origin)
    {
        float32_t3 dist = position - origin;
        float cosThetaMax = nbl::hlsl::sqrt(1.0 - radius2 / nbl::hlsl::dot(dist, dist));
        return 2.0 * numbers::pi<float> * (1.0 - cosThetaMax);
    }

    // should this be in material system?
    float deferredPdf(Light light, Ray ray)
    {
        return 1.0 / getSolidAngle(ray.origin);
    }

    float generate_and_pdf()
    {
        // TODO
    }

    float32_t3 generate_and

    float32_t3 position;
    float32_t radius2;
    uint32_t bsdfLightIDs;
};

template<>
struct Intersection<PIT_RECTANGLE> : IIntersection
{

};

template<>
struct Intersection<PIT_TRIANGLE> : IIntersection
{

};

}
}
}

#endif