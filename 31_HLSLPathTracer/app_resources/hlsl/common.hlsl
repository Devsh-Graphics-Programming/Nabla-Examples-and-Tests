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

enum ProceduralShapeType : uint16_t
{
    PST_SPHERE,
    PST_TRIANGLE,
    PST_RECTANGLE
};

template<ProceduralShapeType type>
struct Shape;

template<>
struct Shape<PST_SPHERE>
{
    static Shape<PST_SPHERE> create(NBL_CONST_REF_ARG(float32_t3) position, float32_t radius, uint32_t bsdfID, uint32_t lightID)
    {
        Shape<PST_SPHERE> retval;
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

    float deferredPdf(Light light, Ray ray)
    {
        return 1.0 / getSolidAngle(ray.origin);
    }

    template<class Aniso>
    float generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, float32_t3 xi, uint32_t objectID)
    {
        float32_t3 Z = position - origin;
        const float distanceSQ = nbl::hlsl::dot(Z,Z);
        const float cosThetaMax2 = 1.0 - radius2 / distanceSQ;
        if (cosThetaMax2 > 0.0)
        {
            const float rcpDistance = 1.0 / nbl::hlsl::sqrt(distanceSQ);
            Z *= rcpDistance;
        
            const float cosThetaMax = nbl::hlsl::sqrt(cosThetaMax2);
            const float cosTheta = nbl::hlsl::mix(1.0, cosThetaMax, xi.x);

            vec3 L = Z * cosTheta;

            const float cosTheta2 = cosTheta * cosTheta;
            const float sinTheta = nbl::hlsl::sqrt(1.0 - cosTheta2);
            float sinPhi, cosPhi;
            math::sincos(2.0 * numbers::pi<float> * xi.y - numbers::pi<float>, sinPhi, cosPhi);
            float32_t2x3 XY = math::frisvad<float>(Z);
        
            L += (XY[0] * cosPhi + XY[1] * sinPhi) * sinTheta;
        
            newRayMaxT = (cosTheta - nbl::hlsl::sqrt(cosTheta2 - cosThetaMax2)) / rcpDistance;
            pdf = 1.0 / (2.0 * numbers::pi<float> * (1.0 - cosThetaMax));
            return L;
        }
        pdf = 0.0;
        return float32_t3(0.0,0.0,0.0);
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t ObjSize = 5;

    float32_t3 position;
    float32_t radius2;
    uint32_t bsdfLightIDs;
};

template<>
struct Shape<PST_TRIANGLE>
{
    static Shape<PST_TRIANGLE> create(NBL_CONST_REF_ARG(float32_t3) vertex0, NBL_CONST_REF_ARG(float32_t3) vertex1, NBL_CONST_REF_ARG(float32_t3) vertex2, uint32_t bsdfID, uint32_t lightID)
    {
        Shape<PST_TRIANGLE> retval;
        retval.vertex0 = vertex0;
        retval.vertex1 = vertex1;
        retval.vertex2 = vertex2;
        retval.bsdfLightIDs = spirv::bitFieldInsert<uint32_t>(bsdfID, lightID, 16, 16);
        return retval;
    }

    float intersect(NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(float32_t3) direction)
    {
        const float32_t3 edges[2] = { vertex1 - vertex0, vertex2 - vertex0 };

        const float32_t3 h = nbl::hlsl::cross(direction, edges[1]);
        const float a = nbl::hlsl::dot(edges[0], h);

        const float32_t3 relOrigin = origin - vertex0;

        const float u = nbl::hlsl::dot(relOrigin, h) / a;

        const float32_t3 q = nbl::hlsl::cross(relOrigin, edges[0]);
        const float v = nbl::hlsl::dot(direction, q) / a;

        const float t = nbl::hlsl::dot(edges[1], q) / a;

        const bool intersection = t > 0.f && u >= 0.f && v >= 0.f && (u + v) <= 1.f;
        return intersection ? t : numeric_limits<float>::infinity;
    }

    float32_t3 getNormalTimesArea()
    {
        const float32_t3 edges[2] = { vertex1 - vertex0, vertex2 - vertex0 };
        return nbl::hlsl::cross(edges[0], edges[1]) * 0.5f;
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t ObjSize = 10;

    float32_t3 vertex0;
    float32_t3 vertex1;
    float32_t3 vertex2;
    uint32_t bsdfLightIDs;
};

template<>
struct Shape<PST_RECTANGLE>
{
    static Shape<PST_TRIANGLE> create(NBL_CONST_REF_ARG(float32_t3) offset, NBL_CONST_REF_ARG(float32_t3) edge0, NBL_CONST_REF_ARG(float32_t3) edge1, uint32_t bsdfID, uint32_t lightID)
    {
        Shape<PST_TRIANGLE> retval;
        retval.offset = offset;
        retval.edge0 = edge0;
        retval.edge1 = edge1;
        retval.bsdfLightIDs = spirv::bitFieldInsert<uint32_t>(bsdfID, lightID, 16, 16);
        return retval;
    }

    float intersect(NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(float32_t3) direction)
    {
        const float32_t3 h = nbl::hlsl::cross(direction, edge1);
        const float a = nbl::hlsl::dot(edge0, h);

        const float32_t3 relOrigin = origin - offset;

        const float u = nbl::hlsl::dot(relOrigin,h)/a;

        const float32_t3 q = nbl::hlsl::cross(relOrigin, edge0);
        const float v = nbl::hlsl::dot(direction, q) / a;

        const float t = nbl::hlsl::dot(edge1, q) / a;

        const bool intersection = t > 0.f && u >= 0.f && v >= 0.f && u <= 1.f && v <= 1.f;
        return intersection ? t : numeric_limits<float>::infinity;
    }

    float32_t3 getNormalTimesArea()
    {
        return nbl::hlsl::cross(edge0, edge1);
    }

    void getNormalBasis(NBL_REF_ARG(float32_t3x3) basis, NBL_REF_ARG(float32_t2) extents)
    {
        extents = float32_t2(nbl::hlsl::length(edge0), nbl::hlsl::length(edge1));
        basis[0] = edge0 / extents[0];
        basis[1] = edge1 / extents[1];
        basis[2] = normalize(cross(basis[0],basis[1]));

        basis = nbl::hlsl::transpose<matrix3x3_type>(basis);    // TODO: double check transpose
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t ObjSize = 10;

    float32_t3 offset;
    float32_t3 edge0;
    float32_t3 edge1;
    uint32_t bsdfLightIDs;
};

}
}
}

#endif