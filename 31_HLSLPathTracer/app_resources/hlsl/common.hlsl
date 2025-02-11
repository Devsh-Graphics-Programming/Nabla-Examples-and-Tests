#ifndef _NBL_HLSL_EXT_PATHTRACING_COMMON_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACING_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>
#include <nbl/builtin/hlsl/shapes/triangle.hlsl>
#include <nbl/builtin/hlsl/shapes/rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl>
//#include <nbl/builtin/hlsl/shapes/rectangle.hlsl>

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

template<class Spectrum>
struct Light
{
    Spectrum radiance;
    uint32_t objectID;
};

enum ProceduralShapeType : uint16_t
{
    PST_SPHERE,
    PST_TRIANGLE,
    PST_RECTANGLE
};

enum PTPolygonMethod : uint16_t
{
    PPM_AREA,
    PPM_SOLID_ANGLE,
    PPM_APPROX_PROJECTED_SOLID_ANGLE
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

    float deferredPdf(NBL_CONST_REF_ARG(Light light), NBL_CONST_REF_ARG(Ray) ray)
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
        retval.polygonMethod = PPM_SOLID_ANGLE;
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

    float deferredPdf(NBL_CONST_REF_ARG(Light light), NBL_CONST_REF_ARG(Ray<float>) ray)
    {
        const float32_t3 L = ray.direction;
        switch (polygonMethod)
        {
            case PPM_AREA:
            {
                const float dist = ray.intersectionT;
                return dist * dist / nbl::hlsl::abs(nbl::hlsl::dot(getNormalTimesArea()), L);
            }
            break;
            case PPM_SOLID_ANGLE:
            {
                shapes::SphericalTriangle<float> st = shapes::SphericalTriangle<float>::create(vertex0, vertex1, vertex2, ray.origin);
                const float rcpProb = st.solidAngleOfTriangle();
                // if `rcpProb` is NAN then the triangle's solid angle was close to 0.0 
                return rcpProb > numeric_limits<float>::min ? (1.0 / rcpProb) : numeric_limits<float>::max;
            }
            break;
            case PPM_APPROX_PROJECTED_SOLID_ANGLE:
            {
                shapes::SphericalTriangle<float> st = shapes::SphericalTriangle<float>::create(vertex0, vertex1, vertex2, ray.origin);
                const float pdf = st.projectedSolidAngleOfTriangle(ray.normalAtOrigin, ray.wasBSDFAtOrigin, L);
                // if `pdf` is NAN then the triangle's projected solid angle was close to 0.0, if its close to INF then the triangle was very small
                return pdf < numeric_limits<float>::max ? pdf : 0.0;
            }
            break;
            default:
                return 0.0;
        }
    }

    template<class Aniso>
    float32_t3 generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, float32_t3 xi, uint32_t objectID)
    {
        switch(polygonMethod)
        {
            case PPM_AREA:
            {
                const float32_t3 edge0 = vertex1 - vertex0;
                const float32_t3 edge1 = vertex2 - vertex0;
                const float sqrtU = nbl::hlsl::sqrt(xi.x);
                float32_t3 pnt = vertex0 + edge0 * (1.0 - sqrtU) + edge1 * sqrtU * xi.y;
                float32_t3 L = pnt - origin;
                
                const float distanceSq = nbl::hlsl::dot(L,L);
                const float rcpDistance = 1.0 / nbl::hlsl::sqrt(distanceSq);
                L *= rcpDistance;
                
                pdf = distanceSq / nbl::hlsl::abs(nbl::hlsl::dot(nbl::hlsl::cross(edge0, edge1) * 0.5f, L));
                newRayMaxT = 1.0 / rcpDistance;
                return L;
            }
            break;
            case PPM_SOLID_ANGLE:
            {
                float rcpPdf;

                shapes::SphericalTriangle<float> st = shapes::SphericalTriangle<float>::create(vertex0, vertex1, vertex2, ray.origin);
                sampling::SphericalTriangle<float> sst = sampling::SphericalTriangle<float>::create(st);

                const float32_t3 L = sst.generate(rcpPdf, xi.xy);

                pdf = rcpPdf > numeric_limits<float>::min ? (1.0 / rcpPdf) : 0.0;

                const float32_t3 N = getNormalTimesArea();
                newRayMaxT = nbl::hlsl::dot(N, vertex0 - origin) / nbl::hlsl::dot(N, L);
                return L;
            }
            break;
            case PPM_APPROX_PROJECTED_SOLID_ANGLE:
            {
                float rcpPdf;

                shapes::SphericalTriangle<float> st = shapes::SphericalTriangle<float>::create(vertex0, vertex1, vertex2, ray.origin);
                sampling::ProjectedSphericalTriangle<float> sst = sampling::ProjectedSphericalTriangle<float>::create(st);
            
                const float32_t3 L = sst.generate(rcpPdf, interaction.N, isBSDF, xi.xy);

                pdf = rcpPdf > numeric_limits<float>::min ? (1.0 / rcpPdf) : 0.0;

                const float32_t3 N = getNormalTimesArea();
                newRayMaxT = nbl::hlsl::dot(N, vertex0 - origin) / nbl::hlsl::dot(N, L);
                return L;
            }
            break;
            default:
                return (float32_t3)0.0;
        }
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t ObjSize = 10;

    float32_t3 vertex0;
    float32_t3 vertex1;
    float32_t3 vertex2;
    uint32_t bsdfLightIDs;
    PTPolygonMethod polygonMethod;
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
        retval.polygonMethod = PPM_SOLID_ANGLE;
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

    float deferredPdf(NBL_CONST_REF_ARG(Light light), NBL_CONST_REF_ARG(Ray<float>) ray)
    {
        switch (polygonMethod)
        {
            case PPM_AREA:
            {
                const float dist = ray.intersectionT;
                return dist * dist / nbl::hlsl::abs(nbl::hlsl::dot(getNormalTimesArea(), L));
            }
            break;
            // #ifdef TRIANGLE_REFERENCE ?
            case PPM_SOLID_ANGLE:
            {
                float pdf;
                float32_t3x3 rectNormalBasis;
                float32_t2 rectExtents;
                getNormalBasis(rectNormalBasis, rectExtents);
                shapes::SphericalRectangle<float> sphR0 = shapes::SphericalRectangle<float>::create(ray.origin, offset, rectNormalBasis);
                float solidAngle = sphR0.solidAngleOfRectangle(rectExtents);
                if (solidAngle > numeric_limits<float>::min)
                    pdf = 1.f / solidAngle;
                else
                    pdf = numeric_limits<float>::infinity;
                return pdf;
            }
            break;
            case PPM_APPROX_PROJECTED_SOLID_ANGLE:
            {
                return numeric_limits<float>::infinity;
            }
            break;
            default:
                return numeric_limits<float>::infinity;
        }
    }

    template<class Aniso>
    float32_t3 generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, float32_t3 xi, uint32_t objectID)
    {
        const float32_t3 N = getNormalTimesArea();
        const float32_t3 origin2origin = offset - origin;

        switch (polygonMethod)
        {
            case PPM_AREA:
            {
                float32_t3 L = origin2origin + edge0 * xi.x + edge1 * xi.y;
                const float distSq = nbl::hlsl::dot(L, L);
                const float rcpDist = 1.0 / nbl::hlsl::sqrt(distSq);
                L *= rcpDist;
                pdf = distSq / nbl::hlsl::abs(nbl::hlsl::dot(N, L));
                newRayMaxT = 1.0 / rcpDist;
                return L;
            }
            break;
            // #ifdef TRIANGLE_REFERENCE ?
            case PPM_SOLID_ANGLE:
            {
                float pdf;
                float32_t3x3 rectNormalBasis;
                float32_t2 rectExtents;
                getNormalBasis(rectNormalBasis, rectExtents);
                shapes::SphericalRectangle<float> sphR0 = shapes::SphericalRectangle<float>::create(origin, offset, rectNormalBasis);
                float32_t3 L = (float32_t3)0.0;
                float solidAngle = sphR0.solidAngleOfRectangle(rectExtents);

                sampling::SphericalRectangle<float> ssph = sampling::SphericalRectangle<float>::create(sphR0);
                float32_t2 sphUv = ssph.generate(rectExtents, xi.xy, solidAngle);
                if (solidAngle > numeric_limits<float>::min)
                {
                    float32_t3 sph_sample = sphUv[0] * edge0 + sphUv[1] * edge1 + offset;
                    L = nbl::hlsl::normalize(sph_sample - origin);
                    pdf = 1.f / solidAngle;
                }
                else
                    pdf = numeric_limits<float>::infinity;

                newRayMaxT = nbl::hlsl::dot(N, origin2origin) / nbl::hlsl::dot(N, L);
                return L;
            }
            break;
            case PPM_APPROX_PROJECTED_SOLID_ANGLE:
            {
                pdf = numeric_limits<float>::infinity;
                return (float32_t3)0.0;
            }
            break;
            default:
                pdf = numeric_limits<float>::infinity;
                return (float32_t3)0.0;
        }
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t ObjSize = 10;

    float32_t3 offset;
    float32_t3 edge0;
    float32_t3 edge1;
    uint32_t bsdfLightIDs;
    PTPolygonMethod polygonMethod;
};

}
}
}

#endif