#ifndef _NBL_HLSL_EXT_PATHTRACING_COMMON_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACING_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>
#include <nbl/builtin/hlsl/shapes/triangle.hlsl>
#include <nbl/builtin/hlsl/shapes/rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>

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

enum ProceduralShapeType : uint16_t
{
    PST_SPHERE,
    PST_TRIANGLE,
    PST_RECTANGLE
};

struct ObjectID
{
    static ObjectID create(uint32_t id, uint32_t mode, ProceduralShapeType shapeType)
    {
        ObjectID retval;
        retval.id = id;
        retval.mode = mode;
        retval.shapeType = shapeType;
        return retval;
    }

    uint32_t id;
    uint32_t mode;
    ProceduralShapeType shapeType;
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
    vector3_type normalAtOrigin;
    bool wasBSDFAtOrigin;

    // mutable
    scalar_type intersectionT;
    ObjectID objectID;

    Payload<T> payload;
};

template<class Spectrum>
struct Light
{
    using spectral_type = Spectrum;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t INVALID_ID = 0xffffu;

    static Light<spectral_type> create(NBL_CONST_REF_ARG(spectral_type) radiance, NBL_CONST_REF_ARG(ObjectID) objectID)
    {
        Light<spectral_type> retval;
        retval.radiance = radiance;
        retval.objectID = objectID;
        return retval;
    }

    spectral_type radiance;
    ObjectID objectID;
};

template<class Spectrum>
struct BxDFNode
{
    using spectral_type = Spectrum;
    using params_type = bxdf::SBxDFCreationParams<float, spectral_type>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t INVALID_ID = 0xffffu;

    // for diffuse bxdfs
    static BxDFNode<Spectrum> create(uint32_t materialType, bool isAniso, NBL_CONST_REF_ARG(float32_t2) A, NBL_CONST_REF_ARG(spectral_type) albedo)
    {
        BxDFNode<Spectrum> retval;
        retval.albedo = albedo;
        retval.materialType = materialType;
        retval.params.is_aniso = isAniso;
        retval.params.A = hlsl::max<float32_t2>(A, 1e-4);
        retval.params.ior0 = (spectral_type)1.0;
        retval.params.ior1 = (spectral_type)1.0;
        return retval;
    }

    // for conductor + dielectric
    static BxDFNode<Spectrum> create(uint32_t materialType, bool isAniso, NBL_CONST_REF_ARG(float32_t2) A, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        BxDFNode<Spectrum> retval;
        retval.albedo = (spectral_type)1.0;
        retval.materialType = materialType;
        retval.params.is_aniso = isAniso;
        retval.params.A = hlsl::max<float32_t2>(A, 1e-4);
        retval.params.ior0 = ior0;
        retval.params.ior1 = ior1;
        return retval;
    }

    spectral_type albedo;
    uint32_t materialType;
    params_type params;
};

template<typename T>
struct Tolerance
{
    NBL_CONSTEXPR_STATIC_INLINE float INTERSECTION_ERROR_BOUND_LOG2 = -8.0;

    static T __common(uint32_t depth)
    {
        float depthRcp = 1.0 / float(depth);
        return INTERSECTION_ERROR_BOUND_LOG2;
    }

    static T getStart(uint32_t depth)
    {
        return nbl::hlsl::exp2(__common(depth));
    }

    static T getEnd(uint32_t depth)
    {
        return 1.0 - nbl::hlsl::exp2(__common(depth) + 1.0);
    }
};

enum PTPolygonMethod : uint16_t
{
    PPM_AREA,
    PPM_SOLID_ANGLE,
    PPM_APPROX_PROJECTED_SOLID_ANGLE
};

// namespace Intersector
// {
// // ray query method
// // ray query struct holds AS info
// // pass in address to vertex/index buffers?

// // ray tracing pipeline method

// // procedural data store: [obj count] [intersect type] [obj1] [obj2] [...]

// struct IntersectData
// {
//     enum Mode : uint32_t    // enum class?
//     {
//         RAY_QUERY,
//         RAY_TRACING,
//         PROCEDURAL
//     };

//     NBL_CONSTEXPR_STATIC_INLINE uint32_t DataSize = 128;

//     uint32_t mode : 2;
//     uint32_t unused : 30;   // possible space for flags
//     uint32_t data[DataSize];
// };
// }

enum IntersectMode : uint32_t
{
    IM_RAY_QUERY,
    IM_RAY_TRACING,
    IM_PROCEDURAL
};

namespace NextEventEstimator
{
// procedural data store: [light count] [event type] [obj]

struct Event
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t DataSize = 16;

    uint32_t mode : 2;
    uint32_t unused : 30;   // possible space for flags
    uint32_t data[DataSize];
};
}

template<ProceduralShapeType type>
struct Shape;

template<>
struct Shape<PST_SPHERE>
{
    static Shape<PST_SPHERE> create(NBL_CONST_REF_ARG(float32_t3) position, float32_t radius2, uint32_t bsdfLightIDs)
    {
        Shape<PST_SPHERE> retval;
        retval.position = position;
        retval.radius2 = radius2;
        retval.bsdfLightIDs = bsdfLightIDs;
        return retval;
    }

    static Shape<PST_SPHERE> create(NBL_CONST_REF_ARG(float32_t3) position, float32_t radius, uint32_t bsdfID, uint32_t lightID)
    {
        uint32_t bsdfLightIDs = glsl::bitfieldInsert<uint32_t>(bsdfID, lightID, 16, 16);
        return create(position, radius * radius, bsdfLightIDs);
    }

    // return intersection distance if found, nan otherwise
    float intersect(NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(float32_t3) direction)
    {
        float32_t3 relOrigin = origin - position;
        float relOriginLen2 = hlsl::dot<float32_t3>(relOrigin, relOrigin);

        float dirDotRelOrigin = hlsl::dot<float32_t3>(direction, relOrigin);
        float det = radius2 - relOriginLen2 + dirDotRelOrigin * dirDotRelOrigin;

        // do some speculative math here
        float detsqrt = hlsl::sqrt<float32_t>(det);
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
        float cosThetaMax = hlsl::sqrt<float32_t>(1.0 - radius2 / hlsl::dot<float32_t3>(dist, dist));
        return 2.0 * numbers::pi<float> * (1.0 - cosThetaMax);
    }

    template<typename Ray>
    float deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        return 1.0 / getSolidAngle(ray.origin);
    }

    template<class Aniso>
    float32_t3 generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, float32_t3 xi)
    {
        float32_t3 Z = position - origin;
        const float distanceSQ = hlsl::dot<float32_t3>(Z,Z);
        const float cosThetaMax2 = 1.0 - radius2 / distanceSQ;
        if (cosThetaMax2 > 0.0)
        {
            const float rcpDistance = 1.0 / hlsl::sqrt<float32_t>(distanceSQ);
            Z *= rcpDistance;

            const float cosThetaMax = hlsl::sqrt<float32_t>(cosThetaMax2);
            const float cosTheta = nbl::hlsl::mix<float>(1.0, cosThetaMax, xi.x);

            float32_t3 L = Z * cosTheta;

            const float cosTheta2 = cosTheta * cosTheta;
            const float sinTheta = hlsl::sqrt<float32_t>(1.0 - cosTheta2);
            float sinPhi, cosPhi;
            math::sincos<float>(2.0 * numbers::pi<float> * xi.y - numbers::pi<float>, sinPhi, cosPhi);
            float32_t3 X, Y;
            math::frisvad<float32_t3>(Z, X, Y);

            L += (X * cosPhi + Y * sinPhi) * sinTheta;

            newRayMaxT = (cosTheta - hlsl::sqrt<float32_t>(cosTheta2 - cosThetaMax2)) / rcpDistance;
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
    static Shape<PST_TRIANGLE> create(NBL_CONST_REF_ARG(float32_t3) vertex0, NBL_CONST_REF_ARG(float32_t3) vertex1, NBL_CONST_REF_ARG(float32_t3) vertex2, uint32_t bsdfLightIDs)
    {
        Shape<PST_TRIANGLE> retval;
        retval.vertex0 = vertex0;
        retval.vertex1 = vertex1;
        retval.vertex2 = vertex2;
        retval.bsdfLightIDs = bsdfLightIDs;
        retval.polygonMethod = PPM_SOLID_ANGLE;
        return retval;
    }

    static Shape<PST_TRIANGLE> create(NBL_CONST_REF_ARG(float32_t3) vertex0, NBL_CONST_REF_ARG(float32_t3) vertex1, NBL_CONST_REF_ARG(float32_t3) vertex2, uint32_t bsdfID, uint32_t lightID)
    {
        uint32_t bsdfLightIDs = glsl::bitfieldInsert<uint32_t>(bsdfID, lightID, 16, 16);
        return create(vertex0, vertex1, vertex2, bsdfLightIDs);
    }

    float intersect(NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(float32_t3) direction)
    {
        const float32_t3 edges[2] = { vertex1 - vertex0, vertex2 - vertex0 };

        const float32_t3 h = hlsl::cross<float32_t3>(direction, edges[1]);
        const float a = hlsl::dot<float32_t3>(edges[0], h);

        const float32_t3 relOrigin = origin - vertex0;

        const float u = hlsl::dot<float32_t3>(relOrigin, h) / a;

        const float32_t3 q = hlsl::cross<float32_t3>(relOrigin, edges[0]);
        const float v = hlsl::dot<float32_t3>(direction, q) / a;

        const float t = hlsl::dot<float32_t3>(edges[1], q) / a;

        const bool intersection = t > 0.f && u >= 0.f && v >= 0.f && (u + v) <= 1.f;
        return intersection ? t : bit_cast<float, uint32_t>(numeric_limits<float>::infinity);
    }

    float32_t3 getNormalTimesArea()
    {
        const float32_t3 edges[2] = { vertex1 - vertex0, vertex2 - vertex0 };
        return hlsl::cross<float32_t3>(edges[0], edges[1]) * 0.5f;
    }

    template<typename Ray>
    float deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        const float32_t3 L = ray.direction;
        switch (polygonMethod)
        {
            case PPM_AREA:
            {
                const float dist = ray.intersectionT;
                const float32_t3 L = ray.direction;
                return dist * dist / hlsl::abs<float32_t>(hlsl::dot<float32_t3>(getNormalTimesArea(), L));
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
                sampling::ProjectedSphericalTriangle<float> pst = sampling::ProjectedSphericalTriangle<float>::create(st);
                const float pdf = pst.pdf(ray.normalAtOrigin, ray.wasBSDFAtOrigin, L);
                // if `pdf` is NAN then the triangle's projected solid angle was close to 0.0, if its close to INF then the triangle was very small
                return pdf < numeric_limits<float>::max ? pdf : 0.0;
            }
            break;
            default:
                return 0.0;
        }
    }

    template<class Aniso>
    float32_t3 generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, float32_t3 xi)
    {
        switch(polygonMethod)
        {
            case PPM_AREA:
            {
                const float32_t3 edge0 = vertex1 - vertex0;
                const float32_t3 edge1 = vertex2 - vertex0;
                const float sqrtU = hlsl::sqrt<float32_t>(xi.x);
                float32_t3 pnt = vertex0 + edge0 * (1.0 - sqrtU) + edge1 * sqrtU * xi.y;
                float32_t3 L = pnt - origin;

                const float distanceSq = hlsl::dot<float32_t3>(L,L);
                const float rcpDistance = 1.0 / hlsl::sqrt<float32_t>(distanceSq);
                L *= rcpDistance;

                pdf = distanceSq / hlsl::abs<float32_t>(hlsl::dot<float32_t3>(hlsl::cross<float32_t3>(edge0, edge1) * 0.5f, L));
                newRayMaxT = 1.0 / rcpDistance;
                return L;
            }
            break;
            case PPM_SOLID_ANGLE:
            {
                float rcpPdf;

                shapes::SphericalTriangle<float> st = shapes::SphericalTriangle<float>::create(vertex0, vertex1, vertex2, origin);
                sampling::SphericalTriangle<float> sst = sampling::SphericalTriangle<float>::create(st);

                const float32_t3 L = sst.generate(rcpPdf, xi.xy);

                pdf = rcpPdf > numeric_limits<float>::min ? (1.0 / rcpPdf) : 0.0;

                const float32_t3 N = getNormalTimesArea();
                newRayMaxT = hlsl::dot<float32_t3>(N, vertex0 - origin) / hlsl::dot<float32_t3>(N, L);
                return L;
            }
            break;
            case PPM_APPROX_PROJECTED_SOLID_ANGLE:
            {
                float rcpPdf;

                shapes::SphericalTriangle<float> st = shapes::SphericalTriangle<float>::create(vertex0, vertex1, vertex2, origin);
                sampling::ProjectedSphericalTriangle<float> sst = sampling::ProjectedSphericalTriangle<float>::create(st);

                const float32_t3 L = sst.generate(rcpPdf, interaction.isotropic.N, isBSDF, xi.xy);

                pdf = rcpPdf > numeric_limits<float>::min ? (1.0 / rcpPdf) : 0.0;

                const float32_t3 N = getNormalTimesArea();
                newRayMaxT = hlsl::dot<float32_t3>(N, vertex0 - origin) / hlsl::dot<float32_t3>(N, L);
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
    static Shape<PST_RECTANGLE> create(NBL_CONST_REF_ARG(float32_t3) offset, NBL_CONST_REF_ARG(float32_t3) edge0, NBL_CONST_REF_ARG(float32_t3) edge1, uint32_t bsdfLightIDs)
    {
        Shape<PST_RECTANGLE> retval;
        retval.offset = offset;
        retval.edge0 = edge0;
        retval.edge1 = edge1;
        retval.bsdfLightIDs = bsdfLightIDs;
        retval.polygonMethod = PPM_SOLID_ANGLE;
        return retval;
    }

    static Shape<PST_RECTANGLE> create(NBL_CONST_REF_ARG(float32_t3) offset, NBL_CONST_REF_ARG(float32_t3) edge0, NBL_CONST_REF_ARG(float32_t3) edge1, uint32_t bsdfID, uint32_t lightID)
    {
        uint32_t bsdfLightIDs = glsl::bitfieldInsert<uint32_t>(bsdfID, lightID, 16, 16);
        return create(offset, edge0, edge1, bsdfLightIDs);
    }

    float intersect(NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(float32_t3) direction)
    {
        const float32_t3 h = hlsl::cross<float32_t3>(direction, edge1);
        const float a = hlsl::dot<float32_t3>(edge0, h);

        const float32_t3 relOrigin = origin - offset;

        const float u = hlsl::dot<float32_t3>(relOrigin,h)/a;

        const float32_t3 q = hlsl::cross<float32_t3>(relOrigin, edge0);
        const float v = hlsl::dot<float32_t3>(direction, q) / a;

        const float t = hlsl::dot<float32_t3>(edge1, q) / a;

        const bool intersection = t > 0.f && u >= 0.f && v >= 0.f && u <= 1.f && v <= 1.f;
        return intersection ? t : bit_cast<float, uint32_t>(numeric_limits<float>::infinity);
    }

    float32_t3 getNormalTimesArea()
    {
        return hlsl::cross<float32_t3>(edge0, edge1);
    }

    void getNormalBasis(NBL_REF_ARG(float32_t3x3) basis, NBL_REF_ARG(float32_t2) extents)
    {
        extents = float32_t2(nbl::hlsl::length(edge0), nbl::hlsl::length(edge1));
        basis[0] = edge0 / extents[0];
        basis[1] = edge1 / extents[1];
        basis[2] = normalize(cross(basis[0],basis[1]));

        basis = nbl::hlsl::transpose<float32_t3x3>(basis);    // TODO: double check transpose
    }

    template<typename Ray>
    float deferredPdf(NBL_CONST_REF_ARG(Ray) ray)
    {
        switch (polygonMethod)
        {
            case PPM_AREA:
            {
                const float dist = ray.intersectionT;
                const float32_t3 L = ray.direction;
                return dist * dist / hlsl::abs<float32_t>(hlsl::dot<float32_t3>(getNormalTimesArea(), L));
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
    float32_t3 generate_and_pdf(NBL_REF_ARG(float32_t) pdf, NBL_REF_ARG(float32_t) newRayMaxT, NBL_CONST_REF_ARG(float32_t3) origin, NBL_CONST_REF_ARG(Aniso) interaction, bool isBSDF, float32_t3 xi)
    {
        const float32_t3 N = getNormalTimesArea();
        const float32_t3 origin2origin = offset - origin;

        switch (polygonMethod)
        {
            case PPM_AREA:
            {
                float32_t3 L = origin2origin + edge0 * xi.x + edge1 * xi.y;
                const float distSq = hlsl::dot<float32_t3>(L, L);
                const float rcpDist = 1.0 / hlsl::sqrt<float32_t>(distSq);
                L *= rcpDist;
                pdf = distSq / hlsl::abs<float32_t>(hlsl::dot<float32_t3>(N, L));
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

                newRayMaxT = hlsl::dot<float32_t3>(N, origin2origin) / hlsl::dot<float32_t3>(N, L);
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
