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
#include <nbl/builtin/hlsl/bxdf/common.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ext
{

template<typename T>    // TODO make type T Spectrum
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
    PST_NONE = 0,
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

    // polygon method == PPM_APPROX_PROJECTED_SOLID_ANGLE
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

    static Light<spectral_type> create(NBL_CONST_REF_ARG(spectral_type) radiance, uint32_t objId, uint32_t mode, ProceduralShapeType shapeType)
    {
        Light<spectral_type> retval;
        retval.radiance = radiance;
        retval.objectID = ObjectID::create(objId, mode, shapeType);
        return retval;
    }

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

template<typename Scalar, typename Spectrum NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SBxDFCreationParams
{
    bool is_aniso;
    vector<Scalar, 2> A;    // roughness
    Spectrum ior0;          // source ior
    Spectrum ior1;          // destination ior
    Scalar eta;             // in most cases, eta will be calculated from ior0 and ior1; see monochromeEta in pathtracer.hlsl
};

template<class Spectrum>
struct BxDFNode
{
    using spectral_type = Spectrum;
    using params_type = SBxDFCreationParams<float, spectral_type>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t INVALID_ID = 0xffffu;

    // for diffuse bxdfs
    static BxDFNode<Spectrum> create(uint32_t materialType, bool isAniso, NBL_CONST_REF_ARG(float32_t2) A, NBL_CONST_REF_ARG(spectral_type) albedo)
    {
        BxDFNode<Spectrum> retval;
        retval.albedo = albedo;
        retval.materialType = materialType;
        retval.params.is_aniso = isAniso;
        retval.params.A = hlsl::max<float32_t2>(A, (float32_t2)1e-4);
        retval.params.ior0 = (spectral_type)1.0;
        retval.params.ior1 = (spectral_type)1.0;
        return retval;
    }

    // for conductor, ior0 = eta, ior1 = etak
    // for dielectric, eta = ior1/ior0
    static BxDFNode<Spectrum> create(uint32_t materialType, bool isAniso, NBL_CONST_REF_ARG(float32_t2) A, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        BxDFNode<Spectrum> retval;
        retval.albedo = (spectral_type)1.0;
        retval.materialType = materialType;
        retval.params.is_aniso = isAniso;
        retval.params.A = hlsl::max<float32_t2>(A, (float32_t2)1e-4);
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

enum IntersectMode : uint32_t
{
    IM_RAY_QUERY,
    IM_RAY_TRACING,
    IM_PROCEDURAL
};

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
        const float radiusRcp = hlsl::rsqrt<float32_t>(radius2);
        return (hitPosition - position) * radiusRcp;
    }

    float getSolidAngle(NBL_CONST_REF_ARG(float32_t3) origin)
    {
        float32_t3 dist = position - origin;
        float cosThetaMax = hlsl::sqrt<float32_t>(1.0 - radius2 / hlsl::dot<float32_t3>(dist, dist));
        return 2.0 * numbers::pi<float> * (1.0 - cosThetaMax);
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

    NBL_CONSTEXPR_STATIC_INLINE uint32_t ObjSize = 10;

    float32_t3 vertex0;
    float32_t3 vertex1;
    float32_t3 vertex2;
    uint32_t bsdfLightIDs;
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
