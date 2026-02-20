#ifndef _PATHTRACER_EXAMPLE_EXAMPLE_COMMON_INCLUDED_
#define _PATHTRACER_EXAMPLE_EXAMPLE_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/shapes/spherical_triangle.hlsl"
#include "nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl"
#include "nbl/builtin/hlsl/sampling/spherical_triangle.hlsl"
#include "nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl"
#include "nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl"

using namespace nbl;
using namespace hlsl;

enum ProceduralShapeType : uint16_t
{
    PST_NONE = 0,
    PST_SPHERE,
    PST_TRIANGLE,
    PST_RECTANGLE
};

enum IntersectMode : uint32_t
{
    IM_RAY_QUERY,
    IM_RAY_TRACING,
    IM_PROCEDURAL
};

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

enum PTPolygonMethod : uint16_t
{
    PPM_AREA,
    PPM_SOLID_ANGLE,
    PPM_APPROX_PROJECTED_SOLID_ANGLE
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

template<typename T, PTPolygonMethod PPM>
struct Ray
{
    using this_t = Ray<T,PPM>;
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    // immutable
    vector3_type origin;
    vector3_type direction;

    // mutable
    scalar_type intersectionT;
    ObjectID objectID;

    Payload<T> payload;

    void initData(const vector3_type _origin, const vector3_type _direction, const vector3_type _normalAtOrigin, bool _wasBSDFAtOrigin)
    {
        origin = _origin;
        direction = _direction;
    }

    void initPayload()
    {
        payload.accumulation = hlsl::promote<vector3_type>(0.0);
        payload.otherTechniqueHeuristic = scalar_type(0.0); // needed for direct eye-light paths
        payload.throughput = hlsl::promote<vector3_type>(1.0);
    }

    vector3_type foundEmissiveMIS(scalar_type pdfSq)
    {
        return payload.throughput / (scalar_type(1.0) + pdfSq * payload.otherTechniqueHeuristic);
    }

    void addPayloadContribution(const vector3_type contribution)
    {
        payload.accumulation += contribution;
    }

    void setPayloadMISWeights(const vector3_type throughput, const scalar_type otherTechniqueHeuristic)
    {
        payload.throughput = throughput;
        payload.otherTechniqueHeuristic = otherTechniqueHeuristic;
    }

    vector3_type getPayloadThroughput() NBL_CONST_MEMBER_FUNC { return payload.throughput; }
};

template<typename T>
struct Ray<T, PPM_APPROX_PROJECTED_SOLID_ANGLE>
{
    using this_t = Ray<T,PPM_APPROX_PROJECTED_SOLID_ANGLE>;
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    // immutable
    vector3_type origin;
    vector3_type direction;

    vector3_type normalAtOrigin;
    bool wasBSDFAtOrigin;

    // mutable
    scalar_type intersectionT;
    ObjectID objectID;

    Payload<T> payload;

    void initData(const vector3_type _origin, const vector3_type _direction, const vector3_type _normalAtOrigin, bool _wasBSDFAtOrigin)
    {
        origin = _origin;
        direction = _direction;
        normalAtOrigin = _normalAtOrigin;
        wasBSDFAtOrigin = _wasBSDFAtOrigin;
    }

    void initPayload()
    {
        payload.accumulation = hlsl::promote<vector3_type>(0.0);
        payload.otherTechniqueHeuristic = scalar_type(0.0); // needed for direct eye-light paths
        payload.throughput = hlsl::promote<vector3_type>(1.0);
    }

    vector3_type foundEmissiveMIS(scalar_type pdfSq)
    {
        return payload.throughput / (scalar_type(1.0) + pdfSq * payload.otherTechniqueHeuristic);
    }

    void addPayloadContribution(const vector3_type contribution)
    {
        payload.accumulation += contribution;
    }

    void setPayloadMISWeights(const vector3_type throughput, const scalar_type otherTechniqueHeuristic)
    {
        payload.throughput = throughput;
        payload.otherTechniqueHeuristic = otherTechniqueHeuristic;
    }

    vector3_type getPayloadThroughput() NBL_CONST_MEMBER_FUNC { return payload.throughput; }
};

template<class Spectrum>
struct Light
{
    using spectral_type = Spectrum;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t INVALID_ID = 0xffffu;

    static Light<spectral_type> create(uint32_t emissiveMatID, uint32_t objId, uint32_t mode, ProceduralShapeType shapeType)
    {
        Light<spectral_type> retval;
        retval.emissiveMatID = emissiveMatID;
        retval.objectID = ObjectID::create(objId, mode, shapeType);
        return retval;
    }

    static Light<spectral_type> create(uint32_t emissiveMatID, NBL_CONST_REF_ARG(ObjectID) objectID)
    {
        Light<spectral_type> retval;
        retval.emissiveMatID = emissiveMatID;
        retval.objectID = objectID;
        return retval;
    }

    uint32_t emissiveMatID;
    ObjectID objectID;
};

template<typename T>
struct Tolerance
{
    NBL_CONSTEXPR_STATIC_INLINE T INTERSECTION_ERROR_BOUND_LOG2 = -8.0;

    static T __common(uint32_t depth)
    {
        T depthRcp = 1.0 / T(depth);
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

enum MaterialType : uint32_t    // enum class?
{
    DIFFUSE = 0u,
    CONDUCTOR,
    DIELECTRIC,
    IRIDESCENT_CONDUCTOR,
    IRIDESCENT_DIELECTRIC,
    EMISSIVE
};

template<typename Scalar, typename Spectrum NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SBxDFCreationParams
{
    bool is_aniso;
    vector<Scalar, 2> A;    // roughness
    Spectrum ior0;          // source ior
    Spectrum ior1;          // destination ior
    Spectrum iork;          // destination iork (for iridescent only)
    Scalar eta;             // in most cases, eta will be calculated from ior0 and ior1; see monochromeEta in path_tracing/unidirectional.hlsl
};

template<class Spectrum>
struct BxDFNode
{
    using spectral_type = Spectrum;
    using scalar_type = typename vector_traits<Spectrum>::scalar_type;
    using vector2_type = vector<scalar_type, 2>;
    using params_type = SBxDFCreationParams<scalar_type, spectral_type>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t INVALID_ID = 0xffffu;

    // for diffuse bxdfs
    static BxDFNode<Spectrum> create(uint32_t materialType, bool isAniso, NBL_CONST_REF_ARG(vector2_type) A, NBL_CONST_REF_ARG(spectral_type) albedo)
    {
        BxDFNode<Spectrum> retval;
        retval.albedo = albedo;
        retval.materialType = materialType;
        retval.params.is_aniso = isAniso;
        retval.params.A = hlsl::max(A, hlsl::promote<vector2_type>(1e-3));
        retval.params.ior0 = hlsl::promote<spectral_type>(1.0);
        retval.params.ior1 = hlsl::promote<spectral_type>(1.0);
        return retval;
    }

    // for conductor, ior0 = eta, ior1 = etak
    // for dielectric, eta = ior1/ior0
    static BxDFNode<Spectrum> create(uint32_t materialType, bool isAniso, NBL_CONST_REF_ARG(vector2_type) A, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        BxDFNode<Spectrum> retval;
        retval.albedo = hlsl::promote<spectral_type>(1.0);
        retval.materialType = materialType;
        retval.params.is_aniso = isAniso;
        retval.params.A = hlsl::max(A, hlsl::promote<vector2_type>(1e-3));
        retval.params.ior0 = ior0;
        retval.params.ior1 = ior1;
        return retval;
    }

    // for iridescent bxdfs, ior0 = thin film ior, ior1+iork1 = base mat ior (k for conductor base)
    static BxDFNode<Spectrum> create(uint32_t materialType, bool isAniso, scalar_type A, scalar_type Dinc, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1, NBL_CONST_REF_ARG(spectral_type) iork1)
    {
        BxDFNode<Spectrum> retval;
        retval.albedo = hlsl::promote<spectral_type>(1.0);
        retval.materialType = materialType;
        retval.params.is_aniso = isAniso;
        retval.params.A = vector2_type(hlsl::max(A, 1e-3), Dinc);
        retval.params.ior0 = ior0;
        retval.params.ior1 = ior1;
        retval.params.iork = iork1;
        return retval;
    }

    // for emissive materials
    static BxDFNode<Spectrum> create(uint32_t materialType, NBL_CONST_REF_ARG(spectral_type) radiance)
    {
        BxDFNode<Spectrum> retval;
        retval.albedo = radiance;
        retval.materialType = materialType;
        return retval;
    }

    scalar_type getNEEProb()
    {
        const scalar_type alpha = materialType != MaterialType::DIFFUSE ? params.A[0] : 1.0;
        return hlsl::min(8.0 * alpha, 1.0);
    }

    spectral_type albedo;   // also stores radiance for emissive
    uint32_t materialType;
    params_type params;
};


template<typename T, ProceduralShapeType type>
struct Shape;

template<typename T>
struct Shape<T, PST_SPHERE>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    static Shape<T, PST_SPHERE> create(NBL_CONST_REF_ARG(vector3_type) position, float32_t radius2, uint32_t bsdfLightIDs)
    {
        Shape<T, PST_SPHERE> retval;
        retval.position = position;
        retval.radius2 = radius2;
        retval.bsdfLightIDs = bsdfLightIDs;
        return retval;
    }

    static Shape<T, PST_SPHERE> create(NBL_CONST_REF_ARG(vector3_type) position, scalar_type radius, uint32_t bsdfID, uint32_t lightID)
    {
        uint32_t bsdfLightIDs = glsl::bitfieldInsert<uint32_t>(bsdfID, lightID, 16, 16);
        return create(position, radius * radius, bsdfLightIDs);
    }

    void updateTransform(NBL_CONST_REF_ARG(float32_t3x4) m)
    {
        position = float3(m[0].w, m[1].w, m[2].w);
        radius2 = m[0].x * m[0].x;
    }

    // return intersection distance if found, nan otherwise
    scalar_type intersect(NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(vector3_type) direction)
    {
        vector3_type relOrigin = origin - position;
        scalar_type relOriginLen2 = hlsl::dot(relOrigin, relOrigin);

        scalar_type dirDotRelOrigin = hlsl::dot(direction, relOrigin);
        scalar_type det = radius2 - relOriginLen2 + dirDotRelOrigin * dirDotRelOrigin;

        // do some speculative math here
        scalar_type detsqrt = hlsl::sqrt(det);
        return -dirDotRelOrigin + (relOriginLen2 > radius2 ? (-detsqrt) : detsqrt);
    }

    vector3_type getNormal(NBL_CONST_REF_ARG(vector3_type) hitPosition)
    {
        const scalar_type radiusRcp = hlsl::rsqrt(radius2);
        return (hitPosition - position) * radiusRcp;
    }

    scalar_type getSolidAngle(NBL_CONST_REF_ARG(vector3_type) origin)
    {
        vector3_type dist = position - origin;
        scalar_type cosThetaMax = hlsl::sqrt(1.0 - radius2 / hlsl::dot(dist, dist));
        return 2.0 * numbers::pi<scalar_type> * (1.0 - cosThetaMax);
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t ObjSize = 5;

    vector3_type position;
    float32_t radius2;
    uint32_t bsdfLightIDs;
};

template<typename T>
struct Shape<T, PST_TRIANGLE>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    static Shape<T, PST_TRIANGLE> create(NBL_CONST_REF_ARG(vector3_type) vertex0, NBL_CONST_REF_ARG(vector3_type) vertex1, NBL_CONST_REF_ARG(vector3_type) vertex2, uint32_t bsdfLightIDs)
    {
        Shape<T, PST_TRIANGLE> retval;
        retval.vertex0 = vertex0;
        retval.vertex1 = vertex1;
        retval.vertex2 = vertex2;
        retval.bsdfLightIDs = bsdfLightIDs;
        return retval;
    }

    static Shape<T, PST_TRIANGLE> create(NBL_CONST_REF_ARG(vector3_type) vertex0, NBL_CONST_REF_ARG(vector3_type) vertex1, NBL_CONST_REF_ARG(vector3_type) vertex2, uint32_t bsdfID, uint32_t lightID)
    {
        uint32_t bsdfLightIDs = glsl::bitfieldInsert<uint32_t>(bsdfID, lightID, 16, 16);
        return create(vertex0, vertex1, vertex2, bsdfLightIDs);
    }

    void updateTransform(NBL_CONST_REF_ARG(float32_t3x4) m)
    {
        // Define triangle in local space
        float3 localVertex0 = float3(0.0, 0.0, 0.0);
        float3 localVertex1 = float3(1.0, 0.0, 0.0);
        float3 localVertex2 = float3(0.0, 1.0, 0.0);
        
        // Transform each vertex
        vertex0 = mul(m, float4(localVertex0, 1.0)).xyz;
        vertex1 = mul(m, float4(localVertex1, 1.0)).xyz;
        vertex2 = mul(m, float4(localVertex2, 1.0)).xyz;
    }

    scalar_type intersect(NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(vector3_type) direction)
    {
        const vector3_type edges[2] = { vertex1 - vertex0, vertex2 - vertex0 };

        const vector3_type h = hlsl::cross<vector3_type>(direction, edges[1]);
        const scalar_type a = hlsl::dot<vector3_type>(edges[0], h);

        const vector3_type relOrigin = origin - vertex0;

        const scalar_type u = hlsl::dot<vector3_type>(relOrigin, h) / a;

        const vector3_type q = hlsl::cross<vector3_type>(relOrigin, edges[0]);
        const scalar_type v = hlsl::dot<vector3_type>(direction, q) / a;

        const scalar_type t = hlsl::dot<vector3_type>(edges[1], q) / a;

        const bool intersection = t > 0.f && u >= 0.f && v >= 0.f && (u + v) <= 1.f;
        return intersection ? t : bit_cast<scalar_type, uint32_t>(numeric_limits<scalar_type>::infinity);
    }

    vector3_type getNormalTimesArea()
    {
        const vector3_type edges[2] = { vertex1 - vertex0, vertex2 - vertex0 };
        return hlsl::cross<vector3_type>(edges[0], edges[1]) * 0.5f;
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t ObjSize = 10;

    vector3_type vertex0;
    vector3_type vertex1;
    vector3_type vertex2;
    uint32_t bsdfLightIDs;
};

template<typename T>
struct Shape<T, PST_RECTANGLE>
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    static Shape<T, PST_RECTANGLE> create(NBL_CONST_REF_ARG(vector3_type) offset, NBL_CONST_REF_ARG(vector3_type) edge0, NBL_CONST_REF_ARG(vector3_type) edge1, uint32_t bsdfLightIDs)
    {
        Shape<T, PST_RECTANGLE> retval;
        retval.offset = offset;
        retval.edge0 = edge0;
        retval.edge1 = edge1;
        retval.bsdfLightIDs = bsdfLightIDs;
        return retval;
    }

    static Shape<T, PST_RECTANGLE> create(NBL_CONST_REF_ARG(vector3_type) offset, NBL_CONST_REF_ARG(vector3_type) edge0, NBL_CONST_REF_ARG(vector3_type) edge1, uint32_t bsdfID, uint32_t lightID)
    {
        uint32_t bsdfLightIDs = glsl::bitfieldInsert<uint32_t>(bsdfID, lightID, 16, 16);
        return create(offset, edge0, edge1, bsdfLightIDs);
    }

    void updateTransform(NBL_CONST_REF_ARG(float32_t3x4) m)
    {
        // Define rectangle in local space
        float3 localVertex0 = float3(0.0, 0.0, 0.0);
        float3 localVertex1 = float3(1.0, 0.0, 0.0);
        float3 localVertex2 = float3(0.0, 1.0, 0.0);

        // Transform each vertex
        float3 vertex0 = mul(m, float4(localVertex0, 1.0)).xyz;
        float3 vertex1 = mul(m, float4(localVertex1, 1.0)).xyz;
        float3 vertex2 = mul(m, float4(localVertex2, 1.0)).xyz;

        // Extract offset and edges from transformed vertices
        offset = vertex0;
        edge0 = vertex1 - vertex0;
        edge1 = vertex2 - vertex0;
    }

    scalar_type intersect(NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(vector3_type) direction)
    {
        const vector3_type h = hlsl::cross<vector3_type>(direction, edge1);
        const scalar_type a = hlsl::dot<vector3_type>(edge0, h);

        const vector3_type relOrigin = origin - offset;

        const scalar_type u = hlsl::dot<vector3_type>(relOrigin,h)/a;

        const vector3_type q = hlsl::cross<vector3_type>(relOrigin, edge0);
        const scalar_type v = hlsl::dot<vector3_type>(direction, q) / a;

        const scalar_type t = hlsl::dot<vector3_type>(edge1, q) / a;

        const bool intersection = t > 0.f && u >= 0.f && v >= 0.f && u <= 1.f && v <= 1.f;
        return intersection ? t : bit_cast<scalar_type, uint32_t>(numeric_limits<scalar_type>::infinity);
    }

    vector3_type getNormalTimesArea()
    {
        return hlsl::cross<vector3_type>(edge0, edge1);
    }

    void getNormalBasis(NBL_REF_ARG(matrix<scalar_type, 3, 3>) basis, NBL_REF_ARG(vector<scalar_type, 2>) extents)
    {
        extents = vector<scalar_type, 2>(nbl::hlsl::length(edge0), nbl::hlsl::length(edge1));
        basis[0] = edge0 / extents[0];
        basis[1] = edge1 / extents[1];
        basis[2] = nbl::hlsl::normalize(nbl::hlsl::cross(basis[0],basis[1]));
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t ObjSize = 10;

    vector3_type offset;
    vector3_type edge0;
    vector3_type edge1;
    uint32_t bsdfLightIDs;
};

#endif
