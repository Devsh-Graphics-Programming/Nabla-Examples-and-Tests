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

template<typename T>
struct Payload
{
    using this_t = Payload<T>;
    using scalar_type = T;
    using spectral_type = vector<T, 3>;

    spectral_type accumulation;
    scalar_type otherTechniqueHeuristic;
    spectral_type throughput;
    // #ifdef KILL_DIFFUSE_SPECULAR_PATHS
    // bool hasDiffuse;
    // #endif
};

enum NEEPolygonMethod : uint16_t
{
    PPM_AREA,
    PPM_SOLID_ANGLE,
    PPM_APPROX_PROJECTED_SOLID_ANGLE
};

struct ObjectID
{
    static ObjectID create(uint16_t id, ProceduralShapeType shapeType)
    {
        ObjectID retval;
        retval.id = id;
        retval.shapeType = shapeType;
        return retval;
    }

    NBL_CONSTEXPR_STATIC_INLINE uint16_t INVALID_ID = 0x3fffu;

    uint16_t id : 14u;
    ProceduralShapeType shapeType : 2u;
};

struct LightID
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t INVALID_ID = 0xffffu;

    uint16_t id;
};

struct MaterialID
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t INVALID_ID = 0xffffu;

    uint16_t id;
};

template<typename Payload, NEEPolygonMethod PPM>
struct Ray
{
    using this_t = Ray<Payload,PPM>;
    using payload_type = Payload;
    using scalar_type = typename payload_type::scalar_type;
    using vector3_type = vector<scalar_type, 3>;

    // immutable
    vector3_type origin;
    vector3_type direction;

    // mutable
    scalar_type intersectionT;

    payload_type payload;
    using spectral_type = typename payload_type::spectral_type;

    void init(const vector3_type _origin, const vector3_type _direction)
    {
        origin = _origin;
        direction = _direction;
    }

    template<class Interaction>
    void initInteraction(NBL_CONST_REF_ARG(Interaction) interaction)
    {
        // empty, only for projected solid angle
    }

    void initPayload()
    {
        payload.accumulation = hlsl::promote<spectral_type>(0.0);
        payload.otherTechniqueHeuristic = scalar_type(0.0); // needed for direct eye-light paths
        payload.throughput = hlsl::promote<spectral_type>(1.0);
    }

    spectral_type foundEmissiveMIS(scalar_type pdfSq)
    {
        return hlsl::mix(hlsl::promote<spectral_type>(1.0), payload.throughput / (scalar_type(1.0) + pdfSq * payload.otherTechniqueHeuristic),
            payload.otherTechniqueHeuristic > numeric_limits<scalar_type>::min);
    }

    void addPayloadContribution(const spectral_type contribution)
    {
        payload.accumulation += contribution;
    }
    spectral_type getPayloadAccumulatiion() { return payload.accumulation; }

    void setPayloadMISWeights(const spectral_type throughput, const scalar_type otherTechniqueHeuristic)
    {
        payload.throughput = throughput;
        payload.otherTechniqueHeuristic = otherTechniqueHeuristic;
    }

    void setT(scalar_type t) { intersectionT = t; }
    scalar_type getT() NBL_CONST_MEMBER_FUNC { return intersectionT; }

    spectral_type getPayloadThroughput() NBL_CONST_MEMBER_FUNC { return payload.throughput; }
};

template<typename Payload>
struct Ray<Payload, PPM_APPROX_PROJECTED_SOLID_ANGLE>
{
    using this_t = Ray<Payload,PPM_APPROX_PROJECTED_SOLID_ANGLE>;
    using payload_type = Payload;
    using scalar_type = typename payload_type::scalar_type;
    using vector3_type = vector<scalar_type, 3>;

    // immutable
    vector3_type origin;
    vector3_type direction;

    vector3_type normalAtOrigin;
    bool wasBSDFAtOrigin;

    // mutable
    scalar_type intersectionT;

    payload_type payload;
    using spectral_type = typename payload_type::spectral_type;

    void init(const vector3_type _origin, const vector3_type _direction)
    {
        origin = _origin;
        direction = _direction;
    }

    template<class Interaction>
    void initInteraction(NBL_CONST_REF_ARG(Interaction) interaction)
    {
        normalAtOrigin = interaction.getN();
        wasBSDFAtOrigin = interaction.isMaterialBSDF();
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
    vector3_type getPayloadAccumulatiion() { return payload.accumulation; }

    void setPayloadMISWeights(const vector3_type throughput, const scalar_type otherTechniqueHeuristic)
    {
        payload.throughput = throughput;
        payload.otherTechniqueHeuristic = otherTechniqueHeuristic;
    }

    void setT(scalar_type t) { intersectionT = t; }
    scalar_type getT() NBL_CONST_MEMBER_FUNC { return intersectionT; }

    vector3_type getPayloadThroughput() NBL_CONST_MEMBER_FUNC { return payload.throughput; }
};

template<class Spectrum>
struct Light
{
    using spectral_type = Spectrum;

    static Light<spectral_type> create(uint32_t emissiveMatID, uint32_t objId, ProceduralShapeType shapeType)
    {
        Light<spectral_type> retval;
        retval.emissiveMatID.id = uint16_t(emissiveMatID);
        retval.objectID = ObjectID::create(uint16_t(objId), shapeType);
        return retval;
    }

    static Light<spectral_type> create(uint32_t emissiveMatID, NBL_CONST_REF_ARG(ObjectID) objectID)
    {
        Light<spectral_type> retval;
        retval.emissiveMatID.id = uint16_t(emissiveMatID);
        retval.objectID = objectID;
        return retval;
    }

    MaterialID emissiveMatID;
    ObjectID objectID;
};

template<typename T>
struct Tolerance
{
    NBL_CONSTEXPR_STATIC_INLINE T INTERSECTION_ERROR_BOUND_LOG2 = -8.0;

    static T __common(uint16_t depth)
    {
        T depthRcp = 1.0 / T(depth);
        return INTERSECTION_ERROR_BOUND_LOG2;
    }

    static T getStart(uint16_t depth)
    {
        return nbl::hlsl::exp2(__common(depth));
    }

    static T getEnd(uint16_t depth)
    {
        return 1.0 - nbl::hlsl::exp2(__common(depth) + 1.0);
    }

    template<class Ray>
    static void adjust(NBL_REF_ARG(Ray) ray, const vector<T, 3> adjDirection, uint16_t depth)
    {
        ray.origin += adjDirection * ray.intersectionT * getStart(depth);
    }
};

enum MaterialType : uint32_t
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

template<class RayDirInfo, class Spectrum NBL_PRIMARY_REQUIRES(bxdf::ray_dir_info::Basic<RayDirInfo> && concepts::FloatingPointLikeVectorial<Spectrum>)
struct PTIsotropicInteraction
{
    using this_t = PTIsotropicInteraction<RayDirInfo, Spectrum>;
    using ray_dir_info_type = RayDirInfo;
    using scalar_type = typename RayDirInfo::scalar_type;
    using vector3_type = typename RayDirInfo::vector3_type;
    using spectral_type = vector3_type;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static this_t create(NBL_CONST_REF_ARG(RayDirInfo) normalizedV, const vector3_type normalizedN)
    {
        this_t retval;
        retval.V = normalizedV;
        retval.N = normalizedN;
        retval.NdotV = nbl::hlsl::dot<vector3_type>(retval.N, retval.V.getDirection());
        retval.NdotV2 = retval.NdotV * retval.NdotV;
        retval.luminosityContributionHint = hlsl::promote<spectral_type>(1.0);

        return retval;
    }

    RayDirInfo getV() NBL_CONST_MEMBER_FUNC { return V; }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return N; }
    scalar_type getNdotV(bxdf::BxDFClampMode _clamp = bxdf::BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC
    {
        return bxdf::conditionalAbsOrMax<scalar_type>(NdotV, _clamp);
    }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return NdotV2; }

    bxdf::PathOrigin getPathOrigin() NBL_CONST_MEMBER_FUNC { return bxdf::PathOrigin::PO_SENSOR; }
    spectral_type getLuminosityContributionHint() NBL_CONST_MEMBER_FUNC { return luminosityContributionHint; }
    bool isMaterialBSDF() NBL_CONST_MEMBER_FUNC { return b_isMaterialBSDF; }

    RayDirInfo V;
    vector3_type N;
    scalar_type NdotV;
    scalar_type NdotV2;
    bool b_isMaterialBSDF;

    spectral_type luminosityContributionHint;
};

template<class IsotropicInteraction NBL_PRIMARY_REQUIRES(bxdf::surface_interactions::Isotropic<IsotropicInteraction>)
struct PTAnisotropicInteraction
{
    using this_t = PTAnisotropicInteraction<IsotropicInteraction>;
    using isotropic_interaction_type = IsotropicInteraction;
    using ray_dir_info_type = typename isotropic_interaction_type::ray_dir_info_type;
    using scalar_type = typename ray_dir_info_type::scalar_type;
    using vector3_type = typename ray_dir_info_type::vector3_type;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;
    using spectral_type = typename isotropic_interaction_type::spectral_type;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static this_t create(
        NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic,
        const vector3_type normalizedT,
        const vector3_type normalizedB
    )
    {
        this_t retval;
        retval.isotropic = isotropic;

        retval.T = normalizedT;
        retval.B = normalizedB;

        retval.TdotV = nbl::hlsl::dot<vector3_type>(retval.isotropic.getV().getDirection(), retval.T);
        retval.BdotV = nbl::hlsl::dot<vector3_type>(retval.isotropic.getV().getDirection(), retval.B);

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic, const vector3_type normalizedT)
    {
        return create(isotropic, normalizedT, cross(isotropic.getN(), normalizedT));
    }
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic)
    {
        vector3_type T, B;
        math::frisvad<vector3_type>(isotropic.getN(), T, B);
        return create(isotropic, nbl::hlsl::normalize<vector3_type>(T), nbl::hlsl::normalize<vector3_type>(B));
    }

    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) normalizedV, const vector3_type normalizedN)
    {
        isotropic_interaction_type isotropic = isotropic_interaction_type::create(normalizedV, normalizedN);
        return create(isotropic);
    }

    ray_dir_info_type getV() NBL_CONST_MEMBER_FUNC { return isotropic.getV(); }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return isotropic.getN(); }
    scalar_type getNdotV(bxdf::BxDFClampMode _clamp = bxdf::BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC { return isotropic.getNdotV(_clamp); }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return isotropic.getNdotV2(); }
    bxdf::PathOrigin getPathOrigin() NBL_CONST_MEMBER_FUNC { return isotropic.getPathOrigin(); }
    spectral_type getLuminosityContributionHint() NBL_CONST_MEMBER_FUNC { return isotropic.getLuminosityContributionHint(); }
    bool isMaterialBSDF() NBL_CONST_MEMBER_FUNC { return isotropic.isMaterialBSDF(); }
    isotropic_interaction_type getIsotropic() NBL_CONST_MEMBER_FUNC { return isotropic; }

    vector3_type getT() NBL_CONST_MEMBER_FUNC { return T; }
    vector3_type getB() NBL_CONST_MEMBER_FUNC { return B; }
    scalar_type getTdotV() NBL_CONST_MEMBER_FUNC { return TdotV; }
    scalar_type getTdotV2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getTdotV(); return t*t; }
    scalar_type getBdotV() NBL_CONST_MEMBER_FUNC { return BdotV; }
    scalar_type getBdotV2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getBdotV(); return t*t; }

    vector3_type getTangentSpaceV() NBL_CONST_MEMBER_FUNC { return vector3_type(TdotV, BdotV, isotropic.getNdotV()); }
    matrix3x3_type getToTangentSpace() NBL_CONST_MEMBER_FUNC { return matrix3x3_type(T, B, isotropic.getN()); }
    matrix3x3_type getFromTangentSpace() NBL_CONST_MEMBER_FUNC { return nbl::hlsl::transpose<matrix3x3_type>(matrix3x3_type(T, B, isotropic.getN())); }

    isotropic_interaction_type isotropic;
    vector3_type T;
    vector3_type B;
    scalar_type TdotV;
    scalar_type BdotV;
};

template<class LS, class Interaction, class Spectrum NBL_STRUCT_CONSTRAINABLE>
struct PTIsoConfiguration;

#define CONF_ISO bxdf::LightSample<LS> && bxdf::surface_interactions::Isotropic<Interaction> && !bxdf::surface_interactions::Anisotropic<Interaction> && concepts::FloatingPointLikeVectorial<Spectrum>

template<class LS, class Interaction, class Spectrum>
NBL_PARTIAL_REQ_TOP(CONF_ISO)
struct PTIsoConfiguration<LS,Interaction,Spectrum NBL_PARTIAL_REQ_BOT(CONF_ISO) >
#undef CONF_ISO
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = false;

    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using monochrome_type = vector<scalar_type, 1>;

    using isotropic_interaction_type = Interaction;
    using anisotropic_interaction_type = PTAnisotropicInteraction<isotropic_interaction_type>;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
};

template<class LS, class Interaction, class MicrofacetCache, class Spectrum NBL_STRUCT_CONSTRAINABLE>
struct PTIsoMicrofacetConfiguration;

#define MICROFACET_CONF_ISO bxdf::LightSample<LS> && bxdf::surface_interactions::Isotropic<Interaction> && !bxdf::surface_interactions::Anisotropic<Interaction> && bxdf::CreatableIsotropicMicrofacetCache<MicrofacetCache> && !bxdf::AnisotropicMicrofacetCache<MicrofacetCache> && concepts::FloatingPointLikeVectorial<Spectrum>

template<class LS, class Interaction, class MicrofacetCache, class Spectrum>
NBL_PARTIAL_REQ_TOP(MICROFACET_CONF_ISO)
struct PTIsoMicrofacetConfiguration<LS,Interaction,MicrofacetCache,Spectrum NBL_PARTIAL_REQ_BOT(MICROFACET_CONF_ISO) > : PTIsoConfiguration<LS, Interaction, Spectrum>
#undef MICROFACET_CONF_ISO
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = false;

    using base_type = PTIsoConfiguration<LS, Interaction, Spectrum>;

    using matrix3x3_type = matrix<typename base_type::scalar_type,3,3>;

    using isocache_type = MicrofacetCache;
    using anisocache_type = bxdf::SAnisotropicMicrofacetCache<MicrofacetCache>;
};

#endif
