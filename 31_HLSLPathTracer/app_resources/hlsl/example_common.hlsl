#ifndef _NBL_HLSL_PATHTRACING_EXAMPLE_COMMON_INCLUDED_
#define _NBL_HLSL_PATHTRACING_EXAMPLE_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

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

#endif
