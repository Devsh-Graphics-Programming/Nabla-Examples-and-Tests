#ifndef _NBL_HLSL_EXT_PATHTRACING_COMMON_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACING_COMMON_INCLUDED_

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

}
}
}

#endif