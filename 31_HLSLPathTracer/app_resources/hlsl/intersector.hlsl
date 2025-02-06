#ifndef _NBL_HLSL_EXT_INTERSECTOR_INCLUDED_
#define _NBL_HLSL_EXT_INTERSECTOR_INCLUDED_

#include "common.hlsl"
#include <nbl/builtin/hlsl/limits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace Intersector
{

// ray query method
// ray query struct holds AS info
// pass in address to vertex/index buffers?

// ray tracing pipeline method

// procedural data store: [obj count] [intersect type] [obj1] [obj2] [...]

struct IntersectData
{
    enum class Mode : uint32_t
    {
        RAY_QUERY,
        RAY_TRACING,
        PROCEDURAL
    };

    NBL_CONSTEXPR_STATIC_INLINE uint32_t DataSize = 128;

    uint32_t mode : 1;
    unit32_t unused : 31;   // possible space for flags
    uint32_t data[DataSize];
};

template<class Ray>
struct Comprehensive
{
    using scalar_type = typename Ray::scalar_type;
    using ray_type = Ray;

    static int traceProcedural(NBL_REF_ARG(ray_type) ray, NBL_REF_ARG(IntersectData) intersect)
    {
        const bool anyHit = ray.intersectionT != numeric_limits<scalar_type>::max;
        const uint32_t objCount = intersect.data[0];
        const ProceduralIntersectionType type = intersect.data[1];

        int objectID = -1;
        for (int i = 0; i < objCount; i++)
        {
            float t;
            switch (type)
            {
                case PIT_SPHERE:
                {
                    float32_t3 position = float32_t3(asfloat(intersect.data[2 + i * Intersection<PIT_SPHERE>::ObjSize]), asfloat(intersect.data[2 + i * Intersection<PIT_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Intersection<PIT_SPHERE>::ObjSize + 2]));
                    Intersection<PIT_SPHERE> sphere = Intersection<PIT_SPHERE>::create(position, asfloat(intersect.data[2 + i * Intersection<PIT_SPHERE>::ObjSize + 3]), intersect.data[2 + i * Intersection<PIT_SPHERE>::ObjSize + 4]);
                    t = sphere.intersect(ray.origin, ray.direction);
                }
                break;
                case PIT_TRIANGLE:
                {
                    float32_t3 vertex0 = float32_t3(asfloat(intersect.data[2 + i * Intersection<PIT_TRIANGLE>::ObjSize]), asfloat(intersect.data[2 + i * Intersection<PIT_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Intersection<PIT_TRIANGLE>::ObjSize + 2]));
                    float32_t3 vertex1 = float32_t3(asfloat(intersect.data[2 + i * Intersection<PIT_TRIANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + i * Intersection<PIT_SPHERE>::ObjSize + 4]), asfloat(intersect.data[2 + i * Intersection<PIT_TRIANGLE>::ObjSize + 5]));
                    float32_t3 vertex2 = float32_t3(asfloat(intersect.data[2 + i * Intersection<PIT_TRIANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + i * Intersection<PIT_SPHERE>::ObjSize + 7]), asfloat(intersect.data[2 + i * Intersection<PIT_TRIANGLE>::ObjSize + 8]));
                    Intersection<PIT_TRIANGLE> tri = Intersection<PIT_TRIANGLE>::create(vertex0, vertex1, vertex2, intersect.data[2 + i * Intersection<PIT_TRIANGLE>::ObjSize + 9]);
                    t = tri.intersect(ray.origin, ray.direction);
                }
                break;
                case PIT_RECTANGLE:
                {
                    float32_t3 offset = float32_t3(asfloat(intersect.data[2 + i * Intersection<PIT_RECTANGLE>::ObjSize]), asfloat(intersect.data[2 + i * Intersection<PIT_RECTANGLE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Intersection<PIT_RECTANGLE>::ObjSize + 2]));
                    float32_t3 edge0 = float32_t3(asfloat(intersect.data[2 + i * Intersection<PIT_RECTANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + i * Intersection<PIT_RECTANGLE>::ObjSize + 4]), asfloat(intersect.data[2 + i * Intersection<PIT_RECTANGLE>::ObjSize + 5]));
                    float32_t3 edge1 = float32_t3(asfloat(intersect.data[2 + i * Intersection<PIT_RECTANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + i * Intersection<PIT_RECTANGLE>::ObjSize + 7]), asfloat(intersect.data[2 + i * Intersection<PIT_RECTANGLE>::ObjSize + 8]));
                    Intersection<PIT_RECTANGLE> rect = Intersection<PIT_RECTANGLE>::create(offset, edge0, edge1, intersect.data[2 + i * Intersection<PIT_RECTANGLE>::ObjSize + 9]);
                    t = rect.intersect(ray.origin, ray.direction);
                }
                break;
                default:
                    t = numeric_limits<float>::infinity;
                    break;
            }
            
            bool closerIntersection = t > 0.0 && t < ray.intersectionT;

            ray.intersectionT = closerIntersection ? t : ray.intersectionT;
            objectID = closerIntersection ? i : objectID;
            
            // allowing early out results in a performance regression, WTF!?
            //if (anyHit && closerIntersection)
            //break;
        }
        return objectID;
    }

    static int traceRay(NBL_REF_ARG(ray_type) ray, NBL_REF_ARG(IntersectData) intersect)
    {
        const IntersectData::Mode mode = intersect.mode;
        switch (mode)
        {
            case IntersectData::Mode::RAY_QUERY:
            {
                // TODO: do ray query stuff
            }
            break;
            case IntersectData::Mode::RAY_TRACING:
            {
                // TODO: do ray tracing stuff
            }
            break;
            case IntersectData::Mode::PROCEDURAL:
            {
                return traceProcedural(ray, intersect);
            }
            break;
            default:
                return -1;
        }
    }
};

// does everything in traceray in ex 30
// template<class Ray>
// struct Procedural
// {
//     using scalar_type = typename Ray::scalar_type;
//     using ray_type = Ray;

//     static int traceRay(NBL_REF_ARG(ray_type) ray, IIntersection objects[32], int objCount)
//     {
//         const bool anyHit = ray.intersectionT != numeric_limits<scalar_type>::max;

//         int objectID = -1;
//         for (int i = 0; i < objCount; i++)
//         {
//             float t;
//             if (objects[i].type == PIT_SPHERE)  // we don't know what type of intersection it is so cast, there has to be a better way to do this
//             {
//                 Intersection<PIT_SPHERE> sphere = (Intersection<PIT_SPHERE>)objects[i];
//                 t = sphere.intersect(ray.origin, ray.direction);
//             }
//             // TODO: other types
            
//             bool closerIntersection = t > 0.0 && t < ray.intersectionT;

//             ray.intersectionT = closerIntersection ? t : ray.intersectionT;
//             objectID = closerIntersection ? i : objectID;
            
//             // allowing early out results in a performance regression, WTF!?
//             //if (anyHit && closerIntersection)
//             //break;
//         }
//         return objectID;
//     }

//     // TODO? traceray with vertex/index buffer
// };

}
}
}
}

#endif