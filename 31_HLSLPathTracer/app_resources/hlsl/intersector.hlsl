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
    static IntersectData encode(uint32_t mode, ProceduralShapeType type, NBL_CONST_REF_ARG(Scene) scene)
    {
        IntersectData retval;
        retval.mode = mode;

        uint32_t objCount = (type == PST_SPHERE) ? scene.sphereCount :
                            (type == PST_TRIANGLE) ? scene.triangleCount :
                            (type == PST_RECTANGLE) ? scene.rectangleCount :
                            -1;
        retval.data[0] = objCount;
        retval.data[1] = type;
        
        switch (type)
        {
            case PST_SPHERE:
            {
                for (int i = 0; i < objCount; i++)
                {
                    Shape<PST_SPHERE> sphere = scene.spheres[i];
                    retval.data[2 + i * Shape<PST_SPHERE>::ObjSize] = asuint(sphere.position.x);
                    retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 1] = asuint(sphere.position.y);
                    retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 2] = asuint(sphere.position.z);
                    retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 3] = asuint(sphere.radius);
                    retval.data[2 + i * Shape<PST_SPHERE>::ObjSize + 4] = sphere.bsdfLightIDs;
                }
            }
            break;
            case PST_TRIANGLE:
            {
                for (int i = 0; i < objCount; i++)
                {
                    Shape<PST_TRIANGLE> tri = scene.triangles[i];
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize] = asuint(tri.vertex0.x);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 1] = asuint(tri.vertex0.y);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 2] = asuint(tri.vertex0.z);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 3] = asuint(tri.vertex1.x);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 4] = asuint(tri.vertex1.y);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 5] = asuint(tri.vertex1.z);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 6] = asuint(tri.vertex2.x);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 7] = asuint(tri.vertex2.y);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 8] = asuint(tri.vertex2.z);
                    retval.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 9] = tri.bsdfLightIDs;
                }
            }
            break;
            case PST_RECTANGLE:
            {
                for (int i = 0; i < objCount; i++)
                {
                    Shape<PST_RECTANGLE> rect = scene.rectangles[i];
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize] = asuint(rect.offset.x);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 1] = asuint(rect.offset.y);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 2] = asuint(rect.offset.z);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 3] = asuint(rect.edge0.x);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 4] = asuint(rect.edge0.y);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 5] = asuint(rect.edge0.z);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 6] = asuint(rect.edge1.x);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 7] = asuint(rect.edge1.y);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 8] = asuint(rect.edge1.z);
                    retval.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 9] = rect.bsdfLightIDs;
                }
            }
            break;
            default:
                // for ASes
                break;
        }
        return retval;        
    }

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
        const ProceduralShapeType type = intersect.data[1];

        int objectID = ray.objectID;
        for (int i = 0; i < objCount; i++)
        {
            float t;
            switch (type)
            {
                case PST_SPHERE:
                {
                    float32_t3 position = float32_t3(asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 2]));
                    Shape<PST_SPHERE> sphere = Shape<PST_SPHERE>::create(position, asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 3]), intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 4]);
                    t = sphere.intersect(ray.origin, ray.direction);
                }
                break;
                case PST_TRIANGLE:
                {
                    float32_t3 vertex0 = float32_t3(asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 2]));
                    float32_t3 vertex1 = float32_t3(asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 4]), asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 5]));
                    float32_t3 vertex2 = float32_t3(asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 7]), asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 8]));
                    Shape<PST_TRIANGLE> tri = Shape<PST_TRIANGLE>::create(vertex0, vertex1, vertex2, intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 9]);
                    t = tri.intersect(ray.origin, ray.direction);
                }
                break;
                case PST_RECTANGLE:
                {
                    float32_t3 offset = float32_t3(asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 2]));
                    float32_t3 edge0 = float32_t3(asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 4]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 5]));
                    float32_t3 edge1 = float32_t3(asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 7]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 8]));
                    Shape<PST_RECTANGLE> rect = Shape<PST_RECTANGLE>::create(offset, edge0, edge1, intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 9]);
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
//             if (objects[i].type == PST_SPHERE)  // we don't know what type of intersection it is so cast, there has to be a better way to do this
//             {
//                 Shape<PST_SPHERE> sphere = (Shape<PST_SPHERE>)objects[i];
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