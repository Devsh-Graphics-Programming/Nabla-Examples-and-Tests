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
    enum Mode : uint32_t    // enum class?
    {
        RAY_QUERY,
        RAY_TRACING,
        PROCEDURAL
    };

    NBL_CONSTEXPR_STATIC_INLINE uint32_t DataSize = 128;

    uint32_t mode : 1;
    uint32_t unused : 31;   // possible space for flags
    uint32_t data[DataSize];
};

template<class Ray>
struct Comprehensive
{
    using scalar_type = typename Ray::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using ray_type = Ray;

    static ObjectID traceProcedural(NBL_REF_ARG(ray_type) ray, NBL_REF_ARG(IntersectData) intersect)
    {
        const bool anyHit = ray.intersectionT != numeric_limits<scalar_type>::max;
        const uint32_t objCount = intersect.data[0];
        const ProceduralShapeType type = (ProceduralShapeType)intersect.data[1];

        ObjectID objectID = ray.objectID;
        objectID.mode = IntersectData::Mode::PROCEDURAL;
        objectID.shapeType = type;
        for (int i = 0; i < objCount; i++)
        {
            float t;
            switch (type)
            {
                case PST_SPHERE:
                {
                    vector3_type position = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 2]));
                    Shape<PST_SPHERE> sphere = Shape<PST_SPHERE>::create(position, asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 3]), intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 4]);
                    t = sphere.intersect(ray.origin, ray.direction);
                }
                break;
                case PST_TRIANGLE:
                {
                    vector3_type vertex0 = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 2]));
                    vector3_type vertex1 = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 4]), asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 5]));
                    vector3_type vertex2 = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 7]), asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 8]));
                    Shape<PST_TRIANGLE> tri = Shape<PST_TRIANGLE>::create(vertex0, vertex1, vertex2, intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 9]);
                    t = tri.intersect(ray.origin, ray.direction);
                }
                break;
                case PST_RECTANGLE:
                {
                    vector3_type offset = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 2]));
                    vector3_type edge0 = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 4]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 5]));
                    vector3_type edge1 = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 7]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 8]));
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
            objectID.id = closerIntersection ? i : objectID.id;
            
            // allowing early out results in a performance regression, WTF!?
            //if (anyHit && closerIntersection)
            //break;
        }
        return objectID;
    }

    static ObjectID traceRay(NBL_REF_ARG(ray_type) ray, NBL_REF_ARG(IntersectData) intersect)
    {
        const IntersectData::Mode mode = (IntersectData::Mode)intersect.mode;
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
            {
                ObjectID objID;
                objID.id = -1;
                return objID;
            }
        }
    }

    template<typename Scene>
    static ObjectID traceRay(NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(Scene) scene)
    {
        IntersectData data;

        ObjectID objectID;
        objectID.id = -1;  // start with no intersect
                
        // prodedural shapes
        if (scene.sphereCount > 0)
        {
            data = scene.toIntersectData(ext::Intersector::IntersectData::Mode::PROCEDURAL, PST_SPHERE);
            objectID = traceRay(ray, data);
        }

        if (scene.triangleCount > 0)
        {
            data = scene.toIntersectData(ext::Intersector::IntersectData::Mode::PROCEDURAL, PST_TRIANGLE);
            objectID = traceRay(ray, data);
        }

        if (scene.rectangleCount > 0)
        {
            data = scene.toIntersectData(ext::Intersector::IntersectData::Mode::PROCEDURAL, PST_RECTANGLE);
            objectID = traceRay(ray, data);
        }

        // TODO: trace AS

        return objectID;
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
