#ifndef _NBL_HLSL_EXT_INTERSECTOR_INCLUDED_
#define _NBL_HLSL_EXT_INTERSECTOR_INCLUDED_

#include "common.hlsl"
#include "scene.hlsl"
#include <nbl/builtin/hlsl/limits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace Intersector
{

template<class Ray, typename Light, typename BxdfNode>
struct Comprehensive
{
    using scalar_type = typename Ray::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using ray_type = Ray;

    using light_type = Light;
    using bxdfnode_type = BxdfNode;
    using scene_type = Scene<light_type, bxdfnode_type>;

    static ObjectID traceRay(NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(scene_type) scene)
    {
        ObjectID objectID;
        objectID.id = -1;

        // prodedural shapes
        for (int i = 0; i < scene.sphereCount; i++)
        {
            float t = scene.spheres[i].intersect(ray.origin, ray.direction);

            bool closerIntersection = t > 0.0 && t < ray.intersectionT;

            if (closerIntersection)
            {
                ray.intersectionT = t;
                objectID.id = i;
                objectID.mode = IM_PROCEDURAL;
                objectID.shapeType = PST_SPHERE;
            }
        }
        for (int i = 0; i < scene.triangleCount; i++)
        {
            float t = scene.triangles[i].intersect(ray.origin, ray.direction);

            bool closerIntersection = t > 0.0 && t < ray.intersectionT;

            if (closerIntersection)
            {
                ray.intersectionT = t;
                objectID.id = i;
                objectID.mode = IM_PROCEDURAL;
                objectID.shapeType = PST_TRIANGLE;
            }
        }
        for (int i = 0; i < scene.rectangleCount; i++)
        {
            float t = scene.rectangles[i].intersect(ray.origin, ray.direction);

            bool closerIntersection = t > 0.0 && t < ray.intersectionT;

            if (closerIntersection)
            {
                ray.intersectionT = t;
                objectID.id = i;
                objectID.mode = IM_PROCEDURAL;
                objectID.shapeType = PST_TRIANGLE;
            }
        }

        // TODO: trace AS

        return objectID;
    }

    // note for future consideration: still need to encode to IntersectData?
    // obsolete?
    // static ObjectID traceProcedural(NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(IntersectData) intersect)
    // {
    //     const bool anyHit = ray.intersectionT != numeric_limits<scalar_type>::max;
    //     const uint32_t objCount = intersect.data[0];
    //     const ProceduralShapeType type = (ProceduralShapeType)intersect.data[1];

    //     ObjectID objectID = ray.objectID;
    //     objectID.mode = IM_PROCEDURAL;
    //     objectID.shapeType = type;
    //     for (int i = 0; i < objCount; i++)
    //     {
    //         float t;
    //         switch (type)
    //         {
    //             case PST_SPHERE:
    //             {
    //                 vector3_type position = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 2]));
    //                 Shape<PST_SPHERE> sphere = Shape<PST_SPHERE>::create(position, asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 3]), intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 4]);
    //                 t = sphere.intersect(ray.origin, ray.direction);
    //             }
    //             break;
    //             case PST_TRIANGLE:
    //             {
    //                 vector3_type vertex0 = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 2]));
    //                 vector3_type vertex1 = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 4]), asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 5]));
    //                 vector3_type vertex2 = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + i * Shape<PST_SPHERE>::ObjSize + 7]), asfloat(intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 8]));
    //                 Shape<PST_TRIANGLE> tri = Shape<PST_TRIANGLE>::create(vertex0, vertex1, vertex2, intersect.data[2 + i * Shape<PST_TRIANGLE>::ObjSize + 9]);
    //                 t = tri.intersect(ray.origin, ray.direction);
    //             }
    //             break;
    //             case PST_RECTANGLE:
    //             {
    //                 vector3_type offset = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 1]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 2]));
    //                 vector3_type edge0 = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 4]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 5]));
    //                 vector3_type edge1 = vector3_type(asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 7]), asfloat(intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 8]));
    //                 Shape<PST_RECTANGLE> rect = Shape<PST_RECTANGLE>::create(offset, edge0, edge1, intersect.data[2 + i * Shape<PST_RECTANGLE>::ObjSize + 9]);
    //                 t = rect.intersect(ray.origin, ray.direction);
    //             }
    //             break;
    //             default:
    //                 t = numeric_limits<float>::infinity;
    //             break;
    //         }

    //         bool closerIntersection = t > 0.0 && t < ray.intersectionT;

    //         ray.intersectionT = closerIntersection ? t : ray.intersectionT;
    //         objectID.id = closerIntersection ? i : objectID.id;

    //         // allowing early out results in a performance regression, WTF!?
    //         //if (anyHit && closerIntersection)
    //         //break;
    //     }
    //     return objectID;
    // }

    // obsolete?
    // static ObjectID traceRay(NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(IntersectData) intersect)
    // {
    //     const uint32_t mode = intersect.mode;
    //     switch (mode)
    //     {
    //         case IM_RAY_QUERY:
    //         {
    //             // TODO: do ray query stuff
    //         }
    //         break;
    //         case IM_RAY_TRACING:
    //         {
    //             // TODO: do ray tracing stuff
    //         }
    //         break;
    //         case IM_PROCEDURAL:
    //         {
    //             return traceProcedural(ray, intersect);
    //         }
    //         break;
    //         default:
    //         {
    //             return ObjectID::create(-1, 0, PST_SPHERE);
    //         }
    //     }
    //     return ObjectID::create(-1, 0, PST_SPHERE);
    // }

    // static ObjectID traceRay(NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(scene_type) scene)
    // {
    //     IntersectData data;

    //     ObjectID objectID;
    //     objectID.id = -1;  // start with no intersect

    //     // prodedural shapes
    //     if (scene.sphereCount > 0)
    //     {
    //         data = scene.toIntersectData(ext::Intersector::IntersectData::Mode::PROCEDURAL, PST_SPHERE);
    //         objectID = traceRay(ray, data);
    //     }

    //     if (scene.triangleCount > 0)
    //     {
    //         data = scene.toIntersectData(ext::Intersector::IntersectData::Mode::PROCEDURAL, PST_TRIANGLE);
    //         objectID = traceRay(ray, data);
    //     }

    //     if (scene.rectangleCount > 0)
    //     {
    //         data = scene.toIntersectData(ext::Intersector::IntersectData::Mode::PROCEDURAL, PST_RECTANGLE);
    //         objectID = traceRay(ray, data);
    //     }

    //     // TODO: trace AS

    //     return objectID;
    // }
};

}
}
}
}

#endif
