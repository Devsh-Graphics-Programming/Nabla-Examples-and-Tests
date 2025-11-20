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

template<class Ray, class Scene>
struct Comprehensive
{
    using scalar_type = typename Ray::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using ray_type = Ray;
    using scene_type = Scene;

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
};

}
}
}
}

#endif
