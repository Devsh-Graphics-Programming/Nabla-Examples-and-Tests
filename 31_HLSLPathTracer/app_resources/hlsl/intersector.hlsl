#ifndef _NBL_HLSL_EXT_INTERSECTOR_INCLUDED_
#define _NBL_HLSL_EXT_INTERSECTOR_INCLUDED_

#include "example_common.hlsl"
#include <nbl/builtin/hlsl/limits.hlsl>

using namespace nbl;
using namespace hlsl;

template<class Ray, class Scene>
struct Intersector
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
        NBL_UNROLL for (int i = 0; i < scene_type::SphereCount; i++)
        {
            float t = scene.getSphere(i).intersect(ray.origin, ray.direction);

            bool closerIntersection = t > 0.0 && t < ray.intersectionT;

            if (closerIntersection)
            {
                ray.intersectionT = t;
                objectID.id = i;
                objectID.mode = IM_PROCEDURAL;
                objectID.shapeType = PST_SPHERE;
            }
        }
        NBL_UNROLL for (int i = 0; i < scene_type::TriangleCount; i++)
        {
            float t = scene.getTriangle(i).intersect(ray.origin, ray.direction);

            bool closerIntersection = t > 0.0 && t < ray.intersectionT;

            if (closerIntersection)
            {
                ray.intersectionT = t;
                objectID.id = i;
                objectID.mode = IM_PROCEDURAL;
                objectID.shapeType = PST_TRIANGLE;
            }
        }
        NBL_UNROLL for (int i = 0; i < scene_type::RectangleCount; i++)
        {
            float t = scene.getRectangle(i).intersect(ray.origin, ray.direction);

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

#endif
