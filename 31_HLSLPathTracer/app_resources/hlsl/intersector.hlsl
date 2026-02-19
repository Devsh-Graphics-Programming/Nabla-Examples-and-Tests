#ifndef _PATHTRACER_EXAMPLE_INTERSECTOR_INCLUDED_
#define _PATHTRACER_EXAMPLE_INTERSECTOR_INCLUDED_

#include "example_common.hlsl"
#include <nbl/builtin/hlsl/limits.hlsl>

using namespace nbl;
using namespace hlsl;

template<class Ray, class Scene, class AnisoInteraction>
struct Intersector
{
    using scalar_type = typename Ray::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using ray_type = Ray;
    using scene_type = Scene;
    using object_handle_type = ObjectID;

    using anisotropic_interaction_type = AnisoInteraction;
    using isotropic_interaction_type = typename anisotropic_interaction_type::isotropic_interaction_type;
    using ray_dir_info_type = typename anisotropic_interaction_type::ray_dir_info_type;

    struct SIntersectData
    {
        bool foundHit;
        vector3_type intersection;
        isotropic_interaction_type iso_interaction;
        anisotropic_interaction_type aniso_interaction;
    };
    using intersect_data_type = SIntersectData;

    static intersect_data_type traceRay(NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(scene_type) scene)
    {
        object_handle_type objectID;
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
                objectID.shapeType = PST_RECTANGLE;
            }
        }

        // TODO: trace AS

        ray.objectID = objectID;

        intersect_data_type retval;
        retval.foundHit = objectID.id != -1;
        if (retval.foundHit)
        {
            retval.intersection = ray.origin + ray.direction * ray.intersectionT;
            typename scene_type::mat_light_id_type matLightID = scene.getMatLightIDs(objectID);
            vector3_type N = scene.getNormal(objectID, retval.intersection);
            N = nbl::hlsl::normalize(N);
            ray_dir_info_type V;
            V.setDirection(-ray.direction);
            retval.iso_interaction = isotropic_interaction_type::create(V, N);
            retval.aniso_interaction = anisotropic_interaction_type::create(retval.iso_interaction);
        }

        return retval;
    }

    static scalar_type traceShadowRay(NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(scene_type) scene, NBL_CONST_REF_ARG(object_handle_type) objectID)
    {
        // prodedural shapes
        NBL_UNROLL for (int i = 0; i < scene_type::SphereCount; i++)
        {
            float t = scene.getSphere(i).intersect(ray.origin, ray.direction);
            bool closerIntersection = t > 0.0 && t < ray.intersectionT;

            if (closerIntersection)
                return 0.0;
        }
        NBL_UNROLL for (int i = 0; i < scene_type::TriangleCount; i++)
        {
            float t = scene.getTriangle(i).intersect(ray.origin, ray.direction);
            bool closerIntersection = t > 0.0 && t < ray.intersectionT;

            if (closerIntersection)
                return 0.0;
        }
        NBL_UNROLL for (int i = 0; i < scene_type::RectangleCount; i++)
        {
            float t = scene.getRectangle(i).intersect(ray.origin, ray.direction);
            bool closerIntersection = t > 0.0 && t < ray.intersectionT;

            if (closerIntersection)
                return 0.0;
        }

        return 1.0;
    }
};

#endif
