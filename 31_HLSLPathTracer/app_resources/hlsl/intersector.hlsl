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
        using object_handle_type = object_handle_type;
        using vector3_type = vector3_type;
        using interaction_type = anisotropic_interaction_type;

        object_handle_type objectID;
        vector3_type position;
        interaction_type aniso_interaction;

        bool foundHit() NBL_CONST_MEMBER_FUNC { return !hlsl::isnan(position.x); }
        object_handle_type getObjectID() NBL_CONST_MEMBER_FUNC { return objectID; }
        vector3_type getPosition() NBL_CONST_MEMBER_FUNC { return position; }
        interaction_type getInteraction() NBL_CONST_MEMBER_FUNC { return aniso_interaction; }
    };
    using closest_hit_type = SIntersectData;

    static closest_hit_type traceClosestHit(NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(scene_type) scene)
    {
        object_handle_type objectID;
        objectID.id = object_handle_type::INVALID_ID;

        // prodedural shapes
        NBL_UNROLL for (int i = 0; i < scene_type::SphereCount; i++)
        {
            float t = scene.getSphere(i).intersect(ray.origin, ray.direction);

            bool closerIntersection = t > 0.0 && t < ray.intersectionT;

            if (closerIntersection)
            {
                ray.intersectionT = t;
                objectID.id = uint16_t(i);
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
                objectID.id = uint16_t(i);
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
                objectID.id = uint16_t(i);
                objectID.shapeType = PST_RECTANGLE;
            }
        }

        closest_hit_type retval;
        retval.objectID = objectID;
        retval.position = hlsl::promote<vector3_type>(bit_cast<scalar_type>(numeric_limits<scalar_type>::quiet_NaN));

        bool foundHit = objectID.id != object_handle_type::INVALID_ID;
        if (foundHit)
        {
            retval.position = ray.origin + ray.direction * ray.intersectionT;
            typename scene_type::mat_light_id_type matLightID = scene.getMatLightIDs(objectID);
            vector3_type N = scene.getNormal(objectID, retval.position);
            N = nbl::hlsl::normalize(N);
            ray_dir_info_type V;
            V.setDirection(-ray.direction);
            isotropic_interaction_type iso_interaction = isotropic_interaction_type::create(V, N);
            iso_interaction.luminosityContributionHint = hlsl::normalize(colorspace::scRGBtoXYZ[1] * ray.getPayloadThroughput());
            retval.aniso_interaction = anisotropic_interaction_type::create(iso_interaction);
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
