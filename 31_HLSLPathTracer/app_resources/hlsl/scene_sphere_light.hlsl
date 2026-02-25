#ifndef _PATHTRACER_EXAMPLE_SCENE_SPHERE_LIGHT_INCLUDED_
#define _PATHTRACER_EXAMPLE_SCENE_SPHERE_LIGHT_INCLUDED_

#include "scene_base.hlsl"

using namespace nbl;
using namespace hlsl;

struct SceneSphereLight : SceneBase
{
    using scalar_type = float;
    using vector3_type = vector<scalar_type, 3>;
    using this_t = SceneSphereLight;
    using base_t = SceneBase;
    using object_handle_type = ObjectID;
    using mat_light_id_type = base_t::mat_light_id_type;

    using ray_dir_info_t = bxdf::ray_dir_info::SBasic<float>;
    using interaction_type = bxdf::surface_interactions::SIsotropic<ray_dir_info_t, spectral_t>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t SphereCount = base_t::SCENE_SPHERE_COUNT + base_t::SCENE_LIGHT_COUNT;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t TriangleCount = 0u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RectangleCount = 0u;

    static const Shape<scalar_type, PST_SPHERE> light_spheres[1];

    Shape<scalar_type, PST_SPHERE> getSphere(uint32_t idx)
    {
        assert(idx < SphereCount);
        if (idx < base_t::SCENE_SPHERE_COUNT)
            return base_t::scene_spheres[idx];
        else
            return light_spheres[idx-base_t::SCENE_SPHERE_COUNT];
    }

    Shape<scalar_type, PST_TRIANGLE> getTriangle(uint32_t idx)
    {
        assert(false);
        Shape<scalar_type, PST_TRIANGLE> dummy;
        return dummy;
    }

    Shape<scalar_type, PST_RECTANGLE> getRectangle(uint32_t idx)
    {
        assert(false);
        Shape<scalar_type, PST_RECTANGLE> dummy;
        return dummy;
    }

     void updateLight(NBL_CONST_REF_ARG(float32_t3x4) generalPurposeLightMatrix)
    {
        light_spheres[0].updateTransform(generalPurposeLightMatrix);
    }
    
    mat_light_id_type getMatLightIDs(NBL_CONST_REF_ARG(object_handle_type) objectID)
    {
        assert(objectID.shapeType == PST_SPHERE);
        return mat_light_id_type::createFromPacked(getSphere(objectID.id).bsdfLightIDs);
    }

    template<class Ray>
    interaction_type getInteraction(NBL_CONST_REF_ARG(object_handle_type) objectID, NBL_CONST_REF_ARG(vector3_type) intersection, NBL_CONST_REF_ARG(Ray) ray)
    {
        assert(objectID.shapeType == PST_SPHERE);
        vector3_type N = getSphere(objectID.id).getNormal(intersection);
        N = hlsl::normalize(N);
        ray_dir_info_t V;
        V.setDirection(-ray.direction);
        interaction_type interaction = interaction_type::create(V, N);
        interaction.luminosityContributionHint = hlsl::normalize(colorspace::scRGBtoXYZ[1] * ray.getPayloadThroughput());
        return interaction;
    }
};

const Shape<float, PST_SPHERE> SceneSphereLight::light_spheres[1] = {
    Shape<float, PST_SPHERE>::create(float3(-1.5, 1.5, 0.0), 0.3, SceneBase::SCENE_BXDF_COUNT-1u/*last in mat arr*/, 0u)
};

using scene_type = SceneSphereLight;

NBL_CONSTEXPR ProceduralShapeType LIGHT_TYPE = PST_SPHERE;
using light_type = Light<spectral_t>;

static const light_type lights[scene_type::SCENE_LIGHT_COUNT] = {
    light_type::create(SceneBase::SCENE_BXDF_COUNT-1u/*last in mat arr*/, scene_type::SCENE_SPHERE_COUNT, LIGHT_TYPE)
};

#endif
