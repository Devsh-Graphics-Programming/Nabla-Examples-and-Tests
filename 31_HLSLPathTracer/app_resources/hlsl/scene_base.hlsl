#ifndef _PATHTRACER_EXAMPLE_SCENE_BASE_INCLUDED_
#define _PATHTRACER_EXAMPLE_SCENE_BASE_INCLUDED_

#include "example_common.hlsl"

using namespace nbl;
using namespace hlsl;

struct SceneBase
{
    using scalar_type = float;
    using vector3_type = vector<scalar_type, 3>;
    using light_type = Light<vector3_type>;
    using light_id_type = LightID;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t SCENE_SPHERE_COUNT = 10u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SCENE_LIGHT_COUNT = 1u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SCENE_BXDF_COUNT = 10u;

    static const Shape<scalar_type, PST_SPHERE> scene_spheres[SCENE_SPHERE_COUNT];

    struct MatLightID
    {
        using light_id_type = LightID;
        using material_id_type = MaterialID;

        light_id_type lightID;
        material_id_type matID;

        static MatLightID createFromPacked(uint32_t packedID)
        {
            MatLightID retval;
            retval.lightID.id = uint16_t(glsl::bitfieldExtract(packedID, 16, 16));
            retval.matID.id = uint16_t(glsl::bitfieldExtract(packedID, 0, 16));
            return retval;
        }

        light_id_type getLightID() NBL_CONST_MEMBER_FUNC { return lightID; }
        material_id_type getMaterialID() NBL_CONST_MEMBER_FUNC { return matID; }

        bool isLight() NBL_CONST_MEMBER_FUNC { return lightID.id != light_id_type::INVALID_ID; }
        bool canContinuePath() NBL_CONST_MEMBER_FUNC { return matID.id != material_id_type::INVALID_ID; }
    };
    using mat_light_id_type = MatLightID;
};

const Shape<float, PST_SPHERE> SceneBase::scene_spheres[SCENE_SPHERE_COUNT] = {
    Shape<float, PST_SPHERE>::create(float3(0.0, -100.5, -1.0), 100.0, 0u, SceneBase::light_id_type::INVALID_ID),
    Shape<float, PST_SPHERE>::create(float3(2.0, 0.0, -1.0), 0.5, 1u, SceneBase::light_id_type::INVALID_ID),
    Shape<float, PST_SPHERE>::create(float3(0.0, 0.0, -1.0), 0.5, 2u, SceneBase::light_id_type::INVALID_ID),
    Shape<float, PST_SPHERE>::create(float3(-2.0, 0.0, -1.0), 0.5, 3u, SceneBase::light_id_type::INVALID_ID),
    Shape<float, PST_SPHERE>::create(float3(2.0, 0.0, 1.0), 0.5, 4u, SceneBase::light_id_type::INVALID_ID),
    Shape<float, PST_SPHERE>::create(float3(0.0, 0.0, 1.0), 0.5, 4u, SceneBase::light_id_type::INVALID_ID),
    Shape<float, PST_SPHERE>::create(float3(-2.0, 0.0, 1.0), 0.5, 5u, SceneBase::light_id_type::INVALID_ID),
    Shape<float, PST_SPHERE>::create(float3(0.5, 1.0, 0.5), 0.5, 6u, SceneBase::light_id_type::INVALID_ID),
    Shape<float, PST_SPHERE>::create(float3(-4.0, 0.0, 1.0), 0.5, 7u, SceneBase::light_id_type::INVALID_ID),
    Shape<float, PST_SPHERE>::create(float3(-4.0, 0.0, -1.0), 0.5, 8u, SceneBase::light_id_type::INVALID_ID)
};

using spectral_t = vector<float, 3>;
using bxdfnode_type = BxDFNode<spectral_t>;

static const bxdfnode_type bxdfs[SceneBase::SCENE_BXDF_COUNT] = {
    bxdfnode_type::create(MaterialType::DIFFUSE, false, float2(0,0), spectral_t(0.8,0.8,0.8)),
    bxdfnode_type::create(MaterialType::DIFFUSE, false, float2(0,0), spectral_t(0.8,0.4,0.4)),
    bxdfnode_type::create(MaterialType::DIFFUSE, false, float2(0,0), spectral_t(0.4,0.8,0.4)),
    bxdfnode_type::create(MaterialType::CONDUCTOR, false, float2(0,0), spectral_t(1.02,1.02,1.3), spectral_t(1.0,1.0,2.0)),
    bxdfnode_type::create(MaterialType::CONDUCTOR, false, float2(0,0), spectral_t(1.02,1.3,1.02), spectral_t(1.0,2.0,1.0)),
    bxdfnode_type::create(MaterialType::CONDUCTOR, false, float2(0.15,0.15), spectral_t(1.02,1.3,1.02), spectral_t(1.0,2.0,1.0)),
    bxdfnode_type::create(MaterialType::DIELECTRIC, false, float2(0.0625,0.0625), spectral_t(1,1,1), spectral_t(1.4,1.45,1.5)),
    bxdfnode_type::create(MaterialType::IRIDESCENT_CONDUCTOR, false, 0.0, 505.0, spectral_t(1.39,1.39,1.39), spectral_t(1.2,1.2,1.2), spectral_t(0.5,0.5,0.5)),
    bxdfnode_type::create(MaterialType::IRIDESCENT_DIELECTRIC, false, 0.0, 400.0, spectral_t(1.7,1.7,1.7), spectral_t(1.0,1.0,1.0), spectral_t(0,0,0)),
    bxdfnode_type::create(MaterialType::EMISSIVE, LightEminence)
};

#endif
