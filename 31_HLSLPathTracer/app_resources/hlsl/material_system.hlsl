#ifndef _NBL_HLSL_EXT_MATERIAL_SYSTEM_INCLUDED_
#define _NBL_HLSL_EXT_MATERIAL_SYSTEM_INCLUDED_

#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/bxdf/common.hlsl>

#include "example_common.hlsl"

using namespace nbl;
using namespace hlsl;

enum MaterialType : uint32_t    // enum class?
{
    DIFFUSE,
    CONDUCTOR,
    DIELECTRIC
};

template<class BxDFNode, class DiffuseBxDF, class ConductorBxDF, class DielectricBxDF, class Scene>  // NOTE: these bxdfs should match the ones in Scene BxDFNode
struct MaterialSystem
{
    using this_t = MaterialSystem<BxDFNode, DiffuseBxDF, ConductorBxDF, DielectricBxDF, Scene>;
    using scalar_type = typename DiffuseBxDF::scalar_type;      // types should be same across all 3 bxdfs
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using measure_type = typename DiffuseBxDF::spectral_type;
    using sample_type = typename DiffuseBxDF::sample_type;
    using ray_dir_info_type = typename sample_type::ray_dir_info_type;
    using quotient_pdf_type = typename DiffuseBxDF::quotient_pdf_type;
    using anisotropic_interaction_type = typename DiffuseBxDF::anisotropic_interaction_type;
    using isotropic_interaction_type = typename anisotropic_interaction_type::isotropic_interaction_type;
    using anisocache_type = typename ConductorBxDF::anisocache_type;
    using isocache_type = typename anisocache_type::isocache_type;
    using create_params_t = SBxDFCreationParams<scalar_type, measure_type>;

    using bxdfnode_type = BxDFNode;
    using diffuse_op_type = DiffuseBxDF;
    using conductor_op_type = ConductorBxDF;
    using dielectric_op_type = DielectricBxDF;

    static bool isBSDF(uint32_t material)
    {
        return (material == MaterialType::DIFFUSE) ? bxdf::traits<diffuse_op_type>::type == bxdf::BT_BSDF :
                (material == MaterialType::CONDUCTOR) ? bxdf::traits<conductor_op_type>::type == bxdf::BT_BSDF :
                bxdf::traits<dielectric_op_type>::type == bxdf::BT_BSDF;
    }

    // these are specific for the bxdfs used for this example
    void fillBxdfParams(uint32_t material, NBL_CONST_REF_ARG(create_params_t) cparams)
    {
        switch(material)
        {
            case MaterialType::DIFFUSE:
            {
                using creation_t = typename diffuse_op_type::creation_type;
                creation_t params;
                params.A = cparams.A.x;
                diffuseBxDF = diffuse_op_type::create(params);
            }
            break;
            case MaterialType::CONDUCTOR:
            {
                conductorBxDF.ndf = conductor_op_type::ndf_type::create(cparams.A.x);
                conductorBxDF.fresnel = conductor_op_type::fresnel_type::create(cparams.ior0,cparams.ior1);
            }
            break;
            case MaterialType::DIELECTRIC:
            {
                using oriented_eta_t = bxdf::fresnel::OrientedEtas<typename dielectric_op_type::monochrome_type>;
                oriented_eta_t orientedEta = oriented_eta_t::create(1.0, hlsl::promote<typename dielectric_op_type::monochrome_type>(cparams.eta));
                dielectricBxDF.ndf = dielectric_op_type::ndf_type::create(cparams.A.x);
                dielectricBxDF.fresnel = dielectric_op_type::fresnel_type::create(orientedEta);
            }
            break;
            default:
                return;
        }
    }

    measure_type eval(uint32_t material, NBL_CONST_REF_ARG(create_params_t) cparams, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) _cache)
    {
        fillBxdfParams(material, cparams);
        switch(material)
        {
            case MaterialType::DIFFUSE:
            {
                return diffuseBxDF.eval(_sample, interaction);
            }
            break;
            case MaterialType::CONDUCTOR:
            {
                return conductorBxDF.eval(_sample, interaction, _cache);
            }
            break;
            case MaterialType::DIELECTRIC:
            {
                return dielectricBxDF.eval(_sample, interaction, _cache);
            }
            break;
            default:
                return hlsl::promote<measure_type>(0.0);
        }
    }

    sample_type generate(uint32_t material, NBL_CONST_REF_ARG(create_params_t) cparams, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector3_type) u, NBL_REF_ARG(anisocache_type) _cache)
    {
        fillBxdfParams(material, cparams);
        switch(material)
        {
            case MaterialType::DIFFUSE:
            {
                return diffuseBxDF.generate(interaction, u.xy);
            }
            break;
            case MaterialType::CONDUCTOR:
            {
                return conductorBxDF.generate(interaction, u.xy, _cache);
            }
            break;
            case MaterialType::DIELECTRIC:
            {
                return dielectricBxDF.generate(interaction, u, _cache);
            }
            break;
            default:
            {
                ray_dir_info_type L;
                L.makeInvalid();
                return sample_type::create(L, hlsl::promote<vector3_type>(0.0));
            }
        }

        ray_dir_info_type L;
        L.makeInvalid();
        return sample_type::create(L, hlsl::promote<vector3_type>(0.0));
    }

    quotient_pdf_type quotient_and_pdf(uint32_t material, NBL_CONST_REF_ARG(create_params_t) cparams, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) _cache)
    {
        const float minimumProjVectorLen = 0.00000001;  // TODO: still need this check?
        if (interaction.getNdotV(bxdf::BxDFClampMode::BCM_ABS) > minimumProjVectorLen && _sample.getNdotL(bxdf::BxDFClampMode::BCM_ABS) > minimumProjVectorLen)
        {
            fillBxdfParams(material, cparams);
            switch(material)
            {
                case MaterialType::DIFFUSE:
                {
                    return diffuseBxDF.quotient_and_pdf(_sample, interaction);
                }
                break;
                case MaterialType::CONDUCTOR:
                {
                    return conductorBxDF.quotient_and_pdf(_sample, interaction, _cache);
                }
                break;
                case MaterialType::DIELECTRIC:
                {
                    return dielectricBxDF.quotient_and_pdf(_sample, interaction, _cache);
                }
                break;
                default:
                    return quotient_pdf_type::create(hlsl::promote<measure_type>(0.0), 0.0);
            }
        }
        return quotient_pdf_type::create(hlsl::promote<measure_type>(0.0), 0.0);
    }

    DiffuseBxDF diffuseBxDF;
    ConductorBxDF conductorBxDF;
    DielectricBxDF dielectricBxDF;

    bxdfnode_type bxdfs[Scene::SCENE_BXDF_COUNT];
    uint32_t bxdfCount;
};

#endif
