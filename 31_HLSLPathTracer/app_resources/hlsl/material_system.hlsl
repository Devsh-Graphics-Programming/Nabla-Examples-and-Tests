#ifndef _PATHTRACER_EXAMPLE_MATERIAL_SYSTEM_INCLUDED_
#define _PATHTRACER_EXAMPLE_MATERIAL_SYSTEM_INCLUDED_

#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/bxdf/common.hlsl>

#include "example_common.hlsl"

using namespace nbl;
using namespace hlsl;

enum MaterialType : uint32_t    // enum class?
{
    DIFFUSE = 0u,
    CONDUCTOR,
    DIELECTRIC,
    IRIDESCENT_CONDUCTOR,
    IRIDESCENT_DIELECTRIC,
};

template<class BxDFNode, class DiffuseBxDF, class ConductorBxDF, class DielectricBxDF, class IridescentConductorBxDF, class IridescentDielectricBxDF, class Scene>  // NOTE: these bxdfs should match the ones in Scene BxDFNode
struct MaterialSystem
{
    using this_t = MaterialSystem<BxDFNode, DiffuseBxDF, ConductorBxDF, DielectricBxDF, IridescentConductorBxDF, IridescentDielectricBxDF, Scene>;
    using scalar_type = typename DiffuseBxDF::scalar_type;      // types should be same across all 3 bxdfs
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using material_id_type = uint32_t;
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
    using iri_conductor_op_type = IridescentConductorBxDF;
    using iri_dielectric_op_type = IridescentDielectricBxDF;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t IsBSDFPacked = uint32_t(bxdf::traits<diffuse_op_type>::type == bxdf::BT_BSDF) << uint32_t(MaterialType::DIFFUSE) &
                                                        uint32_t(bxdf::traits<conductor_op_type>::type == bxdf::BT_BSDF) << uint32_t(MaterialType::CONDUCTOR) &
                                                        uint32_t(bxdf::traits<dielectric_op_type>::type == bxdf::BT_BSDF) << uint32_t(MaterialType::DIELECTRIC) &
                                                        uint32_t(bxdf::traits<iri_conductor_op_type>::type == bxdf::BT_BSDF) << uint32_t(MaterialType::IRIDESCENT_CONDUCTOR) &
                                                        uint32_t(bxdf::traits<iri_dielectric_op_type>::type == bxdf::BT_BSDF) << uint32_t(MaterialType::IRIDESCENT_DIELECTRIC);

    bool isBSDF(material_id_type matID)
    {
        MaterialType matType = (MaterialType)bxdfs[matID].materialType;
        return bool(IsBSDFPacked & (1u << matID));
    }

    // these are specific for the bxdfs used for this example
    void fillBxdfParams(material_id_type matID)
    {
        create_params_t cparams = bxdfs[matID].params;
        MaterialType matType = (MaterialType)bxdfs[matID].materialType;
        switch(matType)
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
            case MaterialType::IRIDESCENT_CONDUCTOR:
            {
                iridescentConductorBxDF.ndf = iri_conductor_op_type::ndf_type::create(cparams.A.x);
                using creation_params_t = typename iri_conductor_op_type::fresnel_type::creation_params_type;
                creation_params_t params;
                params.Dinc = cparams.A.y;
                params.ior1 = hlsl::promote<float32_t3>(1.0);
                params.ior2 = cparams.ior0;
                params.ior3 = cparams.ior1;
                params.iork3 = cparams.iork;
                iridescentConductorBxDF.fresnel = iri_conductor_op_type::fresnel_type::create(params);
            }
            break;
            case MaterialType::IRIDESCENT_DIELECTRIC:
            {
                iridescentDielectricBxDF.ndf = iri_dielectric_op_type::ndf_type::create(cparams.A.x);
                using creation_params_t = typename iri_dielectric_op_type::fresnel_type::creation_params_type;
                creation_params_t params;
                params.Dinc = cparams.A.y;
                params.ior1 = hlsl::promote<float32_t3>(1.0);
                params.ior2 = cparams.ior0;
                params.ior3 = cparams.ior1;
                iridescentDielectricBxDF.fresnel = iri_dielectric_op_type::fresnel_type::create(params);
            }
            break;
            default:
                return;
        }
    }

    measure_type eval(material_id_type matID, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) _cache)
    {
        fillBxdfParams(matID);
        MaterialType matType = (MaterialType)bxdfs[matID].materialType;
        switch(matType)
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
            case MaterialType::IRIDESCENT_CONDUCTOR:
            {
                return iridescentConductorBxDF.eval(_sample, interaction, _cache);
            }
            break;
            case MaterialType::IRIDESCENT_DIELECTRIC:
            {
                return iridescentDielectricBxDF.eval(_sample, interaction, _cache);
            }
            break;
            default:
                return hlsl::promote<measure_type>(0.0);
        }
    }

    sample_type generate(material_id_type matID, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector3_type) u, NBL_REF_ARG(anisocache_type) _cache)
    {
        fillBxdfParams(matID);
        MaterialType matType = (MaterialType)bxdfs[matID].materialType;
        switch(matType)
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
            case MaterialType::IRIDESCENT_CONDUCTOR:
            {
                return iridescentConductorBxDF.generate(interaction, u.xy, _cache);
            }
            break;
            case MaterialType::IRIDESCENT_DIELECTRIC:
            {
                return iridescentDielectricBxDF.generate(interaction, u, _cache);
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

    quotient_pdf_type quotient_and_pdf(material_id_type matID, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) _cache)
    {
        const float minimumProjVectorLen = 0.00000001;  // TODO: still need this check?
        if (interaction.getNdotV(bxdf::BxDFClampMode::BCM_ABS) > minimumProjVectorLen && _sample.getNdotL(bxdf::BxDFClampMode::BCM_ABS) > minimumProjVectorLen)
        {
            fillBxdfParams(matID);
            MaterialType matType = (MaterialType)bxdfs[matID].materialType;
            switch(matType)
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
                case MaterialType::IRIDESCENT_CONDUCTOR:
                {
                    return iridescentConductorBxDF.quotient_and_pdf(_sample, interaction, _cache);
                }
                break;
                case MaterialType::IRIDESCENT_DIELECTRIC:
                {
                    return iridescentDielectricBxDF.quotient_and_pdf(_sample, interaction, _cache);
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
    IridescentConductorBxDF iridescentConductorBxDF;
    IridescentDielectricBxDF iridescentDielectricBxDF;

    bxdfnode_type bxdfs[Scene::SCENE_BXDF_COUNT];
    uint32_t bxdfCount;
};

#endif
