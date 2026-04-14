#ifndef _PATHTRACER_EXAMPLE_MATERIAL_SYSTEM_INCLUDED_
#define _PATHTRACER_EXAMPLE_MATERIAL_SYSTEM_INCLUDED_

#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/bxdf/common.hlsl>

#include "example_common.hlsl"

using namespace nbl;
using namespace hlsl;

template<class BxDFNode, class DiffuseBxDF, class ConductorBxDF, class DielectricBxDF, class IridescentConductorBxDF, class IridescentDielectricBxDF, class NormalMappedDiffuseBxDF, class Scene>  // NOTE: these bxdfs should match the ones in Scene BxDFNode
struct MaterialSystem
{
    using this_t = MaterialSystem<BxDFNode, DiffuseBxDF, ConductorBxDF, DielectricBxDF, IridescentConductorBxDF, IridescentDielectricBxDF, NormalMappedDiffuseBxDF, Scene>;
    using scalar_type = typename DiffuseBxDF::scalar_type;      // types should be same across all bxdfs
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using material_id_type = MaterialID;
    using measure_type = typename DiffuseBxDF::spectral_type;
    using sample_type = typename DiffuseBxDF::sample_type;
    using ray_dir_info_type = typename sample_type::ray_dir_info_type;
    using quotient_weight_type = typename DiffuseBxDF::quotient_weight_type;
    using value_weight_type = typename DiffuseBxDF::value_weight_type;
    using anisotropic_interaction_type = typename DiffuseBxDF::anisotropic_interaction_type;
    using isotropic_interaction_type = typename anisotropic_interaction_type::isotropic_interaction_type;
    using anisocache_type = typename ConductorBxDF::anisocache_type;
    using isocache_type = typename anisocache_type::isocache_type;
    using cache_type = PTMaterialSystemCache<isocache_type, anisocache_type, DiffuseBxDF, ConductorBxDF, DielectricBxDF, IridescentConductorBxDF, IridescentDielectricBxDF, NormalMappedDiffuseBxDF>;
    using create_params_t = SBxDFCreationParams<scalar_type, measure_type>;

    using bxdfnode_type = BxDFNode;
    using diffuse_op_type = DiffuseBxDF;
    using conductor_op_type = ConductorBxDF;
    using dielectric_op_type = DielectricBxDF;
    using iri_conductor_op_type = IridescentConductorBxDF;
    using iri_dielectric_op_type = IridescentDielectricBxDF;
    using normal_mapped_diffuse_op_type = NormalMappedDiffuseBxDF;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t IsBSDFPacked = uint32_t(bxdf::traits<diffuse_op_type>::type == bxdf::BT_BSDF) << uint32_t(MaterialType::DIFFUSE) |
                                                        uint32_t(bxdf::traits<conductor_op_type>::type == bxdf::BT_BSDF) << uint32_t(MaterialType::CONDUCTOR) |
                                                        uint32_t(bxdf::traits<dielectric_op_type>::type == bxdf::BT_BSDF) << uint32_t(MaterialType::DIELECTRIC) |
                                                        uint32_t(bxdf::traits<iri_conductor_op_type>::type == bxdf::BT_BSDF) << uint32_t(MaterialType::IRIDESCENT_CONDUCTOR) |
                                                        uint32_t(bxdf::traits<iri_dielectric_op_type>::type == bxdf::BT_BSDF) << uint32_t(MaterialType::IRIDESCENT_DIELECTRIC) |
                                                        uint32_t(bxdf::traits<normal_mapped_diffuse_op_type>::type == bxdf::BT_BSDF) << uint32_t(MaterialType::NORMAL_MAPPED_DIFFUSE);

    bool isBSDF(material_id_type matID)
    {
        MaterialType matType = (MaterialType)bxdfs[matID.id].materialType;
        return bool(IsBSDFPacked & (1u << matID.id));
    }

    bxdfnode_type getBxDFNode(material_id_type matID, NBL_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        interaction.isotropic.b_isMaterialBSDF = isBSDF(matID);
        return bxdfs[matID.id];
    }

    scalar_type setMonochromeEta(material_id_type matID, measure_type throughputCIE_Y)
    {
        bxdfnode_type bxdf = bxdfs[matID.id];
        const measure_type eta = bxdf.params.ior1 / bxdf.params.ior0;
        const scalar_type monochromeEta = hlsl::dot<vector3_type>(throughputCIE_Y, eta) / (throughputCIE_Y.r + throughputCIE_Y.g + throughputCIE_Y.b);  // TODO: imaginary eta?
        bxdfs[matID.id].params.eta = monochromeEta;
        return monochromeEta;
    }

    cache_type getCacheFromSampleInteraction(material_id_type matID, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        const scalar_type monochromeEta = setMonochromeEta(matID, interaction.getLuminosityContributionHint());
        using monochrome_type = typename dielectric_op_type::monochrome_type;
        bxdf::fresnel::OrientedEtas<monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<monochrome_type>::create(interaction.getNdotV(), hlsl::promote<monochrome_type>(monochromeEta));
        cache_type _cache;
        _cache.aniso_cache = anisocache_type::template create<anisotropic_interaction_type, sample_type>(interaction, _sample, orientedEta);
        fillBxdfParams(matID, _cache);
        return _cache;
    }

    // these are specific for the bxdfs used for this example
    void fillBxdfParams(material_id_type matID, NBL_REF_ARG(cache_type) _cache)
    {
        create_params_t cparams = bxdfs[matID.id].params;
        MaterialType matType = (MaterialType)bxdfs[matID.id].materialType;
        switch(matType)
        {
            case MaterialType::DIFFUSE:
            {
                using creation_t = typename diffuse_op_type::creation_type;
                creation_t params;
                params.A = cparams.A.x;
                _cache.diffuseBxDF = diffuse_op_type::create(params);
            }
            break;
            case MaterialType::CONDUCTOR:
            {
                _cache.conductorBxDF.ndf = conductor_op_type::ndf_type::create(cparams.A.x);
                _cache.conductorBxDF.fresnel = conductor_op_type::fresnel_type::create(cparams.ior0,cparams.ior1);
            }
            break;
            case MaterialType::DIELECTRIC:
            {
                using oriented_eta_t = bxdf::fresnel::OrientedEtas<typename dielectric_op_type::monochrome_type>;
                oriented_eta_t orientedEta = oriented_eta_t::create(1.0, hlsl::promote<typename dielectric_op_type::monochrome_type>(cparams.eta));
                _cache.dielectricBxDF.ndf = dielectric_op_type::ndf_type::create(cparams.A.x);
                _cache.dielectricBxDF.fresnel = dielectric_op_type::fresnel_type::create(orientedEta);
            }
            break;
            case MaterialType::IRIDESCENT_CONDUCTOR:
            {
                _cache.iridescentConductorBxDF.ndf = iri_conductor_op_type::ndf_type::create(cparams.A.x);
                using creation_params_t = typename iri_conductor_op_type::fresnel_type::creation_params_type;
                creation_params_t params;
                params.Dinc = cparams.A.y;
                params.ior1 = hlsl::promote<float32_t3>(1.0);
                params.ior2 = cparams.ior0;
                params.ior3 = cparams.ior1;
                params.iork3 = cparams.iork;
                _cache.iridescentConductorBxDF.fresnel = iri_conductor_op_type::fresnel_type::create(params);
            }
            break;
            case MaterialType::IRIDESCENT_DIELECTRIC:
            {
                _cache.iridescentDielectricBxDF.ndf = iri_dielectric_op_type::ndf_type::create(cparams.A.x);
                using creation_params_t = typename iri_dielectric_op_type::fresnel_type::creation_params_type;
                creation_params_t params;
                params.Dinc = cparams.A.y;
                params.ior1 = hlsl::promote<float32_t3>(1.0);
                params.ior2 = cparams.ior0;
                params.ior3 = cparams.ior1;
                _cache.iridescentDielectricBxDF.fresnel = iri_dielectric_op_type::fresnel_type::create(params);
            }
            break;
            default:
                return;
        }
    }

    value_weight_type evalAndWeight(material_id_type matID, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        cache_type _cache = getCacheFromSampleInteraction(matID, _sample, interaction);
        MaterialType matType = (MaterialType)bxdfs[matID.id].materialType;
        switch(matType)
        {
            case MaterialType::DIFFUSE:
            {
                value_weight_type ret = _cache.diffuseBxDF.evalAndWeight(_sample, interaction.isotropic);
                ret._value *= bxdfs[matID.id].albedo;
                return ret;
            }
            case MaterialType::CONDUCTOR:
            {
                return _cache.conductorBxDF.evalAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
            }
            case MaterialType::DIELECTRIC:
            {
                return _cache.dielectricBxDF.evalAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
            }
            case MaterialType::IRIDESCENT_CONDUCTOR:
            {
                return _cache.iridescentConductorBxDF.evalAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
            }
            case MaterialType::IRIDESCENT_DIELECTRIC:
            {
                return _cache.iridescentDielectricBxDF.evalAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
            }
            case MaterialType::NORMAL_MAPPED_DIFFUSE:
            {
                _cache.normalMappedDiffuseBxDF.shadingNormal = interaction.getN();
                _cache.normalMappedDiffuseBxDF.shadingBasis = interaction.getToTangentSpace();
                typename normal_mapped_diffuse_op_type::bxdf_type nested_brdf;
                _cache.normalMappedDiffuseBxDF.nested_brdf = nested_brdf;
                anisotropic_interaction_type interaction_Np = _cache.normalMappedDiffuseBxDF.template buildInteraction<typename bxdfnode_type::normals_accessor>(bxdfs[matID.id].normals, interaction.getIntersectUV(), interaction.getFromTangentSpace(), interaction.getV());
                value_weight_type ret = _cache.normalMappedDiffuseBxDF.evalAndWeight(_sample, interaction_Np.isotropic);
                ret._value *= bxdfs[matID.id].albedo;
                
                // vector3_type localN;
                // bxdfs[matID.id].normals.get(localN, interaction.getIntersectUV());
                // localN = hlsl::promote<vector3_type>(2.0) * localN - hlsl::promote<vector3_type>(1.0);
                // localN = hlsl::normalize(hlsl::mul(interaction.getFromTangentSpace(), localN));
                // ret._value = hlsl::promote<vector3_type>(0.5) * localN + hlsl::promote<vector3_type>(0.5);

                // vector3_type localN;
                // bxdfs[matID.id].normals.get(localN, interaction.getIntersectUV());
                // ret._value = localN;

                return ret;
            }
            default:
                return value_weight_type::create(0.0, 0.0);
        }
    }

    sample_type generate(material_id_type matID, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector3_type) u, NBL_REF_ARG(cache_type) _cache)
    {
        fillBxdfParams(matID, _cache);
        MaterialType matType = (MaterialType)bxdfs[matID.id].materialType;
        switch(matType)
        {
            case MaterialType::DIFFUSE:
            {
                typename diffuse_op_type::isocache_type dummycache;
                return _cache.diffuseBxDF.generate(interaction, u.xy, dummycache);
            }
            case MaterialType::CONDUCTOR:
            {
                return _cache.conductorBxDF.generate(interaction, u.xy, _cache.aniso_cache);
            }
            case MaterialType::DIELECTRIC:
            {
                return _cache.dielectricBxDF.generate(interaction, u, _cache.aniso_cache);
            }
            case MaterialType::IRIDESCENT_CONDUCTOR:
            {
                return _cache.iridescentConductorBxDF.generate(interaction, u.xy, _cache.aniso_cache);
            }
            case MaterialType::IRIDESCENT_DIELECTRIC:
            {
                return _cache.iridescentDielectricBxDF.generate(interaction, u, _cache.aniso_cache);
            }
            case MaterialType::NORMAL_MAPPED_DIFFUSE:
            {
                _cache.normalMappedDiffuseBxDF.shadingNormal = interaction.getN();
                _cache.normalMappedDiffuseBxDF.shadingBasis = interaction.getToTangentSpace();
                typename normal_mapped_diffuse_op_type::bxdf_type nested_brdf;
                _cache.normalMappedDiffuseBxDF.nested_brdf = nested_brdf;
                anisotropic_interaction_type interaction_Np = _cache.normalMappedDiffuseBxDF.template buildInteraction<typename bxdfnode_type::normals_accessor>(bxdfs[matID.id].normals, interaction.getIntersectUV(), interaction.getFromTangentSpace(), interaction.getV());
                typename normal_mapped_diffuse_op_type::isocache_type cache;
                sample_type s = _cache.normalMappedDiffuseBxDF.generate(interaction_Np.isotropic, u.xy, cache);
                _cache.sampleIsShadowed = cache.sampleIsShadowed;
                return s;
            }
            default:
            {
                ray_dir_info_type L;
                L.makeInvalid();
                return sample_type::create(L, hlsl::promote<vector3_type>(0.0));
            }
        }
    }

    scalar_type pdf(material_id_type matID, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        cache_type _cache = getCacheFromSampleInteraction(matID, _sample, interaction);
        MaterialType matType = (MaterialType)bxdfs[matID.id].materialType;
        switch(matType)
        {
            case MaterialType::DIFFUSE:
            {
                return _cache.diffuseBxDF.forwardPdf(_sample, interaction.isotropic);
            }
            case MaterialType::CONDUCTOR:
            {
                return _cache.conductorBxDF.forwardPdf(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
            }
            case MaterialType::DIELECTRIC:
            {
                return _cache.dielectricBxDF.forwardPdf(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
            }
            case MaterialType::IRIDESCENT_CONDUCTOR:
            {
                return _cache.iridescentConductorBxDF.forwardPdf(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
            }
            case MaterialType::IRIDESCENT_DIELECTRIC:
            {
                return _cache.iridescentDielectricBxDF.forwardPdf(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
            }
            case MaterialType::NORMAL_MAPPED_DIFFUSE:
            {
                _cache.normalMappedDiffuseBxDF.shadingNormal = interaction.getN();
                _cache.normalMappedDiffuseBxDF.shadingBasis = interaction.getToTangentSpace();
                typename normal_mapped_diffuse_op_type::bxdf_type nested_brdf;
                _cache.normalMappedDiffuseBxDF.nested_brdf = nested_brdf;
                anisotropic_interaction_type interaction_Np = _cache.normalMappedDiffuseBxDF.template buildInteraction<typename bxdfnode_type::normals_accessor>(bxdfs[matID.id].normals, interaction.getIntersectUV(), interaction.getFromTangentSpace(), interaction.getV());
                return _cache.normalMappedDiffuseBxDF.forwardPdf(_sample, interaction_Np.isotropic);
            }
            default:
                return scalar_type(0.0);
        }
    }

    quotient_weight_type quotientAndWeight(material_id_type matID, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(cache_type) _cache)
    {
        const float minimumProjVectorLen = 0.00000001;  // TODO: still need this check?
        if (interaction.getNdotV(bxdf::BxDFClampMode::BCM_ABS) > minimumProjVectorLen && _sample.getNdotL(bxdf::BxDFClampMode::BCM_ABS) > minimumProjVectorLen)
        {
            MaterialType matType = (MaterialType)bxdfs[matID.id].materialType;
            switch(matType)
            {
                case MaterialType::DIFFUSE:
                {
                    typename diffuse_op_type::isocache_type dummycache;
                    quotient_weight_type ret = _cache.diffuseBxDF.quotientAndWeight(_sample, interaction.isotropic, dummycache);
                    ret._quotient *= bxdfs[matID.id].albedo;
                    return ret;
                }
                case MaterialType::CONDUCTOR:
                {
                    return _cache.conductorBxDF.quotientAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
                }
                case MaterialType::DIELECTRIC:
                {
                    return _cache.dielectricBxDF.quotientAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
                }
                case MaterialType::IRIDESCENT_CONDUCTOR:
                {
                    return _cache.iridescentConductorBxDF.quotientAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
                }
                case MaterialType::IRIDESCENT_DIELECTRIC:
                {
                    return _cache.iridescentDielectricBxDF.quotientAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
                }
                case MaterialType::NORMAL_MAPPED_DIFFUSE:
                {
                    _cache.normalMappedDiffuseBxDF.shadingNormal = interaction.getN();
                    _cache.normalMappedDiffuseBxDF.shadingBasis = interaction.getToTangentSpace();
                    typename normal_mapped_diffuse_op_type::bxdf_type nested_brdf;
                    _cache.normalMappedDiffuseBxDF.nested_brdf = nested_brdf;

                    anisotropic_interaction_type interaction_Np = _cache.normalMappedDiffuseBxDF.template buildInteraction<typename bxdfnode_type::normals_accessor>(bxdfs[matID.id].normals, interaction.getIntersectUV(), interaction.getFromTangentSpace(), interaction.getV());
                    typename normal_mapped_diffuse_op_type::isocache_type cache;
                    cache.sampleIsShadowed = _cache.sampleIsShadowed;
                    quotient_weight_type ret = _cache.normalMappedDiffuseBxDF.quotientAndWeight(_sample, interaction_Np.isotropic, cache);
                    ret._quotient *= bxdfs[matID.id].albedo;

                    // vector3_type localN;
                    // bxdfs[matID.id].normals.get(localN, interaction.getIntersectUV());
                    // localN = hlsl::promote<vector3_type>(2.0) * localN - hlsl::promote<vector3_type>(1.0);
                    // localN = hlsl::normalize(hlsl::mul(interaction.getFromTangentSpace(), localN));
                    // ret._quotient = hlsl::promote<vector3_type>(0.5) * localN + hlsl::promote<vector3_type>(0.5);

                    // vector3_type localN;
                    // bxdfs[matID.id].normals.get(localN, interaction.getIntersectUV());
                    // ret._quotient = localN;

                    return ret;
                }
                default:
                    break;
            }
        }
        return quotient_weight_type::create(0.0, 0.0);
    }

    bool hasEmission(material_id_type matID)
    {
        MaterialType matType = (MaterialType)bxdfs[matID.id].materialType;
        return matType == MaterialType::EMISSIVE;
    }

    measure_type getEmission(material_id_type matID, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        if (hasEmission(matID))
            return bxdfs[matID.id].albedo;
        return hlsl::promote<measure_type>(0.0);
    }

    bxdfnode_type bxdfs[Scene::SCENE_BXDF_COUNT];
};

#endif
