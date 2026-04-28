#ifndef _PATHTRACER_EXAMPLE_MATERIAL_SYSTEM_INCLUDED_
#define _PATHTRACER_EXAMPLE_MATERIAL_SYSTEM_INCLUDED_

#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/bxdf/common.hlsl>

#include "example_common.hlsl"

using namespace nbl;
using namespace hlsl;

template<class BxDFNode, class DiffuseBxDF, class ConductorBxDF, class DielectricBxDF, class IridescentConductorBxDF, class IridescentDielectricBxDF, class NormalMappedDiffuseBxDF, class Scene>  // NOTE: these bxdfs should match the ones in Scene BxDFNode, TODO: THEN TAKE THEM FROM THE SCENE!
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
                _cache.diffuseToData(diffuse_op_type::create(params));
            }
            break;
            case MaterialType::CONDUCTOR:
            {
                conductor_op_type bxdf;
                bxdf.ndf = conductor_op_type::ndf_type::create(cparams.A.x);
                bxdf.fresnel = conductor_op_type::fresnel_type::create(cparams.ior0,cparams.ior1);
                _cache.conductorToData(bxdf);
            }
            break;
            case MaterialType::DIELECTRIC:
            {
                using oriented_eta_t = bxdf::fresnel::OrientedEtas<typename dielectric_op_type::monochrome_type>;
                oriented_eta_t orientedEta = oriented_eta_t::create(1.0, hlsl::promote<typename dielectric_op_type::monochrome_type>(cparams.eta));
                dielectric_op_type bxdf;
                bxdf.ndf = dielectric_op_type::ndf_type::create(cparams.A.x);
                bxdf.fresnel = dielectric_op_type::fresnel_type::create(orientedEta);
                _cache.dielectricToData(bxdf);
            }
            break;
            case MaterialType::IRIDESCENT_CONDUCTOR:
            {
                using creation_params_t = typename iri_conductor_op_type::fresnel_type::creation_params_type;
                creation_params_t params;
                params.Dinc = cparams.A.y;
                params.ior1 = hlsl::promote<float32_t3>(1.0);
                params.ior2 = cparams.ior0;
                params.ior3 = cparams.ior1;
                params.iork3 = cparams.iork;
                iri_conductor_op_type bxdf;
                bxdf.ndf = iri_conductor_op_type::ndf_type::create(cparams.A.x);
                bxdf.fresnel = iri_conductor_op_type::fresnel_type::create(params);
                _cache.iridescentConductorToData(bxdf);
            }
            break;
            case MaterialType::IRIDESCENT_DIELECTRIC:
            {
                using creation_params_t = typename iri_dielectric_op_type::fresnel_type::creation_params_type;
                creation_params_t params;
                params.Dinc = cparams.A.y;
                params.ior1 = hlsl::promote<float32_t3>(1.0);
                params.ior2 = cparams.ior0;
                params.ior3 = cparams.ior1;
                iri_dielectric_op_type bxdf;
                bxdf.ndf = iri_dielectric_op_type::ndf_type::create(cparams.A.x);
                bxdf.fresnel = iri_dielectric_op_type::fresnel_type::create(params);
                _cache.iridescentDielectricToData(bxdf);
            }
            break;
            default:
                return;
        }
    }

    value_weight_type evalAndWeight(material_id_type matID, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        // TODO: call only fillBxdfParams, should probably split the cache away from the bxdf node
        cache_type _cache;
        fillBxdfParams(matID, _cache);
        MaterialType matType = (MaterialType)bxdfs[matID.id].materialType;
        switch(matType)
        {
            case MaterialType::DIFFUSE:
            {
                diffuse_op_type bxdf = _cache.dataToDiffuse();
                value_weight_type ret = bxdf.evalAndWeight(_sample, interaction.isotropic);
                ret._value *= bxdfs[matID.id].albedo;
                return ret;
            }
            case MaterialType::CONDUCTOR:
            {
                conductor_op_type bxdf = _cache.dataToConductor();
                return bxdf.evalAndWeight(_sample, interaction.isotropic);
            }
            case MaterialType::DIELECTRIC:
            {
                dielectric_op_type bxdf = _cache.dataToDielectric();
                return bxdf.evalAndWeight(_sample, interaction.isotropic);
            }
            case MaterialType::IRIDESCENT_CONDUCTOR:
            {
                iri_conductor_op_type bxdf = _cache.dataToIridescentConductor();
                return bxdf.evalAndWeight(_sample, interaction.isotropic);
            }
            case MaterialType::IRIDESCENT_DIELECTRIC:
            {
                iri_dielectric_op_type bxdf = _cache.dataToIridescentDielectric();
                return bxdf.evalAndWeight(_sample, interaction.isotropic);
            }
            case MaterialType::NORMAL_MAPPED_DIFFUSE:
            {
                normal_mapped_diffuse_op_type bxdf;
                bxdf.shadingNormal = interaction.getN();
                bxdf.shadingBasis = interaction.getToTangentSpace();
                typename normal_mapped_diffuse_op_type::bxdf_type nested_brdf;
                bxdf.nested_brdf = nested_brdf;
                anisotropic_interaction_type interaction_Np = bxdf.template buildInteraction<typename bxdfnode_type::normals_accessor>(bxdfs[matID.id].normals, interaction.getIntersectUV(), interaction.getFromTangentSpace(), interaction.getV());
                sample_type sample_Np = sample_type::create(_sample.getL(), interaction_Np.getN());
                value_weight_type ret = bxdf.evalAndWeight(sample_Np, interaction_Np.isotropic);
                ret._value *= bxdfs[matID.id].albedo;
                return ret;
            }
            default:
                return value_weight_type::create(0.0, 0.0);
        }
    }

    sample_type generate(material_id_type matID, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector3_type) u, NBL_REF_ARG(cache_type) _cache)
    {
        // TODO: should probably split the caches, no aniso cache needed (generate should overwrite it
        fillBxdfParams(matID, _cache);
        MaterialType matType = (MaterialType)bxdfs[matID.id].materialType;
        switch(matType)
        {
            case MaterialType::DIFFUSE:
            {
                typename diffuse_op_type::isocache_type dummycache;
                diffuse_op_type bxdf = _cache.dataToDiffuse();
                return bxdf.generate(interaction, u.xy, dummycache);
            }
            case MaterialType::CONDUCTOR:
            {
                conductor_op_type bxdf = _cache.dataToConductor();
                return bxdf.generate(interaction, u.xy, _cache.aniso_cache);
            }
            case MaterialType::DIELECTRIC:
            {
                dielectric_op_type bxdf = _cache.dataToDielectric();
                return bxdf.generate(interaction, u, _cache.aniso_cache);
            }
            case MaterialType::IRIDESCENT_CONDUCTOR:
            {
                iri_conductor_op_type bxdf = _cache.dataToIridescentConductor();
                return bxdf.generate(interaction, u.xy, _cache.aniso_cache);
            }
            case MaterialType::IRIDESCENT_DIELECTRIC:
            {
                iri_dielectric_op_type bxdf = _cache.dataToIridescentDielectric();
                return bxdf.generate(interaction, u, _cache.aniso_cache);
            }
            case MaterialType::NORMAL_MAPPED_DIFFUSE:
            {
                normal_mapped_diffuse_op_type bxdf;
                bxdf.shadingNormal = interaction.getN();
                bxdf.shadingBasis = interaction.getToTangentSpace();
                typename normal_mapped_diffuse_op_type::bxdf_type nested_brdf;
                bxdf.nested_brdf = nested_brdf;
                anisotropic_interaction_type interaction_Np = bxdf.template buildInteraction<typename bxdfnode_type::normals_accessor>(bxdfs[matID.id].normals, interaction.getIntersectUV(), interaction.getFromTangentSpace(), interaction.getV());
                typename normal_mapped_diffuse_op_type::isocache_type cache;
                sample_type s = bxdf.generate(interaction_Np, u.xy, cache);
                _cache.sampleFromNt = cache.sampleFromNt;
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
                    typename diffuse_op_type::isocache_type dummycache; // diffuse doesn't actually have a cache (struct is empty)
                    diffuse_op_type bxdf = _cache.dataToDiffuse();
                    quotient_weight_type ret = bxdf.quotientAndWeight(_sample, interaction.isotropic, dummycache);
                    ret._quotient *= bxdfs[matID.id].albedo;
                    return ret;
                }
                case MaterialType::CONDUCTOR:
                {
                    conductor_op_type bxdf = _cache.dataToConductor();
                    return bxdf.quotientAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
                }
                case MaterialType::DIELECTRIC:
                {
                    dielectric_op_type bxdf = _cache.dataToDielectric();
                    return bxdf.quotientAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
                }
                case MaterialType::IRIDESCENT_CONDUCTOR:
                {
                    iri_conductor_op_type bxdf = _cache.dataToIridescentConductor();
                    return bxdf.quotientAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
                }
                case MaterialType::IRIDESCENT_DIELECTRIC:
                {
                    iri_dielectric_op_type bxdf = _cache.dataToIridescentDielectric();
                    return bxdf.quotientAndWeight(_sample, interaction.isotropic, _cache.aniso_cache.iso_cache);
                }
                case MaterialType::NORMAL_MAPPED_DIFFUSE:
                {
                    normal_mapped_diffuse_op_type bxdf;
                    bxdf.shadingNormal = interaction.getN();
                    bxdf.shadingBasis = interaction.getToTangentSpace();
                    typename normal_mapped_diffuse_op_type::bxdf_type nested_brdf;
                    bxdf.nested_brdf = nested_brdf;
                    anisotropic_interaction_type interaction_Np = bxdf.template buildInteraction<typename bxdfnode_type::normals_accessor>(bxdfs[matID.id].normals, interaction.getIntersectUV(), interaction.getFromTangentSpace(), interaction.getV());
                    typename normal_mapped_diffuse_op_type::isocache_type cache;
                    cache.sampleFromNt = _cache.sampleFromNt;
                    cache.sampleIsShadowed = _cache.sampleIsShadowed;
                    sample_type sample_Np = sample_type::create(_sample.getL(), interaction_Np.getN());
                    quotient_weight_type ret = bxdf.quotientAndWeight(sample_Np, interaction_Np.isotropic, cache);
                    ret._quotient *= bxdfs[matID.id].albedo;
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
