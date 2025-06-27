#ifndef _NBL_HLSL_EXT_MATERIAL_SYSTEM_INCLUDED_
#define _NBL_HLSL_EXT_MATERIAL_SYSTEM_INCLUDED_

#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/bxdf/common.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace MaterialSystem
{

enum MaterialType : uint32_t    // enum class?
{
    DIFFUSE,
    CONDUCTOR,
    DIELECTRIC
};

template<class DiffuseBxDF, class ConductorBxDF, class DielectricBxDF>
struct MaterialParams
{
    using this_t = MaterialParams<DiffuseBxDF, ConductorBxDF, DielectricBxDF>;
    using sample_type = typename DiffuseBxDF::sample_type;
    using anisotropic_interaction_type = typename DiffuseBxDF::anisotropic_interaction_type;
    using isotropic_interaction_type = typename anisotropic_interaction_type::isotropic_interaction_type;
    using anisocache_type = typename ConductorBxDF::anisocache_type;
    using isocache_type = typename anisocache_type::isocache_type;

    using diffuse_params_type = typename DiffuseBxDF::params_isotropic_t;
    using conductor_params_type = typename ConductorBxDF::params_isotropic_t;
    using dielectric_params_type = typename DielectricBxDF::params_isotropic_t;

    // we're only doing isotropic for this example
    static this_t create(sample_type _sample, isotropic_interaction_type _interaction, isocache_type _cache, bxdf::BxDFClampMode _clamp)
    {
        this_t retval;
        retval._Sample = _sample;
        retval.interaction = _interaction;
        retval.cache = _cache;
        retval.clampMode = _clamp;
        return retval;
    }

    diffuse_params_type getDiffuseParams()
    {
        return diffuse_params_type::create(_Sample, interaction, clampMode);
    }

    conductor_params_type getConductorParams()
    {
        return conductor_params_type::create(_Sample, interaction, cache, clampMode);
    }

    dielectric_params_type getDielectricParams()
    {
        return dielectric_params_type::create(_Sample, interaction, cache, clampMode);
    }

    sample_type _Sample;
    isotropic_interaction_type interaction;
    isocache_type cache;
    bxdf::BxDFClampMode clampMode;
};

template<class DiffuseBxDF, class ConductorBxDF, class DielectricBxDF>  // NOTE: these bxdfs should match the ones in Scene BxDFNode
struct System
{
    using this_t = System<DiffuseBxDF, ConductorBxDF, DielectricBxDF>;
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
    using params_t = MaterialParams<DiffuseBxDF, ConductorBxDF, DielectricBxDF>;
    using create_params_t = bxdf::SBxDFCreationParams<scalar_type, measure_type>;

    using diffuse_op_type = DiffuseBxDF;
    using conductor_op_type = ConductorBxDF;
    using dielectric_op_type = DielectricBxDF;

    static this_t create(NBL_CONST_REF_ARG(create_params_t) diffuseParams, NBL_CONST_REF_ARG(create_params_t) conductorParams, NBL_CONST_REF_ARG(create_params_t) dielectricParams)
    {
        this_t retval;
        retval.diffuseBxDF = diffuse_op_type::create(diffuseParams);
        retval.conductorBxDF = conductor_op_type::create(conductorParams);
        retval.dielectricBxDF = dielectric_op_type::create(dielectricParams);
        return retval;
    }

    measure_type eval(uint32_t material, NBL_CONST_REF_ARG(create_params_t) cparams, NBL_CONST_REF_ARG(params_t) params)
    {
        switch(material)
        {
            case MaterialType::DIFFUSE:
            {
                diffuseBxDF.init(cparams);
                return (measure_type)diffuseBxDF.eval(params.getDiffuseParams());
            }
            break;
            case MaterialType::CONDUCTOR:
            {
                conductorBxDF.init(cparams);
                return conductorBxDF.eval(params.getConductorParams());
            }
            break;
            case MaterialType::DIELECTRIC:
            {
                dielectricBxDF.init(cparams);
                return dielectricBxDF.eval(params.getDielectricParams());
            }
            break;
            default:
                return (measure_type)0.0;
        }
    }

    sample_type generate(uint32_t material, NBL_CONST_REF_ARG(create_params_t) cparams, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector3_type) u, NBL_REF_ARG(anisocache_type) _cache)
    {
        switch(material)
        {
            case MaterialType::DIFFUSE:
            {
                diffuseBxDF.init(cparams);
                return diffuseBxDF.generate(interaction, u.xy);
            }
            break;
            case MaterialType::CONDUCTOR:
            {
                conductorBxDF.init(cparams);
                return conductorBxDF.generate(interaction, u.xy, _cache);
            }
            break;
            case MaterialType::DIELECTRIC:
            {
                dielectricBxDF.init(cparams);
                return dielectricBxDF.generate(interaction, u, _cache);
            }
            break;
            default:
            {
                ray_dir_info_type L;
                L.direction = (vector3_type)0;
                return sample_type::create(L, 0, (vector3_type)0);
            }
        }

        ray_dir_info_type L;
        L.direction = (vector3_type)0;
        return sample_type::create(L, 0, (vector3_type)0);
    }

    quotient_pdf_type quotient_and_pdf(uint32_t material, NBL_CONST_REF_ARG(create_params_t) cparams, NBL_CONST_REF_ARG(params_t) params)
    {
        const float minimumProjVectorLen = 0.00000001;
        if (params.interaction.getNdotV() > minimumProjVectorLen && params._Sample.getNdotL() > minimumProjVectorLen)
        {
            switch(material)
            {
                case MaterialType::DIFFUSE:
                {
                    diffuseBxDF.init(cparams);
                    return diffuseBxDF.quotient_and_pdf(params.getDiffuseParams());
                }
                break;
                case MaterialType::CONDUCTOR:
                {
                    conductorBxDF.init(cparams);
                    return conductorBxDF.quotient_and_pdf(params.getConductorParams());
                }
                break;
                case MaterialType::DIELECTRIC:
                {
                    dielectricBxDF.init(cparams);
                    return dielectricBxDF.quotient_and_pdf(params.getDielectricParams());
                }
                break;
                default:
                    return quotient_pdf_type::create((measure_type)0.0, 0.0);
            }
        }
        return quotient_pdf_type::create((measure_type)0.0, 0.0);
    }

    DiffuseBxDF diffuseBxDF;
    ConductorBxDF conductorBxDF;
    DielectricBxDF dielectricBxDF;
};

}
}
}
}

#endif
