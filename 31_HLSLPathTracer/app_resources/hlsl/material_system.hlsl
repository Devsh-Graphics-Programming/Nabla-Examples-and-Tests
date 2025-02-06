#ifndef _NBL_HLSL_EXT_MATERIAL_SYSTEM_INCLUDED_
#define _NBL_HLSL_EXT_MATERIAL_SYSTEM_INCLUDED_

#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace MaterialSystem
{

struct Material
{
    enum class Type : uint32_t
    {
        DIFFUSE,
        CONDUCTOR,
        DIELECTRIC
    };

    NBL_CONSTEXPR_STATIC_INLINE uint32_t DataSize = 32;

    uint32_t type : 1;
    unit32_t unused : 31;   // possible space for flags
    uint32_t data[DataSize];
};

template<class DiffuseBxDF, class ConductorBxDF, class DielectricBxDF>
struct System
{
    using this_t = System<DiffuseBxDF, ConductorBxDF, DielectricBxDF>;
    using scalar_type = typename DiffuseBxDF::scalar_type;      // types should be same across all 3 bxdfs
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using measure_type = typename DiffuseBxDF::spectral_type;
    using quotient_pdf_type = typename DiffuseBxDF::quotient_pdf_type;
    using anisotropic_type = typename DiffuseBxDF::anisotropic_type;
    using anisocache_type = typename ConductorBxDF::anisocache_type;
    using params_t = SBxDFParams<scalar_type>;

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, measure_type>) diffuseParams, NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, measure_type>) conductorParams, NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, measure_type>) dielectricParams)
    {
        diffuseBxDF = DiffuseBxDF::create(diffuseParams);
        conductorBxDF = DiffuseBxDF::create(conductorParams);
        dielectricBxDF = DiffuseBxDF::create(dielectricParams);
    }

    static measure_type eval(NBL_CONST_REF_ARG(Material) material, NBL_CONST_REF_ARG(params_t) params)
    {
        switch(material.type)
        {
            case DIFFUSE:
            {
                return (measure_type)diffuseBxDF.eval(params);
            }
            break;
            case CONDUCTOR:
            {
                return conductorBxDF.eval(params);
            }
            break;
            case DIELECTRIC:
            {
                return dielectricBxDF.eval(params);
            }
            break;
            default:
                return (measure_type)0.0;
        }
    }

    static vector3_type generate(NBL_CONST_REF_ARG(Material) material, anisotropic_type interaction, vector2_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        switch(material.type)
        {
            case DIFFUSE:
            {
                return diffuseBxDF.generate(interaction, u);
            }
            break;
            case CONDUCTOR:
            {
                return conductorBxDF.generate(interaction, u, cache);
            }
            break;
            case DIELECTRIC:
            {
                return dielectricBxDF.generate(interaction, u, cache);
            }
            break;
            default:
                return (vector3_type)numeric_limits<float>::infinity;
        }
    }

    static quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(Material) material, NBL_CONST_REF_ARG(params_t) params)
    {
        const float minimumProjVectorLen = 0.00000001;
        if (params.NdotV > minimumProjVectorLen && params.NdotL > minimumProjVectorLen)
        {
            switch(material.type)
            {
                case DIFFUSE:
                {
                    return diffuseBxDF.quotient_and_pdf(params);
                }
                break;
                case CONDUCTOR:
                {
                    return conductorBxDF.quotient_and_pdf(params);
                }
                break;
                case DIELECTRIC:
                {
                    return dielectricBxDF.quotient_and_pdf(params);
                }
                break;
                default:
                    return quotient_pdf_type::create((measure_type)0.0, numeric_limits<float>::infinity);
            }
        }
        return quotient_pdf_type::create((measure_type)0.0, numeric_limits<float>::infinity);
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