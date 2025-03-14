#ifndef _NBL_HLSL_EXT_NEXT_EVENT_ESTIMATOR_INCLUDED_
#define _NBL_HLSL_EXT_NEXT_EVENT_ESTIMATOR_INCLUDED_

#include "common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace NextEventEstimator
{

template<typename Light, typename Ray, class LightSample, class Aniso>
struct Estimator
{
    using scalar_type = typename Ray::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using ray_type = Ray;
    using light_type = Light;
    using spectral_type = typename Light::spectral_type;
    using interaction_type = Aniso;
    using quotient_pdf_type = bxdf::quotient_and_pdf<spectral_type, scalar_type>;
    using sample_type = LightSample;
    using ray_dir_info_type = typename sample_type::ray_dir_info_type;

    static spectral_type proceduralDeferredEvalAndPdf(NBL_REF_ARG(scalar_type) pdf, NBL_CONST_REF_ARG(light_type) light, NBL_CONST_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(Event) event)
    {
        const uint32_t lightCount = event.data[0];
        const ProceduralShapeType type = (ProceduralShapeType)event.data[1];

        pdf = 1.0 / lightCount;
        switch (type)
        {
            case PST_SPHERE:
            {
                const vector3_type position = vector3_type(
                    bit_cast<float32_t>(event.data[2]),
                    bit_cast<float32_t>(event.data[3]),
                    bit_cast<float32_t>(event.data[4]));
                Shape<PST_SPHERE> sphere = Shape<PST_SPHERE>::create(position, bit_cast<float32_t>(event.data[5]), event.data[6]);
                pdf *= sphere.template deferredPdf<ray_type>(ray);
            }
            break;
            case PST_TRIANGLE:
            {
                const vector3_type vertex0 = vector3_type(
                    bit_cast<float32_t>(event.data[2]),
                    bit_cast<float32_t>(event.data[3]),
                    bit_cast<float32_t>(event.data[4]));
                const vector3_type vertex1 = vector3_type(
                    bit_cast<float32_t>(event.data[5]), 
                    bit_cast<float32_t>(event.data[6]),
                    bit_cast<float32_t>(event.data[7]));
                const vector3_type vertex2 = vector3_type(
                    bit_cast<float32_t>(event.data[8]),
                    bit_cast<float32_t>(event.data[9]),
                    bit_cast<float32_t>(event.data[10]));
                Shape<PST_TRIANGLE> tri = Shape<PST_TRIANGLE>::create(vertex0, vertex1, vertex2, event.data[11]);
                pdf *= tri.template deferredPdf<ray_type>(ray);
            }
            break;
            case PST_RECTANGLE:
            {
                const vector3_type offset = vector3_type(
                    bit_cast<float32_t>(event.data[2]),
                    bit_cast<float32_t>(event.data[3]),
                    bit_cast<float32_t>(event.data[4]));
                const vector3_type edge0 = vector3_type(
                    bit_cast<float32_t>(event.data[5]),
                    bit_cast<float32_t>(event.data[6]),
                    bit_cast<float32_t>(event.data[7]));
                const vector3_type edge1 = vector3_type(
                    bit_cast<float32_t>(event.data[8]),
                    bit_cast<float32_t>(event.data[9]),
                    bit_cast<float32_t>(event.data[10]));
                Shape<PST_RECTANGLE> rect = Shape<PST_RECTANGLE>::create(offset, edge0, edge1, event.data[11]);
                pdf *= rect.template deferredPdf<ray_type>(ray);
            }
            break;
            default:
                pdf = bit_cast<float>(numeric_limits<float>::infinity);
                break;
        }

        return light.radiance;
    }

    static spectral_type deferredEvalAndPdf(NBL_REF_ARG(scalar_type) pdf, NBL_CONST_REF_ARG(light_type) light, NBL_CONST_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(Event) event)
    {
        // const IntersectMode mode = (IntersectMode)event.mode;
        // switch (mode)
        // {
        //     case IM_RAY_QUERY:
        //     {
        //         // TODO: do ray query stuff
        //     }
        //     break;
        //     case IM_RAY_TRACING:
        //     {
        //         // TODO: do ray tracing stuff
        //     }
        //     break;
        //     case IM_PROCEDURAL:
        //     {
                return proceduralDeferredEvalAndPdf(pdf, light, ray, event);
        //     }
        //     break;
        //     default:
        //         return (spectral_type)0.0;
        // }
        // return (spectral_type)0.0;
    }

    static sample_type procedural_generate_and_quotient_and_pdf(NBL_REF_ARG(quotient_pdf_type) quotient_pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(light_type) light, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(interaction_type) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi, uint32_t depth, NBL_CONST_REF_ARG(Event) event)
    {
        const uint32_t lightCount = event.data[0];
        const ProceduralShapeType type = (ProceduralShapeType)event.data[1];

        sample_type L;
        scalar_type pdf;
        switch (type)
        {
            case PST_SPHERE:
            {
                const vector3_type position = vector3_type(
                    bit_cast<float32_t>(event.data[2]),
                    bit_cast<float32_t>(event.data[3]),
                    bit_cast<float32_t>(event.data[4]));
                Shape<PST_SPHERE> sphere = Shape<PST_SPHERE>::create(position, bit_cast<float32_t>(event.data[5]), event.data[6]);

                const vector3_type sampleL = sphere.template generate_and_pdf<interaction_type>(pdf, newRayMaxT, origin, interaction, isBSDF, xi);
                const vector3_type V = interaction.isotropic.V.getDirection();
                const scalar_type VdotL = nbl::hlsl::dot<vector3_type>(V, sampleL);
                ray_dir_info_type rayL;
                rayL.direction = sampleL;
                L = sample_type::create(rayL,VdotL,interaction.T,interaction.B,interaction.isotropic.N);
            }
            break;
            case PST_TRIANGLE:
            {
                const vector3_type vertex0 = vector3_type(
                    bit_cast<float32_t>(event.data[2]),
                    bit_cast<float32_t>(event.data[3]),
                    bit_cast<float32_t>(event.data[4]));
                const vector3_type vertex1 = vector3_type(
                    bit_cast<float32_t>(event.data[5]), 
                    bit_cast<float32_t>(event.data[6]),
                    bit_cast<float32_t>(event.data[7]));
                const vector3_type vertex2 = vector3_type(
                    bit_cast<float32_t>(event.data[8]),
                    bit_cast<float32_t>(event.data[9]),
                    bit_cast<float32_t>(event.data[10]));
                Shape<PST_TRIANGLE> tri = Shape<PST_TRIANGLE>::create(vertex0, vertex1, vertex2, event.data[11]);

                const vector3_type sampleL = tri.template generate_and_pdf<interaction_type>(pdf, newRayMaxT, origin, interaction, isBSDF, xi);
                const vector3_type V = interaction.isotropic.V.getDirection();
                const scalar_type VdotL = nbl::hlsl::dot<vector3_type>(V, sampleL);
                ray_dir_info_type rayL;
                rayL.direction = sampleL;
                L = sample_type::create(rayL,VdotL,interaction.T,interaction.B,interaction.isotropic.N);
            }
            break;
            case PST_RECTANGLE:
            {
                const vector3_type offset = vector3_type(
                    bit_cast<float32_t>(event.data[2]),
                    bit_cast<float32_t>(event.data[3]),
                    bit_cast<float32_t>(event.data[4]));
                const vector3_type edge0 = vector3_type(
                    bit_cast<float32_t>(event.data[5]),
                    bit_cast<float32_t>(event.data[6]),
                    bit_cast<float32_t>(event.data[7]));
                const vector3_type edge1 = vector3_type(
                    bit_cast<float32_t>(event.data[8]),
                    bit_cast<float32_t>(event.data[9]),
                    bit_cast<float32_t>(event.data[10]));
                Shape<PST_RECTANGLE> rect = Shape<PST_RECTANGLE>::create(offset, edge0, edge1, event.data[11]);

                const vector3_type sampleL = rect.template generate_and_pdf<interaction_type>(pdf, newRayMaxT, origin, interaction, isBSDF, xi);
                const vector3_type V = interaction.isotropic.V.getDirection();
                const scalar_type VdotL = nbl::hlsl::dot<vector3_type>(V, sampleL);
                ray_dir_info_type rayL;
                rayL.direction = sampleL;
                L = sample_type::create(rayL,VdotL,interaction.T,interaction.B,interaction.isotropic.N);
            }
            break;
            default:
                pdf = bit_cast<float>(numeric_limits<float>::infinity);
                break;
        }

        newRayMaxT *= Tolerance<scalar_type>::getEnd(depth);
        pdf *= 1.0 / scalar_type(lightCount);
        spectral_type quo = light.radiance / pdf;
        quotient_pdf = quotient_pdf_type::create(quo, pdf);

        return L;
    }

    static sample_type generate_and_quotient_and_pdf(NBL_REF_ARG(quotient_pdf_type) quotient_pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(light_type) light, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(interaction_type) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi, uint32_t depth, NBL_CONST_REF_ARG(Event) event)
    {
        const IntersectMode mode = (IntersectMode)event.mode;
        sample_type L;
        // switch (mode)
        // {
        //     case IM_RAY_QUERY:
        //     {
        //         // TODO: do ray query stuff
        //     }
        //     break;
        //     case IM_RAY_TRACING:
        //     {
        //         // TODO: do ray tracing stuff
        //     }
        //     break;
        //     case IM_PROCEDURAL:
        //     {
                return procedural_generate_and_quotient_and_pdf(quotient_pdf, newRayMaxT, light, origin, interaction, isBSDF, xi, depth, event);
        //     }
        //     break;
        //     default:
        //     {
        //         return L;
        //     }
        // }
        // return L;
    }
};

}
}
}
}

#endif
