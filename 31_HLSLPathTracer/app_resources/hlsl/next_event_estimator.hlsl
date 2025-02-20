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

// procedural data store: [light count] [intersect type] [obj]

struct Event
{
    enum Mode : uint32_t    // enum class?
    {
        RAY_QUERY,
        RAY_TRACING,
        PROCEDURAL
    };

    NBL_CONSTEXPR_STATIC_INLINE uint32_t DataSize = 16;

    uint32_t mode : 1;
    unit32_t unused : 31;   // possible space for flags
    uint32_t data[DataSize];
};

template<typename Light, typename Ray, class LightSample, class Aniso>
struct Estimator
{
    using scalar_type = typename Ray::scalar_type;
    using ray_type = Ray;
    using light_type = Light;
    using spectral_type = typename Light::spectral_type;
    using interaction_type = Aniso;
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using sample_type = LightSample;

    static spectral_type proceduralDeferredEvalAndPdf(NBL_REF_ARG(scalar_type) pdf, NBL_CONST_REF_ARG(light_type) light, NBL_CONST_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(Event) event)
    {
        const uint32_t lightCount = event.data[0];
        const ProceduralShapeType type = event.data[1];

        pdf = 1.0 / lightCount;
        switch (type)
        {
            case PST_SPHERE:
            {
                float32_t3 position = float32_t3(asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize]), asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 2]));
                Shape<PST_SPHERE> sphere = Shape<PST_SPHERE>::create(position, asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 3]), intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 4]);
                pdf *= sphere.template deferredPdf<ray_type>(ray);
            }
            break;
            case PST_TRIANGLE:
            {
                float32_t3 vertex0 = float32_t3(asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize]), asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 2]));
                float32_t3 vertex1 = float32_t3(asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 4]), asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 5]));
                float32_t3 vertex2 = float32_t3(asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 7]), asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 8]));
                Shape<PST_TRIANGLE> tri = Shape<PST_TRIANGLE>::create(vertex0, vertex1, vertex2, intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 9]);
                pdf *= tri.template deferredPdf<ray_type>(ray);
            }
            break;
            case PST_RECTANGLE:
            {
                float32_t3 offset = float32_t3(asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 1]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 2]));
                float32_t3 edge0 = float32_t3(asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 4]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 5]));
                float32_t3 edge1 = float32_t3(asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 7]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 8]));
                Shape<PST_RECTANGLE> rect = Shape<PST_RECTANGLE>::create(offset, edge0, edge1, intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 9]);
                pdf *= rect.template deferredPdf<ray_type>(ray);
            }
            break;
            default:
                pdf = numeric_limits<float>::infinity;
                break;
        }

        return light.radiance;
    }

    static spectral_type deferredEvalAndPdf(NBL_REF_ARG(scalar_type) pdf, NBL_CONST_REF_ARG(light_type) light, NBL_CONST_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(Event) event)
    {
        const Event::Mode mode = event.mode;
        switch (mode)
        {
            case Event::Mode::RAY_QUERY:
            {
                // TODO: do ray query stuff
            }
            break;
            case Event::Mode::RAY_TRACING:
            {
                // TODO: do ray tracing stuff
            }
            break;
            case Event::Mode::PROCEDURAL:
            {
                return proceduralDeferredEvalAndPdf(pdf, light, ray, event);
            }
            break;
            default:
                return (spectral_type)0.0;
        }
    }

    static sample_type procedural_generate_and_quotient_and_pdf(NBL_REF_ARG(quotient_pdf_type) quotient_pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(interaction_type) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi, unit32_t depth, NBL_CONST_REF_ARG(Event) event)
    {
        const uint32_t lightCount = event.data[0];
        const ProceduralShapeType type = event.data[1];

        sample_type L;
        scalar_type pdf;
        switch (type)
        {
            case PST_SPHERE:
            {
                float32_t3 position = float32_t3(asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize]), asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 2]));
                Shape<PST_SPHERE> sphere = Shape<PST_SPHERE>::create(position, asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 3]), intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 4]);
                L = sphere.template generate_and_pdf<interaction_type>(pdf, newRayMaxT, origin, interaction, isBSDF, xi);
            }
            break;
            case PST_TRIANGLE:
            {
                float32_t3 vertex0 = float32_t3(asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize]), asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 1]), asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 2]));
                float32_t3 vertex1 = float32_t3(asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 4]), asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 5]));
                float32_t3 vertex2 = float32_t3(asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + Shape<PST_SPHERE>::ObjSize + 7]), asfloat(intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 8]));
                Shape<PST_TRIANGLE> tri = Shape<PST_TRIANGLE>::create(vertex0, vertex1, vertex2, intersect.data[2 + Shape<PST_TRIANGLE>::ObjSize + 9]);
                L = tri.template generate_and_pdf<interaction_type>(pdf, newRayMaxT, origin, interaction, isBSDF, xi);
            }
            break;
            case PST_RECTANGLE:
            {
                float32_t3 offset = float32_t3(asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 1]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 2]));
                float32_t3 edge0 = float32_t3(asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 3]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 4]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 5]));
                float32_t3 edge1 = float32_t3(asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 6]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 7]), asfloat(intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 8]));
                Shape<PST_RECTANGLE> rect = Shape<PST_RECTANGLE>::create(offset, edge0, edge1, intersect.data[2 + Shape<PST_RECTANGLE>::ObjSize + 9]);
                L = rect.template generate_and_pdf<interaction_type>(pdf, newRayMaxT, origin, interaction, isBSDF, xi);
            }
            break;
            default:
                pdf = numeric_limits<float>::infinity;
                break;
        }

        newRayMaxT *= Tolerance<scalar_type>::getEnd(depth);
        pdf *= 1.0 / lightCount;
        spectral_type quo = light.radiance / pdf;
        quotient_pdf = quotient_pdf_type::create(quo, pdf);

        return L;
    }

    static sample_type generate_and_quotient_and_pdf(NBL_REF_ARG(quotient_pdf_type) quotient_pdf, NBL_REF_ARG(scalar_type) newRayMaxT, NBL_CONST_REF_ARG(vector3_type) origin, NBL_CONST_REF_ARG(interaction_type) interaction, bool isBSDF, NBL_CONST_REF_ARG(vector3_type) xi, unit32_t depth, NBL_CONST_REF_ARG(Event) event)
    {
        const Event::Mode mode = event.mode;
        switch (mode)
        {
            case Event::Mode::RAY_QUERY:
            {
                // TODO: do ray query stuff
            }
            break;
            case Event::Mode::RAY_TRACING:
            {
                // TODO: do ray tracing stuff
            }
            break;
            case Event::Mode::PROCEDURAL:
            {
                return procedural_generate_and_quotient_and_pdf(newRayMaxT, origin, interaction, isBSDF, xi, depth, event);
            }
            break;
            default:
            {
                sample_type L;
                return L;
            }
        }
    }
};

}
}
}
}

#endif
