#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"
// #include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"

[[vk::binding(0,0)]] RWStructuredBuffer<float3> buff;

using namespace nbl::hlsl;

using ray_dir_info_t = bxdf::ray_dir_info::SBasic<float>;
using iso_interaction = bxdf::surface_interactions::SIsotropic<ray_dir_info_t>;
using aniso_interaction = bxdf::surface_interactions::SAnisotropic<iso_interaction>;
using sample_t = bxdf::SLightSample<ray_dir_info_t>;
using iso_cache = bxdf::SIsotropicMicrofacetCache<float>;
using aniso_cache = bxdf::SAnisotropicMicrofacetCache<iso_cache>;
using quotient_pdf_t = sampling::quotient_and_pdf<float32_t3, float>;
using spectral_t = vector<float, 3>;

using iso_config_t = bxdf::SConfiguration<sample_t, iso_interaction, spectral_t>;
using aniso_config_t = bxdf::SConfiguration<sample_t, aniso_interaction, spectral_t>;
using iso_microfacet_config_t = bxdf::SMicrofacetConfiguration<sample_t, iso_interaction, iso_cache, spectral_t>;
using aniso_microfacet_config_t = bxdf::SMicrofacetConfiguration<sample_t, aniso_interaction, aniso_cache, spectral_t>;

[numthreads(WORKGROUP_SIZE,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    bxdf::reflection::SLambertianBxDF<iso_config_t> lambertianBRDF;
    bxdf::reflection::SOrenNayarBxDF<iso_config_t> orenNayarBRDF;
    bxdf::reflection::SBeckmannIsotropicBxDF<iso_microfacet_config_t> beckmannIsoBRDF;
    bxdf::reflection::SBeckmannAnisotropicBxDF<aniso_microfacet_config_t> beckmannAnisoBRDF;
    bxdf::reflection::SGGXIsotropicBxDF<iso_microfacet_config_t> ggxIsoBRDF;
    bxdf::reflection::SGGXAnisotropicBxDF<aniso_microfacet_config_t> ggxAnisoBRDF;

    bxdf::transmission::SLambertianBxDF<iso_config_t> lambertianBSDF;
    bxdf::transmission::SSmoothDielectricBxDF<iso_config_t> smoothDielectricBSDF;
    bxdf::transmission::SSmoothThinDielectricBxDF<iso_config_t> thinSmoothDielectricBSDF;
    bxdf::transmission::SBeckmannDielectricIsotropicBxDF<iso_microfacet_config_t> beckmannIsoBSDF;
    bxdf::transmission::SBeckmannDielectricAnisotropicBxDF<aniso_microfacet_config_t> beckmannAnisoBSDF;
    bxdf::transmission::SGGXDielectricIsotropicBxDF<iso_microfacet_config_t> ggxIsoBSDF;
    bxdf::transmission::SGGXDielectricAnisotropicBxDF<aniso_microfacet_config_t> ggxAnisoBSDF;


    // do some nonsense calculations, but call all the relevant functions
    ray_dir_info_t V;
    V.direction = nbl::hlsl::normalize<float3>(float3(1, 1, 1));
    const float3 N = float3(0, 1, 0);
    float3 T, B;
    math::frisvad<float32_t3>(N, T, B);
    const float3 u = float3(0.5, 0.5, 0);

    iso_interaction isointer = iso_interaction::create(V, N);
    aniso_interaction anisointer = aniso_interaction::create(isointer, T, B);
    aniso_cache cache;

    float3 L = float3(0,0,0);
    float3 q = float3(0,0,0);
    sample_t s = lambertianBRDF.generate(anisointer, u.xy);
    L += s.L.direction;

    s = orenNayarBRDF.generate(anisointer, u.xy);
    L += s.L.direction;

    quotient_pdf_t qp = orenNayarBRDF.quotient_and_pdf(s, isointer);
    L -= qp.quotient;

    s = beckmannAnisoBRDF.generate(anisointer, u.xy, cache);
    L += s.L.direction;

    qp = beckmannAnisoBRDF.quotient_and_pdf(s, anisointer, cache);
    L -= qp.quotient;

    s = ggxAnisoBRDF.generate(anisointer, u.xy, cache);
    L += s.L.direction;

    qp = ggxAnisoBRDF.quotient_and_pdf(s, anisointer, cache);
    L -= qp.quotient;

    s = lambertianBSDF.generate(anisointer, u);
    L += s.L.direction;

    s = thinSmoothDielectricBSDF.generate(anisointer, u);
    L += s.L.direction;

    qp = thinSmoothDielectricBSDF.quotient_and_pdf(s, isointer);
    L -= qp.quotient;

    s = ggxAnisoBSDF.generate(anisointer, u, cache);
    L += s.L.direction;

    qp = ggxAnisoBSDF.quotient_and_pdf(s, anisointer, cache);
    L -= qp.quotient;

    buff[ID.x] = L;
}