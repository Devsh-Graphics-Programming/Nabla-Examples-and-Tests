#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"

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

[numthreads(WORKGROUP_SIZE,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t> lambertianBRDF;
    bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t> orenNayarBRDF;
    bxdf::reflection::SBeckmannBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t> beckmannBRDF;
    bxdf::reflection::SGGXBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t> ggxBRDF;

    bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t> lambertianBSDF;
    bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t, false> smoothDielectricBSDF;
    bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t, true> thinSmoothDielectricBSDF;
    bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t> beckmannBSDF;
    bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t> ggxBSDF;


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

    bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>::params_isotropic_t params0 = bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>::params_isotropic_t::create(s, isointer, bxdf::BxDFClampMode::BCM_MAX);
    quotient_pdf_t qp = orenNayarBRDF.quotient_and_pdf(params0);
    L -= qp.quotient;

    s = beckmannBRDF.generate(anisointer, u.xy, cache);
    L += s.L.direction;

    bxdf::reflection::SBeckmannBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t>::params_anisotropic_t params1 = bxdf::reflection::SBeckmannBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t>::params_anisotropic_t::create(s, anisointer, cache, bxdf::BxDFClampMode::BCM_MAX);
    qp = beckmannBRDF.quotient_and_pdf(params1);
    L -= qp.quotient;

    s = ggxBRDF.generate(anisointer, u.xy, cache);
    L += s.L.direction;

    bxdf::reflection::SGGXBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t>::params_anisotropic_t params2 = bxdf::reflection::SGGXBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t>::params_anisotropic_t::create(s, anisointer, cache, bxdf::BxDFClampMode::BCM_MAX);
    qp = ggxBRDF.quotient_and_pdf(params2);
    L -= qp.quotient;

    s = lambertianBSDF.generate(anisointer, u);
    L += s.L.direction;

    s = thinSmoothDielectricBSDF.generate(anisointer, u);
    L += s.L.direction;

    bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t, true>::params_isotropic_t params3 = bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t, true>::params_isotropic_t::create(s, isointer, bxdf::BxDFClampMode::BCM_ABS);
    qp = thinSmoothDielectricBSDF.quotient_and_pdf(params3);
    L -= qp.quotient;

    s = ggxBSDF.generate(anisointer, u, cache);
    L += s.L.direction;

    bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t>::params_anisotropic_t params4 = bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_interaction, aniso_interaction, iso_cache, aniso_cache, spectral_t>::params_anisotropic_t::create(s, anisointer, cache, bxdf::BxDFClampMode::BCM_ABS);
    qp = ggxBSDF.quotient_and_pdf(params4);
    L -= qp.quotient;

    buff[ID.x] = L;
}