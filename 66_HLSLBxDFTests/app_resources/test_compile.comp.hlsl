#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"

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
    bxdf::reflection::SLambertian<iso_config_t> lambertianBRDF;
    bxdf::reflection::SOrenNayar<iso_config_t> orenNayarBRDF;
    bxdf::reflection::SDeltaDistribution<iso_config_t> deltaDistBRDF;
    bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t> beckmannIsoBRDF;
    bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t> beckmannAnisoBRDF;
    bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t> ggxIsoBRDF;
    bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t> ggxAnisoBRDF;

    bxdf::transmission::SLambertian<iso_config_t> lambertianBSDF;
    bxdf::transmission::SOrenNayar<iso_config_t> orenNayarBSDF;
    bxdf::transmission::SSmoothDielectric<iso_config_t> smoothDielectricBSDF;
    bxdf::transmission::SThinSmoothDielectric<iso_config_t> thinSmoothDielectricBSDF;
    bxdf::transmission::SDeltaDistribution<iso_config_t> deltaDistBSDF;
    bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t> beckmannIsoBSDF;
    bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t> beckmannAnisoBSDF;
    bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t> ggxIsoBSDF;
    bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t> ggxAnisoBSDF;


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

    typename bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>::query_type query0 = beckmannAnisoBRDF.createQuery(s, anisointer);
    qp = beckmannAnisoBRDF.quotient_and_pdf(query0, s, anisointer, cache);
    L -= qp.quotient;

    s = ggxAnisoBRDF.generate(anisointer, u.xy, cache);
    L += s.L.direction;

    typename bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>::query_type query1 = ggxAnisoBRDF.createQuery(s, anisointer);
    qp = ggxAnisoBRDF.quotient_and_pdf(query1, s, anisointer, cache);
    L -= qp.quotient;

    s = lambertianBSDF.generate(anisointer, u);
    L += s.L.direction;

    s = thinSmoothDielectricBSDF.generate(anisointer, u);
    L += s.L.direction;

    qp = thinSmoothDielectricBSDF.quotient_and_pdf(s, isointer);
    L -= qp.quotient;

    s = ggxAnisoBSDF.generate(anisointer, u, cache);
    L += s.L.direction;

    typename bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>::query_type query2 = ggxAnisoBSDF.createQuery(s, anisointer);
    qp = ggxAnisoBSDF.quotient_and_pdf(query2, s, anisointer, cache);
    L -= qp.quotient;

    buff[ID.x] = L;
}