#ifndef BXDFTESTS_TESTS_HLSL
#define BXDFTESTS_TESTS_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"
#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"

namespace nbl
{
namespace hlsl
{

using ray_dir_info_t = bxdf::ray_dir_info::SBasic<float>;
using iso_interaction = bxdf::surface_interactions::SIsotropic<ray_dir_info_t>;
using aniso_interaction = bxdf::surface_interactions::SAnisotropic<ray_dir_info_t>;
using sample_t = bxdf::SLightSample<ray_dir_info_t>;
using iso_cache = bxdf::SIsotropicMicrofacetCache<float>;
using aniso_cache = bxdf::SAnisotropicMicrofacetCache<float>;
using quotient_pdf_t = bxdf::quotient_and_pdf<float32_t3, float>;

inline float32_t4 testLambertianBRDF()
{
    const uint32_t2 state = uint32_t2(10u, 42u);
    nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(state);
    NBL_CONSTEXPR float h = 0.001;
    float32_t2 u = float32_t2(rng(), rng());

    ray_dir_info_t V;
    V.direction = glm::normalize(float32_t3(rng(), rng(), rng()));  // TODO: use cpp compat version
    float32_t3 N = float32_t3(0, 1, 0);
    iso_interaction isointer = iso_interaction::create(V, N);
    aniso_interaction anisointer = aniso_interaction::create(isointer, float32_t3(0, 0, 1), float32_t3(1, 0, 0));   // TODO: random T and cross B
    sample_t s, sx, sy;

    bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction> lambertian = bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>::create();
    s = lambertian.generate(anisointer, u);
    sx = lambertian.generate(anisointer, u + float32_t2(h,0));
    sy = lambertian.generate(anisointer, u + float32_t2(0,h));
    quotient_pdf_t pdf = lambertian.quotient_and_pdf(s, isointer);
    float32_t3 brdf = float32_t3(lambertian.eval(s, isointer));

    // get jacobian
    float32_t2x2 m = float32_t2x2(sx.TdotL - s.TdotL, sy.TdotL - s.TdotL, sx.BdotL - s.BdotL, sy.BdotL - s.BdotL);
    float det = nbl::hlsl::determinant<float, 2>(m);

    return float32_t4(nbl::hlsl::abs<float32_t3>(pdf.value() - brdf), nbl::hlsl::abs<float>(det*pdf.pdf/s.NdotL) * 0.5);
}

inline float32_t4 testLambertianBSDF()
{
    ray_dir_info_t V;
    V.direction = float32_t3(0.3, 0.4, 0.5);
    float32_t3 N = float32_t3(0, 1, 0);
    iso_interaction isointer = iso_interaction::create(V, N);
    aniso_interaction anisointer = aniso_interaction::create(isointer, float32_t3(0, 0, 1), float32_t3(1, 0, 0));
    sample_t s;

    bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction> lambertian = bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>::create();
    s = lambertian.generate(anisointer, float32_t3(0.5, 0.5, 0.0));
    quotient_pdf_t pdf = lambertian.quotient_and_pdf(s, isointer);
    float32_t3 bsdf = float32_t3(lambertian.eval(s, isointer));

    // get jacobian

    return float32_t4(nbl::hlsl::abs<float32_t3>(pdf.value() - bsdf), 0);
}

}
}

#endif