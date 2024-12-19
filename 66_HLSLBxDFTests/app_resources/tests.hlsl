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

inline float32_t3 testLambertianBRDF()
{
    ray_dir_info_t V;
    V.direction = float32_t3(0.3, 0.4, 0.5);
    float32_t3 N = float32_t3(0, 1, 0);
    iso_interaction isointer = iso_interaction::create(V, N);
    aniso_interaction anisointer = aniso_interaction::create(isointer, float32_t3(0, 0, 1), float32_t3(1, 0, 0));
    sample_t s;

    bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction> lambertian = bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>::create();
    s = lambertian.generate(anisointer, float32_t2(0.5, 0.0));
    quotient_pdf_t pdf = lambertian.quotient_and_pdf(s, isointer);
    float32_t3 brdf = float32_t3(lambertian.eval(s, isointer));

    // get jacobian

    return abs<float32_t3>(pdf.value() - brdf);
}

inline float32_t3 testLambertianBSDF()
{
    ray_dir_info_t V;
    V.direction = float32_t3(0.3, 0.4, 0.5);
    float32_t3 N = float32_t3(0, 1, 0);
    iso_interaction isointer = iso_interaction::create(V, N);
    aniso_interaction anisointer = aniso_interaction::create(isointer, float32_t3(0, 0, 1), float32_t3(1, 0, 0));
    sample_t s;

    bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction> lambertian = bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>::create();
    s = lambertian.generate(anisointer, float32_t2(0.5, 0.0));
    quotient_pdf_t pdf = lambertian.quotient_and_pdf(s, isointer);
    float32_t3 bsdf = float32_t3(lambertian.eval(s, isointer));

    // get jacobian

    return abs<float32_t3>(pdf.value() - bsdf);
}

}
}

#endif