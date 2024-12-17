#ifndef BXDFTESTS_TESTS_HLSL
#define BXDFTESTS_TESTS_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"
#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"

using namespace nbl;
using namespace hlsl;

using ray_dir_info_t = ray_dir_info::SBasic<float>;
using iso_interaction = surface_interactions::SIsotropic<ray_dir_info_t>;
using aniso_interaction = surface_interactions::SAnisotropic<ray_dir_info_t>;
using sample_t = SLightSample<ray_dir_info_t>;
using quotient_pdf_t = quotient_and_pdf<float32_t3, float>;

float32_t3 testLambertianBRDF()
{
    ray_dir_info_t V;
    V.direction = float32_t3(0.3, 0.4, 0.5);
    float32_t3 N = float32_t3(0, 1, 0);
    iso_interaction isointer = iso_interaction::create(V, N);
    aniso_interaction anisointer = aniso_interaction::create(isointer, float32_t3(0, 0, 1), float32_t3(1, 0, 0));
    sample_t s;

    reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction> lambertian = reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>::create();
    s = lambertian.generate(anisointer, float32_t2(0.5, 0.0));
    quotient_pdf_t pdf = lambertian.quotient_and_pdf(s, isointer);
    float32_t3 brdf = float32_t3(lambertian.eval(s, isointer));

    // get jacobian

    return abs<float>(pdf.value() - brdf);
}

#endif