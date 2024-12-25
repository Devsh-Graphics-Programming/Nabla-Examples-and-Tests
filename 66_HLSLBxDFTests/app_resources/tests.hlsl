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

inline float rngFloat01(NBL_REF_ARG(nbl::hlsl::Xoroshiro64Star) rng)
{
    return (float)rng() / numeric_limits<uint32_t>::max;
}

inline float32_t3 rngFloat301(NBL_REF_ARG(nbl::hlsl::Xoroshiro64Star) rng)
{
    return float32_t3(rngFloat01(rng), rngFloat01(rng), rngFloat01(rng));
}


struct SBxDFTestResources
{
    static SBxDFTestResources create(uint32_t2 seed)
    {
        SBxDFTestResources retval;

        retval.rng = nbl::hlsl::Xoroshiro64Star::construct(seed);
        retval.u = float32_t3(rngFloat01(retval.rng), rngFloat01(retval.rng), 0.0);

        retval.V.direction = nbl::hlsl::normalize<float32_t3>(rngFloat301(retval.rng));
        retval.N = nbl::hlsl::normalize<float32_t3>(rngFloat301(retval.rng));
        retval.T = nbl::hlsl::normalize<float32_t3>(rngFloat301(retval.rng));

        retval.T = nbl::hlsl::normalize<float32_t3>(retval.T - nbl::hlsl::dot<float32_t3>(retval.T, retval.N) * retval.N); // gram schmidt
        retval.B = nbl::hlsl::cross<float32_t3>(retval.N, retval.T);

        retval.alpha.x = rngFloat01(retval.rng);
        retval.alpha.y = rngFloat01(retval.rng);
        retval.ior = float32_t3x2(1.02, 1.0,      // randomize at some point?
                                        1.3, 2.0,
                                        1.02, 1.0);
        return retval;
    }

    float h = 0.001;

    nbl::hlsl::Xoroshiro64Star rng;
    ray_dir_info_t V;
    float32_t3 N;
    float32_t3 T;
    float32_t3 B;

    float32_t3 u;
    float32_t2 alpha;
    float32_t3x2 ior;
};

inline float32_t4 testLambertianBRDF()
{
    const uint32_t2 state = uint32_t2(10u, 42u);
    SBxDFTestResources rc = SBxDFTestResources::create(state);

    iso_interaction isointer = iso_interaction::create(rc.V, rc.N);
    aniso_interaction anisointer = aniso_interaction::create(isointer, rc.T, rc.B);
    
    sample_t s, sx, sy;
    bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction> lambertian = bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>::create();
    s = lambertian.generate(anisointer, rc.u.xy);
    sx = lambertian.generate(anisointer, rc.u.xy + float32_t2(rc.h,0));
    sy = lambertian.generate(anisointer, rc.u.xy + float32_t2(0,rc.h));
    quotient_pdf_t pdf = lambertian.quotient_and_pdf(s, isointer);
    float32_t3 brdf = float32_t3(lambertian.eval(s, isointer));

    // get jacobian
    float32_t2x2 m = float32_t2x2(sx.TdotL - s.TdotL, sy.TdotL - s.TdotL, sx.BdotL - s.BdotL, sy.BdotL - s.BdotL);
    float det = nbl::hlsl::determinant<float, 2>(m);

    return float32_t4(nbl::hlsl::abs<float32_t3>(pdf.value() - brdf), nbl::hlsl::abs<float>(det*pdf.pdf/s.NdotL) * 0.5);
}

inline float32_t4 testBeckmannBRDF()
{
    const uint32_t2 state = uint32_t2(10u, 42u);
    SBxDFTestResources rc = SBxDFTestResources::create(state);

    iso_interaction isointer = iso_interaction::create(rc.V, rc.N);
    aniso_interaction anisointer = aniso_interaction::create(isointer, rc.T, rc.B);
    aniso_cache cache;
    
    sample_t s, sx, sy;
    bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache> beckmann = bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,rc.alpha.y,rc.ior);
    s = beckmann.generate(anisointer, rc.u.xy, cache);
    sx = beckmann.generate(anisointer, rc.u.xy + float32_t2(rc.h,0), cache);
    sy = beckmann.generate(anisointer, rc.u.xy + float32_t2(0,rc.h), cache);
    quotient_pdf_t pdf = beckmann.quotient_and_pdf(s, anisointer, cache);
    float32_t3 brdf = float32_t3(beckmann.eval(s, anisointer, cache));

    // get jacobian
    float32_t2x2 m = float32_t2x2(sx.TdotL - s.TdotL, sy.TdotL - s.TdotL, sx.BdotL - s.BdotL, sy.BdotL - s.BdotL);
    float det = nbl::hlsl::determinant<float, 2>(m);

    return float32_t4(nbl::hlsl::abs<float32_t3>(pdf.value() - brdf), nbl::hlsl::abs<float>(det*pdf.pdf/s.NdotL) * 0.5);
}

inline float32_t4 testLambertianBSDF()
{
    const uint32_t2 state = uint32_t2(12u, 69u);
    SBxDFTestResources rc = SBxDFTestResources::create(state);

    iso_interaction isointer = iso_interaction::create(rc.V, rc.N);
    aniso_interaction anisointer = aniso_interaction::create(isointer, rc.T, rc.B);
    
    sample_t s, sx, sy;
    bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction> lambertian = bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>::create();
    s = lambertian.generate(anisointer, rc.u);
    sx = lambertian.generate(anisointer, rc.u + float32_t3(rc.h,0,0));
    sy = lambertian.generate(anisointer, rc.u + float32_t3(0,rc.h,0));
    quotient_pdf_t pdf = lambertian.quotient_and_pdf(s, isointer);
    float32_t3 brdf = float32_t3(lambertian.eval(s, isointer));

    // get jacobian
    float32_t2x2 m = float32_t2x2(sx.TdotL - s.TdotL, sy.TdotL - s.TdotL, sx.BdotL - s.BdotL, sy.BdotL - s.BdotL);
    float det = nbl::hlsl::determinant<float, 2>(m);

    return float32_t4(nbl::hlsl::abs<float32_t3>(pdf.value() - brdf), nbl::hlsl::abs<float>(det*pdf.pdf/s.NdotL) * 0.5);
}

}
}

#endif