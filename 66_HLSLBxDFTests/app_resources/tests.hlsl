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
        retval.eta = 1.3;// rngFloat01(retval.rng) + 1.0;
        retval.ior = float32_t3x2(1.02, 1.0,      // randomize at some point?
                                1.3, 2.0,
                                1.02, 1.0);
        retval.eta2 = float32_t3(1.02, 1.3, 1.02);  // TODO: check correctness?
        retval.luma_coeff = float32_t3(0.2126, 0.7152, 0.0722); // luma coefficients for Rec. 709
        return retval;
    }

    ray_dir_info_t dV(int axis)
    {
        float32_t3 d = (float32_t3)0.0;
        d[axis] += h;
        ray_dir_info_t retval;
        retval.direction = V.direction + d;
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
    float eta;
    float32_t3x2 ior;

    float32_t3 eta2;    // what is this?
    float32_t3 luma_coeff;
};

struct STestMeta
{
    float32_t4 result;
#ifndef __HLSL_VERSION
    std::string bxdfName;
    std::string testName;
#endif
};

template<class BxDF>
struct TestBase
{
    void init(uint32_t2 seed)
    {
        rc = SBxDFTestResources::create(seed);

        isointer = iso_interaction::create(rc.V, rc.N);
        anisointer = aniso_interaction::create(isointer, rc.T, rc.B);
    }

    SBxDFTestResources rc;
    BxDF bxdf;

    iso_interaction isointer;
    aniso_interaction anisointer;

    STestMeta meta;
};


template<class BxDF>
struct TestBxDF : TestBase<BxDF>
{
    using base_t = TestBase<BxDF>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf = BxDF::create();  // default to lambertian bxdf
#ifndef __HLSL_VERSION
        base_t::meta.bxdfName = "LambertianBxDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>> : TestBase<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>>
{
    using base_t = TestBase<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf = bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>::create(_rc.alpha.x);
#ifndef __HLSL_VERSION
        base_t::meta.bxdfName = "OrenNayarBRDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>> : TestBase<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>>
{
    using base_t = TestBase<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>>;

    template<bool aniso>
    void initBxDF(SBxDFTestResources _rc)
    {
        if (aniso)
        {
            base_t::bxdf = bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,rc.alpha.y,rc.ior);
#ifndef __HLSL_VERSION
            base_t::meta.bxdfName = "BeckmannBRDF Aniso";
#endif
        }
        else
        {
            base_t::bxdf = bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,rc.ior);
#ifndef __HLSL_VERSION
            base_t::meta.bxdfName = "BeckmannBRDF";
#endif
        }
    }
};

template<>
struct TestBxDF<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>> : TestBase<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>>
{
    using base_t = TestBase<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>>;

    template<bool aniso>
    void initBxDF(SBxDFTestResources _rc)
    {
        if (aniso)
        {
            base_t::bxdf = bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,rc.alpha.y,rc.ior);
#ifndef __HLSL_VERSION
            base_t::meta.bxdfName = "GGXBRDF Aniso";
#endif
        }
        else
        {
            base_t::bxdf = bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,rc.ior);
#ifndef __HLSL_VERSION
            base_t::meta.bxdfName = "GGXBRDF";
#endif
        }
    }
};

template<>
struct TestBxDF<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache>> : TestBase<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache>>
{
    using base_t = TestBase<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf = bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache>::create(rc.eta);
#ifndef __HLSL_VERSION
        base_t::meta.bxdfName = "SmoothDielectricBSDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>> : TestBase<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>>
{
    using base_t = TestBase<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf = bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>::create(rc.eta2,rc.luma_coeff);
#ifndef __HLSL_VERSION
        base_t::meta.bxdfName = "ThinSmoothDielectricBSDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>> : TestBase<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>>
{
    using base_t = TestBase<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>>;

    template<bool aniso>
    void initBxDF(SBxDFTestResources _rc)
    {
        if (aniso)
        {
            base_t::bxdf = bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>::create(rc.eta,rc.alpha.x,rc.alpha.y);
#ifndef __HLSL_VERSION
            base_t::meta.bxdfName = "BeckmannBSDF Aniso";
#endif
        }
        else
        {
            base_t::bxdf = bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>::create(rc.eta,rc.alpha.x);
#ifndef __HLSL_VERSION
            base_t::meta.bxdfName = "BeckmannBSDF";
#endif
        }
    }
};

template<>
struct TestBxDF<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>> : TestBase<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>>
{
    using base_t = TestBase<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>>;

    template<bool aniso>
    void initBxDF(SBxDFTestResources _rc)
    {
        if (aniso)
        {
            base_t::bxdf = bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>::create(rc.eta,rc.alpha.x,rc.alpha.y);
#ifndef __HLSL_VERSION
            base_t::meta.bxdfName = "GGXBSDF Aniso";
#endif
        }
        else
        {
            base_t::bxdf = bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>::create(rc.eta,rc.alpha.x);
#ifndef __HLSL_VERSION
            base_t::meta.bxdfName = "GGXBSDF";
#endif
        }
    }
};


template<class T>
struct is_basic_brdf : bool_constant<
    is_same<T, bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::value ||
    is_same<T, bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>>::value
> {};

template<class T>
struct is_microfacet_brdf : bool_constant<
    is_same<T, bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>>::value ||
    is_same<T, bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>>::value
> {};

template<class T>
struct is_basic_bsdf : bool_constant<
    is_same<T, bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::value ||
    is_same<T, bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache>>::value ||
    is_same<T, bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>>::value
> {};

template<class T>
struct is_microfacet_bsdf : bool_constant<
    is_same<T, bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>>::value ||
    is_same<T, bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>>::value
> {};

template<class T>
NBL_CONSTEXPR bool is_basic_brdf_v = is_basic_brdf<T>::value;
template<class T>
NBL_CONSTEXPR bool is_microfacet_brdf_v = is_microfacet_brdf<T>::value;
template<class T>
NBL_CONSTEXPR bool is_basic_bsdf_v = is_basic_bsdf<T>::value;
template<class T>
NBL_CONSTEXPR bool is_microfacet_bsdf_v = is_microfacet_bsdf<T>::value;


template<class BxDF, bool aniso = false>
struct TestUOffset : TestBxDF<BxDF>
{
    using base_t = TestBase<BxDF>;
    using this_t = TestUOffset<BxDF, aniso>;

    float32_t4 test()
    {
        sample_t s, sx, sy;
        quotient_pdf_t pdf;
        float32_t3 brdf;
        aniso_cache cache, dummy;

        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy);
            sx = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy + float32_t2(base_t::rc.h,0));
            sy = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy + float32_t2(0,base_t::rc.h));
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy, cache);
            sx = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy + float32_t2(base_t::rc.h,0), dummy);
            sy = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy + float32_t2(0,base_t::rc.h), dummy);
        }
        if NBL_CONSTEXPR_FUNC (is_basic_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u);
            float32_t3 ux = base_t::rc.u + float32_t3(base_t::rc.h,0,0);
            sx = base_t::bxdf.generate(base_t::anisointer, ux);
            float32_t3 uy = base_t::rc.u + float32_t3(0,base_t::rc.h,0);
            sy = base_t::bxdf.generate(base_t::anisointer, uy);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u, cache);
            float32_t3 ux = base_t::rc.u + float32_t3(base_t::rc.h,0,0);
            sx = base_t::bxdf.generate(base_t::anisointer, ux, dummy);
            float32_t3 uy = base_t::rc.u + float32_t3(0,base_t::rc.h,0);
            sy = base_t::bxdf.generate(base_t::anisointer, uy, dummy);
        }
        
        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF> || is_basic_bsdf_v<BxDF>)
        {
            pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer);
            brdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer));
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                pdf = base_t::bxdf.quotient_and_pdf(s, base_t::anisointer, cache);
                brdf = float32_t3(base_t::bxdf.eval(s, base_t::anisointer, cache));
            }
            else
            {
                iso_cache isocache = (iso_cache)cache;
                pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer, isocache);
                brdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer, isocache));
            }
        }

        // get jacobian
        float32_t2x2 m = float32_t2x2(sx.TdotL - s.TdotL, sy.TdotL - s.TdotL, sx.BdotL - s.BdotL, sy.BdotL - s.BdotL);
        float det = nbl::hlsl::determinant<float32_t2x2>(m);

        return float32_t4(nbl::hlsl::abs<float32_t3>(pdf.value() - brdf), nbl::hlsl::abs<float>(det*pdf.pdf/s.NdotL) * 0.5);
    }

    static STestMeta run(uint32_t2 seed)
    {
        this_t t;
        t.init(seed);
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
            t.template initBxDF<aniso>(t.rc);
        else
            t.initBxDF(t.rc);

        t.meta.result = t.test();
        t.meta.testName = "u offset";
        return t.meta;
    }
};


template<class BxDF, bool aniso = false>
struct TestVOffset : TestBxDF<BxDF>
{
    using base_t = TestBase<BxDF>;
    using this_t = TestVOffset<BxDF, aniso>;

    float32_t4 test()
    {
        sample_t s, sx, sy, sz;
        quotient_pdf_t pdf;
        float32_t3 brdf;
        aniso_cache cache, dummy;

        iso_interaction isointerx = iso_interaction::create(base_t::rc.dV(0), base_t::rc.N);
        aniso_interaction anisointerx = aniso_interaction::create(isointerx, base_t::rc.T, base_t::rc.B);
        iso_interaction isointery = iso_interaction::create(base_t::rc.dV(1), base_t::rc.N);
        aniso_interaction anisointery = aniso_interaction::create(isointery, base_t::rc.T, base_t::rc.B);
        iso_interaction isointerz = iso_interaction::create(base_t::rc.dV(2), base_t::rc.N);
        aniso_interaction anisointerz = aniso_interaction::create(isointerz, base_t::rc.T, base_t::rc.B);

        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy);
            sx = base_t::bxdf.generate(anisointerx, base_t::rc.u.xy);
            sy = base_t::bxdf.generate(anisointery, base_t::rc.u.xy);
            sz = base_t::bxdf.generate(anisointerz, base_t::rc.u.xy);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy, cache);
            sx = base_t::bxdf.generate(anisointerx, base_t::rc.u.xy, dummy);
            sy = base_t::bxdf.generate(anisointery, base_t::rc.u.xy, dummy);
            sz = base_t::bxdf.generate(anisointerz, base_t::rc.u.xy, dummy);
        }
        if NBL_CONSTEXPR_FUNC (is_basic_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u);
            float32_t3 ux = base_t::rc.u + float32_t3(base_t::rc.h,0,0);
            sx = base_t::bxdf.generate(anisointerx, ux);
            float32_t3 uy = base_t::rc.u + float32_t3(0,base_t::rc.h,0);
            sy = base_t::bxdf.generate(anisointery, uy);
            float32_t3 uz = base_t::rc.u + float32_t3(0,0,base_t::rc.h);
            sz = base_t::bxdf.generate(anisointerz, uz);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u, cache);
            float32_t3 ux = base_t::rc.u + float32_t3(base_t::rc.h,0,0);
            sx = base_t::bxdf.generate(anisointerx, ux, dummy);
            float32_t3 uy = base_t::rc.u + float32_t3(0,base_t::rc.h,0);
            sy = base_t::bxdf.generate(anisointery, uy, dummy);
            float32_t3 uz = base_t::rc.u + float32_t3(0,0,base_t::rc.h);
            sz = base_t::bxdf.generate(anisointerz, uz, dummy);
        }
        
        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF> || is_basic_bsdf_v<BxDF>)
        {
            pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer);
            brdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer));
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                pdf = base_t::bxdf.quotient_and_pdf(s, base_t::anisointer, cache);
                brdf = float32_t3(base_t::bxdf.eval(s, base_t::anisointer, cache));
            }
            else
            {
                iso_cache isocache = (iso_cache)cache;
                pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer, isocache);
                brdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer, isocache));
            }
        }

        // get jacobian
        float32_t3x3 m = float32_t3x3(sx.TdotL - s.TdotL, sy.TdotL - s.TdotL, sx.TdotL - s.TdotL, sx.BdotL - s.BdotL, sy.BdotL - s.BdotL, sz.BdotL - s.BdotL, sx.NdotL - s.NdotL, sy.NdotL - s.NdotL, sz.NdotL - s.NdotL);
        float det = nbl::hlsl::determinant<float32_t3x3>(m);

        return float32_t4(nbl::hlsl::abs<float32_t3>(pdf.value() - brdf), nbl::hlsl::abs<float>(det*pdf.pdf/s.NdotL) * 0.5);
    }

    static STestMeta run(uint32_t2 seed)
    {
        this_t t;
        t.init(seed);
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
            t.template initBxDF<aniso>(t.rc);
        else
            t.initBxDF(t.rc);

        t.meta.result = t.test();
        t.meta.testName = "V offset";
        return t.meta;
    }
};

}
}

#endif