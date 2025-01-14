#ifndef BXDFTESTS_TESTS_HLSL
#define BXDFTESTS_TESTS_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"
#include "nbl/builtin/hlsl/random/pcg.hlsl"
#include "nbl/builtin/hlsl/sampling/uniform.hlsl"
#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"

#ifndef __HLSL_VERSION
#include <glm/gtc/quaternion.hpp>
#endif

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
using spectral_t = vector<float, 3>;

using bool32_t3 = vector<bool, 3>;

// uint32_t pcg_hash(uint32_t v)
// {
//     uint32_t state = v * 747796405u + 2891336453u;
//     uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
//     return (word >> 22u) ^ word;
// }

// uint32_t2 pcg2d_hash(uint32_t v)
// {
//     return uint32_t2(pcg_hash(v), pcg_hash(v+1));
// }

namespace impl
{

inline float rngFloat01(NBL_REF_ARG(nbl::hlsl::Xoroshiro64Star) rng)
{
    return (float)rng() / numeric_limits<uint32_t>::max;
}

template<typename T>
struct RNGUniformDist;

template<>
struct RNGUniformDist<float32_t>
{
    static float32_t __call(NBL_REF_ARG(nbl::hlsl::Xoroshiro64Star) rng)
    {
        return rngFloat01(rng);
    }
};

template<uint16_t N>
struct RNGUniformDist<vector<float32_t, N>>
{
    static vector<float32_t, N> __call(NBL_REF_ARG(nbl::hlsl::Xoroshiro64Star) rng)
    {
        vector<float32_t, N> retval;
        for (int i = 0; i < N; i++)
            retval[i] = rngFloat01(rng);
        return retval;
    }
};

}

template<typename T>
T rngUniformDist(NBL_REF_ARG(nbl::hlsl::Xoroshiro64Star) rng)
{
    return impl::RNGUniformDist<T>::__call(rng);
}


struct SBxDFTestResources
{
    static SBxDFTestResources create(uint32_t2 seed)
    {
        SBxDFTestResources retval;
        retval.rng = nbl::hlsl::Xoroshiro64Star::construct(seed);
        retval.u = float32_t3(rngUniformDist<float32_t2>(retval.rng), 0.0);

        retval.V.direction = nbl::hlsl::normalize<float32_t3>(uniform_sphere_generate<float>(rngUniformDist<float32_t2>(retval.rng)));
        retval.N = nbl::hlsl::normalize<float32_t3>(uniform_sphere_generate<float>(rngUniformDist<float32_t2>(retval.rng)));
        
        float32_t2x3 tb = math::frisvad<float>(retval.N);
#ifndef __HLSL_VERSION
        const float angle = 2 * numbers::pi<float> * rngUniformDist<float>(retval.rng);
        glm::quat rot = glm::angleAxis(angle, retval.N);
        retval.T = rot * tb[0];
        retval.B = rot * tb[1];
#else
        retval.T = tb[0];
        retval.B = tb[1];
#endif

        retval.alpha.x = rngUniformDist<float>(retval.rng);
        retval.alpha.y = rngUniformDist<float>(retval.rng);
        retval.eta = 1.3;
        retval.ior = float32_t2(1.3, 2.0);
        retval.luma_coeff = float32_t3(0.2126, 0.7152, 0.0722); // luma coefficients for Rec. 709
        return retval;
    }

    // epsilon
    float eps = 1e-3;

    nbl::hlsl::Xoroshiro64Star rng;
    ray_dir_info_t V;
    float32_t3 N;
    float32_t3 T;
    float32_t3 B;

    float32_t3 u;
    float32_t2 alpha;
    float eta;
    float32_t2 ior;
    float32_t3 luma_coeff;
};

enum ErrorType : uint32_t
{
    NOERR = 0,
    NEGATIVE_VAL,   // pdf/quotient/eval < 0
    PDF_ZERO,           // pdf = 0
    QUOTIENT_INF,       // quotient -> inf
    JACOBIAN,
    PDF_EVAL_DIFF,
    RECIPROCITY
};

struct TestBase
{
    void init(uint32_t2 seed)
    {
        rc = SBxDFTestResources::create(seed);

        isointer = iso_interaction::create(rc.V, rc.N);
        anisointer = aniso_interaction::create(isointer, rc.T, rc.B);
    }

    virtual void compute() {}

    SBxDFTestResources rc;

    iso_interaction isointer;
    aniso_interaction anisointer;

#ifndef __HLSL_VERSION
    std::string name = "base";
#endif
};

struct FailureCallback
{
    virtual void __call(ErrorType error, NBL_REF_ARG(TestBase) failedFor) {}
};

template<class BxDF>
struct TestBxDFBase : TestBase
{
    BxDF bxdf;
};

template<class BxDF>
struct TestBxDF : TestBxDFBase<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf = BxDF::create();  // default to lambertian bxdf
#ifndef __HLSL_VERSION
        base_t::name = "Lambertian BxDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>> : TestBxDFBase<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>
{
    using base_t = TestBxDFBase<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf = bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>::create(_rc.alpha.x);
#ifndef __HLSL_VERSION
        base_t::name = "OrenNayar BRDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>> : TestBxDFBase<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>
{
    using base_t = TestBxDFBase<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>;

    template<bool aniso>
    void initBxDF(SBxDFTestResources _rc)
    {
        if (aniso)
        {
            base_t::bxdf = bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>::create(rc.alpha.x,rc.alpha.y,(float32_t3)(rc.ior.x),(float32_t3)(rc.ior.y));
#ifndef __HLSL_VERSION
            base_t::name = "Beckmann Aniso BRDF";
#endif
        }
        else
        {
            base_t::bxdf = bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>::create(rc.alpha.x,(float32_t3)(rc.ior.x),(float32_t3)(rc.ior.y));
#ifndef __HLSL_VERSION
            base_t::name = "Beckmann BRDF";
#endif
        }
    }
};

template<>
struct TestBxDF<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>> : TestBxDFBase<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>
{
    using base_t = TestBxDFBase<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>;

    template<bool aniso>
    void initBxDF(SBxDFTestResources _rc)
    {
        if (aniso)
        {
            base_t::bxdf = bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>::create(rc.alpha.x,rc.alpha.y,(float32_t3)(rc.ior.x),(float32_t3)(rc.ior.y));
#ifndef __HLSL_VERSION
            base_t::name = "GGX Aniso BRDF";
#endif
        }
        else
        {
            base_t::bxdf = bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>::create(rc.alpha.x,(float32_t3)(rc.ior.x),(float32_t3)(rc.ior.y));
#ifndef __HLSL_VERSION
            base_t::name = "GGX BRDF";
#endif
        }
    }
};

template<>
struct TestBxDF<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>> : TestBxDFBase<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf = bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>::create(rc.eta);
#ifndef __HLSL_VERSION
        base_t::name = "Smooth dielectric BSDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>> : TestBxDFBase<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf = bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>::create(float32_t3(rc.eta * rc.eta),rc.luma_coeff);
#ifndef __HLSL_VERSION
        base_t::name = "Thin smooth dielectric BSDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>> : TestBxDFBase<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>;

    template<bool aniso>
    void initBxDF(SBxDFTestResources _rc)
    {
        if (aniso)
        {
            base_t::bxdf = bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>::create(rc.eta,rc.alpha.x,rc.alpha.y);
#ifndef __HLSL_VERSION
            base_t::name = "Beckmann Dielectric Aniso BSDF";
#endif
        }
        else
        {
            base_t::bxdf = bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>::create(rc.eta,rc.alpha.x);
#ifndef __HLSL_VERSION
            base_t::name = "Beckmann Dielectric BSDF";
#endif
        }
    }
};

template<>
struct TestBxDF<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>> : TestBxDFBase<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>;

    template<bool aniso>
    void initBxDF(SBxDFTestResources _rc)
    {
        if (aniso)
        {
            base_t::bxdf = bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>::create(rc.eta,rc.alpha.x,rc.alpha.y);
#ifndef __HLSL_VERSION
            base_t::name = "GGX Dielectric Aniso BSDF";
#endif
        }
        else
        {
            base_t::bxdf = bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>::create(rc.eta,rc.alpha.x);
#ifndef __HLSL_VERSION
            base_t::name = "GGX Dielectric BSDF";
#endif
        }
    }
};


template<class T>
struct is_basic_brdf : bool_constant<
    is_same<T, bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::value ||
    is_same<T, bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::value
> {};

template<class T>
struct is_microfacet_brdf : bool_constant<
    is_same<T, bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::value ||
    is_same<T, bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::value
> {};

template<class T>
struct is_basic_bsdf : bool_constant<
    is_same<T, bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::value ||
    is_same<T, bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::value ||
    is_same<T, bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::value
> {};

template<class T>
struct is_microfacet_bsdf : bool_constant<
    is_same<T, bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::value ||
    is_same<T, bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::value
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
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestUOffset<BxDF, aniso>;

    void compute() override
    {
        aniso_cache cache, dummy;

        float32_t3 ux = base_t::rc.u + float32_t3(base_t::rc.eps,0,0);
        float32_t3 uy = base_t::rc.u + float32_t3(0,base_t::rc.eps,0);

        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy);
            sx = base_t::bxdf.generate(base_t::anisointer, ux.xy);
            sy = base_t::bxdf.generate(base_t::anisointer, uy.xy);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy, cache);
            sx = base_t::bxdf.generate(base_t::anisointer, ux.xy, dummy);
            sy = base_t::bxdf.generate(base_t::anisointer, uy.xy, dummy);
        }
        if NBL_CONSTEXPR_FUNC (is_basic_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u);
            sx = base_t::bxdf.generate(base_t::anisointer, ux);
            sy = base_t::bxdf.generate(base_t::anisointer, uy);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u, cache);
            sx = base_t::bxdf.generate(base_t::anisointer, ux, dummy);
            sy = base_t::bxdf.generate(base_t::anisointer, uy, dummy);
        }
        
        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF> || is_basic_bsdf_v<BxDF>)
        {
            pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer);
            bsdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer));
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                pdf = base_t::bxdf.quotient_and_pdf(s, base_t::anisointer, cache);
                bsdf = float32_t3(base_t::bxdf.eval(s, base_t::anisointer, cache));
            }
            else
            {
                iso_cache isocache = (iso_cache)cache;
                pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer, isocache);
                bsdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer, isocache));
            }
        }
    }

    ErrorType test()
    {
        compute();

        if (nbl::hlsl::abs<float>(pdf.pdf) < base_t::rc.eps)  // something generated cannot have 0 probability of getting generated
            return PDF_ZERO;

        if (!all<bool32_t3>(pdf.quotient < (float32_t3)numeric_limits<float>::infinity))    // importance sampler's job to prevent inf
            return QUOTIENT_INF;

        if (all<bool32_t3>(nbl::hlsl::abs<float32_t3>(bsdf) < (float32_t3)base_t::rc.eps) || all<bool32_t3>(pdf.quotient < (float32_t3)base_t::rc.eps))
            return NOERR;    // produces an "impossible" sample

        // get jacobian
        float32_t2x2 m = float32_t2x2(sx.TdotL - s.TdotL, sy.TdotL - s.TdotL, sx.BdotL - s.BdotL, sy.BdotL - s.BdotL);
        float det = nbl::hlsl::determinant<float32_t2x2>(m);

        bool jacobian_test = nbl::hlsl::abs<float>(det*pdf.pdf/s.NdotL) < base_t::rc.eps;
        if (!jacobian_test)
            return JACOBIAN;

        bool32_t3 diff_test = nbl::hlsl::max<float32_t3>(pdf.value() / bsdf, bsdf / pdf.value()) <= (float32_t3)(1 + base_t::rc.eps);
        if (!all<bool32_t3>(diff_test))
            return PDF_EVAL_DIFF;

        return NOERR;
    }

    static void run(uint32_t seed, NBL_REF_ARG(FailureCallback) cb)
    {
        uint32_t2 state = pcg32x2(seed);

        this_t t;
        t.init(state);
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
            t.template initBxDF<aniso>(t.rc);
        else
            t.initBxDF(t.rc);
        
        ErrorType e = t.test();
        if (e != NOERR)
            cb.__call(e, t);
    }

    sample_t s, sx, sy;
    quotient_pdf_t pdf;
    float32_t3 bsdf;
};

}
}

#endif