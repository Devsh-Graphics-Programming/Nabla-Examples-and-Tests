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
#include <glm/gtx/hash.hpp>
#include <unordered_map>
#include <cmath>
#include <format>
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
using params_t = bxdf::SBxDFParams<float>;

using bool32_t3 = vector<bool, 3>;

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

template<typename T>
bool checkEq(T a, T b, float32_t eps)
{
    return nbl::hlsl::all<vector<bool, T::length()>>(nbl::hlsl::max<T>(a / b, b / a) <= (T)(1 + eps));
}

template<typename T>
bool checkLt(T a, T b)
{
    return nbl::hlsl::all<vector<bool, T::length()>>(a < b);
}

template<typename T>
bool checkZero(T a, float32_t eps)
{
    return nbl::hlsl::all<vector<bool, T::length()>>(nbl::hlsl::abs<T>(a) < (T)eps);
}

template<>
bool checkZero<float32_t>(float32_t a, float32_t eps)
{
    return nbl::hlsl::abs<float32_t>(a) < eps;
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

    float eps = 1e-3;   // epsilon
    uint32_t state;     // init state seed, for debugging

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
    BET_NONE = 0,
    BET_NEGATIVE_VAL,       // pdf/quotient/eval < 0
    BET_PDF_ZERO,           // pdf = 0
    BET_QUOTIENT_INF,       // quotient -> inf
    BET_JACOBIAN,
    BET_PDF_EVAL_DIFF,
    BET_RECIPROCITY,
    BET_PRINT_MSG
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
    std::string errMsg = "";
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
struct TestJacobian : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestJacobian<BxDF, aniso>;

    virtual void compute() override
    {
        aniso_cache cache, dummy;
        iso_cache isocache;
        params_t params;

        float32_t3 ux = base_t::rc.u + float32_t3(base_t::rc.eps,0,0);
        float32_t3 uy = base_t::rc.u + float32_t3(0,base_t::rc.eps,0);

        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy);
            sx = base_t::bxdf.generate(base_t::anisointer, ux.xy);
            sy = base_t::bxdf.generate(base_t::anisointer, uy.xy);
            params = params_t::template create<sample_t, iso_interaction>(s, base_t::isointer, bxdf::BCM_MAX);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy, cache);
            sx = base_t::bxdf.generate(base_t::anisointer, ux.xy, dummy);
            sy = base_t::bxdf.generate(base_t::anisointer, uy.xy, dummy);

            if NBL_CONSTEXPR_FUNC (aniso)
                params = params_t::template create<sample_t, aniso_interaction, aniso_cache>(s, base_t::anisointer, cache, bxdf::BCM_MAX);
            else
            {
                isocache = (iso_cache)cache;
                params = params_t::template create<sample_t, iso_interaction, iso_cache>(s, base_t::isointer, isocache, bxdf::BCM_MAX);
            }
        }
        if NBL_CONSTEXPR_FUNC (is_basic_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u);
            sx = base_t::bxdf.generate(base_t::anisointer, ux);
            sy = base_t::bxdf.generate(base_t::anisointer, uy);
            params = params_t::template create<sample_t, iso_interaction>(s, base_t::isointer, bxdf::BCM_ABS);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u, cache);
            sx = base_t::bxdf.generate(base_t::anisointer, ux, dummy);
            sy = base_t::bxdf.generate(base_t::anisointer, uy, dummy);

            if NBL_CONSTEXPR_FUNC (aniso)
                params = params_t::template create<sample_t, aniso_interaction, aniso_cache>(s, base_t::anisointer, cache, bxdf::BCM_ABS);
            else
            {
                isocache = (iso_cache)cache;
                params = params_t::template create<sample_t, iso_interaction, iso_cache>(s, base_t::isointer, isocache, bxdf::BCM_ABS);
            }
        }

        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF> || is_basic_bsdf_v<BxDF>)
        {
            pdf = base_t::bxdf.quotient_and_pdf(params);
            bsdf = float32_t3(base_t::bxdf.eval(params));
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                pdf = base_t::bxdf.quotient_and_pdf(params);
                bsdf = float32_t3(base_t::bxdf.eval(params));
            }
            else
            {
                pdf = base_t::bxdf.quotient_and_pdf(params);
                bsdf = float32_t3(base_t::bxdf.eval(params));
            }
        }
    }

    ErrorType test()
    {
        compute();

        if (checkZero<float>(pdf.pdf, 1e-5))  // something generated cannot have 0 probability of getting generated
            return BET_PDF_ZERO;

        if (!checkLt<float32_t3>(pdf.quotient, (float32_t3)numeric_limits<float>::infinity))    // importance sampler's job to prevent inf
            return BET_QUOTIENT_INF;

        if (checkZero<float32_t3>(bsdf, 1e-5) || checkZero<float32_t3>(pdf.quotient, 1e-5))
            return BET_NONE;    // produces an "impossible" sample

        if (checkLt<float32_t3>(bsdf, (float32_t3)0.0) || checkLt<float32_t3>(pdf.quotient, (float32_t3)0.0) || pdf.pdf < 0.0)
            return BET_NEGATIVE_VAL;

        // get BET_jacobian
        float32_t2x2 m = float32_t2x2(sx.TdotL - s.TdotL, sy.TdotL - s.TdotL, sx.BdotL - s.BdotL, sy.BdotL - s.BdotL);
        float det = nbl::hlsl::determinant<float32_t2x2>(m);

        if (!checkZero<float>(det * pdf.pdf / s.NdotL, 1e-5))
            return BET_JACOBIAN;

        if (!checkEq<float32_t3>(pdf.value(), bsdf, 5e-2))
            return BET_PDF_EVAL_DIFF;

        return BET_NONE;
    }

    static void run(uint32_t seed, NBL_REF_ARG(FailureCallback) cb)
    {
        uint32_t2 state = pcg32x2(seed);

        this_t t;
        t.init(state);
        t.rc.state = seed;
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
            t.template initBxDF<aniso>(t.rc);
        else
            t.initBxDF(t.rc);
        
        ErrorType e = t.test();
        if (e != BET_NONE)
            cb.__call(e, t);
    }

    sample_t s, sx, sy;
    quotient_pdf_t pdf;
    float32_t3 bsdf;
};

template<class BxDF, bool aniso = false>
struct TestReciprocity : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestReciprocity<BxDF, aniso>;

    virtual void compute() override
    {
        aniso_cache cache, rec_cache;
        iso_cache isocache, rec_isocache;

        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy);
            params = params_t::template create<sample_t, iso_interaction>(s, base_t::isointer, bxdf::BCM_MAX);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy, cache);

            if NBL_CONSTEXPR_FUNC (aniso)
                params = params_t::template create<sample_t, aniso_interaction, aniso_cache>(s, base_t::anisointer, cache, bxdf::BCM_MAX);
            else
            {
                isocache = (iso_cache)cache;
                params = params_t::template create<sample_t, iso_interaction, iso_cache>(s, base_t::isointer, isocache, bxdf::BCM_MAX);
            }
        }
        if NBL_CONSTEXPR_FUNC (is_basic_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u);
            params = params_t::template create<sample_t, iso_interaction>(s, base_t::isointer, bxdf::BCM_ABS);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u, cache);

            if NBL_CONSTEXPR_FUNC (aniso)
                params = params_t::template create<sample_t, aniso_interaction, aniso_cache>(s, base_t::anisointer, cache, bxdf::BCM_ABS);
            else
            {
                isocache = (iso_cache)cache;
                params = params_t::template create<sample_t, iso_interaction, iso_cache>(s, base_t::isointer, isocache, bxdf::BCM_ABS);
            }
        }

        float32_t3x3 toTangentSpace = base_t::anisointer.getToTangentSpace();
        ray_dir_info_t rec_V = s.L;
        ray_dir_info_t rec_localV = ray_dir_info_t::transform(toTangentSpace, rec_V);
        ray_dir_info_t rec_localL = ray_dir_info_t::transform(toTangentSpace, base_t::rc.V);
        rec_s = sample_t::createFromTangentSpace(rec_localV.direction, rec_localL, base_t::anisointer.getFromTangentSpace());

        iso_interaction rec_isointer = iso_interaction::create(rec_V, base_t::rc.N);
        aniso_interaction rec_anisointer = aniso_interaction::create(rec_isointer, base_t::rc.T, base_t::rc.B);
        rec_cache = cache;
        rec_cache.VdotH = cache.LdotH;
        rec_cache.LdotH = cache.VdotH;

        rec_isocache = (iso_cache)rec_cache;

        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF>)
            rec_params = params_t::template create<sample_t, iso_interaction>(rec_s, rec_isointer, bxdf::BCM_MAX);
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
                rec_params = params_t::template create<sample_t, aniso_interaction, aniso_cache>(rec_s, rec_anisointer, rec_cache, bxdf::BCM_MAX);
            else
            {
                rec_isocache = (iso_cache)rec_cache;
                rec_params = params_t::template create<sample_t, iso_interaction, iso_cache>(rec_s, rec_isointer, rec_isocache, bxdf::BCM_MAX);
            }
        }
        if NBL_CONSTEXPR_FUNC (is_basic_bsdf_v<BxDF>)
            rec_params = params_t::template create<sample_t, iso_interaction>(rec_s, rec_isointer, bxdf::BCM_ABS);
        if NBL_CONSTEXPR_FUNC (is_microfacet_bsdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
                rec_params = params_t::template create<sample_t, aniso_interaction, aniso_cache>(rec_s, rec_anisointer, rec_cache, bxdf::BCM_ABS);
            else
            {
                rec_isocache = (iso_cache)rec_cache;
                rec_params = params_t::template create<sample_t, iso_interaction, iso_cache>(rec_s, rec_isointer, rec_isocache, bxdf::BCM_ABS);
            }
        }
        
        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF> || is_basic_bsdf_v<BxDF>)
        {
            bsdf = float32_t3(base_t::bxdf.eval(params));
            rec_bsdf = float32_t3(base_t::bxdf.eval(rec_params));
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                bsdf = float32_t3(base_t::bxdf.eval(params));
                rec_bsdf = float32_t3(base_t::bxdf.eval(rec_params));
            }
            else
            {
                bsdf = float32_t3(base_t::bxdf.eval(params));
                rec_bsdf = float32_t3(base_t::bxdf.eval(rec_params));
            }
        }
    }

    ErrorType test()
    {
        compute();

        if (checkZero<float32_t3>(bsdf, 1e-5))
            return BET_NONE;    // produces an "impossible" sample

        if (checkLt<float32_t3>(bsdf, (float32_t3)0.0))
            return BET_NEGATIVE_VAL;

        float32_t3 a = bsdf * nbl::hlsl::abs<float>(params.NdotV);
        float32_t3 b = rec_bsdf * nbl::hlsl::abs<float>(rec_params.NdotV);
        if (!(a == b))  // avoid division by 0
            if (!checkEq<float32_t3>(a, b, 1e-2))
                return BET_RECIPROCITY;

        return BET_NONE;
    }

    static void run(uint32_t seed, NBL_REF_ARG(FailureCallback) cb)
    {
        uint32_t2 state = pcg32x2(seed);

        this_t t;
        t.init(state);
        t.rc.state = seed;
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
            t.template initBxDF<aniso>(t.rc);
        else
            t.initBxDF(t.rc);
        
        ErrorType e = t.test();
        if (e != BET_NONE)
            cb.__call(e, t);
    }

    sample_t s, rec_s;
    float32_t3 bsdf, rec_bsdf;
    params_t params, rec_params;
};

#ifndef __HLSL_VERSION  // because unordered_map
template<class BxDF, bool aniso = false>
struct TestBucket : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestBucket<BxDF, aniso>;

    void clearBuckets()
    {
        for (float z = -1.0f; z < 1.0f; z += stride)
        {
            for (float y = -1.0f; y < 1.0f; y += stride)
            {
                for (float x = -1.0f; x < 1.0f; x += stride)
                {
                    buckets[float32_t3(x, y, z)] = 0;
                }
            }
        }
    }

    float bin(float a)
    {
        float diff = std::fmod(a, stride);
        float b = (a < 0) ? -0.2f : 0.0f;
        return a - diff + b;
    }

    virtual void compute() override
    {
        clearBuckets();

        aniso_cache cache, dummy;
        iso_cache isocache;
        params_t params;

        sample_t s;
        quotient_pdf_t pdf;
        float32_t3 bsdf;

        NBL_CONSTEXPR uint32_t samples = 500;
        for (uint32_t i = 0; i < samples; i++)
        {
            float32_t3 u = float32_t3(rngUniformDist<float32_t2>(base_t::rc.rng), 0.0);

            if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF>)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u.xy);
                params = params_t::template create<sample_t, iso_interaction>(s, base_t::isointer, bxdf::BCM_MAX);
            }
            if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF>)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u.xy, cache);

                if NBL_CONSTEXPR_FUNC (aniso)
                    params = params_t::template create<sample_t, aniso_interaction, aniso_cache>(s, base_t::anisointer, cache, bxdf::BCM_MAX);
                else
                {
                    isocache = (iso_cache)cache;
                    params = params_t::template create<sample_t, iso_interaction, iso_cache>(s, base_t::isointer, isocache, bxdf::BCM_MAX);
                }
            }
            if NBL_CONSTEXPR_FUNC (is_basic_bsdf_v<BxDF>)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u);
                params = params_t::template create<sample_t, iso_interaction>(s, base_t::isointer, bxdf::BCM_ABS);
            }
            if NBL_CONSTEXPR_FUNC (is_microfacet_bsdf_v<BxDF>)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u, cache);

                if NBL_CONSTEXPR_FUNC (aniso)
                    params = params_t::template create<sample_t, aniso_interaction, aniso_cache>(s, base_t::anisointer, cache, bxdf::BCM_ABS);
                else
                {
                    isocache = (iso_cache)cache;
                    params = params_t::template create<sample_t, iso_interaction, iso_cache>(s, base_t::isointer, isocache, bxdf::BCM_ABS);
                }
            }

            // put s into bucket
            // TODO: probably change to bucket with polar coords
            const float32_t3 L = s.L.direction;
            float32_t3 bucket = float32_t3(bin(L.x), bin(L.y), bin(L.z));

            if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF> || is_basic_bsdf_v<BxDF>)
            {
                pdf = base_t::bxdf.quotient_and_pdf(params);
                bsdf = float32_t3(base_t::bxdf.eval(params));
            }
            if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
            {
                if NBL_CONSTEXPR_FUNC (aniso)
                {
                    pdf = base_t::bxdf.quotient_and_pdf(params);
                    bsdf = float32_t3(base_t::bxdf.eval(params));
                }
                else
                {
                    pdf = base_t::bxdf.quotient_and_pdf(params);
                    bsdf = float32_t3(base_t::bxdf.eval(params));
                }
            }

            // check pdf == INF
            if (pdf.pdf == numeric_limits<float>::infinity)
                buckets[bucket] += 1;
        }

#ifndef __HLSL_VERSION
        for (auto const& b : buckets) {
            if (!selective || b.second > 0)
            {
                const float32_t3 v = b.first;
                base_t::errMsg += std::format("({:.1f},{:.1f},{:.1f}): {}\n", v.x, v.y, v.z, b.second);
            }
        }
#endif
    }

    ErrorType test()
    {
        compute();

        return (base_t::errMsg.length() == 0) ? BET_NONE : BET_PRINT_MSG;
    }

    static void run(uint32_t seed, NBL_REF_ARG(FailureCallback) cb)
    {
        uint32_t2 state = pcg32x2(seed);

        this_t t;
        t.init(state);
        t.rc.state = seed;
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
            t.template initBxDF<aniso>(t.rc);
        else
            t.initBxDF(t.rc);
        
        ErrorType e = t.test();
        if (e != BET_NONE)
            cb.__call(e, t);
    }

    bool selective = true;  // print only buckets with count > 0
    float stride = 0.2f;
    std::unordered_map<float32_t3, uint32_t, std::hash<float32_t3>> buckets;
};
#endif

}
}

#endif