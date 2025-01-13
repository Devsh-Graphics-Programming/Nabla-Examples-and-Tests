#ifndef BXDFTESTS_TESTS_HLSL
#define BXDFTESTS_TESTS_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"
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

using bool32_t3 = vector<bool, 3>;

uint32_t pcg_hash(uint32_t v)
{
    uint32_t state = v * 747796405u + 2891336453u;
	uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

uint32_t2 pcg2d_hash(uint32_t v)
{
    return uint32_t2(pcg_hash(v), pcg_hash(v+1));
}

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

    float h = 0.001;

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
    NEGATIVE_VAL = 0,   // pdf/quotient/eval < 0
    PDF_ZERO,           // pdf = 0
    QUOTIENT_INF,       // quotient -> inf
    JACOBIAN,
    PDF_EVAL_DIFF,
    RECIPROCITY
};

struct FailureCallback
{
    void __call(ErrorType error, NBL_CONST_REF_ARG(SBxDFTestResources) failedFor, NBL_CONST_REF_ARG(sample_t) failedAt) {}
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
};


template<class BxDF>
struct TestBxDF : TestBase<BxDF>
{
    using base_t = TestBase<BxDF>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf = BxDF::create();  // default to lambertian bxdf
    }
};

template<>
struct TestBxDF<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>> : TestBase<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>>
{
    using base_t = TestBase<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf = bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>::create(_rc.alpha.x);
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
            base_t::bxdf = bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,rc.alpha.y,float32_t3x2(rc.ior,rc.ior,rc.ior));
        }
        else
        {
            base_t::bxdf = bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,float32_t3x2(rc.ior,rc.ior,rc.ior));
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
            base_t::bxdf = bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,rc.alpha.y,float32_t3x2(rc.ior,rc.ior,rc.ior));
        }
        else
        {
            base_t::bxdf = bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,float32_t3x2(rc.ior,rc.ior,rc.ior));
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
    }
};

template<>
struct TestBxDF<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>> : TestBase<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>>
{
    using base_t = TestBase<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf = bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>::create(float32_t3(rc.eta * rc.eta),rc.luma_coeff);
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
        }
        else
        {
            base_t::bxdf = bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>::create(rc.eta,rc.alpha.x);
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
        }
        else
        {
            base_t::bxdf = bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>::create(rc.eta,rc.alpha.x);
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

    bool test(NBL_CONST_REF_ARG(FailureCallback) cb)
    {
        sample_t s, sx, sy;
        quotient_pdf_t pdf;
        float32_t3 bsdf;
        aniso_cache cache, dummy;

        float32_t3 ux = base_t::rc.u + float32_t3(base_t::rc.h,0,0);
        float32_t3 uy = base_t::rc.u + float32_t3(0,base_t::rc.h,0);

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

        if (nbl::hlsl::abs<float>(pdf.pdf) < 1e-3)
        {
            cb::__call(PDF_ZERO, base_t::rc, s);
            return false;
        }

        // get jacobian
        float32_t2x2 m = float32_t2x2(sx.TdotL - s.TdotL, sy.TdotL - s.TdotL, sx.BdotL - s.BdotL, sy.BdotL - s.BdotL);
        float det = nbl::hlsl::determinant<float32_t2x2>(m);

        bool jacobian_test = nbl::hlsl::abs<float>(det*pdf.pdf/s.NdotL) < 1e-3;
        if (!jacobian_test)
            cb::__call(JACOBIAN, base_t::rc, s);

        bool diff_test = nbl::hlsl::abs<float32_t3>(pdf.value() - bsdf) < 1e-3;
        if (!diff_test)
            cb::__call(PDF_EVAL_DIFF, base_t::rc, s);

        return true;
    }

    static void run(uint32_t seed, NBL_CONST_REF_ARG(FailureCallback) cb)
    {
        uint32_t2 state = pcg2d_hash(seed);

        this_t t;
        t.init(state);
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
            t.template initBxDF<aniso>(t.rc);
        else
            t.initBxDF(t.rc);
        t.test(cb);
    }
};

}
}

#endif