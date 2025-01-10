#ifndef BXDFTESTS_TESTS_HLSL
#define BXDFTESTS_TESTS_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"
#include "nbl/builtin/hlsl/sampling/uniform.hlsl"
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
        core::quaternion rot;
        float angle = 2 * numbers::pi<float> * rngUniformDist<float>(retval.rng);
        core::vectorSIMDf N = core::vectorSIMDf(retval.N.x, retval.N.y, retval.N.z);
        rot.toAngleAxis(angle, N);
        core::vectorSIMDf tmp = rot.transformVect(core::vectorSIMDf(tb[0].x, tb[0].y, tb[0].z));
        retval.T = float32_t3(tmp[0],tmp[1],tmp[2]);
        tmp = rot.transformVect(core::vectorSIMDf(tb[1].x, tb[1].y, tb[1].z));
        retval.B = float32_t3(tmp[0],tmp[1],tmp[2]);
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
            base_t::bxdf = bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,rc.alpha.y,float32_t3x2(rc.ior,rc.ior,rc.ior));
#ifndef __HLSL_VERSION
            base_t::meta.bxdfName = "BeckmannBRDF Aniso";
#endif
        }
        else
        {
            base_t::bxdf = bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,float32_t3x2(rc.ior,rc.ior,rc.ior));
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
            base_t::bxdf = bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,rc.alpha.y,float32_t3x2(rc.ior,rc.ior,rc.ior));
#ifndef __HLSL_VERSION
            base_t::meta.bxdfName = "GGXBRDF Aniso";
#endif
        }
        else
        {
            base_t::bxdf = bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>::create(rc.alpha.x,float32_t3x2(rc.ior,rc.ior,rc.ior));
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
        base_t::bxdf = bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>::create(float32_t3(rc.eta * rc.eta),rc.luma_coeff);
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

        return float32_t4(nbl::hlsl::abs<float32_t3>(pdf.value() - brdf), nbl::hlsl::abs<float>(det*pdf.pdf/s.NdotL));
    }

    static STestMeta run(uint32_t seed)
    {
        uint32_t2 state = pcg2d_hash(seed);

        this_t t;
        t.init(state);
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
            t.template initBxDF<aniso>(t.rc);
        else
            t.initBxDF(t.rc);

        t.meta.result = t.test();
        t.meta.testName = "u offset";
        return t.meta;
    }
};

}
}

#endif