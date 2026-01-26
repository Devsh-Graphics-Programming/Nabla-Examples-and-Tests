#ifndef BXDFTESTS_TESTS_COMMON_HLSL
#define BXDFTESTS_TESTS_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"
#include "nbl/builtin/hlsl/random/pcg.hlsl"
#include "nbl/builtin/hlsl/random/dim_adaptor_recursive.hlsl"
#include "nbl/builtin/hlsl/sampling/uniform_spheres.hlsl"
#include "nbl/builtin/hlsl/math/quaternions.hlsl"
#include "nbl/builtin/hlsl/math/polar.hlsl"
#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"
#include "nbl/builtin/hlsl/testing/relative_approx_compare.hlsl"

using namespace nbl;
using namespace hlsl;

using spectral_t = hlsl::vector<float, 3>;
using ray_dir_info_t = bxdf::ray_dir_info::SBasic<float>;
using iso_interaction = bxdf::surface_interactions::SIsotropic<ray_dir_info_t, spectral_t>;
using aniso_interaction = bxdf::surface_interactions::SAnisotropic<iso_interaction>;
using sample_t = bxdf::SLightSample<ray_dir_info_t>;
using iso_cache = bxdf::SIsotropicMicrofacetCache<float>;
using aniso_cache = bxdf::SAnisotropicMicrofacetCache<iso_cache>;
using quotient_pdf_t = sampling::quotient_and_pdf<float32_t3, float>;

using iso_config_t = bxdf::SConfiguration<sample_t, iso_interaction, spectral_t>;
using aniso_config_t = bxdf::SConfiguration<sample_t, aniso_interaction, spectral_t>;
using iso_microfacet_config_t = bxdf::SMicrofacetConfiguration<sample_t, iso_interaction, iso_cache, spectral_t>;
using aniso_microfacet_config_t = bxdf::SMicrofacetConfiguration<sample_t, aniso_interaction, aniso_cache, spectral_t>;

using bool32_t3 = hlsl::vector<bool, 3>;

template<typename T NBL_PRIMARY_REQUIRES(concepts::UnsignedIntegral<T>)
struct ConvertToFloat01
{
    using ret_t = conditional_t<vector_traits<T>::Dimension==1, float, hlsl::vector<float, vector_traits<T>::Dimension> >;

    static ret_t __call(T x)
    {
        return ret_t(x) / hlsl::promote<ret_t>(numeric_limits<uint32_t>::max);
    }
};

template<typename T>
bool checkLt(T a, T b)
{
    return nbl::hlsl::all<hlsl::vector<bool, vector_traits<T>::Dimension> >(a < b);
}

template<typename T>
bool checkZero(T a, float32_t eps)
{
    return nbl::hlsl::all<hlsl::vector<bool, vector_traits<T>::Dimension> >(nbl::hlsl::abs<T>(a) < hlsl::promote<T>(eps));
}

template<>
bool checkZero<float32_t>(float32_t a, float32_t eps)
{
    return nbl::hlsl::abs<float32_t>(a) < eps;
}

struct SBxDFTestResources
{
    static SBxDFTestResources create(uint32_t _halfseed)
    {
        random::PCG32 pcg = random::PCG32::construct(_halfseed);
        uint32_t2 seed = nbl::hlsl::random::DimAdaptorRecursive<nbl::hlsl::random::PCG32, 2>::__call(pcg);

        SBxDFTestResources retval;
        retval.rng = nbl::hlsl::Xoroshiro64Star::construct(seed);
        retval.u = ConvertToFloat01<uint32_t3>::__call(retval.rng_vec<3>());
        retval.u.x = hlsl::clamp(retval.u.x, retval.eps, 1.f-retval.eps);
        retval.u.y = hlsl::clamp(retval.u.y, retval.eps, 1.f-retval.eps);
        retval.u.z = hlsl::mix(0.0, 1.0, retval.u.z > 0.5);

        retval.V.direction = nbl::hlsl::normalize<float32_t3>(sampling::UniformSphere<float>::generate(ConvertToFloat01<uint32_t2>::__call(retval.rng_vec<2>())));
        retval.N = nbl::hlsl::normalize<float32_t3>(sampling::UniformSphere<float>::generate(ConvertToFloat01<uint32_t2>::__call(retval.rng_vec<2>())));
        
        float32_t3 tangent, bitangent;
        math::frisvad<float32_t3>(retval.N, tangent, bitangent);
        tangent = nbl::hlsl::normalize<float32_t3>(tangent);
        bitangent = nbl::hlsl::normalize<float32_t3>(bitangent);

        const float angle = 2.0f * numbers::pi<float> * ConvertToFloat01<uint32_t>::__call(retval.rng());
        math::quaternion<float> rot = math::quaternion<float>::create(retval.N, angle);
        retval.T = rot.transformVector(tangent);
        retval.B = rot.transformVector(bitangent);

        retval.alpha.x = hlsl::max(ConvertToFloat01<uint32_t>::__call(retval.rng()), 1e-4f);
        retval.alpha.y = hlsl::max(ConvertToFloat01<uint32_t>::__call(retval.rng()), 1e-4f);
        retval.eta = ConvertToFloat01<uint32_t3>::__call(retval.rng_vec<3>()) * hlsl::promote<float32_t3>(1.5) + hlsl::promote<float32_t3>(1.1); // range [1.1,2.6], also only do eta = eta/1.0 (air)
        retval.etak = ConvertToFloat01<uint32_t3>::__call(retval.rng_vec<3>()) * hlsl::promote<float32_t3>(1.5) + hlsl::promote<float32_t3>(1.1); // same as above
        retval.luma_coeff = colorspace::scRGBtoXYZ[1];

        retval.Dinc = ConvertToFloat01<uint32_t>::__call(retval.rng()) * 2400.0f + 100.0f;
        retval.etaThinFilm = ConvertToFloat01<uint32_t>::__call(retval.rng()) * 0.5 + 1.1f; // range [1.1,1.6]
        return retval;
    }

    template<uint32_t D>
    hlsl::vector<uint32_t,D> rng_vec()
    {
        // don't construct an adaptor, use a static call which takes base RNG by reference, so modifies its state while producing numbers
        return nbl::hlsl::random::DimAdaptorRecursive<nbl::hlsl::Xoroshiro64Star, D>::__call(rng);
    }

    float eps = 1e-3;   // epsilon
    uint32_t halfSeed;     // init state seed, expect each seed to be unique so threads don't clash writing to same filename

    nbl::hlsl::Xoroshiro64Star rng;
    ray_dir_info_t V;
    float32_t3 N;
    float32_t3 T;
    float32_t3 B;

    float32_t3 u;
    float32_t2 alpha;
    float32_t3 eta;
    float32_t3 etak;
    float32_t3 luma_coeff;

    // thin film stuff;
    float Dinc; // in nm [100, 2500]
    float etaThinFilm;
};

// refer to config to see which params are used in which test
struct STestInitParams
{
    bool logInfo;
    uint32_t halfSeed;  // state used to get vec2 seed from hash, default: iteration no.
    uint32_t samples;   // num samples generated for distribution tests, e.g. chi2, bucket, etc.
    uint32_t thetaSplits;
    uint32_t phiSplits;
    uint16_t writeFrequencies;
    bool immediateFail;
    bool verbose;
};

enum TestResult
{
    BTR_NOBREAK = 0,
    BTR_NONE = 1,
    BTR_PRINT_MSG = 2,

    BTR_ERROR_NEGATIVE_VAL = -1,        // pdf/quotient/eval < 0
    BTR_ERROR_GENERATED_SAMPLE_NAN_PDF = -2,    // pdf = 0
    BTR_ERROR_QUOTIENT_INF = -3,        // quotient -> inf
    BTR_ERROR_JACOBIAN_TEST_FAIL = -4,  // jacobian * pdf != 0
    BTR_ERROR_PDF_EVAL_DIFF = -5,       // quotient * pdf != eval
    BTR_ERROR_NO_RECIPROCITY = -6,      // eval(incoming) != eval(outgoing)
    BTR_ERROR_GENERATED_H_INVALID = -7, // generated H is invalid
    BTR_ERROR_REFLECTANCE_OUT_OF_RANGE = -8,    // reflectance not [0, 1]
    BTR_ERROR_QUOTIENT_SUM_TOO_LARGE = -9,  // accumulated quotient >= 1.0
    
    BTR_INVALID_TEST_CONFIG = 3         // returned when test values are outside of expected usage
};

struct TestBase
{
    void init(uint32_t halfSeed)
    {
        rc = SBxDFTestResources::create(halfSeed);

        isointer = iso_interaction::create(rc.V, rc.N);
        isointer.luminosityContributionHint = rc.luma_coeff;
        anisointer = aniso_interaction::create(isointer, rc.T, rc.B);
    }

    SBxDFTestResources rc;

    iso_interaction isointer;
    aniso_interaction anisointer;

#ifndef __HLSL_VERSION
    std::string name = "base";
    std::string errMsg = "";
#endif
};

template<class TestT>
struct FailureCallback
{
    virtual void __call(TestResult error, NBL_REF_ARG(TestT) failedFor, bool logInfo) {}
};

template<class BxDF>
struct TestBxDFBase : TestBase
{
    using bxdf_t = BxDF;
    BxDF bxdf;
};

template<class BxDF>
struct TestBxDF : TestBxDFBase<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;

    void initBxDF(SBxDFTestResources _rc)
    {
        // default to lambertian bxdf
#ifndef __HLSL_VERSION
        base_t::name = "Lambertian BxDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::reflection::SOrenNayar<iso_config_t>> : TestBxDFBase<bxdf::reflection::SOrenNayar<iso_config_t>>
{
    using base_t = TestBxDFBase<bxdf::reflection::SOrenNayar<iso_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf_t::creation_type params;
        params.A = _rc.alpha.x;
        base_t::bxdf = bxdf::reflection::SOrenNayar<iso_config_t>::create(params);
#ifndef __HLSL_VERSION
        base_t::name = "OrenNayar BRDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::reflection::SDeltaDistribution<iso_config_t>> : TestBxDFBase<bxdf::reflection::SDeltaDistribution<iso_config_t>>
{
    using base_t = TestBxDFBase<bxdf::reflection::SDeltaDistribution<iso_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
#ifndef __HLSL_VERSION
        base_t::name = "Delta Distribution BRDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>> : TestBxDFBase<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>>
{
    using base_t = TestBxDFBase<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(_rc.alpha.x);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(_rc.eta,_rc.etak);
#ifndef __HLSL_VERSION
        base_t::name = "Beckmann BRDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>> : TestBxDFBase<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>>
{
    using base_t = TestBxDFBase<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(_rc.alpha.x, _rc.alpha.y);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(_rc.eta,_rc.etak);
#ifndef __HLSL_VERSION
        base_t::name = "Beckmann Aniso BRDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>> : TestBxDFBase<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>>
{
    using base_t = TestBxDFBase<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(_rc.alpha.x);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(_rc.eta,_rc.etak);
#ifndef __HLSL_VERSION
        base_t::name = "GGX BRDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>> : TestBxDFBase<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>>
{
    using base_t = TestBxDFBase<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(_rc.alpha.x, _rc.alpha.y);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(_rc.eta,_rc.etak);
#ifndef __HLSL_VERSION
        base_t::name = "GGX Aniso BRDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::reflection::SIridescent<iso_microfacet_config_t>> : TestBxDFBase<bxdf::reflection::SIridescent<iso_microfacet_config_t>>
{
    using base_t = TestBxDFBase<bxdf::reflection::SIridescent<iso_microfacet_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(_rc.alpha.x);
        using creation_params_t = base_t::bxdf_t::fresnel_type::creation_params_type;
        creation_params_t params;
        params.Dinc = _rc.Dinc;
        params.ior1 = hlsl::promote<float32_t3>(1.0);
        params.ior2 = hlsl::promote<float32_t3>(_rc.etaThinFilm);
        params.ior3 = _rc.eta;
        params.iork3 = _rc.etak;
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(params);
#ifndef __HLSL_VERSION
        base_t::name = "Iridescent BRDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::transmission::SOrenNayar<iso_config_t>> : TestBxDFBase<bxdf::transmission::SOrenNayar<iso_config_t>>
{
   using base_t = TestBxDFBase<bxdf::transmission::SOrenNayar<iso_config_t>>;

   void initBxDF(SBxDFTestResources _rc)
   {
        base_t::bxdf_t::creation_type params;
        params.A = _rc.alpha.x;
        base_t::bxdf = bxdf::transmission::SOrenNayar<iso_config_t>::create(params);
#ifndef __HLSL_VERSION
        base_t::name = "OrenNayar BSDF";
#endif
   }
};

template<>
struct TestBxDF<bxdf::transmission::SSmoothDielectric<iso_config_t>> : TestBxDFBase<bxdf::transmission::SSmoothDielectric<iso_config_t>>
{
   using base_t = TestBxDFBase<bxdf::transmission::SSmoothDielectric<iso_config_t>>;

   void initBxDF(SBxDFTestResources _rc)
   {
        base_t::bxdf.orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(base_t::isointer.getNdotV(bxdf::BxDFClampMode::BCM_ABS), hlsl::promote<typename base_t::bxdf_t::monochrome_type>(_rc.eta.x));
#ifndef __HLSL_VERSION
        base_t::name = "Smooth dielectric BSDF";
#endif
   }
};

template<>
struct TestBxDF<bxdf::transmission::SThinSmoothDielectric<iso_config_t>> : TestBxDFBase<bxdf::transmission::SThinSmoothDielectric<iso_config_t>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SThinSmoothDielectric<iso_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        using spectral_type = typename base_t::bxdf_t::spectral_type;
        base_t::bxdf.fresnel = bxdf::fresnel::Dielectric<spectral_type>::create(bxdf::fresnel::OrientedEtas<spectral_type>::create(base_t::isointer.getNdotV(bxdf::BxDFClampMode::BCM_ABS), hlsl::promote<spectral_type>(_rc.eta.x)));
#ifndef __HLSL_VERSION
        base_t::name = "Thin smooth dielectric BSDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::transmission::SDeltaDistribution<iso_config_t>> : TestBxDFBase<bxdf::transmission::SDeltaDistribution<iso_config_t>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SDeltaDistribution<iso_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
#ifndef __HLSL_VERSION
        base_t::name = "Delta Distribution BSDF";
#endif
    }
};

template<class Config>
struct TestBxDF<bxdf::transmission::SBeckmannDielectricIsotropic<Config>> : TestBxDFBase<bxdf::transmission::SBeckmannDielectricIsotropic<Config>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SBeckmannDielectricIsotropic<Config>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(1.0, hlsl::promote<typename base_t::bxdf_t::monochrome_type>(_rc.eta.x));
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(_rc.alpha.x);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(orientedEta);
#ifndef __HLSL_VERSION
        base_t::name = "Beckmann Dielectric BSDF";
#endif
    }
};

template<class Config>
struct TestBxDF<bxdf::transmission::SBeckmannDielectricAnisotropic<Config>> : TestBxDFBase<bxdf::transmission::SBeckmannDielectricAnisotropic<Config>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SBeckmannDielectricAnisotropic<Config>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(1.0, hlsl::promote<typename base_t::bxdf_t::monochrome_type>(_rc.eta.x));
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(_rc.alpha.x, _rc.alpha.y);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(orientedEta);
#ifndef __HLSL_VERSION
        base_t::name = "Beckmann Dielectric Aniso BSDF";
#endif
    }
};

template<class Config>
struct TestBxDF<bxdf::transmission::SGGXDielectricIsotropic<Config>> : TestBxDFBase<bxdf::transmission::SGGXDielectricIsotropic<Config>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SGGXDielectricIsotropic<Config>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(1.0, hlsl::promote<typename base_t::bxdf_t::monochrome_type>(_rc.eta.x));
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(_rc.alpha.x);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(orientedEta);
#ifndef __HLSL_VERSION
        base_t::name = "GGX Dielectric BSDF";
#endif
    }
};

template<class Config>
struct TestBxDF<bxdf::transmission::SGGXDielectricAnisotropic<Config>> : TestBxDFBase<bxdf::transmission::SGGXDielectricAnisotropic<Config>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SGGXDielectricAnisotropic<Config>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(1.0, hlsl::promote<typename base_t::bxdf_t::monochrome_type>(_rc.eta.x));
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(_rc.alpha.x, _rc.alpha.y);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(orientedEta);
#ifndef __HLSL_VERSION
        base_t::name = "GGX Dielectric Aniso BSDF";
#endif
    }
};

template<class Config>
struct TestBxDF<bxdf::transmission::SIridescent<Config>> : TestBxDFBase<bxdf::transmission::SIridescent<Config>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SIridescent<Config>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(_rc.alpha.x);
        using creation_params_t = base_t::bxdf_t::fresnel_type::creation_params_type;
        creation_params_t params;
        params.Dinc = _rc.Dinc;
        params.ior1 = hlsl::promote<float32_t3>(1.0);
        params.ior2 = hlsl::promote<float32_t3>(_rc.etaThinFilm);
        params.ior3 = hlsl::promote<float32_t3>(_rc.eta.x);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(params);
#ifndef __HLSL_VERSION
        base_t::name = "Iridescent BSDF";
#endif
    }
};


namespace reciprocity_test_impl
{
template<class RayDirInfo, class Spectrum NBL_PRIMARY_REQUIRES(bxdf::ray_dir_info::Basic<RayDirInfo> && concepts::FloatingPointLikeVectorial<Spectrum>)
struct SIsotropic
{
    using this_t = SIsotropic<RayDirInfo, Spectrum>;
    using ray_dir_info_type = RayDirInfo;
    using scalar_type = typename RayDirInfo::scalar_type;
    using vector3_type = typename RayDirInfo::vector3_type;
    using spectral_type = Spectrum;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static this_t create(NBL_CONST_REF_ARG(RayDirInfo) normalizedV, const vector3_type normalizedN)
    {
        this_t retval;
        retval.V = normalizedV;
        retval.N = normalizedN;
        retval.NdotV = nbl::hlsl::dot<vector3_type>(retval.N, retval.V.getDirection());
        retval.NdotV2 = retval.NdotV * retval.NdotV;
        retval.luminosityContributionHint = hlsl::promote<spectral_type>(1.0);

        return retval;
    }

    template<typename I NBL_FUNC_REQUIRES(bxdf::surface_interactions::Isotropic<I>)
    static this_t copy(NBL_CONST_REF_ARG(I) other)
    {
        this_t retval;
        retval.V = other.getV();
        retval.N = other.getN();
        retval.NdotV = other.getNdotV();
        retval.NdotV2 = other.getNdotV2();
        retval.pathOrigin = bxdf::PathOrigin::PO_SENSOR;
        retval.luminosityContributionHint = other.luminosityContributionHint;
        return retval;
    }

    RayDirInfo getV() NBL_CONST_MEMBER_FUNC { return V; }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return N; }
    scalar_type getNdotV(bxdf::BxDFClampMode _clamp = bxdf::BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC
    {
        return bxdf::conditionalAbsOrMax<scalar_type>(NdotV, _clamp);
    }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return NdotV2; }

    bxdf::PathOrigin getPathOrigin() NBL_CONST_MEMBER_FUNC { return pathOrigin; }
    spectral_type getLuminosityContributionHint() NBL_CONST_MEMBER_FUNC { return luminosityContributionHint; }

    RayDirInfo V;
    vector3_type N;
    scalar_type NdotV;
    scalar_type NdotV2;
    bxdf::PathOrigin pathOrigin;
    spectral_type luminosityContributionHint;
};

template<class IsotropicInteraction NBL_PRIMARY_REQUIRES(bxdf::surface_interactions::Isotropic<IsotropicInteraction>)
struct SAnisotropic
{
    using this_t = SAnisotropic<IsotropicInteraction>;
    using isotropic_interaction_type = IsotropicInteraction;
    using ray_dir_info_type = typename isotropic_interaction_type::ray_dir_info_type;
    using scalar_type = typename ray_dir_info_type::scalar_type;
    using vector3_type = typename ray_dir_info_type::vector3_type;
    using matrix3x3_type = hlsl::matrix<scalar_type, 3, 3>;
    using spectral_type = typename isotropic_interaction_type::spectral_type;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static this_t create(
        NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic,
        const vector3_type normalizedT,
        const vector3_type normalizedB
    )
    {
        this_t retval;
        retval.isotropic = isotropic;

        retval.T = normalizedT;
        retval.B = normalizedB;

        retval.TdotV = nbl::hlsl::dot<vector3_type>(retval.isotropic.getV().getDirection(), retval.T);
        retval.BdotV = nbl::hlsl::dot<vector3_type>(retval.isotropic.getV().getDirection(), retval.B);

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic, const vector3_type normalizedT)
    {
        return create(isotropic, normalizedT, cross(isotropic.getN(), normalizedT));
    }
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic)
    {
        vector3_type T, B;
        math::frisvad<vector3_type>(isotropic.getN(), T, B);
        return create(isotropic, nbl::hlsl::normalize<vector3_type>(T), nbl::hlsl::normalize<vector3_type>(B));
    }

    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) normalizedV, const vector3_type normalizedN)
    {
        isotropic_interaction_type isotropic = isotropic_interaction_type::create(normalizedV, normalizedN);
        return create(isotropic);
    }

    template<typename I NBL_FUNC_REQUIRES(bxdf::surface_interactions::Anisotropic<I>)
    static this_t copy(NBL_CONST_REF_ARG(I) other)
    {
        this_t retval;
        retval.isotropic = isotropic_interaction_type::template copy<typename I::isotropic_interaction_type>(other.isotropic);
        retval.T = other.getT();
        retval.B = other.getB();
        retval.TdotV = other.getTdotV();
        retval.BdotV = other.getBdotV();
        return retval;
    }

    ray_dir_info_type getV() NBL_CONST_MEMBER_FUNC { return isotropic.getV(); }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return isotropic.getN(); }
    scalar_type getNdotV(bxdf::BxDFClampMode _clamp = bxdf::BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC { return isotropic.getNdotV(_clamp); }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return isotropic.getNdotV2(); }
    bxdf::PathOrigin getPathOrigin() NBL_CONST_MEMBER_FUNC { return isotropic.getPathOrigin(); }
    spectral_type getLuminosityContributionHint() NBL_CONST_MEMBER_FUNC { return isotropic.getLuminosityContributionHint(); }

    vector3_type getT() NBL_CONST_MEMBER_FUNC { return T; }
    vector3_type getB() NBL_CONST_MEMBER_FUNC { return B; }
    scalar_type getTdotV() NBL_CONST_MEMBER_FUNC { return TdotV; }
    scalar_type getTdotV2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getTdotV(); return t*t; }
    scalar_type getBdotV() NBL_CONST_MEMBER_FUNC { return BdotV; }
    scalar_type getBdotV2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getBdotV(); return t*t; }

    vector3_type getTangentSpaceV() NBL_CONST_MEMBER_FUNC { return vector3_type(TdotV, BdotV, isotropic.getNdotV()); }
    matrix3x3_type getToTangentSpace() NBL_CONST_MEMBER_FUNC { return matrix3x3_type(T, B, isotropic.getN()); }
    matrix3x3_type getFromTangentSpace() NBL_CONST_MEMBER_FUNC { return nbl::hlsl::transpose<matrix3x3_type>(matrix3x3_type(T, B, isotropic.getN())); }

    isotropic_interaction_type isotropic;
    vector3_type T;
    vector3_type B;
    scalar_type TdotV;
    scalar_type BdotV;
};


template<class LS, class Interaction, class MicrofacetCache, class Spectrum>
struct CustomIsoMicrofacetConfiguration;

template<class LS, class Interaction, class MicrofacetCache, class Spectrum>
NBL_BOOL_CONCEPT CustomMicrofacetConfigIso = bxdf::LightSample<LS> && bxdf::surface_interactions::Isotropic<Interaction> && !bxdf::surface_interactions::Anisotropic<Interaction> && bxdf::CreatableIsotropicMicrofacetCache<MicrofacetCache> && !bxdf::AnisotropicMicrofacetCache<MicrofacetCache> && concepts::FloatingPointLikeVectorial<Spectrum>;

template<class LS, class Interaction, class MicrofacetCache, class Spectrum>
NBL_PARTIAL_REQ_TOP(CustomMicrofacetConfigIso<LS, Interaction, MicrofacetCache, Spectrum>)
struct CustomIsoMicrofacetConfiguration<LS,Interaction,MicrofacetCache,Spectrum NBL_PARTIAL_REQ_BOT(CustomMicrofacetConfigIso<LS, Interaction, MicrofacetCache, Spectrum>) >
#undef MICROFACET_CONF_ISO
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = false;

    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = hlsl::vector<scalar_type, 2>;
    using vector3_type = hlsl::vector<scalar_type, 3>;
    using monochrome_type = hlsl::vector<scalar_type, 1>;
    using matrix3x3_type = hlsl::matrix<scalar_type,3,3>;
    using isotropic_interaction_type = Interaction;
    using anisotropic_interaction_type = reciprocity_test_impl::SAnisotropic<isotropic_interaction_type>;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = MicrofacetCache;
    using anisocache_type = bxdf::SAnisotropicMicrofacetCache<MicrofacetCache>;
};
}

using rectest_iso_interaction = reciprocity_test_impl::SIsotropic<ray_dir_info_t, spectral_t>;
using rectest_aniso_interaction = reciprocity_test_impl::SAnisotropic<rectest_iso_interaction>;
using rectest_iso_microfacet_config_t = reciprocity_test_impl::CustomIsoMicrofacetConfiguration<sample_t, rectest_iso_interaction, iso_cache, spectral_t>;
using rectest_aniso_microfacet_config_t = bxdf::SMicrofacetConfiguration<sample_t, rectest_aniso_interaction, aniso_cache, spectral_t>;

#endif
