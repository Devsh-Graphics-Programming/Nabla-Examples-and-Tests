#ifndef BXDFTESTS_TESTS_COMMON_HLSL
#define BXDFTESTS_TESTS_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"
#include "nbl/builtin/hlsl/random/pcg.hlsl"
#include "nbl/builtin/hlsl/random/dim_adaptor_recursive.hlsl"
#include "nbl/builtin/hlsl/sampling/uniform_spheres.hlsl"
#include "nbl/builtin/hlsl/math/linalg/transform.hlsl"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/builtin/hlsl/math/polar.hlsl"
#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"

#ifndef __HLSL_VERSION
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <format>
#include <functional>

#include "ImfRgbaFile.h"
#include "ImfArray.h"
#include "ImfHeader.h"

#include "ImfNamespace.h"
#include <iostream>

#include "nlohmann/json.hpp"

namespace IMF = Imf;
namespace IMATH = Imath;

using namespace IMF;
using namespace IMATH;
using json = nlohmann::json;
#endif

namespace nbl
{
namespace hlsl
{

using ray_dir_info_t = bxdf::ray_dir_info::SBasic<float>;
using iso_interaction = bxdf::surface_interactions::SIsotropic<ray_dir_info_t>;
using aniso_interaction = bxdf::surface_interactions::SAnisotropic<iso_interaction>;
using sample_t = bxdf::SLightSample<ray_dir_info_t>;
using iso_cache = bxdf::SIsotropicMicrofacetCache<float>;
using aniso_cache = bxdf::SAnisotropicMicrofacetCache<iso_cache>;
using quotient_pdf_t = sampling::quotient_and_pdf<float32_t3, float>;
using spectral_t = vector<float, 3>;

using iso_config_t = bxdf::SConfiguration<sample_t, iso_interaction, spectral_t>;
using aniso_config_t = bxdf::SConfiguration<sample_t, aniso_interaction, spectral_t>;
using iso_microfacet_config_t = bxdf::SMicrofacetConfiguration<sample_t, iso_interaction, iso_cache, spectral_t>;
using aniso_microfacet_config_t = bxdf::SMicrofacetConfiguration<sample_t, aniso_interaction, aniso_cache, spectral_t>;

using bool32_t3 = vector<bool, 3>;

template<typename T>
struct ConvertToFloat01
{
    using ret_t = conditional_t<vector_traits<T>::Dimension==1, float, vector<float, vector_traits<T>::Dimension> >;

    static ret_t __call(T x)
    {
        return ret_t(x) / hlsl::promote<ret_t>(numeric_limits<uint32_t>::max);
    }
};

template<typename T>
bool checkEq(T a, T b, float32_t eps)
{
    T _a = hlsl::abs(a);
    T _b = hlsl::abs(b);
    return nbl::hlsl::all<vector<bool, vector_traits<T>::Dimension> >(nbl::hlsl::max<T>(_a / _b, _b / _a) <= hlsl::promote<T>(1 + eps));
}

template<typename T>
bool checkLt(T a, T b)
{
    return nbl::hlsl::all<vector<bool, vector_traits<T>::Dimension> >(a < b);
}

template<typename T>
bool checkZero(T a, float32_t eps)
{
    return nbl::hlsl::all<vector<bool, vector_traits<T>::Dimension> >(nbl::hlsl::abs<T>(a) < hlsl::promote<T>(eps));
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
        nbl::hlsl::random::DimAdaptorRecursive<nbl::hlsl::Xoroshiro64Star, 2> rng_vec2 = nbl::hlsl::random::DimAdaptorRecursive<nbl::hlsl::Xoroshiro64Star, 2>::construct(retval.rng);
        nbl::hlsl::random::DimAdaptorRecursive<nbl::hlsl::Xoroshiro64Star, 3> rng_vec3 = nbl::hlsl::random::DimAdaptorRecursive<nbl::hlsl::Xoroshiro64Star, 3>::construct(retval.rng);
        retval.u = ConvertToFloat01<uint32_t3>::__call(rng_vec3());
        retval.u.x = hlsl::clamp(retval.u.x, retval.eps, 1.f-retval.eps);
        retval.u.y = hlsl::clamp(retval.u.y, retval.eps, 1.f-retval.eps);
        // retval.u.z = 0.0;

        retval.V.direction = nbl::hlsl::normalize<float32_t3>(sampling::UniformSphere<float>::generate(ConvertToFloat01<uint32_t2>::__call(rng_vec2())));
        retval.N = nbl::hlsl::normalize<float32_t3>(sampling::UniformSphere<float>::generate(ConvertToFloat01<uint32_t2>::__call(rng_vec2())));
        // if (hlsl::dot(retval.N, retval.V.direction) < 0)
        //     retval.V.direction = -retval.V.direction;
        
        float32_t3 tangent, bitangent;
        math::frisvad<float32_t3>(retval.N, tangent, bitangent);
        tangent = nbl::hlsl::normalize<float32_t3>(tangent);
        bitangent = nbl::hlsl::normalize<float32_t3>(bitangent);

        const float angle = 2.0f * numbers::pi<float> * ConvertToFloat01<uint32_t>::__call(retval.rng());
        float32_t4x4 rot = math::linalg::promote_affine<4, 4>(math::linalg::rotation_mat(angle, retval.N));
        retval.T = mul(rot, float32_t4(tangent,1)).xyz;
        retval.B = mul(rot, float32_t4(bitangent,1)).xyz;

        retval.alpha.x = ConvertToFloat01<uint32_t>::__call(retval.rng());
        retval.alpha.y = ConvertToFloat01<uint32_t>::__call(retval.rng());
        retval.eta = ConvertToFloat01<uint32_t2>::__call(rng_vec2()) * hlsl::promote<float32_t2>(1.5) + hlsl::promote<float32_t2>(1.1); // range [1.1,2.6], also only do eta = eta/1.0 (air)
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
    float32_t2 eta; // (eta, etak)
    float32_t3 luma_coeff;
};

struct STestInitParams
{
    bool logInfo;
    uint32_t state;
    uint32_t samples;
    uint32_t thetaSplits;
    uint32_t phiSplits;
    bool writeFrequencies;
    bool immediateFail;
    bool verbose;
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
    BET_GENERATE_H,

    BET_NOBREAK,    // not an error code, ones after this don't break
    BET_INVALID,
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

    virtual ErrorType compute() { return BET_NONE; }

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
    virtual void __call(ErrorType error, NBL_REF_ARG(TestBase) failedFor, bool logInfo) {}
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
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(rc.alpha.x);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(hlsl::promote<float32_t3>(rc.eta.x),hlsl::promote<float32_t3>(rc.eta.y));
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
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(rc.alpha.x, rc.alpha.y);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(hlsl::promote<float32_t3>(rc.eta.x),hlsl::promote<float32_t3>(rc.eta.y));
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
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(rc.alpha.x);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(hlsl::promote<float32_t3>(rc.eta.x),hlsl::promote<float32_t3>(rc.eta.y));
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
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(rc.alpha.x, rc.alpha.y);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(hlsl::promote<float32_t3>(rc.eta.x),hlsl::promote<float32_t3>(rc.eta.y));
#ifndef __HLSL_VERSION
        base_t::name = "GGX Aniso BRDF";
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
        base_t::bxdf.orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(base_t::isointer.getNdotV(bxdf::BxDFClampMode::BCM_ABS), hlsl::promote<typename base_t::bxdf_t::monochrome_type>(rc.eta.x));
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
        base_t::bxdf.fresnel = bxdf::fresnel::Dielectric<spectral_type>::create(bxdf::fresnel::OrientedEtas<spectral_type>::create(base_t::isointer.getNdotV(bxdf::BxDFClampMode::BCM_ABS), hlsl::promote<spectral_type>(rc.eta.x)));
        base_t::bxdf.luminosityContributionHint = rc.luma_coeff;
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

template<>
struct TestBxDF<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>> : TestBxDFBase<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(1.0, hlsl::promote<typename base_t::bxdf_t::monochrome_type>(rc.eta.x));
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(rc.alpha.x);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(orientedEta);
#ifndef __HLSL_VERSION
        base_t::name = "Beckmann Dielectric BSDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>> : TestBxDFBase<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(1.0, hlsl::promote<typename base_t::bxdf_t::monochrome_type>(rc.eta.x));
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(rc.alpha.x, rc.alpha.y);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(orientedEta);
#ifndef __HLSL_VERSION
        base_t::name = "Beckmann Dielectric Aniso BSDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>> : TestBxDFBase<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(1.0, hlsl::promote<typename base_t::bxdf_t::monochrome_type>(rc.eta.x));
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(rc.alpha.x);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(orientedEta);
#ifndef __HLSL_VERSION
        base_t::name = "GGX Dielectric BSDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>> : TestBxDFBase<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(1.0, hlsl::promote<typename base_t::bxdf_t::monochrome_type>(rc.eta.x));
        base_t::bxdf.ndf = base_t::bxdf_t::ndf_type::create(rc.alpha.x, rc.alpha.y);
        base_t::bxdf.fresnel = base_t::bxdf_t::fresnel_type::create(orientedEta);
#ifndef __HLSL_VERSION
        base_t::name = "GGX Dielectric Aniso BSDF";
#endif
    }
};

}
}

#endif
