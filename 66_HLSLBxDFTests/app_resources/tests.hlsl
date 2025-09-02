#ifndef BXDFTESTS_TESTS_HLSL
#define BXDFTESTS_TESTS_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"
#include "nbl/builtin/hlsl/random/pcg.hlsl"
#include "nbl/builtin/hlsl/random/dim_adaptor_recursive.hlsl"
#include "nbl/builtin/hlsl/sampling/uniform_spheres.hlsl"
#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"

#ifndef __HLSL_VERSION
#include <glm/gtc/quaternion.hpp>
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

// custom hash for float32_t2
template<>
struct std::hash<float32_t2>
{
    size_t operator()(const float32_t2& v) const noexcept
    {
        size_t v1 = std::hash<float32_t>()(v.x);
        size_t v2 = std::hash<float32_t>()(v.y);
        return v1 ^ (v2 << 1);
    }
};
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
    return nbl::hlsl::all<vector<bool, vector_traits<T>::Dimension> >(nbl::hlsl::max<T>(a / b, b / a) <= (T)(1 + eps));
}

template<typename T>
bool checkLt(T a, T b)
{
    return nbl::hlsl::all<vector<bool, vector_traits<T>::Dimension> >(a < b);
}

template<typename T>
bool checkZero(T a, float32_t eps)
{
    return nbl::hlsl::all<vector<bool, vector_traits<T>::Dimension> >(nbl::hlsl::abs<T>(a) < (T)eps);
}

template<>
bool checkZero<float32_t>(float32_t a, float32_t eps)
{
    return nbl::hlsl::abs<float32_t>(a) < eps;
}

#ifndef __HLSL_VERSION
// because atan2 is not in tgmath.hlsl yet

// takes in normalized vectors
inline float32_t3 polarToCartesian(float32_t2 theta_phi)
{
    return float32_t3(std::cos(theta_phi.y) * std::cos(theta_phi.x),
                        std::sin(theta_phi.y) * std::cos(theta_phi.x),
                        std::sin(theta_phi.x));
}

inline float32_t2 cartesianToPolar(float32_t3 coords)
{
    return float32_t2(std::acos(clamp<float>(coords.z, -1, 1)), std::atan2(coords.y, coords.x));
}
#endif

struct SBxDFTestResources
{
    static SBxDFTestResources create(uint32_t2 seed)
    {
        SBxDFTestResources retval;
        retval.rng = nbl::hlsl::Xoroshiro64Star::construct(seed);
        retval.u = float32_t3(rngUniformDist<float32_t2>(retval.rng), 0.0);

        retval.V.direction = nbl::hlsl::normalize<float32_t3>(sampling::UniformSphere<float>::generate(rngUniformDist<float32_t2>(retval.rng)));
        retval.N = nbl::hlsl::normalize<float32_t3>(sampling::UniformSphere<float>::generate(rngUniformDist<float32_t2>(retval.rng)));
        
        float32_t3 tangent, bitangent;
        math::frisvad<float32_t3>(retval.N, tangent, bitangent);
        tangent = nbl::hlsl::normalize<float32_t3>(tangent);
        bitangent = nbl::hlsl::normalize<float32_t3>(bitangent);
#ifndef __HLSL_VERSION
        const float angle = 2 * numbers::pi<float> * rngUniformDist<float>(retval.rng);
        glm::quat rot = glm::angleAxis(angle, retval.N);
        retval.T = rot * tangent;
        retval.B = rot * bitangent;
#else
        retval.T = tangent;
        retval.B = bitangent;
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

struct STestInitParams
{
    bool logInfo;
    uint32_t state;
    uint32_t samples;
    uint32_t thetaSplits;
    uint32_t phiSplits;
    bool writeFrequencies;
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
        base_t::bxdf = BxDF::create();  // default to lambertian bxdf
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
        base_t::bxdf = bxdf::reflection::SOrenNayar<iso_config_t>::create(_rc.alpha.x);
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
        base_t::bxdf = bxdf::reflection::SDeltaDistribution<iso_config_t>::create();
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
        base_t::bxdf = bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>::create(rc.alpha.x,hlsl::promote<float32_t3>(rc.ior.x),hlsl::promote<float32_t3>(rc.ior.y));
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
        base_t::bxdf = bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>::create(rc.alpha.x,rc.alpha.y,hlsl::promote<float32_t3>(rc.ior.x),hlsl::promote<float32_t3>(rc.ior.y));
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
        base_t::bxdf = bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>::create(rc.alpha.x,hlsl::promote<float32_t3>(rc.ior.x),hlsl::promote<float32_t3>(rc.ior.y));
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
        base_t::bxdf = bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>::create(rc.alpha.x,rc.alpha.y,hlsl::promote<float32_t3>(rc.ior.x),hlsl::promote<float32_t3>(rc.ior.y));
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
        base_t::bxdf = bxdf::transmission::SOrenNayar<iso_config_t>::create(_rc.alpha.x);
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
        base_t::bxdf.orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(base_t::isointer.getNdotV(bxdf::BxDFClampMode::BCM_ABS), hlsl::promote<typename base_t::bxdf_t::monochrome_type>(rc.eta));
#ifndef __HLSL_VERSION
        base_t::name = "Smooth dielectric BSDF";
#endif
    }
};

template<>
struct TestBxDF<bxdf::transmission::SSmoothThinDielectric<iso_config_t>> : TestBxDFBase<bxdf::transmission::SSmoothThinDielectric<iso_config_t>>
{
    using base_t = TestBxDFBase<bxdf::transmission::SSmoothThinDielectric<iso_config_t>>;

    void initBxDF(SBxDFTestResources _rc)
    {
        using spectral_type = typename base_t::bxdf_t::spectral_type;
        bxdf::fresnel::Dielectric<spectral_type> f = bxdf::fresnel::Dielectric<spectral_type>::create(bxdf::fresnel::OrientedEtas<spectral_type>::create(base_t::isointer.getNdotV(bxdf::BxDFClampMode::BCM_ABS), hlsl::promote<spectral_type>(rc.eta)));
        base_t::bxdf = bxdf::transmission::SSmoothThinDielectric<iso_config_t>::create(f,rc.luma_coeff);
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
        base_t::bxdf = bxdf::transmission::SDeltaDistribution<iso_config_t>::create();
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
        bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(base_t::isointer.getNdotV(), hlsl::promote<typename base_t::bxdf_t::monochrome_type>(rc.eta));
        base_t::bxdf = bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>::create(orientedEta,rc.alpha.x);
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
        bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(base_t::anisointer.getNdotV(), hlsl::promote<typename base_t::bxdf_t::monochrome_type>(rc.eta));
        base_t::bxdf = bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>::create(orientedEta,rc.alpha.x,rc.alpha.y);
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
            bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(base_t::isointer.getNdotV(), hlsl::promote<typename base_t::bxdf_t::monochrome_type>(rc.eta));
        base_t::bxdf = bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>::create(orientedEta,rc.alpha.x);
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
        bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename base_t::bxdf_t::monochrome_type>::create(base_t::anisointer.getNdotV(), hlsl::promote<typename base_t::bxdf_t::monochrome_type>(rc.eta));
        base_t::bxdf = bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>::create(orientedEta,rc.alpha.x,rc.alpha.y);
#ifndef __HLSL_VERSION
        base_t::name = "GGX Dielectric Aniso BSDF";
#endif
    }
};


template<class T>
struct is_basic_brdf : bool_constant<
    is_same<T, bxdf::reflection::SLambertian<iso_config_t> >::value ||
    is_same<T, bxdf::reflection::SLambertian<aniso_config_t> >::value ||
    is_same<T, bxdf::reflection::SOrenNayar<iso_config_t>>::value ||
    is_same<T, bxdf::reflection::SOrenNayar<aniso_config_t>>::value ||
    is_same<T, bxdf::reflection::SDeltaDistribution<iso_config_t> >::value ||
    is_same<T, bxdf::reflection::SDeltaDistribution<aniso_config_t> >::value
> {};

template<class T>
struct is_microfacet_brdf : bool_constant<
    is_same<T, bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>>::value ||
    is_same<T, bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>>::value ||
    is_same<T, bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>>::value ||
    is_same<T, bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>>::value
> {};

template<class T>
struct is_basic_bsdf : bool_constant<
    is_same<T, bxdf::transmission::SLambertian<iso_config_t>>::value ||
    is_same<T, bxdf::transmission::SLambertian<aniso_config_t>>::value ||
    is_same<T, bxdf::transmission::SSmoothDielectric<iso_config_t>>::value ||
    is_same<T, bxdf::transmission::SSmoothDielectric<aniso_config_t>>::value ||
    is_same<T, bxdf::transmission::SSmoothThinDielectric<iso_config_t>>::value ||
    is_same<T, bxdf::transmission::SSmoothThinDielectric<aniso_config_t>>::value ||
    is_same<T, bxdf::transmission::SDeltaDistribution<iso_config_t>>::value ||
    is_same<T, bxdf::transmission::SDeltaDistribution<aniso_config_t>>::value
> {};

template<class T>
struct is_microfacet_bsdf : bool_constant<
    is_same<T, bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>>::value ||
    is_same<T, bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>>::value ||
    is_same<T, bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>>::value ||
    is_same<T, bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>>::value
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

    virtual ErrorType compute() override
    {
        aniso_cache cache, dummy;
        iso_cache isocache, dummy_iso;

        float32_t3 ux = base_t::rc.u + float32_t3(base_t::rc.eps,0,0);
        float32_t3 uy = base_t::rc.u + float32_t3(0,base_t::rc.eps,0);

        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::isointer, base_t::rc.u.xy);
            sx = base_t::bxdf.generate(base_t::isointer, ux.xy);
            sy = base_t::bxdf.generate(base_t::isointer, uy.xy);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy, cache);
                sx = base_t::bxdf.generate(base_t::anisointer, ux.xy, dummy);
                sy = base_t::bxdf.generate(base_t::anisointer, uy.xy, dummy);
            }
            else
            {
                s = base_t::bxdf.generate(base_t::isointer, base_t::rc.u.xy, isocache);
                sx = base_t::bxdf.generate(base_t::isointer, ux.xy, dummy_iso);
                sy = base_t::bxdf.generate(base_t::isointer, uy.xy, dummy_iso);
            }
        }
        if NBL_CONSTEXPR_FUNC (is_basic_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u);
            sx = base_t::bxdf.generate(base_t::anisointer, ux);
            sy = base_t::bxdf.generate(base_t::anisointer, uy);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_bsdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u, cache);
                sx = base_t::bxdf.generate(base_t::anisointer, ux, dummy);
                sy = base_t::bxdf.generate(base_t::anisointer, uy, dummy);
            }
            else
            {
                s = base_t::bxdf.generate(base_t::isointer, base_t::rc.u, isocache);
                sx = base_t::bxdf.generate(base_t::isointer, ux, dummy_iso);
                sy = base_t::bxdf.generate(base_t::isointer, uy, dummy_iso);
            }
        }

        // TODO: add checks with need clamp trait
        if (bxdf::traits<BxDF>::type == bxdf::BT_BRDF)
        {
            if (s.getNdotL() <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }
        else if (bxdf::traits<BxDF>::type == bxdf::BT_BSDF)
        {
            if (abs<float>(s.getNdotL()) <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
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
                typename BxDF::query_type query = base_t::bxdf.createQuery(s, base_t::anisointer);
                pdf = base_t::bxdf.quotient_and_pdf(query, s, base_t::anisointer, cache);
                bsdf = float32_t3(base_t::bxdf.eval(query, s, base_t::anisointer, cache));
            }
            else
            {
                typename BxDF::query_type query = base_t::bxdf.createQuery(s, base_t::isointer);
                pdf = base_t::bxdf.quotient_and_pdf(query, s, base_t::isointer, isocache);
                bsdf = float32_t3(base_t::bxdf.eval(query, s, base_t::isointer, isocache));
            }
        }

        return BET_NONE;
    }

    ErrorType test()
    {
        if (bxdf::traits<BxDF>::type == bxdf::BT_BRDF)
        {    
            if (base_t::isointer.getNdotV() <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }        
        else if (bxdf::traits<BxDF>::type == bxdf::BT_BSDF)
        {
            if (abs<float>(base_t::isointer.getNdotV()) <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }

        ErrorType res = compute();
        if (res != BET_NONE)
            return res;

        if (checkZero<float>(pdf.pdf, 1e-5))  // something generated cannot have 0 probability of getting generated
            return BET_PDF_ZERO;

        if (!checkLt<float32_t3>(pdf.quotient, (float32_t3)bit_cast<float, uint32_t>(numeric_limits<float>::infinity)))    // importance sampler's job to prevent inf
            return BET_QUOTIENT_INF;

        if (checkZero<float32_t3>(bsdf, 1e-5) || checkZero<float32_t3>(pdf.quotient, 1e-5))
            return BET_NONE;    // produces an "impossible" sample

        if (checkLt<float32_t3>(bsdf, (float32_t3)0.0) || checkLt<float32_t3>(pdf.quotient, (float32_t3)0.0) || pdf.pdf < 0.0)
            return BET_NEGATIVE_VAL;

        // get BET_jacobian
        float32_t2x2 m = float32_t2x2(sx.getTdotL() - s.getTdotL(), sy.getTdotL() - s.getTdotL(), sx.getBdotL() - s.getBdotL(), sy.getBdotL() - s.getBdotL());
        float det = nbl::hlsl::determinant<float32_t2x2>(m);

        if (!checkZero<float>(det * pdf.pdf / s.getNdotL(), 1e-5))
            return BET_JACOBIAN;

        if (!checkEq<float32_t3>(pdf.value(), bsdf, 5e-2))
            return BET_PDF_EVAL_DIFF;

        return BET_NONE;
    }

    static void run(NBL_CONST_REF_ARG(STestInitParams) initparams, NBL_REF_ARG(FailureCallback) cb)
    {
        random::PCG32 pcg = random::PCG32::construct(initparams.state);
        random::DimAdaptorRecursive<random::PCG32, 2> rand2d = random::DimAdaptorRecursive<random::PCG32, 2>::construct(pcg);
        uint32_t2 state = rand2d();

        this_t t;
        t.init(state);
        t.rc.state = initparams.state;
        t.initBxDF(t.rc);
        
        ErrorType e = t.test();
        if (e != BET_NONE)
            cb.__call(e, t, initparams.logInfo);
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

    virtual ErrorType compute() override
    {
        aniso_cache cache, rec_cache;
        iso_cache isocache, rec_isocache;

        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u.xy, cache);
            }
            else
            {
                s = base_t::bxdf.generate(base_t::isointer, base_t::rc.u.xy, isocache);
            }
        }
        if NBL_CONSTEXPR_FUNC (is_basic_bsdf_v<BxDF>)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u);
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_bsdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u, cache);
            }
            else
            {
                s = base_t::bxdf.generate(base_t::isointer, base_t::rc.u, isocache);
            }
        }

        // TODO: add checks with need clamp trait
        if (bxdf::traits<BxDF>::type == bxdf::BT_BRDF)
        {
            if (s.getNdotL() <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }
        else if (bxdf::traits<BxDF>::type == bxdf::BT_BSDF)
        {
            if (abs<float>(s.getNdotL()) <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }

        float32_t3x3 toTangentSpace = base_t::anisointer.getToTangentSpace();
        ray_dir_info_t rec_V = s.getL();
        ray_dir_info_t rec_localV = rec_V.transform(toTangentSpace);
        ray_dir_info_t rec_localL = base_t::rc.V.transform(toTangentSpace);
        rec_s = sample_t::createFromTangentSpace(rec_localL, base_t::anisointer.getFromTangentSpace());

        rec_isointer = iso_interaction::create(rec_V, base_t::rc.N);
        rec_anisointer = aniso_interaction::create(rec_isointer, base_t::rc.T, base_t::rc.B);
        rec_cache = cache;
        rec_cache.iso_cache.VdotH = cache.iso_cache.getLdotH();
        rec_cache.iso_cache.LdotH = cache.iso_cache.getVdotH();
        rec_isocache = isocache;
        rec_isocache.VdotH = isocache.getLdotH();
        rec_isocache.LdotH = isocache.getVdotH();
        
        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF> || is_basic_bsdf_v<BxDF>)
        {
            bsdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer));
            rec_bsdf = float32_t3(base_t::bxdf.eval(rec_s, rec_isointer));
        }
        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                typename BxDF::query_type query = base_t::bxdf.createQuery(s, base_t::anisointer);
                bsdf = float32_t3(base_t::bxdf.eval(query, s, base_t::anisointer, cache));
                query = base_t::bxdf.createQuery(rec_s, rec_anisointer);
                rec_bsdf = float32_t3(base_t::bxdf.eval(query, rec_s, rec_anisointer, rec_cache));
            }
            else
            {
                typename BxDF::query_type query = base_t::bxdf.createQuery(s, base_t::isointer);
                bsdf = float32_t3(base_t::bxdf.eval(query, s, base_t::isointer, isocache));
                query = base_t::bxdf.createQuery(rec_s, rec_isointer);
                rec_bsdf = float32_t3(base_t::bxdf.eval(query, rec_s, rec_isointer, rec_isocache));
            }
        }

        return BET_NONE;
    }

    ErrorType test()
    {
        if (bxdf::traits<BxDF>::type == bxdf::BT_BRDF)
        {    
            if (base_t::isointer.getNdotV() <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }        
        else if (bxdf::traits<BxDF>::type == bxdf::BT_BSDF)
        {
            if (abs<float>(base_t::isointer.getNdotV()) <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }

        ErrorType res = compute();
        if (res != BET_NONE)
            return res;

        if (checkZero<float32_t3>(bsdf, 1e-5))
            return BET_NONE;    // produces an "impossible" sample

        if (checkLt<float32_t3>(bsdf, (float32_t3)0.0))
            return BET_NEGATIVE_VAL;

        float32_t3 a = bsdf * nbl::hlsl::abs<float>(base_t::isointer.getNdotV());
        float32_t3 b = rec_bsdf * nbl::hlsl::abs<float>(rec_isointer.getNdotV());
        if (!(a == b))  // avoid division by 0
            if (!checkEq<float32_t3>(a, b, 1e-2))
                return BET_RECIPROCITY;

        return BET_NONE;
    }

    static void run(NBL_CONST_REF_ARG(STestInitParams) initparams, NBL_REF_ARG(FailureCallback) cb)
    {
        random::PCG32 pcg = random::PCG32::construct(initparams.state);
        random::DimAdaptorRecursive<random::PCG32, 2> rand2d = random::DimAdaptorRecursive<random::PCG32, 2>::construct(pcg);
        uint32_t2 state = rand2d();

        this_t t;
        t.init(state);
        t.rc.state = initparams.state;
        t.initBxDF(t.rc);
        
        ErrorType e = t.test();
        if (e != BET_NONE)
            cb.__call(e, t, initparams.logInfo);
    }

    sample_t s, rec_s;
    float32_t3 bsdf, rec_bsdf;
    iso_interaction rec_isointer;
    aniso_interaction rec_anisointer;
    // params_t params, rec_params;
};

#ifndef __HLSL_VERSION  // because unordered_map
template<class BxDF, bool aniso = false>
struct TestBucket : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestBucket<BxDF, aniso>;

    void clearBuckets()
    {
        for (float y = -1.0f; y < 1.0f; y += stride)
        {
            for (float x = -1.0f; x < 1.0f; x += stride)
            {
                buckets[float32_t2(x, y)] = 0;
            }
        }
    }

    float bin(float a)
    {
        float diff = std::fmod(a, stride);
        float b = (a < 0) ? -stride : 0.0f;
        return a - diff + b;
    }

    virtual ErrorType compute() override
    {
        clearBuckets();

        aniso_cache cache;
        iso_cache isocache;

        sample_t s;
        quotient_pdf_t pdf;
        float32_t3 bsdf;

        for (uint32_t i = 0; i < numSamples; i++)
        {
            float32_t3 u = float32_t3(rngUniformDist<float32_t2>(base_t::rc.rng), 0.0);

            if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF>)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u.xy);
            }
            if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF>)
            {
                if NBL_CONSTEXPR_FUNC (aniso)
                {
                    s = base_t::bxdf.generate(base_t::anisointer, u.xy, cache);
                }
                else
                {
                    s = base_t::bxdf.generate(base_t::isointer, u.xy, isocache);
                }
            }
            if NBL_CONSTEXPR_FUNC (is_basic_bsdf_v<BxDF>)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u);
            }
            if NBL_CONSTEXPR_FUNC (is_microfacet_bsdf_v<BxDF>)
            {
                if NBL_CONSTEXPR_FUNC (aniso)
                {
                    s = base_t::bxdf.generate(base_t::anisointer, u, cache);
                }
                else
                {
                    s = base_t::bxdf.generate(base_t::isointer, u, isocache);
                }
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
                    typename BxDF::query_type query = base_t::bxdf.createQuery(s, base_t::anisointer);
                    pdf = base_t::bxdf.quotient_and_pdf(query, s, base_t::anisointer, cache);
                    bsdf = float32_t3(base_t::bxdf.eval(query, s, base_t::anisointer, cache));
                }
                else
                {
                    typename BxDF::query_type query = base_t::bxdf.createQuery(s, base_t::isointer);
                    pdf = base_t::bxdf.quotient_and_pdf(query, s, base_t::isointer, isocache);
                    bsdf = float32_t3(base_t::bxdf.eval(query, s, base_t::isointer, isocache));
                }
            }

            // put s into bucket
            float32_t3x3 toTangentSpace = base_t::anisointer.getToTangentSpace();
            const ray_dir_info_t localL = s.getL().transform(toTangentSpace);
            const float32_t2 coords = cartesianToPolar(localL.direction);
            float32_t2 bucket = float32_t2(bin(coords.x * numbers::inv_pi<float>), bin(coords.y * 0.5f * numbers::inv_pi<float>));

            if (pdf.pdf == bit_cast<float>(numeric_limits<float>::infinity))
                buckets[bucket] += 1;
        }

#ifndef __HLSL_VERSION
        // double check this conversion makes sense
        for (auto const& b : buckets) {
            if (!selective || b.second > 0)
            {
                const float32_t3 v = polarToCartesian(b.first * float32_t2(1, 2) * numbers::pi<float>);
                base_t::errMsg += std::format("({:.3f},{:.3f},{:.3f}): {}\n", v.x, v.y, v.z, b.second);
            }
        }
#endif
        return BET_NONE;
    }

    ErrorType test()
    {
        if (bxdf::traits<BxDF>::type == bxdf::BT_BRDF)
        {    
            if (base_t::isointer.getNdotV() <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }        
        else if (bxdf::traits<BxDF>::type == bxdf::BT_BSDF)
        {
            if (abs<float>(base_t::isointer.getNdotV()) <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }

        ErrorType res = compute();
        if (res != BET_NONE)
            return res;

        return (base_t::errMsg.length() == 0) ? BET_NONE : BET_PRINT_MSG;
    }

    static void run(NBL_CONST_REF_ARG(STestInitParams) initparams, NBL_REF_ARG(FailureCallback) cb)
    {
        random::PCG32 pcg = random::PCG32::construct(initparams.state);
        random::DimAdaptorRecursive<random::PCG32, 2> rand2d = random::DimAdaptorRecursive<random::PCG32, 2>::construct(pcg);
        uint32_t2 state = rand2d();

        this_t t;
        t.init(state);
        t.rc.state = initparams.state;
        t.numSamples = initparams.samples;
        t.initBxDF(t.rc);
        
        ErrorType e = t.test();
        if (e != BET_NONE)
            cb.__call(e, t, initparams.logInfo);
    }

    bool selective = true;  // print only buckets with count > 0
    float stride = 0.2f;
    uint32_t numSamples = 500;
    std::unordered_map<float32_t2, uint32_t, std::hash<float32_t2>> buckets;
};

inline float adaptiveSimpson(const std::function<float(float)>& f, float x0, float x1, float eps = 1e-6, int depth = 6)
{
    int count = 0;
    std::function<float(float, float, float, float, float, float, float, float, int)> integrate = 
    [&](float a, float b, float c, float fa, float fb, float fc, float I, float eps, int depth)
    {
        float d = 0.5f * (a + b);
        float e = 0.5f * (b + c);
        float fd = f(d);
        float fe = f(e);

        float h = c - a;
        float I0 = (1.0f / 12.0f) * h * (fa + 4 * fd + fb);
        float I1 = (1.0f / 12.0f) * h * (fb + 4 * fe + fc);
        float Ip = I0 + I1;
        count++;

        if (depth <= 0 || std::abs(Ip - I) < 15 * eps)
            return Ip + (1.0f / 15.0f) * (Ip - I);

        return integrate(a, d, b, fa, fd, fb, I0, .5f * eps, depth - 1) +
                integrate(b, e, c, fb, fe, fc, I1, .5f * eps, depth - 1);
    };
    
    float a = x0;
    float b = 0.5f * (x0 + x1);
    float c = x1;
    float fa = f(a);
    float fb = f(b);
    float fc = f(c);
    float I = (c - a) * (1.0f / 6.0f) * (fa + 4.f * fb + fc);
    return integrate(a, b, c, fa, fb, fc, I, eps, depth);
}

inline float adaptiveSimpson2D(const std::function<float(float, float)>& f, float32_t2 x0, float32_t2 x1, float eps = 1e-6, int depth = 6)
{
    const auto integrate = [&](float y) -> float
    {
        return adaptiveSimpson(std::bind(f, std::placeholders::_1, y), x0.x, x1.x, eps, depth);
    };
    return adaptiveSimpson(integrate, x0.y, x1.y, eps, depth);
}

// adapted from pbrt chi2 test: https://github.com/mmp/pbrt-v4/blob/792aaaa08d97dbedf11a3bb23e246b6443d847b4/src/pbrt/bsdfs_test.cpp#L280
template<class BxDF, bool aniso = false>
struct TestChi2 : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestChi2<BxDF, aniso>;

    void clearBuckets()
    {
        const uint32_t freqSize = thetaSplits * phiSplits;
        countFreq.resize(freqSize);
        std::fill(countFreq.begin(), countFreq.end(), 0);
        integrateFreq.resize(freqSize);
        std::fill(integrateFreq.begin(), integrateFreq.end(), 0);
    }

    double RLGamma(double a, double x) {
        const double epsilon = 0.000000000000001;
        const double big = 4503599627370496.0;
        const double bigInv = 2.22044604925031308085e-16;
        assert(a >= 0 && x >= 0);

        if (x == 0)
            return 0.0f;

        double ax = (a * std::log(x)) - x - std::lgamma(a);
        if (ax < -709.78271289338399)
            return a < x ? 1.0 : 0.0;

        if (x <= 1 || x <= a)
        {
            double r2 = a;
            double c2 = 1;
            double ans2 = 1;

            do {
                r2 = r2 + 1;
                c2 = c2 * x / r2;
                ans2 += c2;
            } while ((c2 / ans2) > epsilon);

            return std::exp(ax) * ans2 / a;
        }

        int c = 0;
        double y = 1 - a;
        double z = x + y + 1;
        double p3 = 1;
        double q3 = x;
        double p2 = x + 1;
        double q2 = z * x;
        double ans = p2 / q2;
        double error;

        do {
            c++;
            y += 1;
            z += 2;
            double yc = y * c;
            double p = (p2 * z) - (p3 * yc);
            double q = (q2 * z) - (q3 * yc);

            if (q != 0)
            {
                double nextans = p / q;
                error = std::abs((ans - nextans) / nextans);
                ans = nextans;
            }
            else
            {
                error = 1;
            }

            p3 = p2;
            p2 = p;
            q3 = q2;
            q2 = q;

            if (std::abs(p) > big)
            {
                p3 *= bigInv;
                p2 *= bigInv;
                q3 *= bigInv;
                q2 *= bigInv;
            }
        } while (error > epsilon);

        return 1.0 - (std::exp(ax) * ans);
    }

    double chi2CDF(double x, int dof)
    {
        if (dof < 1 || x < 0)
        {
            return 0.0;
        }
        else if (dof == 2)
        {
            return 1.0 - std::exp(-0.5 * x);
        }
        else
        {
            return RLGamma(0.5 * dof, 0.5 * x);
        }
    }

    Imf::Rgba mapColor(float v, float vmin, float vmax)
    {
        Imf::Rgba c(1, 1, 1);
        float diff = vmax - vmin;
        v = clamp<float>(v, vmin, vmax);

        if (v < (vmin + 0.25f * diff))
        {
            c.r = 0;
            c.g = 4.f * (v - vmin) / diff;
        }
        else if (v < (vmin + 0.5f * diff))
        {
            c.r = 0;
            c.b = 1.f + 4.f * (vmin + 0.25f * diff - v) / diff;
        }
        else if (v < (vmin + 0.75f * diff))
        {
            c.r = 4.f * (v - vmin - 0.5f * diff) / diff;
            c.b = 0;
        }
        else
        {
            c.g = 1.f + 4.f * (vmin + 0.75f * diff - v) / diff;
            c.b = 0;
        }

        return c;
    }

    void writeToEXR()
    {
        std::string filename = std::format("chi2test_{}_{}.exr", base_t::rc.state, base_t::name);

        int totalWidth = phiSplits;
        int totalHeight = 2 * thetaSplits + 1;
        float maxFreq = max<float>(maxCountFreq, maxIntFreq);
        
        Array2D<Rgba> pixels(totalWidth, totalHeight);
        for (int y = 0; y < thetaSplits; y++)
            for (int x = 0; x < phiSplits; x++)
                pixels[y][x] = mapColor(countFreq[y * phiSplits + x], 0.f, maxFreq);

        // for (int x = 0; x < phiSplits; x++)
        //     pixels[thetaSplits][x] = Rgba(1, 1, 1);

        for (int y = 0; y < thetaSplits; y++)
            for (int x = 0; x < phiSplits; x++)
                pixels[thetaSplits + y][x] = mapColor(integrateFreq[y * phiSplits + x], 0.f, maxFreq);
    
        Header header(totalWidth, totalHeight);
        RgbaOutputFile file(filename.c_str(), header, WRITE_RGBA);
        file.setFrameBuffer(&pixels[0][0], 1, totalWidth+1);
        file.writePixels(totalHeight);
    }

    virtual ErrorType compute() override
    {
        clearBuckets();

        float thetaFactor = thetaSplits * numbers::inv_pi<float>;
        float phiFactor = phiSplits * 0.5f * numbers::inv_pi<float>;

        sample_t s;
        iso_cache isocache;
        aniso_cache cache;
        for (uint32_t i = 0; i < numSamples; i++)
        {
            float32_t3 u = float32_t3(rngUniformDist<float32_t2>(base_t::rc.rng), 0.0);

            if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF>)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u.xy);
            }
            if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF>)
            {
                if NBL_CONSTEXPR_FUNC(aniso)
                    s = base_t::bxdf.generate(base_t::anisointer, u.xy, cache);
                else
                    s = base_t::bxdf.generate(base_t::isointer, u.xy, isocache);
            }
            if NBL_CONSTEXPR_FUNC (is_basic_bsdf_v<BxDF>)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u);
            }
            if NBL_CONSTEXPR_FUNC (is_microfacet_bsdf_v<BxDF>)
            {
                if NBL_CONSTEXPR_FUNC(aniso)
                    s = base_t::bxdf.generate(base_t::anisointer, u, cache);
                else
                    s = base_t::bxdf.generate(base_t::isointer, u, isocache);
            }

            // put s into bucket
            float32_t2 coords = cartesianToPolar(s.getL().getDirection()) * float32_t2(thetaFactor, phiFactor);
            if (coords.y < 0)
                coords.y += 2.f * numbers::pi<float> * phiFactor;

            int thetaBin = clamp<int>((int)std::floor(coords.x), 0, thetaSplits - 1);
            int phiBin = clamp<int>((int)std::floor(coords.y), 0, phiSplits - 1);

            uint32_t freqidx = thetaBin * phiSplits + phiBin;
            countFreq[freqidx] += 1;

            if (write_frequencies && maxCountFreq < countFreq[freqidx])
                maxCountFreq = countFreq[freqidx];
        }

        thetaFactor = 1.f / thetaFactor;
        phiFactor = 1.f / phiFactor;

        uint32_t intidx = 0;
        for (int i = 0; i < thetaSplits; i++)
        {
            for (int j = 0; j < phiSplits; j++)
            {
                uint32_t lastidx = intidx;
                integrateFreq[intidx++] = numSamples * adaptiveSimpson2D([&](float theta, float phi) -> float
                    {
                        float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
                        float cosPhi = std::cos(phi), sinPhi = std::sin(phi);

                        float32_t3x3 toTangentSpace = base_t::anisointer.getToTangentSpace();
                        ray_dir_info_t localV = base_t::rc.V.transform(toTangentSpace);
                        ray_dir_info_t L;
                        L.direction = float32_t3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
                        ray_dir_info_t localL = L.transform(toTangentSpace);
                        sample_t s = sample_t::createFromTangentSpace(localL, base_t::anisointer.getFromTangentSpace());

                        quotient_pdf_t pdf;
                        if NBL_CONSTEXPR_FUNC (is_basic_brdf_v<BxDF> || is_basic_bsdf_v<BxDF>)
                        {
                            pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer);
                        }
                        if NBL_CONSTEXPR_FUNC (is_microfacet_brdf_v<BxDF> || is_microfacet_bsdf_v<BxDF>)
                        {
                            if NBL_CONSTEXPR_FUNC (aniso)
                            {
                                aniso_cache cache = aniso_cache::template createForReflection<aniso_interaction,sample_t>(base_t::anisointer, s);
                                typename BxDF::query_type query = base_t::bxdf.createQuery(s, base_t::anisointer);
                                pdf = base_t::bxdf.quotient_and_pdf(query, s, base_t::anisointer, cache);
                            }
                            else
                            {
                                aniso_cache cache = aniso_cache::template createForReflection<aniso_interaction,sample_t>(base_t::anisointer, s);
                                typename BxDF::query_type query = base_t::bxdf.createQuery(s, base_t::isointer);
                                pdf = base_t::bxdf.quotient_and_pdf(query, s, base_t::isointer, cache.iso_cache);
                            }
                        }
                        return pdf.pdf == bit_cast<float>(numeric_limits<float>::infinity) ? 0.0 : pdf.pdf * sinTheta;
                    },
                    float32_t2(i * thetaFactor, j * phiFactor), float32_t2((i + 1) * thetaFactor, (j + 1) * phiFactor));

                if (write_frequencies && maxIntFreq < integrateFreq[lastidx])
                    maxIntFreq = integrateFreq[lastidx];
            }
        }

        return BET_NONE;
    }

    ErrorType test()
    {
        if (bxdf::traits<BxDF>::type == bxdf::BT_BRDF)
            if (base_t::isointer.getNdotV() <= numeric_limits<float>::min)
                return BET_INVALID;
        else if (bxdf::traits<BxDF>::type == bxdf::BT_BSDF)
            if (abs<float>(base_t::isointer.getNdotV()) <= numeric_limits<float>::min)
                return BET_INVALID;

        ErrorType res = compute();
        if (res != BET_NONE)
            return res;

        if (write_frequencies)
            writeToEXR();

        // chi2
        std::vector<Cell> cells(thetaSplits * phiSplits);
        for (uint32_t i = 0; i < cells.size(); i++)
        {
            cells[i].expFreq = integrateFreq[i];
            cells[i].index = i;
        }
        std::sort(cells.begin(), cells.end(), [](const Cell& a, const Cell& b)
        {
            return a.expFreq < b.expFreq;
        });

        float pooledFreqs = 0, pooledExpFreqs = 0, chsq = 0;
        int pooledCells = 0, dof = 0;

        for (const Cell& c : cells)
        {
            if (integrateFreq[c.index] == 0)
            {
                if (countFreq[c.index] > numSamples * 1e-5)
                {
                    base_t::errMsg = std::format("expected frequency of 0 for c but found {} samples", countFreq[c.index]);
                    return BET_PRINT_MSG;
                }
            }
            else if (integrateFreq[c.index] < minFreq)
            {
                pooledFreqs += countFreq[c.index];
                pooledExpFreqs += integrateFreq[c.index];
                pooledCells++;
            }
            else if (pooledExpFreqs > 0 && pooledExpFreqs < minFreq)
            {
                pooledFreqs += countFreq[c.index];
                pooledExpFreqs += integrateFreq[c.index];
                pooledCells++;
            }
            else
            {
                float diff = countFreq[c.index] - integrateFreq[c.index];
                chsq += (diff * diff) / integrateFreq[c.index];
                dof++;
            }
        }

        if (pooledExpFreqs > 0 || pooledFreqs > 0)
        {
            float diff = pooledFreqs - pooledExpFreqs;
            chsq += (diff * diff) / pooledExpFreqs;
            dof++;
        }
        dof -= 1;

        if (dof <= 0)
        {
            base_t::errMsg = std::format("degrees of freedom {} too low", dof);
            return BET_PRINT_MSG;
        }

        float pval = 1.0f - static_cast<float>(chi2CDF(chsq, dof));
        float alpha = 1.0f - std::pow(1.0f - threshold, 1.0f / numTests);

        if (pval < alpha || !std::isfinite(pval))
        {
            base_t::errMsg = std::format("chi2 test: rejected the null hypothesis (p-value = {:.3f}, significance level = {:.3f}", pval, alpha);
            return BET_PRINT_MSG;
        }

        return BET_NONE;
    }

    static void run(NBL_CONST_REF_ARG(STestInitParams) initparams, NBL_REF_ARG(FailureCallback) cb)
    {
        random::PCG32 pcg = random::PCG32::construct(initparams.state);
        random::DimAdaptorRecursive<random::PCG32, 2> rand2d = random::DimAdaptorRecursive<random::PCG32, 2>::construct(pcg);
        uint32_t2 state = rand2d();

        this_t t;
        t.init(state);
        t.rc.state = initparams.state;
        t.numSamples = initparams.samples;
        t.thetaSplits = initparams.thetaSplits;
        t.phiSplits = initparams.phiSplits;
        t.write_frequencies = initparams.writeFrequencies;
        t.initBxDF(t.rc);
        
        ErrorType e = t.test();
        if (e != BET_NONE)
            cb.__call(e, t, initparams.logInfo);
    }

    struct Cell {
        float expFreq;
        uint32_t index;
    };

    uint32_t thetaSplits = 80;
    uint32_t phiSplits = 160;
    uint32_t numSamples = 1000000;

    uint32_t threshold = 1e-2;
    uint32_t minFreq = 5;
    uint32_t numTests = 5;
    
    bool write_frequencies = true;
    float maxCountFreq;
    float maxIntFreq;

    std::vector<float> countFreq;
    std::vector<float> integrateFreq;
};
#endif

}
}

#endif