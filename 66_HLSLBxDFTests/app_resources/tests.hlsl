#ifndef BXDFTESTS_TESTS_HLSL
#define BXDFTESTS_TESTS_HLSL

#include "tests_common.hlsl"

namespace nbl
{
namespace hlsl
{

template<class BxDF, bool aniso = false>
struct TestJacobian : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestJacobian<BxDF, aniso>;
    using traits_t = bxdf::traits<BxDF>;

    virtual ErrorType compute() override
    {
        aniso_cache cache, dummy;
        iso_cache isocache, dummy_iso;

        float32_t3 ux = base_t::rc.u + float32_t3(base_t::rc.eps,0,0);
        float32_t3 uy = base_t::rc.u + float32_t3(0,base_t::rc.eps,0);

        if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BRDF && !traits_t::IsMicrofacet)
        {
            s = base_t::bxdf.generate(base_t::isointer, base_t::rc.u.xy);
            sx = base_t::bxdf.generate(base_t::isointer, ux.xy);
            sy = base_t::bxdf.generate(base_t::isointer, uy.xy);
        }
        if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BRDF && traits_t::IsMicrofacet)
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
        if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BSDF && !traits_t::IsMicrofacet)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u);
            sx = base_t::bxdf.generate(base_t::anisointer, ux);
            sy = base_t::bxdf.generate(base_t::anisointer, uy);
        }
        if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
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

        if (!(s.isValid() && sx.isValid() && sy.isValid()))
            return BET_INVALID;

        if (traits_t::type == bxdf::BT_BRDF)
        {
            if (s.getNdotL() <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }
        else if (traits_t::type == bxdf::BT_BSDF)
        {
            if (abs<float>(s.getNdotL()) <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }

        if NBL_CONSTEXPR_FUNC (!traits_t::IsMicrofacet)
        {
            pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer);
            bsdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer));
            transmitted = base_t::isointer.getNdotV() * s.getNdotL() < 0.f;
        }
        if NBL_CONSTEXPR_FUNC (traits_t::IsMicrofacet)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                pdf = base_t::bxdf.quotient_and_pdf(s, base_t::anisointer, cache);
                bsdf = float32_t3(base_t::bxdf.eval(s, base_t::anisointer, cache));
                transmitted = cache.isTransmission();
            }
            else
            {
                pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer, isocache);
                bsdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer, isocache));
                transmitted = isocache.isTransmission();
            }
        }

        return BET_NONE;
    }

    ErrorType test()
    {
        if (traits_t::type == bxdf::BT_BRDF)
        {    
            if (base_t::isointer.getNdotV() <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }        
        else if (traits_t::type == bxdf::BT_BSDF)
        {
            if (abs<float>(base_t::isointer.getNdotV()) <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }

        ErrorType res = compute();
        if (res != BET_NONE)
            return res;

        if (checkZero<float>(pdf.pdf, 1e-5) && !checkZero<float32_t3>(pdf.quotient, 1e-5))  // something generated cannot have 0 probability of getting generated
            return BET_PDF_ZERO;

        if (!checkLt<float32_t3>(pdf.quotient, (float32_t3)bit_cast<float, uint32_t>(numeric_limits<float>::infinity)))    // importance sampler's job to prevent inf
            return BET_QUOTIENT_INF;

        if (checkZero<float32_t3>(bsdf, 1e-5) || checkZero<float32_t3>(pdf.quotient, 1e-5))
            return BET_NONE;    // produces an "impossible" sample

        if (checkLt<float32_t3>(bsdf, (float32_t3)0.0) || checkLt<float32_t3>(pdf.quotient, (float32_t3)0.0) || pdf.pdf < 0.0)
            return BET_NEGATIVE_VAL;

        // get jacobian
        float32_t2x2 m = float32_t2x2(
            sx.getTdotL() - s.getTdotL(), sy.getTdotL() - s.getTdotL(),
            sx.getBdotL() - s.getBdotL(), sy.getBdotL() - s.getBdotL()
        );
        float det = nbl::hlsl::determinant<float32_t2x2>(m);

        if (!checkZero<float>(det * pdf.pdf / s.getNdotL(), 1e-4))
            return BET_JACOBIAN;

        float32_t3 quo_pdf = pdf.value();
        if (!checkEq<float32_t3>(quo_pdf, bsdf, 1e-4))
        {
#ifndef __HLSL_VERSION
            if (verbose)
                base_t::errMsg += std::format("transmitted={}, quotient*pdf=[{},{},{}]    eval=[{},{},{}]",
                    transmitted ? "true" : "false",
                    quo_pdf.x, quo_pdf.y, quo_pdf.z,
                    bsdf.x, bsdf.y, bsdf.z);
#endif
            return BET_PDF_EVAL_DIFF;
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
        t.verbose = initparams.verbose;
        t.initBxDF(t.rc);
        
        ErrorType e = t.test();
        if (e != BET_NONE)
            cb.__call(e, t, initparams.logInfo);
    }

    sample_t s, sx, sy;
    quotient_pdf_t pdf;
    float32_t3 bsdf;
    bool transmitted;
    bool verbose;
};

template<class BxDF, bool aniso = false>
struct TestReciprocity : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestReciprocity<BxDF, aniso>;
    using traits_t = bxdf::traits<BxDF>;

    using iso_interaction_t = typename BxDF::isotropic_interaction_type;
    using aniso_interaction_t = typename BxDF::anisotropic_interaction_type;

    virtual ErrorType compute() override
    {
        aniso_cache cache, rec_cache;
        iso_cache isocache, rec_isocache;

        if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
        {
            isointer = iso_interaction_t::template copy<iso_interaction>(base_t::isointer);
            anisointer = aniso_interaction_t::template copy<aniso_interaction>(base_t::anisointer);
        }
        else
        {
            isointer = base_t::isointer;
            anisointer = base_t::anisointer;
        }

        if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BRDF && !traits_t::IsMicrofacet)
        {
            s = base_t::bxdf.generate(anisointer, base_t::rc.u.xy);
        }
        if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BRDF && traits_t::IsMicrofacet)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                s = base_t::bxdf.generate(anisointer, base_t::rc.u.xy, cache);
            }
            else
            {
                s = base_t::bxdf.generate(isointer, base_t::rc.u.xy, isocache);
            }
        }
        if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BSDF && !traits_t::IsMicrofacet)
        {
            s = base_t::bxdf.generate(anisointer, base_t::rc.u);
        }
        if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                s = base_t::bxdf.generate(anisointer, base_t::rc.u, cache);
            }
            else
            {
                s = base_t::bxdf.generate(isointer, base_t::rc.u, isocache);
            }
        }

        if (!s.isValid())
            return BET_INVALID;

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

        float32_t3x3 toTangentSpace = anisointer.getToTangentSpace();
        ray_dir_info_t rec_V = s.getL();
        ray_dir_info_t rec_localV = rec_V.transform(toTangentSpace);
        ray_dir_info_t rec_localL = base_t::rc.V.transform(toTangentSpace);
        rec_s = sample_t::createFromTangentSpace(rec_localL, anisointer.getFromTangentSpace());

        rec_isointer = iso_interaction_t::create(rec_V, base_t::rc.N);
        rec_isointer.luminosityContributionHint = isointer.luminosityContributionHint;
        rec_anisointer = aniso_interaction_t::create(rec_isointer, base_t::rc.T, base_t::rc.B);
        rec_cache = cache;
        rec_cache.iso_cache.VdotH = cache.iso_cache.getLdotH();
        rec_cache.iso_cache.LdotH = cache.iso_cache.getVdotH();
        rec_isocache = isocache;
        rec_isocache.VdotH = isocache.getLdotH();
        rec_isocache.LdotH = isocache.getVdotH();
        
        if NBL_CONSTEXPR_FUNC (!traits_t::IsMicrofacet)
        {
            bsdf = float32_t3(base_t::bxdf.eval(s, isointer));
            rec_bsdf = float32_t3(base_t::bxdf.eval(rec_s, rec_isointer));
        }
        if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BRDF && traits_t::IsMicrofacet)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                bsdf = float32_t3(base_t::bxdf.eval(s, anisointer, cache));
                rec_bsdf = float32_t3(base_t::bxdf.eval(rec_s, rec_anisointer, rec_cache));
            }
            else
            {
                bsdf = float32_t3(base_t::bxdf.eval(s, isointer, isocache));
                rec_bsdf = float32_t3(base_t::bxdf.eval(rec_s, rec_isointer, rec_isocache));
            }
        }
        if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
        {
            if NBL_CONSTEXPR_FUNC (aniso)
            {
                anisointer.isotropic.pathOrigin = bxdf::PathOrigin::PO_SENSOR;
                bsdf = float32_t3(base_t::bxdf.eval(s, anisointer, cache));
                rec_anisointer.isotropic.pathOrigin = bxdf::PathOrigin::PO_LIGHT;
                rec_bsdf = float32_t3(base_t::bxdf.eval(rec_s, rec_anisointer, rec_cache));
            }
            else
            {
                isointer.pathOrigin = bxdf::PathOrigin::PO_SENSOR;
                bsdf = float32_t3(base_t::bxdf.eval(s, isointer, isocache));
                rec_isointer.pathOrigin = bxdf::PathOrigin::PO_LIGHT;
                rec_bsdf = float32_t3(base_t::bxdf.eval(rec_s, rec_isointer, rec_isocache));
            }
        }

        transmitted = aniso ? cache.isTransmission() : isocache.isTransmission();

#ifndef __HLSL_VERSION
        if (verbose)
            base_t::errMsg += std::format("isTransmission: {}, NdotV: {}, NdotL: {}, VdotH: {}, LdotH: {}, NdotH: {}",
                transmitted ? "true" : "false",
                isointer.getNdotV(), s.getNdotL(),
                aniso ? cache.getVdotH() : isocache.getVdotH(), aniso ? cache.getLdotH() : isocache.getLdotH(), aniso ? cache.getAbsNdotH() : isocache.getAbsNdotH());
#endif

        return BET_NONE;
    }

    ErrorType test()
    {
        if (traits_t::type == bxdf::BT_BRDF)
        {    
            if (base_t::isointer.getNdotV() <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }        
        else if (traits_t::type == bxdf::BT_BSDF)
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

        float32_t3 a = bsdf / hlsl::abs(s.getNdotL());
        float32_t3 b = rec_bsdf / hlsl::abs(rec_s.getNdotL());
        if (!(a == b))  // avoid division by 0
            if (!checkEq<float32_t3>(a, b, 1e-2))
            {
#ifndef __HLSL_VERSION
                if (verbose)
                    base_t::errMsg += std::format("    front=[{},{},{}]    rec=[{},{},{}]",
                        a.x, a.y, a.z,
                        b.x, b.y, b.z);
#endif
                return BET_RECIPROCITY;
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
        t.verbose = initparams.verbose;
        t.initBxDF(t.rc);
        
        ErrorType e = t.test();
        if (e != BET_NONE)
            cb.__call(e, t, initparams.logInfo);
    }

    sample_t s, rec_s;
    float32_t3 bsdf, rec_bsdf;
    iso_interaction_t isointer, rec_isointer;
    aniso_interaction_t anisointer, rec_anisointer;
    bool transmitted;
    bool verbose;
};

#ifndef __HLSL_VERSION  // because unordered_map
template<class BxDF, bool aniso = false>
struct TestBucket : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestBucket<BxDF, aniso>;
    using traits_t = bxdf::traits<BxDF>;

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

        nbl::hlsl::random::DimAdaptorRecursive<nbl::hlsl::Xoroshiro64Star, 3> rng_vec3 = nbl::hlsl::random::DimAdaptorRecursive<nbl::hlsl::Xoroshiro64Star, 3>::construct(base_t::rc.rng);
        for (uint32_t i = 0; i < numSamples; i++)
        {
            float32_t3 u = ConvertToFloat01<uint32_t3>::__call(rng_vec3());
            // u.z = 0.0;

            if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BRDF && !traits_t::IsMicrofacet)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u.xy);
            }
            if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BRDF && traits_t::IsMicrofacet)
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
            if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BSDF && !traits_t::IsMicrofacet)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u);
            }
            if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
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

            if (!s.isValid())
                continue;

            if NBL_CONSTEXPR_FUNC (!traits_t::IsMicrofacet)
            {
                pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer);
                bsdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer));
            }
            if NBL_CONSTEXPR_FUNC (traits_t::IsMicrofacet)
            {
                if NBL_CONSTEXPR_FUNC (aniso)
                {
                    pdf = base_t::bxdf.quotient_and_pdf(s, base_t::anisointer, cache);
                    bsdf = float32_t3(base_t::bxdf.eval(s, base_t::anisointer, cache));
                }
                else
                {
                    pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer, isocache);
                    bsdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer, isocache));
                }
            }

            // put s into bucket
            float32_t3x3 toTangentSpace = base_t::anisointer.getToTangentSpace();
            const ray_dir_info_t localL = s.getL().transform(toTangentSpace);
            math::Polar<float> polarCoords = math::Polar<float>::createFromCartesian(localL.getDirection());
            float32_t2 bucket = float32_t2(bin(polarCoords.theta * numbers::inv_pi<float>), bin(polarCoords.phi * 0.5f * numbers::inv_pi<float>));

            if (pdf.pdf == bit_cast<float>(numeric_limits<float>::infinity))
                buckets[bucket] += 1;
        }

#ifndef __HLSL_VERSION
        // double check this conversion makes sense
        for (auto const& b : buckets) {
            if (!selective || b.second > 0)
            {
                math::Polar<float> polarCoords;
                polarCoords.theta = b.first.x * numbers::pi<float>;
                polarCoords.phi = b.first.y * 2.f * numbers::pi<float>;
                const float32_t3 v = polarCoords.getCartesian();
                base_t::errMsg += std::format("({:.3f},{:.3f},{:.3f}): {}\n", v.x, v.y, v.z, b.second);
            }
        }
#endif
        return BET_NONE;
    }

    ErrorType test()
    {
        if (traits_t::type == bxdf::BT_BRDF)
        {    
            if (base_t::isointer.getNdotV() <= bit_cast<float>(numeric_limits<float>::min))
                return BET_INVALID;
        }        
        else if (traits_t::type == bxdf::BT_BSDF)
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
    using traits_t = bxdf::traits<BxDF>;

    void clearBuckets()
    {
        const uint32_t freqSize = thetaSplits * phiSplits;
        countFreq.resize(freqSize);
        std::fill(countFreq.begin(), countFreq.end(), 0);
        integrateFreq.resize(freqSize);
        std::fill(integrateFreq.begin(), integrateFreq.end(), 0);
        maxCountFreq = 0.f;
        maxIntFreq = 0.f;
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
        
        // write sample count from generate, top half
        Array2D<Rgba> pixels(totalWidth, totalHeight);
        for (int y = 0; y < thetaSplits; y++)
            for (int x = 0; x < phiSplits; x++)
                pixels[y][x] = mapColor(countFreq[y * phiSplits + x], 0.f, maxCountFreq);

        // for (int x = 0; x < phiSplits; x++)
        //     pixels[thetaSplits][x] = Rgba(1, 1, 1);

        // write values of pdf, bottom half
        for (int y = 0; y < thetaSplits; y++)
            for (int x = 0; x < phiSplits; x++)
                pixels[thetaSplits + y][x] = mapColor(integrateFreq[y * phiSplits + x], 0.f, maxIntFreq);
    
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
        nbl::hlsl::random::DimAdaptorRecursive<nbl::hlsl::Xoroshiro64Star, 3> rng_vec3 = nbl::hlsl::random::DimAdaptorRecursive<nbl::hlsl::Xoroshiro64Star, 3>::construct(base_t::rc.rng);
        for (uint32_t i = 0; i < numSamples; i++)
        {
            float32_t3 u = ConvertToFloat01<uint32_t3>::__call(rng_vec3());
            u.x = hlsl::clamp(u.x, base_t::rc.eps, 1.f-base_t::rc.eps);
            u.y = hlsl::clamp(u.y, base_t::rc.eps, 1.f-base_t::rc.eps);
            // u.z = 0.0;

            if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BRDF && !traits_t::IsMicrofacet)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u.xy);
            }
            if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BRDF && traits_t::IsMicrofacet)
            {
                if NBL_CONSTEXPR_FUNC(aniso)
                    s = base_t::bxdf.generate(base_t::anisointer, u.xy, cache);
                else
                    s = base_t::bxdf.generate(base_t::isointer, u.xy, isocache);
            }
            if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BSDF && !traits_t::IsMicrofacet)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u);
            }
            if NBL_CONSTEXPR_FUNC (traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
            {
                if NBL_CONSTEXPR_FUNC(aniso)
                    s = base_t::bxdf.generate(base_t::anisointer, u, cache);
                else
                    s = base_t::bxdf.generate(base_t::isointer, u, isocache);
            }

            if (!s.isValid())
                continue;

            // put s into bucket
            math::Polar<float> polarCoords = math::Polar<float>::createFromCartesian(s.getL().getDirection());
            polarCoords.theta *= thetaFactor;
            polarCoords.phi *= phiFactor;
            if (polarCoords.phi < 0)
                polarCoords.phi += 2.f * numbers::pi<float> * phiFactor;

            int thetaBin = clamp<int>((int)std::floor(polarCoords.theta), 0, thetaSplits - 1);
            int phiBin = clamp<int>((int)std::floor(polarCoords.phi), 0, phiSplits - 1);

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

                        ray_dir_info_t V = base_t::rc.V;
                        ray_dir_info_t L;
                        L.direction = hlsl::normalize(float32_t3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));
                        float32_t3 N = base_t::anisointer.getN();
                        float NdotL = hlsl::dot(N, L.direction);

                        float32_t3 T = base_t::anisointer.getT();
                        float32_t3 B = base_t::anisointer.getB();
                        sample_t s = sample_t::create(L, T, B, NdotL);

                        NBL_IF_CONSTEXPR(traits_t::IsMicrofacet)
                        {
                            const float NdotV = base_t::anisointer.getNdotV();
                            NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BRDF)
                                if (NdotV < 0.f) return 0.f;

                            float eta = 1.f;
                            const float NdotL = s.getNdotL();
                            if (NdotV * NdotL < 0.f)
                                eta = NdotV < 0.f ? 1.f/base_t::rc.eta.x : base_t::rc.eta.x;
                            float32_t3 H = hlsl::normalize(V.getDirection() + L.getDirection() * eta);
                            float VdotH = hlsl::dot(V.getDirection(), H);
                            if (NdotV * VdotH < 0.f)
                            {
                                H = -H;
                                VdotH = -VdotH;
                            }

                            cache.iso_cache.VdotH = VdotH;
                            cache.iso_cache.LdotH = hlsl::dot(L.getDirection(), H);
                            cache.iso_cache.VdotL = hlsl::dot(V.getDirection(), L.getDirection());
                            cache.iso_cache.absNdotH = hlsl::abs(hlsl::dot(N, H));
                            cache.iso_cache.NdotH2 = cache.iso_cache.absNdotH * cache.iso_cache.absNdotH;

                            if (!cache.isValid(bxdf::fresnel::OrientedEtas<vector<float,1> >::create(1.f, hlsl::promote<vector<float,1> >(eta))))
                                return 0.f;

                            const float32_t3 T = base_t::anisointer.getT();
                            const float32_t3 B = base_t::anisointer.getB();
                            cache.fillTangents(T, B, H);
                        }

                        float pdf;
                        if NBL_CONSTEXPR_FUNC (!traits_t::IsMicrofacet)
                        {
                            pdf = base_t::bxdf.pdf(s, base_t::isointer);
                        }
                        if NBL_CONSTEXPR_FUNC (traits_t::IsMicrofacet)
                        {
                            if NBL_CONSTEXPR_FUNC (aniso)
                            {
                                pdf = base_t::bxdf.pdf(s, base_t::anisointer, cache);
                            }
                            else
                            {
                                pdf = base_t::bxdf.pdf(s, base_t::isointer, cache.iso_cache);
                            }
                        }
                        return pdf * sinTheta;
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
        if (traits_t::type == bxdf::BT_BRDF)
            if (base_t::isointer.getNdotV() <= numeric_limits<float>::min)
                return BET_INVALID;
        else if (traits_t::type == bxdf::BT_BSDF)
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