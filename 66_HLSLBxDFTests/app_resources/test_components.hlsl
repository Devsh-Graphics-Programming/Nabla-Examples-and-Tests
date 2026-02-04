#ifndef BXDFTESTS_TEST_COMPONENTS_HLSL
#define BXDFTESTS_TEST_COMPONENTS_HLSL

#include "tests_common.hlsl"

template<class BxDF, bool aniso = false>    // only for cook torrance bxdfs
struct TestNDF : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestNDF<BxDF, aniso>;
    using traits_t = bxdf::traits<BxDF>;

    TestResult compute()
    {
        aniso_cache dummy;
        iso_cache dummy_iso;

        // avoid cases where ux or uy might end up outside the input domain when eps is added
        if (!checkLt<float32_t3>(base_t::rc.u, hlsl::promote<float32_t3>(1.0-base_t::rc.eps)))
            return BTR_INVALID_TEST_CONFIG;

        base_t::rc.u.z = hlsl::mix(0.0, 1.0, base_t::rc.u.z > 0.5);
        float32_t3 ux = base_t::rc.u + float32_t3(eps,0,0);
        float32_t3 uy = base_t::rc.u + float32_t3(0,eps,0);

        NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BRDF && traits_t::IsMicrofacet)
        {
            NBL_IF_CONSTEXPR(aniso)
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
        NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
        {
            NBL_IF_CONSTEXPR(aniso)
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

        if (!BxDF::ndf_type::GuaranteedVNDF && !(s.isValid() && sx.isValid() && sy.isValid()))
            return BTR_INVALID_TEST_CONFIG;

        using ndf_type = typename base_t::bxdf_t::ndf_type;
        using quant_type = typename ndf_type::quant_type;
        using quant_query_type = typename ndf_type::quant_query_type;
        using dg1_query_type = typename ndf_type::dg1_query_type;
        using fresnel_type = typename base_t::bxdf_t::fresnel_type;

        float reflectance;
        bool isNdfInfinity;
        bool transmitted;
        NBL_IF_CONSTEXPR(aniso)
        {
            dg1_query_type dq = base_t::bxdf.ndf.template createDG1Query<aniso_interaction, aniso_cache>(base_t::anisointer, cache);
            fresnel_type _f = base_t::bxdf_t::__getOrientedFresnel(base_t::bxdf.fresnel, base_t::anisointer.getNdotV());
            quant_query_type qq = bxdf::impl::quant_query_helper<ndf_type, fresnel_type, base_t::bxdf_t::IsBSDF>::template __call<aniso_interaction, aniso_cache>(base_t::bxdf.ndf, _f, base_t::anisointer, cache);
            quant_type DG1 = base_t::bxdf.ndf.template DG1<sample_t, aniso_interaction>(dq, qq, s, base_t::anisointer, isNdfInfinity);
            dg1 = DG1.projectedLightMeasure;

            float VdotH = cache.getVdotH();
            NBL_IF_CONSTEXPR (traits_t::type == bxdf::BT_BSDF)
                VdotH = hlsl::abs(VdotH);
            reflectance = _f(VdotH)[0];
            transmitted = cache.isTransmission();
        }
        else
        {
            dg1_query_type dq = base_t::bxdf.ndf.template createDG1Query<iso_interaction, iso_cache>(base_t::isointer, isocache);
            fresnel_type _f = base_t::bxdf_t::__getOrientedFresnel(base_t::bxdf.fresnel, base_t::isointer.getNdotV());
            quant_query_type qq = bxdf::impl::quant_query_helper<ndf_type, fresnel_type, base_t::bxdf_t::IsBSDF>::template __call<iso_interaction, iso_cache>(base_t::bxdf.ndf, _f, base_t::isointer, isocache);
            quant_type DG1 = base_t::bxdf.ndf.template DG1<sample_t, iso_interaction>(dq, qq, s, base_t::isointer, isNdfInfinity);
            dg1 = DG1.projectedLightMeasure;

            float VdotH = isocache.getVdotH();
            NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BSDF)
                VdotH = hlsl::abs(VdotH);
            reflectance = _f(VdotH)[0];
            transmitted = isocache.isTransmission();
        }

        if (isNdfInfinity)
            return BTR_INVALID_TEST_CONFIG;

        if (reflectance < 0.f || reflectance > 1.f)
        {
#ifndef __HLSL_VERSION
            if (verbose)
                base_t::errMsg += std::format("reflectance={}, eta={}, transmitted={}", reflectance, base_t::rc.eta.x, transmitted ? "true" : "false");
#endif
            return BTR_ERROR_REFLECTANCE_OUT_OF_RANGE;
        }

        return BTR_NONE;
    }

    TestResult test()
    {
        if (traits_t::type == bxdf::BT_BRDF)
        {    
            if (base_t::isointer.getNdotV() <= bit_cast<float>(numeric_limits<float>::min))
                return BTR_INVALID_TEST_CONFIG;
        }        
        else if (traits_t::type == bxdf::BT_BSDF)
        {
            if (hlsl::abs(base_t::isointer.getNdotV()) <= bit_cast<float>(numeric_limits<float>::min))
                return BTR_INVALID_TEST_CONFIG;
        }

        TestResult res = compute();
        if (res != BTR_NONE)
            return res;

        const float absNdotL = hlsl::abs(s.getNdotL());
        if (absNdotL <= bit_cast<float>(numeric_limits<float>::min))
            return BTR_INVALID_TEST_CONFIG;

        // get jacobian
        float32_t2x2 m = float32_t2x2(
            sx.getTdotL() - s.getTdotL(), sy.getTdotL() - s.getTdotL(),
            sx.getBdotL() - s.getBdotL(), sy.getBdotL() - s.getBdotL()
        );
        float det = nbl::hlsl::determinant<float32_t2x2>(m) / (eps * eps);
        
        float jacobi_dg1 = det * dg1 / absNdotL;
        if (!checkZero<float>(det, 1e-3) && !testing::relativeApproxCompare<float>(jacobi_dg1, 1.0, 0.1))
        {
#ifndef __HLSL_VERSION
            if (verbose)
                base_t::errMsg += std::format("VdotH={}, NdotV={}, LdotH={}, NdotL={}, eta={}, alpha=[{},{}] Jacobian={}, DG1={}, Jacobian*DG1={}",
                                        aniso ? cache.getVdotH() : isocache.getVdotH(), aniso ? base_t::anisointer.getNdotV() : base_t::isointer.getNdotV(),
                                        aniso ? cache.getLdotH() : isocache.getLdotH(), s.getNdotL(), base_t::rc.eta.x, base_t::rc.alpha.x, base_t::rc.alpha.y,
                                        det, dg1, jacobi_dg1);
#endif
            return BTR_ERROR_JACOBIAN_TEST_FAIL;
        }

        return BTR_NONE;
    }

    static void run(NBL_CONST_REF_ARG(STestInitParams) initparams, NBL_REF_ARG(FailureCallback<this_t>) cb)
    {
        this_t t;
        t.init(initparams.halfSeed);
        t.rc.halfSeed = initparams.halfSeed;
        t.verbose = initparams.verbose;
        t.initBxDF(t.rc);
        
        TestResult e = t.test();
        if (e != BTR_NONE)
            cb.__call(e, t, initparams.logInfo);
    }

    float eps = 1e-5;
    sample_t s, sx, sy;
    aniso_cache cache;
    iso_cache isocache;
    float dg1;
    bool verbose;
};

template<class BxDF, bool aniso = false>
struct TestCTGenerateH : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestCTGenerateH<BxDF, aniso>;
    using traits_t = bxdf::traits<BxDF>;

    TestResult compute()
    {
        counter.reset();

        sample_t s;
        iso_cache isocache;
        aniso_cache cache;
        for (uint32_t i = 0; i < numSamples; i++)
        {
            float32_t3 u = ConvertToFloat01<uint32_t3>::__call(base_t::rc.rng_vec<3>());

            NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BRDF && !traits_t::IsMicrofacet)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u.xy);
            }
            NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BRDF && traits_t::IsMicrofacet)
            {
                NBL_IF_CONSTEXPR(aniso)
                    s = base_t::bxdf.generate(base_t::anisointer, u.xy, cache);
                else
                    s = base_t::bxdf.generate(base_t::isointer, u.xy, isocache);
            }
            NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BSDF && !traits_t::IsMicrofacet)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u);
            }
            NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
            {
                NBL_IF_CONSTEXPR(aniso)
                    s = base_t::bxdf.generate(base_t::anisointer, u, cache);
                else
                    s = base_t::bxdf.generate(base_t::isointer, u, isocache);
            }

            if (!BxDF::ndf_type::GuaranteedVNDF && !s.isValid())
                continue;

            bool transmitted;
            float NdotV, VdotH;
            float dotProductVdotL, VdotL;
            NBL_IF_CONSTEXPR(aniso)
            {
                NdotV = base_t::anisointer.getNdotV();
                VdotH = cache.getVdotH();
                transmitted = cache.isTransmission();
                dotProductVdotL = hlsl::dot(base_t::anisointer.getV().getDirection(), s.getL().getDirection());
                VdotL = cache.getVdotL();
            }
            else
            {
                NdotV = base_t::isointer.getNdotV();
                VdotH = isocache.getVdotH();
                transmitted = isocache.isTransmission();
                dotProductVdotL = hlsl::dot(base_t::isointer.getV().getDirection(), s.getL().getDirection());
                VdotL = isocache.getVdotL();
            }

            if (!(NdotV * VdotH >= 0.f))
            {
                if (immediateFail)
                {
                    base_t::errMsg += std::format("first failed case (NdotV*VdotH): i={}, u=[{},{},{}] NdotV={}, VdotH={}", i, u.x, u.y, u.z, NdotV, VdotH);
                    return BTR_WARNING_GENERATED_H_INVALID;
                }
                else
                {
                    counter.NdotVVdotHfail++;
                    transmitted ? counter.transmitted++ : counter.reflected++;
                }
            }

            if (!checkZero<float>(dotProductVdotL - VdotL, 1e-4))
            {
                if (immediateFail)
                {
                    base_t::errMsg += std::format("first failed case (compare VdotL): i={}, u=[{},{},{}] {}!={}", i, u.x, u.y, u.z, dotProductVdotL, VdotL);
                    return BTR_WARNING_GENERATED_H_INVALID;
                }
                else
                {
                    counter.VdotLfail++;
                    transmitted ? counter.transmitted++ : counter.reflected++;
                }
            }

            counter.total++;
        }

        float totalFails = counter.totalFails();
        if (totalFails > 0)
        {
            base_t::errMsg += std::format("fail count={} out of {} valid samples: [{}] NdotV*VdotH, [{}] compare VdotL, [{}] transmitted, [{}] reflected, alpha=[{},{}]",
                                totalFails, counter.total, counter.NdotVVdotHfail, counter.VdotLfail,
                                counter.transmitted, counter.reflected, base_t::rc.alpha.x, base_t::rc.alpha.y);
            return BTR_WARNING_GENERATED_H_INVALID;
        }

        return BTR_NONE;
    }

    TestResult test()
    {
        if (traits_t::type == bxdf::BT_BRDF)
            if (base_t::isointer.getNdotV() <= numeric_limits<float>::min)
                return BTR_INVALID_TEST_CONFIG;
        else if (traits_t::type == bxdf::BT_BSDF)
            if (hlsl::abs(base_t::isointer.getNdotV()) <= numeric_limits<float>::min)
                return BTR_INVALID_TEST_CONFIG;

        TestResult res = compute();
        if (res != BTR_NONE)
            return res;

        return BTR_NONE;
    }

    static void run(NBL_CONST_REF_ARG(STestInitParams) initparams, NBL_REF_ARG(FailureCallback<this_t>) cb)
    {
        this_t t;
        t.init(initparams.halfSeed);
        t.rc.halfSeed = initparams.halfSeed;
        t.numSamples = initparams.samples;
        t.immediateFail = initparams.immediateFail;
        t.initBxDF(t.rc);
        
        TestResult e = t.test();
        if (e != BTR_NONE)
            cb.__call(e, t, initparams.logInfo);
    }

    struct Counter
    {
        uint32_t NdotVVdotHfail;
        uint32_t VdotLfail;
        uint32_t reflected;
        uint32_t transmitted;
        uint32_t total;

        void reset()
        {
            NdotVVdotHfail = 0;
            VdotLfail = 0;
            reflected = 0;
            transmitted = 0;
            total = 0;
        }

        float totalFails() { return NdotVVdotHfail + VdotLfail; }
    };

    bool immediateFail = false;
    uint32_t numSamples = 1000000;
    Counter counter;
};

#endif
