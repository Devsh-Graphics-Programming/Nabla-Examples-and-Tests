#ifndef BXDFTESTS_TESTS_HLSL
#define BXDFTESTS_TESTS_HLSL

#include "tests_common.hlsl"

template<class BxDF, bool aniso = false>
struct TestJacobian : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestJacobian<BxDF, aniso>;
    using traits_t = bxdf::traits<BxDF>;

    TestResult compute()
    {
        aniso_cache cache, dummy;
        iso_cache isocache, dummy_iso;

        // avoid cases where ux or uy might end up outside the input domain when eps is added
        if (!checkLt<float32_t3>(base_t::rc.u, hlsl::promote<float32_t3>(1.0-base_t::rc.eps)))
            return BTR_INVALID_TEST_CONFIG;

        float32_t3 ux = base_t::rc.u + float32_t3(base_t::rc.eps,0,0);
        float32_t3 uy = base_t::rc.u + float32_t3(0,base_t::rc.eps,0);

        NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BRDF && !traits_t::IsMicrofacet)
        {
            s = base_t::bxdf.generate(base_t::isointer, base_t::rc.u.xy);
            sx = base_t::bxdf.generate(base_t::isointer, ux.xy);
            sy = base_t::bxdf.generate(base_t::isointer, uy.xy);
        }
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
        NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BSDF && !traits_t::IsMicrofacet)
        {
            s = base_t::bxdf.generate(base_t::anisointer, base_t::rc.u);
            sx = base_t::bxdf.generate(base_t::anisointer, ux);
            sy = base_t::bxdf.generate(base_t::anisointer, uy);
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

        // TODO: might want to distinguish between invalid H and sample produced below hemisphere
        if (!(s.isValid() && sx.isValid() && sy.isValid()))
            return BTR_INVALID_TEST_CONFIG;

        NBL_IF_CONSTEXPR(!traits_t::IsMicrofacet)
        {
            sampledLi = base_t::bxdf.quotient_and_pdf(s, base_t::isointer);
            Li = float32_t3(base_t::bxdf.eval(s, base_t::isointer));
            transmitted = base_t::isointer.getNdotV() * s.getNdotL() < 0.f;
        }
        NBL_IF_CONSTEXPR(traits_t::IsMicrofacet)
        {
            NBL_IF_CONSTEXPR(aniso)
            {
                sampledLi = base_t::bxdf.quotient_and_pdf(s, base_t::anisointer, cache);
                Li = float32_t3(base_t::bxdf.eval(s, base_t::anisointer, cache));
                transmitted = cache.isTransmission();
            }
            else
            {
                sampledLi = base_t::bxdf.quotient_and_pdf(s, base_t::isointer, isocache);
                Li = float32_t3(base_t::bxdf.eval(s, base_t::isointer, isocache));
                transmitted = isocache.isTransmission();
            }
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

        if (sampledLi.pdf < 0.f)    // pdf should not be negative
            return BTR_ERROR_NEGATIVE_VAL;

        if (sampledLi.pdf < bit_cast<float>(numeric_limits<float>::min))   // there's exceptional cases where pdf=0, so we check here to avoid adding all edge-cases, but quotient must be positive afterwards
            return BTR_NONE;

        if (checkLt<float32_t3>(Li, hlsl::promote<float32_t3>(0.0)) || checkLt<float32_t3>(sampledLi.quotient, hlsl::promote<float32_t3>(0.0)))
            return BTR_ERROR_NEGATIVE_VAL;

        if (!checkLt<float32_t3>(sampledLi.quotient, hlsl::promote<float32_t3>(bit_cast<float, uint32_t>(numeric_limits<float>::infinity))))    // importance sampler's job to prevent inf
            return BTR_ERROR_QUOTIENT_INF;

        // we've already checked above if:
        // 1. PDF is positive
        // 2. quotient is positive and (1) already checked
        // So if we must have `eval == quotient*pdf` , then eval must also be positive
        // However for mixture of, or singular delta BxDF the bsdf can be less due to removal of Dirac-Delta lobes from the eval method, which is why allow `BTR_NONE` in this case
        if (checkZero<float32_t3>(Li, 1e-5) || checkZero<float32_t3>(sampledLi.quotient, 1e-5))
            return BTR_NONE;

        if (hlsl::isnan(sampledLi.pdf))
            return BTR_ERROR_GENERATED_SAMPLE_NAN_PDF;

        // get jacobian
        float32_t2x2 m = float32_t2x2(
            sx.getTdotL() - s.getTdotL(), sy.getTdotL() - s.getTdotL(),
            sx.getBdotL() - s.getBdotL(), sy.getBdotL() - s.getBdotL()
        );
        float det = nbl::hlsl::determinant<float32_t2x2>(m);

        if (hlsl::isinf(sampledLi.pdf))
        {
            // if pdf is infinite then density is infinite and no differential area inbetween samples
            if (!checkZero<float>(det, numeric_limits<float>::min * base_t::rc.eps * base_t::rc.eps))
                return BTR_ERROR_JACOBIAN_TEST_FAIL;
            // valid behaviour, but obviously can't check eval = quotient*pdf
            return BTR_NONE;
        }
        else if (checkZero<float>(det, numeric_limits<float>::min * base_t::rc.eps * base_t::rc.eps))
        {
#ifndef __HLSL_VERSION
            if (verbose)
                base_t::errMsg += std::format("determinant={}", det);
#endif
            return BTR_ERROR_JACOBIAN_TEST_FAIL;
        }

        float32_t3 quo_pdf = sampledLi.value();
        if (!testing::relativeApproxCompare<float32_t3>(quo_pdf, Li, 1e-4))
        {
#ifndef __HLSL_VERSION
            if (verbose)
                base_t::errMsg += std::format("transmitted={}, quotient*pdf=[{},{},{}]    eval=[{},{},{}]",
                    transmitted ? "true" : "false",
                    quo_pdf.x, quo_pdf.y, quo_pdf.z,
                    Li.x, Li.y, Li.z);
#endif
            return BTR_ERROR_PDF_EVAL_DIFF;
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

    sample_t s, sx, sy;
    quotient_pdf_t sampledLi;
    float32_t3 Li;
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

    TestResult compute()
    {
        aniso_cache cache, rec_cache;
        iso_cache isocache, rec_isocache;

        NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
        {
            isointer = iso_interaction_t::template copy<iso_interaction>(base_t::isointer);
            anisointer = aniso_interaction_t::template copy<aniso_interaction>(base_t::anisointer);
        }
        else
        {
            isointer = base_t::isointer;
            anisointer = base_t::anisointer;
        }

        NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BRDF && !traits_t::IsMicrofacet)
        {
            s = base_t::bxdf.generate(anisointer, base_t::rc.u.xy);
        }
        NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BRDF && traits_t::IsMicrofacet)
        {
            NBL_IF_CONSTEXPR(aniso)
            {
                s = base_t::bxdf.generate(anisointer, base_t::rc.u.xy, cache);
            }
            else
            {
                s = base_t::bxdf.generate(isointer, base_t::rc.u.xy, isocache);
            }
        }
        NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BSDF && !traits_t::IsMicrofacet)
        {
            s = base_t::bxdf.generate(anisointer, base_t::rc.u);
        }
        NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
        {
            NBL_IF_CONSTEXPR(aniso)
            {
                s = base_t::bxdf.generate(anisointer, base_t::rc.u, cache);
            }
            else
            {
                s = base_t::bxdf.generate(isointer, base_t::rc.u, isocache);
            }
        }

        // TODO: might want to distinguish between invalid H and sample produced below hemisphere
        if (!s.isValid())
            return BTR_INVALID_TEST_CONFIG;

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
        
        NBL_IF_CONSTEXPR(!traits_t::IsMicrofacet)
        {
            Li = float32_t3(base_t::bxdf.eval(s, isointer));
            recLi = float32_t3(base_t::bxdf.eval(rec_s, rec_isointer));
        }
        NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BRDF && traits_t::IsMicrofacet)
        {
            NBL_IF_CONSTEXPR(aniso)
            {
                Li = float32_t3(base_t::bxdf.eval(s, anisointer, cache));
                recLi = float32_t3(base_t::bxdf.eval(rec_s, rec_anisointer, rec_cache));
            }
            else
            {
                Li = float32_t3(base_t::bxdf.eval(s, isointer, isocache));
                recLi = float32_t3(base_t::bxdf.eval(rec_s, rec_isointer, rec_isocache));
            }
        }
        NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
        {
            NBL_IF_CONSTEXPR(aniso)
            {
                anisointer.isotropic.pathOrigin = bxdf::PathOrigin::PO_SENSOR;
                Li = float32_t3(base_t::bxdf.eval(s, anisointer, cache));
                rec_anisointer.isotropic.pathOrigin = bxdf::PathOrigin::PO_LIGHT;
                recLi = float32_t3(base_t::bxdf.eval(rec_s, rec_anisointer, rec_cache));
            }
            else
            {
                isointer.pathOrigin = bxdf::PathOrigin::PO_SENSOR;
                Li = float32_t3(base_t::bxdf.eval(s, isointer, isocache));
                rec_isointer.pathOrigin = bxdf::PathOrigin::PO_LIGHT;
                recLi = float32_t3(base_t::bxdf.eval(rec_s, rec_isointer, rec_isocache));
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

        if (checkZero<float32_t3>(Li, 1e-5))
            return BTR_NONE;    // produces an "impossible" sample

        if (checkLt<float32_t3>(Li, (float32_t3)0.0))
            return BTR_ERROR_NEGATIVE_VAL;

        float32_t3 a = Li / hlsl::abs(s.getNdotL());
        float32_t3 b = recLi / hlsl::abs(rec_s.getNdotL());
        if (!(a == b))  // avoid division by 0
            if (!testing::relativeApproxCompare<float32_t3>(a, b, 1e-2))
            {
#ifndef __HLSL_VERSION
                if (verbose)
                    base_t::errMsg += std::format("    front=[{},{},{}]    rec=[{},{},{}]",
                        a.x, a.y, a.z,
                        b.x, b.y, b.z);
#endif
                return BTR_ERROR_NO_RECIPROCITY;
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

    sample_t s, rec_s;
    float32_t3 Li, recLi;
    iso_interaction_t isointer, rec_isointer;
    aniso_interaction_t anisointer, rec_anisointer;
    bool transmitted;
    bool verbose;
};

#endif
