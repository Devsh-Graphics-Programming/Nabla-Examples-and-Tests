#ifndef BXDFTESTS_TESTS_HLSL
#define BXDFTESTS_TESTS_HLSL

#include "tests_common.hlsl"

template<class BxDF, bool aniso = false>
struct TestJacobian : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestJacobian<BxDF, aniso>;
    using traits_t = bxdf::traits<BxDF>;

    ErrorType compute()
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
            if (hlsl::abs(s.getNdotL()) <= bit_cast<float>(numeric_limits<float>::min))
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
            if (hlsl::abs(base_t::isointer.getNdotV()) <= bit_cast<float>(numeric_limits<float>::min))
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
        this_t t;
        t.init(initparams.halfSeed);
        t.rc.halfSeed = initparams.halfSeed;
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

    ErrorType compute()
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
            if (hlsl::abs(s.getNdotL()) <= bit_cast<float>(numeric_limits<float>::min))
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
            if (hlsl::abs(base_t::isointer.getNdotV()) <= bit_cast<float>(numeric_limits<float>::min))
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
                return BET_NO_RECIPROCITY;
            }

        return BET_NONE;
    }

    static void run(NBL_CONST_REF_ARG(STestInitParams) initparams, NBL_REF_ARG(FailureCallback) cb)
    {
        this_t t;
        t.init(initparams.halfSeed);
        t.rc.halfSeed = initparams.halfSeed;
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

#endif
