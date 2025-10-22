#ifndef BXDFTESTS_TEST_COMPONENTS_HLSL
#define BXDFTESTS_TEST_COMPONENTS_HLSL

#include "tests_common.hlsl"

namespace nbl
{
namespace hlsl
{

template<class BxDF, bool aniso = false>    // only for cook torrance bxdfs
struct TestNDF : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestNDF<BxDF, aniso>;
    using traits_t = bxdf::traits<BxDF>;

    virtual ErrorType compute() override
    {
        aniso_cache dummy;
        iso_cache dummy_iso;

        float32_t3 ux = base_t::rc.u + float32_t3(eps,0,0);
        float32_t3 uy = base_t::rc.u + float32_t3(0,eps,0);

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

        // test jacobian is correct
        // float32_t3x3 fromTangentSpace = base_t::anisointer.getFromTangentSpace();
        // ray_dir_info_t tmpL;
        // tmpL.setDirection(sampling::ProjectedSphere<float>::generate(base_t::rc.u));
        // s = sample_t::createFromTangentSpace(tmpL, fromTangentSpace);
        // tmpL.setDirection(sampling::ProjectedSphere<float>::generate(ux));
        // sx = sample_t::createFromTangentSpace(tmpL, fromTangentSpace);
        // tmpL.setDirection(sampling::ProjectedSphere<float>::generate(uy));
        // sy = sample_t::createFromTangentSpace(tmpL, fromTangentSpace);

        if (!(s.isValid() && sx.isValid() && sy.isValid()))
            return BET_INVALID;

        // TODO: add checks with need clamp trait
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

        using ndf_type = typename base_t::bxdf_t::ndf_type;
        using quant_type = typename ndf_type::quant_type;
        using quant_query_type = typename ndf_type::quant_query_type;
        using dg1_query_type = typename ndf_type::dg1_query_type;
        using fresnel_type = typename base_t::bxdf_t::fresnel_type;

        NBL_IF_CONSTEXPR(aniso)
        {
            dg1_query_type dq = base_t::bxdf.ndf.template createDG1Query<aniso_interaction, aniso_cache>(base_t::anisointer, cache);
            fresnel_type _f = bxdf::impl::getOrientedFresnel<fresnel_type, base_t::bxdf_t::IsBSDF>::__call(base_t::bxdf.fresnel, base_t::anisointer.getNdotV());
            quant_query_type qq = bxdf::impl::quant_query_helper<ndf_type, fresnel_type, base_t::bxdf_t::IsBSDF>::template __call<aniso_cache>(base_t::bxdf.ndf, _f, cache);
            quant_type DG1 = base_t::bxdf.ndf.template DG1<sample_t, aniso_interaction>(dq, qq, s, base_t::anisointer);
            dg1 = DG1.microfacetMeasure * hlsl::abs(cache.getVdotH() / base_t::anisointer.getNdotV());
            NdotH = cache.getAbsNdotH();
        }
        else
        {
            dg1_query_type dq = base_t::bxdf.ndf.template createDG1Query<iso_interaction, iso_cache>(base_t::isointer, isocache);
            fresnel_type _f = bxdf::impl::getOrientedFresnel<fresnel_type, base_t::bxdf_t::IsBSDF>::__call(base_t::bxdf.fresnel, base_t::isointer.getNdotV());
            quant_query_type qq = bxdf::impl::quant_query_helper<ndf_type, fresnel_type, base_t::bxdf_t::IsBSDF>::template __call<iso_cache>(base_t::bxdf.ndf, _f, isocache);
            quant_type DG1 = base_t::bxdf.ndf.template DG1<sample_t, iso_interaction>(dq, qq, s, base_t::isointer);
            dg1 = DG1.microfacetMeasure * hlsl::abs(isocache.getVdotH() / base_t::isointer.getNdotV());
            NdotH = isocache.getAbsNdotH();
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

        // get jacobian
        float32_t2x2 m = float32_t2x2(
            sx.getTdotL() - s.getTdotL(), sy.getTdotL() - s.getTdotL(),
            sx.getBdotL() - s.getBdotL(), sy.getBdotL() - s.getBdotL()
        );
        float det = nbl::hlsl::determinant<float32_t2x2>(m) / (eps * eps);
        
        float jacobi_dg1_ndoth = det * dg1 * NdotH;
        if (!checkZero<float>(jacobi_dg1_ndoth - 1.f, 1e-4))
        {
#ifndef __HLSL_VERSION
            if (verbose)
                base_t::errMsg += std::format("VdotH={}, LdotH={}, Jacobian={}, DG1={}, NdotH={}, Jacobian*DG1*NdotH={}",
                                        aniso ? cache.getVdotH() : isocache.getVdotH(), aniso ? cache.getLdotH() : isocache.getLdotH(),
                                        det, dg1, NdotH, jacobi_dg1_ndoth);
#endif
            return BET_JACOBIAN;
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

    float eps = 1e-5;
    sample_t s, sx, sy;
    aniso_cache cache;
    iso_cache isocache;
    float dg1, NdotH;
    bool verbose;
};

#ifndef __HLSL_VERSION
template<class BxDF, bool aniso = false>
struct TestCTGenerateH : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestCTGenerateH<BxDF, aniso>;
    using traits_t = bxdf::traits<BxDF>;

    virtual ErrorType compute() override
    {
        counter.reset();

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

            bool transmitted;
            float NdotV, VdotH;
            NBL_IF_CONSTEXPR(aniso)
            {
                NdotV = base_t::anisointer.getNdotV();
                VdotH = cache.getVdotH();
                transmitted = cache.isTransmission();
            }
            else
            {
                NdotV = base_t::isointer.getNdotV();
                VdotH = isocache.getVdotH();
                transmitted = isocache.isTransmission();
            }

            if (!(NdotV * VdotH >= 0.f))
            {
                if (immediateFail)
                {
                    base_t::errMsg += std::format("first failed case: u=[{},{},{}] NdotV={}, VdotH={}", u.x, u.y, u.z, NdotV, VdotH);
                    return BET_GENERATE_H;
                }
                else
                {
                    counter.fail++;
                    transmitted ? counter.transmitted++ : counter.reflected++;
                }
            }

            counter.total++;
        }

        if (counter.fail > 0)
        {
            base_t::errMsg += std::format("fail count={} out of {} valid samples: {} are transmitted, {} reflected, alpha=[{},{}]",
                                counter.fail, counter.total, counter.transmitted, counter.reflected, base_t::rc.alpha.x, base_t::rc.alpha.y);
            return BET_GENERATE_H;
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
        t.immediateFail = initparams.immediateFail;
        t.initBxDF(t.rc);
        
        ErrorType e = t.test();
        if (e != BET_NONE)
            cb.__call(e, t, initparams.logInfo);
    }

    struct Counter
    {
        uint32_t fail;
        uint32_t reflected;
        uint32_t transmitted;
        uint32_t total;

        void reset()
        {
            fail = 0;
            reflected = 0;
            transmitted = 0;
            total = 0;
        }
    };

    bool immediateFail = false;
    uint32_t numSamples = 1000000;
    Counter counter;
};
#endif

}
}

#endif