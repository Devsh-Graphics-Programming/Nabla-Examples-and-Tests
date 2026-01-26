#ifndef BXDFTESTS_TESTS_H
#define BXDFTESTS_TESTS_H
// cpp only tests

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <format>
#include <functional>

#include "app_resources/tests_common.hlsl"
#include "nbl/builtin/hlsl/visualization/turbo.hlsl"
#include "nbl/builtin/hlsl/math/quadrature/adaptive_simpson.hlsl"

template<class BxDF, bool aniso = false>
struct TestModifiedWhiteFurnace : TestBxDF<BxDF>
{
    using base_t = TestBxDFBase<BxDF>;
    using this_t = TestModifiedWhiteFurnace<BxDF, aniso>;
    using traits_t = bxdf::traits<BxDF>;

    TestResult compute()
    {
        accumulatedQuotient = float32_t3(0.f, 0.f, 0.f);

        aniso_cache cache;
        iso_cache isocache;

        sample_t s;
        quotient_pdf_t sampledLi;

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
                {
                    s = base_t::bxdf.generate(base_t::anisointer, u.xy, cache);
                }
                else
                {
                    s = base_t::bxdf.generate(base_t::isointer, u.xy, isocache);
                }
            }
            NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BSDF && !traits_t::IsMicrofacet)
            {
                s = base_t::bxdf.generate(base_t::anisointer, u);
            }
            NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BSDF && traits_t::IsMicrofacet)
            {
                NBL_IF_CONSTEXPR(aniso)
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

            NBL_IF_CONSTEXPR(!traits_t::IsMicrofacet)
            {
                sampledLi = base_t::bxdf.quotient_and_pdf(s, base_t::isointer);
            }
            NBL_IF_CONSTEXPR(traits_t::IsMicrofacet)
            {
                NBL_IF_CONSTEXPR(aniso)
                {
                    sampledLi = base_t::bxdf.quotient_and_pdf(s, base_t::anisointer, cache);
                }
                else
                {
                    sampledLi = base_t::bxdf.quotient_and_pdf(s, base_t::isointer, isocache);
                }
            }

            if (hlsl::isinf(sampledLi.pdf))
                accumulatedQuotient += sampledLi.quotient;
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

        if (nbl::hlsl::any<hlsl::vector<bool, 3> >(accumulatedQuotient > hlsl::promote<float32_t3>(1.f)))
        {
            base_t::errMsg += std::format("({}, {}, {})", accumulatedQuotient.x, accumulatedQuotient.y, accumulatedQuotient.z);
            return BTR_ERROR_QUOTIENT_SUM_TOO_LARGE;
        }

        return BTR_NONE;
    }

    static void run(NBL_CONST_REF_ARG(STestInitParams) initparams, NBL_REF_ARG(FailureCallback<this_t>) cb)
    {
        this_t t;
        t.init(initparams.halfSeed);
        t.rc.halfSeed = initparams.halfSeed;
        t.numSamples = initparams.samples;
        t.initBxDF(t.rc);

        TestResult e = t.test();
        if (e != BTR_NONE)
            cb.__call(e, t, initparams.logInfo);
    }

    uint32_t numSamples;
    float32_t3 accumulatedQuotient;
};

template<class BxDF, bool aniso>
struct CalculatePdfSinTheta
{
    using traits_t = bxdf::traits<BxDF>;

    float operator()(float theta, float phi) NBL_CONST_MEMBER_FUNC
    {
        float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
        float cosPhi = std::cos(phi), sinPhi = std::sin(phi);

        ray_dir_info_t L;
        L.direction = hlsl::normalize(float32_t3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));
        float32_t3 N = anisointer.getN();
        float NdotL = hlsl::dot(N, L.direction);

        const float32_t3 T = anisointer.getT();
        const float32_t3 B = anisointer.getB();
        sample_t s = sample_t::create(L, T, B, NdotL);
        aniso_cache cache;

        float tmpeta = 1.f;
        NBL_IF_CONSTEXPR(traits_t::IsMicrofacet)
        {
            const float NdotV = anisointer.getNdotV();
            NBL_IF_CONSTEXPR(traits_t::type == bxdf::BT_BRDF)
                if (NdotV < 0.f) return 0.f;

            const float NdotL = s.getNdotL();
            if (NdotV * NdotL < 0.f)
                tmpeta = NdotV < 0.f ? 1.f / eta : eta;
            float32_t3 H = hlsl::normalize(V.getDirection() + L.getDirection() * tmpeta);
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

            if (!cache.isValid(bxdf::fresnel::OrientedEtas<hlsl::vector<float, 1> >::create(1.f, hlsl::promote<hlsl::vector<float, 1> >(tmpeta))))
                return 0.f;

            cache.fillTangents(T, B, H);
        }

        float pdf;
        NBL_IF_CONSTEXPR(!traits_t::IsMicrofacet)
        {
            pdf = bxdf.pdf(s, isointer);
        }
        NBL_IF_CONSTEXPR(traits_t::IsMicrofacet)
        {
            NBL_IF_CONSTEXPR(aniso)
            {
                pdf = bxdf.pdf(s, anisointer, cache);
            }
            else
            {
                pdf = bxdf.pdf(s, isointer, cache.iso_cache);
            }
        }

        return pdf * sinTheta;
    }

    BxDF bxdf;
    ray_dir_info_t V;
    iso_interaction isointer;
    aniso_interaction anisointer;
    float eta;
};

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

    double chi2CDF(double x, int dof)
    {
        if (dof < 1 || x < 0)
        {
            return 0.0;
        }
        else if (dof == 2)
        {
            return 1.0 - hlsl::exp(-0.5 * x);
        }
        else
        {
            return hlsl::gamma(0.5 * dof, 0.5 * x);
        }
    }

    enum WriteFrequenciesToEXR : uint16_t
    {
        WFE_DONT_WRITE = 0,
        WFE_WRITE_ERRORS = 1,
        WFE_WRITE_ALL = 2
    };

    TestResult compute()
    {
        clearBuckets();

        float thetaFactor = thetaSplits * numbers::inv_pi<float>;
        float phiFactor = phiSplits * 0.5f * numbers::inv_pi<float>;

        uint32_t numObservedSamples = 0u;
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
                if NBL_CONSTEXPR_FUNC(aniso)
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
            numObservedSamples++;

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
                CalculatePdfSinTheta<BxDF, aniso> pdfSinTheta;
                pdfSinTheta.bxdf = base_t::bxdf;
                pdfSinTheta.V = base_t::rc.V;
                pdfSinTheta.isointer = base_t::isointer;
                pdfSinTheta.anisointer = base_t::anisointer;
                pdfSinTheta.eta = base_t::rc.eta.x;
                integrateFreq[intidx++] = numObservedSamples * math::quadrature::AdaptiveSimpson2D<CalculatePdfSinTheta<BxDF, aniso>, float>::__call(
                    pdfSinTheta, float32_t2(i * thetaFactor, j * phiFactor), float32_t2((i + 1) * thetaFactor, (j + 1) * phiFactor));

                if (write_frequencies && maxIntFreq < integrateFreq[lastidx])
                    maxIntFreq = integrateFreq[lastidx];
            }
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
                    return BTR_PRINT_MSG;
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
            return BTR_PRINT_MSG;
        }

        float pval = 1.0f - static_cast<float>(chi2CDF(chsq, dof));
        float alpha = 1.0f - std::pow(1.0f - threshold, 1.0f / numTests);

        if (pval < alpha || !std::isfinite(pval))
        {
            base_t::errMsg = std::format("chi2 test: rejected the null hypothesis (p-value = {:.3f}, significance level = {:.3f}", pval, alpha);
            return BTR_PRINT_MSG;
        }

        return BTR_NONE;
    }

    static void run(NBL_CONST_REF_ARG(STestInitParams) initparams, NBL_REF_ARG(FailureCallback<this_t>) cb)
    {
        this_t t;
        t.init(initparams.halfSeed);
        t.rc.halfSeed = initparams.halfSeed;
        t.numSamples = initparams.samples;
        t.thetaSplits = initparams.thetaSplits;
        t.phiSplits = initparams.phiSplits;
        t.write_frequencies = static_cast<WriteFrequenciesToEXR>(initparams.writeFrequencies);
        t.initBxDF(t.rc);

        TestResult e = t.test();
        if (e != BTR_NONE)
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
    
    WriteFrequenciesToEXR write_frequencies;
    float maxCountFreq;
    float maxIntFreq;

    std::vector<float> countFreq;
    std::vector<float> integrateFreq;
};

#endif
