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

// because unordered_map -- next time, do fixed size array of atomic offsets and linked lists (for readback and verification on cpu)
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

    TestResult compute()
    {
        clearBuckets();

        aniso_cache cache;
        iso_cache isocache;

        sample_t s;
        quotient_pdf_t pdf;
        //float32_t3 bsdf;

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
                pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer);
                //bsdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer));
            }
            NBL_IF_CONSTEXPR(traits_t::IsMicrofacet)
            {
                NBL_IF_CONSTEXPR(aniso)
                {
                    pdf = base_t::bxdf.quotient_and_pdf(s, base_t::anisointer, cache);
                    //bsdf = float32_t3(base_t::bxdf.eval(s, base_t::anisointer, cache));
                }
                else
                {
                    pdf = base_t::bxdf.quotient_and_pdf(s, base_t::isointer, isocache);
                    //bsdf = float32_t3(base_t::bxdf.eval(s, base_t::isointer, isocache));
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

        return (base_t::errMsg.length() == 0) ? BTR_NONE : BTR_PRINT_MSG;
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

    smart_refctd_ptr<ICPUImage> writeToCPUImage()
    {
        const uint32_t totalWidth = phiSplits;
        const uint32_t totalHeight = 2 * thetaSplits;
        const auto format = E_FORMAT::EF_R32G32B32A32_SFLOAT;

        IImage::SCreationParams imageParams = {};
        imageParams.type = IImage::E_TYPE::ET_2D;
        imageParams.format = format;
        imageParams.extent = { totalWidth, totalHeight, 1 };
        imageParams.mipLevels = 1;
        imageParams.arrayLayers = 1;
        imageParams.samples = ICPUImage::ESCF_1_BIT;
        imageParams.usage = IImage::EUF_SAMPLED_BIT;

        smart_refctd_ptr<ICPUImage> image = ICPUImage::create(std::move(imageParams));
        assert(image);

        const size_t bufferSize = totalWidth * totalHeight * getTexelOrBlockBytesize(format);
        {
            auto imageRegions = make_refctd_dynamic_array<smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull);
            auto& region = imageRegions->front();
            region.bufferImageHeight = 0u;
            region.bufferOffset = 0ull;
            region.bufferRowLength = totalWidth;
            region.imageExtent = { totalWidth, totalHeight, 1 };
            region.imageOffset = { 0u, 0u, 0u };
            region.imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
            region.imageSubresource.baseArrayLayer = 0u;
            region.imageSubresource.layerCount = 1;
            region.imageSubresource.mipLevel = 0;

            image->setBufferAndRegions(ICPUBuffer::create({ bufferSize }), std::move(imageRegions));
        }

        uint8_t* bytePtr = reinterpret_cast<uint8_t*>(image->getBuffer()->getPointer());

        // write sample count from generate, top half
        for (uint64_t j = 0u; j < thetaSplits; ++j)
            for (uint64_t i = 0; i < phiSplits; ++i)
            {
                //float32_t3 pixelColor = mapColor(countFreq[j * phiSplits + i], 0.f, maxCountFreq);
                float32_t3 pixelColor = hlsl::visualization::Turbo::map(countFreq[j * phiSplits + i] / maxCountFreq);
                double decodedPixel[4] = { pixelColor[0], pixelColor[1], pixelColor[2], 1 };

                const uint64_t pixelIndex = j * phiSplits + i;
                asset::encodePixelsRuntime(format, bytePtr + pixelIndex * asset::getTexelOrBlockBytesize(format), decodedPixel);
            }

        // write values of pdf, bottom half
        for (uint64_t j = 0u; j < thetaSplits; ++j)
            for (uint64_t i = 0; i < phiSplits; ++i)
            {
                //float32_t3 pixelColor = mapColor(integrateFreq[j * phiSplits + i], 0.f, maxIntFreq);
                float32_t3 pixelColor = hlsl::visualization::Turbo::map(integrateFreq[j * phiSplits + i] / maxIntFreq);
                double decodedPixel[4] = { pixelColor[0], pixelColor[1], pixelColor[2], 1 };

                const uint64_t pixelIndex = (thetaSplits + j) * phiSplits + i;
                asset::encodePixelsRuntime(format, bytePtr + pixelIndex * asset::getTexelOrBlockBytesize(format), decodedPixel);
            }

        return image;
    }

    void writeToEXR(asset::IAssetManager* assetManager)
    {
        std::string filename = std::format("chi2test_{}_{}.exr", base_t::rc.halfSeed, base_t::name);

        auto cpuImage = writeToCPUImage();
        ICPUImageView::SCreationParams imgViewParams;
        imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
        imgViewParams.format = cpuImage->getCreationParameters().format;
        imgViewParams.image = smart_refctd_ptr(cpuImage);
        imgViewParams.viewType = ICPUImageView::ET_2D;
        imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
        smart_refctd_ptr<nbl::asset::ICPUImageView> imageView = ICPUImageView::create(std::move(imgViewParams));

        IAssetWriter::SAssetWriteParams wp(imageView.get());
        assetManager->writeAsset(filename, wp);
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

            // TODO: might want to distinguish between invalid H and sample produced below hemisphere?
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

                            if (!cache.isValid(bxdf::fresnel::OrientedEtas<hlsl::vector<float,1> >::create(1.f, hlsl::promote<hlsl::vector<float,1> >(eta))))
                                return 0.f;

                            const float32_t3 T = base_t::anisointer.getT();
                            const float32_t3 B = base_t::anisointer.getB();
                            cache.fillTangents(T, B, H);
                        }

                        float pdf;
                        NBL_IF_CONSTEXPR(!traits_t::IsMicrofacet)
                        {
                            pdf = base_t::bxdf.pdf(s, base_t::isointer);
                        }
                        NBL_IF_CONSTEXPR(traits_t::IsMicrofacet)
                        {
                            NBL_IF_CONSTEXPR(aniso)
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

        return BTR_NONE;
    }

    TestResult test(asset::IAssetManager* assetManager)
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
                    if (write_frequencies == WFE_WRITE_ERRORS)
                        writeToEXR(assetManager);
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
            if (write_frequencies == WFE_WRITE_ERRORS)
                writeToEXR(assetManager);
            base_t::errMsg = std::format("degrees of freedom {} too low", dof);
            return BTR_PRINT_MSG;
        }

        float pval = 1.0f - static_cast<float>(chi2CDF(chsq, dof));
        float alpha = 1.0f - std::pow(1.0f - threshold, 1.0f / numTests);

        if (pval < alpha || !std::isfinite(pval))
        {
            if (write_frequencies == WFE_WRITE_ERRORS)
                writeToEXR(assetManager);
            base_t::errMsg = std::format("chi2 test: rejected the null hypothesis (p-value = {:.3f}, significance level = {:.3f}", pval, alpha);
            return BTR_PRINT_MSG;
        }

        if (write_frequencies == WFE_WRITE_ALL)
            writeToEXR(assetManager);

        return BTR_NONE;
    }

    static void run(NBL_CONST_REF_ARG(STestInitParams) initparams, NBL_REF_ARG(FailureCallback<this_t>) cb, asset::IAssetManager* assetManager)
    {
        this_t t;
        t.init(initparams.halfSeed);
        t.rc.halfSeed = initparams.halfSeed;
        t.numSamples = initparams.samples;
        t.thetaSplits = initparams.thetaSplits;
        t.phiSplits = initparams.phiSplits;
        t.write_frequencies = static_cast<WriteFrequenciesToEXR>(initparams.writeFrequencies);
        t.initBxDF(t.rc);

        TestResult e = t.test(assetManager);
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
