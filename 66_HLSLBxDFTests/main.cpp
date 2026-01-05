#include <nabla.h>
#include <fstream>
#include <iomanip>
#include <ranges>
#include <execution>

#include "nbl/examples/examples.hpp"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;
using namespace nbl::hlsl;
using namespace nbl::examples;

#include "app_resources/test_components.hlsl"
#include "app_resources/tests.hlsl"
#include "tests.h"
#include "nbl/builtin/hlsl/math/angle_adding.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf.hlsl"
#include "nbl/builtin/hlsl/bxdf/fresnel.hlsl"

#include "nlohmann/json.hpp"

using json = nlohmann::json;

#define FOR_EACH_BEGIN_EX(r, ex) std::for_each(ex, r.begin(), r.end(), [&](uint32_t i) {
#define FOR_EACH_BEGIN(r) std::for_each(std::execution::par_unseq, r.begin(), r.end(), [&](uint32_t i) {
#define FOR_EACH_END });

#define RUN_TEST_OF_TYPE(TEST_TYPE, INIT_PARAMS) {\
    PrintFailureCallback<BOOST_PP_REMOVE_PARENS(TEST_TYPE)> cb;\
    cb.logger = m_logger;\
    BOOST_PP_REMOVE_PARENS(TEST_TYPE)::run(INIT_PARAMS, cb);\
}\

#define RUN_CHI2_TEST_WRITE_EXR(TEST_TYPE, INIT_PARAMS) {\
    PrintFailureCallback<BOOST_PP_REMOVE_PARENS(TEST_TYPE)> cb;\
    cb.logger = m_logger;\
    BOOST_PP_REMOVE_PARENS(TEST_TYPE) t;\
    t.init(initparams.halfSeed);\
    t.rc.halfSeed = initparams.halfSeed;\
    t.numSamples = initparams.samples;\
    t.thetaSplits = initparams.thetaSplits;\
    t.phiSplits = initparams.phiSplits;\
    t.write_frequencies = static_cast<BOOST_PP_REMOVE_PARENS(TEST_TYPE)::WriteFrequenciesToEXR>(initparams.writeFrequencies);\
    t.initBxDF(t.rc);\
    TestResult e = t.test();\
    if (e != BTR_INVALID_TEST_CONFIG)\
    {\
    if (e != BTR_NONE)\
    {\
        if (initparams.writeFrequencies >= BOOST_PP_REMOVE_PARENS(TEST_TYPE)::WFE_WRITE_ERRORS)\
            writeToEXR<BOOST_PP_REMOVE_PARENS(TEST_TYPE)>(t);\
        cb.__call(e, t, initparams.logInfo);\
    }\
    else if (initparams.writeFrequencies == BOOST_PP_REMOVE_PARENS(TEST_TYPE)::WFE_WRITE_ALL)\
            writeToEXR<BOOST_PP_REMOVE_PARENS(TEST_TYPE)>(t);\
    }\
}\

class HLSLBxDFTests final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
    using device_base_t = application_templates::MonoDeviceApplication;
    using asset_base_t = BuiltinResourcesApplication;

public:
    HLSLBxDFTests(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

    inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;

        if (!asset_base_t::onAppInitialized(std::move(system)))
            return false;

        std::ifstream f("../app_resources/config.json");
        if (f.fail())
        {
            m_logger->log("could not open config file", ILogger::ELL_ERROR);
            return false;
        }
        try
        {
            testconfigs = json::parse(f);
        }
        catch (json::parse_error& ex)
        {
            m_logger->log("parse_error.%d failed to parse config file at byte %u: %s", ILogger::ELL_ERROR, ex.id, ex.byte, ex.what());
            return false;
        }

        // test compile with dxc
        {
            IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = m_logger.get();
            lp.workingDirectory = "app_resources"; // virtual root
            auto key = nbl::this_example::builtin::build::get_spirv_key<"shader">(m_device.get());
            auto bundle = m_assetMgr->getAsset(key.c_str(), lp);

            const auto assets = bundle.getContents();
            if (assets.empty())
                m_logger->log("Could not load shader!", ILogger::ELL_ERROR);

            // Cast down the asset to its proper type
            auto shader = IAsset::castDown<IShader>(assets[0]);

            if (!shader)
                m_logger->log("compile shader test failed!", ILogger::ELL_ERROR);
        }

        // test concepts, not comprehensive
        static_assert(bxdf::surface_interactions::Isotropic<iso_interaction>);
        static_assert(bxdf::surface_interactions::Isotropic<aniso_interaction>);
        static_assert(bxdf::surface_interactions::Anisotropic<aniso_interaction>);

        static_assert(bxdf::CreatableIsotropicMicrofacetCache<iso_cache>);
        static_assert(bxdf::ReadableIsotropicMicrofacetCache<aniso_cache>);
        static_assert(bxdf::AnisotropicMicrofacetCache<aniso_cache>);

        using ndf_beckmann_t = bxdf::ndf::Beckmann<float, false, bxdf::ndf::MTT_REFLECT>;
        static_assert(bxdf::ndf::NDF<ndf_beckmann_t>);
        using ndf_ggx_t = bxdf::ndf::GGX<float, true, bxdf::ndf::MTT_REFLECT_REFRACT>;
        static_assert(bxdf::ndf::NDF<ndf_ggx_t>);

        using fresnel_schlick_t = bxdf::fresnel::Schlick<float32_t3>;
        static_assert(bxdf::fresnel::Fresnel<fresnel_schlick_t>);

        static_assert(bxdf::bxdf_concepts::IsotropicBxDF<bxdf::reflection::SLambertian<iso_config_t>>);
        static_assert(bxdf::bxdf_concepts::IsotropicBxDF<bxdf::reflection::SOrenNayar<iso_config_t>>);
        static_assert(bxdf::bxdf_concepts::IsotropicBxDF<bxdf::transmission::SSmoothDielectric<iso_config_t>>);

        static_assert(bxdf::bxdf_concepts::MicrofacetBxDF<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>>);
        static_assert(bxdf::bxdf_concepts::MicrofacetBxDF<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>>);
        static_assert(bxdf::bxdf_concepts::MicrofacetBxDF<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>>);
        static_assert(bxdf::bxdf_concepts::MicrofacetBxDF<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>>);

        runTests();

        return true;
    }

    void workLoopBody() override {}

    bool keepRunning() override { return false; }

    bool onAppTerminated() override
    {
        return device_base_t::onAppTerminated();
    }

private:
    template<class TestT>
    struct PrintFailureCallback : FailureCallback<TestT>
    {
        void __call(TestResult error, NBL_REF_ARG(TestT) failedFor, bool logInfo) override
        {
            switch (error)
            {
            case BTR_INVALID_TEST_CONFIG:
                if (logInfo)
                    logger->log("seed %u: %s skipping test due to invalid NdotV/NdotL config", ILogger::ELL_INFO, failedFor.rc.halfSeed, failedFor.name.c_str());
                break;
            case BTR_ERROR_NEGATIVE_VAL:
                logger->log("seed %u: %s pdf/quotient/eval < 0", ILogger::ELL_ERROR, failedFor.rc.halfSeed, failedFor.name.c_str());
                break;
            case BTR_ERROR_GENERATED_SAMPLE_NON_POSITIVE_PDF:
                logger->log("seed %u: %s generated sample has pdf = 0", ILogger::ELL_ERROR, failedFor.rc.halfSeed, failedFor.name.c_str());
                break;
            case BTR_ERROR_QUOTIENT_INF:
                logger->log("seed %u: %s quotient -> inf", ILogger::ELL_ERROR, failedFor.rc.halfSeed, failedFor.name.c_str());
                break;
            case BTR_ERROR_JACOBIAN_TEST_FAIL:
                logger->log("seed %u: %s failed the jacobian * pdf test    %s", ILogger::ELL_ERROR, failedFor.rc.halfSeed, failedFor.name.c_str(), failedFor.errMsg.c_str());
                break;
            case BTR_ERROR_PDF_EVAL_DIFF:
                logger->log("seed %u: %s quotient * pdf != eval    %s", ILogger::ELL_ERROR, failedFor.rc.halfSeed, failedFor.name.c_str(), failedFor.errMsg.c_str());
                break;
            case BTR_ERROR_NO_RECIPROCITY:
                logger->log("seed %u: %s failed the reciprocity test    %s", ILogger::ELL_ERROR, failedFor.rc.halfSeed, failedFor.name.c_str(), failedFor.errMsg.c_str());
                break;
            case BTR_ERROR_REFLECTANCE_OUT_OF_RANGE:
                logger->log("seed %u: %s reflectance not between 0 and 1    %s", ILogger::ELL_ERROR, failedFor.rc.halfSeed, failedFor.name.c_str(), failedFor.errMsg.c_str());
                break;
            case BTR_PRINT_MSG:
                logger->log("seed %u: %s error message    %s", ILogger::ELL_ERROR, failedFor.rc.halfSeed, failedFor.name.c_str(), failedFor.errMsg.c_str());
                break;
            case BTR_ERROR_GENERATED_H_INVALID:
                logger->log("seed %u: %s failed invalid H configuration generated    %s", ILogger::ELL_WARNING, failedFor.rc.halfSeed, failedFor.name.c_str(), failedFor.errMsg.c_str());
                break;
            default:
                logger->log("seed %u: %s unknown error", ILogger::ELL_ERROR, failedFor.rc.halfSeed, failedFor.name.c_str());
            }

#ifdef _NBL_DEBUG
            for (volatile bool repeat = true; IsDebuggerPresent() && repeat && error < BTR_NOBREAK; )
            {
                repeat = false;
                _NBL_DEBUG_BREAK_IF(true);
                failedFor.compute();
            }
#endif
        }

        smart_refctd_ptr<ILogger> logger;
    };

    void runTests()
    {
        const bool logInfo = testconfigs["logInfo"];

        // test jacobian * pdf
        uint32_t runs = testconfigs["TestJacobian"]["runs"];
        auto rJacobian = std::ranges::views::iota(0u, runs);
        FOR_EACH_BEGIN(rJacobian)
            STestInitParams initparams{ .logInfo = logInfo };
            initparams.halfSeed = i;
            initparams.verbose = testconfigs["TestJacobian"]["verbose"];

            RUN_TEST_OF_TYPE((TestJacobian<bxdf::reflection::SLambertian<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::reflection::SOrenNayar<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::reflection::SDeltaDistribution<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::reflection::SIridescent<iso_microfacet_config_t>, false>), initparams);

            RUN_TEST_OF_TYPE((TestJacobian<bxdf::transmission::SLambertian<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::transmission::SOrenNayar<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::transmission::SSmoothDielectric<iso_config_t> >), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::transmission::SThinSmoothDielectric<iso_config_t> >), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::transmission::SDeltaDistribution<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestJacobian<bxdf::transmission::SIridescent<iso_microfacet_config_t>, false>), initparams);
        FOR_EACH_END


        // test reciprocity
        runs = testconfigs["TestReciprocity"]["runs"];
        auto rReciprocity = std::ranges::views::iota(0u, runs);
        FOR_EACH_BEGIN(rReciprocity)
            STestInitParams initparams{ .logInfo = logInfo };
            initparams.halfSeed = i;
            initparams.verbose = testconfigs["TestReciprocity"]["verbose"];

            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::reflection::SLambertian<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::reflection::SOrenNayar<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::reflection::SDeltaDistribution<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::reflection::SIridescent<iso_microfacet_config_t>, false>), initparams);

            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::transmission::SLambertian<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::transmission::SOrenNayar<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::transmission::SSmoothDielectric<iso_config_t> >), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::transmission::SThinSmoothDielectric<iso_config_t> >), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::transmission::SDeltaDistribution<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::transmission::SBeckmannDielectricIsotropic<rectest_iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::transmission::SBeckmannDielectricAnisotropic<rectest_aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::transmission::SGGXDielectricIsotropic<rectest_iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::transmission::SGGXDielectricAnisotropic<rectest_aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestReciprocity<bxdf::transmission::SIridescent<rectest_iso_microfacet_config_t>, false>), initparams);
        FOR_EACH_END


        // test buckets of inf
        runs = testconfigs["TestBucket"]["runs"];
        auto rBucket = std::ranges::views::iota(0u, runs);
        FOR_EACH_BEGIN(rBucket)
            STestInitParams initparams{ .logInfo = logInfo };
            initparams.halfSeed = i;
            initparams.samples = testconfigs["TestBucket"]["samples"];

            RUN_TEST_OF_TYPE((TestBucket<bxdf::reflection::SLambertian<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::reflection::SOrenNayar<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::reflection::SIridescent<iso_microfacet_config_t>, false>), initparams);

            RUN_TEST_OF_TYPE((TestBucket<bxdf::transmission::SLambertian<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::transmission::SOrenNayar<iso_config_t>>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestBucket<bxdf::transmission::SIridescent<iso_microfacet_config_t>, false>), initparams);
        FOR_EACH_END


        // chi2 test for sampling and pdf
        runs = testconfigs["TestChi2"]["runs"];
        auto rChi2 = std::ranges::views::iota(0u, runs);
        FOR_EACH_BEGIN_EX(rChi2, std::execution::par_unseq)
            STestInitParams initparams{ .logInfo = logInfo };
            initparams.halfSeed = i;
            initparams.samples = testconfigs["TestChi2"]["samples"];
            initparams.thetaSplits = testconfigs["TestChi2"]["thetaSplits"];
            initparams.phiSplits = testconfigs["TestChi2"]["phiSplits"];
            initparams.writeFrequencies = testconfigs["TestChi2"]["writeFrequencies"];

            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::reflection::SLambertian<iso_config_t>>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::reflection::SOrenNayar<iso_config_t>>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::reflection::SIridescent<iso_microfacet_config_t>, false>), initparams);

            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::transmission::SLambertian<iso_config_t>>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::transmission::SOrenNayar<iso_config_t>>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_CHI2_TEST_WRITE_EXR((TestChi2<bxdf::transmission::SIridescent<iso_microfacet_config_t>, false>), initparams);
        FOR_EACH_END

        // testing ndf jacobian * dg1, ONLY for cook torrance bxdfs
        runs = testconfigs["TestNDF"]["runs"];
        auto rNdf = std::ranges::views::iota(0u, runs);
        FOR_EACH_BEGIN(rNdf)
            STestInitParams initparams{ .logInfo = logInfo };
            initparams.halfSeed = i;
            initparams.verbose = testconfigs["TestNDF"]["verbose"];

            RUN_TEST_OF_TYPE((TestNDF<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestNDF<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestNDF<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestNDF<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>, true>), initparams);

            RUN_TEST_OF_TYPE((TestNDF<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestNDF<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestNDF<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestNDF<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>, true>), initparams);
        FOR_EACH_END

        // test generated H that NdotV*VdotH>=0.0, VdotL calculation
        runs = testconfigs["TestCTGenerateH"]["runs"];
        auto rGenerateH = std::ranges::views::iota(0u, runs);
        FOR_EACH_BEGIN_EX(rGenerateH, std::execution::par_unseq)
            STestInitParams initparams{ .logInfo = logInfo };
            initparams.halfSeed = i;
            initparams.samples = testconfigs["TestCTGenerateH"]["samples"];
            initparams.immediateFail = testconfigs["TestCTGenerateH"]["immediateFail"];

            RUN_TEST_OF_TYPE((TestCTGenerateH<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestCTGenerateH<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestCTGenerateH<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestCTGenerateH<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>, true>), initparams);

            RUN_TEST_OF_TYPE((TestCTGenerateH<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestCTGenerateH<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>, true>), initparams);
            RUN_TEST_OF_TYPE((TestCTGenerateH<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>, false>), initparams);
            RUN_TEST_OF_TYPE((TestCTGenerateH<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>, true>), initparams);
        FOR_EACH_END

            // test arccos angle sums
        {
            Xoroshiro64Star rng = Xoroshiro64Star::construct(uint32_t2(4, 2));
            math::sincos_accumulator<float> angle_adder;

            auto Sin = [&](const float cosA) -> float
            {
                return nbl::hlsl::sqrt(1.f - cosA * cosA);
            };

            for (uint32_t i = 0; i < 10; i++)
            {
                const float a = ConvertToFloat01<uint32_t>::__call(rng()) * 2.f - 1.f;
                const float b = ConvertToFloat01<uint32_t>::__call(rng()) * 2.f - 1.f;
                const float c = ConvertToFloat01<uint32_t>::__call(rng()) * 2.f - 1.f;
                const float d = ConvertToFloat01<uint32_t>::__call(rng()) * 2.f - 1.f;

                const float exAB = acos(a) + acos(b);
                angle_adder = math::sincos_accumulator<float>::create(a, Sin(a));
                angle_adder.addAngle(b, Sin(b));
                float res = angle_adder.getSumofArccos();
                if (!checkEq<float>(res, exAB, 1e-3))
                    fprintf(stderr, "[ERROR] angle adding (2 angles) failed! expected %f, got %f\n", exAB, res);

                const float exABCD = exAB + acos(c) + acos(d);
                angle_adder = math::sincos_accumulator<float>::create(a, Sin(a));
                angle_adder.addAngle(b, Sin(b));
                angle_adder.addAngle(c, Sin(c));
                angle_adder.addAngle(d, Sin(d));
                res = angle_adder.getSumofArccos();
                if (!checkEq<float>(res, exABCD, 1e-3))
                    fprintf(stderr, "[ERROR] angle adding (4 angles) failed! expected %f, got %f\n", exABCD, res);
            }
        }
    }

    template<class Chi2Test>
    static smart_refctd_ptr<ICPUImage> writeToCPUImage(const Chi2Test& test)
    {
        const uint32_t totalWidth = test.phiSplits;
        const uint32_t totalHeight = 2 * test.thetaSplits;
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
        for (uint64_t j = 0u; j < test.thetaSplits; ++j)
            for (uint64_t i = 0u; i < test.phiSplits; ++i)
            {
                float32_t3 pixelColor = hlsl::visualization::Turbo::map(test.countFreq[j * test.phiSplits + i] / test.maxCountFreq);
                double decodedPixel[4] = { pixelColor[0], pixelColor[1], pixelColor[2], 1 };

                const uint64_t pixelIndex = j * test.phiSplits + i;
                asset::encodePixelsRuntime(format, bytePtr + pixelIndex * asset::getTexelOrBlockBytesize(format), decodedPixel);
            }

        // write values of pdf, bottom half
        for (uint64_t j = 0u; j < test.thetaSplits; ++j)
            for (uint64_t i = 0u; i < test.phiSplits; ++i)
            {
                float32_t3 pixelColor = hlsl::visualization::Turbo::map(test.integrateFreq[j * test.phiSplits + i] / test.maxIntFreq);
                double decodedPixel[4] = { pixelColor[0], pixelColor[1], pixelColor[2], 1 };

                const uint64_t pixelIndex = (test.thetaSplits + j) * test.phiSplits + i;
                asset::encodePixelsRuntime(format, bytePtr + pixelIndex * asset::getTexelOrBlockBytesize(format), decodedPixel);
            }

        return image;
    }

    template<class Chi2Test>
    void writeToEXR(const Chi2Test& test)
    {
        std::string filename = std::format("chi2test_{}_{}.exr", test.rc.halfSeed, test.name);

        auto cpuImage = writeToCPUImage<Chi2Test>(test);
        ICPUImageView::SCreationParams imgViewParams;
        imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
        imgViewParams.format = cpuImage->getCreationParameters().format;
        imgViewParams.image = smart_refctd_ptr(cpuImage);
        imgViewParams.viewType = ICPUImageView::ET_2D;
        imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
        smart_refctd_ptr<nbl::asset::ICPUImageView> imageView = ICPUImageView::create(std::move(imgViewParams));

        IAssetWriter::SAssetWriteParams wp(imageView.get());
        m_assetMgr->writeAsset(filename, wp);
    }

    json testconfigs;
};

NBL_MAIN_FUNC(HLSLBxDFTests)
