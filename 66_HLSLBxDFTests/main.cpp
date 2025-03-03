#include <nabla.h>
#include <fstream>
#include <iomanip>
#include <ranges>
#include <execution>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include "nbl/system/CColoredStdoutLoggerANSI.h"
#include "nbl/system/IApplicationFramework.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;
using namespace nbl::hlsl;

#include "app_resources/tests.hlsl"

struct PrintFailureCallback : FailureCallback
{
    void __call(ErrorType error, NBL_REF_ARG(TestBase) failedFor, bool logInfo) override
    {
        switch (error)
        {
        case BET_INVALID:
            if (logInfo)
                fprintf(stderr, "[INFO] seed %u: %s skipping test due to invalid NdotV/NdotL config\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_NEGATIVE_VAL:
            fprintf(stderr, "[ERROR] seed %u: %s pdf/quotient/eval < 0\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_PDF_ZERO:
            fprintf(stderr, "[ERROR] seed %u: %s pdf = 0\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_QUOTIENT_INF:
            fprintf(stderr, "[ERROR] seed %u: %s quotient -> inf\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_JACOBIAN:
            fprintf(stderr, "[ERROR] seed %u: %s failed the jacobian * pdf test\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_PDF_EVAL_DIFF:
            fprintf(stderr, "[ERROR] seed %u: %s quotient * pdf != eval\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_RECIPROCITY:
            fprintf(stderr, "[ERROR] seed %u: %s failed the reciprocity test\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_PRINT_MSG:
            fprintf(stderr, "[ERROR] seed %u: %s error message\n%s\n", failedFor.rc.state, failedFor.name.c_str(), failedFor.errMsg.c_str());
            break;
        default:
            fprintf(stderr, "[ERROR] seed %u: %s unknown error\n", failedFor.rc.state, failedFor.name.c_str());
        }

        // TODO: #ifdef NBL_ENABLE_DEBUGBREAK
        for (volatile bool repeat = true; IsDebuggerPresent() && repeat && error < BET_NOBREAK; )
        {
            repeat = false;
            __debugbreak();
            failedFor.compute();
        }
    }
};

#define FOR_EACH_BEGIN(r) std::for_each(std::execution::par_unseq, r.begin(), r.end(), [&](uint32_t i) {
#define FOR_EACH_END });

int main(int argc, char** argv)
{
    std::cout << std::fixed << std::setprecision(4);

    std::ifstream f("../app_resources/config.json");
    if (f.fail())
    {
        fprintf(stderr, "[ERROR] could not open config file\n");
        return -1;
    }
    json testconfigs;
    try
    {
        testconfigs = json::parse(f);
    }
    catch (json::parse_error& ex)
    {
        fprintf(stderr, "[ERROR] parse_error.%d failed to parse config file at byte %u: %s\n", ex.id, ex.byte, ex.what());
        return -1;
    }

    // test compile with dxc
    {
        smart_refctd_ptr<system::ISystem> m_system = system::IApplicationFramework::createSystem();
        smart_refctd_ptr<system::ILogger> m_logger = core::make_smart_refctd_ptr<system::CColoredStdoutLoggerANSI>(system::ILogger::DefaultLogMask());
        m_logger->log("Logger Created!", system::ILogger::ELL_INFO);
        smart_refctd_ptr<asset::IAssetManager> m_assetMgr = make_smart_refctd_ptr<asset::IAssetManager>(smart_refctd_ptr(m_system));

        path CWD = system::path(argv[0]).parent_path().generic_string() + "/";
        path localInputCWD = CWD / "../";
        auto resourceArchive =
#ifdef NBL_EMBED_BUILTIN_RESOURCES
            make_smart_refctd_ptr<nbl::this_example::builtin::CArchive>(smart_refctd_ptr(m_logger));
#else
            make_smart_refctd_ptr<system::CMountDirectoryArchive>(localInputCWD/"app_resources", smart_refctd_ptr(m_logger), m_system.get());
#endif
        m_system->mount(std::move(resourceArchive), "app_resources");

        constexpr uint32_t WorkgroupSize = 256;
        const std::string WorkgroupSizeAsStr = std::to_string(WorkgroupSize);
        const std::string filePath = "app_resources/test_compile.comp.hlsl";

        IAssetLoader::SAssetLoadParams lparams = {};
        lparams.logger = m_logger.get();
        lparams.workingDirectory = "";
        auto bundle = m_assetMgr->getAsset(filePath, lparams);
        if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
        {
            m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
            exit(-1);
        }

        const auto assets = bundle.getContents();
        assert(assets.size() == 1);
        smart_refctd_ptr<ICPUShader> shaderSrc = IAsset::castDown<ICPUShader>(assets[0]);

        smart_refctd_ptr<ICPUShader> shader = shaderSrc;
        auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(m_system));
        CHLSLCompiler::SOptions options = {};
        options.stage = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE;
        options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
        options.spirvOptimizer = nullptr;
        // if you don't set the logger and source identifier you'll have no meaningful errors
        options.preprocessorOptions.sourceIdentifier = shaderSrc->getFilepathHint();
        options.preprocessorOptions.logger = m_logger.get();
        options.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder();
        const IShaderCompiler::SMacroDefinition WorkgroupSizeDefine = { "WORKGROUP_SIZE", WorkgroupSizeAsStr };
        options.preprocessorOptions.extraDefines = { &WorkgroupSizeDefine,&WorkgroupSizeDefine + 1 };
        if (!(shader = compiler->compileToSPIRV((const char*)shaderSrc->getContent()->getPointer(), options)))
            fprintf(stderr, "[ERROR] compile shader test failed!\n");
    }


    const bool logInfo = testconfigs["logInfo"];
    PrintFailureCallback cb;

    // test jacobian * pdf
    uint32_t runs = testconfigs["TestJacobian"]["runs"];
    auto rJacobian = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN(rJacobian)
    STestInitParams initparams{ .logInfo = logInfo };
    initparams.state = i;

    TestJacobian<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(initparams, cb);

    TestJacobian<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, false>>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(initparams, cb);
    FOR_EACH_END


    // test reciprocity
    runs = testconfigs["TestReciprocity"]["runs"];
    auto rReciprocity = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN(rReciprocity)
    STestInitParams initparams{ .logInfo = logInfo };
    initparams.state = i;

    TestReciprocity<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);

    TestReciprocity<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, false>>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    FOR_EACH_END


    // test buckets of inf
    // NOTE: can safely ignore any errors for smooth dielectric BxDFs because pdf SHOULD be inf
    runs = testconfigs["TestBucket"]["runs"];
    auto rBucket = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN(rBucket)
    STestInitParams initparams{ .logInfo = logInfo };
    initparams.state = i;
    initparams.samples = testconfigs["TestBucket"]["samples"];

    TestBucket<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestBucket<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestBucket<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestBucket<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestBucket<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestBucket<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);

    TestBucket<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    //TestBucket<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, false>>::run(initparams, cb);
    //TestBucket<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(initparams, cb);
    TestBucket<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestBucket<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestBucket<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestBucket<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    FOR_EACH_END


    // chi2 test for sampling and pdf
    runs = testconfigs["TestChi2"]["runs"];
    auto rChi2 = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN(rChi2)
    STestInitParams initparams{ .logInfo = logInfo };
    initparams.state = i;
    initparams.samples = testconfigs["TestChi2"]["samples"];
    initparams.thetaSplits = testconfigs["TestChi2"]["thetaSplits"];
    initparams.phiSplits = testconfigs["TestChi2"]["phiSplits"];
    initparams.writeFrequencies = testconfigs["TestChi2"]["writeFrequencies"];

    TestChi2<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestChi2<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestChi2<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestChi2<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestChi2<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestChi2<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);

    TestChi2<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestChi2<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, false>>::run(initparams, cb);
    TestChi2<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(initparams, cb);
    TestChi2<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestChi2<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestChi2<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestChi2<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    FOR_EACH_END

        // test arccos angle sums
    {
        Xoroshiro64Star rng = Xoroshiro64Star::construct(uint32_t2(4, 2));
        for (uint32_t i = 0; i < 10; i++)
        {
            const float a = rng() * numbers::pi<float>;
            const float b = rng() * numbers::pi<float>;
            const float c = rng() * numbers::pi<float>;
            const float d = rng() * numbers::pi<float>;

            const float exAB = acos(a) + acos(b);
            float res = math::getSumofArccosAB(a, b);
            if (res != exAB)
                fprintf(stderr, "[ERROR] math::getSumofArccosAB failed! expected %f, got %f\n", exAB, res);

            const float exABCD = exAB + acos(c) + acos(d);
            res = math::getSumofArccosABCD(a, b, c, d);
            if (res != exABCD)
                fprintf(stderr, "[ERROR] math::getSumofArccosABCD failed! expected %f, got %f\n", exABCD, res);
        }
    }

    return 0;
}