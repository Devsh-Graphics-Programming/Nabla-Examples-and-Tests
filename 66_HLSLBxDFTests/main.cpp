#include <nabla.h>
#include <fstream>
#include <iomanip>
#include <ranges>
#include <execution>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "CArchive.h"
#endif
#include "nbl/system/CColoredStdoutLoggerANSI.h"
#include "nbl/system/IApplicationFramework.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;
using namespace nbl::hlsl;

#include "app_resources/test_components.hlsl"
#include "app_resources/tests.hlsl"
#include "nbl/builtin/hlsl/math/angle_adding.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf.hlsl"
#include "nbl/builtin/hlsl/bxdf/fresnel.hlsl"

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
            fprintf(stderr, "[ERROR] seed %u: %s failed the jacobian * pdf test    %s\n", failedFor.rc.state, failedFor.name.c_str(), failedFor.errMsg.c_str());
            break;
        case BET_PDF_EVAL_DIFF:
            fprintf(stderr, "[ERROR] seed %u: %s quotient * pdf != eval    %s\n", failedFor.rc.state, failedFor.name.c_str(), failedFor.errMsg.c_str());
            break;
        case BET_RECIPROCITY:
            fprintf(stderr, "[ERROR] seed %u: %s failed the reciprocity test    %s\n", failedFor.rc.state, failedFor.name.c_str(), failedFor.errMsg.c_str());
            break;
        case BET_PRINT_MSG:
            fprintf(stderr, "[ERROR] seed %u: %s error message\n%s\n", failedFor.rc.state, failedFor.name.c_str(), failedFor.errMsg.c_str());
            break;
        case BET_GENERATE_H:
            fprintf(stderr, "[ERROR] seed %u: %s failed invalid H configuration generated    %s\n", failedFor.rc.state, failedFor.name.c_str(), failedFor.errMsg.c_str());
            break;
        default:
            fprintf(stderr, "[ERROR] seed %u: %s unknown error\n", failedFor.rc.state, failedFor.name.c_str());
        }

#ifdef _NBL_DEBUG
        for (volatile bool repeat = true; IsDebuggerPresent() && repeat && error < BET_NOBREAK; )
        {
            repeat = false;
            _NBL_DEBUG_BREAK_IF(true);
            failedFor.compute();
        }
#endif
    }
};

#define FOR_EACH_BEGIN_EX(r, ex) std::for_each(ex, r.begin(), r.end(), [&](uint32_t i) {
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
        auto shaderSrc = smart_refctd_ptr_static_cast<IShader>(assets[0]);

        auto shader = shaderSrc;
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

    assert(bxdf::surface_interactions::Isotropic<iso_interaction>);
    assert(bxdf::surface_interactions::Isotropic<aniso_interaction>);
    assert(bxdf::surface_interactions::Anisotropic<aniso_interaction>);

    assert(bxdf::CreatableIsotropicMicrofacetCache<iso_cache>);
    assert(bxdf::ReadableIsotropicMicrofacetCache<aniso_cache>);
    assert(bxdf::AnisotropicMicrofacetCache<aniso_cache>);

    using ndf_beckmann_t = bxdf::ndf::Beckmann<float, false, bxdf::ndf::MTT_REFLECT>;
    assert(bxdf::ndf::NDF<ndf_beckmann_t>);
    using ndf_ggx_t = bxdf::ndf::GGX<float, true, bxdf::ndf::MTT_REFLECT_REFRACT>;
    assert(bxdf::ndf::NDF<ndf_ggx_t>);

    using fresnel_schlick_t = bxdf::fresnel::Schlick<float32_t3>;
    assert(bxdf::fresnel::Fresnel<fresnel_schlick_t>);

    assert(bxdf::bxdf_concepts::IsotropicBxDF<bxdf::reflection::SLambertian<iso_config_t>>);
    assert(bxdf::bxdf_concepts::IsotropicBxDF<bxdf::reflection::SOrenNayar<iso_config_t>>);
    assert(bxdf::bxdf_concepts::IsotropicBxDF<bxdf::transmission::SSmoothDielectric<iso_config_t>>);

    assert(bxdf::bxdf_concepts::MicrofacetBxDF<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>>);
    assert(bxdf::bxdf_concepts::MicrofacetBxDF<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>>);
    assert(bxdf::bxdf_concepts::MicrofacetBxDF<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>>);
    assert(bxdf::bxdf_concepts::MicrofacetBxDF<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>>);

    const bool logInfo = testconfigs["logInfo"];
    PrintFailureCallback cb;

    // test jacobian * pdf
    uint32_t runs = testconfigs["TestJacobian"]["runs"];
    auto rJacobian = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN(rJacobian)
    STestInitParams initparams{ .logInfo = logInfo };
    initparams.state = i;
    initparams.verbose = testconfigs["TestJacobian"]["verbose"];

    TestJacobian<bxdf::reflection::SLambertian<iso_config_t>>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SOrenNayar<iso_config_t>>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SDeltaDistribution<iso_config_t>>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>,true>::run(initparams, cb);
    //TestJacobian<bxdf::reflection::SIridescent<iso_microfacet_config_t>, false>::run(initparams, cb);

    TestJacobian<bxdf::transmission::SLambertian<iso_config_t>>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SOrenNayar<iso_config_t>>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SSmoothDielectric<iso_config_t> >::run(initparams, cb);
    TestJacobian<bxdf::transmission::SThinSmoothDielectric<iso_config_t> >::run(initparams, cb);
    TestJacobian<bxdf::transmission::SDeltaDistribution<iso_config_t>>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>,true>::run(initparams, cb);
    //TestJacobian<bxdf::transmission::SIridescent<iso_microfacet_config_t>, false>::run(initparams, cb);
    FOR_EACH_END


    // test reciprocity
    runs = testconfigs["TestReciprocity"]["runs"];
    auto rReciprocity = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN(rReciprocity)
    STestInitParams initparams{ .logInfo = logInfo };
    initparams.state = 3;
    initparams.verbose = testconfigs["TestReciprocity"]["verbose"];

    TestReciprocity<bxdf::reflection::SLambertian<iso_config_t>>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SOrenNayar<iso_config_t>>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SDeltaDistribution<iso_config_t>>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    //TestReciprocity<bxdf::reflection::SIridescent<iso_microfacet_config_t>, false>::run(initparams, cb);

    TestReciprocity<bxdf::transmission::SLambertian<iso_config_t>>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SOrenNayar<iso_config_t>>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SSmoothDielectric<iso_config_t>>::run(initparams, cb);    
    TestReciprocity<bxdf::transmission::SThinSmoothDielectric<iso_config_t>>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SDeltaDistribution<iso_config_t>>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SBeckmannDielectricIsotropic<rectest_iso_microfacet_config_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SBeckmannDielectricAnisotropic<rectest_aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SGGXDielectricIsotropic<rectest_iso_microfacet_config_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SGGXDielectricAnisotropic<rectest_aniso_microfacet_config_t>, true>::run(initparams, cb);
    //TestReciprocity<bxdf::transmission::SIridescent<iso_microfacet_config_t>, false>::run(initparams, cb);
    FOR_EACH_END


    // test buckets of inf
    // NOTE: can safely ignore any errors for smooth dielectric BxDFs because pdf SHOULD be inf
    runs = testconfigs["TestBucket"]["runs"];
    auto rBucket = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN(rBucket)
    STestInitParams initparams{ .logInfo = logInfo };
    initparams.state = i;
    initparams.samples = testconfigs["TestBucket"]["samples"];

    TestBucket<bxdf::reflection::SLambertian<iso_config_t>>::run(initparams, cb);
    TestBucket<bxdf::reflection::SOrenNayar<iso_config_t>>::run(initparams, cb);
    TestBucket<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestBucket<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestBucket<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestBucket<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    //TestBucket<bxdf::reflection::SIridescent<iso_microfacet_config_t>, false>::run(initparams, cb);

    TestBucket<bxdf::transmission::SLambertian<iso_config_t>>::run(initparams, cb);
    TestBucket<bxdf::transmission::SOrenNayar<iso_config_t>>::run(initparams, cb);
    TestBucket<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestBucket<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestBucket<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestBucket<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    //TestBucket<bxdf::transmission::SIridescent<iso_microfacet_config_t>, false>::run(initparams, cb);
    FOR_EACH_END


    // chi2 test for sampling and pdf
    runs = testconfigs["TestChi2"]["runs"];
    auto rChi2 = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN_EX(rChi2, std::execution::par_unseq)
    STestInitParams initparams{ .logInfo = logInfo };
    initparams.state = i;
    initparams.samples = testconfigs["TestChi2"]["samples"];
    initparams.thetaSplits = testconfigs["TestChi2"]["thetaSplits"];
    initparams.phiSplits = testconfigs["TestChi2"]["phiSplits"];
    initparams.writeFrequencies = testconfigs["TestChi2"]["writeFrequencies"];

    TestChi2<bxdf::reflection::SLambertian<iso_config_t>>::run(initparams, cb);
    TestChi2<bxdf::reflection::SOrenNayar<iso_config_t>>::run(initparams, cb);
    TestChi2<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestChi2<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestChi2<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestChi2<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    //TestChi2<bxdf::reflection::SIridescent<iso_microfacet_config_t>, false>::run(initparams, cb);

    TestChi2<bxdf::transmission::SLambertian<iso_config_t>>::run(initparams, cb);
    TestChi2<bxdf::transmission::SOrenNayar<iso_config_t>>::run(initparams, cb);
    TestChi2<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestChi2<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestChi2<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestChi2<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    //TestChi2<bxdf::transmission::SIridescent<iso_microfacet_config_t>, false>::run(initparams, cb);
    FOR_EACH_END

#if 0
    // testing ndf jacobian * dg1, ONLY for cook torrance bxdfs
    runs = testconfigs["TestNDF"]["runs"];
    auto rNdf = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN(rNdf)
        STestInitParams initparams{ .logInfo = logInfo };
    initparams.state = i;
    initparams.verbose = testconfigs["TestNDF"]["verbose"];

    TestNDF<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestNDF<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestNDF<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestNDF<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);

    TestNDF<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestNDF<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestNDF<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestNDF<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    FOR_EACH_END
#endif
#if 0
    // test generated H that NdotV*VdotH>=0.0, VdotL calculation
    runs = testconfigs["TestCTGenerateH"]["runs"];
    auto rGenerateH = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN_EX(rGenerateH, std::execution::par_unseq)
    STestInitParams initparams{ .logInfo = logInfo };
    initparams.state = i;
    initparams.samples = testconfigs["TestCTGenerateH"]["samples"];
    initparams.immediateFail = testconfigs["TestCTGenerateH"]["immediateFail"];

    TestCTGenerateH<bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestCTGenerateH<bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestCTGenerateH<bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestCTGenerateH<bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);

    TestCTGenerateH<bxdf::transmission::SBeckmannDielectricIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestCTGenerateH<bxdf::transmission::SBeckmannDielectricAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    TestCTGenerateH<bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>, false>::run(initparams, cb);
    TestCTGenerateH<bxdf::transmission::SGGXDielectricAnisotropic<aniso_microfacet_config_t>, true>::run(initparams, cb);
    FOR_EACH_END
#endif

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
            angle_adder.addCosine(b, Sin(b));
            float res = angle_adder.getSumofArccos();
            if (!checkEq<float>(res, exAB, 1e-3))
                fprintf(stderr, "[ERROR] angle adding (2 angles) failed! expected %f, got %f\n", exAB, res);

            const float exABCD = exAB + acos(c) + acos(d);
            angle_adder = math::sincos_accumulator<float>::create(a, Sin(a));
            angle_adder.addCosine(b, Sin(b));
            angle_adder.addCosine(c, Sin(c));
            angle_adder.addCosine(d, Sin(d));
            res = angle_adder.getSumofArccos();
            if (!checkEq<float>(res, exABCD, 1e-3))
                fprintf(stderr, "[ERROR] angle adding (4 angles) failed! expected %f, got %f\n", exABCD, res);
        }
    }

    return 0;
}