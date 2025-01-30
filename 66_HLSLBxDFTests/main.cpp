#include <nabla.h>
#include <iostream>
#include <iomanip>
#include <ranges>
#include <execution>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

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

    std::ifstream f("app_resources/config.json");
    json testconfigs = json::parse(f);

    //auto r5 = std::ranges::views::iota(0u, 5u);
    //auto r10 = std::ranges::views::iota(0u, 10u);
    PrintFailureCallback cb;

    // test jacobian * pdf
    uint32_t runs = testconfigs["TestJacobian"]["runs"];
    auto rJacobian = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN(rJacobian)
    STestInitParams initparams;
    initparams.state = i;

    TestJacobian<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(initparams, cb);

    TestJacobian<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    //TestJacobian<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::run(initparams, cb);
    //TestJacobian<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestJacobian<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(initparams, cb);
    FOR_EACH_END


    // test reciprocity
    runs = testconfigs["TestReciprocity"]["runs"];
    auto rReciprocity = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN(rReciprocity)
    STestInitParams initparams;
    initparams.state = i;

    TestReciprocity<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);

    TestReciprocity<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    //TestReciprocity<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::run(initparams, cb);
    //TestReciprocity<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestReciprocity<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    FOR_EACH_END


    // test buckets of inf
    runs = testconfigs["TestBucket"]["runs"];
    auto rBucket = std::ranges::views::iota(0u, runs);
    FOR_EACH_BEGIN(rBucket)
    STestInitParams initparams;
    initparams.state = i;
    initparams.samples = testconfigs["TestBucket"]["samples"];

    TestBucket<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestBucket<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    TestBucket<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestBucket<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestBucket<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestBucket<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);

    TestBucket<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(initparams, cb);
    //TestBucket<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::run(initparams, cb);
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
    STestInitParams initparams;
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
    //TestChi2<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::run(initparams, cb);
    //TestChi2<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(initparams, cb);
    TestChi2<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestChi2<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    TestChi2<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(initparams, cb);
    TestChi2<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(initparams, cb);
    FOR_EACH_END

    return 0;
}