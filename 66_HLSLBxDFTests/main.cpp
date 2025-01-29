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
    void __call(ErrorType error, NBL_REF_ARG(TestBase) failedFor) override
    {
        switch (error)
        {
        case BET_NEGATIVE_VAL:
            fprintf(stderr, "seed %u: %s pdf/quotient/eval < 0\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_PDF_ZERO:
            fprintf(stderr, "seed %u: %s pdf = 0\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_QUOTIENT_INF:
            fprintf(stderr, "seed %u: %s quotient -> inf\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_JACOBIAN:
            fprintf(stderr, "seed %u: %s failed the jacobian * pdf test\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_PDF_EVAL_DIFF:
            fprintf(stderr, "seed %u: %s quotient * pdf != eval\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_RECIPROCITY:
            fprintf(stderr, "seed %u: %s failed the reciprocity test\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        case BET_PRINT_MSG:
            fprintf(stderr, "seed %u: %s error message\n%s\n", failedFor.rc.state, failedFor.name.c_str(), failedFor.errMsg.c_str());
            break;
        default:
            fprintf(stderr, "seed %u: %s unknown error\n", failedFor.rc.state, failedFor.name.c_str());
        }

        // TODO: #ifdef NBL_ENABLE_DEBUGBREAK
        for (volatile bool repeat = true; IsDebuggerPresent() && repeat && (error != BET_PRINT_MSG); )
        {
            repeat = false;
            __debugbreak();
            failedFor.compute();
        }
    }
};

#define FOR_EACH_BEGIN(r) std::for_each(std::execution::seq, r.begin(), r.end(), [&](uint32_t i) {
#define FOR_EACH_END });

int main(int argc, char** argv)
{
    std::cout << std::fixed << std::setprecision(4);

    auto r5 = std::ranges::views::iota(0u, 5u);
    auto r10 = std::ranges::views::iota(0u, 10u);
    PrintFailureCallback cb;

    // test jacobian * pdf
    FOR_EACH_BEGIN(r10)
    TestJacobian<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    TestJacobian<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    TestJacobian<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestJacobian<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);
    TestJacobian<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestJacobian<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(i, cb);

    TestJacobian<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    //TestJacobian<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::run(i, cb);
    //TestJacobian<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(i, cb);
    TestJacobian<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestJacobian<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);
    TestJacobian<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestJacobian<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(i, cb);
    FOR_EACH_END


    // test reciprocity
    FOR_EACH_BEGIN(r10)
    TestReciprocity<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    TestReciprocity<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    TestReciprocity<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestReciprocity<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);
    TestReciprocity<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestReciprocity<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);

    TestReciprocity<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    //TestReciprocity<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::run(i, cb);
    //TestReciprocity<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(i, cb);
    TestReciprocity<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestReciprocity<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);
    TestReciprocity<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestReciprocity<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);
    FOR_EACH_END


    // test buckets of inf
    FOR_EACH_BEGIN(r10)
    TestBucket<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    TestBucket<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    TestBucket<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestBucket<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);
    TestBucket<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestBucket<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);

    TestBucket<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    //TestBucket<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::run(i, cb);
    //TestBucket<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(i, cb);
    TestBucket<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestBucket<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);
    TestBucket<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    TestBucket<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);
    FOR_EACH_END
    
    //TestChi2<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(5u, cb);
    //TestChi2<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(1u, cb);
    TestChi2<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(10u, cb);


    // chi2 test for sampling and pdf
    //FOR_EACH_BEGIN(r5)
    //TestChi2<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    //TestChi2<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    //TestChi2<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    //TestChi2<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);
    //TestChi2<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    //TestChi2<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);

    //TestChi2<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    ////TestChi2<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::run(i, cb);
    ////TestChi2<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(i, cb);
    //TestChi2<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    //TestChi2<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);
    //TestChi2<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, false>::run(i, cb);
    //TestChi2<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>, true>::run(i, cb);
    //FOR_EACH_END

    return 0;
}