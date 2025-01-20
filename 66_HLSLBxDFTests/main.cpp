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
            fprintf(stderr, "seed %u: %s failed the reprocity test\n", failedFor.rc.state, failedFor.name.c_str());
            break;
        default:
            fprintf(stderr, "seed %u: %s unknown error\n", failedFor.rc.state, failedFor.name.c_str());
        }

        for (volatile bool repeat = true; IsDebuggerPresent() && repeat; )
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

    auto r = std::ranges::views::iota(0u, 10u);
    PrintFailureCallback cb;

    // test u offset, 2 axis
    FOR_EACH_BEGIN(r)
    TestUOffset<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    TestUOffset<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    TestUOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,false>::run(i, cb);
    TestUOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(i, cb);
    TestUOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,false>::run(i, cb);
    TestUOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(i, cb);

    TestUOffset<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(i, cb);
    //TestUOffset<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::run(i, cb);
    //TestUOffset<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(i, cb);
    TestUOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,false>::run(i, cb);
    TestUOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(i, cb);
    TestUOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,false>::run(i, cb);
    TestUOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(i, cb);
    FOR_EACH_END


    // test reciprocity
    FOR_EACH_BEGIN(r)
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
    

    return 0;
}