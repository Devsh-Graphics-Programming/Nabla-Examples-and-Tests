#include <nabla.h>
#include <iostream>
#include <iomanip>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl::hlsl;

#include "app_resources/tests.hlsl"

struct PrintFailureCallback : FailureCallback
{
    void __call(ErrorType error, NBL_REF_ARG(TestBase) failedFor) override
    {
        switch (error)
        {
        case NEGATIVE_VAL:
            fprintf(stderr, "%s pdf/quotient/eval < 0\n", failedFor.name.c_str());
            break;
        case PDF_ZERO:
            fprintf(stderr, "%s pdf = 0\n", failedFor.name.c_str());
            break;
        case QUOTIENT_INF:
            fprintf(stderr, "%s quotient -> inf\n", failedFor.name.c_str());
            break;
        case JACOBIAN:
            fprintf(stderr, "%s failed the jacobian * pdf test\n", failedFor.name.c_str());
            break;
        case PDF_EVAL_DIFF:
            fprintf(stderr, "%s quotient * pdf - eval not 0\n", failedFor.name.c_str());
            break;
        case RECIPROCITY:
            fprintf(stderr, "%s failed the reprocity test\n", failedFor.name.c_str());
            break;
        default:
            fprintf(stderr, "%s unknown error\n", failedFor.name.c_str());
        }

        for (volatile bool repeat = true; IsDebuggerPresent() && repeat; )
        {
            repeat = false;
            __debugbreak();
            failedFor.compute();
        }
    }
};

int main(int argc, char** argv)
{
    std::cout << std::fixed << std::setprecision(4);

    const uint32_t state = 69u;

    PrintFailureCallback cb;

    // test u offset, 2 axis
    TestUOffset<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(state, cb);
    TestUOffset<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(state, cb);
    TestUOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,false>::run(state, cb);
    TestUOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(state, cb);
    TestUOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,false>::run(state, cb);
    TestUOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(state, cb);

    TestUOffset<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>>::run(state, cb);
    //TestUOffset<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>>::run(state);
    //TestUOffset<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t, true>>::run(state);
    TestUOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,false>::run(state, cb);
    TestUOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(state, cb);
    TestUOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,false>::run(state, cb);
    TestUOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>,true>::run(state, cb);

    return 0;
}