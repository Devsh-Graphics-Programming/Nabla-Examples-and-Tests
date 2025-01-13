#include <nabla.h>
#include <iostream>
#include <iomanip>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl::hlsl;

#include "app_resources/tests.hlsl"

#define ASSERT_ZERO(x) (assert(all<bool32_t4>((x.result) < float32_t4(1e-3))));

//void printResult(const STestMeta& m)
//{
//    std::cout << m.testName << " " << m.bxdfName << "\t"
//        << m.result.x << " "
//        << m.result.y << " "
//        << m.result.z << " "
//        << m.result.w << "\n";
//}

struct PrintFailureCallback : FailureCallback
{
    void __call(ErrorType error, NBL_CONST_REF_ARG(SBxDFTestResources) failedFor, NBL_CONST_REF_ARG(sample_t) failedAt)
    {
        switch (error)
        {
        case NEGATIVE_VAL:
            fprintf(stderr, "pdf/quotient/eval < 0\n");
            break;
        case PDF_ZERO:
            fprintf(stderr, "pdf = 0\n");
            break;
        case QUOTIENT_INF:
            fprintf(stderr, "quotient -> inf\n");
            break;
        case JACOBIAN:
            fprintf(stderr, "failed the jacobian * pdf test\n");
            break;
        case PDF_EVAL_DIFF:
            fprintf(stderr, "quotient * pdf - eval not 0\n");
            break;
        case RECIPROCITY:
            fprintf(stderr, "failed the reprocity test\n");
            break;
        default:
            fprintf(stderr, "unknown error\n");
        }

        for (volatile bool repeat = true; IsDebuggerPresent() && repeat; )
        {
            repeat = false;
            __debugbreak();
            //failedFor.test(failedAt);
        }
    }
};

int main(int argc, char** argv)
{
    std::cout << std::fixed << std::setprecision(4);

    const uint32_t state = 69u;

    PrintFailureCallback cb;

    // test u offset, 2 axis
    TestUOffset<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state, cb);
    TestUOffset<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state, cb);
    TestUOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state, cb);
    TestUOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state, cb);
    TestUOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state, cb);
    TestUOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state, cb);

    TestUOffset<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state, cb);
    //TestUOffset<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache>>::run(state);
    //TestUOffset<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>>::run(state);
    TestUOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state, cb);
    TestUOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state, cb);
    TestUOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state, cb);
    TestUOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state, cb);

    return 0;
}