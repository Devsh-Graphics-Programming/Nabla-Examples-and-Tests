#include <nabla.h>
#include <iostream>
#include <iomanip>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl::hlsl;

#include "app_resources/tests.hlsl"

#define ASSERT_ZERO(x) (assert(all<bool32_t4>((x.result) < float32_t4(1e-3))));

void printResult(const STestMeta& m)
{
    std::cout << m.testName << " " << m.bxdfName << "\t"
        << m.result.x << " "
        << m.result.y << " "
        << m.result.z << " "
        << m.result.w << "\n";
}

int main(int argc, char** argv)
{
    std::cout << std::fixed << std::setprecision(4);

    using bool32_t4 = vector<bool, 4>;

    const uint32_t2 state = uint32_t2(10u, 42u);    // (12u, 69u)

    // test u offset, 2 axis
    printResult((TestUOffset<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));
    printResult((TestUOffset<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));
    printResult(TestUOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state));
    printResult((TestUOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state)));
    printResult(TestUOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state));
    printResult((TestUOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state)));

    printResult((TestUOffset<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));
    //printResult(TestUOffset<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache>>::run(state));
    //printResult(TestUOffset<bxdf::transmission::SSmoothDielectricBxDF<sample_t, iso_cache, aniso_cache, true>>::run(state));
    printResult(TestUOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state));
    printResult((TestUOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state)));
    printResult(TestUOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state));
    printResult(TestUOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state));

    return 0;
}