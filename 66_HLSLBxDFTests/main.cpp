#include <nabla.h>
#include <iostream>
#include <iomanip>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl::hlsl;

#include "app_resources/tests.hlsl"

#define ASSERT_ZERO(x) (assert(all<bool32_t4>((x) < float32_t4(1e-3))));

void printFloat4(const float32_t4& result)
{
    std::cout << result.x << " "
        << result.y << " "
        << result.z << " "
        << result.w << "\n";
}

int main(int argc, char** argv)
{
    std::cout << std::fixed << std::setprecision(4);

    using bool32_t4 = vector<bool, 4>;

    const uint32_t2 state = uint32_t2(12u, 69u);    // (12u, 69u)

    // test u offset, 2 axis
    ASSERT_ZERO((TestUOffset<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));
    ASSERT_ZERO((TestUOffset<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));
    printFloat4(TestUOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state));
    ASSERT_ZERO((TestUOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state)));
    printFloat4(TestUOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state));
    ASSERT_ZERO((TestUOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state)));

    ASSERT_ZERO((TestUOffset<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));
    printFloat4(TestUOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state));
    ASSERT_ZERO((TestUOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state)));
    printFloat4(TestUOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state));
    printFloat4(TestUOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state));

    // test v offset, 3 axis
    ASSERT_ZERO((TestVOffset<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));
    ASSERT_ZERO((TestVOffset<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));
    printFloat4(TestVOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>, false>::run(state));
    ASSERT_ZERO((TestVOffset<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>, true>::run(state)));
    printFloat4(TestVOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>, false>::run(state));
    ASSERT_ZERO((TestVOffset<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>, true>::run(state)));

    ASSERT_ZERO((TestVOffset<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));
    printFloat4(TestVOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>, false>::run(state));
    ASSERT_ZERO((TestVOffset<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>, true>::run(state)));
    printFloat4(TestVOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>, false>::run(state));
    printFloat4(TestVOffset<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>, true>::run(state));

    return 0;
}