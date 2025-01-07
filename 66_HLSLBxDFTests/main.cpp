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

    const uint32_t2 state = uint32_t2(10u, 42u);    // (12u, 69u)

    // brdfs
    // diffuse
    ASSERT_ZERO(testLambertianBRDF2());

    ASSERT_ZERO((TestUOffsetBasicBRDF<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));
    ASSERT_ZERO((TestUOffsetBasicBRDF<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));

    // specular
    printFloat4(TestUOffsetMicrofacetBRDF<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state));

    ASSERT_ZERO((TestUOffsetMicrofacetBRDF<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state)));
    
    printFloat4(TestUOffsetMicrofacetBRDF<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>, false>::run(state));

    ASSERT_ZERO((TestUOffsetMicrofacetBRDF<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state)));

    // bxdfs
    // diffuse
    ASSERT_ZERO((TestUOffsetBasicBSDF<bxdf::transmission::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::run(state)));

    // specular
    printFloat4(TestUOffsetMicrofacetBSDF<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state));

    printFloat4(TestUOffsetMicrofacetBSDF<bxdf::transmission::SBeckmannDielectricBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state)); // this one's fine

    printFloat4(TestUOffsetMicrofacetBSDF<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>,false>::run(state));

    printFloat4(TestUOffsetMicrofacetBSDF<bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache>,true>::run(state));

    return 0;
}