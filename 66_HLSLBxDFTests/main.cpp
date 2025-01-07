#include <nabla.h>
#include <iostream>
#include <iomanip>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl::hlsl;

#include "app_resources/tests.hlsl"

#define ASSERT_ZERO(x) assert(all<bool32_t4>(x < float32_t4(1e-4)));

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

    // brdfs
    // diffuse
    float32_t4 result = testLambertianBRDF2();
    ASSERT_ZERO(result);

    TestUOffsetBasicBRDF<bxdf::reflection::SLambertianBxDF<sample_t, iso_interaction, aniso_interaction>>::run(uint32_t2(10u, 42u));
    ASSERT_ZERO(result);

    TestUOffsetBasicBRDF<bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction>>::run(uint32_t2(10u, 42u));
    ASSERT_ZERO(result);

    // specular
    //result = testBlinnPhongBRDF();
    //printFloat4(result);

    TestUOffsetMicrofacetBRDF<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>,false>::run(uint32_t2(10u, 42u));
    ASSERT_ZERO(result);

    TestUOffsetMicrofacetBRDF<bxdf::reflection::SBeckmannBxDF<sample_t, iso_cache, aniso_cache>,true>::run(uint32_t2(10u, 42u));
    ASSERT_ZERO(result);

    TestUOffsetMicrofacetBRDF<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>,false>::run(uint32_t2(10u, 42u));
    ASSERT_ZERO(result);

    TestUOffsetMicrofacetBRDF<bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache>,true>::run(uint32_t2(10u, 42u));
    ASSERT_ZERO(result);

    // bxdfs
    result = testLambertianBSDF();
    ASSERT_ZERO(result);

    result = testBeckmannBSDF();
    printFloat4(result);

    result = testBeckmannAnisoBSDF();
    printFloat4(result);

    result = testGGXBSDF();
    printFloat4(result);

    result = testGGXAnisoBSDF();
    printFloat4(result);

    return 0;
}