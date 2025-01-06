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
    float32_t4 result = testLambertianBRDF();
    ASSERT_ZERO(result);

    result = testLambertianBRDF2();
    assert(all<bool32_t4>(result < float32_t4(1e-4)));

    result = testOrenNayarBRDF();
    assert(all<bool32_t4>(result < float32_t4(1e-4)));

    // specular
    //result = testBlinnPhongBRDF();
    //printFloat4(result);

    result = testBeckmannBRDF();
    assert(all<bool32_t4>(result < float32_t4(1e-4)));

    result = testGGXBRDF();
    printFloat4(result);

    result = testGGXAnisoBRDF();
    printFloat4(result);

    // bxdfs
    result = testLambertianBSDF();
    assert(all<bool32_t4>(result < float32_t4(1e-4)));

    result = testBeckmannBSDF();
    assert(all<bool32_t4>(result < float32_t4(1e-4)));

    return 0;
}