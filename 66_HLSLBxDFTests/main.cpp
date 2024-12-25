#include <nabla.h>
#include <iostream>
#include <iomanip>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl::hlsl;

#include "app_resources/tests.hlsl"

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

    float32_t4 result = testLambertianBRDF();
    printFloat4(result);

    result = testBeckmannBRDF();
    printFloat4(result);

    result = testLambertianBSDF();
    printFloat4(result);

    return 0;
}