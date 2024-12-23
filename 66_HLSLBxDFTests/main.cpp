#include <nabla.h>
#include <iostream>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl::hlsl;

#include "app_resources/tests.hlsl"

int main(int argc, char** argv)
{
    float32_t4 result = testLambertianBRDF();

    std::cout << result.x << " "
            << result.y << " "
            << result.z << " "
            << result.w << "\n";

    result = testLambertianBSDF();

    std::cout << result.x << " "
        << result.y << " "
        << result.z << " "
        << result.w << "\n";

    return 0;
}