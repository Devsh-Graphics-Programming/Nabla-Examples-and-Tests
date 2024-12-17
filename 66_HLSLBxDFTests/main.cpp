#include <nabla.h>
#include <iostream>

#include "app_resources/tests.hlsl"

int main(int argc, char** argv)
{
    float32_t3 result = testLambertianBRDF();

    std::cout << result.x << " "
            << result.y << " "
            << result.z << "\n";

    return 0;
}