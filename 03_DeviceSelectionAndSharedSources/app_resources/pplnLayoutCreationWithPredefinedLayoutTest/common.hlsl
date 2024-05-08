// This is a magical header that provides most HLSL types and intrinsics in C++
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

struct PSInput
{
    [[vk::location(2)]] nointerpolation uint4 data1 : COLOR1;
    [[vk::location(5)]] float2 data2 : COLOR2;
};

struct SomeType
{
    uint32_t a;
	uint32_t b[5];
	uint32_t c[10];
};

NBL_CONSTEXPR uint32_t WorkgroupSize = 256;