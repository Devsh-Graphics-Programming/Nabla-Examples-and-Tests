//// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#pragma shader_stage(compute)

#define SHADER_CRASHING_ASSERT(expr) \
    do { \
        [branch] if (!(expr)) \
          vk::RawBufferStore<uint32_t>(0xdeadbeefBADC0FFbull,0x45u,4u); \
    } while(true)

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

// #include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
// #include <nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl>
// #include <nbl/builtin/hlsl/colorspace/EOTF.hlsl>
// #include <nbl/builtin/hlsl/colorspace/OETF.hlsl>

//#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

#include <nbl/builtin/hlsl/mpl.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>

[numthreads(1, 1, 1)]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    const float3 TEST_VEC = float3(1.0f, 2.0f, 3.0f);
    
    // test functions from EOTF.hlsl
    //nbl::hlsl::colorspace::eotf::identity<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::eotf::impl_shared_2_4<float3>(TEST_VEC, 0.5f);
    //nbl::hlsl::colorspace::eotf::sRGB<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::eotf::Display_P3<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::eotf::DCI_P3_XYZ<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::eotf::SMPTE_170M<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::eotf::SMPTE_ST2084<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::eotf::HDR10_HLG<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::eotf::AdobeRGB<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::eotf::Gamma_2_2<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::eotf::ACEScc<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::eotf::ACEScct<float3>(TEST_VEC);
    
    // test functions from OETF.hlsl
    //nbl::hlsl::colorspace::oetf::identity<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::oetf::impl_shared_2_4<float3>(TEST_VEC, 0.5f);
    //nbl::hlsl::colorspace::oetf::sRGB<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::oetf::Display_P3<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::oetf::DCI_P3_XYZ<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::oetf::SMPTE_170M<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::oetf::SMPTE_ST2084<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::oetf::HDR10_HLG<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::oetf::AdobeRGB<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::oetf::Gamma_2_2<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::oetf::ACEScc<float3>(TEST_VEC);
    //nbl::hlsl::colorspace::oetf::ACEScct<float3>(TEST_VEC);
    
    // xoroshiro tests
    //const uint32_t2 state = uint32_t2(12u, 34u);
    //nbl::hlsl::Xoroshiro64Star xoroshiro64Star = nbl::hlsl::Xoroshiro64Star::construct(state);
    //xoroshiro64Star();
    //nbl::hlsl::Xoroshiro64StarStar xoroshiro64StarStar = nbl::hlsl::Xoroshiro64StarStar::construct(state);
    //xoroshiro64StarStar();
    
    //nbl::hlsl::mpl::countl_zero<2ull>::value;
    
    // TODO: test if std::rotl/r == nbl::hlsl::rotr/l == nbl::hlsl::mpl::rotr/l
    
    uint32_t mplRotlResult0 = nbl::hlsl::mpl::rotl<uint32_t, 2u, 1>::value;
    uint32_t mplRotlResult1 = nbl::hlsl::mpl::rotl<uint32_t, 2u, -1>::value;
    uint32_t mplRotrResult0 = nbl::hlsl::mpl::rotr<uint32_t, 2u, 1>::value;
    uint32_t mplRotrResult1 = nbl::hlsl::mpl::rotr<uint32_t, 2u, -1>::value;
    
    uint32_t rotlResult0 = nbl::hlsl::mpl::rotl<uint32_t, 2u, 1>::value;
    uint32_t rotlResult1 = nbl::hlsl::mpl::rotl<uint32_t, 2u, -1>::value;
    uint32_t rotrResult0 = nbl::hlsl::mpl::rotr<uint32_t, 2u, 1>::value;
    uint32_t rotrResult1 = nbl::hlsl::mpl::rotr<uint32_t, 2u, -1>::value;
    
    SHADER_CRASHING_ASSERT(rotlResult0 == mplRotlResult0);
    SHADER_CRASHING_ASSERT(rotlResult1 == mplRotlResult1);
    SHADER_CRASHING_ASSERT(rotrResult0 == mplRotrResult0);
    SHADER_CRASHING_ASSERT(rotrResult1 == mplRotrResult1);

    // TODO: more tests and compare with cpp version as well
    // countl_zero test
    {
        static const uint16_t TEST_VALUE_0 = 5;
        static const uint32_t TEST_VALUE_1 = 0x80000000u;
        static const uint32_t TEST_VALUE_2 = 0x8000000000000000u;
        static const uint32_t TEST_VALUE_3 = 0x00000001u;
        static const uint32_t TEST_VALUE_4 = 0x0000000000000001u;

        uint16_t compileTimeCountLZero = nbl::hlsl::mpl::countl_zero<TEST_VALUE_0>::value;
        uint16_t runTimeCountLZero = nbl::hlsl::countl_zero(TEST_VALUE_0);
        SHADER_CRASHING_ASSERT(compileTimeCountLZero == runTimeCountLZero);

        compileTimeCountLZero = nbl::hlsl::mpl::countl_zero<TEST_VALUE_1>::value;
        runTimeCountLZero = nbl::hlsl::countl_zero(TEST_VALUE_1);
        SHADER_CRASHING_ASSERT(compileTimeCountLZero == runTimeCountLZero);

        compileTimeCountLZero = nbl::hlsl::mpl::countl_zero<TEST_VALUE_2>::value;
        runTimeCountLZero = nbl::hlsl::countl_zero(TEST_VALUE_2);
        SHADER_CRASHING_ASSERT(compileTimeCountLZero == runTimeCountLZero);

        compileTimeCountLZero = nbl::hlsl::mpl::countl_zero<TEST_VALUE_3>::value;
        runTimeCountLZero = nbl::hlsl::countl_zero(TEST_VALUE_3);
        SHADER_CRASHING_ASSERT(compileTimeCountLZero == runTimeCountLZero);

        compileTimeCountLZero = nbl::hlsl::mpl::countl_zero<TEST_VALUE_4>::value;
        runTimeCountLZero = nbl::hlsl::countl_zero(TEST_VALUE_4);
        SHADER_CRASHING_ASSERT(compileTimeCountLZero == runTimeCountLZero);
    }
}
