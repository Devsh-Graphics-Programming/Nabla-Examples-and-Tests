//// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#if 1

#pragma shader_stage(compute)

#define SHADER_CRASHING_ASSERT(expr) \
    do { \
        [branch] if (!(expr)) \
          vk::RawBufferStore<uint32_t>(0xdeadbeefBADC0FFbull,0x45u,4u); \
    } while(true)

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

//#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
//#include <nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl>
//#include <nbl/builtin/hlsl/colorspace/EOTF.hlsl>
//#include <nbl/builtin/hlsl/colorspace/OETF.hlsl>

#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

#include <nbl/builtin/hlsl/mpl.hlsl>

[numthreads(1, 1, 1)]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    ;
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
    const uint32_t2 state = uint32_t2(12u, 34u);
    nbl::hlsl::Xoroshiro64Star xoroshiro64Star = nbl::hlsl::Xoroshiro64Star::construct(state);
    xoroshiro64Star();
    nbl::hlsl::Xoroshiro64StarStar xoroshiro64StarStar = nbl::hlsl::Xoroshiro64StarStar::construct(state);
    xoroshiro64StarStar();
    
    nbl::hlsl::mpl::clz<2ull>::value;
    
    // TODO: test if std::rotl/r == nbl::hlsl::rotr/l == nbl::hlsl::mpl::rotr/l
    // TODO: fix nbl::hlsl::mpl::countl_zero and test if std::countl_zero == nbl::hlsl::countl_zero == nbl::hlsl::mpl::countl_zero
        
}

#else

#pragma shader_stage(compute)

#include <nbl/builtin/hlsl/type_traits.hlsl>

namespace nbl
{
namespace hlsl
{
    template<uint16_t bits_log2>
    struct clz_masks
    {
        // static const uint16_t SHIFT = uint16_t(1)<<(bits_log2-1);
        // static const uint64_t LO_MASK = (1ull<<SHIFT)-1;
    
        static const uint16_t SHIFT = type_traits::conditional<bool(bits_log2),type_traits::integral_constant<uint16_t,uint16_t(1)<<(bits_log2-1)>,type_traits::integral_constant<uint16_t,0> >::type::value;
        static const uint64_t LO_MASK = type_traits::conditional<bool(bits_log2),type_traits::integral_constant<uint64_t,(1ull<<SHIFT)-1>,type_traits::integral_constant<uint64_t,0> >::type::value;
    };
    
    template<>
    struct clz_masks<0>
    {
        static const uint16_t shift = 0;
        static const uint64_t lo_mask = 0;
    };
}
}

[numthreads(1, 1, 1)]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    //SHADER_CRASHING_ASSERT(false);
}

#endif