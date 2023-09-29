// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma shader_stage(compute)

#define SHADER_CRASHING_ASSERT(expr) \
    do { \
        [branch] if (!(expr)) \
          vk::RawBufferStore<uint32_t>(0xdeadbeefBADC0FFbull,0x45u,4u); \
    } while(true)

#include <nbl/builtin/hlsl/cpp_compat/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/colorspace/EOTF.hlsl>
#include <nbl/builtin/hlsl/colorspace/OETF.hlsl>

[numthreads(1, 1, 1)]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    SHADER_CRASHING_ASSERT(true);
    const float3 TEST_VEC = float3(1.0f, 2.0f, 3.0f);
    
    nbl::hlsl::colorspace::eotf::identity<float3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::impl_shared_2_4<float3>(TEST_VEC, 0.5f);
    nbl::hlsl::colorspace::eotf::sRGB<float3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::Display_P3<float3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::DCI_P3_XYZ<float3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::SMPTE_170M<float3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::SMPTE_ST2084<float3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::HDR10_HLG<float3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::AdobeRGB<float3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::Gamma_2_2<float3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::ACEScc<float3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::ACEScct<float3>(TEST_VEC);
    
    nbl::hlsl::colorspace::oetf::identity<float3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::impl_shared_2_4<float3>(TEST_VEC, 0.5f);
    nbl::hlsl::colorspace::oetf::sRGB<float3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::Display_P3<float3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::DCI_P3_XYZ<float3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::SMPTE_170M<float3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::SMPTE_ST2084<float3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::HDR10_HLG<float3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::AdobeRGB<float3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::Gamma_2_2<float3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::ACEScc<float3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::ACEScct<float3>(TEST_VEC);
}
