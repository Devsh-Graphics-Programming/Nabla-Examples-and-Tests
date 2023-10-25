//// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#pragma shader_stage(compute)

#define STATIC_ASSERT(C) { nbl::hlsl::conditional<C, int, void>::type a = 0; }

#define IS_SAME(L,R) nbl::hlsl::is_same<L,R>::value
#define SHADER_CRASHING_ASSERT(expr) \
{ \
    bool con = (expr); \
    do { \
        [branch] if (!con) \
          vk::RawBufferStore<uint32_t>(0xdeadbeefBADC0FFbull,0x45u,4u); \
    } while(!con); \
} 


#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/colorspace/EOTF.hlsl>
#include <nbl/builtin/hlsl/colorspace/OETF.hlsl>

#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

#include <nbl/builtin/hlsl/mpl.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>

#include <nbl/builtin/hlsl/limits.hlsl>

struct PushConstants
{
	uint2 imgSize;
};

[[vk::push_constant]]
PushConstants u_pushConstants;

[[vk::binding(0, 0)]] Texture2D<float4> inImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> outImage;
[[vk::binding(2, 0)]] Buffer<float4> inBuffer;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> outBuffer;


template<int A>
struct Spec
{
    static const int value = Spec<A-1>::value + 1;
};

template<>
struct Spec<0>
{
    static const int value = 0;
};

Buffer<float32_t4>  unbounded[];

template<class T>
bool val(T) { return nbl::hlsl::is_unbounded_array<T>::value; }


template<typename T, uint32_t N>
struct array
{
  T data[N];
};

void fill(uint3 invocationID, float val)
{
    outImage[invocationID.xy] = float4(val,val,val,val);
    outBuffer[invocationID.x * invocationID.y] = float4(val,val,val,val);
}

void fill(uint3 invocationID, float4 val)
{
    outImage[invocationID.xy] = val;
    outBuffer[invocationID.x * invocationID.y] = val;
}


[numthreads(8, 8, 1)]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    fill(invocationID, 1);
    const float32_t3 TEST_VEC = float32_t3(1.0f, 2.0f, 3.0f);

    fill(invocationID, 2);
    // test functions from EOTF.hlsl
    nbl::hlsl::colorspace::eotf::identity<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::impl_shared_2_4<float32_t3>(TEST_VEC, 0.5f);
    nbl::hlsl::colorspace::eotf::sRGB<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::Display_P3<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::DCI_P3_XYZ<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::SMPTE_170M<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::SMPTE_ST2084<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::HDR10_HLG<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::AdobeRGB<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::Gamma_2_2<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::ACEScc<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::eotf::ACEScct<float32_t3>(TEST_VEC);
    
    fill(invocationID, 3);

    // test functions from OETF.hlsl
    nbl::hlsl::colorspace::oetf::identity<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::impl_shared_2_4<float32_t3>(TEST_VEC, 0.5f);
    nbl::hlsl::colorspace::oetf::sRGB<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::Display_P3<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::DCI_P3_XYZ<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::SMPTE_170M<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::SMPTE_ST2084<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::HDR10_HLG<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::AdobeRGB<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::Gamma_2_2<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::ACEScc<float32_t3>(TEST_VEC);
    nbl::hlsl::colorspace::oetf::ACEScct<float32_t3>(TEST_VEC);
    
    fill(invocationID, 4);
    // xoroshiro tests
    const uint32_t2 state = uint32_t2(12u, 34u);
    nbl::hlsl::Xoroshiro64Star xoroshiro64Star = nbl::hlsl::Xoroshiro64Star::construct(state);
    xoroshiro64Star();
    nbl::hlsl::Xoroshiro64StarStar xoroshiro64StarStar = nbl::hlsl::Xoroshiro64StarStar::construct(state);
    xoroshiro64StarStar();
    
    //nbl::hlsl::mpl::countl_zero<2ull>::value;
    
    // TODO: test if std::rotl/r == nbl::hlsl::rotr/l == nbl::hlsl::mpl::rotr/l
    
    // uint32_t mplRotlResult0 = nbl::hlsl::mpl::rotl<uint32_t, 2u, 1>::value;
    // uint32_t mplRotlResult1 = nbl::hlsl::mpl::rotl<uint32_t, 2u, -1>::value;
    // uint32_t mplRotrResult0 = nbl::hlsl::mpl::rotr<uint32_t, 2u, 1>::value;
    // uint32_t mplRotrResult1 = nbl::hlsl::mpl::rotr<uint32_t, 2u, -1>::value;
    
    // uint32_t rotlResult0 = nbl::hlsl::mpl::rotl<uint32_t, 2u, 1>::value;
    // uint32_t rotlResult1 = nbl::hlsl::mpl::rotl<uint32_t, 2u, -1>::value;
    // uint32_t rotrResult0 = nbl::hlsl::mpl::rotr<uint32_t, 2u, 1>::value;
    // uint32_t rotrResult1 = nbl::hlsl::mpl::rotr<uint32_t, 2u, -1>::value;
    
    // SHADER_CRASHING_ASSERT(rotlResult0 == mplRotlResult0);
    // SHADER_CRASHING_ASSERT(rotlResult1 == mplRotlResult1);
    // SHADER_CRASHING_ASSERT(rotrResult0 == mplRotrResult0);
    // SHADER_CRASHING_ASSERT(rotrResult1 == mplRotrResult1);

    // TODO: more tests and compare with cpp version as well
    fill(invocationID, 5);
    // countl_zero test
    {
        static const uint16_t TEST_VALUE_0 = 5;
        static const uint32_t TEST_VALUE_1 = 0x80000000u;
        static const uint32_t TEST_VALUE_2 = 0x8000000000000000u;
        static const uint32_t TEST_VALUE_3 = 0x00000001u;
        static const uint32_t TEST_VALUE_4 = 0x0000000000000001u;
        

        fill(invocationID, 5.01);
        uint16_t compileTimeCountLZero = nbl::hlsl::mpl::countl_zero<uint16_t, TEST_VALUE_0>::value;
        uint16_t runTimeCountLZero = nbl::hlsl::countl_zero(TEST_VALUE_0);
        fill(invocationID, float4(5.1, compileTimeCountLZero, runTimeCountLZero, 0));
        SHADER_CRASHING_ASSERT(compileTimeCountLZero == runTimeCountLZero);

        compileTimeCountLZero = nbl::hlsl::mpl::countl_zero<uint32_t, TEST_VALUE_1>::value;
        runTimeCountLZero = nbl::hlsl::countl_zero(TEST_VALUE_1);
        fill(invocationID, float4(5.2, compileTimeCountLZero, runTimeCountLZero, 0));
        SHADER_CRASHING_ASSERT(compileTimeCountLZero == runTimeCountLZero);

        compileTimeCountLZero = nbl::hlsl::mpl::countl_zero<uint32_t, TEST_VALUE_2>::value;
        runTimeCountLZero = nbl::hlsl::countl_zero(TEST_VALUE_2);
        fill(invocationID, float4(5.3, compileTimeCountLZero, runTimeCountLZero, 0));
        SHADER_CRASHING_ASSERT(compileTimeCountLZero == runTimeCountLZero);

        compileTimeCountLZero = nbl::hlsl::mpl::countl_zero<uint32_t, TEST_VALUE_3>::value;
        runTimeCountLZero = nbl::hlsl::countl_zero(TEST_VALUE_3);
        fill(invocationID, float4(5.4, compileTimeCountLZero, runTimeCountLZero, 0));
        SHADER_CRASHING_ASSERT(compileTimeCountLZero == runTimeCountLZero);

        compileTimeCountLZero = nbl::hlsl::mpl::countl_zero<uint32_t, TEST_VALUE_4>::value;
        runTimeCountLZero = nbl::hlsl::countl_zero(TEST_VALUE_4);
        fill(invocationID, float4(5.5, compileTimeCountLZero, runTimeCountLZero, 0));
        SHADER_CRASHING_ASSERT(compileTimeCountLZero == runTimeCountLZero);
    }

    {
        bool A = Spec<3>::value == 3;
    }
    {
        bool A = nbl::hlsl::is_integral<int>::value;
    }
    {
        bool A = val(unbounded);
    }
    fill(invocationID, 6);
    {
        float4 v;
        fill(invocationID, float4(alignof(v.x), alignof(v), 0, 0));
        SHADER_CRASHING_ASSERT(alignof(v.x) == alignof(v));
    }
    
    {
        float4 v;
        const volatile float4 u;

        STATIC_ASSERT(IS_SAME(decltype(v.x), nbl::hlsl::impl::add_lvalue_reference<decltype(v.x)>::type));
        STATIC_ASSERT(nbl::hlsl::impl::is_reference<decltype(v.x)>::value);
        STATIC_ASSERT(IS_SAME(float,nbl::hlsl::impl::remove_reference<decltype(v.x)>::type));
        STATIC_ASSERT(IS_SAME(decltype(v.x),nbl::hlsl::impl::add_lvalue_reference<float>::type));
        STATIC_ASSERT(IS_SAME(decltype(v.x),nbl::hlsl::impl::add_lvalue_reference<nbl::hlsl::impl::remove_reference<decltype(v.x)>::type>::type));
        
        STATIC_ASSERT(IS_SAME(float,nbl::hlsl::remove_cvref<decltype(v.x)>::type));
        STATIC_ASSERT(IS_SAME(nbl::hlsl::remove_cv<decltype(v.x)>::type,nbl::hlsl::impl::add_lvalue_reference<float>::type));
        STATIC_ASSERT(IS_SAME(nbl::hlsl::remove_cv<decltype(v.x)>::type,nbl::hlsl::impl::add_lvalue_reference<nbl::hlsl::remove_cvref<decltype(v.x)>::type>::type));
    }
    fill(invocationID, 7);
    {
        float x[4][4];
        STATIC_ASSERT(IS_SAME(nbl::hlsl::remove_extent<decltype(x)>::type, float[4]));
        STATIC_ASSERT(IS_SAME(nbl::hlsl::remove_all_extents<decltype(x)>::type, float));
    }
    fill(invocationID, 8);
    {
        STATIC_ASSERT(IS_SAME(nbl::hlsl::make_signed<int16_t>::type,   nbl::hlsl::make_signed<uint16_t>::type));
        STATIC_ASSERT(IS_SAME(nbl::hlsl::make_unsigned<int16_t>::type, nbl::hlsl::make_unsigned<uint16_t>::type));

        STATIC_ASSERT(IS_SAME(nbl::hlsl::make_signed<int32_t>::type,   nbl::hlsl::make_signed<uint32_t>::type));
        STATIC_ASSERT(IS_SAME(nbl::hlsl::make_unsigned<int32_t>::type, nbl::hlsl::make_unsigned<uint32_t>::type));

        STATIC_ASSERT(IS_SAME(nbl::hlsl::make_signed<int64_t>::type,   nbl::hlsl::make_signed<uint64_t>::type));
        STATIC_ASSERT(IS_SAME(nbl::hlsl::make_unsigned<int64_t>::type, nbl::hlsl::make_unsigned<uint64_t>::type));
    }

    {
        int Q[3][4][5];
        STATIC_ASSERT(3 == (nbl::hlsl::extent<decltype(Q), 0>::value));
        STATIC_ASSERT(4 == (nbl::hlsl::extent<decltype(Q), 1>::value));
        STATIC_ASSERT(5 == (nbl::hlsl::extent<decltype(Q), 2>::value));
        STATIC_ASSERT(0 == (nbl::hlsl::extent<decltype(Q), 3>::value));
    }
    fill(invocationID, 9);
    {
        float32_t a;

        a = nbl::hlsl::numeric_limits<float32_t>::min;
        a = nbl::hlsl::numeric_limits<float32_t>::max;
        a = nbl::hlsl::numeric_limits<float32_t>::lowest;
        a = nbl::hlsl::numeric_limits<float32_t>::epsilon;
        a = nbl::hlsl::numeric_limits<float32_t>::round_error;
        a = nbl::hlsl::numeric_limits<float32_t>::infinity;
        a = nbl::hlsl::numeric_limits<float32_t>::quiet_NaN;
        a = nbl::hlsl::numeric_limits<float32_t>::signaling_NaN;
        a = nbl::hlsl::numeric_limits<float32_t>::denorm_min;
    }

    fill(invocationID, 10);

    fill(invocationID, -1);
}
