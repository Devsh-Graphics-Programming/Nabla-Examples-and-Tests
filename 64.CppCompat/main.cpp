// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// ERM THIS IS PROBABLY WRONG, consult Arek!
#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <assert.h>
#include <nabla.h>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/mpl.hlsl>
#include <nbl/builtin/hlsl/barycentric/utils.hlsl>


// Every header limits.hlsl includes need to be included before here for this to work
#undef _NBL_BUILTIN_HLSL_NUMERIC_LIMITS_INCLUDED_
#define __HLSL_VERSION
#include <nbl/builtin/hlsl/limits.hlsl>
#undef __HLSL_VERSION

#include "../common/CommonAPI.h"


using namespace nbl;
using namespace core;
using namespace ui;
using namespace nbl::hlsl;


struct S
{
    float32_t3 f;
};

struct T
{
    float32_t       a;
    float32_t3      b;
    S               c;
    float32_t2x3    d;
    float32_t2x3    e;
    int             f[3];
    float32_t2      g[2];
    float32_t4      h;
};

// numeric limits API
// is_specialized
// is_signed
// is_integer
// is_exact
// has_infinity
// has_quiet_NaN
// has_signaling_NaN
// has_denorm
// has_denorm_loss
// round_style
// is_iec559
// is_bounded
// is_modulo
// digits
// digits10
// max_digits10
// radix
// min_exponent
// min_exponent10
// max_exponent
// max_exponent10
// traps
// tinyness_before


template<class T>
constexpr bool val(T a)
{
    return std::is_const_v<T>;
}

template<class T> 
bool equal(T l, auto r)
{
    if constexpr (is_integral<T>::value)
        return l == r;
    
    return 0==memcmp(&l, &r, sizeof(T));
}



int main()
{
    // TODO: later this whole test should be templated so we can check all `T` not just `float`, but for this we need `type_traits`
  
    // DO NOT EVER THINK TO CHANGE `using type1 = vector<type,1>` to `using type1 = type` EVER!
    static_assert(!std::is_same_v<float32_t1,float32_t>);
    static_assert(!std::is_same_v<float64_t1,float64_t>);
    static_assert(!std::is_same_v<int32_t1,int32_t>);
    static_assert(!std::is_same_v<uint32_t1,uint32_t>);
    //static_assert(!std::is_same_v<vector<T,1>,T>);

    // checking matrix memory layout
    {
        float32_t4x3 a;
        float32_t3x4 b;
        float32_t3 v;
        float32_t4 u;
        mul(a, b);
        mul(b, a);
        mul(a, v);
        mul(v, b);
        mul(u, a);
        mul(b, u);

        float32_t4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        a - a;
        b + b;
        static_assert(std::is_same_v<float32_t4x4, decltype(mul(a, b))>);
        static_assert(std::is_same_v<float32_t3x3, decltype(mul(b, a))>);
        static_assert(std::is_same_v<float32_t4, decltype(mul(a, v))>);
        static_assert(std::is_same_v<float32_t4, decltype(mul(v, b))>);
        static_assert(std::is_same_v<float32_t3, decltype(mul(u, a))>);
        static_assert(std::is_same_v<float32_t3, decltype(mul(b, u))>);

    }

    // making sure linear operators returns the correct type
    static_assert(std::is_same_v<float32_t4x4, std::remove_cvref_t<decltype(float32_t4x4() = float32_t4x4())>>);
    static_assert(std::is_same_v<float32_t4x4, std::remove_cvref_t<decltype(float32_t4x4() + float32_t4x4())>>);
    static_assert(std::is_same_v<float32_t4x4, std::remove_cvref_t<decltype(float32_t4x4() - float32_t4x4())>>);
    static_assert(std::is_same_v<float32_t4x4, std::remove_cvref_t<decltype(mul(float32_t4x4(), float32_t4x4()))>>);

    // checking scalar packing
    static_assert(offsetof(T, a) == 0);
    static_assert(offsetof(T, b) == offsetof(T, a) + sizeof(T::a));
    static_assert(offsetof(T, c) == offsetof(T, b) + sizeof(T::b));
    static_assert(offsetof(T, d) == offsetof(T, c) + sizeof(T::c));
    static_assert(offsetof(T, e) == offsetof(T, d) + sizeof(T::d));
    static_assert(offsetof(T, f) == offsetof(T, e) + sizeof(T::e));
    static_assert(offsetof(T, g) == offsetof(T, f) + sizeof(T::f));
    static_assert(offsetof(T, h) == offsetof(T, g) + sizeof(T::g));
    
    // use some functions
    float32_t3 x;
    float32_t2x3 y;
    float32_t3x3 z;
    barycentric::reconstructBarycentrics(x, y);
    barycentric::reconstructBarycentrics(x, z);

    auto zero = cross(x,x);
    auto lenX2 = dot(x,x);
    float32_t3x3 z_inv = inverse(z);
    auto mid = lerp(x,x,0.5f);
    auto w = transpose(y);
    
   
    auto test_type_limits = []<class T>() 
    {
        using L = std::numeric_limits<T>;
        using R = nbl::hlsl::numeric_limits<T>;

        
        #define TEST_AND_LOG(var) \
            { \
                auto lhs = L::var; \
                auto rhs = R::var; \
                if(!equal(lhs, rhs)) \
                { \
                    std::cout << typeid(T).name() << " " << #var << " does not match : " << double(lhs) << " - " << double(rhs) << "\n"; \
                    abort(); \
                } \
            }

        TEST_AND_LOG(is_specialized);
        TEST_AND_LOG(is_signed);
        TEST_AND_LOG(is_integer);
        TEST_AND_LOG(is_exact);
        TEST_AND_LOG(has_infinity);
        TEST_AND_LOG(has_quiet_NaN);
        TEST_AND_LOG(has_signaling_NaN);
        TEST_AND_LOG(has_denorm);
        TEST_AND_LOG(has_denorm_loss);
        TEST_AND_LOG(round_style);
        TEST_AND_LOG(is_iec559);
        TEST_AND_LOG(is_bounded);
        TEST_AND_LOG(is_modulo);
        TEST_AND_LOG(digits);
        TEST_AND_LOG(digits10);
        TEST_AND_LOG(max_digits10);
        TEST_AND_LOG(radix);
        TEST_AND_LOG(min_exponent);
        TEST_AND_LOG(min_exponent10);
        TEST_AND_LOG(max_exponent);
        TEST_AND_LOG(max_exponent10);
        TEST_AND_LOG(traps);
        TEST_AND_LOG(tinyness_before);

        TEST_AND_LOG(min());
        TEST_AND_LOG(max());
        TEST_AND_LOG(lowest());
        TEST_AND_LOG(epsilon());
        TEST_AND_LOG(round_error());
        TEST_AND_LOG(infinity());
        TEST_AND_LOG(quiet_NaN());
        TEST_AND_LOG(signaling_NaN());
        TEST_AND_LOG(denorm_min());
    };

    test_type_limits.template operator()<float>();
    test_type_limits.template operator()<double>();
    test_type_limits.template operator()<int8_t>();
    test_type_limits.template operator()<int16_t>();
    test_type_limits.template operator()<int32_t>();
    test_type_limits.template operator()<int64_t>();
    test_type_limits.template operator()<uint8_t>();
    test_type_limits.template operator()<uint16_t>();
    test_type_limits.template operator()<uint32_t>();
    test_type_limits.template operator()<uint64_t>();
    

    // test HLSL side

    {
        auto initOutput = CommonAPI::InitWithDefaultExt(CommonAPI::InitParams {.appName = "64.CppCompat" });
        auto logger = std::move(initOutput.logger);
        auto assetManager = std::move(initOutput.assetManager);
        auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);

        const char* pathToShader = "../test.hlsl";
        core::smart_refctd_ptr<video::IGPUSpecializedShader> specializedShader = nullptr;
        {
            video::IGPUObjectFromAssetConverter CPU2GPU;
            asset::IAssetLoader::SAssetLoadParams params = {};
            params.logger = logger.get();
            auto specShader_cpu = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(pathToShader, params).getContents().begin());
            specializedShader = CPU2GPU.getGPUObjectsFromAssets(&specShader_cpu, &specShader_cpu + 1, cpu2gpuParams)->front();
        }
        assert(specializedShader);
    }
}
