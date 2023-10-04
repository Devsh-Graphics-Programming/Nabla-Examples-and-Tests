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
#include <nbl/builtin/hlsl/barycentric/utils.hlsl>

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


template<class T>
constexpr bool val(T a)
{
    return std::is_const_v<T>;
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
