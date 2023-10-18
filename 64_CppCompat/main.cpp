// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <assert.h>
#include <nabla.h>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include <nbl/builtin/hlsl/barycentric/utils.hlsl>

using namespace nbl;
using namespace core;
using namespace ui;
using namespace hlsl;



struct S {
    float3 f;
};

struct T {
    float    a;
    float3   b;
    S        c;
    float2x3 d;
    float2x3 e;
    int      f[3];
    float2   g[2];
    float4   h;
};



int main()
{
    {
        float4x3 a;
        float3x4 b;
        float3 v;
        float4 u;
        mul(a, b);
        mul(b, a);
        mul(a, v);
        mul(v, b);
        mul(u, a);
        mul(b, u);

        float4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        a - a;
        b + b;
        static_assert(std::is_same_v<float4x4, decltype(mul(a, b))>);
        static_assert(std::is_same_v<float3x3, decltype(mul(b, a))>);
        static_assert(std::is_same_v<float4, decltype(mul(a, v))>);
        static_assert(std::is_same_v<float4, decltype(mul(v, b))>);
        static_assert(std::is_same_v<float3, decltype(mul(u, a))>);
        static_assert(std::is_same_v<float3, decltype(mul(b, u))>);

    }

    static_assert(std::is_same_v<float4x4, std::remove_cvref_t<decltype(float4x4() = float4x4())>>);
    static_assert(std::is_same_v<float4x4, std::remove_cvref_t<decltype(float4x4() + float4x4())>>);
    static_assert(std::is_same_v<float4x4, std::remove_cvref_t<decltype(float4x4() - float4x4())>>);
    static_assert(std::is_same_v<float4x4, std::remove_cvref_t<decltype(mul(float4x4(), float4x4()))>>);

    static_assert(offsetof(T, a) == 0);
    static_assert(offsetof(T, b) == offsetof(T, a) + sizeof(T::a));
    static_assert(offsetof(T, c) == offsetof(T, b) + sizeof(T::b));
    static_assert(offsetof(T, d) == offsetof(T, c) + sizeof(T::c));
    static_assert(offsetof(T, e) == offsetof(T, d) + sizeof(T::d));
    static_assert(offsetof(T, f) == offsetof(T, e) + sizeof(T::e));
    static_assert(offsetof(T, g) == offsetof(T, f) + sizeof(T::f));
    static_assert(offsetof(T, h) == offsetof(T, g) + sizeof(T::g));
    
    float3 x;
    float2x3 y;
    float3x3 z;
    barycentric::reconstructBarycentrics(x, y);
    barycentric::reconstructBarycentrics(x, z);

}
