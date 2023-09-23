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

#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl>

using namespace nbl;
using namespace core;
using namespace ui;
using namespace hlsl;

// encodeCIEXYZ.hlsl matrices
constexpr glm::mat3 nbl_glsl_scRGBtoXYZ = glm::mat3(
    glm::vec3(0.4124564, 0.2126729, 0.0193339),
    glm::vec3(0.3575761, 0.7151522, 0.1191920),
    glm::vec3(0.1804375, 0.0721750, 0.9503041)
);

constexpr glm::mat3 nbl_glsl_Display_P3toXYZ = glm::mat3(
    glm::vec3(0.4865709, 0.2289746, 0.0000000),
    glm::vec3(0.2656677, 0.6917385, 0.0451134),
    glm::vec3(0.1982173, 0.0792869, 1.0439444)
);

constexpr glm::mat3 nbl_glsl_DCI_P3toXYZ = glm::mat3(
    glm::vec3(1.0, 0.0, 0.0),
    glm::vec3(0.0, 1.0, 0.0),
    glm::vec3(0.0, 0.0, 1.0)
);

constexpr glm::mat3 nbl_glsl_BT2020toXYZ = glm::mat3(
    glm::vec3(0.6369580, 0.2627002, 0.0000000),
    glm::vec3(0.1446169, 0.6779981, 0.0280727),
    glm::vec3(0.1688810, 0.0593017, 1.0609851)
);

constexpr glm::mat3 nbl_glsl_AdobeRGBtoXYZ = glm::mat3(
    glm::vec3(0.57667, 0.29734, 0.02703),
    glm::vec3(0.18556, 0.62736, 0.07069),
    glm::vec3(0.18823, 0.07529, 0.99134)
);

constexpr glm::mat3 nbl_glsl_ACES2065_1toXYZ = glm::mat3(
    glm::vec3(0.9525523959, 0.3439664498, 0.0000000000),
    glm::vec3(0.0000000000, 0.7281660966, 0.0000000000),
    glm::vec3(0.0000936786, -0.0721325464, 1.0088251844)
);

constexpr glm::mat3 nbl_glsl_ACEScctoXYZ = glm::mat3(
    glm::vec3(0.6624542, 0.2722287, -0.0055746),
    glm::vec3(0.1340042, 0.6740818, 0.6740818),
    glm::vec3(0.1561877, 0.0536895, 1.0103391)
);

// decodeCIEXYZ.hlsl matrices
constexpr glm::mat3 nbl_glsl_XYZtoscRGB = glm::mat3(
    glm::vec3(3.2404542, -0.9692660, 0.0556434),
    glm::vec3(-1.5371385, 1.8760108, -0.2040259),
    glm::vec3(-0.4985314, 0.0415560, 1.0572252)
);

constexpr glm::mat3 nbl_glsl_XYZtoDisplay_P3 = glm::mat3(
    glm::vec3(2.4934969, -0.8294890, 0.0358458),
    glm::vec3(-0.9313836, 1.7626641, -0.0761724),
    glm::vec3(-0.4027108, 0.0236247, 0.9568845)
);

constexpr glm::mat3 nbl_glsl_XYZtoDCI_P3 = glm::mat3(glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0));

constexpr glm::mat3 nbl_glsl_XYZtoBT2020 = glm::mat3(
    glm::vec3(1.7166512, -0.6666844, 0.0176399),
    glm::vec3(-0.3556708, 1.6164812, -0.0427706),
    glm::vec3(-0.2533663, 0.0157685, 0.9421031)
);

constexpr glm::mat3 nbl_glsl_XYZtoAdobeRGB = glm::mat3(
    glm::vec3(2.04159, -0.96924, 0.01344),
    glm::vec3(-0.56501, 1.87597, -0.11836),
    glm::vec3(-0.34473, 0.04156, 1.01517)
);

constexpr glm::mat3 nbl_glsl_XYZtoACES2065_1 = glm::mat3(
    glm::vec3(1.0498110175, 0.0000000000, -0.0000974845),
    glm::vec3(-0.4959030231, 1.3733130458, 0.0982400361),
    glm::vec3(0.0000000000, 0.0000000000, 0.9912520182)
);

constexpr glm::mat3 nbl_glsl_XYZtoACEScc = glm::mat3(
    glm::vec3(1.6410234, -0.6636629, 0.0117219),
    glm::vec3(-0.3248033, 1.6153316, -0.0082844),
    glm::vec3(-0.2364247, 0.0167563, 0.9883949)
);

constexpr uint32_t COLOR_MATRIX_CNT = 14u;
constexpr std::array<float3x3, COLOR_MATRIX_CNT> hlslColorMatrices = {
    colorspace::scRGBtoXYZ, colorspace::Display_P3toXYZ, colorspace::DCI_P3toXYZ,
    colorspace::BT2020toXYZ, colorspace::AdobeRGBtoXYZ, colorspace::ACES2065_1toXYZ,
    colorspace::ACEScctoXYZ, colorspace::decode::XYZtoscRGB, colorspace::decode::XYZtoDisplay_P3,
    colorspace::decode::XYZtoDCI_P3, colorspace::decode::XYZtoBT2020, colorspace::decode::XYZtoAdobeRGB,
    colorspace::decode::XYZtoACES2065_1, colorspace::decode::XYZtoACEScc
};
constexpr std::array<glm::mat3, COLOR_MATRIX_CNT> glslColorMatrices = {
    nbl_glsl_scRGBtoXYZ, nbl_glsl_Display_P3toXYZ, nbl_glsl_DCI_P3toXYZ,
    nbl_glsl_BT2020toXYZ, nbl_glsl_AdobeRGBtoXYZ, nbl_glsl_ACES2065_1toXYZ,
    nbl_glsl_ACEScctoXYZ, nbl_glsl_XYZtoscRGB, nbl_glsl_XYZtoDisplay_P3,
    nbl_glsl_XYZtoDCI_P3, nbl_glsl_XYZtoBT2020, nbl_glsl_XYZtoAdobeRGB,
    nbl_glsl_XYZtoACES2065_1, nbl_glsl_XYZtoACEScc
};

void testColorMatrices()
{
    constexpr std::array<float3, 3> unitVectors = {
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f),
        float3(0.0f, 0.0f, 1.0f)
    };

    for (uint32_t matrixIdx = 0u; matrixIdx < COLOR_MATRIX_CNT; matrixIdx++)
    {
        const auto& hlslMatrix = hlslColorMatrices[matrixIdx];
        const auto& glslMatrix = glslColorMatrices[matrixIdx];

        for (uint32_t i = 0u; i < 3u; i++)
        {
            assert(glslMatrix[i] == mul(hlslMatrix, float3(i == 0 ? 1.f : 0.f, i == 1 ? 1.f : 0.f, i == 2 ? 1.f : 0.f)));
            assert(mul(hlslMatrix, unitVectors[i]) == glslMatrix * unitVectors[i]);
        }
    }
}

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

    // color matrix tests:
    testColorMatrices();
}
