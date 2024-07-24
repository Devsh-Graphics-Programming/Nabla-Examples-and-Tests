//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/emulated_float64_t.hlsl>

NBL_CONSTEXPR uint32_t WORKGROUP_SIZE = 1;

using namespace nbl;
using namespace hlsl;

struct ConstructorTestValues
{
    int32_t int32;
    int64_t int64;
    uint32_t uint32;
    uint64_t uint64;
    float32_t float32;
    float64_t float64;
};

struct TestValues
{
    double a;
    double b;
    // constructors
    
    //nbl::hlsl::emulated_float64_t::storage_t int16CreateVal;
    nbl::hlsl::emulated_float64_t::storage_t int32CreateVal;
    nbl::hlsl::emulated_float64_t::storage_t int64CreateVal;
    
    // TODO:
    //nbl::hlsl::emulated_float64_t::storage_t uint16CreateVal;
    nbl::hlsl::emulated_float64_t::storage_t uint32CreateVal;
    nbl::hlsl::emulated_float64_t::storage_t uint64CreateVal;
    // TODO:
    //nbl::hlsl::emulated_float64_t::storage_t float16CreateVal;
    nbl::hlsl::emulated_float64_t::storage_t float32CreateVal;
    nbl::hlsl::emulated_float64_t::storage_t float64CreateVal;

    // arithmetic
    nbl::hlsl::emulated_float64_t::storage_t additionVal;
    nbl::hlsl::emulated_float64_t::storage_t substractionVal;
    nbl::hlsl::emulated_float64_t::storage_t multiplicationVal;
    nbl::hlsl::emulated_float64_t::storage_t divisionVal;

    // relational
    int lessOrEqualVal;
    int greaterOrEqualVal;
    int equalVal;
    int notEqualVal;
    int lessVal;
    int greaterVal;
};