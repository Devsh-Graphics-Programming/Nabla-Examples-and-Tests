//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/emulated/float64_t.hlsl>
#include <nbl/builtin/hlsl/portable/float64_t.hlsl>
#include <nbl/builtin/hlsl/portable/vector_t.hlsl>
#include <nbl/builtin/hlsl/portable/matrix_t.hlsl>

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t WORKGROUP_SIZE = 1;

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

struct PushConstants
{
    uint64_t a;
    uint64_t b;
    ConstructorTestValues constrTestVals;
};

template<bool FastMath, bool FlushDenormToZero>
struct TestValues
{
    uint64_t a;
    uint64_t b;
   
    // constructors
    typename nbl::hlsl::emulated_float64_t<FastMath, FlushDenormToZero>::storage_t int32CreateVal;
    typename nbl::hlsl::emulated_float64_t<FastMath, FlushDenormToZero>::storage_t int64CreateVal;
    typename nbl::hlsl::emulated_float64_t<FastMath, FlushDenormToZero>::storage_t uint32CreateVal;
    typename nbl::hlsl::emulated_float64_t<FastMath, FlushDenormToZero>::storage_t uint64CreateVal;
    typename nbl::hlsl::emulated_float64_t<FastMath, FlushDenormToZero>::storage_t float32CreateVal;
    typename nbl::hlsl::emulated_float64_t<FastMath, FlushDenormToZero>::storage_t float64CreateVal;

    // arithmetic
    typename nbl::hlsl::emulated_float64_t<FastMath, FlushDenormToZero>::storage_t additionVal;
    typename nbl::hlsl::emulated_float64_t<FastMath, FlushDenormToZero>::storage_t substractionVal;
    typename nbl::hlsl::emulated_float64_t<FastMath, FlushDenormToZero>::storage_t multiplicationVal;
    typename nbl::hlsl::emulated_float64_t<FastMath, FlushDenormToZero>::storage_t divisionVal;

    // relational
    int lessOrEqualVal;
    int greaterOrEqualVal;
    int equalVal;
    int notEqualVal;
    int lessVal;
    int greaterVal;
};