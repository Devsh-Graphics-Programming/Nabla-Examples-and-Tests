//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/emulated_float64_t.hlsl>

NBL_CONSTEXPR uint32_t WORKGROUP_SIZE = 1;

struct TestValues
{
    // "constructors"
    emulated::emulated_float64_t::storage_t intCreateVal;
    emulated::emulated_float64_t::storage_t uintCreateVal;
    emulated::emulated_float64_t::storage_t uint64CreateVal;
    emulated::emulated_float64_t::storage_t floatCreateVal;
    emulated::emulated_float64_t::storage_t doubleCreateVal;
    //emulated::emulated_float64_t::create(min16int(2));

    // arithmetic
    emulated::emulated_float64_t::storage_t additionVal;
    emulated::emulated_float64_t::storage_t substractionVal;
    emulated::emulated_float64_t::storage_t multiplicationVal;
    emulated::emulated_float64_t::storage_t divisionVal;

    // relational
    bool lessOrEqualVal;
    bool greaterOrEqualVal;
    bool equalVal;
    bool notEqualVal;
    bool lessVal;
    bool greaterVal;

    // conversion
    bool convertionToBoolVal;
    int convertionToIntVal;
    uint32_t convertionToUint32Val;
    uint64_t convertionToUint64Val;
    float convertionToFloatVal;
    double convertionToDoubleVal;
    //bool convertionToHalfVal;
};