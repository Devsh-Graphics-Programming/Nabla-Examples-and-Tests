//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_59_QUATERNION_COMMON_INCLUDED_
#define _NBL_EXAMPLES_TESTS_59_QUATERNION_COMMON_INCLUDED_

// because DXC doesn't properly support `_Static_assert`
// TODO: add a message, and move to macros.h or cpp_compat
#define STATIC_ASSERT(...) { nbl::hlsl::conditional<__VA_ARGS__, int, void>::type a = 0; }

#include <boost/preprocessor.hpp>

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


#include <nbl/builtin/hlsl/barycentric/utils.hlsl>
#include <nbl/builtin/hlsl/member_test_macros.hlsl>
#include <nbl/builtin/hlsl/device_capabilities_traits.hlsl>

#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>

#include <nbl/builtin/hlsl/math/quaternions.hlsl>

using namespace nbl::hlsl;
struct QuaternionInputTestValues
{
    math::quaternion<float> quat0;
    math::quaternion<float> quat1;
    float32_t3 axis;
    float angle;
    float pitch;
    float yaw;
    float roll;
    float32_t3x3 rotationMat;
    float factor;
    float32_t3 someVec;
};

struct QuaternionTestValues
{
    math::quaternion<float> quatFromAngleAxis;
    math::quaternion<float> quatFromEulerAngles;
    math::quaternion<float> quatFromMat;
    float32_t3x3 rotationMat;
    math::quaternion<float> quatMult;
    math::quaternion<float> quatSlerp;
    float32_t3 transformedVec;
};

struct QuaternionTestExecutor
{
    void operator()(NBL_CONST_REF_ARG(QuaternionInputTestValues) input, NBL_REF_ARG(QuaternionTestValues) output)
    {
        output.quatFromAngleAxis = math::quaternion<float>::create(input.axis, input.angle);
        output.quatFromEulerAngles = math::quaternion<float>::create(input.pitch, input.yaw, input.roll);
        output.quatFromMat = math::quaternion<float>::create(input.rotationMat);
        output.rotationMat = _static_cast<float32_t3x3>(input.quat0);
        output.quatMult = input.quat0 * input.quat1;
        output.quatSlerp = math::quaternion<float>::slerp(input.quat0, input.quat1, input.factor);
        output.transformedVec = input.quat0.transformVector(input.someVec, true);
    }
};

#endif
