//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_59_QUATERNION_COMMON_INCLUDED_
#define _NBL_EXAMPLES_TESTS_59_QUATERNION_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/math/quaternions.hlsl>

using namespace nbl::hlsl;
struct QuaternionInputTestValues
{
    math::quaternion<float> quat0;
    math::quaternion<float> quat1;
    math::quaternion<float> quat2;
    math::quaternion<float> quat3;
    float32_t3 axis;
    float angle;
    float pitch;
    float yaw;
    float roll;
    float32_t3x3 rotationMat;
    float scaleFactor;
    float32_t3x3 scaleRotationMat;
    float interpolationFactor;
    float32_t3 someVec;
};

struct QuaternionTestValues
{
    math::quaternion<float> quatFromAngleAxis;
    math::quaternion<float> quatFromEulerAngles;
    math::quaternion<float> quatFromMat;
    math::quaternion<float> quatFromScaledMat;
    float32_t3x3 rotationMat;
    float32_t3x3 scaleRotationMat;
    math::quaternion<float> quatMult;
    math::quaternion<float> quatSlerp;
    math::quaternion<float> quatFlerp;
    math::quaternion<float> quatScaledMult;
    float32_t3 transformedVec;
};

struct QuaternionTestExecutor
{
    void operator()(NBL_CONST_REF_ARG(QuaternionInputTestValues) input, NBL_REF_ARG(QuaternionTestValues) output)
    {
        output.quatFromAngleAxis = math::quaternion<float>::create(input.axis, input.angle);
        output.quatFromEulerAngles = math::quaternion<float>::create(input.pitch, input.yaw, input.roll);
        output.quatFromMat = math::quaternion<float>::create(input.rotationMat);
        output.quatFromScaledMat = math::quaternion<float>::create(input.scaleRotationMat);

        output.rotationMat = _static_cast<float32_t3x3>(input.quat0);
        output.scaleRotationMat = _static_cast<float32_t3x3>(input.quat2);

        output.quatMult = input.quat0 * input.quat1;
        output.quatSlerp = math::quaternion<float>::slerp(input.quat0, input.quat1, input.interpolationFactor);
        output.quatFlerp = math::quaternion<float>::flerp(input.quat0, input.quat1, input.interpolationFactor);
        output.transformedVec = input.quat0.transformVector(input.someVec, true);

        output.quatScaledMult = input.quat2 * input.quat3;
    }
};

#endif
