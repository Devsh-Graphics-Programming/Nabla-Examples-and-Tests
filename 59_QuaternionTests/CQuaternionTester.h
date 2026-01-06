#ifndef _NBL_EXAMPLES_TESTS_59_QUATERNION_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_59_QUATERNION_TESTER_INCLUDED_


#include <glm/detail/type_quat.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>

#include "nbl/examples/examples.hpp"
#include "app_resources/common.hlsl"
#include "nbl/examples/Tester/ITester.h"

using namespace nbl;

class CQuaternionTester final : public ITester<QuaternionInputTestValues, QuaternionTestValues, QuaternionTestExecutor>
{
    using base_t = ITester<QuaternionInputTestValues, QuaternionTestValues, QuaternionTestExecutor>;

public:
    CQuaternionTester(const uint32_t testBatchCount)
        : base_t(testBatchCount) {};

private:
    QuaternionInputTestValues generateInputTestValues() override
    {
        std::uniform_real_distribution<float> realDistribution(-100.0f, 100.0f);
        std::uniform_real_distribution<float> realDistributionSmall(1.0f, 4.0f);
        std::uniform_int_distribution<int> intDistribution(-100, 100);
        std::uniform_int_distribution<int> coinFlipDistribution(0, 1);

        QuaternionInputTestValues testInput;
        testInput.axis = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.angle = realDistribution(getRandomEngine());
        testInput.quat0 = math::quaternion<float>::create(float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())), realDistribution(getRandomEngine()));
        testInput.quat1 = math::quaternion<float>::create(float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())), realDistribution(getRandomEngine()));
        testInput.pitch = realDistribution(getRandomEngine());
        testInput.yaw = realDistribution(getRandomEngine());
        testInput.roll = realDistribution(getRandomEngine());
        testInput.factor = realDistribution(getRandomEngine());

        return testInput;
    }

    QuaternionTestValues determineExpectedResults(const QuaternionInputTestValues& testInput) override
    {
        const auto glmquat0 = glm::quat(testInput.quat0.data.w, testInput.quat0.data.x, testInput.quat0.data.y, testInput.quat0.data.z);
        const auto glmquat1 = glm::quat(testInput.quat1.data.w, testInput.quat1.data.x, testInput.quat1.data.y, testInput.quat1.data.z);

        QuaternionTestValues expected;
        {
            const auto glmquat = glm::angleAxis(testInput.angle, testInput.axis);
            expected.quatFromAngleAxis.data.x = glmquat.data.data[0];
            expected.quatFromAngleAxis.data.y = glmquat.data.data[1];
            expected.quatFromAngleAxis.data.z = glmquat.data.data[2];
            expected.quatFromAngleAxis.data.w = glmquat.data.data[3];
        }
        {
            const auto rotmat = glm::yawPitchRoll(testInput.yaw, testInput.pitch, testInput.roll);
            const auto glmquat = glm::quat_cast(rotmat);
            expected.quatFromEulerAngles.data.x = glmquat.data.data[0];
            expected.quatFromEulerAngles.data.y = glmquat.data.data[1];
            expected.quatFromEulerAngles.data.z = glmquat.data.data[2];
            expected.quatFromEulerAngles.data.w = glmquat.data.data[3];
        }
        {
            const auto rotmat = glm::mat3_cast(glmquat0);
            expected.rotationMat[0] = rotmat[0];
            expected.rotationMat[1] = rotmat[1];
            expected.rotationMat[2] = rotmat[2];
        }
        {
            const auto mult = glmquat0 * glmquat1;
            expected.quatMult.data.x = mult.data.data[0];
            expected.quatMult.data.y = mult.data.data[1];
            expected.quatMult.data.z = mult.data.data[2];
            expected.quatMult.data.w = mult.data.data[3];
        }
        {
            const auto lerped = glm::lerp(glmquat0, glmquat1, testInput.factor);
            expected.quatLerp.data.x = lerped.data.data[0];
            expected.quatLerp.data.y = lerped.data.data[1];
            expected.quatLerp.data.z = lerped.data.data[2];
            expected.quatLerp.data.w = lerped.data.data[3];
        }

        return expected;
    }

    void verifyTestResults(const QuaternionTestValues& expectedTestValues, const QuaternionTestValues& testValues, const size_t testIteration, const uint32_t seed, TestType testType) override
    {
        verifyTestValue("create from axis angle", expectedTestValues.quatFromAngleAxis.data, testValues.quatFromAngleAxis.data, testIteration, seed, testType, 1e-2);
        verifyTestValue("create from Euler angles", expectedTestValues.quatFromEulerAngles.data, testValues.quatFromEulerAngles.data, testIteration, seed, testType, 1e-2);

        verifyTestValue("construct matrix", expectedTestValues.rotationMat, testValues.rotationMat, testIteration, seed, testType, 1e-2);

        verifyTestValue("multiply quat", expectedTestValues.quatMult.data, testValues.quatMult.data, testIteration, seed, testType, 1e-2);
        verifyTestValue("lerp quat", expectedTestValues.quatLerp.data, testValues.quatLerp.data, testIteration, seed, testType, 1e-2);
    }
};

#endif
