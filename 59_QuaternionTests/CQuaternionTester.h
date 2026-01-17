#ifndef _NBL_EXAMPLES_TESTS_59_QUATERNION_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_59_QUATERNION_TESTER_INCLUDED_

#define GLM_FORCE_RADIANS
#include <glm/detail/type_quat.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

#include "nbl/examples/examples.hpp"
#include "app_resources/common.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include <nbl/builtin/hlsl/testing/orientation_compare.hlsl>
#include <nbl/builtin/hlsl/testing/vector_length_compare.hlsl>

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
        std::uniform_real_distribution<float> realDistribution(-1.0f, 1.0f);
        std::uniform_real_distribution<float> realDistribution01(0.0f, 1.0f);
        std::uniform_real_distribution<float> realDistributionRad(-numbers::pi<float>, numbers::pi<float>);

        QuaternionInputTestValues testInput;
        testInput.axis = hlsl::normalize(float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())));
        testInput.angle = realDistributionRad(getRandomEngine());
        testInput.quat0 = math::quaternion<float>::create(float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())), realDistribution(getRandomEngine()));
        testInput.quat0 = hlsl::normalize(testInput.quat0);
        testInput.quat1 = math::quaternion<float>::create(float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())), realDistribution(getRandomEngine()));
        testInput.quat1 = hlsl::normalize(testInput.quat1);
        testInput.quat2 = testInput.quat0 * realDistribution(getRandomEngine()) * 1000.f;
        testInput.quat3 = testInput.quat1 * realDistribution(getRandomEngine()) * 1000.f;
        testInput.pitch = realDistributionRad(getRandomEngine());
        testInput.yaw = realDistributionRad(getRandomEngine());
        testInput.roll = realDistributionRad(getRandomEngine());
        testInput.rotationMat = float32_t3x3(glm::rotate(realDistributionRad(getRandomEngine()), hlsl::normalize(float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())))));
        testInput.scaleFactor = realDistribution01(getRandomEngine()) * 1000.f;

        testInput.scaleRotationMat = testInput.rotationMat; 
        testInput.scaleRotationMat *= testInput.scaleFactor;

        testInput.interpolationFactor = realDistribution01(getRandomEngine());
        testInput.someVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));

        return testInput;
    }

    QuaternionTestValues determineExpectedResults(const QuaternionInputTestValues& testInput) override
    {
        const auto glmquat0 = glm::quat(testInput.quat0.data.w, testInput.quat0.data.x, testInput.quat0.data.y, testInput.quat0.data.z);
        const auto glmquat1 = glm::quat(testInput.quat1.data.w, testInput.quat1.data.x, testInput.quat1.data.y, testInput.quat1.data.z);
        const auto glmquat2 = glm::quat(testInput.quat2.data.w, testInput.quat2.data.x, testInput.quat2.data.y, testInput.quat2.data.z);
        const auto glmquat3 = glm::quat(testInput.quat3.data.w, testInput.quat3.data.x, testInput.quat3.data.y, testInput.quat3.data.z);

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
            glm::mat3x3 rotmat;
            rotmat[0] = testInput.rotationMat[0];
            rotmat[1] = testInput.rotationMat[1];
            rotmat[2] = testInput.rotationMat[2];
            const auto glmquat = glm::quat_cast(rotmat);
            expected.quatFromMat.data.x = glmquat.data.data[0];
            expected.quatFromMat.data.y = glmquat.data.data[1];
            expected.quatFromMat.data.z = glmquat.data.data[2];
            expected.quatFromMat.data.w = glmquat.data.data[3];

            expected.quatFromScaledMat.data = hlsl::normalize(expected.quatFromMat.data) * testInput.scaleFactor;
        }
        {
            const auto rotmat = glm::transpose(glm::mat3_cast(glmquat0));
            expected.rotationMat[0] = rotmat[0];
            expected.rotationMat[1] = rotmat[1];
            expected.rotationMat[2] = rotmat[2];
        }
        {
            const auto rotmat = transpose(glm::mat3_cast(glmquat2));
            expected.scaleRotationMat[0] = rotmat[0];
            expected.scaleRotationMat[1] = rotmat[1];
            expected.scaleRotationMat[2] = rotmat[2];
        }
        {
            const auto mult = glmquat0 * glmquat1;
            expected.quatMult.data.x = mult.data.data[0];
            expected.quatMult.data.y = mult.data.data[1];
            expected.quatMult.data.z = mult.data.data[2];
            expected.quatMult.data.w = mult.data.data[3];
        }
        {
            const auto slerped = glm::slerp(glmquat0, glmquat1, testInput.interpolationFactor);
            expected.quatSlerp.data.x = slerped.data.data[0];
            expected.quatSlerp.data.y = slerped.data.data[1];
            expected.quatSlerp.data.z = slerped.data.data[2];
            expected.quatSlerp.data.w = slerped.data.data[3];

            expected.quatFlerp.data = expected.quatSlerp.data;
        }
        {
            const auto mult = glmquat2 * glmquat3;
            expected.quatScaledMult.data.x = mult.data.data[0];
            expected.quatScaledMult.data.y = mult.data.data[1];
            expected.quatScaledMult.data.z = mult.data.data[2];
            expected.quatScaledMult.data.w = mult.data.data[3];
        }
        expected.transformedVec = glmquat0 * testInput.someVec;

        return expected;
    }

    void verifyTestResults(const QuaternionTestValues& expectedTestValues, const QuaternionTestValues& testValues, const size_t testIteration, const uint32_t seed, TestType testType) override
    {
        verifyVectorTestValue("create from axis angle", expectedTestValues.quatFromAngleAxis.data, testValues.quatFromAngleAxis.data, testIteration, seed, testType, 1e-2, true);
        verifyVectorTestValue("create from Euler angles", expectedTestValues.quatFromEulerAngles.data, testValues.quatFromEulerAngles.data, testIteration, seed, testType, 1e-2, true);
        verifyVectorTestValue("create from rotation matrix", expectedTestValues.quatFromMat.data, testValues.quatFromMat.data, testIteration, seed, testType, 1e-2, true);
        verifyScaledVectorTestValue("create from scale rotation matrix", expectedTestValues.quatFromScaledMat.data, testValues.quatFromScaledMat.data, testIteration, seed, testType, 1e-4, 1e-2);

        verifyTestValue("construct matrix", expectedTestValues.rotationMat, testValues.rotationMat, testIteration, seed, testType, 1e-2);
        verifyTestValue("construct matrix (scaled)", expectedTestValues.scaleRotationMat, testValues.scaleRotationMat, testIteration, seed, testType, 1e-2);

        verifyVectorTestValue("multiply quat", expectedTestValues.quatMult.data, testValues.quatMult.data, testIteration, seed, testType, 1e-2, true);
        verifyVectorTestValue("slerp quat", expectedTestValues.quatSlerp.data, testValues.quatSlerp.data, testIteration, seed, testType, 1e-2, true);
        verifyVectorTestValue("flerp quat", expectedTestValues.quatFlerp.data, testValues.quatFlerp.data, testIteration, seed, testType, 1e-1, true);
        verifyTestValue("transform vector", expectedTestValues.transformedVec, testValues.transformedVec, testIteration, seed, testType, 1e-2);

        verifyScaledVectorTestValue("multiply scaled quat", expectedTestValues.quatScaledMult.data, testValues.quatScaledMult.data, testIteration, seed, testType, 1e-4, 1e-2);
    }

    template<typename T>
    void verifyScaledVectorTestValue(const std::string& memberName, const T& expectedVal, const T& testVal,
        const size_t testIteration, const uint32_t seed, const TestType testType, const float64_t maxAbsoluteDifference, const float64_t maxRelativeDifference)
    {
        if (nbl::hlsl::testing::orientationCompare(expectedVal, testVal, maxRelativeDifference) &&
            nbl::hlsl::testing::vectorLengthCompare(expectedVal, testVal, maxAbsoluteDifference, maxRelativeDifference))
            return;

        printTestFail<T>(memberName, expectedVal, testVal, testIteration, seed, testType);
    }

    template<typename T>
    void verifyVectorTestValue(const std::string& memberName, const T& expectedVal, const T& testVal,
        const size_t testIteration, const uint32_t seed, const TestType testType, const float64_t maxAllowedDifference, const bool testOrientation)
    {
        if (compareVectorTestValues<T>(expectedVal, testVal, maxAllowedDifference, testOrientation))
            return;

        printTestFail<T>(memberName, expectedVal, testVal, testIteration, seed, testType);
    }

    template<typename T> requires concepts::FloatingPointLikeVectorial<T>
    bool compareVectorTestValues(const T& lhs, const T& rhs, const float64_t maxAllowedDifference, const bool testOrientation)
    {
        if (testOrientation)
            return nbl::hlsl::testing::orientationCompare(lhs, rhs, maxAllowedDifference);
        return nbl::hlsl::testing::relativeApproxCompare(lhs, rhs, maxAllowedDifference);
    }
};

#endif
