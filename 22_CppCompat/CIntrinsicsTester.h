#ifndef _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_C_INTRINSICS_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_C_INTRINSICS_TESTER_INCLUDED_


#include "nbl/examples/examples.hpp"

#include "app_resources/common.hlsl"
#include "ITester.h"


using namespace nbl;

class CIntrinsicsTester final : public ITester
{
public:
    void performTests()
    {
        std::random_device rd;
        std::mt19937 mt(rd());

        std::uniform_real_distribution<float> realDistributionNeg(-50.0f, -1.0f);
        std::uniform_real_distribution<float> realDistributionPos(1.0f, 50.0f);
        std::uniform_real_distribution<float> realDistributionZeroToOne(0.0f, 1.0f);
        std::uniform_real_distribution<float> realDistribution(-100.0f, 100.0f);
        std::uniform_real_distribution<float> realDistributionSmall(1.0f, 4.0f);
        std::uniform_int_distribution<int> intDistribution(-100, 100);
        std::uniform_int_distribution<uint32_t> uintDistribution(0, 100);

        m_logger->log("intrinsics.hlsl TESTS:", system::ILogger::ELL_PERFORMANCE);
        for (int i = 0; i < Iterations; ++i)
        {
            // Set input thest values that will be used in both CPU and GPU tests
            IntrinsicsIntputTestValues testInput;
            testInput.bitCount = intDistribution(mt);
            testInput.crossLhs = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.crossRhs = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.clampVal = realDistribution(mt);
            testInput.clampMin = realDistributionNeg(mt);
            testInput.clampMax = realDistributionPos(mt);
            testInput.length = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.normalize = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.dotLhs = float32_t3(realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt));
            testInput.dotRhs = float32_t3(realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt));
            testInput.determinant = float32_t3x3(
                realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt),
                realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt),
                realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt)
            );
            testInput.findMSB = realDistribution(mt);
            testInput.findLSB = realDistribution(mt);
            testInput.inverse = float32_t3x3(
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt)
            );
            testInput.transpose = float32_t3x3(
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt)
            );
            testInput.mulLhs = float32_t3x3(
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt)
            );
            testInput.mulRhs = float32_t3x3(
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt)
            );
            testInput.minA = realDistribution(mt);
            testInput.minB = realDistribution(mt);
            testInput.maxA = realDistribution(mt);
            testInput.maxB = realDistribution(mt);
            testInput.rsqrt = realDistributionPos(mt);
            testInput.bitReverse = realDistribution(mt);
            testInput.frac = realDistribution(mt);
            testInput.mixX = realDistributionNeg(mt);
            testInput.mixY = realDistributionPos(mt);
            testInput.mixA = realDistributionZeroToOne(mt);
            testInput.sign = realDistribution(mt);
            testInput.radians = realDistribution(mt);
            testInput.degrees = realDistribution(mt);
            testInput.stepEdge = realDistribution(mt);
            testInput.stepX = realDistribution(mt);
            testInput.smoothStepEdge0 = realDistributionNeg(mt);
            testInput.smoothStepEdge1 = realDistributionPos(mt);
            testInput.smoothStepX = realDistribution(mt);
            testInput.addCarryA = std::numeric_limits<uint32_t>::max() - uintDistribution(mt);
            testInput.addCarryB = uintDistribution(mt);
            testInput.subBorrowA = uintDistribution(mt);
            testInput.subBorrowB = uintDistribution(mt);

            testInput.bitCountVec = int32_t3(intDistribution(mt), intDistribution(mt), intDistribution(mt));
            testInput.clampValVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.clampMinVec = float32_t3(realDistributionNeg(mt), realDistributionNeg(mt), realDistributionNeg(mt));
            testInput.clampMaxVec = float32_t3(realDistributionPos(mt), realDistributionPos(mt), realDistributionPos(mt));
            testInput.findMSBVec = uint32_t3(uintDistribution(mt), uintDistribution(mt), uintDistribution(mt));
            testInput.findLSBVec = uint32_t3(uintDistribution(mt), uintDistribution(mt), uintDistribution(mt));
            testInput.minAVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.minBVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.maxAVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.maxBVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.rsqrtVec = float32_t3(realDistributionPos(mt), realDistributionPos(mt), realDistributionPos(mt));
            testInput.bitReverseVec = uint32_t3(uintDistribution(mt), uintDistribution(mt), uintDistribution(mt));
            testInput.fracVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.mixXVec = float32_t3(realDistributionNeg(mt), realDistributionNeg(mt), realDistributionNeg(mt));
            testInput.mixYVec = float32_t3(realDistributionPos(mt), realDistributionPos(mt), realDistributionPos(mt));
            testInput.mixAVec = float32_t3(realDistributionZeroToOne(mt), realDistributionZeroToOne(mt), realDistributionZeroToOne(mt));

            testInput.signVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.radiansVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.degreesVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.stepEdgeVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.stepXVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.smoothStepEdge0Vec = float32_t3(realDistributionNeg(mt), realDistributionNeg(mt), realDistributionNeg(mt));
            testInput.smoothStepEdge1Vec = float32_t3(realDistributionPos(mt), realDistributionPos(mt), realDistributionPos(mt));
            testInput.smoothStepXVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.faceForwardN = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.faceForwardI = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.faceForwardNref = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.reflectI = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.reflectN = glm::normalize(float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt)));
            testInput.refractI = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.refractN = glm::normalize(float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt)));
            testInput.refractEta = realDistribution(mt);
            testInput.addCarryAVec = uint32_t3(std::numeric_limits<uint32_t>::max() - uintDistribution(mt), std::numeric_limits<uint32_t>::max() - uintDistribution(mt), std::numeric_limits<uint32_t>::max() - uintDistribution(mt));
            testInput.addCarryBVec = uint32_t3(uintDistribution(mt), uintDistribution(mt), uintDistribution(mt));
            testInput.subBorrowAVec = uint32_t3(uintDistribution(mt), uintDistribution(mt), uintDistribution(mt));
            testInput.subBorrowBVec = uint32_t3(uintDistribution(mt), uintDistribution(mt), uintDistribution(mt));

            // use std library or glm functions to determine expected test values, the output of functions from intrinsics.hlsl will be verified against these values
            IntrinsicsTestValues expected;
            expected.bitCount = glm::bitCount(testInput.bitCount);
            expected.clamp = glm::clamp(testInput.clampVal, testInput.clampMin, testInput.clampMax);
            expected.length = glm::length(testInput.length);
            expected.dot = glm::dot(testInput.dotLhs, testInput.dotRhs);
            expected.determinant = glm::determinant(reinterpret_cast<typename float32_t3x3::Base const&>(testInput.determinant));
            expected.findMSB = glm::findMSB(testInput.findMSB);
            expected.findLSB = glm::findLSB(testInput.findLSB);
            expected.min = glm::min(testInput.minA, testInput.minB);
            expected.max = glm::max(testInput.maxA, testInput.maxB);
            expected.rsqrt = (1.0f / std::sqrt(testInput.rsqrt));
            expected.mix = std::lerp(testInput.mixX, testInput.mixY, testInput.mixA);
            expected.sign = glm::sign(testInput.sign);
            expected.radians = glm::radians(testInput.radians);
            expected.degrees = glm::degrees(testInput.degrees);
            expected.step = glm::step(testInput.stepEdge, testInput.stepX);
            expected.smoothStep = glm::smoothstep(testInput.smoothStepEdge0, testInput.smoothStepEdge1, testInput.smoothStepX);

            expected.addCarry.result = glm::uaddCarry(testInput.addCarryA, testInput.addCarryB, expected.addCarry.carry);
            expected.subBorrow.result = glm::usubBorrow(testInput.subBorrowA, testInput.subBorrowB, expected.subBorrow.borrow);

            expected.frac = testInput.frac - std::floor(testInput.frac);
            expected.bitReverse = glm::bitfieldReverse(testInput.bitReverse);

            expected.normalize = glm::normalize(testInput.normalize);
            expected.cross = glm::cross(testInput.crossLhs, testInput.crossRhs);
            expected.bitCountVec = int32_t3(glm::bitCount(testInput.bitCountVec.x), glm::bitCount(testInput.bitCountVec.y), glm::bitCount(testInput.bitCountVec.z));
            expected.clampVec = float32_t3(
                glm::clamp(testInput.clampValVec.x, testInput.clampMinVec.x, testInput.clampMaxVec.x),
                glm::clamp(testInput.clampValVec.y, testInput.clampMinVec.y, testInput.clampMaxVec.y),
                glm::clamp(testInput.clampValVec.z, testInput.clampMinVec.z, testInput.clampMaxVec.z)
            );
            expected.findMSBVec = glm::findMSB(testInput.findMSBVec);
            expected.findLSBVec = glm::findLSB(testInput.findLSBVec);
            expected.minVec = float32_t3(
                glm::min(testInput.minAVec.x, testInput.minBVec.x),
                glm::min(testInput.minAVec.y, testInput.minBVec.y),
                glm::min(testInput.minAVec.z, testInput.minBVec.z)
            );
            expected.maxVec = float32_t3(
                glm::max(testInput.maxAVec.x, testInput.maxBVec.x),
                glm::max(testInput.maxAVec.y, testInput.maxBVec.y),
                glm::max(testInput.maxAVec.z, testInput.maxBVec.z)
            );
            expected.rsqrtVec = float32_t3(1.0f / std::sqrt(testInput.rsqrtVec.x), 1.0f / std::sqrt(testInput.rsqrtVec.y), 1.0f / std::sqrt(testInput.rsqrtVec.z));
            expected.bitReverseVec = glm::bitfieldReverse(testInput.bitReverseVec);
            expected.fracVec = float32_t3(
                testInput.fracVec.x - std::floor(testInput.fracVec.x),
                testInput.fracVec.y - std::floor(testInput.fracVec.y),
                testInput.fracVec.z - std::floor(testInput.fracVec.z));
            expected.mixVec.x = std::lerp(testInput.mixXVec.x, testInput.mixYVec.x, testInput.mixAVec.x);
            expected.mixVec.y = std::lerp(testInput.mixXVec.y, testInput.mixYVec.y, testInput.mixAVec.y);
            expected.mixVec.z = std::lerp(testInput.mixXVec.z, testInput.mixYVec.z, testInput.mixAVec.z);

            expected.signVec = glm::sign(testInput.signVec);
            expected.radiansVec = glm::radians(testInput.radiansVec);
            expected.degreesVec = glm::degrees(testInput.degreesVec);
            expected.stepVec = glm::step(testInput.stepEdgeVec, testInput.stepXVec);
            expected.smoothStepVec = glm::smoothstep(testInput.smoothStepEdge0Vec, testInput.smoothStepEdge1Vec, testInput.smoothStepXVec);
            expected.faceForward = glm::faceforward(testInput.faceForwardN, testInput.faceForwardI, testInput.faceForwardNref);
            expected.reflect = glm::reflect(testInput.reflectI, testInput.reflectN);
            expected.refract = glm::refract(testInput.refractI, testInput.refractN, testInput.refractEta);

            expected.addCarryVec.result = glm::uaddCarry(testInput.addCarryAVec, testInput.addCarryBVec, expected.addCarryVec.carry);
            expected.subBorrowVec.result = glm::usubBorrow(testInput.subBorrowAVec, testInput.subBorrowBVec, expected.subBorrowVec.borrow);

            auto mulGlm = nbl::hlsl::mul(testInput.mulLhs, testInput.mulRhs);
            expected.mul = reinterpret_cast<float32_t3x3&>(mulGlm);
            auto transposeGlm = glm::transpose(reinterpret_cast<typename float32_t3x3::Base const&>(testInput.transpose));
            expected.transpose = reinterpret_cast<float32_t3x3&>(transposeGlm);
            auto inverseGlm = glm::inverse(reinterpret_cast<typename float32_t3x3::Base const&>(testInput.inverse));
            expected.inverse = reinterpret_cast<float32_t3x3&>(inverseGlm);

            performCpuTests(testInput, expected);
            performGpuTests(testInput, expected);
        }
        m_logger->log("intrinsics.hlsl TESTS DONE.", system::ILogger::ELL_PERFORMANCE);
    }

private:
    inline static constexpr int Iterations = 100u;

    void performCpuTests(const IntrinsicsIntputTestValues& commonTestInputValues, const IntrinsicsTestValues& expectedTestValues)
    {
        IntrinsicsTestValues cpuTestValues;

        cpuTestValues.fillTestValues(commonTestInputValues);
        verifyTestValues(expectedTestValues, cpuTestValues, ITester::TestType::CPU);

    }

    void performGpuTests(const IntrinsicsIntputTestValues& commonTestInputValues, const IntrinsicsTestValues& expectedTestValues)
    {
        IntrinsicsTestValues gpuTestValues;
        gpuTestValues = dispatch<IntrinsicsIntputTestValues, IntrinsicsTestValues>(commonTestInputValues);
        verifyTestValues(expectedTestValues, gpuTestValues, ITester::TestType::GPU);
    }

    void verifyTestValues(const IntrinsicsTestValues& expectedTestValues, const IntrinsicsTestValues& testValues, ITester::TestType testType)
    {
        verifyTestValue("bitCount", expectedTestValues.bitCount, testValues.bitCount, testType);
        verifyTestValue("clamp", expectedTestValues.clamp, testValues.clamp, testType);
        verifyTestValue("length", expectedTestValues.length, testValues.length, testType);
        verifyTestValue("dot", expectedTestValues.dot, testValues.dot, testType);
        verifyTestValue("determinant", expectedTestValues.determinant, testValues.determinant, testType);
        verifyTestValue("findMSB", expectedTestValues.findMSB, testValues.findMSB, testType);
        verifyTestValue("findLSB", expectedTestValues.findLSB, testValues.findLSB, testType);
        verifyTestValue("min", expectedTestValues.min, testValues.min, testType);
        verifyTestValue("max", expectedTestValues.max, testValues.max, testType);
        verifyTestValue("rsqrt", expectedTestValues.rsqrt, testValues.rsqrt, testType);
        verifyTestValue("frac", expectedTestValues.frac, testValues.frac, testType);
        verifyTestValue("bitReverse", expectedTestValues.bitReverse, testValues.bitReverse, testType);
        verifyTestValue("mix", expectedTestValues.mix, testValues.mix, testType);
        verifyTestValue("sign", expectedTestValues.sign, testValues.sign, testType);
        verifyTestValue("radians", expectedTestValues.radians, testValues.radians, testType);
        verifyTestValue("degrees", expectedTestValues.degrees, testValues.degrees, testType);
        verifyTestValue("step", expectedTestValues.step, testValues.step, testType);
        verifyTestValue("smoothStep", expectedTestValues.smoothStep, testValues.smoothStep, testType);
        verifyTestValue("addCarryResult", expectedTestValues.addCarry.result, testValues.addCarry.result, testType);
        verifyTestValue("addCarryCarry", expectedTestValues.addCarry.carry, testValues.addCarry.carry, testType);
        // Disabled: current glm implementation is wrong
        //verifyTestValue("subBorrowResult", expectedTestValues.subBorrow.result, testValues.subBorrow.result, testType);
        //verifyTestValue("subBorrowBorrow", expectedTestValues.subBorrow.borrow, testValues.subBorrow.borrow, testType);

        verifyTestVector3dValue("normalize", expectedTestValues.normalize, testValues.normalize, testType);
        verifyTestVector3dValue("cross", expectedTestValues.cross, testValues.cross, testType);
        verifyTestVector3dValue("bitCountVec", expectedTestValues.bitCountVec, testValues.bitCountVec, testType);
        verifyTestVector3dValue("clampVec", expectedTestValues.clampVec, testValues.clampVec, testType);
        verifyTestVector3dValue("findMSBVec", expectedTestValues.findMSBVec, testValues.findMSBVec, testType);
        verifyTestVector3dValue("findLSBVec", expectedTestValues.findLSBVec, testValues.findLSBVec, testType);
        verifyTestVector3dValue("minVec", expectedTestValues.minVec, testValues.minVec, testType);
        verifyTestVector3dValue("maxVec", expectedTestValues.maxVec, testValues.maxVec, testType);
        verifyTestVector3dValue("rsqrtVec", expectedTestValues.rsqrtVec, testValues.rsqrtVec, testType);
        verifyTestVector3dValue("bitReverseVec", expectedTestValues.bitReverseVec, testValues.bitReverseVec, testType);
        verifyTestVector3dValue("fracVec", expectedTestValues.fracVec, testValues.fracVec, testType);
        verifyTestVector3dValue("mixVec", expectedTestValues.mixVec, testValues.mixVec, testType);

        verifyTestVector3dValue("signVec", expectedTestValues.signVec, testValues.signVec, testType);
        verifyTestVector3dValue("radiansVec", expectedTestValues.radiansVec, testValues.radiansVec, testType);
        verifyTestVector3dValue("degreesVec", expectedTestValues.degreesVec, testValues.degreesVec, testType);
        verifyTestVector3dValue("stepVec", expectedTestValues.stepVec, testValues.stepVec, testType);
        verifyTestVector3dValue("smoothStepVec", expectedTestValues.smoothStepVec, testValues.smoothStepVec, testType);
        verifyTestVector3dValue("faceForward", expectedTestValues.faceForward, testValues.faceForward, testType);
        verifyTestVector3dValue("reflect", expectedTestValues.reflect, testValues.reflect, testType);
        verifyTestVector3dValue("refract", expectedTestValues.refract, testValues.refract, testType);
        verifyTestVector3dValue("addCarryVecResult", expectedTestValues.addCarryVec.result, testValues.addCarryVec.result, testType);
        verifyTestVector3dValue("addCarryVecCarry", expectedTestValues.addCarryVec.carry, testValues.addCarryVec.carry, testType);
        // Disabled: current glm implementation is wrong
        //verifyTestVector3dValue("subBorrowVecResult", expectedTestValues.subBorrowVec.result, testValues.subBorrowVec.result, testType);
        //verifyTestVector3dValue("subBorrowVecBorrow", expectedTestValues.subBorrowVec.borrow, testValues.subBorrowVec.borrow, testType);

        verifyTestMatrix3x3Value("mul", expectedTestValues.mul, testValues.mul, testType);
        verifyTestMatrix3x3Value("transpose", expectedTestValues.transpose, testValues.transpose, testType);
        verifyTestMatrix3x3Value("inverse", expectedTestValues.inverse, testValues.inverse, testType);
    }
};

#endif