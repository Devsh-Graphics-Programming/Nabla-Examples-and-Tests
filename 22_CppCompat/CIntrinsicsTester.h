#ifndef _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_C_INTRINSICS_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_C_INTRINSICS_TESTER_INCLUDED_


#include "nbl/examples/examples.hpp"

#include "app_resources/common.hlsl"


using namespace nbl;

class CIntrinsicsTester final : public ITester<IntrinsicsIntputTestValues, IntrinsicsTestValues, IntrinsicsTestExecutor>
{
    using base_t = ITester<IntrinsicsIntputTestValues, IntrinsicsTestValues, IntrinsicsTestExecutor>;

public:
    CIntrinsicsTester(const uint32_t testBatchCount)
        : base_t(testBatchCount) {};

private:
    IntrinsicsIntputTestValues generateInputTestValues() override
    {
        std::uniform_real_distribution<float> realDistributionNeg(-50.0f, -1.0f);
        std::uniform_real_distribution<float> realDistributionPos(1.0f, 50.0f);
        std::uniform_real_distribution<float> realDistributionZeroToOne(0.0f, 1.0f);
        std::uniform_real_distribution<float> realDistribution(-100.0f, 100.0f);
        std::uniform_real_distribution<float> realDistributionSmall(1.0f, 4.0f);
        std::uniform_int_distribution<int> intDistribution(-100, 100);
        std::uniform_int_distribution<uint32_t> uintDistribution(0, 100);

        IntrinsicsIntputTestValues testInput;
        testInput.bitCount = intDistribution(getRandomEngine());
        testInput.crossLhs = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.crossRhs = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.clampVal = realDistribution(getRandomEngine());
        testInput.clampMin = realDistributionNeg(getRandomEngine());
        testInput.clampMax = realDistributionPos(getRandomEngine());
        testInput.length = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.normalize = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.dotLhs = float32_t3(realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()));
        testInput.dotRhs = float32_t3(realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()));
        testInput.determinant = float32_t3x3(
            realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()),
            realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()),
            realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine())
        );
        testInput.findMSB = realDistribution(getRandomEngine());
        testInput.findLSB = realDistribution(getRandomEngine());
        testInput.inverse = float32_t3x3(
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()),
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()),
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())
        );
        testInput.transpose = float32_t3x3(
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()),
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()),
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())
        );
        testInput.mulLhs = float32_t3x3(
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()),
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()),
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())
        );
        testInput.mulRhs = float32_t3x3(
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()),
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()),
            realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())
        );
        testInput.minA = realDistribution(getRandomEngine());
        testInput.minB = realDistribution(getRandomEngine());
        testInput.maxA = realDistribution(getRandomEngine());
        testInput.maxB = realDistribution(getRandomEngine());
        testInput.rsqrt = realDistributionPos(getRandomEngine());
        testInput.bitReverse = realDistribution(getRandomEngine());
        testInput.frac = realDistribution(getRandomEngine());
        testInput.mixX = realDistributionNeg(getRandomEngine());
        testInput.mixY = realDistributionPos(getRandomEngine());
        testInput.mixA = realDistributionZeroToOne(getRandomEngine());
        testInput.sign = realDistribution(getRandomEngine());
        testInput.radians = realDistribution(getRandomEngine());
        testInput.degrees = realDistribution(getRandomEngine());
        testInput.stepEdge = realDistribution(getRandomEngine());
        testInput.stepX = realDistribution(getRandomEngine());
        testInput.smoothStepEdge0 = realDistributionNeg(getRandomEngine());
        testInput.smoothStepEdge1 = realDistributionPos(getRandomEngine());
        testInput.smoothStepX = realDistribution(getRandomEngine());

        testInput.bitCountVec = int32_t3(intDistribution(getRandomEngine()), intDistribution(getRandomEngine()), intDistribution(getRandomEngine()));
        testInput.clampValVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.clampMinVec = float32_t3(realDistributionNeg(getRandomEngine()), realDistributionNeg(getRandomEngine()), realDistributionNeg(getRandomEngine()));
        testInput.clampMaxVec = float32_t3(realDistributionPos(getRandomEngine()), realDistributionPos(getRandomEngine()), realDistributionPos(getRandomEngine()));
        testInput.findMSBVec = uint32_t3(uintDistribution(getRandomEngine()), uintDistribution(getRandomEngine()), uintDistribution(getRandomEngine()));
        testInput.findLSBVec = uint32_t3(uintDistribution(getRandomEngine()), uintDistribution(getRandomEngine()), uintDistribution(getRandomEngine()));
        testInput.minAVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.minBVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.maxAVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.maxBVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.rsqrtVec = float32_t3(realDistributionPos(getRandomEngine()), realDistributionPos(getRandomEngine()), realDistributionPos(getRandomEngine()));
        testInput.bitReverseVec = uint32_t3(uintDistribution(getRandomEngine()), uintDistribution(getRandomEngine()), uintDistribution(getRandomEngine()));
        testInput.fracVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.mixXVec = float32_t3(realDistributionNeg(getRandomEngine()), realDistributionNeg(getRandomEngine()), realDistributionNeg(getRandomEngine()));
        testInput.mixYVec = float32_t3(realDistributionPos(getRandomEngine()), realDistributionPos(getRandomEngine()), realDistributionPos(getRandomEngine()));
        testInput.mixAVec = float32_t3(realDistributionZeroToOne(getRandomEngine()), realDistributionZeroToOne(getRandomEngine()), realDistributionZeroToOne(getRandomEngine()));

        testInput.signVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.radiansVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.degreesVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.stepEdgeVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.stepXVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.smoothStepEdge0Vec = float32_t3(realDistributionNeg(getRandomEngine()), realDistributionNeg(getRandomEngine()), realDistributionNeg(getRandomEngine()));
        testInput.smoothStepEdge1Vec = float32_t3(realDistributionPos(getRandomEngine()), realDistributionPos(getRandomEngine()), realDistributionPos(getRandomEngine()));
        testInput.smoothStepXVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.faceForwardN = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.faceForwardI = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.faceForwardNref = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.reflectI = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.reflectN = glm::normalize(float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())));
        testInput.refractI = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.refractN = glm::normalize(float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine())));
        testInput.refractEta = realDistribution(getRandomEngine());

        return testInput;
    }

    IntrinsicsTestValues determineExpectedResults(const IntrinsicsIntputTestValues& testInput) override
    {
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

        return expected;
    }

    void verifyTestResults(const IntrinsicsTestValues& expectedTestValues, const IntrinsicsTestValues& testValues, const size_t testIteration, const uint32_t seed, TestType testType) override
    {
        verifyTestValue("bitCount", expectedTestValues.bitCount, testValues.bitCount, testIteration, seed, testType);
        verifyTestValue("clamp", expectedTestValues.clamp, testValues.clamp, testIteration, seed, testType);
        verifyTestValue("length", expectedTestValues.length, testValues.length, testIteration, seed, testType, 0.0001);
        verifyTestValue("dot", expectedTestValues.dot, testValues.dot, testIteration, seed, testType, 0.00001);
        verifyTestValue("determinant", expectedTestValues.determinant, testValues.determinant, testIteration, seed, testType);
        verifyTestValue("findMSB", expectedTestValues.findMSB, testValues.findMSB, testIteration, seed, testType);
        verifyTestValue("findLSB", expectedTestValues.findLSB, testValues.findLSB, testIteration, seed, testType);
        verifyTestValue("min", expectedTestValues.min, testValues.min, testIteration, seed, testType);
        verifyTestValue("max", expectedTestValues.max, testValues.max, testIteration, seed, testType);
        verifyTestValue("rsqrt", expectedTestValues.rsqrt, testValues.rsqrt, testIteration, seed, testType);
        verifyTestValue("frac", expectedTestValues.frac, testValues.frac, testIteration, seed, testType);
        verifyTestValue("bitReverse", expectedTestValues.bitReverse, testValues.bitReverse, testIteration, seed, testType);
        verifyTestValue("mix", expectedTestValues.mix, testValues.mix, testIteration, seed, testType);
        verifyTestValue("sign", expectedTestValues.sign, testValues.sign, testIteration, seed, testType);
        verifyTestValue("radians", expectedTestValues.radians, testValues.radians, testIteration, seed, testType, 0.00001);
        verifyTestValue("degrees", expectedTestValues.degrees, testValues.degrees, testIteration, seed, testType, 0.001);
        verifyTestValue("step", expectedTestValues.step, testValues.step, testIteration, seed, testType);
        verifyTestValue("smoothStep", expectedTestValues.smoothStep, testValues.smoothStep, testIteration, seed, testType);
        verifyTestValue("addCarryResult", expectedTestValues.addCarry.result, testValues.addCarry.result, testIteration, seed, testType);
        verifyTestValue("addCarryCarry", expectedTestValues.addCarry.carry, testValues.addCarry.carry, testIteration, seed, testType);
        verifyTestValue("subBorrowResult", expectedTestValues.subBorrow.result, testValues.subBorrow.result, testIteration, seed, testType);
        verifyTestValue("subBorrowBorrow", expectedTestValues.subBorrow.borrow, testValues.subBorrow.borrow, testIteration, seed, testType);

        verifyTestValue("normalize", expectedTestValues.normalize, testValues.normalize, testIteration, seed, testType, 0.000001);
        verifyTestValue("cross", expectedTestValues.cross, testValues.cross, testIteration, seed, testType);
        verifyTestValue("bitCountVec", expectedTestValues.bitCountVec, testValues.bitCountVec, testIteration, seed, testType);
        verifyTestValue("clampVec", expectedTestValues.clampVec, testValues.clampVec, testIteration, seed, testType);
        verifyTestValue("findMSBVec", expectedTestValues.findMSBVec, testValues.findMSBVec, testIteration, seed, testType);
        verifyTestValue("findLSBVec", expectedTestValues.findLSBVec, testValues.findLSBVec, testIteration, seed, testType);
        verifyTestValue("minVec", expectedTestValues.minVec, testValues.minVec, testIteration, seed, testType);
        verifyTestValue("maxVec", expectedTestValues.maxVec, testValues.maxVec, testIteration, seed, testType);
        verifyTestValue("rsqrtVec", expectedTestValues.rsqrtVec, testValues.rsqrtVec, testIteration, seed, testType);
        verifyTestValue("bitReverseVec", expectedTestValues.bitReverseVec, testValues.bitReverseVec, testIteration, seed, testType);
        verifyTestValue("fracVec", expectedTestValues.fracVec, testValues.fracVec, testIteration, seed, testType);
        verifyTestValue("mixVec", expectedTestValues.mixVec, testValues.mixVec, testIteration, seed, testType);

        verifyTestValue("signVec", expectedTestValues.signVec, testValues.signVec, testIteration, seed, testType);
        verifyTestValue("radiansVec", expectedTestValues.radiansVec, testValues.radiansVec, testIteration, seed, testType, 0.00001);
        verifyTestValue("degreesVec", expectedTestValues.degreesVec, testValues.degreesVec, testIteration, seed, testType, 0.001);
        verifyTestValue("stepVec", expectedTestValues.stepVec, testValues.stepVec, testIteration, seed, testType);
        verifyTestValue("smoothStepVec", expectedTestValues.smoothStepVec, testValues.smoothStepVec, testIteration, seed, testType);
        verifyTestValue("faceForward", expectedTestValues.faceForward, testValues.faceForward, testIteration, seed, testType);
        verifyTestValue("reflect", expectedTestValues.reflect, testValues.reflect, testIteration, seed, testType, 0.0001);
        verifyTestValue("refract", expectedTestValues.refract, testValues.refract, testIteration, seed, testType, 0.01);
        verifyTestValue("addCarryVecResult", expectedTestValues.addCarryVec.result, testValues.addCarryVec.result, testIteration, seed, testType);
        verifyTestValue("addCarryVecCarry", expectedTestValues.addCarryVec.carry, testValues.addCarryVec.carry, testIteration, seed, testType);
        verifyTestValue("subBorrowVecResult", expectedTestValues.subBorrowVec.result, testValues.subBorrowVec.result, testIteration, seed, testType);
        verifyTestValue("subBorrowVecBorrow", expectedTestValues.subBorrowVec.borrow, testValues.subBorrowVec.borrow, testIteration, seed, testType);

        verifyTestValue("mul", expectedTestValues.mul, testValues.mul, testIteration, seed, testType);
        verifyTestValue("transpose", expectedTestValues.transpose, testValues.transpose, testIteration, seed, testType);
        verifyTestValue("inverse", expectedTestValues.inverse, testValues.inverse, testIteration, seed, testType);
    }
};

#endif