#ifndef _NBL_EXAMPLES_TESTS_74_QUANTIZED_SEQUENCE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_74_QUANTIZED_SEQUENCE_TESTER_INCLUDED_

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

class CQuantizedSequenceTester final : public ITester<QuantizedSequenceInputTestValues, QuantizedSequenceTestValues, QuantizedSequenceTestExecutor>
{
    using base_t = ITester<QuantizedSequenceInputTestValues, QuantizedSequenceTestValues, QuantizedSequenceTestExecutor>;

public:
    CQuantizedSequenceTester(const uint32_t testBatchCount)
        : base_t(testBatchCount) {};

private:
    QuantizedSequenceInputTestValues generateInputTestValues() override
    {
        std::uniform_real_distribution<float> realDistribution(0.0f, 1.0f);
        std::uniform_int_distribution<uint32_t> uint32Distribution(0, std::numeric_limits<uint32_t>::max());

        QuantizedSequenceInputTestValues testInput;
        testInput.scalar = uint32Distribution(getRandomEngine());
        testInput.uvec2 = uint32_t2(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));
        testInput.uvec3 = uint32_t3(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));
        testInput.uvec4 = uint32_t4(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));

        testInput.unorm3 = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));

        testInput.scrambleKey3 = uint32_t3(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));

        return testInput;
    }

    QuantizedSequenceTestValues determineExpectedResults(const QuantizedSequenceInputTestValues& testInput) override
    {
        QuantizedSequenceTestValues expected;
        expected.uintDim1 = testInput.scalar;
        {
            for (uint32_t i = 0; i < 2; i++)
                expected.uintDim2[i] = testInput.uvec2[i] >> 16u;
        }
        {
            for (uint32_t i = 0; i < 3; i++)
                expected.uintDim3[i] = testInput.uvec3[i] >> 22u;
        }
        {
            for (uint32_t i = 0; i < 4; i++)
                expected.uintDim4[i] = testInput.uvec4[i] >> 24u;
        }

        expected.uintVec2_Dim2 = testInput.uvec2;
        {
            for (uint32_t i = 0; i < 3; i++)
                expected.uintVec2_Dim3[i] = testInput.uvec3[i] >> 11u;
        }

        expected.uintVec3_Dim3 = testInput.uvec3;
        expected.uintVec4_Dim4 = testInput.uvec4;

        {
            const uint32_t fullWidthMultiplier = (1u << 31u) - 1u;
            uint32_t3 stored;
            for (uint32_t i = 0; i < 3; i++)
                stored[i] = uint32_t(testInput.unorm3[i] * fullWidthMultiplier) >> 11u;
            expected.unorm3_predecode = float32_t3(stored ^ testInput.scrambleKey3) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t multiplier = (1u << 21u) - 1u;
            uint32_t3 stored, scrambleKey;
            for (uint32_t i = 0; i < 3; i++)
            {
                stored[i] = uint32_t(testInput.unorm3[i] * multiplier) >> 11u;
                scrambleKey[i] = testInput.scrambleKey3[i] >> 11u;
            }
            expected.unorm3_postdecode = float32_t3(stored ^ scrambleKey) * bit_cast<float>(0x35000004u);
        }

        return expected;
    }

    void verifyTestResults(const QuantizedSequenceTestValues& expectedTestValues, const QuantizedSequenceTestValues& testValues, const size_t testIteration, const uint32_t seed, TestType testType) override
    {
        verifyTestValue("get uint from dim 1", expectedTestValues.uintDim1, testValues.uintDim1, testIteration, seed, testType);
        verifyTestValue("get uint2 from dim 1", expectedTestValues.uintDim2, testValues.uintDim2, testIteration, seed, testType);
        verifyTestValue("get uint3 from dim 1", expectedTestValues.uintDim3, testValues.uintDim3, testIteration, seed, testType);
        verifyTestValue("get uint4 from dim 1", expectedTestValues.uintDim4, testValues.uintDim4, testIteration, seed, testType);

        verifyTestValue("get uint2 from dim 2", expectedTestValues.uintVec2_Dim2, testValues.uintVec2_Dim2, testIteration, seed, testType);
        verifyTestValue("get uint3 from dim 2", expectedTestValues.uintVec2_Dim3, testValues.uintVec2_Dim3, testIteration, seed, testType);

        verifyTestValue("get uint3 from dim 3", expectedTestValues.uintVec3_Dim3, testValues.uintVec3_Dim3, testIteration, seed, testType);
        verifyTestValue("get uint4 from dim 4", expectedTestValues.uintVec4_Dim4, testValues.uintVec4_Dim4, testIteration, seed, testType);

        verifyTestValue("encode/decode unorm3 from uint2 (fullwidth)", expectedTestValues.unorm3_predecode, testValues.unorm3_predecode, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm3 from uint2", expectedTestValues.unorm3_postdecode, testValues.unorm3_postdecode, testIteration, seed, testType);
    }

};

#endif
