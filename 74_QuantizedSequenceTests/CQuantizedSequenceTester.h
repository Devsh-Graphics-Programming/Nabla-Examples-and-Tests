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
        std::uniform_real_distribution<float> realDistribution(-1.0f, 1.0f);
        std::uniform_real_distribution<float> realDistribution01(0.0f, 1.0f);
        std::uniform_int_distribution<uint32_t> uint32Distribution(0, std::numeric_limits<uint32_t>::max());

        QuantizedSequenceInputTestValues testInput;
        testInput.scalar = uint32Distribution(getRandomEngine());
        testInput.uvec2 = uint32_t2(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));
        testInput.uvec3 = uint32_t3(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));
        testInput.uvec4 = uint32_t4(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));

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

        return expected;
    }

    void verifyTestResults(const QuantizedSequenceTestValues& expectedTestValues, const QuantizedSequenceTestValues& testValues, const size_t testIteration, const uint32_t seed, TestType testType) override
    {
        verifyTestValue("get uint3", expectedTestValues.uintVec2_Dim3, testValues.uintVec2_Dim3, testIteration, seed, testType);
    }

};

#endif
