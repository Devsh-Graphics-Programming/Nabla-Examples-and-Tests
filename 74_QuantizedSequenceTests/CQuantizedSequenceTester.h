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
        testInput.uintVec3 = uint32_t3(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));

        return testInput;
    }

    QuantizedSequenceTestValues determineExpectedResults(const QuantizedSequenceInputTestValues& testInput) override
    {
        QuantizedSequenceTestValues expected;

        {
            for (uint32_t i = 0; i < 3; i++)
                expected.uintVec3[i] = testInput.uintVec3[i] >> 11u;
        }

        return expected;
    }

    void verifyTestResults(const QuantizedSequenceTestValues& expectedTestValues, const QuantizedSequenceTestValues& testValues, const size_t testIteration, const uint32_t seed, TestType testType) override
    {
        verifyTestValue("get uint3", expectedTestValues.uintVec3, testValues.uintVec3, testIteration, seed, testType);
    }

};

#endif
