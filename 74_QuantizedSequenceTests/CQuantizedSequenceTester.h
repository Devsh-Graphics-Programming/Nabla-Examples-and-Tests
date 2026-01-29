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
        std::uniform_int_distribution<uint16_t> uint16Distribution(0, std::numeric_limits<uint16_t>::max());

        QuantizedSequenceInputTestValues testInput;
        testInput.scalar = uint16Distribution(getRandomEngine());
        testInput.u16vec2 = uint32_t2(uint16Distribution(getRandomEngine()), uint16Distribution(getRandomEngine()));
        testInput.u16vec3 = uint32_t3(uint16Distribution(getRandomEngine()), uint16Distribution(getRandomEngine()), uint16Distribution(getRandomEngine()));
        testInput.u16vec4 = uint32_t4(uint16Distribution(getRandomEngine()), uint16Distribution(getRandomEngine()), uint16Distribution(getRandomEngine()), uint16Distribution(getRandomEngine()));

        testInput.scalar16 = uint32Distribution(getRandomEngine());
        testInput.uvec2 = uint32_t2(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));
        testInput.uvec3 = uint32_t3(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));
        testInput.uvec4 = uint32_t4(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));

        testInput.unorm1 = float32_t1(realDistribution(getRandomEngine()));
        testInput.unorm2 = float32_t2(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.unorm3 = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.unorm4 = float32_t4(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));

        testInput.scrambleKey1 = uint32_t1(uint32Distribution(getRandomEngine()));
        testInput.scrambleKey2 = uint32_t2(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));
        testInput.scrambleKey3 = uint32_t3(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));
        testInput.scrambleKey4 = uint32_t4(uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()), uint32Distribution(getRandomEngine()));

        return testInput;
    }

    QuantizedSequenceTestValues determineExpectedResults(const QuantizedSequenceInputTestValues& testInput) override
    {
        QuantizedSequenceTestValues expected;
        // test create/set/get
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

        expected.u16Dim1 = testInput.scalar16;
        {
            for (uint32_t i = 0; i < 2; i++)
                expected.u16Dim2[i] = testInput.u16vec2[i] >> 8u;
        }
        {
            for (uint32_t i = 0; i < 3; i++)
                expected.u16Dim3[i] = testInput.u16vec3[i] >> 11u;
        }
        {
            for (uint32_t i = 0; i < 4; i++)
                expected.u16Dim4[i] = testInput.u16vec4[i] >> 12u;
        }

        expected.u16Vec2_Dim2 = testInput.u16vec2;
        {
            for (uint32_t i = 0; i < 4; i++)
                expected.u16Vec2_Dim4[i] = testInput.u16vec4[i] >> 8u;
        }

        expected.u16Vec3_Dim3 = testInput.u16vec3;
        expected.u16Vec4_Dim4 = testInput.u16vec4;

        // test encode/decode uint32, dim 1..4
        {
            const uint32_t fullWidthMultiplier = (1u << 31u) - 1u;
            uint32_t1 stored;
            stored[0] = uint32_t(testInput.unorm1[0] * fullWidthMultiplier);
            expected.unorm1_pre_u32 = float32_t1(stored ^ testInput.scrambleKey1) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t multiplier = (1u << 31u) - 1u;
            uint32_t1 stored;
            stored[0] = uint32_t(testInput.unorm1[0] * multiplier);
            expected.unorm1_post_u32 = float32_t1(stored ^ testInput.scrambleKey1) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t bitsPerComponent = 16u;
            const uint32_t discardBits = 32u - bitsPerComponent;
            const uint32_t fullWidthMultiplier = (1u << 31u) - 1u;
            uint32_t2 stored;
            for (uint32_t i = 0; i < 2; i++)
                stored[i] = uint32_t(testInput.unorm2[i] * fullWidthMultiplier) >> discardBits;
            expected.unorm2_pre_u32 = float32_t2(stored ^ testInput.scrambleKey2) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t bitsPerComponent = 16u;
            const uint32_t discardBits = 32u - bitsPerComponent;
            const uint32_t multiplier = (1u << bitsPerComponent) - 1u;
            uint32_t2 stored, scrambleKey;
            for (uint32_t i = 0; i < 2; i++)
            {
                stored[i] = uint32_t(testInput.unorm2[i] * multiplier) >> discardBits;
                scrambleKey[i] = testInput.scrambleKey2[i] >> discardBits;
            }
            expected.unorm2_post_u32 = float32_t2(stored ^ scrambleKey) * bit_cast<float>(0x37800080u);
        }
        {
            const uint32_t bitsPerComponent = 10u;
            const uint32_t discardBits = 32u - bitsPerComponent;
            const uint32_t fullWidthMultiplier = (1u << 31u) - 1u;
            uint32_t3 stored;
            for (uint32_t i = 0; i < 3; i++)
                stored[i] = uint32_t(testInput.unorm3[i] * fullWidthMultiplier) >> discardBits;
            expected.unorm3_pre_u32 = float32_t3(stored ^ testInput.scrambleKey3) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t bitsPerComponent = 10u;
            const uint32_t discardBits = 32u - bitsPerComponent;
            const uint32_t multiplier = (1u << bitsPerComponent) - 1u;
            uint32_t3 stored, scrambleKey;
            for (uint32_t i = 0; i < 3; i++)
            {
                stored[i] = uint32_t(testInput.unorm3[i] * multiplier) >> discardBits;
                scrambleKey[i] = testInput.scrambleKey3[i] >> discardBits;
            }
            expected.unorm3_post_u32 = float32_t3(stored ^ scrambleKey) * bit_cast<float>(0x3a802008u);
        }
        {
            const uint32_t bitsPerComponent = 8u;
            const uint32_t discardBits = 32u - bitsPerComponent;
            const uint32_t fullWidthMultiplier = (1u << 31u) - 1u;
            uint32_t4 stored;
            for (uint32_t i = 0; i < 4; i++)
                stored[i] = uint32_t(testInput.unorm4[i] * fullWidthMultiplier) >> discardBits;
            expected.unorm4_pre_u32 = float32_t4(stored ^ testInput.scrambleKey4) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t bitsPerComponent = 8u;
            const uint32_t discardBits = 32u - bitsPerComponent;
            const uint32_t multiplier = (1u << bitsPerComponent) - 1u;
            uint32_t4 stored, scrambleKey;
            for (uint32_t i = 0; i < 4; i++)
            {
                stored[i] = uint32_t(testInput.unorm4[i] * multiplier) >> discardBits;
                scrambleKey[i] = testInput.scrambleKey4[i] >> discardBits;
            }
            expected.unorm4_post_u32 = float32_t4(stored ^ scrambleKey) * bit_cast<float>(0x3b808081u);
        }

        // test encode/decode uint32_tN storage, dim == N
        {
            const uint32_t fullWidthMultiplier = (1u << 31u) - 1u;
            uint32_t2 stored;
            for (uint32_t i = 0; i < 2; i++)
                stored[i] = uint32_t(testInput.unorm2[i] * fullWidthMultiplier);
            expected.unorm2_pre_u32t2 = float32_t2(stored ^ testInput.scrambleKey2) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t multiplier = (1u << 31u) - 1u;
            uint32_t2 stored;
            for (uint32_t i = 0; i < 2; i++)
                stored[i] = uint32_t(testInput.unorm2[i] * multiplier);
            expected.unorm2_post_u32t2 = float32_t2(stored ^ testInput.scrambleKey2) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t fullWidthMultiplier = (1u << 31u) - 1u;
            uint32_t3 stored;
            for (uint32_t i = 0; i < 3; i++)
                stored[i] = uint32_t(testInput.unorm3[i] * fullWidthMultiplier);
            expected.unorm3_pre_u32t3 = float32_t3(stored ^ testInput.scrambleKey3) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t multiplier = (1u << 31u) - 1u;
            uint32_t3 stored;
            for (uint32_t i = 0; i < 3; i++)
                stored[i] = uint32_t(testInput.unorm3[i] * multiplier);
            expected.unorm3_post_u32t3 = float32_t3(stored ^ testInput.scrambleKey3) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t fullWidthMultiplier = (1u << 31u) - 1u;
            uint32_t4 stored;
            for (uint32_t i = 0; i < 4; i++)
                stored[i] = uint32_t(testInput.unorm4[i] * fullWidthMultiplier);
            expected.unorm4_pre_u32t4 = float32_t4(stored ^ testInput.scrambleKey4) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t multiplier = (1u << 31u) - 1u;
            uint32_t4 stored;
            for (uint32_t i = 0; i < 4; i++)
                stored[i] = uint32_t(testInput.unorm4[i] * multiplier);
            expected.unorm4_post_u32t4 = float32_t4(stored ^ testInput.scrambleKey4) * bit_cast<float>(0x2f800004u);
        }

        // test encode/decode uint32_t2 storage, dim 3
        {
            const uint32_t bitsPerComponent = 21u;
            const uint32_t discardBits = 32u - bitsPerComponent;
            const uint32_t fullWidthMultiplier = (1u << 31u) - 1u;
            uint32_t3 stored;
            for (uint32_t i = 0; i < 3; i++)
                stored[i] = uint32_t(testInput.unorm3[i] * fullWidthMultiplier) >> discardBits;
            expected.unorm3_pre_u32t2 = float32_t3(stored ^ testInput.scrambleKey3) * bit_cast<float>(0x2f800004u);
        }
        {
            const uint32_t bitsPerComponent = 21u;
            const uint32_t discardBits = 32u - bitsPerComponent;
            const uint32_t multiplier = (1u << bitsPerComponent) - 1u;
            uint32_t3 stored, scrambleKey;
            for (uint32_t i = 0; i < 3; i++)
            {
                stored[i] = uint32_t(testInput.unorm3[i] * multiplier) >> discardBits;
                scrambleKey[i] = testInput.scrambleKey3[i] >> discardBits;
            }
            expected.unorm3_post_u32t2 = float32_t3(stored ^ scrambleKey) * bit_cast<float>(0x35000004u);
        }

        return expected;
    }

    void verifyTestResults(const QuantizedSequenceTestValues& expectedTestValues, const QuantizedSequenceTestValues& testValues, const size_t testIteration, const uint32_t seed, TestType testType) override
    {
        verifyTestValue("get uint from u32", expectedTestValues.uintDim1, testValues.uintDim1, testIteration, seed, testType);
        verifyTestValue("get uint2 from u32", expectedTestValues.uintDim2, testValues.uintDim2, testIteration, seed, testType);
        verifyTestValue("get uint3 from u32", expectedTestValues.uintDim3, testValues.uintDim3, testIteration, seed, testType);
        verifyTestValue("get uint4 from u32", expectedTestValues.uintDim4, testValues.uintDim4, testIteration, seed, testType);

        verifyTestValue("get uint2 from u32 vec2", expectedTestValues.uintVec2_Dim2, testValues.uintVec2_Dim2, testIteration, seed, testType);
        verifyTestValue("get uint3 from u32 vec2", expectedTestValues.uintVec2_Dim3, testValues.uintVec2_Dim3, testIteration, seed, testType);

        verifyTestValue("get uint3 from u32 vec3", expectedTestValues.uintVec3_Dim3, testValues.uintVec3_Dim3, testIteration, seed, testType);
        verifyTestValue("get uint4 from u32 vec4", expectedTestValues.uintVec4_Dim4, testValues.uintVec4_Dim4, testIteration, seed, testType);

        verifyTestValue("get uint from u16", expectedTestValues.u16Dim1, testValues.u16Dim1, testIteration, seed, testType);
        verifyTestValue("get uint2 from u16", expectedTestValues.u16Dim2, testValues.u16Dim2, testIteration, seed, testType);
        verifyTestValue("get uint3 from u16", expectedTestValues.u16Dim3, testValues.u16Dim3, testIteration, seed, testType);
        verifyTestValue("get uint4 from u16", expectedTestValues.u16Dim3, testValues.u16Dim3, testIteration, seed, testType);

        verifyTestValue("get uint2 from u16 vec2", expectedTestValues.u16Vec2_Dim2, testValues.u16Vec2_Dim2, testIteration, seed, testType);
        verifyTestValue("get uint4 from u16 vec2", expectedTestValues.u16Vec2_Dim4, testValues.u16Vec2_Dim4, testIteration, seed, testType);

        verifyTestValue("get uint3 from u16 vec3", expectedTestValues.u16Vec3_Dim3, testValues.u16Vec3_Dim3, testIteration, seed, testType);
        verifyTestValue("get uint4 from u16 vec4", expectedTestValues.u16Vec4_Dim4, testValues.u16Vec4_Dim4, testIteration, seed, testType);

        verifyTestValue("encode/decode unorm from u32 (fullwidth)", expectedTestValues.unorm1_pre_u32, testValues.unorm1_pre_u32, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm from u32", expectedTestValues.unorm1_post_u32, testValues.unorm1_post_u32, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm2 from u32 (fullwidth)", expectedTestValues.unorm2_pre_u32, testValues.unorm2_pre_u32, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm2 from u32", expectedTestValues.unorm2_post_u32, testValues.unorm2_post_u32, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm3 from u32 (fullwidth)", expectedTestValues.unorm3_pre_u32, testValues.unorm3_pre_u32, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm3 from u32", expectedTestValues.unorm3_post_u32, testValues.unorm3_post_u32, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm4 from u32 (fullwidth)", expectedTestValues.unorm4_pre_u32, testValues.unorm4_pre_u32, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm4 from u32", expectedTestValues.unorm4_post_u32, testValues.unorm4_post_u32, testIteration, seed, testType);

        verifyTestValue("encode/decode unorm2 from u32 vec2 (fullwidth)", expectedTestValues.unorm2_pre_u32t2, testValues.unorm2_pre_u32t2, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm2 from u32 vec2", expectedTestValues.unorm2_post_u32t2, testValues.unorm2_post_u32t2, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm3 from u32 vec3 (fullwidth)", expectedTestValues.unorm3_pre_u32t3, testValues.unorm3_pre_u32t3, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm3 from u32 vec3", expectedTestValues.unorm3_post_u32t3, testValues.unorm3_post_u32t3, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm4 from u32 vec4 (fullwidth)", expectedTestValues.unorm4_pre_u32t4, testValues.unorm4_pre_u32t4, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm4 from u32 vec4", expectedTestValues.unorm4_post_u32t4, testValues.unorm4_post_u32t4, testIteration, seed, testType);

        verifyTestValue("encode/decode unorm3 from u32 vec2 (fullwidth)", expectedTestValues.unorm3_pre_u32t2, testValues.unorm3_pre_u32t2, testIteration, seed, testType);
        verifyTestValue("encode/decode unorm3 from u32 vec2", expectedTestValues.unorm3_post_u32t2, testValues.unorm3_post_u32t2, testIteration, seed, testType);
    }

};

#endif
