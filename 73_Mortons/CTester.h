#ifndef _NBL_EXAMPLES_TESTS_12_MORTON_C_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_12_MORTON_C_TESTER_INCLUDED_

#include <nabla.h>
#include "app_resources/testCommon.hlsl"
#include "../common/include/nbl/examples/Tester/ITester.h"

using namespace nbl;

class CTester final : public ITester<InputTestValues, TestValues, TestExecutor>
{
    using base_t = ITester<InputTestValues, TestValues, TestExecutor>;

public:
    CTester(const uint32_t testIterationCount)
        : base_t(testIterationCount) {};

private:
    InputTestValues generateInputTestValues() override
    {
        std::random_device rd;
        std::mt19937 mt(rd());

        std::uniform_int_distribution<uint16_t> shortDistribution(uint16_t(0), std::numeric_limits<uint16_t>::max());
        std::uniform_int_distribution<uint32_t> intDistribution(uint32_t(0), std::numeric_limits<uint32_t>::max());
        std::uniform_int_distribution<uint64_t> longDistribution(uint64_t(0), std::numeric_limits<uint64_t>::max());

        // Set input thest values that will be used in both CPU and GPU tests
        InputTestValues testInput;

        testInput.generatedA = longDistribution(mt);
        testInput.generatedB = longDistribution(mt);

        uint32_t generatedShift = intDistribution(mt) & uint32_t(63);
        testInput.shift = generatedShift;

        testInput.coordX = longDistribution(mt);
        testInput.coordY = longDistribution(mt);
        testInput.coordZ = longDistribution(mt);
        testInput.coordW = longDistribution(mt);

        return testInput;
    }

    TestValues determineExpectedResults(const InputTestValues& testInput) override
    {
        // use std library or glm functions to determine expected test values, the output of functions from intrinsics.hlsl will be verified against these values
        TestValues expected;

        {
            const uint64_t generatedA = testInput.generatedA;
            const uint64_t generatedB = testInput.generatedB;
            const uint32_t generatedShift = testInput.shift;

            expected.emulatedAnd = _static_cast<emulated_uint64_t>(generatedA & generatedB);
            expected.emulatedOr = _static_cast<emulated_uint64_t>(generatedA | generatedB);
            expected.emulatedXor = _static_cast<emulated_uint64_t>(generatedA ^ generatedB);
            expected.emulatedNot = _static_cast<emulated_uint64_t>(~generatedA);
            expected.emulatedPlus = _static_cast<emulated_uint64_t>(generatedA + generatedB);
            expected.emulatedMinus = _static_cast<emulated_uint64_t>(generatedA - generatedB);
            expected.emulatedLess = uint32_t(generatedA < generatedB);
            expected.emulatedLessEqual = uint32_t(generatedA <= generatedB);
            expected.emulatedGreater = uint32_t(generatedA > generatedB);
            expected.emulatedGreaterEqual = uint32_t(generatedA >= generatedB);

            expected.emulatedLeftShifted = _static_cast<emulated_uint64_t>(generatedA << generatedShift);
            expected.emulatedUnsignedRightShifted = _static_cast<emulated_uint64_t>(generatedA >> generatedShift);
            expected.emulatedSignedRightShifted = _static_cast<emulated_int64_t>(static_cast<int64_t>(generatedA) >> generatedShift);
        }
        {
            uint64_t2 Vec2A = { testInput.coordX, testInput.coordY };
            uint64_t2 Vec2B = { testInput.coordZ, testInput.coordW };

            uint16_t2 Vec2ASmall = uint16_t2(Vec2A & smallBitsMask_2);
            uint16_t2 Vec2BSmall = uint16_t2(Vec2B & smallBitsMask_2);
            uint16_t2 Vec2AMedium = uint16_t2(Vec2A & mediumBitsMask_2);
            uint16_t2 Vec2BMedium = uint16_t2(Vec2B & mediumBitsMask_2);
            uint32_t2 Vec2AFull = uint32_t2(Vec2A & fullBitsMask_2);
            uint32_t2 Vec2BFull = uint32_t2(Vec2B & fullBitsMask_2);

            uint64_t3 Vec3A = { testInput.coordX, testInput.coordY, testInput.coordZ };
            uint64_t3 Vec3B = { testInput.coordY, testInput.coordZ, testInput.coordW };

            uint16_t3 Vec3ASmall = uint16_t3(Vec3A & smallBitsMask_3);
            uint16_t3 Vec3BSmall = uint16_t3(Vec3B & smallBitsMask_3);
            uint16_t3 Vec3AMedium = uint16_t3(Vec3A & mediumBitsMask_3);
            uint16_t3 Vec3BMedium = uint16_t3(Vec3B & mediumBitsMask_3);
            uint32_t3 Vec3AFull = uint32_t3(Vec3A & fullBitsMask_3);
            uint32_t3 Vec3BFull = uint32_t3(Vec3B & fullBitsMask_3);

            uint64_t4 Vec4A = { testInput.coordX, testInput.coordY, testInput.coordZ, testInput.coordW };
            uint64_t4 Vec4B = { testInput.coordY, testInput.coordZ, testInput.coordW, testInput.coordX };

            uint16_t4 Vec4ASmall = uint16_t4(Vec4A & smallBitsMask_4);
            uint16_t4 Vec4BSmall = uint16_t4(Vec4B & smallBitsMask_4);
            uint16_t4 Vec4AMedium = uint16_t4(Vec4A & mediumBitsMask_4);
            uint16_t4 Vec4BMedium = uint16_t4(Vec4B & mediumBitsMask_4);
            uint16_t4 Vec4AFull = uint16_t4(Vec4A & fullBitsMask_4);
            uint16_t4 Vec4BFull = uint16_t4(Vec4B & fullBitsMask_4);

            // Signed vectors can't just have their highest bits masked off, for them to preserve sign we also need to left shift then right shift them
            // so their highest bits are all 0s or 1s depending on the sign of the number they encode

            int16_t2 Vec2ASignedSmall = int16_t2(Vec2ASmall << uint16_t(16 - smallBits_2)) >> int16_t(16 - smallBits_2);
            int16_t2 Vec2BSignedSmall = int16_t2(Vec2BSmall << uint16_t(16 - smallBits_2)) >> int16_t(16 - smallBits_2);
            int16_t2 Vec2ASignedMedium = int16_t2(Vec2AMedium << uint16_t(16 - mediumBits_2)) >> int16_t(16 - mediumBits_2);
            int16_t2 Vec2BSignedMedium = int16_t2(Vec2BMedium << uint16_t(16 - mediumBits_2)) >> int16_t(16 - mediumBits_2);
            int32_t2 Vec2ASignedFull = int32_t2(Vec2AFull << uint32_t(32 - fullBits_2)) >> int32_t(32 - fullBits_2);
            int32_t2 Vec2BSignedFull = int32_t2(Vec2BFull << uint32_t(32 - fullBits_2)) >> int32_t(32 - fullBits_2);

            int16_t3 Vec3ASignedSmall = int16_t3(Vec3ASmall << uint16_t(16 - smallBits_3)) >> int16_t(16 - smallBits_3);
            int16_t3 Vec3BSignedSmall = int16_t3(Vec3BSmall << uint16_t(16 - smallBits_3)) >> int16_t(16 - smallBits_3);
            int16_t3 Vec3ASignedMedium = int16_t3(Vec3AMedium << uint16_t(16 - mediumBits_3)) >> int16_t(16 - mediumBits_3);
            int16_t3 Vec3BSignedMedium = int16_t3(Vec3BMedium << uint16_t(16 - mediumBits_3)) >> int16_t(16 - mediumBits_3);
            int32_t3 Vec3ASignedFull = int32_t3(Vec3AFull << uint32_t(32 - fullBits_3)) >> int32_t(32 - fullBits_3);
            int32_t3 Vec3BSignedFull = int32_t3(Vec3BFull << uint32_t(32 - fullBits_3)) >> int32_t(32 - fullBits_3);

            int16_t4 Vec4ASignedSmall = int16_t4(Vec4ASmall << uint16_t(16 - smallBits_4)) >> int16_t(16 - smallBits_4);
            int16_t4 Vec4BSignedSmall = int16_t4(Vec4BSmall << uint16_t(16 - smallBits_4)) >> int16_t(16 - smallBits_4);
            int16_t4 Vec4ASignedMedium = int16_t4(Vec4AMedium << uint16_t(16 - mediumBits_4)) >> int16_t(16 - mediumBits_4);
            int16_t4 Vec4BSignedMedium = int16_t4(Vec4BMedium << uint16_t(16 - mediumBits_4)) >> int16_t(16 - mediumBits_4);
            int16_t4 Vec4ASignedFull = int16_t4(Vec4AFull << uint16_t(16 - fullBits_4)) >> int16_t(16 - fullBits_4);
            int16_t4 Vec4BSignedFull = int16_t4(Vec4BFull << uint16_t(16 - fullBits_4)) >> int16_t(16 - fullBits_4);

            // Plus
            expected.mortonPlus_small_2 = morton::code<false, smallBits_2, 2>::create(Vec2ASmall + Vec2BSmall);
            expected.mortonPlus_medium_2 = morton::code<false, mediumBits_2, 2>::create(Vec2AMedium + Vec2BMedium);
            expected.mortonPlus_full_2 = morton::code<false, fullBits_2, 2>::create(Vec2AFull + Vec2BFull);
            expected.mortonPlus_emulated_2 = morton::code<false, fullBits_2, 2, emulated_uint64_t>::create(Vec2AFull + Vec2BFull);

            expected.mortonPlus_small_3 = morton::code<false, smallBits_3, 3>::create(Vec3ASmall + Vec3BSmall);
            expected.mortonPlus_medium_3 = morton::code<false, mediumBits_3, 3>::create(Vec3AMedium + Vec3BMedium);
            expected.mortonPlus_full_3 = morton::code<false, fullBits_3, 3>::create(Vec3AFull + Vec3BFull);
            expected.mortonPlus_emulated_3 = morton::code<false, fullBits_3, 3, emulated_uint64_t>::create(Vec3AFull + Vec3BFull);

            expected.mortonPlus_small_4 = morton::code<false, smallBits_4, 4>::create(Vec4ASmall + Vec4BSmall);
            expected.mortonPlus_medium_4 = morton::code<false, mediumBits_4, 4>::create(Vec4AMedium + Vec4BMedium);
            expected.mortonPlus_full_4 = morton::code<false, fullBits_4, 4>::create(Vec4AFull + Vec4BFull);
            expected.mortonPlus_emulated_4 = morton::code<false, fullBits_4, 4, emulated_uint64_t>::create(Vec4AFull + Vec4BFull);

            // Minus
            expected.mortonMinus_small_2 = morton::code<false, smallBits_2, 2>::create(Vec2ASmall - Vec2BSmall);
            expected.mortonMinus_medium_2 = morton::code<false, mediumBits_2, 2>::create(Vec2AMedium - Vec2BMedium);
            expected.mortonMinus_full_2 = morton::code<false, fullBits_2, 2>::create(Vec2AFull - Vec2BFull);
            expected.mortonMinus_emulated_2 = morton::code<false, fullBits_2, 2, emulated_uint64_t>::create(Vec2AFull - Vec2BFull);

            expected.mortonMinus_small_3 = morton::code<false, smallBits_3, 3>::create(Vec3ASmall - Vec3BSmall);
            expected.mortonMinus_medium_3 = morton::code<false, mediumBits_3, 3>::create(Vec3AMedium - Vec3BMedium);
            expected.mortonMinus_full_3 = morton::code<false, fullBits_3, 3>::create(Vec3AFull - Vec3BFull);
            expected.mortonMinus_emulated_3 = morton::code<false, fullBits_3, 3, emulated_uint64_t>::create(Vec3AFull - Vec3BFull);

            expected.mortonMinus_small_4 = morton::code<false, smallBits_4, 4>::create(Vec4ASmall - Vec4BSmall);
            expected.mortonMinus_medium_4 = morton::code<false, mediumBits_4, 4>::create(Vec4AMedium - Vec4BMedium);
            expected.mortonMinus_full_4 = morton::code<false, fullBits_4, 4>::create(Vec4AFull - Vec4BFull);
            expected.mortonMinus_emulated_4 = morton::code<false, fullBits_4, 4, emulated_uint64_t>::create(Vec4AFull - Vec4BFull);

            // Coordinate-wise equality
            expected.mortonEqual_small_2 = uint32_t2(glm::equal(Vec2ASmall, Vec2BSmall));
            expected.mortonEqual_medium_2 = uint32_t2(glm::equal(Vec2AMedium, Vec2BMedium));
            expected.mortonEqual_full_2 = uint32_t2(glm::equal(Vec2AFull, Vec2BFull));
            expected.mortonEqual_emulated_2 = uint32_t2(glm::equal(Vec2AFull, Vec2BFull));

            expected.mortonEqual_small_3 = uint32_t3(glm::equal(Vec3ASmall, Vec3BSmall));
            expected.mortonEqual_medium_3 = uint32_t3(glm::equal(Vec3AMedium, Vec3BMedium));
            expected.mortonEqual_full_3 = uint32_t3(glm::equal(Vec3AFull, Vec3BFull));
            expected.mortonEqual_emulated_3 = uint32_t3(glm::equal(Vec3AFull, Vec3BFull));

            expected.mortonEqual_small_4 = uint32_t4(glm::equal(Vec4ASmall, Vec4BSmall));
            expected.mortonEqual_medium_4 = uint32_t4(glm::equal(Vec4AMedium, Vec4BMedium));
            expected.mortonEqual_full_4 = uint32_t4(glm::equal(Vec4AFull, Vec4BFull));

            // Coordinate-wise unsigned inequality (just testing with less)
            expected.mortonUnsignedLess_small_2 = uint32_t2(glm::lessThan(Vec2ASmall, Vec2BSmall));
            expected.mortonUnsignedLess_medium_2 = uint32_t2(glm::lessThan(Vec2AMedium, Vec2BMedium));
            expected.mortonUnsignedLess_full_2 = uint32_t2(glm::lessThan(Vec2AFull, Vec2BFull));
            expected.mortonUnsignedLess_emulated_2 = uint32_t2(glm::lessThan(Vec2AFull, Vec2BFull));

            expected.mortonUnsignedLess_small_3 = uint32_t3(glm::lessThan(Vec3ASmall, Vec3BSmall));
            expected.mortonUnsignedLess_medium_3 = uint32_t3(glm::lessThan(Vec3AMedium, Vec3BMedium));
            expected.mortonUnsignedLess_full_3 = uint32_t3(glm::lessThan(Vec3AFull, Vec3BFull));
            expected.mortonUnsignedLess_emulated_3 = uint32_t3(glm::lessThan(Vec3AFull, Vec3BFull));

            expected.mortonUnsignedLess_small_4 = uint32_t4(glm::lessThan(Vec4ASmall, Vec4BSmall));
            expected.mortonUnsignedLess_medium_4 = uint32_t4(glm::lessThan(Vec4AMedium, Vec4BMedium));
            expected.mortonUnsignedLess_full_4 = uint32_t4(glm::lessThan(Vec4AFull, Vec4BFull));

            // Coordinate-wise signed inequality
            expected.mortonSignedLess_small_2 = uint32_t2(glm::lessThan(Vec2ASignedSmall, Vec2BSignedSmall));
            expected.mortonSignedLess_medium_2 = uint32_t2(glm::lessThan(Vec2ASignedMedium, Vec2BSignedMedium));
            expected.mortonSignedLess_full_2 = uint32_t2(glm::lessThan(Vec2ASignedFull, Vec2BSignedFull));

            expected.mortonSignedLess_small_3 = uint32_t3(glm::lessThan(Vec3ASignedSmall, Vec3BSignedSmall));
            expected.mortonSignedLess_medium_3 = uint32_t3(glm::lessThan(Vec3ASignedMedium, Vec3BSignedMedium));
            expected.mortonSignedLess_full_3 = uint32_t3(glm::lessThan(Vec3ASignedFull, Vec3BSignedFull));

            expected.mortonSignedLess_small_4 = uint32_t4(glm::lessThan(Vec4ASignedSmall, Vec4BSignedSmall));
            expected.mortonSignedLess_medium_4 = uint32_t4(glm::lessThan(Vec4ASignedMedium, Vec4BSignedMedium));
            expected.mortonSignedLess_full_4 = uint32_t4(glm::lessThan(Vec4ASignedFull, Vec4BSignedFull));

            uint16_t castedShift = uint16_t(testInput.shift);
            // Left-shift
            expected.mortonLeftShift_small_2 = morton::code<false, smallBits_2, 2>::create((Vec2ASmall << uint16_t(castedShift % smallBits_2)) & uint16_t(smallBitsMask_2));
            expected.mortonLeftShift_medium_2 = morton::code<false, mediumBits_2, 2>::create((Vec2AMedium << uint16_t(castedShift % mediumBits_2)) & uint16_t(mediumBitsMask_2));
            expected.mortonLeftShift_full_2 = morton::code<false, fullBits_2, 2>::create((Vec2AFull << uint32_t(castedShift % fullBits_2)) & uint32_t(fullBitsMask_2));
            expected.mortonLeftShift_emulated_2 = morton::code<false, fullBits_2, 2, emulated_uint64_t>::create((Vec2AFull << uint32_t(castedShift % fullBits_2)) & uint32_t(fullBitsMask_2));

            expected.mortonLeftShift_small_3 = morton::code<false, smallBits_3, 3>::create((Vec3ASmall << uint16_t(castedShift % smallBits_3)) & uint16_t(smallBitsMask_3));
            expected.mortonLeftShift_medium_3 = morton::code<false, mediumBits_3, 3>::create((Vec3AMedium << uint16_t(castedShift % mediumBits_3)) & uint16_t(mediumBitsMask_3));
            expected.mortonLeftShift_full_3 = morton::code<false, fullBits_3, 3>::create((Vec3AFull << uint32_t(castedShift % fullBits_3)) & uint32_t(fullBitsMask_3));
            expected.mortonLeftShift_emulated_3 = morton::code<false, fullBits_3, 3, emulated_uint64_t>::create((Vec3AFull << uint32_t(castedShift % fullBits_3)) & uint32_t(fullBitsMask_3));

            expected.mortonLeftShift_small_4 = morton::code<false, smallBits_4, 4>::create((Vec4ASmall << uint16_t(castedShift % smallBits_4)) & uint16_t(smallBitsMask_4));
            expected.mortonLeftShift_medium_4 = morton::code<false, mediumBits_4, 4>::create((Vec4AMedium << uint16_t(castedShift % mediumBits_4)) & uint16_t(mediumBitsMask_4));
            expected.mortonLeftShift_full_4 = morton::code<false, fullBits_4, 4>::create((Vec4AFull << uint16_t(castedShift % fullBits_4)) & uint16_t(fullBitsMask_4));
            expected.mortonLeftShift_emulated_4 = morton::code<false, fullBits_4, 4, emulated_uint64_t>::create((Vec4AFull << uint16_t(castedShift % fullBits_4)) & uint16_t(fullBitsMask_4));

            // Unsigned right-shift
            expected.mortonUnsignedRightShift_small_2 = morton::code<false, smallBits_2, 2>::create((Vec2ASmall >> uint16_t(castedShift % smallBits_2)) & uint16_t(smallBitsMask_2));
            expected.mortonUnsignedRightShift_medium_2 = morton::code<false, mediumBits_2, 2>::create((Vec2AMedium >> uint16_t(castedShift % mediumBits_2)) & uint16_t(mediumBitsMask_2));
            expected.mortonUnsignedRightShift_full_2 = morton::code<false, fullBits_2, 2>::create((Vec2AFull >> uint32_t(castedShift % fullBits_2)) & uint32_t(fullBitsMask_2));
            expected.mortonUnsignedRightShift_emulated_2 = morton::code<false, fullBits_2, 2, emulated_uint64_t>::create((Vec2AFull >> uint32_t(castedShift % fullBits_2)) & uint32_t(fullBitsMask_2));

            expected.mortonUnsignedRightShift_small_3 = morton::code<false, smallBits_3, 3>::create((Vec3ASmall >> uint16_t(castedShift % smallBits_3)) & uint16_t(smallBitsMask_3));
            expected.mortonUnsignedRightShift_medium_3 = morton::code<false, mediumBits_3, 3>::create((Vec3AMedium >> uint16_t(castedShift % mediumBits_3)) & uint16_t(mediumBitsMask_3));
            expected.mortonUnsignedRightShift_full_3 = morton::code<false, fullBits_3, 3>::create((Vec3AFull >> uint32_t(castedShift % fullBits_3)) & uint32_t(fullBitsMask_3));
            expected.mortonUnsignedRightShift_emulated_3 = morton::code<false, fullBits_3, 3, emulated_uint64_t>::create((Vec3AFull >> uint32_t(castedShift % fullBits_3)) & uint32_t(fullBitsMask_3));

            expected.mortonUnsignedRightShift_small_4 = morton::code<false, smallBits_4, 4>::create((Vec4ASmall >> uint16_t(castedShift % smallBits_4)) & uint16_t(smallBitsMask_4));
            expected.mortonUnsignedRightShift_medium_4 = morton::code<false, mediumBits_4, 4>::create((Vec4AMedium >> uint16_t(castedShift % mediumBits_4)) & uint16_t(mediumBitsMask_4));
            expected.mortonUnsignedRightShift_full_4 = morton::code<false, fullBits_4, 4>::create((Vec4AFull >> uint16_t(castedShift % fullBits_4)) & uint16_t(fullBitsMask_4));
            expected.mortonUnsignedRightShift_emulated_4 = morton::code<false, fullBits_4, 4, emulated_uint64_t>::create((Vec4AFull >> uint16_t(castedShift % fullBits_4)) & uint16_t(fullBitsMask_4));

            // Signed right-shift
            expected.mortonSignedRightShift_small_2 = morton::code<true, smallBits_2, 2>::create((Vec2ASignedSmall >> int16_t(castedShift % smallBits_2)) & int16_t(smallBitsMask_2));
            expected.mortonSignedRightShift_medium_2 = morton::code<true, mediumBits_2, 2>::create((Vec2ASignedMedium >> int16_t(castedShift % mediumBits_2)) & int16_t(mediumBitsMask_2));
            expected.mortonSignedRightShift_full_2 = morton::code<true, fullBits_2, 2>::create((Vec2ASignedFull >> int32_t(castedShift % fullBits_2)) & int32_t(fullBitsMask_2));

            expected.mortonSignedRightShift_small_3 = morton::code<true, smallBits_3, 3>::create((Vec3ASignedSmall >> int16_t(castedShift % smallBits_3)) & int16_t(smallBitsMask_3));
            expected.mortonSignedRightShift_medium_3 = morton::code<true, mediumBits_3, 3>::create((Vec3ASignedMedium >> int16_t(castedShift % mediumBits_3)) & int16_t(mediumBitsMask_3));
            expected.mortonSignedRightShift_full_3 = morton::code<true, fullBits_3, 3>::create((Vec3ASignedFull >> int32_t(castedShift % fullBits_3)) & int32_t(fullBitsMask_3));

            expected.mortonSignedRightShift_small_4 = morton::code<true, smallBits_4, 4>::create((Vec4ASignedSmall >> int16_t(castedShift % smallBits_4)) & int16_t(smallBitsMask_4));
            expected.mortonSignedRightShift_medium_4 = morton::code<true, mediumBits_4, 4>::create((Vec4ASignedMedium >> int16_t(castedShift % mediumBits_4)) & int16_t(mediumBitsMask_4));
            expected.mortonSignedRightShift_full_4 = morton::code<true, fullBits_4, 4>::create((Vec4ASignedFull >> int16_t(castedShift % fullBits_4)) & int16_t(fullBitsMask_4));
        }

        return expected;
    }

    void verifyTestResults(const TestValues& expectedTestValues, const TestValues& testValues, ITester::TestType testType) override
    {
        verifyTestValue("emulatedAnd", expectedTestValues.emulatedAnd, testValues.emulatedAnd, testType);
        verifyTestValue("emulatedOr", expectedTestValues.emulatedOr, testValues.emulatedOr, testType);
        verifyTestValue("emulatedXor", expectedTestValues.emulatedXor, testValues.emulatedXor, testType);
        verifyTestValue("emulatedNot", expectedTestValues.emulatedNot, testValues.emulatedNot, testType);
        verifyTestValue("emulatedPlus", expectedTestValues.emulatedPlus, testValues.emulatedPlus, testType);
        verifyTestValue("emulatedMinus", expectedTestValues.emulatedMinus, testValues.emulatedMinus, testType);
        verifyTestValue("emulatedLess", expectedTestValues.emulatedLess, testValues.emulatedLess, testType);
        verifyTestValue("emulatedLessEqual", expectedTestValues.emulatedLessEqual, testValues.emulatedLessEqual, testType);
        verifyTestValue("emulatedGreater", expectedTestValues.emulatedGreater, testValues.emulatedGreater, testType);
        verifyTestValue("emulatedGreaterEqual", expectedTestValues.emulatedGreaterEqual, testValues.emulatedGreaterEqual, testType);
        verifyTestValue("emulatedLeftShifted", expectedTestValues.emulatedLeftShifted, testValues.emulatedLeftShifted, testType);
        verifyTestValue("emulatedUnsignedRightShifted", expectedTestValues.emulatedUnsignedRightShifted, testValues.emulatedUnsignedRightShifted, testType);
        verifyTestValue("emulatedSignedRightShifted", expectedTestValues.emulatedSignedRightShifted, testValues.emulatedSignedRightShifted, testType);

        // Morton Plus
        verifyTestValue("mortonPlus_small_2", expectedTestValues.mortonPlus_small_2, testValues.mortonPlus_small_2, testType);
        verifyTestValue("mortonPlus_medium_2", expectedTestValues.mortonPlus_medium_2, testValues.mortonPlus_medium_2, testType);
        verifyTestValue("mortonPlus_full_2", expectedTestValues.mortonPlus_full_2, testValues.mortonPlus_full_2, testType);
        verifyTestValue("mortonPlus_emulated_2", expectedTestValues.mortonPlus_emulated_2, testValues.mortonPlus_emulated_2, testType);

        verifyTestValue("mortonPlus_small_3", expectedTestValues.mortonPlus_small_3, testValues.mortonPlus_small_3, testType);
        verifyTestValue("mortonPlus_medium_3", expectedTestValues.mortonPlus_medium_3, testValues.mortonPlus_medium_3, testType);
        verifyTestValue("mortonPlus_full_3", expectedTestValues.mortonPlus_full_3, testValues.mortonPlus_full_3, testType);
        verifyTestValue("mortonPlus_emulated_3", expectedTestValues.mortonPlus_emulated_3, testValues.mortonPlus_emulated_3, testType);

        verifyTestValue("mortonPlus_small_4", expectedTestValues.mortonPlus_small_4, testValues.mortonPlus_small_4, testType);
        verifyTestValue("mortonPlus_medium_4", expectedTestValues.mortonPlus_medium_4, testValues.mortonPlus_medium_4, testType);
        verifyTestValue("mortonPlus_full_4", expectedTestValues.mortonPlus_full_4, testValues.mortonPlus_full_4, testType);
        verifyTestValue("mortonPlus_emulated_4", expectedTestValues.mortonPlus_emulated_4, testValues.mortonPlus_emulated_4, testType);

        // Morton Minus
        verifyTestValue("mortonMinus_small_2", expectedTestValues.mortonMinus_small_2, testValues.mortonMinus_small_2, testType);
        verifyTestValue("mortonMinus_medium_2", expectedTestValues.mortonMinus_medium_2, testValues.mortonMinus_medium_2, testType);
        verifyTestValue("mortonMinus_full_2", expectedTestValues.mortonMinus_full_2, testValues.mortonMinus_full_2, testType);
        verifyTestValue("mortonMinus_emulated_2", expectedTestValues.mortonMinus_emulated_2, testValues.mortonMinus_emulated_2, testType);

        verifyTestValue("mortonMinus_small_3", expectedTestValues.mortonMinus_small_3, testValues.mortonMinus_small_3, testType);
        verifyTestValue("mortonMinus_medium_3", expectedTestValues.mortonMinus_medium_3, testValues.mortonMinus_medium_3, testType);
        verifyTestValue("mortonMinus_full_3", expectedTestValues.mortonMinus_full_3, testValues.mortonMinus_full_3, testType);
        verifyTestValue("mortonMinus_emulated_3", expectedTestValues.mortonMinus_emulated_3, testValues.mortonMinus_emulated_3, testType);

        verifyTestValue("mortonMinus_small_4", expectedTestValues.mortonMinus_small_4, testValues.mortonMinus_small_4, testType);
        verifyTestValue("mortonMinus_medium_4", expectedTestValues.mortonMinus_medium_4, testValues.mortonMinus_medium_4, testType);
        verifyTestValue("mortonMinus_full_4", expectedTestValues.mortonMinus_full_4, testValues.mortonMinus_full_4, testType);
        verifyTestValue("mortonMinus_emulated_4", expectedTestValues.mortonMinus_emulated_4, testValues.mortonMinus_emulated_4, testType);

        // Morton coordinate-wise equality
        verifyTestValue("mortonEqual_small_2", expectedTestValues.mortonEqual_small_2, testValues.mortonEqual_small_2, testType);
        verifyTestValue("mortonEqual_medium_2", expectedTestValues.mortonEqual_medium_2, testValues.mortonEqual_medium_2, testType);
        verifyTestValue("mortonEqual_full_2", expectedTestValues.mortonEqual_full_2, testValues.mortonEqual_full_2, testType);
        verifyTestValue("mortonEqual_emulated_2", expectedTestValues.mortonEqual_emulated_2, testValues.mortonEqual_emulated_2, testType);

        verifyTestValue("mortonEqual_small_3", expectedTestValues.mortonEqual_small_3, testValues.mortonEqual_small_3, testType);
        verifyTestValue("mortonEqual_medium_3", expectedTestValues.mortonEqual_medium_3, testValues.mortonEqual_medium_3, testType);
        verifyTestValue("mortonEqual_full_3", expectedTestValues.mortonEqual_full_3, testValues.mortonEqual_full_3, testType);
        verifyTestValue("mortonEqual_emulated_3", expectedTestValues.mortonEqual_emulated_3, testValues.mortonEqual_emulated_3, testType);

        verifyTestValue("mortonEqual_small_4", expectedTestValues.mortonEqual_small_4, testValues.mortonEqual_small_4, testType);
        verifyTestValue("mortonEqual_medium_4", expectedTestValues.mortonEqual_medium_4, testValues.mortonEqual_medium_4, testType);
        verifyTestValue("mortonEqual_full_4", expectedTestValues.mortonEqual_full_4, testValues.mortonEqual_full_4, testType);

        // Morton coordinate-wise unsigned inequality
        verifyTestValue("mortonUnsignedLess_small_2", expectedTestValues.mortonUnsignedLess_small_2, testValues.mortonUnsignedLess_small_2, testType);
        verifyTestValue("mortonUnsignedLess_medium_2", expectedTestValues.mortonUnsignedLess_medium_2, testValues.mortonUnsignedLess_medium_2, testType);
        verifyTestValue("mortonUnsignedLess_full_2", expectedTestValues.mortonUnsignedLess_full_2, testValues.mortonUnsignedLess_full_2, testType);
        verifyTestValue("mortonUnsignedLess_emulated_2", expectedTestValues.mortonUnsignedLess_emulated_2, testValues.mortonUnsignedLess_emulated_2, testType);

        verifyTestValue("mortonUnsignedLess_small_3", expectedTestValues.mortonUnsignedLess_small_3, testValues.mortonUnsignedLess_small_3, testType);
        verifyTestValue("mortonUnsignedLess_medium_3", expectedTestValues.mortonUnsignedLess_medium_3, testValues.mortonUnsignedLess_medium_3, testType);
        verifyTestValue("mortonUnsignedLess_full_3", expectedTestValues.mortonUnsignedLess_full_3, testValues.mortonUnsignedLess_full_3, testType);
        verifyTestValue("mortonUnsignedLess_emulated_3", expectedTestValues.mortonUnsignedLess_emulated_3, testValues.mortonUnsignedLess_emulated_3, testType);

        verifyTestValue("mortonUnsignedLess_small_4", expectedTestValues.mortonUnsignedLess_small_4, testValues.mortonUnsignedLess_small_4, testType);
        verifyTestValue("mortonUnsignedLess_medium_4", expectedTestValues.mortonUnsignedLess_medium_4, testValues.mortonUnsignedLess_medium_4, testType);
        verifyTestValue("mortonUnsignedLess_full_4", expectedTestValues.mortonUnsignedLess_full_4, testValues.mortonUnsignedLess_full_4, testType);

        // Morton coordinate-wise signed inequality
        verifyTestValue("mortonSignedLess_small_2", expectedTestValues.mortonSignedLess_small_2, testValues.mortonSignedLess_small_2, testType);
        verifyTestValue("mortonSignedLess_medium_2", expectedTestValues.mortonSignedLess_medium_2, testValues.mortonSignedLess_medium_2, testType);
        verifyTestValue("mortonSignedLess_full_2", expectedTestValues.mortonSignedLess_full_2, testValues.mortonSignedLess_full_2, testType);

        verifyTestValue("mortonSignedLess_small_3", expectedTestValues.mortonSignedLess_small_3, testValues.mortonSignedLess_small_3, testType);
        verifyTestValue("mortonSignedLess_medium_3", expectedTestValues.mortonSignedLess_medium_3, testValues.mortonSignedLess_medium_3, testType);
        verifyTestValue("mortonSignedLess_full_3", expectedTestValues.mortonSignedLess_full_3, testValues.mortonSignedLess_full_3, testType);

        verifyTestValue("mortonSignedLess_small_4", expectedTestValues.mortonSignedLess_small_4, testValues.mortonSignedLess_small_4, testType);
        verifyTestValue("mortonSignedLess_medium_4", expectedTestValues.mortonSignedLess_medium_4, testValues.mortonSignedLess_medium_4, testType);
        verifyTestValue("mortonSignedLess_full_4", expectedTestValues.mortonSignedLess_full_4, testValues.mortonSignedLess_full_4, testType);

        // Morton left-shift
        verifyTestValue("mortonLeftShift_small_2", expectedTestValues.mortonLeftShift_small_2, testValues.mortonLeftShift_small_2, testType);
        verifyTestValue("mortonLeftShift_medium_2", expectedTestValues.mortonLeftShift_medium_2, testValues.mortonLeftShift_medium_2, testType);
        verifyTestValue("mortonLeftShift_full_2", expectedTestValues.mortonLeftShift_full_2, testValues.mortonLeftShift_full_2, testType);
        verifyTestValue("mortonLeftShift_emulated_2", expectedTestValues.mortonLeftShift_emulated_2, testValues.mortonLeftShift_emulated_2, testType);

        verifyTestValue("mortonLeftShift_small_3", expectedTestValues.mortonLeftShift_small_3, testValues.mortonLeftShift_small_3, testType);
        verifyTestValue("mortonLeftShift_medium_3", expectedTestValues.mortonLeftShift_medium_3, testValues.mortonLeftShift_medium_3, testType);
        verifyTestValue("mortonLeftShift_full_3", expectedTestValues.mortonLeftShift_full_3, testValues.mortonLeftShift_full_3, testType);
        verifyTestValue("mortonLeftShift_emulated_3", expectedTestValues.mortonLeftShift_emulated_3, testValues.mortonLeftShift_emulated_3, testType);

        verifyTestValue("mortonLeftShift_small_4", expectedTestValues.mortonLeftShift_small_4, testValues.mortonLeftShift_small_4, testType);
        verifyTestValue("mortonLeftShift_medium_4", expectedTestValues.mortonLeftShift_medium_4, testValues.mortonLeftShift_medium_4, testType);
        verifyTestValue("mortonLeftShift_full_4", expectedTestValues.mortonLeftShift_full_4, testValues.mortonLeftShift_full_4, testType);
        verifyTestValue("mortonLeftShift_emulated_4", expectedTestValues.mortonLeftShift_emulated_4, testValues.mortonLeftShift_emulated_4, testType);

        // Morton unsigned right-shift
        verifyTestValue("mortonUnsignedRightShift_small_2", expectedTestValues.mortonUnsignedRightShift_small_2, testValues.mortonUnsignedRightShift_small_2, testType);
        verifyTestValue("mortonUnsignedRightShift_medium_2", expectedTestValues.mortonUnsignedRightShift_medium_2, testValues.mortonUnsignedRightShift_medium_2, testType);
        verifyTestValue("mortonUnsignedRightShift_full_2", expectedTestValues.mortonUnsignedRightShift_full_2, testValues.mortonUnsignedRightShift_full_2, testType);
        verifyTestValue("mortonUnsignedRightShift_emulated_2", expectedTestValues.mortonUnsignedRightShift_emulated_2, testValues.mortonUnsignedRightShift_emulated_2, testType);

        verifyTestValue("mortonUnsignedRightShift_small_3", expectedTestValues.mortonUnsignedRightShift_small_3, testValues.mortonUnsignedRightShift_small_3, testType);
        verifyTestValue("mortonUnsignedRightShift_medium_3", expectedTestValues.mortonUnsignedRightShift_medium_3, testValues.mortonUnsignedRightShift_medium_3, testType);
        verifyTestValue("mortonUnsignedRightShift_full_3", expectedTestValues.mortonUnsignedRightShift_full_3, testValues.mortonUnsignedRightShift_full_3, testType);
        verifyTestValue("mortonUnsignedRightShift_emulated_3", expectedTestValues.mortonUnsignedRightShift_emulated_3, testValues.mortonUnsignedRightShift_emulated_3, testType);

        verifyTestValue("mortonUnsignedRightShift_small_4", expectedTestValues.mortonUnsignedRightShift_small_4, testValues.mortonUnsignedRightShift_small_4, testType);
        verifyTestValue("mortonUnsignedRightShift_medium_4", expectedTestValues.mortonUnsignedRightShift_medium_4, testValues.mortonUnsignedRightShift_medium_4, testType);
        verifyTestValue("mortonUnsignedRightShift_full_4", expectedTestValues.mortonUnsignedRightShift_full_4, testValues.mortonUnsignedRightShift_full_4, testType);
        verifyTestValue("mortonUnsignedRightShift_emulated_4", expectedTestValues.mortonUnsignedRightShift_emulated_4, testValues.mortonUnsignedRightShift_emulated_4, testType);

        // Morton signed right-shift
        verifyTestValue("mortonSignedRightShift_small_2", expectedTestValues.mortonSignedRightShift_small_2, testValues.mortonSignedRightShift_small_2, testType);
        verifyTestValue("mortonSignedRightShift_medium_2", expectedTestValues.mortonSignedRightShift_medium_2, testValues.mortonSignedRightShift_medium_2, testType);
        verifyTestValue("mortonSignedRightShift_full_2", expectedTestValues.mortonSignedRightShift_full_2, testValues.mortonSignedRightShift_full_2, testType);

        verifyTestValue("mortonSignedRightShift_small_3", expectedTestValues.mortonSignedRightShift_small_3, testValues.mortonSignedRightShift_small_3, testType);
        verifyTestValue("mortonSignedRightShift_medium_3", expectedTestValues.mortonSignedRightShift_medium_3, testValues.mortonSignedRightShift_medium_3, testType);
        verifyTestValue("mortonSignedRightShift_full_3", expectedTestValues.mortonSignedRightShift_full_3, testValues.mortonSignedRightShift_full_3, testType);

        verifyTestValue("mortonSignedRightShift_small_4", expectedTestValues.mortonSignedRightShift_small_4, testValues.mortonSignedRightShift_small_4, testType);
        verifyTestValue("mortonSignedRightShift_medium_4", expectedTestValues.mortonSignedRightShift_medium_4, testValues.mortonSignedRightShift_medium_4, testType);
        verifyTestValue("mortonSignedRightShift_full_4", expectedTestValues.mortonSignedRightShift_full_4, testValues.mortonSignedRightShift_full_4, testType);
    }
};

#endif