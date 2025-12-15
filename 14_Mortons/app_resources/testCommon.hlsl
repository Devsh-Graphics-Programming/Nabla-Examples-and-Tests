#include "common.hlsl"

struct TestExecutor
{
	void operator()(NBL_CONST_REF_ARG(InputTestValues) input, NBL_REF_ARG(TestValues) output)
	{
		emulated_uint64_t emulatedA = _static_cast<emulated_uint64_t>(input.generatedA);
		emulated_uint64_t emulatedB = _static_cast<emulated_uint64_t>(input.generatedB);
		emulated_int64_t signedEmulatedA = _static_cast<emulated_int64_t>(input.generatedA);

		// Emulated int tests
		output.emulatedAnd = emulatedA & emulatedB;
		output.emulatedOr = emulatedA | emulatedB;
		output.emulatedXor = emulatedA ^ emulatedB;
		output.emulatedNot = emulatedA.operator~();
		output.emulatedPlus = emulatedA + emulatedB;
		output.emulatedMinus = emulatedA - emulatedB;
		output.emulatedLess = uint32_t(emulatedA < emulatedB);
		output.emulatedLessEqual = uint32_t(emulatedA <= emulatedB);
		output.emulatedGreater = uint32_t(emulatedA > emulatedB);
		output.emulatedGreaterEqual = uint32_t(emulatedA >= emulatedB);

		left_shift_operator<emulated_uint64_t> leftShift;
		output.emulatedLeftShifted = leftShift(emulatedA, input.shift);

		arithmetic_right_shift_operator<emulated_uint64_t> unsignedRightShift;
		output.emulatedUnsignedRightShifted = unsignedRightShift(emulatedA, input.shift);

		arithmetic_right_shift_operator<emulated_int64_t> signedRightShift;
		output.emulatedSignedRightShifted = signedRightShift(signedEmulatedA, input.shift);

		output.emulatedUnaryMinus = signedEmulatedA.operator-();

		// Morton tests
		uint64_t2 Vec2A = { input.coordX, input.coordY };
		uint64_t2 Vec2B = { input.coordZ, input.coordW };

		uint64_t3 Vec3A = { input.coordX, input.coordY, input.coordZ };
		uint64_t3 Vec3B = { input.coordY, input.coordZ, input.coordW };

		uint64_t4 Vec4A = { input.coordX, input.coordY, input.coordZ, input.coordW };
		uint64_t4 Vec4B = { input.coordY, input.coordZ, input.coordW, input.coordX };

		uint16_t2 Vec2ASmall = createAnyBitIntegerVecFromU64Vec<uint16_t, false, smallBits_2, 2>(Vec2A);
		uint16_t2 Vec2BSmall = createAnyBitIntegerVecFromU64Vec<uint16_t, false, smallBits_2, 2>(Vec2B);
		uint16_t2 Vec2AMedium = createAnyBitIntegerVecFromU64Vec<uint16_t, false, mediumBits_2, 2>(Vec2A);
		uint16_t2 Vec2BMedium = createAnyBitIntegerVecFromU64Vec<uint16_t, false, mediumBits_2, 2>(Vec2B);
		uint32_t2 Vec2AFull = createAnyBitIntegerVecFromU64Vec<uint32_t, false, fullBits_2, 2>(Vec2A);
  		uint32_t2 Vec2BFull = createAnyBitIntegerVecFromU64Vec<uint32_t, false, fullBits_2, 2>(Vec2B);

		uint16_t3 Vec3ASmall = createAnyBitIntegerVecFromU64Vec<uint16_t, false, smallBits_3, 3>(Vec3A);
		uint16_t3 Vec3BSmall = createAnyBitIntegerVecFromU64Vec<uint16_t, false, smallBits_3, 3>(Vec3B);
		uint16_t3 Vec3AMedium = createAnyBitIntegerVecFromU64Vec<uint16_t, false, mediumBits_3, 3>(Vec3A);
		uint16_t3 Vec3BMedium = createAnyBitIntegerVecFromU64Vec<uint16_t, false, mediumBits_3, 3>(Vec3B);
		uint32_t3 Vec3AFull = createAnyBitIntegerVecFromU64Vec<uint32_t, false, fullBits_3, 3>(Vec3A);
		uint32_t3 Vec3BFull = createAnyBitIntegerVecFromU64Vec<uint32_t, false, fullBits_3, 3>(Vec3B);

		uint16_t4 Vec4ASmall = createAnyBitIntegerVecFromU64Vec<uint16_t, false, smallBits_4, 4>(Vec4A);
		uint16_t4 Vec4BSmall = createAnyBitIntegerVecFromU64Vec<uint16_t, false, smallBits_4, 4>(Vec4B);
		uint16_t4 Vec4AMedium = createAnyBitIntegerVecFromU64Vec<uint16_t, false, mediumBits_4, 4>(Vec4A);
		uint16_t4 Vec4BMedium = createAnyBitIntegerVecFromU64Vec<uint16_t, false, mediumBits_4, 4>(Vec4B);
		uint16_t4 Vec4AFull = createAnyBitIntegerVecFromU64Vec<uint16_t, false, fullBits_4, 4>(Vec4A);
		uint16_t4 Vec4BFull = createAnyBitIntegerVecFromU64Vec<uint16_t, false, fullBits_4, 4>(Vec4B);

		int16_t2 Vec2ASignedSmall = createAnyBitIntegerVecFromU64Vec<int16_t, true, smallBits_2, 2>(Vec2A);
		int16_t2 Vec2BSignedSmall = createAnyBitIntegerVecFromU64Vec<int16_t, true, smallBits_2, 2>(Vec2B);
		int16_t2 Vec2ASignedMedium = createAnyBitIntegerVecFromU64Vec<int16_t, true,mediumBits_2, 2 >(Vec2A);
		int16_t2 Vec2BSignedMedium = createAnyBitIntegerVecFromU64Vec<int16_t, true, mediumBits_2, 2>(Vec2B);
		int32_t2 Vec2ASignedFull = createAnyBitIntegerVecFromU64Vec<int32_t, true, fullBits_2, 2>(Vec2A);
		int32_t2 Vec2BSignedFull = createAnyBitIntegerVecFromU64Vec<int32_t, true, fullBits_2, 2>(Vec2B);

		int16_t3 Vec3ASignedSmall = createAnyBitIntegerVecFromU64Vec<int16_t, true, smallBits_3, 3>(Vec3A);
		int16_t3 Vec3BSignedSmall = createAnyBitIntegerVecFromU64Vec<int16_t, true, smallBits_3, 3>(Vec3B);
		int16_t3 Vec3ASignedMedium = createAnyBitIntegerVecFromU64Vec<int16_t, true, mediumBits_3, 3>(Vec3A);
		int16_t3 Vec3BSignedMedium = createAnyBitIntegerVecFromU64Vec<int16_t, true, mediumBits_3, 3>(Vec3B);
		int32_t3 Vec3ASignedFull = createAnyBitIntegerVecFromU64Vec<int32_t, true, fullBits_3, 3>(Vec3A);
		int32_t3 Vec3BSignedFull = createAnyBitIntegerVecFromU64Vec<int32_t, true, fullBits_3, 3>(Vec3B);

		int16_t4 Vec4ASignedSmall = createAnyBitIntegerVecFromU64Vec<int16_t, true, smallBits_4, 4>(Vec4A);
		int16_t4 Vec4BSignedSmall = createAnyBitIntegerVecFromU64Vec<int16_t, true, smallBits_4, 4>(Vec4B);
		int16_t4 Vec4ASignedMedium = createAnyBitIntegerVecFromU64Vec<int16_t, true, mediumBits_4, 4>(Vec4A);
		int16_t4 Vec4BSignedMedium = createAnyBitIntegerVecFromU64Vec<int16_t, true, mediumBits_4, 4>(Vec4B);
		int16_t4 Vec4ASignedFull = createAnyBitIntegerVecFromU64Vec<int16_t, true, fullBits_4, 4>(Vec4A);
		int16_t4 Vec4BSignedFull = createAnyBitIntegerVecFromU64Vec<int16_t, true, fullBits_4, 4>(Vec4B);

		morton::code<false, smallBits_2, 2> morton_small_2A = createMortonFromU64Vec<false, smallBits_2, 2>(Vec2A);
		morton::code<false, mediumBits_2, 2> morton_medium_2A = createMortonFromU64Vec<false, mediumBits_2, 2>(Vec2A);
		morton::code<false, fullBits_2, 2> morton_full_2A = createMortonFromU64Vec<false, fullBits_2, 2>(Vec2A);
		morton::code<false, fullBits_2, 2, emulated_uint64_t> morton_emulated_2A = createMortonFromU64Vec<false, fullBits_2, 2, emulated_uint64_t>(Vec2A);
		morton::code<false, smallBits_2, 2> morton_small_2B = createMortonFromU64Vec<false, smallBits_2, 2>(Vec2B);
		morton::code<false, mediumBits_2, 2> morton_medium_2B = createMortonFromU64Vec<false, mediumBits_2, 2>(Vec2B);
		morton::code<false, fullBits_2, 2> morton_full_2B = createMortonFromU64Vec<false, fullBits_2, 2>(Vec2B);
		morton::code<false, fullBits_2, 2, emulated_uint64_t> morton_emulated_2B = createMortonFromU64Vec<false, fullBits_2, 2, emulated_uint64_t>(Vec2B);
	
		morton::code<false, smallBits_3, 3> morton_small_3A = createMortonFromU64Vec<false, smallBits_3, 3>(Vec3A);
		morton::code<false, mediumBits_3, 3> morton_medium_3A = createMortonFromU64Vec<false, mediumBits_3, 3>(Vec3A);
		morton::code<false, fullBits_3, 3> morton_full_3A = createMortonFromU64Vec<false, fullBits_3, 3>(Vec3A);
		morton::code<false, fullBits_3, 3, emulated_uint64_t> morton_emulated_3A = createMortonFromU64Vec<false, fullBits_3, 3, emulated_uint64_t>(Vec3A);
		morton::code<false, smallBits_3, 3> morton_small_3B = createMortonFromU64Vec<false, smallBits_3, 3>(Vec3B);
		morton::code<false, mediumBits_3, 3> morton_medium_3B = createMortonFromU64Vec<false, mediumBits_3, 3>(Vec3B);
		morton::code<false, fullBits_3, 3> morton_full_3B = createMortonFromU64Vec<false, fullBits_3, 3>(Vec3B);
		morton::code<false, fullBits_3, 3, emulated_uint64_t> morton_emulated_3B = createMortonFromU64Vec<false, fullBits_3, 3, emulated_uint64_t>(Vec3B);
	
		morton::code<false, smallBits_4, 4> morton_small_4A = createMortonFromU64Vec<false, smallBits_4, 4>(Vec4A);
		morton::code<false, mediumBits_4, 4> morton_medium_4A = createMortonFromU64Vec<false, mediumBits_4, 4>(Vec4A);
		morton::code<false, fullBits_4, 4> morton_full_4A = createMortonFromU64Vec<false, fullBits_4, 4>(Vec4A);
		morton::code<false, fullBits_4, 4, emulated_uint64_t> morton_emulated_4A = createMortonFromU64Vec<false, fullBits_4, 4, emulated_uint64_t>(Vec4A);
		morton::code<false, smallBits_4, 4> morton_small_4B = createMortonFromU64Vec<false, smallBits_4, 4>(Vec4B);
		morton::code<false, mediumBits_4, 4> morton_medium_4B = createMortonFromU64Vec<false, mediumBits_4, 4>(Vec4B);
		morton::code<false, fullBits_4, 4> morton_full_4B = createMortonFromU64Vec<false, fullBits_4, 4>(Vec4B);
		morton::code<false, fullBits_4, 4, emulated_uint64_t> morton_emulated_4B = createMortonFromU64Vec<false, fullBits_4, 4, emulated_uint64_t>(Vec4B);
	
		morton::code<true, smallBits_2, 2> morton_small_2_signed = createMortonFromU64Vec<true, smallBits_2, 2>(Vec2A);
		morton::code<true, mediumBits_2, 2> morton_medium_2_signed = createMortonFromU64Vec<true, mediumBits_2, 2>(Vec2A);
		morton::code<true, fullBits_2, 2> morton_full_2_signed = createMortonFromU64Vec<true, fullBits_2, 2>(Vec2A);
		morton::code<true, fullBits_2, 2, emulated_uint64_t> morton_emulated_2_signed = createMortonFromU64Vec<true, fullBits_2, 2, emulated_uint64_t>(Vec2A);
	
		morton::code<true, smallBits_3, 3> morton_small_3_signed = createMortonFromU64Vec<true, smallBits_3, 3>(Vec3A);
		morton::code<true, mediumBits_3, 3> morton_medium_3_signed = createMortonFromU64Vec<true, mediumBits_3, 3>(Vec3A);
		morton::code<true, fullBits_3, 3> morton_full_3_signed = createMortonFromU64Vec<true, fullBits_3, 3>(Vec3A);
		morton::code<true, fullBits_3, 3, emulated_uint64_t> morton_emulated_3_signed = createMortonFromU64Vec<true, fullBits_3, 3, emulated_uint64_t>(Vec3A);
	
		morton::code<true, smallBits_4, 4> morton_small_4_signed = createMortonFromU64Vec<true, smallBits_4, 4>(Vec4A);
		morton::code<true, mediumBits_4, 4> morton_medium_4_signed = createMortonFromU64Vec<true, mediumBits_4, 4>(Vec4A);
		morton::code<true, fullBits_4, 4> morton_full_4_signed = createMortonFromU64Vec<true, fullBits_4, 4>(Vec4A);
		morton::code<true, fullBits_4, 4, emulated_uint64_t> morton_emulated_4_signed = createMortonFromU64Vec<true, fullBits_4, 4, emulated_uint64_t>(Vec4A);

    	// Some test and operation is moved to testCommon2.hlsl due to dxc bug that cause compilation failure. Uncomment when the bug is fixed.
		// Plus
		output.mortonPlus_small_2 = morton_small_2A + morton_small_2B;
		output.mortonPlus_medium_2 = morton_medium_2A + morton_medium_2B;
		output.mortonPlus_full_2 = morton_full_2A + morton_full_2B;
		output.mortonPlus_emulated_2 = morton_emulated_2A + morton_emulated_2B;

		output.mortonPlus_small_3 = morton_small_3A + morton_small_3B;
		output.mortonPlus_medium_3 = morton_medium_3A + morton_medium_3B;
		output.mortonPlus_full_3 = morton_full_3A + morton_full_3B;
		output.mortonPlus_emulated_3 = morton_emulated_3A + morton_emulated_3B;

		output.mortonPlus_small_4 = morton_small_4A + morton_small_4B;
		output.mortonPlus_medium_4 = morton_medium_4A + morton_medium_4B;
		output.mortonPlus_full_4 = morton_full_4A + morton_full_4B;
		output.mortonPlus_emulated_4 = morton_emulated_4A + morton_emulated_4B;

		// Minus
		output.mortonMinus_small_2 = morton_small_2A - morton_small_2B;
		output.mortonMinus_medium_2 = morton_medium_2A - morton_medium_2B;
		output.mortonMinus_full_2 = morton_full_2A - morton_full_2B;
		output.mortonMinus_emulated_2 = morton_emulated_2A - morton_emulated_2B;

		output.mortonMinus_small_3 = morton_small_3A - morton_small_3B;
		output.mortonMinus_medium_3 = morton_medium_3A - morton_medium_3B;
		output.mortonMinus_full_3 = morton_full_3A - morton_full_3B;
		output.mortonMinus_emulated_3 = morton_emulated_3A - morton_emulated_3B;

		output.mortonMinus_small_4 = morton_small_4A - morton_small_4B;
		output.mortonMinus_medium_4 = morton_medium_4A - morton_medium_4B;
		output.mortonMinus_full_4 = morton_full_4A - morton_full_4B;
		output.mortonMinus_emulated_4 = morton_emulated_4A - morton_emulated_4B;

		// Coordinate-wise equality
		output.mortonEqual_small_2 = uint32_t2(morton_small_2A.equal<false>(Vec2BSmall));
		output.mortonEqual_medium_2 = uint32_t2(morton_medium_2A.equal<false>(Vec2BMedium));
		output.mortonEqual_full_2 = uint32_t2(morton_full_2A.equal<false>(Vec2BFull));
		output.mortonEqual_emulated_2 = uint32_t2(morton_emulated_2A.equal<false>(Vec2BFull));

		output.mortonEqual_small_3 = uint32_t3(morton_small_3A.equal<false>(Vec3BSmall));
		output.mortonEqual_medium_3 = uint32_t3(morton_medium_3A.equal<false>(Vec3BMedium));
		output.mortonEqual_full_3 = uint32_t3(morton_full_3A.equal<false>(Vec3BFull));
		output.mortonEqual_emulated_3 = uint32_t3(morton_emulated_3A.equal<false>(Vec3BFull));

		output.mortonEqual_small_4 = uint32_t4(morton_small_4A.equal<false>(Vec4BSmall));
		output.mortonEqual_medium_4 = uint32_t4(morton_medium_4A.equal<false>(Vec4BMedium));
		output.mortonEqual_full_4 = uint32_t4(morton_full_4A.equal<false>(Vec4BFull));
    	output.mortonEqual_emulated_4 = uint32_t4(morton_emulated_4A.equal<false>(Vec4BFull));

		// Coordinate-wise unsigned inequality (just testing with less)
		output.mortonUnsignedLess_small_2 = uint32_t2(morton_small_2A.lessThan<false>(Vec2BSmall));
		output.mortonUnsignedLess_medium_2 = uint32_t2(morton_medium_2A.lessThan<false>(Vec2BMedium));
		output.mortonUnsignedLess_full_2 = uint32_t2(morton_full_2A.lessThan<false>(Vec2BFull));
		output.mortonUnsignedLess_emulated_2 = uint32_t2(morton_emulated_2A.lessThan<false>(Vec2BFull));

		output.mortonUnsignedLess_small_3 = uint32_t3(morton_small_3A.lessThan<false>(Vec3BSmall));
		output.mortonUnsignedLess_medium_3 = uint32_t3(morton_medium_3A.lessThan<false>(Vec3BMedium));
		output.mortonUnsignedLess_full_3 = uint32_t3(morton_full_3A.lessThan<false>(Vec3BFull));
		output.mortonUnsignedLess_emulated_3 = uint32_t3(morton_emulated_3A.lessThan<false>(Vec3BFull));

		output.mortonUnsignedLess_small_4 = uint32_t4(morton_small_4A.lessThan<false>(Vec4BSmall));
		output.mortonUnsignedLess_medium_4 = uint32_t4(morton_medium_4A.lessThan<false>(Vec4BMedium));
		output.mortonUnsignedLess_full_4 = uint32_t4(morton_full_4A.lessThan<false>(Vec4BFull));
		// output.mortonUnsignedLess_emulated_4 = uint32_t4(morton_emulated_4A.lessThan<false>(Vec4BFull));

		// Coordinate-wise signed inequality
		output.mortonSignedLess_small_2 = uint32_t2(morton_small_2_signed.lessThan<false>(Vec2BSignedSmall));
		output.mortonSignedLess_medium_2 = uint32_t2(morton_medium_2_signed.lessThan<false>(Vec2BSignedMedium));
		output.mortonSignedLess_full_2 = uint32_t2(morton_full_2_signed.lessThan<false>(Vec2BSignedFull));
		// output.mortonSignedLess_emulated_2 = uint32_t2(morton_emulated_2_signed.lessThan<false>(Vec2BSignedFull)); 

		output.mortonSignedLess_small_3 = uint32_t3(morton_small_3_signed.lessThan<false>(Vec3BSignedSmall));
		output.mortonSignedLess_medium_3 = uint32_t3(morton_medium_3_signed.lessThan<false>(Vec3BSignedMedium));
		output.mortonSignedLess_full_3 = uint32_t3(morton_full_3_signed.lessThan<false>(Vec3BSignedFull));
		// output.mortonSignedLess_emulated_3 = uint32_t3(morton_emulated_3_signed.lessThan<false>(Vec3BSignedFull)); 

		output.mortonSignedLess_small_4 = uint32_t4(morton_small_4_signed.lessThan<false>(Vec4BSignedSmall));
		output.mortonSignedLess_medium_4 = uint32_t4(morton_medium_4_signed.lessThan<false>(Vec4BSignedMedium));
		output.mortonSignedLess_full_4 = uint32_t4(morton_full_4_signed.lessThan<false>(Vec4BSignedFull));
		// output.mortonSignedLess_emulated_4 = uint32_t4(morton_emulated_4_signed.lessThan<false>(Vec4BSignedFull)); 

		// Cast to uint16_t which is what left shift for Mortons expect
		uint16_t castedShift = uint16_t(input.shift);
		// Each left shift clamps to correct bits so the result kinda makes sense
		// Left-shift
		left_shift_operator<morton::code<false, smallBits_2, 2> > leftShiftSmall2;
		output.mortonLeftShift_small_2 = leftShiftSmall2(morton_small_2A, castedShift % smallBits_2);
		left_shift_operator<morton::code<false, mediumBits_2, 2> > leftShiftMedium2;
		output.mortonLeftShift_medium_2 = leftShiftMedium2(morton_medium_2A, castedShift % mediumBits_2);
		left_shift_operator<morton::code<false, fullBits_2, 2> > leftShiftFull2;
		output.mortonLeftShift_full_2 = leftShiftFull2(morton_full_2A, castedShift % fullBits_2);
		left_shift_operator<morton::code<false, fullBits_2, 2, emulated_uint64_t> > leftShiftEmulated2;
		output.mortonLeftShift_emulated_2 = leftShiftEmulated2(morton_emulated_2A, castedShift % fullBits_2);

		left_shift_operator<morton::code<false, smallBits_3, 3> > leftShiftSmall3;
		output.mortonLeftShift_small_3 = leftShiftSmall3(morton_small_3A, castedShift % smallBits_3);
		left_shift_operator<morton::code<false, mediumBits_3, 3> > leftShiftMedium3;
		output.mortonLeftShift_medium_3 = leftShiftMedium3(morton_medium_3A, castedShift % mediumBits_3);
		left_shift_operator<morton::code<false, fullBits_3, 3> > leftShiftFull3;
		output.mortonLeftShift_full_3 = leftShiftFull3(morton_full_3A, castedShift % fullBits_3);
		left_shift_operator<morton::code<false, fullBits_3, 3, emulated_uint64_t> > leftShiftEmulated3;
		output.mortonLeftShift_emulated_3 = leftShiftEmulated3(morton_emulated_3A, castedShift % fullBits_3);
	
		left_shift_operator<morton::code<false, smallBits_4, 4> > leftShiftSmall4;
		output.mortonLeftShift_small_4 = leftShiftSmall4(morton_small_4A, castedShift % smallBits_4);
		left_shift_operator<morton::code<false, mediumBits_4, 4> > leftShiftMedium4;
		output.mortonLeftShift_medium_4 = leftShiftMedium4(morton_medium_4A, castedShift % mediumBits_4);
		left_shift_operator<morton::code<false, fullBits_4, 4> > leftShiftFull4;
		output.mortonLeftShift_full_4 = leftShiftFull4(morton_full_4A, castedShift % fullBits_4);
		left_shift_operator<morton::code<false, fullBits_4, 4, emulated_uint64_t> > leftShiftEmulated4;
		output.mortonLeftShift_emulated_4 = leftShiftEmulated4(morton_emulated_4A, castedShift % fullBits_4);
		
		// Unsigned right-shift
		arithmetic_right_shift_operator<morton::code<false, smallBits_2, 2> > rightShiftSmall2;
		output.mortonUnsignedRightShift_small_2 = rightShiftSmall2(morton_small_2A, castedShift % smallBits_2);
		arithmetic_right_shift_operator<morton::code<false, mediumBits_2, 2> > rightShiftMedium2;
		output.mortonUnsignedRightShift_medium_2 = rightShiftMedium2(morton_medium_2A, castedShift % mediumBits_2);
		arithmetic_right_shift_operator<morton::code<false, fullBits_2, 2> > rightShiftFull2;
		output.mortonUnsignedRightShift_full_2 = rightShiftFull2(morton_full_2A, castedShift % fullBits_2);
		arithmetic_right_shift_operator<morton::code<false, fullBits_2, 2, emulated_uint64_t> > rightShiftEmulated2;
		output.mortonUnsignedRightShift_emulated_2 = rightShiftEmulated2(morton_emulated_2A, castedShift % fullBits_2);
		
		arithmetic_right_shift_operator<morton::code<false, smallBits_3, 3> > rightShiftSmall3;
		output.mortonUnsignedRightShift_small_3 = rightShiftSmall3(morton_small_3A, castedShift % smallBits_3);
		arithmetic_right_shift_operator<morton::code<false, mediumBits_3, 3> > rightShiftMedium3;
		output.mortonUnsignedRightShift_medium_3 = rightShiftMedium3(morton_medium_3A, castedShift % mediumBits_3);
		arithmetic_right_shift_operator<morton::code<false, fullBits_3, 3> > rightShiftFull3;
		output.mortonUnsignedRightShift_full_3 = rightShiftFull3(morton_full_3A, castedShift % fullBits_3);
		arithmetic_right_shift_operator<morton::code<false, fullBits_3, 3, emulated_uint64_t> > rightShiftEmulated3;
		output.mortonUnsignedRightShift_emulated_3 = rightShiftEmulated3(morton_emulated_3A, castedShift % fullBits_3);
		
		arithmetic_right_shift_operator<morton::code<false, smallBits_4, 4> > rightShiftSmall4;
		output.mortonUnsignedRightShift_small_4 = rightShiftSmall4(morton_small_4A, castedShift % smallBits_4);
		arithmetic_right_shift_operator<morton::code<false, mediumBits_4, 4> > rightShiftMedium4;
		output.mortonUnsignedRightShift_medium_4 = rightShiftMedium4(morton_medium_4A, castedShift % mediumBits_4);
		arithmetic_right_shift_operator<morton::code<false, fullBits_4, 4> > rightShiftFull4;
		output.mortonUnsignedRightShift_full_4 = rightShiftFull4(morton_full_4A, castedShift % fullBits_4);
		arithmetic_right_shift_operator<morton::code<false, fullBits_4, 4, emulated_uint64_t> > rightShiftEmulated4;
		output.mortonUnsignedRightShift_emulated_4 = rightShiftEmulated4(morton_emulated_4A, castedShift % fullBits_4);
		
		// Signed right-shift
		arithmetic_right_shift_operator<morton::code<true, smallBits_2, 2> > rightShiftSignedSmall2;
		output.mortonSignedRightShift_small_2 = rightShiftSignedSmall2(morton_small_2_signed, castedShift % smallBits_2);
		arithmetic_right_shift_operator<morton::code<true, mediumBits_2, 2> > rightShiftSignedMedium2;
		output.mortonSignedRightShift_medium_2 = rightShiftSignedMedium2(morton_medium_2_signed, castedShift % mediumBits_2);
		arithmetic_right_shift_operator<morton::code<true, fullBits_2, 2> > rightShiftSignedFull2;
		output.mortonSignedRightShift_full_2 = rightShiftSignedFull2(morton_full_2_signed, castedShift % fullBits_2);
		// arithmetic_right_shift_operator<morton::code<true, fullBits_2, 2, emulated_uint64_t> > rightShiftSignedEmulated2;
		// output.mortonSignedRightShift_emulated_2 = rightShiftSignedEmulated2(morton_emulated_2_signed, castedShift % fullBits_2); 
		
		arithmetic_right_shift_operator<morton::code<true, smallBits_3, 3> > rightShiftSignedSmall3;
		output.mortonSignedRightShift_small_3 = rightShiftSignedSmall3(morton_small_3_signed, castedShift % smallBits_3);
		arithmetic_right_shift_operator<morton::code<true, mediumBits_3, 3> > rightShiftSignedMedium3;
		output.mortonSignedRightShift_medium_3 = rightShiftSignedMedium3(morton_medium_3_signed, castedShift % mediumBits_3);
		arithmetic_right_shift_operator<morton::code<true, fullBits_3, 3> > rightShiftSignedFull3;
		output.mortonSignedRightShift_full_3 = rightShiftSignedFull3(morton_full_3_signed, castedShift % fullBits_3);
		// arithmetic_right_shift_operator<morton::code<true, fullBits_3, 3, emulated_uint64_t> > rightShiftSignedEmulated3;
		// output.mortonSignedRightShift_emulated_3 = rightShiftSignedEmulated3(morton_emulated_3_signed, castedShift % fullBits_3); 
		
		arithmetic_right_shift_operator<morton::code<true, smallBits_4, 4> > rightShiftSignedSmall4;
		output.mortonSignedRightShift_small_4 = rightShiftSignedSmall4(morton_small_4_signed, castedShift % smallBits_4);
		arithmetic_right_shift_operator<morton::code<true, mediumBits_4, 4> > rightShiftSignedMedium4;
		output.mortonSignedRightShift_medium_4 = rightShiftSignedMedium4(morton_medium_4_signed, castedShift % mediumBits_4);
		arithmetic_right_shift_operator<morton::code<true, fullBits_4, 4> > rightShiftSignedFull4;
		output.mortonSignedRightShift_full_4 = rightShiftSignedFull4(morton_full_4_signed, castedShift % fullBits_4);
		// arithmetic_right_shift_operator<morton::code<true, fullBits_4, 4, emulated_uint64_t> > rightShiftSignedEmulated4;
		// output.mortonSignedRightShift_emulated_4 = rightShiftSignedEmulated4(morton_emulated_4_signed, castedShift % fullBits_4); 

}
