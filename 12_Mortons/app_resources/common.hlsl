//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_12_MORTON_COMMON_INCLUDED_
#define _NBL_EXAMPLES_TESTS_12_MORTON_COMMON_INCLUDED_

#include <boost/preprocessor.hpp>

#include <nbl/builtin/hlsl/morton.hlsl>

NBL_CONSTEXPR uint16_t smallBits_2 = 8;
NBL_CONSTEXPR uint16_t mediumBits_2 = 16;
NBL_CONSTEXPR uint16_t fullBits_2 = 32;
NBL_CONSTEXPR uint16_t smallBits_3 = 5;
NBL_CONSTEXPR uint16_t mediumBits_3 = 10;
NBL_CONSTEXPR uint16_t fullBits_3 = 21;
NBL_CONSTEXPR uint16_t smallBits_4 = 4;
NBL_CONSTEXPR uint16_t mediumBits_4 = 8;
NBL_CONSTEXPR uint16_t fullBits_4 = 16;

#ifndef __HLSL_VERSION

constexpr uint64_t smallBitsMask_2 = (uint64_t(1) << smallBits_2) - 1;
constexpr uint64_t mediumBitsMask_2 = (uint64_t(1) << mediumBits_2) - 1;
constexpr uint64_t fullBitsMask_2 = (uint64_t(1) << fullBits_2) - 1;

constexpr uint64_t smallBitsMask_3 = (uint64_t(1) << smallBits_3) - 1;
constexpr uint64_t mediumBitsMask_3 = (uint64_t(1) << mediumBits_3) - 1;
constexpr uint64_t fullBitsMask_3 = (uint64_t(1) << fullBits_3) - 1;

constexpr uint64_t smallBitsMask_4 = (uint64_t(1) << smallBits_4) - 1;
constexpr uint64_t mediumBitsMask_4 = (uint64_t(1) << mediumBits_4) - 1;
constexpr uint64_t fullBitsMask_4 = (uint64_t(1) << fullBits_4) - 1;

#endif

using namespace nbl::hlsl;
struct InputTestValues
{
	// Both tests
	uint32_t shift;

	// Emulated int tests
	uint64_t generatedA;
	uint64_t generatedB;
	
	// Morton tests
	uint64_t coordX;
	uint64_t coordY;
	uint64_t coordZ;
	uint64_t coordW;
};

struct TestValues
{
	// Emulated int tests
	emulated_uint64_t emulatedAnd;
	emulated_uint64_t emulatedOr;
	emulated_uint64_t emulatedXor;
	emulated_uint64_t emulatedNot;
	emulated_uint64_t emulatedPlus;
	emulated_uint64_t emulatedMinus;
	// These are bools but stored as uint because you can't store bools, causes a SPIR-V issue
	uint32_t emulatedLess;
	uint32_t emulatedLessEqual;
	uint32_t emulatedGreater;
	uint32_t emulatedGreaterEqual;
	emulated_uint64_t emulatedLeftShifted;
	emulated_uint64_t emulatedUnsignedRightShifted;
	emulated_int64_t  emulatedSignedRightShifted;

	// Morton tests - for each dimension let's do one small, medium and full-szied (max bits possible) test to cover representation with
	// 16, 32 and 64-bit types. Could make it more exhaustive with macros (test all possible bitwidths)
	// For emulated mortons, we store only the emulated uint64 representing it, because DXC complains about bitcasts otherwise

	// Plus
	morton::code<false, smallBits_2, 2>					  mortonPlus_small_2;
	morton::code<false, mediumBits_2, 2>				  mortonPlus_medium_2;
	morton::code<false, fullBits_2, 2>					  mortonPlus_full_2;
	morton::code<false, fullBits_2, 2, emulated_uint64_t> mortonPlus_emulated_2;
	
	morton::code<false, smallBits_3, 3>					  mortonPlus_small_3;
	morton::code<false, mediumBits_3, 3>				  mortonPlus_medium_3;
	morton::code<false, fullBits_3, 3>					  mortonPlus_full_3;
	morton::code<false, fullBits_3, 3, emulated_uint64_t> mortonPlus_emulated_3;
	
	morton::code<false, smallBits_4, 4>					  mortonPlus_small_4;
	morton::code<false, mediumBits_4, 4>				  mortonPlus_medium_4;
	morton::code<false, fullBits_4, 4>					  mortonPlus_full_4;
	morton::code<false, fullBits_4, 4, emulated_uint64_t> mortonPlus_emulated_4;
	
	// Minus
	morton::code<false, smallBits_2, 2>					  mortonMinus_small_2;
	morton::code<false, mediumBits_2, 2>				  mortonMinus_medium_2;
	morton::code<false, fullBits_2, 2>					  mortonMinus_full_2;
	morton::code<false, fullBits_2, 2, emulated_uint64_t> mortonMinus_emulated_2;
	
	morton::code<false, smallBits_3, 3>					  mortonMinus_small_3;
	morton::code<false, mediumBits_3, 3>				  mortonMinus_medium_3;
	morton::code<false, fullBits_3, 3>					  mortonMinus_full_3;
	morton::code<false, fullBits_3, 3, emulated_uint64_t> mortonMinus_emulated_3;
	
	morton::code<false, smallBits_4, 4>					  mortonMinus_small_4;
	morton::code<false, mediumBits_4, 4>				  mortonMinus_medium_4;
	morton::code<false, fullBits_4, 4>					  mortonMinus_full_4;
	morton::code<false, fullBits_4, 4, emulated_uint64_t> mortonMinus_emulated_4;

	// Coordinate-wise equality (these are bools)
	uint32_t2 mortonEqual_small_2;
	uint32_t2 mortonEqual_medium_2;
	uint32_t2 mortonEqual_full_2;
	uint32_t2 mortonEqual_emulated_2;

	uint32_t3 mortonEqual_small_3;
	uint32_t3 mortonEqual_medium_3;
	uint32_t3 mortonEqual_full_3;
	uint32_t3 mortonEqual_emulated_3;

	uint32_t4 mortonEqual_small_4;
	uint32_t4 mortonEqual_medium_4;
	uint32_t4 mortonEqual_full_4;
	uint32_t4 mortonEqual_emulated_4;

	// Coordinate-wise unsigned inequality (just testing with less, again these are bools)
	uint32_t2 mortonUnsignedLess_small_2;
	uint32_t2 mortonUnsignedLess_medium_2;
	uint32_t2 mortonUnsignedLess_full_2;
	uint32_t2 mortonUnsignedLess_emulated_2;

	uint32_t3 mortonUnsignedLess_small_3;
	uint32_t3 mortonUnsignedLess_medium_3;
	uint32_t3 mortonUnsignedLess_full_3;
	uint32_t3 mortonUnsignedLess_emulated_3;

	uint32_t4 mortonUnsignedLess_small_4;
	uint32_t4 mortonUnsignedLess_medium_4;
	uint32_t4 mortonUnsignedLess_full_4;
	uint32_t4 mortonUnsignedLess_emulated_4;

	// Coordinate-wise signed inequality (bools)
	uint32_t2 mortonSignedLess_small_2;
	uint32_t2 mortonSignedLess_medium_2;
	uint32_t2 mortonSignedLess_full_2;
	uint32_t2 mortonSignedLess_emulated_2;

	uint32_t3 mortonSignedLess_small_3;
	uint32_t3 mortonSignedLess_medium_3;
	uint32_t3 mortonSignedLess_full_3;
	uint32_t3 mortonSignedLess_emulated_3;

	uint32_t4 mortonSignedLess_small_4;
	uint32_t4 mortonSignedLess_medium_4;
	uint32_t4 mortonSignedLess_full_4;
	uint32_t4 mortonSignedLess_emulated_4;

	// Left-shift
	morton::code<false, smallBits_2, 2>					  mortonLeftShift_small_2;
	morton::code<false, mediumBits_2, 2>				  mortonLeftShift_medium_2;
	morton::code<false, fullBits_2, 2>					  mortonLeftShift_full_2;
	morton::code<false, fullBits_2, 2, emulated_uint64_t> mortonLeftShift_emulated_2;

	morton::code<false, smallBits_3, 3>					  mortonLeftShift_small_3;
	morton::code<false, mediumBits_3, 3>				  mortonLeftShift_medium_3;
	morton::code<false, fullBits_3, 3>					  mortonLeftShift_full_3;
	morton::code<false, fullBits_3, 3, emulated_uint64_t> mortonLeftShift_emulated_3;

	morton::code<false, smallBits_4, 4>					  mortonLeftShift_small_4;
	morton::code<false, mediumBits_4, 4>				  mortonLeftShift_medium_4;
	morton::code<false, fullBits_4, 4>					  mortonLeftShift_full_4;
	morton::code<false, fullBits_4, 4, emulated_uint64_t> mortonLeftShift_emulated_4;

	// Unsigned right-shift
	morton::code<false, smallBits_2, 2>					  mortonUnsignedRightShift_small_2;
	morton::code<false, mediumBits_2, 2>				  mortonUnsignedRightShift_medium_2;
	morton::code<false, fullBits_2, 2>					  mortonUnsignedRightShift_full_2;
	morton::code<false, fullBits_2, 2, emulated_uint64_t> mortonUnsignedRightShift_emulated_2;

	morton::code<false, smallBits_3, 3>					  mortonUnsignedRightShift_small_3;
	morton::code<false, mediumBits_3, 3>				  mortonUnsignedRightShift_medium_3;
	morton::code<false, fullBits_3, 3>					  mortonUnsignedRightShift_full_3;
	morton::code<false, fullBits_3, 3, emulated_uint64_t> mortonUnsignedRightShift_emulated_3;

	morton::code<false, smallBits_4, 4>					  mortonUnsignedRightShift_small_4;
	morton::code<false, mediumBits_4, 4>				  mortonUnsignedRightShift_medium_4;
	morton::code<false, fullBits_4, 4>					  mortonUnsignedRightShift_full_4;
	morton::code<false, fullBits_4, 4, emulated_uint64_t> mortonUnsignedRightShift_emulated_4;

	// Signed right-shift
	morton::code<true, smallBits_2, 2>					  mortonSignedRightShift_small_2;
	morton::code<true, mediumBits_2, 2>					  mortonSignedRightShift_medium_2;
	morton::code<true, fullBits_2, 2>					  mortonSignedRightShift_full_2;
	morton::code<true, fullBits_2, 2, emulated_uint64_t>  mortonSignedRightShift_emulated_2;

	morton::code<true, smallBits_3, 3>					  mortonSignedRightShift_small_3;
	morton::code<true, mediumBits_3, 3>					  mortonSignedRightShift_medium_3;
	morton::code<true, fullBits_3, 3>					  mortonSignedRightShift_full_3;
	morton::code<true, fullBits_3, 3, emulated_uint64_t>  mortonSignedRightShift_emulated_3;

	morton::code<true, smallBits_4, 4>					  mortonSignedRightShift_small_4;
	morton::code<true, mediumBits_4, 4>					  mortonSignedRightShift_medium_4;
	morton::code<true, fullBits_4, 4>					  mortonSignedRightShift_full_4;
	morton::code<true, fullBits_4, 4, emulated_uint64_t>  mortonSignedRightShift_emulated_4;

	/*
	void fillSecondTestValues(NBL_CONST_REF_ARG(InputTestValues) input)
	{
		uint64_t2 Vec2A = { input.coordX, input.coordY };
		uint64_t2 Vec2B = { input.coordZ, input.coordW };

		uint64_t3 Vec3A = { input.coordX, input.coordY, input.coordZ };
		uint64_t3 Vec3B = { input.coordY, input.coordZ, input.coordW };

		uint64_t4 Vec4A = { input.coordX, input.coordY, input.coordZ, input.coordW };
		uint64_t4 Vec4B = { input.coordY, input.coordZ, input.coordW, input.coordX };

		int64_t2 Vec2ASigned = int64_t2(Vec2A);
		int64_t2 Vec2BSigned = int64_t2(Vec2B);

		int64_t3 Vec3ASigned = int64_t3(Vec3A);
		int64_t3 Vec3BSigned = int64_t3(Vec3B);

		int64_t4 Vec4ASigned = int64_t4(Vec4A);
		int64_t4 Vec4BSigned = int64_t4(Vec4B);

		morton::code<false, fullBits_4, 4, emulated_uint64_t> morton_emulated_4A = morton::code<false, fullBits_4, 4, emulated_uint64_t>::create(Vec4A);
		morton::code<true, fullBits_2, 2, emulated_uint64_t> morton_emulated_2_signed = morton::code<true, fullBits_2, 2, emulated_uint64_t>::create(Vec2ASigned);
		morton::code<true, fullBits_3, 3, emulated_uint64_t> morton_emulated_3_signed = morton::code<true, fullBits_3, 3, emulated_uint64_t>::create(Vec3ASigned);
		morton::code<true, fullBits_4, 4, emulated_uint64_t> morton_emulated_4_signed = morton::code<true, fullBits_4, 4, emulated_uint64_t>::create(Vec4ASigned);

		output.mortonEqual_emulated_4 = uint32_t4(morton_emulated_4A.equal<false>(uint16_t4(Vec4B)));
		
		output.mortonUnsignedLess_emulated_4 = uint32_t4(morton_emulated_4A.lessThan<false>(uint16_t4(Vec4B)));
		
		mortonSignedLess_emulated_2 = uint32_t2(morton_emulated_2_signed.lessThan<false>(int32_t2(Vec2BSigned))); 
		mortonSignedLess_emulated_3 = uint32_t3(morton_emulated_3_signed.lessThan<false>(int32_t3(Vec3BSigned))); 
		mortonSignedLess_emulated_4 = uint32_t4(morton_emulated_4_signed.lessThan<false>(int16_t4(Vec4BSigned))); 

		uint16_t castedShift = uint16_t(input.shift);

		arithmetic_right_shift_operator<morton::code<true, fullBits_2, 2, emulated_uint64_t> > rightShiftSignedEmulated2;
		mortonSignedRightShift_emulated_2 = rightShiftSignedEmulated2(morton_emulated_2_signed, castedShift); 
		arithmetic_right_shift_operator<morton::code<true, fullBits_3, 3, emulated_uint64_t> > rightShiftSignedEmulated3;
		mortonSignedRightShift_emulated_3 = rightShiftSignedEmulated3(morton_emulated_3_signed, castedShift); 
		arithmetic_right_shift_operator<morton::code<true, fullBits_4, 4, emulated_uint64_t> > rightShiftSignedEmulated4;
		mortonSignedRightShift_emulated_4 = rightShiftSignedEmulated4(morton_emulated_4_signed, castedShift); 
	}
	*/
};

#endif
