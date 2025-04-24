//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_12_MORTON_COMMON_INCLUDED_
#define _NBL_EXAMPLES_TESTS_12_MORTON_COMMON_INCLUDED_

// because DXC doesn't properly support `_Static_assert`
// TODO: add a message, and move to macros.h or cpp_compat
#define STATIC_ASSERT(...) { nbl::hlsl::conditional<__VA_ARGS__, int, void>::type a = 0; }

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

	void fillTestValues(NBL_CONST_REF_ARG(InputTestValues) input)
	{
		emulated_uint64_t emulatedA = _static_cast<emulated_uint64_t>(input.generatedA);
		emulated_uint64_t emulatedB = _static_cast<emulated_uint64_t>(input.generatedB);

		// Emulated int tests
		emulatedAnd = emulatedA & emulatedB;
		emulatedOr = emulatedA | emulatedB;
		emulatedXor = emulatedA ^ emulatedB;
		emulatedNot = emulatedA.operator~();
		emulatedPlus = emulatedA + emulatedB;
		emulatedMinus = emulatedA - emulatedB;
		emulatedLess = uint32_t(emulatedA < emulatedB);
		emulatedLessEqual = uint32_t(emulatedA <= emulatedB);
		emulatedGreater = uint32_t(emulatedA > emulatedB);
		emulatedGreaterEqual = uint32_t(emulatedA >= emulatedB);

		left_shift_operator<emulated_uint64_t> leftShift;
		emulatedLeftShifted = leftShift(emulatedA, input.shift);

		arithmetic_right_shift_operator<emulated_uint64_t> unsignedRightShift;
		emulatedUnsignedRightShifted = unsignedRightShift(emulatedA, input.shift);

		arithmetic_right_shift_operator<emulated_int64_t> signedRightShift;
		emulatedSignedRightShifted = signedRightShift(_static_cast<emulated_int64_t>(emulatedA), input.shift);

		// Morton tests
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

		morton::code<false, smallBits_2, 2> morton_small_2A = morton::code<false, smallBits_2, 2>::create(Vec2A);
		morton::code<false, mediumBits_2, 2> morton_medium_2A = morton::code<false, mediumBits_2, 2>::create(Vec2A);
		morton::code<false, fullBits_2, 2> morton_full_2A = morton::code<false, fullBits_2, 2>::create(Vec2A);
		morton::code<false, fullBits_2, 2, emulated_uint64_t> morton_emulated_2A = morton::code<false, fullBits_2, 2, emulated_uint64_t>::create(Vec2A);
		morton::code<false, smallBits_2, 2> morton_small_2B = morton::code<false, smallBits_2, 2>::create(Vec2B);
		morton::code<false, mediumBits_2, 2> morton_medium_2B = morton::code<false, mediumBits_2, 2>::create(Vec2B);
		morton::code<false, fullBits_2, 2> morton_full_2B = morton::code<false, fullBits_2, 2>::create(Vec2B);
		morton::code<false, fullBits_2, 2, emulated_uint64_t> morton_emulated_2B = morton::code<false, fullBits_2, 2, emulated_uint64_t>::create(Vec2B);

		morton::code<false, smallBits_3, 3> morton_small_3A = morton::code<false, smallBits_3, 3>::create(Vec3A);
		morton::code<false, mediumBits_3, 3> morton_medium_3A = morton::code<false, mediumBits_3, 3>::create(Vec3A);
		morton::code<false, fullBits_3, 3> morton_full_3A = morton::code<false, fullBits_3, 3>::create(Vec3A);
		morton::code<false, fullBits_3, 3, emulated_uint64_t> morton_emulated_3A = morton::code<false, fullBits_3, 3, emulated_uint64_t>::create(Vec3A);
		morton::code<false, smallBits_3, 3> morton_small_3B = morton::code<false, smallBits_3, 3>::create(Vec3B);
		morton::code<false, mediumBits_3, 3> morton_medium_3B = morton::code<false, mediumBits_3, 3>::create(Vec3B);
		morton::code<false, fullBits_3, 3> morton_full_3B = morton::code<false, fullBits_3, 3>::create(Vec3B);
		morton::code<false, fullBits_3, 3, emulated_uint64_t> morton_emulated_3B = morton::code<false, fullBits_3, 3, emulated_uint64_t>::create(Vec3B);

		morton::code<false, smallBits_4, 4> morton_small_4A = morton::code<false, smallBits_4, 4>::create(Vec4A);
		morton::code<false, mediumBits_4, 4> morton_medium_4A = morton::code<false, mediumBits_4, 4>::create(Vec4A);
		morton::code<false, fullBits_4, 4> morton_full_4A = morton::code<false, fullBits_4, 4>::create(Vec4A);
		morton::code<false, fullBits_4, 4, emulated_uint64_t> morton_emulated_4A = morton::code<false, fullBits_4, 4, emulated_uint64_t>::create(Vec4A);
		morton::code<false, smallBits_4, 4> morton_small_4B = morton::code<false, smallBits_4, 4>::create(Vec4B);
		morton::code<false, mediumBits_4, 4> morton_medium_4B = morton::code<false, mediumBits_4, 4>::create(Vec4B);
		morton::code<false, fullBits_4, 4> morton_full_4B = morton::code<false, fullBits_4, 4>::create(Vec4B);
		morton::code<false, fullBits_4, 4, emulated_uint64_t> morton_emulated_4B = morton::code<false, fullBits_4, 4, emulated_uint64_t>::create(Vec4B);

		morton::code<true, smallBits_2, 2> morton_small_2ASigned = morton::code<true, smallBits_2, 2>::create(Vec2ASigned);
		morton::code<true, mediumBits_2, 2> morton_medium_2ASigned = morton::code<true, mediumBits_2, 2>::create(Vec2ASigned);
		morton::code<true, fullBits_2, 2> morton_full_2ASigned = morton::code<true, fullBits_2, 2>::create(Vec2ASigned);
		morton::code<true, fullBits_2, 2, emulated_uint64_t> morton_emulated_2ASigned = morton::code<true, fullBits_2, 2, emulated_uint64_t>::create(Vec2ASigned);
		morton::code<true, smallBits_2, 2> morton_small_2BSigned = morton::code<true, smallBits_2, 2>::create(Vec2BSigned);
		morton::code<true, mediumBits_2, 2> morton_medium_2BSigned = morton::code<true, mediumBits_2, 2>::create(Vec2BSigned);
		morton::code<true, fullBits_2, 2> morton_full_2BSigned = morton::code<true, fullBits_2, 2>::create(Vec2BSigned);
		morton::code<true, fullBits_2, 2, emulated_uint64_t> morton_emulated_2BSigned = morton::code<true, fullBits_2, 2, emulated_uint64_t>::create(Vec2BSigned);

		morton::code<true, smallBits_3, 3> morton_small_3ASigned = morton::code<true, smallBits_3, 3>::create(Vec3ASigned);
		morton::code<true, mediumBits_3, 3> morton_medium_3ASigned = morton::code<true, mediumBits_3, 3>::create(Vec3ASigned);
		morton::code<true, fullBits_3, 3> morton_full_3ASigned = morton::code<true, fullBits_3, 3>::create(Vec3ASigned);
		morton::code<true, fullBits_3, 3, emulated_uint64_t> morton_emulated_3ASigned = morton::code<true, fullBits_3, 3, emulated_uint64_t>::create(Vec3ASigned);
		morton::code<true, smallBits_3, 3> morton_small_3BSigned = morton::code<true, smallBits_3, 3>::create(Vec3BSigned);
		morton::code<true, mediumBits_3, 3> morton_medium_3BSigned = morton::code<true, mediumBits_3, 3>::create(Vec3BSigned);
		morton::code<true, fullBits_3, 3> morton_full_3BSigned = morton::code<true, fullBits_3, 3>::create(Vec3BSigned);
		morton::code<true, fullBits_3, 3, emulated_uint64_t> morton_emulated_3BSigned = morton::code<true, fullBits_3, 3, emulated_uint64_t>::create(Vec3BSigned);

		morton::code<true, smallBits_4, 4> morton_small_4ASigned = morton::code<true, smallBits_4, 4>::create(Vec4ASigned);
		morton::code<true, mediumBits_4, 4> morton_medium_4ASigned = morton::code<true, mediumBits_4, 4>::create(Vec4ASigned);
		morton::code<true, fullBits_4, 4> morton_full_4ASigned = morton::code<true, fullBits_4, 4>::create(Vec4ASigned);
		morton::code<true, fullBits_4, 4, emulated_uint64_t> morton_emulated_4ASigned = morton::code<true, fullBits_4, 4, emulated_uint64_t>::create(Vec4ASigned);
		morton::code<true, smallBits_4, 4> morton_small_4BSigned = morton::code<true, smallBits_4, 4>::create(Vec4BSigned);
		morton::code<true, mediumBits_4, 4> morton_medium_4BSigned = morton::code<true, mediumBits_4, 4>::create(Vec4BSigned);
		morton::code<true, fullBits_4, 4> morton_full_4BSigned = morton::code<true, fullBits_4, 4>::create(Vec4BSigned);
		morton::code<true, fullBits_4, 4, emulated_uint64_t> morton_emulated_4BSigned = morton::code<true, fullBits_4, 4, emulated_uint64_t>::create(Vec4BSigned);

		/*
		left_shift_operator<portable_vector_t<emulated_uint64_t, 4> > leftShiftTemp;
		portable_vector_t<emulated_uint64_t, 4> interleaved = _static_cast<portable_vector_t<emulated_uint64_t, 4> >(uint16_t4(Vec4B)) & morton::impl::coding_mask_v<4, fullBits_4, morton::impl::CodingStages, emulated_uint64_t>;
		
		#define ENCODE_LOOP_ITERATION(I) NBL_IF_CONSTEXPR(fullBits_4 > (uint16_t(1) << I))\
        {\
            interleaved = interleaved | leftShiftTemp(interleaved, (uint16_t(1) << I) * (4 - 1));\
            interleaved = interleaved & _static_cast<emulated_uint64_t>(morton::impl::coding_mask<4, fullBits_4, I>::value);\
        }
		
		ENCODE_LOOP_ITERATION(4)
		ENCODE_LOOP_ITERATION(3)
		ENCODE_LOOP_ITERATION(2)
		ENCODE_LOOP_ITERATION(1)
		ENCODE_LOOP_ITERATION(0)

		#undef ENCODE_LOOP_ITERATION
		// After interleaving, shift each coordinate left by their index
		return leftShiftTemp(interleaved, truncate<vector<uint16_t, Dim> >(vector<uint16_t, 4>(0, 1, 2, 3)));
		
		
		array_get<portable_vector_t<emulated_uint64_t, 4>, emulated_uint64_t> getter;
		emulatedAnd = getter(interleaved, 0);
		*/
		
		// Plus
		mortonPlus_small_2 = morton_small_2A + morton_small_2B;
		mortonPlus_medium_2 = morton_medium_2A + morton_medium_2B;
		mortonPlus_full_2 = morton_full_2A + morton_full_2B;
		mortonPlus_emulated_2 = morton_emulated_2A + morton_emulated_2B;
		
		mortonPlus_small_3 = morton_small_3A + morton_small_3B;
		mortonPlus_medium_3 = morton_medium_3A + morton_medium_3B;
		mortonPlus_full_3 = morton_full_3A + morton_full_3B;
		mortonPlus_emulated_3 = morton_emulated_3A + morton_emulated_3B;

		mortonPlus_small_4 = morton_small_4A + morton_small_4B;
		mortonPlus_medium_4 = morton_medium_4A + morton_medium_4B;
		mortonPlus_full_4 = morton_full_4A + morton_full_4B;
		mortonPlus_emulated_4 = morton_emulated_4A + morton_emulated_4B;
		
		// Minus
		mortonMinus_small_2 = morton_small_2A - morton_small_2B;
		mortonMinus_medium_2 = morton_medium_2A - morton_medium_2B;
		mortonMinus_full_2 = morton_full_2A - morton_full_2B;
		mortonMinus_emulated_2 = morton_emulated_2A - morton_emulated_2B;

		mortonMinus_small_3 = morton_small_3A - morton_small_3B;
		mortonMinus_medium_3 = morton_medium_3A - morton_medium_3B;
		mortonMinus_full_3 = morton_full_3A - morton_full_3B;
		mortonMinus_emulated_3 = morton_emulated_3A - morton_emulated_3B;

		mortonMinus_small_4 = morton_small_4A - morton_small_4B;
		mortonMinus_medium_4 = morton_medium_4A - morton_medium_4B;
		mortonMinus_full_4 = morton_full_4A - morton_full_4B;
		mortonMinus_emulated_4 = morton_emulated_4A - morton_emulated_4B;

		// Coordinate-wise equality
		mortonEqual_small_2 = uint32_t2(morton_small_2A.equal<false>(uint16_t2(Vec2B)));
		mortonEqual_medium_2 = uint32_t2(morton_medium_2A.equal<false>(uint16_t2(Vec2B)));
		mortonEqual_full_2 = uint32_t2(morton_full_2A.equal<false>(uint32_t2(Vec2B)));
		mortonEqual_emulated_2 = uint32_t2(morton_emulated_2A.equal<false>(uint32_t2(Vec2B)));

		mortonEqual_small_3 = uint32_t3(morton_small_3A.equal<false>(uint16_t3(Vec3B)));
		mortonEqual_medium_3 = uint32_t3(morton_medium_3A.equal<false>(uint16_t3(Vec3B)));
		mortonEqual_full_3 = uint32_t3(morton_full_3A.equal<false>(uint32_t3(Vec3B)));
		mortonEqual_emulated_3 = uint32_t3(morton_emulated_3A.equal<false>(uint32_t3(Vec3B)));

		mortonEqual_small_4 = uint32_t4(morton_small_4A.equal<false>(uint16_t4(Vec4B)));
		mortonEqual_medium_4 = uint32_t4(morton_medium_4A.equal<false>(uint16_t4(Vec4B)));
		mortonEqual_full_4 = uint32_t4(morton_full_4A.equal<false>(uint16_t4(Vec4B)));
		mortonEqual_emulated_4 = uint32_t4(morton_emulated_4A.equal<false>(uint16_t4(Vec4B)));
		
		// Coordinate-wise unsigned inequality (just testing with less)
		mortonUnsignedLess_small_2 = uint32_t2(morton_small_2A.lessThan<false>(uint16_t2(Vec2B)));
		mortonUnsignedLess_medium_2 = uint32_t2(morton_medium_2A.lessThan<false>(uint16_t2(Vec2B)));
		mortonUnsignedLess_full_2 = uint32_t2(morton_full_2A.lessThan<false>(uint32_t2(Vec2B)));
		mortonUnsignedLess_emulated_2 = uint32_t2(morton_emulated_2A.lessThan<false>(uint32_t2(Vec2B)));
		
		mortonUnsignedLess_small_3 = uint32_t3(morton_small_3A.lessThan<false>(uint16_t3(Vec3B)));
		mortonUnsignedLess_medium_3 = uint32_t3(morton_medium_3A.lessThan<false>(uint16_t3(Vec3B)));
		mortonUnsignedLess_full_3 = uint32_t3(morton_full_3A.lessThan<false>(uint32_t3(Vec3B)));
		mortonUnsignedLess_emulated_3 = uint32_t3(morton_emulated_3A.lessThan<false>(uint32_t3(Vec3B)));
		
		mortonUnsignedLess_small_4 = uint32_t4(morton_small_4A.lessThan<false>(uint16_t4(Vec4B)));
		mortonUnsignedLess_medium_4 = uint32_t4(morton_medium_4A.lessThan<false>(uint16_t4(Vec4B)));
		mortonUnsignedLess_full_4 = uint32_t4(morton_full_4A.lessThan<false>(uint16_t4(Vec4B)));
		mortonUnsignedLess_emulated_4 = uint32_t4(morton_emulated_4A.lessThan<false>(uint16_t4(Vec4B)));
		
		// Coordinate-wise signed inequality
		mortonSignedLess_small_2 = uint32_t2(morton_small_2ASigned.lessThan<false>(int16_t2(Vec2BSigned)));
		mortonSignedLess_medium_2 = uint32_t2(morton_medium_2ASigned.lessThan<false>(int16_t2(Vec2BSigned)));
		mortonSignedLess_full_2 = uint32_t2(morton_full_2ASigned.lessThan<false>(int32_t2(Vec2BSigned)));
		//mortonSignedLess_emulated_2 = uint32_t2(morton_emulated_2ASigned.lessThan<false>(int32_t2(Vec2BSigned)));

		mortonSignedLess_small_3 = uint32_t3(morton_small_3ASigned.lessThan<false>(int16_t3(Vec3BSigned)));
		mortonSignedLess_medium_3 = uint32_t3(morton_medium_3ASigned.lessThan<false>(int16_t3(Vec3BSigned)));
		mortonSignedLess_full_3 = uint32_t3(morton_full_3ASigned.lessThan<false>(int32_t3(Vec3BSigned)));
		//mortonSignedLess_emulated_3 = uint32_t3(morton_emulated_3ASigned.lessThan<false>(int32_t3(Vec3BSigned)));

		mortonSignedLess_small_4 = uint32_t4(morton_small_4ASigned.lessThan<false>(int16_t4(Vec4BSigned)));
		mortonSignedLess_medium_4 = uint32_t4(morton_medium_4ASigned.lessThan<false>(int16_t4(Vec4BSigned)));
		mortonSignedLess_full_4 = uint32_t4(morton_full_4ASigned.lessThan<false>(int16_t4(Vec4BSigned)));
		//mortonSignedLess_emulated_4 = uint32_t4(morton_emulated_4ASigned.lessThan<false>(int16_t4(Vec4BSigned)));
		
		// Left-shift
		uint16_t castedShift = uint16_t(input.shift);
		left_shift_operator<morton::code<false, smallBits_2, 2> > leftShiftSmall2;
		mortonLeftShift_small_2 = leftShiftSmall2(morton_small_2A, castedShift);
		left_shift_operator<morton::code<false, mediumBits_2, 2> > leftShiftMedium2;
		mortonLeftShift_medium_2 = leftShiftMedium2(morton_medium_2A, castedShift);
		left_shift_operator<morton::code<false, fullBits_2, 2> > leftShiftFull2;
		mortonLeftShift_full_2 = leftShiftFull2(morton_full_2A, castedShift);
		left_shift_operator<morton::code<false, fullBits_2, 2, emulated_uint64_t> > leftShiftEmulated2;
		mortonLeftShift_emulated_2 = leftShiftEmulated2(morton_emulated_2A, castedShift);

		left_shift_operator<morton::code<false, smallBits_3, 3> > leftShiftSmall3;
		mortonLeftShift_small_3 = leftShiftSmall3(morton_small_3A, castedShift);
		left_shift_operator<morton::code<false, mediumBits_3, 3> > leftShiftMedium3;
		mortonLeftShift_medium_3 = leftShiftMedium3(morton_medium_3A, castedShift);
		left_shift_operator<morton::code<false, fullBits_3, 3> > leftShiftFull3;
		mortonLeftShift_full_3 = leftShiftFull3(morton_full_3A, castedShift);
		left_shift_operator<morton::code<false, fullBits_3, 3, emulated_uint64_t> > leftShiftEmulated3;
		mortonLeftShift_emulated_3 = leftShiftEmulated3(morton_emulated_3A, castedShift);

		left_shift_operator<morton::code<false, smallBits_4, 4> > leftShiftSmall4;
		mortonLeftShift_small_4 = leftShiftSmall4(morton_small_4A, castedShift);
		left_shift_operator<morton::code<false, mediumBits_4, 4> > leftShiftMedium4;
		mortonLeftShift_medium_4 = leftShiftMedium4(morton_medium_4A, castedShift);
		left_shift_operator<morton::code<false, fullBits_4, 4> > leftShiftFull4;
		mortonLeftShift_full_4 = leftShiftFull4(morton_full_4A, castedShift);
		left_shift_operator<morton::code<false, fullBits_4, 4, emulated_uint64_t> > leftShiftEmulated4;
		mortonLeftShift_emulated_4 = leftShiftEmulated4(morton_emulated_4A, castedShift);
		
		// Unsigned right-shift
		arithmetic_right_shift_operator<morton::code<false, smallBits_2, 2> > rightShiftSmall2;
		mortonUnsignedRightShift_small_2 = rightShiftSmall2(morton_small_2A, castedShift);
		arithmetic_right_shift_operator<morton::code<false, mediumBits_2, 2> > rightShiftMedium2;
		mortonUnsignedRightShift_medium_2 = rightShiftMedium2(morton_medium_2A, castedShift);
		arithmetic_right_shift_operator<morton::code<false, fullBits_2, 2> > rightShiftFull2;
		mortonUnsignedRightShift_full_2 = rightShiftFull2(morton_full_2A, castedShift);
		arithmetic_right_shift_operator<morton::code<false, fullBits_2, 2, emulated_uint64_t> > rightShiftEmulated2;
		mortonUnsignedRightShift_emulated_2 = rightShiftEmulated2(morton_emulated_2A, castedShift);

		arithmetic_right_shift_operator<morton::code<false, smallBits_3, 3> > rightShiftSmall3;
		mortonUnsignedRightShift_small_3 = rightShiftSmall3(morton_small_3A, castedShift);
		arithmetic_right_shift_operator<morton::code<false, mediumBits_3, 3> > rightShiftMedium3;
		mortonUnsignedRightShift_medium_3 = rightShiftMedium3(morton_medium_3A, castedShift);
		arithmetic_right_shift_operator<morton::code<false, fullBits_3, 3> > rightShiftFull3;
		mortonUnsignedRightShift_full_3 = rightShiftFull3(morton_full_3A, castedShift);
		arithmetic_right_shift_operator<morton::code<false, fullBits_3, 3, emulated_uint64_t> > rightShiftEmulated3;
		mortonUnsignedRightShift_emulated_3 = rightShiftEmulated3(morton_emulated_3A, castedShift);

		arithmetic_right_shift_operator<morton::code<false, smallBits_4, 4> > rightShiftSmall4;
		mortonUnsignedRightShift_small_4 = rightShiftSmall4(morton_small_4A, castedShift);
		arithmetic_right_shift_operator<morton::code<false, mediumBits_4, 4> > rightShiftMedium4;
		mortonUnsignedRightShift_medium_4 = rightShiftMedium4(morton_medium_4A, castedShift);
		arithmetic_right_shift_operator<morton::code<false, fullBits_4, 4> > rightShiftFull4;
		mortonUnsignedRightShift_full_4 = rightShiftFull4(morton_full_4A, castedShift);
		arithmetic_right_shift_operator<morton::code<false, fullBits_4, 4, emulated_uint64_t> > rightShiftEmulated4;
		mortonUnsignedRightShift_emulated_4 = rightShiftEmulated4(morton_emulated_4A, castedShift);

		// Signed right-shift
		arithmetic_right_shift_operator<morton::code<true, smallBits_2, 2> > rightShiftSignedSmall2;
		mortonSignedRightShift_small_2 = rightShiftSignedSmall2(morton_small_2ASigned, castedShift);
		arithmetic_right_shift_operator<morton::code<true, mediumBits_2, 2> > rightShiftSignedMedium2;
		mortonSignedRightShift_medium_2 = rightShiftSignedMedium2(morton_medium_2ASigned, castedShift);
		arithmetic_right_shift_operator<morton::code<true, fullBits_2, 2> > rightShiftSignedFull2;
		mortonSignedRightShift_full_2 = rightShiftSignedFull2(morton_full_2ASigned, castedShift);
		arithmetic_right_shift_operator<morton::code<true, fullBits_2, 2, emulated_uint64_t> > rightShiftSignedEmulated2;
		//mortonSignedRightShift_emulated_2 = rightShiftSignedEmulated2(morton_emulated_2ASigned, castedShift);

		arithmetic_right_shift_operator<morton::code<true, smallBits_3, 3> > rightShiftSignedSmall3;
		mortonSignedRightShift_small_3 = rightShiftSignedSmall3(morton_small_3ASigned, castedShift);
		arithmetic_right_shift_operator<morton::code<true, mediumBits_3, 3> > rightShiftSignedMedium3;
		mortonSignedRightShift_medium_3 = rightShiftSignedMedium3(morton_medium_3ASigned, castedShift);
		arithmetic_right_shift_operator<morton::code<true, fullBits_3, 3> > rightShiftSignedFull3;
		mortonSignedRightShift_full_3 = rightShiftSignedFull3(morton_full_3ASigned, castedShift);
		arithmetic_right_shift_operator<morton::code<true, fullBits_3, 3, emulated_uint64_t> > rightShiftSignedEmulated3;
		//mortonSignedRightShift_emulated_3 = rightShiftSignedEmulated3(morton_emulated_3ASigned, castedShift);

		arithmetic_right_shift_operator<morton::code<true, smallBits_4, 4> > rightShiftSignedSmall4;
		mortonSignedRightShift_small_4 = rightShiftSignedSmall4(morton_small_4ASigned, castedShift);
		arithmetic_right_shift_operator<morton::code<true, mediumBits_4, 4> > rightShiftSignedMedium4;
		mortonSignedRightShift_medium_4 = rightShiftSignedMedium4(morton_medium_4ASigned, castedShift);
		arithmetic_right_shift_operator<morton::code<true, fullBits_4, 4> > rightShiftSignedFull4;
		mortonSignedRightShift_full_4 = rightShiftSignedFull4(morton_full_4ASigned, castedShift);
		arithmetic_right_shift_operator<morton::code<true, fullBits_4, 4, emulated_uint64_t> > rightShiftSignedEmulated4;
		//mortonSignedRightShift_emulated_4 = rightShiftSignedEmulated4(morton_emulated_4ASigned, castedShift);
	}
};

#endif
