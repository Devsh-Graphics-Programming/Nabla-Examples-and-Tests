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

using namespace nbl::hlsl;
template <typename T, bool Signed, uint16_t Bits>
NBL_CONSTEXPR_INLINE_FUNC T createAnyBitIntegerFromU64(uint64_t val)
{
  if(Signed)
  {
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint64_t mask = (uint64_t(1) << (Bits - 1)) - 1;
    // fill excess bit with one
	if (int64_t(val) < 0)
		return T(val) | ~mask;
	else
        return T(val) & mask;
  } else
  {
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint64_t mask = (uint64_t(1) << Bits) - 1;
    return T(val) & mask;
  }
}

template <typename T, bool Signed, uint16_t Bits, uint16_t D>
NBL_CONSTEXPR_INLINE_FUNC vector<T, D> createAnyBitIntegerVecFromU64Vec(vector<uint64_t, D> val)
{
    array_get<portable_vector_t<uint64_t, D>, uint64_t> getter;
    array_set<portable_vector_t<T, D>, T> setter;
	vector<T, D> output;
    NBL_UNROLL
	for (uint16_t i = 0; i < D; i++)
	{
		setter(output, i, createAnyBitIntegerFromU64<T, Signed, Bits>(getter(val, i)));
	}
	return output;
}

template <bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t = uint64_t>
NBL_CONSTEXPR_INLINE_FUNC morton::code<Signed, Bits, D, _uint64_t> createMortonFromU64Vec(const vector<uint64_t, D> vec)
{
	using morton_code_t = morton::code<Signed, Bits, D, _uint64_t>;
	using decode_component_t = typename morton_code_t::decode_component_t;
	return morton_code_t::create(createAnyBitIntegerVecFromU64Vec<decode_component_t, Signed, Bits, D>(vec));
}

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
	emulated_int64_t emulatedUnaryMinus;
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

	
};

#endif
