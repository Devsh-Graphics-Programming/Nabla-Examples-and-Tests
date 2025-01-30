//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_COMMON_INCLUDED_
#define _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_COMMON_INCLUDED_

// because DXC doesn't properly support `_Static_assert`
#define STATIC_ASSERT(...) { nbl::hlsl::conditional<__VA_ARGS__, int, void>::type a = 0; }

#include <boost/preprocessor.hpp>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/colorspace/EOTF.hlsl>
#include <nbl/builtin/hlsl/colorspace/OETF.hlsl>

#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

#include <nbl/builtin/hlsl/mpl.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>

#include <nbl/builtin/hlsl/limits.hlsl>


#include <nbl/builtin/hlsl/barycentric/utils.hlsl>
#include <nbl/builtin/hlsl/member_test_macros.hlsl>
#include <nbl/builtin/hlsl/device_capabilities_traits.hlsl>

#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>

// tgmath.hlsl and intrinsics.hlsl tests

using namespace nbl::hlsl;
struct TgmathIntputTestValues
{
	float floor;
	float isnan;
	float isinf;
	float powX;
	float powY;
	float exp;
	float exp2;
	float log;
	float log2;
	float absF;
	int absI;
	float sqrt;
	float sin;
	float cos;
	float acos;
	float modf;
	float round;
	float roundEven;
	float trunc;
	float ceil;
	float fmaX;
	float fmaY;
	float fmaZ;
	float ldexpArg;
	int ldexpExp;

	float32_t3 floorVec;
	float32_t3 isnanVec;
	float32_t3 isinfVec;
	float32_t3 powXVec;
	float32_t3 powYVec;
	float32_t3 expVec;
	float32_t3 exp2Vec;
	float32_t3 logVec;
	float32_t3 log2Vec;
	float32_t3 absFVec;
	int32_t3 absIVec;
	float32_t3 sqrtVec;
	float32_t3 sinVec;
	float32_t3 cosVec;
	float32_t3 acosVec;
	float32_t3 modfVec;
	float32_t3 roundVec;
	float32_t3 roundEvenVec;
	float32_t3 truncVec;
	float32_t3 ceilVec;
	float32_t3 fmaXVec;
	float32_t3 fmaYVec;
	float32_t3 fmaZVec;
	float32_t3 ldexpArgVec;
	int32_t3 ldexpExpVec;
};

struct TgmathTestValues
{
	float floor;
	int isnan;
	int isinf;
	float pow;
	float exp;
	float exp2;
	float log;
	float log2;
	float absF;
	int absI;
	float sqrt;
	float sin;
	float cos;
	float acos;
	float modf;
	float round;
	float roundEven;
	float trunc;
	float ceil;
	float fma;
	float ldexp;

	float32_t3 floorVec;
#ifndef __HLSL_VERSION
	nbl::hlsl::vector<int, 3> isnanVec;
	nbl::hlsl::vector<int, 3> isinfVec;
#else
	vector<int, 3> isnanVec;
	vector<int, 3> isinfVec;
#endif
	
	float32_t3 powVec;
	float32_t3 expVec;
	float32_t3 exp2Vec;
	float32_t3 logVec;
	float32_t3 log2Vec;
	float32_t3 absFVec;
	int32_t3 absIVec;
	float32_t3 sqrtVec;
	float32_t3 cosVec;
	float32_t3 sinVec;
	float32_t3 acosVec;
	float32_t3 modfVec;
	float32_t3 roundVec;
	float32_t3 roundEvenVec;
	float32_t3 truncVec;
	float32_t3 ceilVec;
	float32_t3 fmaVec;
	float32_t3 ldexpVec;

	void fillTestValues(NBL_CONST_REF_ARG(TgmathIntputTestValues) input)
	{
		floor = nbl::hlsl::floor(input.floor);
		isnan = nbl::hlsl::isnan(input.isnan);
		isinf = nbl::hlsl::isinf(input.isinf);
		pow = nbl::hlsl::pow(input.powX, input.powY);
		exp = nbl::hlsl::exp(input.exp);
		exp2 = nbl::hlsl::exp2(input.exp2);
		log = nbl::hlsl::log(input.log);
		log2 = nbl::hlsl::log2(input.log2);
		absF = nbl::hlsl::abs(input.absF);
		absI = nbl::hlsl::abs(input.absI);
		sqrt = nbl::hlsl::sqrt(input.sqrt);
		sin = nbl::hlsl::sin(input.sin);
		cos = nbl::hlsl::cos(input.cos);
		acos = nbl::hlsl::acos(input.acos);
		modf = nbl::hlsl::modf(input.modf);
		round = nbl::hlsl::round(input.round);
		roundEven = nbl::hlsl::roundEven(input.roundEven);
		trunc = nbl::hlsl::trunc(input.trunc);
		ceil = nbl::hlsl::ceil(input.ceil);
		fma = nbl::hlsl::fma(input.fmaX, input.fmaY, input.fmaZ);
		ldexp = nbl::hlsl::ldexp(input.ldexpArg, input.ldexpExp);

		floorVec = nbl::hlsl::floor(input.floorVec);
		isnanVec = nbl::hlsl::isnan(input.isnanVec);
		isinfVec = nbl::hlsl::isinf(input.isinfVec);
		powVec = nbl::hlsl::pow(input.powXVec, input.powYVec);
		expVec = nbl::hlsl::exp(input.expVec);
		exp2Vec = nbl::hlsl::exp2(input.exp2Vec);
		logVec = nbl::hlsl::log(input.logVec);
		log2Vec = nbl::hlsl::log2(input.log2Vec);
		absFVec = nbl::hlsl::abs(input.absFVec);
		absIVec = nbl::hlsl::abs(input.absIVec);
		sqrtVec = nbl::hlsl::sqrt(input.sqrtVec);
		sinVec = nbl::hlsl::sin(input.sinVec);
		cosVec = nbl::hlsl::cos(input.cosVec);
		acosVec = nbl::hlsl::acos(input.acosVec);
		modfVec = nbl::hlsl::modf(input.modfVec);
		roundVec = nbl::hlsl::round(input.roundVec);
		roundEvenVec = nbl::hlsl::roundEven(input.roundEvenVec);
		truncVec = nbl::hlsl::trunc(input.truncVec);
		ceilVec = nbl::hlsl::ceil(input.ceilVec);
		fmaVec = nbl::hlsl::fma(input.fmaXVec, input.fmaYVec, input.fmaZVec);
		ldexpVec = nbl::hlsl::ldexp(input.ldexpArgVec, input.ldexpExpVec);
	}
};

struct IntrinsicsIntputTestValues
{
	int bitCount;
	float32_t3 crossLhs;
	float32_t3 crossRhs;
	float clampVal;
	float clampMin;
	float clampMax;
	float32_t3 length;
	float32_t3 normalize;
	float32_t3 dotLhs;
	float32_t3 dotRhs;
	float32_t3x3 determinant;
	uint32_t findMSB;
	uint32_t findLSB;
	float32_t3x3 inverse;
	float32_t3x3 transpose;
	float32_t3x3 mulLhs;
	float32_t3x3 mulRhs;
	float minA;
	float minB;
	float maxA;
	float maxB;
	float rsqrt;
	uint32_t bitReverse;
	float frac;
	float mixX;
	float mixY;
	float mixA;

	int32_t3 bitCountVec;
	float32_t3 clampValVec;
	float32_t3 clampMinVec;
	float32_t3 clampMaxVec;
	uint32_t3 findMSBVec;
	uint32_t3 findLSBVec;
	float32_t3 minAVec;
	float32_t3 minBVec;
	float32_t3 maxAVec;
	float32_t3 maxBVec;
	float32_t3 rsqrtVec;
	uint32_t3 bitReverseVec;
	float32_t3 fracVec;
	float32_t3 mixXVec;
	float32_t3 mixYVec;
	float32_t3 mixAVec;
};

struct IntrinsicsTestValues
{
	int bitCount;
	float clamp;
	float length;
	float dot;
	float determinant;
	int findMSB;
	int findLSB;
	float min;
	float max;
	float rsqrt;
	float frac;
	uint32_t bitReverse;
	float mix;

	float32_t3 normalize;
	float32_t3 cross;
	int32_t3 bitCountVec;
	float32_t3 clampVec;
	uint32_t3 findMSBVec;
	uint32_t3 findLSBVec;
	float32_t3 minVec;
	float32_t3 maxVec;
	float32_t3 rsqrtVec;
	uint32_t3 bitReverseVec;
	float32_t3 fracVec;
	float32_t3 mixVec;

	float32_t3x3 mul;
	float32_t3x3 transpose;
	float32_t3x3 inverse;

	void fillTestValues(NBL_CONST_REF_ARG(IntrinsicsIntputTestValues) input)
	{
		bitCount = nbl::hlsl::bitCount(input.bitCount);
		cross = nbl::hlsl::cross(input.crossLhs, input.crossRhs);
		clamp = nbl::hlsl::clamp(input.clampVal, input.clampMin, input.clampMax);
		length = nbl::hlsl::length(input.length);
		normalize = nbl::hlsl::normalize(input.normalize);
		dot = nbl::hlsl::dot(input.dotLhs, input.dotRhs);
		determinant = nbl::hlsl::determinant(input.determinant);
		findMSB = nbl::hlsl::findMSB(input.findMSB);
		findLSB = nbl::hlsl::findLSB(input.findLSB);
		inverse = nbl::hlsl::inverse(input.inverse);
		transpose = nbl::hlsl::transpose(input.transpose);
		mul = nbl::hlsl::mul(input.mulLhs, input.mulRhs);
		// TODO: fix min and max
		//min = nbl::hlsl::min(input.minA, input.minB);
		//max = nbl::hlsl::max(input.maxA, input.maxB);
		rsqrt = nbl::hlsl::rsqrt(input.rsqrt);
		bitReverse = nbl::hlsl::bitReverse(input.bitReverse);
		frac = nbl::hlsl::frac(input.frac);
		mix = nbl::hlsl::mix(input.mixX, input.mixY, input.mixA);

		bitCountVec = nbl::hlsl::bitCount(input.bitCountVec);
		clampVec = nbl::hlsl::clamp(input.clampValVec, input.clampMinVec, input.clampMaxVec);
		findMSBVec = nbl::hlsl::findMSB(input.findMSBVec);
		findLSBVec = nbl::hlsl::findLSB(input.findLSBVec);
		// TODO: fix min and max
		//minVec = nbl::hlsl::min(input.minAVec, input.minBVec);
		//maxVec = nbl::hlsl::max(input.maxAVec, input.maxBVec);
		rsqrtVec = nbl::hlsl::rsqrt(input.rsqrtVec);
		bitReverseVec = nbl::hlsl::bitReverse(input.bitReverseVec);
		fracVec = nbl::hlsl::frac(input.fracVec);
		mixVec = nbl::hlsl::mix(input.mixXVec, input.mixYVec, input.mixAVec);
	}
};

#endif
