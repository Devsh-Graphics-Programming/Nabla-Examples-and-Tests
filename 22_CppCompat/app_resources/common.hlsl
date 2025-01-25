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

// tmath.hlsl and intrinsics.hlsl tests

using namespace nbl::hlsl;
struct TgmathIntputTestValues
{
	float floor;
	float lerpX;
	float lerpY;
	float lerpA;
	float isnan;
	float isinf;
	float powX;
	float powY;
	float exp;
	float exp2;
	float log;
	float absF;
	int absI;
	float sqrt;
	float sin;
	float cos;
	float acos;

	float32_t3 floorVec;
	float32_t3 lerpXVec;
	float32_t3 lerpYVec;
	float32_t3 lerpAVec;
	float32_t3 isnanVec;
	float32_t3 isinfVec;
	float32_t3 powXVec;
	float32_t3 powYVec;
	float32_t3 expVec;
	float32_t3 exp2Vec;
	float32_t3 logVec;
	float32_t3 absFVec;
	int32_t3 absIVec;
	float32_t3 sqrtVec;
	float32_t3 sinVec;
	float32_t3 cosVec;
	float32_t3 acosVec;
};

struct TgmathTestValues
{
	float floor;
	float lerp;
	int isnan;
	int isinf;
	float pow;
	float exp;
	float exp2;
	float log;
	float absF;
	int absI;
	float sqrt;
	float sin;
	float cos;
	float acos;

	float32_t3 floorVec;
	float32_t3 lerpVec;
	int32_t3 isnanVec;
	int32_t3 isinfVec;
	float32_t3 powVec;
	float32_t3 expVec;
	float32_t3 exp2Vec;
	float32_t3 logVec;
	float32_t3 absFVec;
	int32_t3 absIVec;
	float32_t3 sqrtVec;
	float32_t3 cosVec;
	float32_t3 sinVec;
	float32_t3 acosVec;

	void fillTestValues(NBL_CONST_REF_ARG(TgmathIntputTestValues) input)
	{
		floor = nbl::hlsl::floor(input.floor);
		lerp = nbl::hlsl::lerp(input.lerpX, input.lerpY, input.lerpA);
		isnan = nbl::hlsl::isnan(input.isnan);
		isinf = nbl::hlsl::isinf(input.isinf);
		pow = nbl::hlsl::pow(input.powX, input.powY);
		exp = nbl::hlsl::exp(input.exp);
		exp2 = nbl::hlsl::exp2(input.exp2);
		log = nbl::hlsl::log(input.log);
		absF = nbl::hlsl::abs(input.absF);
		absI = nbl::hlsl::abs(input.absI);
		sqrt = nbl::hlsl::sqrt(input.sqrt);
		sin = nbl::hlsl::sin(input.sin);
		cos = nbl::hlsl::cos(input.cos);
		acos = nbl::hlsl::acos(input.acos);

		floorVec = nbl::hlsl::floor(input.floorVec);
		lerpVec = nbl::hlsl::lerp(input.lerpXVec, input.lerpYVec, input.lerpAVec);
		//isnanVec = nbl::hlsl::isnan(input.isnanVec);
		//isinfVec = nbl::hlsl::isinf(input.isinfVec);
		powVec = nbl::hlsl::pow(input.powXVec, input.powYVec);
		expVec = nbl::hlsl::exp(input.expVec);
		exp2Vec = nbl::hlsl::exp2(input.exp2Vec);
		logVec = nbl::hlsl::log(input.logVec);
		absFVec = nbl::hlsl::abs(input.absFVec);
		absIVec = nbl::hlsl::abs(input.absIVec);
		sqrtVec = nbl::hlsl::sqrt(input.sqrtVec);
		sinVec = nbl::hlsl::sin(input.sinVec);
		cosVec = nbl::hlsl::cos(input.cosVec);
		acosVec = nbl::hlsl::acos(input.acosVec);
	}
};

#endif