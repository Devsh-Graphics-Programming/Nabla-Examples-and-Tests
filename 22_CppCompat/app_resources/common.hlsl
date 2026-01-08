//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_COMMON_INCLUDED_
#define _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_COMMON_INCLUDED_

// because DXC doesn't properly support `_Static_assert`
// TODO: add a message, and move to macros.h or cpp_compat
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
	float modfStruct;
	float frexpStruct;
	float tan;
	float asin;
	float atan;
	float sinh;
	float cosh;
	float tanh;
	float asinh;
	float acosh;
	float atanh;
	float atan2X;
	float atan2Y;
	float erf;
	float erfInv;

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
	float32_t3 modfStructVec;
	float32_t3 frexpStructVec;
	float32_t3 tanVec;
	float32_t3 asinVec;
	float32_t3 atanVec;
	float32_t3 sinhVec;
	float32_t3 coshVec;
	float32_t3 tanhVec;
	float32_t3 asinhVec;
	float32_t3 acoshVec;
	float32_t3 atanhVec;
	float32_t3 atan2XVec;
	float32_t3 atan2YVec;
	float32_t3 erfVec;
	float32_t3 erfInvVec;
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
	float tan;
	float asin;
	float atan;
	float sinh;
	float cosh;
	float tanh;
	float asinh;
	float acosh;
	float atanh;
	float atan2;
	float erf;
	float erfInv;

	float32_t3 floorVec;

	// we can't fix this because using namespace nbl::hlsl would cause ambiguous math functions below 
	// and we can't add a nbl::hlsl alias for the builtin hLSL vector type because of https://github.com/microsoft/DirectXShaderCompiler/issues/7035
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
	float32_t3 tanVec;
	float32_t3 asinVec;
	float32_t3 atanVec;
	float32_t3 sinhVec;
	float32_t3 coshVec;
	float32_t3 tanhVec;
	float32_t3 asinhVec;
	float32_t3 acoshVec;
	float32_t3 atanhVec;
	float32_t3 atan2Vec;
	float32_t3 erfVec;
	float32_t3 erfInvVec;

	ModfOutput<float> modfStruct;
	ModfOutput<float32_t3> modfStructVec;
	FrexpOutput<float> frexpStruct;
	FrexpOutput<float32_t3> frexpStructVec;
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
	float sign;
	float radians;
	float degrees;
	float stepEdge;
	float stepX;
	float smoothStepEdge0;
	float smoothStepEdge1;
	float smoothStepX;
	uint32_t addCarryA;
	uint32_t addCarryB;
	uint32_t subBorrowA;
	uint32_t subBorrowB;

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
	float32_t3 signVec;
	float32_t3 radiansVec;
	float32_t3 degreesVec;
	float32_t3 stepEdgeVec;
	float32_t3 stepXVec;
	float32_t3 smoothStepEdge0Vec;
	float32_t3 smoothStepEdge1Vec;
	float32_t3 smoothStepXVec;
	float32_t3 faceForwardN;
	float32_t3 faceForwardI;
	float32_t3 faceForwardNref;
	float32_t3 reflectI;
	float32_t3 reflectN;
	float32_t3 refractI;
	float32_t3 refractN;
	float refractEta;
	uint32_t3 addCarryAVec;
	uint32_t3 addCarryBVec;
	uint32_t3 subBorrowAVec;
	uint32_t3 subBorrowBVec;
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
	float sign;
	float radians;
	float degrees;
	float step;
	float smoothStep;

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
	float32_t3 signVec;
	float32_t3 radiansVec;
	float32_t3 degreesVec;
	float32_t3 stepVec;
	float32_t3 smoothStepVec;
	float32_t3 faceForward;
	float32_t3 reflect;
	float32_t3 refract;

	float32_t3x3 mul;
	float32_t3x3 transpose;
	float32_t3x3 inverse;

	spirv::AddCarryOutput<uint32_t> addCarry;
	spirv::SubBorrowOutput<uint32_t> subBorrow;
	spirv::AddCarryOutput<uint32_t3> addCarryVec;
	spirv::SubBorrowOutput<uint32_t3> subBorrowVec;
};

struct IntrinsicsTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(IntrinsicsIntputTestValues) input, NBL_REF_ARG(IntrinsicsTestValues) output)
	{
		output.bitCount = nbl::hlsl::bitCount(input.bitCount);
		output.cross = nbl::hlsl::cross(input.crossLhs, input.crossRhs);
		output.clamp = nbl::hlsl::clamp(input.clampVal, input.clampMin, input.clampMax);
		output.length = nbl::hlsl::length(input.length);
		output.normalize = nbl::hlsl::normalize(input.normalize);
		output.dot = nbl::hlsl::dot(input.dotLhs, input.dotRhs);
		output.determinant = nbl::hlsl::determinant(input.determinant);
		output.findMSB = nbl::hlsl::findMSB(input.findMSB);
		output.findLSB = nbl::hlsl::findLSB(input.findLSB);
		output.inverse = nbl::hlsl::inverse(input.inverse);
		output.transpose = nbl::hlsl::transpose(input.transpose);
		output.mul = nbl::hlsl::mul(input.mulLhs, input.mulRhs);
		// TODO: fix min and max
		output.min = nbl::hlsl::min(input.minA, input.minB);
		output.max = nbl::hlsl::max(input.maxA, input.maxB);
		output.rsqrt = nbl::hlsl::rsqrt(input.rsqrt);
		output.bitReverse = nbl::hlsl::bitReverse(input.bitReverse);
		output.frac = nbl::hlsl::fract(input.frac);
		output.mix = nbl::hlsl::mix(input.mixX, input.mixY, input.mixA);
		output.sign = nbl::hlsl::sign(input.sign);
		output.radians = nbl::hlsl::radians(input.radians);
		output.degrees = nbl::hlsl::degrees(input.degrees);
		output.step = nbl::hlsl::step(input.stepEdge, input.stepX);
		output.smoothStep = nbl::hlsl::smoothStep(input.smoothStepEdge0, input.smoothStepEdge1, input.smoothStepX);

		output.bitCountVec = nbl::hlsl::bitCount(input.bitCountVec);
		output.clampVec = nbl::hlsl::clamp(input.clampValVec, input.clampMinVec, input.clampMaxVec);
		output.findMSBVec = nbl::hlsl::findMSB(input.findMSBVec);
		output.findLSBVec = nbl::hlsl::findLSB(input.findLSBVec);
		// TODO: fix min and max
		output.minVec = nbl::hlsl::min(input.minAVec, input.minBVec);
		output.maxVec = nbl::hlsl::max(input.maxAVec, input.maxBVec);
		output.rsqrtVec = nbl::hlsl::rsqrt(input.rsqrtVec);
		output.bitReverseVec = nbl::hlsl::bitReverse(input.bitReverseVec);
		output.fracVec = nbl::hlsl::fract(input.fracVec);
		output.mixVec = nbl::hlsl::mix(input.mixXVec, input.mixYVec, input.mixAVec);

		output.signVec = nbl::hlsl::sign(input.signVec);
		output.radiansVec = nbl::hlsl::radians(input.radiansVec);
		output.degreesVec = nbl::hlsl::degrees(input.degreesVec);
		output.stepVec = nbl::hlsl::step(input.stepEdgeVec, input.stepXVec);
		output.smoothStepVec = nbl::hlsl::smoothStep(input.smoothStepEdge0Vec, input.smoothStepEdge1Vec, input.smoothStepXVec);
		output.faceForward = nbl::hlsl::faceForward(input.faceForwardN, input.faceForwardI, input.faceForwardNref);
		output.reflect = nbl::hlsl::reflect(input.reflectI, input.reflectN);
		output.refract = nbl::hlsl::refract(input.refractI, input.refractN, input.refractEta);
		output.addCarry = nbl::hlsl::addCarry(input.addCarryA, input.addCarryB);
		output.subBorrow = nbl::hlsl::subBorrow(input.subBorrowA, input.subBorrowB);
		output.addCarryVec = nbl::hlsl::addCarry(input.addCarryAVec, input.addCarryBVec);
		output.subBorrowVec = nbl::hlsl::subBorrow(input.subBorrowAVec, input.subBorrowBVec);
	}
};

struct TgmathTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(TgmathIntputTestValues) input, NBL_REF_ARG(TgmathTestValues) output)
	{
		output.floor = nbl::hlsl::floor(input.floor);
		output.isnan = nbl::hlsl::isnan(input.isnan);
		output.isinf = nbl::hlsl::isinf(input.isinf);
		output.pow = nbl::hlsl::pow(input.powX, input.powY);
		output.exp = nbl::hlsl::exp(input.exp);
		output.exp2 = nbl::hlsl::exp2(input.exp2);
		output.log = nbl::hlsl::log(input.log);
		output.log2 = nbl::hlsl::log2(input.log2);
		output.absF = nbl::hlsl::abs(input.absF);
		output.absI = nbl::hlsl::abs(input.absI);
		output.sqrt = nbl::hlsl::sqrt(input.sqrt);
		output.sin = nbl::hlsl::sin(input.sin);
		output.cos = nbl::hlsl::cos(input.cos);
		output.tan = nbl::hlsl::tan(input.tan);
		output.asin = nbl::hlsl::asin(input.asin);
		output.atan = nbl::hlsl::atan(input.atan);
		output.sinh = nbl::hlsl::sinh(input.sinh);
		output.cosh = nbl::hlsl::cosh(input.cosh);
		output.tanh = nbl::hlsl::tanh(input.tanh);
		output.asinh = nbl::hlsl::asinh(input.asinh);
		output.acosh = nbl::hlsl::acosh(input.acosh);
		output.atanh = nbl::hlsl::atanh(input.atanh);
		output.atan2 = nbl::hlsl::atan2(input.atan2Y, input.atan2X);
		output.erf = nbl::hlsl::erf(input.erf);
		output.erfInv = nbl::hlsl::erfInv(input.erfInv);
		output.acos = nbl::hlsl::acos(input.acos);
		output.modf = nbl::hlsl::modf(input.modf);
		output.round = nbl::hlsl::round(input.round);
		output.roundEven = nbl::hlsl::roundEven(input.roundEven);
		output.trunc = nbl::hlsl::trunc(input.trunc);
		output.ceil = nbl::hlsl::ceil(input.ceil);
		output.fma = nbl::hlsl::fma(input.fmaX, input.fmaY, input.fmaZ);
		output.ldexp = nbl::hlsl::ldexp(input.ldexpArg, input.ldexpExp);

		output.floorVec = nbl::hlsl::floor(input.floorVec);
		output.isnanVec = nbl::hlsl::isnan(input.isnanVec);
		output.isinfVec = nbl::hlsl::isinf(input.isinfVec);
		output.powVec = nbl::hlsl::pow(input.powXVec, input.powYVec);
		output.expVec = nbl::hlsl::exp(input.expVec);
		output.exp2Vec = nbl::hlsl::exp2(input.exp2Vec);
		output.logVec = nbl::hlsl::log(input.logVec);
		output.log2Vec = nbl::hlsl::log2(input.log2Vec);
		output.absFVec = nbl::hlsl::abs(input.absFVec);
		output.absIVec = nbl::hlsl::abs(input.absIVec);
		output.sqrtVec = nbl::hlsl::sqrt(input.sqrtVec);
		output.sinVec = nbl::hlsl::sin(input.sinVec);
		output.cosVec = nbl::hlsl::cos(input.cosVec);
		output.tanVec = nbl::hlsl::tan(input.tanVec);
		output.asinVec = nbl::hlsl::asin(input.asinVec);
		output.atanVec = nbl::hlsl::atan(input.atanVec);
		output.sinhVec = nbl::hlsl::sinh(input.sinhVec);
		output.coshVec = nbl::hlsl::cosh(input.coshVec);
		output.tanhVec = nbl::hlsl::tanh(input.tanhVec);
		output.asinhVec = nbl::hlsl::asinh(input.asinhVec);
		output.acoshVec = nbl::hlsl::acosh(input.acoshVec);
		output.atanhVec = nbl::hlsl::atanh(input.atanhVec);
		output.atan2Vec = nbl::hlsl::atan2(input.atan2YVec, input.atan2XVec);
		output.acosVec = nbl::hlsl::acos(input.acosVec);
		output.modfVec = nbl::hlsl::modf(input.modfVec);
		output.roundVec = nbl::hlsl::round(input.roundVec);
		output.roundEvenVec = nbl::hlsl::roundEven(input.roundEvenVec);
		output.truncVec = nbl::hlsl::trunc(input.truncVec);
		output.ceilVec = nbl::hlsl::ceil(input.ceilVec);
		output.fmaVec = nbl::hlsl::fma(input.fmaXVec, input.fmaYVec, input.fmaZVec);
		output.ldexpVec = nbl::hlsl::ldexp(input.ldexpArgVec, input.ldexpExpVec);
		output.erfVec = nbl::hlsl::erf(input.erfVec);
		output.erfInvVec = nbl::hlsl::erfInv(input.erfInvVec);

		output.modfStruct = nbl::hlsl::modfStruct(input.modfStruct);
		output.modfStructVec = nbl::hlsl::modfStruct(input.modfStructVec);
		output.frexpStruct = nbl::hlsl::frexpStruct(input.frexpStruct);
		output.frexpStructVec = nbl::hlsl::frexpStruct(input.frexpStructVec);
	}
};

#endif
