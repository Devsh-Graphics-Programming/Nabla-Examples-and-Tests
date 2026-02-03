//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/benchmark/common.hlsl"
#include <nbl/builtin/hlsl/emulated/float64_t.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/gauss_legendre.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl>

using namespace nbl::hlsl;

[[vk::binding(0, 0)]] RWByteAddressBuffer outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

template<typename F64>
struct Random16thPolynomial
{
	void randomizeCoefficients()
	{
		Xoroshiro64Star rng = Xoroshiro64Star::construct(100);

		for (int i = 0; i < CoefficientNumber; ++i)
		{
			uint64_t exp = uint64_t(rng()) % 16 + ieee754::traits<float64_t>::exponentBias;
			exp <<= ieee754::traits<float64_t>::mantissaBitCnt;
			uint64_t mantissa = (uint64_t(rng()) << 32) | uint64_t(rng());
			mantissa &= ieee754::traits<float64_t>::mantissaMask;

			coefficients[i] = bit_cast<F64>(exp | mantissa);
		}
	}

	F64 operator()(F64 x)
	{
		F64 result = coefficients[CoefficientNumber - 1];

		F64 xn = x;
		for (int i = CoefficientNumber - 2; i >= 0; --i)
		{
			result = result + coefficients[i] * xn;
			xn = xn * x;
		}

		return result;
	}

	static const int CoefficientNumber = 16;
	static const bool reasonableCoefficients = true;

	F64 coefficients[CoefficientNumber];
};

template<typename F64>
uint64_t calcIntegral()
{
	Random16thPolynomial<F64> polynomial;
	polynomial.randomizeCoefficients();

	using Integrator = math::quadrature::GaussLegendreIntegration<15, F64, Random16thPolynomial<F64> >;
	F64 integral = Integrator::calculateIntegral(polynomial, _static_cast<F64>(0.0f), _static_cast<F64>(69.0f));

	return bit_cast<uint64_t>(integral);
}

[numthreads(BENCHMARK_WORKGROUP_DIMENSION_SIZE_X, 1, 1)]
[shader("compute")]
void main(uint3 invocationID : SV_DispatchThreadID)
{
	static const uint32_t NativeToEmulatedRatio = 6;
	// slightly more invocations will go to native so `NativeToEmulatedRatio-1 < real ratio <= NativeToEmulatedRatio`
	const bool nativeSubgroup = bool(glsl::gl_SubgroupID() % NativeToEmulatedRatio);

	uint64_t output = 0ull;

	switch (pc.benchmarkMode)
	{
	case NATIVE:
		output = calcIntegral<float64_t>();
		break;
	case EF64_FAST_MATH_ENABLED:
		output = calcIntegral<emulated_float64_t<true, true> >();
		break;
	case EF64_FAST_MATH_DISABLED:
		output = calcIntegral<emulated_float64_t<false, true> >();
		break;
	case SUBGROUP_DIVIDED_WORK:
		if (nativeSubgroup)
			output = calcIntegral<float64_t>();
		else
			output = calcIntegral<emulated_float64_t<true, true> >();
		break;
	case INTERLEAVED:
		output = calcIntegral<emulated_float64_t<true, true> >();
		break;
	}

	for (uint32_t i = 0; i < NativeToEmulatedRatio; ++i)
	{
		switch (pc.benchmarkMode)
		{
		case NATIVE:
		case INTERLEAVED:
			output ^= calcIntegral<float64_t>();
			break;
		case EF64_FAST_MATH_ENABLED:
			output ^= calcIntegral<emulated_float64_t<true, true> >();
			break;
		case EF64_FAST_MATH_DISABLED:
			output ^= calcIntegral<emulated_float64_t<false, true> >();
			break;
		case SUBGROUP_DIVIDED_WORK:
			if (nativeSubgroup)
				output ^= calcIntegral<float64_t>();
			else
				output ^= calcIntegral<emulated_float64_t<true, true> >();
			break;
		}
	}

	const uint32_t offset = sizeof(uint64_t) * invocationID.x;
	outputBuffer.Store<uint64_t>(offset, output);
}
