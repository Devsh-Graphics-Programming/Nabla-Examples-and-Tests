//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#pragma shader_stage(compute)

#include "app_resources/benchmark/common.hlsl"
#include <nbl/builtin/hlsl/emulated/float64_t.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/gauss_legendre.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>

using namespace nbl::hlsl;

[[vk::binding(0, 0)]] RWByteAddressBuffer outputBuffer;
[[vk::push_constant]] BenchmarkPushConstants pc;

// for initial seed of 69, the polynomial is:
//	f(x) = x^15 * 187.804 + x^14 * 11.6964 + x^13 * 2450.9 + x^12 * 88.6756 + x^11 * 3.62408 + x^10 * 11.4605 + x^9 * 53276.3 + x^8 * 16045.4 + x^7 * 2260.61 + x^6 * 8162.57 
// + x^5 * 20.674 + x^4 * 13918.6 + x^3 * 2.36093 + x^2 * 8.72536 + x^1 * 2335.63 + 176.719
// f(1) = 98961.74987
// int from 0 to 69 = 3.11133×10^30

template<typename F64>
struct Random16thPolynomial
{
	void randomizeCoefficients()
	{
		Xoroshiro64Star rng = Xoroshiro64Star::construct(69);

		// can't just do `coefficients[i] = rng()` this will create retarded numbers or special value exponents
		for (int i = 0; i < CoefficientNumber; ++i)
		{
			uint64_t exp = uint64_t(rng()) % 16 + ieee754::traits<float64_t>::exponentBias;
			exp <<= ieee754::traits<float64_t>::mantissaBitCnt;
			uint64_t mantissa = (uint64_t(rng()) << 32) | uint64_t(rng());
			mantissa &= ieee754::traits<float64_t>::mantissaMask;

			coefficients[i] = bit_cast<F64>(exp | mantissa);
		}

		//debugPrintCoefficients();
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

	void debugPrintCoefficients()
	{
		printf("x^15 = %llu, x^14 = %llu, x^13 = %llu, x^12 = %llu, x^11 = %llu, x^10 = %llu, x^9 = %llu, x^8 = %llu, x^7 = %llu, x^6 = %llu, x^5 = %llu, x^4 = %llu, x^3 = %llu, x^2 = %llu, x^1 = %llu, x^0 = %llu",
			bit_cast<uint64_t>(coefficients[0]),
			bit_cast<uint64_t>(coefficients[1]),
			bit_cast<uint64_t>(coefficients[2]),
			bit_cast<uint64_t>(coefficients[3]),
			bit_cast<uint64_t>(coefficients[4]),
			bit_cast<uint64_t>(coefficients[5]),
			bit_cast<uint64_t>(coefficients[6]),
			bit_cast<uint64_t>(coefficients[7]),
			bit_cast<uint64_t>(coefficients[8]),
			bit_cast<uint64_t>(coefficients[9]),
			bit_cast<uint64_t>(coefficients[10]),
			bit_cast<uint64_t>(coefficients[11]),
			bit_cast<uint64_t>(coefficients[12]),
			bit_cast<uint64_t>(coefficients[13]),
			bit_cast<uint64_t>(coefficients[14]),
			bit_cast<uint64_t>(coefficients[15]));
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
void main(uint3 invocationID : SV_DispatchThreadID)
{
	/*if (pc.testEmulatedFloat64)
		printf("testing emulated");
	else
		printf("testing native");*/

	uint64_t output;
	switch (pc.benchmarkMode)
	{
	case NATIVE:
	{
		output = calcIntegral<float64_t>();
		break;
	}
	case EF64_FAST_MATH_ENABLED:
	{
		output = calcIntegral<emulated_float64_t<true, true> >();
		break;
	}
	case EF64_FAST_MATH_DISABLED:
	{
		output = calcIntegral<emulated_float64_t<false, true> >();
		break;
	}
	case SUBGROUP_DIVIDED_WORK:
	{
		const bool emulated = (WaveGetLaneIndex() & 0x1) != 0;
		if (emulated)
			output = calcIntegral<emulated_float64_t<false, true> >();
		else
			output = calcIntegral<float64_t>();

		break;
	}
	case INTERLEAVED:
	{
		uint64_t a = calcIntegral<float64_t>();
		uint64_t b = calcIntegral<emulated_float64_t<false, true> >();
		output = a + b; // addional add operation, don't know any better way to avoid dead code optimization

		break;
	}
	}

	const uint32_t offset = sizeof(uint64_t) * invocationID.x;
	outputBuffer.Store<uint64_t>(offset, output);
}
