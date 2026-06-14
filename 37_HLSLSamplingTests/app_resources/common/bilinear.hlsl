#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_BILINEAR_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_BILINEAR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/bilinear.hlsl>
#include "jacobian_test.hlsl"

using namespace nbl::hlsl;

struct BilinearInputValues
{
	float32_t4 bilinearCoeffs;
	float32_t2 u;
};

struct BilinearTestResults
{
	float32_t2 generated;
	float32_t backwardPdf;
	float32_t forwardPdf;
	float32_t forwardWeight;
	float32_t backwardWeight;
	float32_t jacobianProduct;
};

struct BilinearTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(BilinearInputValues) input, NBL_REF_ARG(BilinearTestResults) output)
	{
		sampling::Bilinear<float32_t> sampler = sampling::Bilinear<float32_t>::create(input.bilinearCoeffs);
		{
			sampling::Bilinear<float32_t>::cache_type cache;
			output.generated = sampler.generate(input.u, cache);
			output.forwardPdf = sampler.forwardPdf(input.u, cache);
			output.forwardWeight = sampler.forwardWeight(input.u, cache);
		}

		{
			output.backwardPdf = sampler.backwardPdf(output.generated);
			output.backwardWeight = sampler.backwardWeight(output.generated);
		}
		// marginFactor = 3: same reasoning as Linear; Bilinear is two Linear stages, so the skewed-
		// coefficient inverse-CDF d^2/du^2 divergence near [0,1]^2 boundary applies on both axes.
		output.jacobianProduct = computeJacobianProduct<JACOBIAN_PLAIN>(sampler, input.u, 1e-3f, 3.0f);

	}
};

#endif
