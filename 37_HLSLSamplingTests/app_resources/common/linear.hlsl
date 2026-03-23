#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_LINEAR_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_LINEAR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/linear.hlsl>

using namespace nbl::hlsl;

struct LinearInputValues
{
	float32_t2 coeffs;
	float32_t u;
};

struct LinearTestResults
{
	float32_t generated;
	float32_t generateInversed;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t roundtripError;
	float32_t jacobianProduct;
};

struct LinearTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(LinearInputValues) input, NBL_REF_ARG(LinearTestResults) output)
	{
		sampling::Linear<float32_t> _sampler = sampling::Linear<float32_t>::create(input.coeffs);
		{
			sampling::Linear<float32_t>::cache_type cache;
			output.generated = _sampler.generate(input.u, cache);
			output.forwardPdf = _sampler.forwardPdf(cache);
		}

		{
			sampling::Linear<float32_t>::cache_type cache;
			output.generateInversed = _sampler.generateInverse(output.generated);
			output.backwardPdf = _sampler.backwardPdf(output.generated);
		}
		output.roundtripError = abs(input.u - output.generateInversed);
		output.jacobianProduct = (float32_t(1.0) / output.forwardPdf) * output.backwardPdf;
	}
};

#endif
