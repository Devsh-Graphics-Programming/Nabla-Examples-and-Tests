#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_BOX_MULLER_TRANSFORM_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_BOX_MULLER_TRANSFORM_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/box_muller_transform.hlsl>

using namespace nbl::hlsl;

struct BoxMullerTransformInputValues
{
	float32_t stddev;
	float32_t2 u;
};

struct BoxMullerTransformTestResults
{
	float32_t2 generated;
	float32_t cachedPdf;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t2 separateBackwardPdf;
};

struct BoxMullerTransformTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(BoxMullerTransformInputValues) input, NBL_REF_ARG(BoxMullerTransformTestResults) output)
	{
		sampling::BoxMullerTransform<float32_t> sampler;
		sampler.stddev = input.stddev;

		{
			sampling::BoxMullerTransform<float32_t>::cache_type cache;
			output.generated = sampler.generate(input.u, cache);
			output.cachedPdf = cache.pdf;
			output.forwardPdf = sampler.forwardPdf(cache);
		}

		output.backwardPdf = sampler.backwardPdf(output.generated);
		output.separateBackwardPdf = sampler.separateBackwardPdf(output.generated);
	}
};

#endif
