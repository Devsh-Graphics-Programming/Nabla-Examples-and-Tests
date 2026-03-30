#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_UNIFORM_HEMISPHERE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_UNIFORM_HEMISPHERE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/uniform_spheres.hlsl>

using namespace nbl::hlsl;

struct UniformHemisphereInputValues
{
	float32_t2 u;
};

struct UniformHemisphereTestResults
{
	float32_t3 generated;
	float32_t pdf;
	float32_t2 inverted;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t roundtripError;
	float32_t jacobianProduct;
};

struct UniformHemisphereTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(UniformHemisphereInputValues) input, NBL_REF_ARG(UniformHemisphereTestResults) output)
	{
		sampling::UniformHemisphere<float32_t> sampler;
		{
			sampling::UniformHemisphere<float32_t>::cache_type cache;
			output.generated = sampler.generate(input.u, cache);
			output.forwardPdf = sampler.forwardPdf(input.u, cache);
		}
		{
			sampling::UniformHemisphere<float32_t>::cache_type cache;
			output.inverted = sampler.generateInverse(output.generated);
			output.backwardPdf = sampler.backwardPdf(output.generated);
		}
		float32_t2 diff = input.u - output.inverted;
		output.roundtripError = nbl::hlsl::length(diff);
		output.jacobianProduct = (float32_t(1.0) / output.forwardPdf) * output.backwardPdf;
	}
};

#endif
