#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_HEMISPHERE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_HEMISPHERE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl>

using namespace nbl::hlsl;

struct ProjectedHemisphereInputValues
{
	float32_t2 u;
};

struct ProjectedHemisphereTestResults
{
	float32_t3 generated;
	float32_t2 inverted;
	float32_t cachedPdf;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t roundtripError;
	float32_t jacobianProduct;
};

struct ProjectedHemisphereTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(ProjectedHemisphereInputValues) input, NBL_REF_ARG(ProjectedHemisphereTestResults) output)
	{
		sampling::ProjectedHemisphere<float32_t> sampler;
		{
			sampling::ProjectedHemisphere<float32_t>::cache_type cache;
			output.generated = sampler.generate(input.u, cache);
			output.cachedPdf = sampler.forwardPdf(output.generated, cache);
			output.forwardPdf = sampler.forwardPdf(output.generated, cache);
		}
		{
			sampling::ProjectedHemisphere<float32_t>::cache_type cache;
			output.inverted = sampler.generateInverse(output.generated);
			output.backwardPdf = sampler.backwardPdf(output.generated);
		}
		float32_t2 diff = input.u - output.inverted;
		output.roundtripError = nbl::hlsl::length(diff);
		output.jacobianProduct = (float32_t(1.0) / output.forwardPdf) * output.backwardPdf;
	}
};

#endif
