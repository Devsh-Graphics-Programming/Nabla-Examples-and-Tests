#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl>

using namespace nbl::hlsl;

struct ProjectedSphereInputValues
{
	float32_t3 u;
};

struct ProjectedSphereTestResults
{
	float32_t3 generated;
	float32_t cachedPdf;
	float32_t forwardPdf;
	float32_t3 modifiedU;
	float32_t3 inverted;
	float32_t backwardPdf;
	// Only xy round-trips accurately; z information is intentionally lost in generateInverse
	// (it maps to 0 or 1 based on sign of generated.z, not the exact original value).
	float32_t roundtripError;
	float32_t jacobianProduct;
};

struct ProjectedSphereTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(ProjectedSphereInputValues) input, NBL_REF_ARG(ProjectedSphereTestResults) output)
	{
		sampling::ProjectedSphere<float32_t> sampler;
		{
			sampling::ProjectedSphere<float32_t>::cache_type cache;
			float32_t3 sample = input.u;
			output.generated = sampler.generate(sample, cache);
			output.cachedPdf = cache.pdf;
			output.forwardPdf = sampler.forwardPdf(cache);
			output.modifiedU = sample;
		}
		{
			sampling::ProjectedSphere<float32_t>::cache_type cache;
			output.inverted = sampler.generateInverse(output.generated, cache);
			output.backwardPdf = sampler.backwardPdf(output.generated);
		}
		float32_t2 xyDiff = output.modifiedU.xy - output.inverted.xy;
		output.roundtripError = nbl::hlsl::length(xyDiff);
		output.jacobianProduct = (float32_t(1.0) / output.forwardPdf) * output.backwardPdf;
	}
};

#endif
