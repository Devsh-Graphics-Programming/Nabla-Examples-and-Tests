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
	float32_t backwardPdf;
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
			output.cachedPdf = sampler.forwardPdf(output.generated, cache);
			output.forwardPdf = sampler.forwardPdf(output.generated, cache);
			output.modifiedU = sample;
		}
		output.backwardPdf = sampler.backwardPdf(output.generated);
	}
};

#endif
