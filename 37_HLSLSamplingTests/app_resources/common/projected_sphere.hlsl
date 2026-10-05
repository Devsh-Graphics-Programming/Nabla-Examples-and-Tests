#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl>
#include "jacobian_test.hlsl"

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
	float32_t forwardWeight;
	float32_t backwardWeight;
	float32_t jacobianProduct;
};

struct ProjectedSphereTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(ProjectedSphereInputValues) input, NBL_REF_ARG(ProjectedSphereTestResults) output)
	{
		sampling::ProjectedSphere<float32_t> sampler;
		{
			sampling::ProjectedSphere<float32_t>::cache_type cache;
			float32_t3 _sample = input.u;
			output.generated = sampler.generate(_sample, cache);
			output.cachedPdf = sampler.forwardPdf(_sample, cache);
			output.forwardPdf = sampler.forwardPdf(_sample, cache);
			output.forwardWeight = sampler.forwardWeight(_sample, cache);
			output.modifiedU = _sample;
		}
		output.backwardPdf = sampler.backwardPdf(output.generated);
		output.backwardWeight = sampler.backwardWeight(output.generated);
		output.jacobianProduct = computeJacobianProduct<JACOBIAN_CONCENTRIC>(sampler, input.u, 1e-3f, 5.0f);
	}
};

#endif
