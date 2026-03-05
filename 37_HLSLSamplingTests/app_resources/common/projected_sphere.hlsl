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
	float32_t pdf;
	float32_t3 modifiedU;
};

struct ProjectedSphereTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(ProjectedSphereInputValues) input, NBL_REF_ARG(ProjectedSphereTestResults) output)
	{
		float32_t3 sample = input.u;
		output.generated = sampling::ProjectedSphere<float32_t>::generate(sample);
		output.pdf = sampling::ProjectedSphere<float32_t>::pdf(nbl::hlsl::abs(output.generated.z));
		output.modifiedU = sample;
	}
};

#endif
