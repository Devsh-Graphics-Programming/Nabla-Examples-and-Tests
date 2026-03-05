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
	float32_t pdf;
};

struct ProjectedHemisphereTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(ProjectedHemisphereInputValues) input, NBL_REF_ARG(ProjectedHemisphereTestResults) output)
	{
		output.generated = sampling::ProjectedHemisphere<float32_t>::generate(input.u);
		output.pdf = sampling::ProjectedHemisphere<float32_t>::pdf(output.generated.z);
	}
};

#endif
