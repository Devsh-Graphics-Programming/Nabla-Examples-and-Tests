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
};

struct UniformHemisphereTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(UniformHemisphereInputValues) input, NBL_REF_ARG(UniformHemisphereTestResults) output)
	{
		output.generated = sampling::UniformHemisphere<float32_t>::generate(input.u);
		output.pdf = sampling::UniformHemisphere<float32_t>::pdf();
	}
};

#endif
