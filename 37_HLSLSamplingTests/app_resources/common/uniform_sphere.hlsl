#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_UNIFORM_SPHERE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_UNIFORM_SPHERE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/uniform_spheres.hlsl>

using namespace nbl::hlsl;

struct UniformSphereInputValues
{
	float32_t2 u;
};

struct UniformSphereTestResults
{
	float32_t3 generated;
	float32_t pdf;
};

struct UniformSphereTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(UniformSphereInputValues) input, NBL_REF_ARG(UniformSphereTestResults) output)
	{
		output.generated = sampling::UniformSphere<float32_t>::generate(input.u);
		output.pdf = sampling::UniformSphere<float32_t>::pdf();
	}
};

#endif
