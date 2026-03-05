#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_LINEAR_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_LINEAR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/linear.hlsl>

using namespace nbl::hlsl;

struct LinearInputValues
{
	float32_t2 coeffs;
	float32_t u;
};

struct LinearTestResults
{
	float32_t generated;
};

struct LinearTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(LinearInputValues) input, NBL_REF_ARG(LinearTestResults) output)
	{
		sampling::Linear<float32_t> _sampler = sampling::Linear<float32_t>::create(input.coeffs);
		output.generated = _sampler.generate(input.u);
	}
};

#endif
