#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_POLAR_MAPPING_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_POLAR_MAPPING_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/polar_mapping.hlsl>

using namespace nbl::hlsl;

struct PolarMappingInputValues
{
	float32_t2 u;
};

struct PolarMappingTestResults
{
	float32_t2 mapped;
	float32_t2 inverted;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t jacobianProduct;
	float32_t roundtripError;
};

struct PolarMappingTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(PolarMappingInputValues) input, NBL_REF_ARG(PolarMappingTestResults) output)
	{
		{
			sampling::PolarMapping<float32_t>::cache_type cache;
			output.mapped = sampling::PolarMapping<float32_t>::generate(input.u, cache);
			output.forwardPdf = sampling::PolarMapping<float32_t>::forwardPdf(output.mapped, cache);
		}
		{
			output.inverted = sampling::PolarMapping<float32_t>::generateInverse(output.mapped);
			output.backwardPdf = sampling::PolarMapping<float32_t>::backwardPdf(input.u);
		}
		float32_t2 diff = input.u - output.inverted;
		output.roundtripError = nbl::hlsl::length(diff);
		output.jacobianProduct = float32_t(1.0 / output.backwardPdf) * output.forwardPdf;
	}
};

#endif
