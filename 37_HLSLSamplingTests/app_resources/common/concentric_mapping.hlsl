#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_CONCENTRIC_MAPPING_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_CONCENTRIC_MAPPING_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/concentric_mapping.hlsl>

using namespace nbl::hlsl;

struct ConcentricMappingInputValues
{
	float32_t2 u;
};

struct ConcentricMappingTestResults
{
	float32_t2 mapped;
	float32_t2 inverted;
	float32_t cachedPdf;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t jacobianProduct;
	float32_t roundtripError;
};

struct ConcentricMappingTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(ConcentricMappingInputValues) input, NBL_REF_ARG(ConcentricMappingTestResults) output)
	{
		{
			sampling::ConcentricMapping<float32_t>::cache_type cache;
			output.mapped = sampling::ConcentricMapping<float32_t>::generate(input.u, cache);
			output.cachedPdf = cache.pdf;
			output.forwardPdf = sampling::ConcentricMapping<float32_t>::forwardPdf(cache);
		}
		{
			output.inverted = sampling::ConcentricMapping<float32_t>::generateInverse(output.mapped);
			output.backwardPdf = sampling::ConcentricMapping<float32_t>::backwardPdf(input.u);
		}
		float32_t2 diff = input.u - output.inverted;
		output.roundtripError = nbl::hlsl::length(diff);
		output.jacobianProduct = float32_t(1.0 / output.backwardPdf) * output.forwardPdf;	
	}
};

#endif
