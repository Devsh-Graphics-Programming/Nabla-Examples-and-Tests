#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_CONCENTRIC_MAPPING_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_CONCENTRIC_MAPPING_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/concentric_mapping.hlsl>
#include "jacobian_test.hlsl"

using namespace nbl::hlsl;

struct ConcentricMappingInputValues
{
	float32_t2 u;
};

struct ConcentricMappingTestResults
{
	float32_t2 mapped;
	float32_t2 inverted;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t forwardWeight;
	float32_t backwardWeight;
	float32_t jacobianProduct;
	float32_t inverseJacobianPdf;
	float32_t2 roundtripError;
};

struct ConcentricMappingTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(ConcentricMappingInputValues) input, NBL_REF_ARG(ConcentricMappingTestResults) output)
	{
		{
			sampling::ConcentricMapping<float32_t>::cache_type cache;
			output.mapped = sampling::ConcentricMapping<float32_t>::generate(input.u, cache);
			output.forwardPdf = sampling::ConcentricMapping<float32_t>::forwardPdf(output.mapped, cache);
			output.forwardWeight = sampling::ConcentricMapping<float32_t>::forwardWeight(output.mapped, cache);
		}
		{
			output.inverted = sampling::ConcentricMapping<float32_t>::generateInverse(output.mapped);
			output.backwardPdf = sampling::ConcentricMapping<float32_t>::backwardPdf(input.u);
			output.backwardWeight = sampling::ConcentricMapping<float32_t>::backwardWeight(input.u);
		}
		output.roundtripError = nbl::hlsl::abs(input.u - output.inverted);
		{
			sampling::ConcentricMapping<float32_t> sampler;
			output.jacobianProduct = computeJacobianProduct<JACOBIAN_CONCENTRIC>(sampler, input.u, 1e-3f, 1.0f);
			// Disk-center singularity: concentric atan2 blows up as r->0.
			const float32_t diskRadius = nbl::hlsl::length(output.mapped);
			output.inverseJacobianPdf = diskRadius < 0.1f
				? JACOBIAN_SKIP_CODOMAIN_SINGULARITY
				: computeInverseJacobianPdf(sampler, output.mapped, output.backwardPdf, 0.0f, 1e30f);
		}
	}
};

#endif
