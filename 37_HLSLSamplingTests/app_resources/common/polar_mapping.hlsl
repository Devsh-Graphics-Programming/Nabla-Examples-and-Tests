#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_POLAR_MAPPING_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_POLAR_MAPPING_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/polar_mapping.hlsl>
#include "jacobian_test.hlsl"

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
	float32_t forwardWeight;
	float32_t backwardWeight;
	float32_t jacobianProduct;
	float32_t inverseJacobianPdf;
	float32_t2 roundtripError;
};

struct PolarMappingTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(PolarMappingInputValues) input, NBL_REF_ARG(PolarMappingTestResults) output)
	{
		{
			sampling::PolarMapping<float32_t>::cache_type cache;
			output.mapped = sampling::PolarMapping<float32_t>::generate(input.u, cache);
			output.forwardPdf = sampling::PolarMapping<float32_t>::forwardPdf(output.mapped, cache);
			output.forwardWeight = sampling::PolarMapping<float32_t>::forwardWeight(output.mapped, cache);
		}
		{
			output.inverted = sampling::PolarMapping<float32_t>::generateInverse(output.mapped);
			output.backwardPdf = sampling::PolarMapping<float32_t>::backwardPdf(input.u);
			output.backwardWeight = sampling::PolarMapping<float32_t>::backwardWeight(input.u);
		}
		output.roundtripError = nbl::hlsl::abs(input.u - output.inverted);

		{
			sampling::PolarMapping<float32_t> sampler;
			// marginFactor = 3: r = sqrt(u.x) gives O(h/u.x) forward-diff bias near u.x=0, so skip
			// u.x within 3*eps of the domain boundary (same reasoning as Linear's skewed-density case).
			output.jacobianProduct = computeJacobianProduct<JACOBIAN_PLAIN>(sampler, input.u, 1e-3f, 3.0f);
			// Two inverse singularities:
			//  - disk center: atan2 diverges as r -> 0
			//  - atan2 branch cut at y=0, x>0: the stencil's +/-eps in y straddles the 2*pi wrap,
			//    producing du.y/eps ~ 1/eps spikes (seen as test values ~305-862 with eps=1e-3).
			const float32_t polarRadius = nbl::hlsl::length(output.mapped);
			const bool onCutBand = nbl::hlsl::abs(output.mapped.y) < 5e-3f && output.mapped.x > 0.0f;
			output.inverseJacobianPdf = (polarRadius < 0.1f || onCutBand)
				? JACOBIAN_SKIP_CODOMAIN_SINGULARITY
				: computeInverseJacobianPdf(sampler, output.mapped, output.backwardPdf, 0.0f, 1e30f);
		}

	}
};

#endif
