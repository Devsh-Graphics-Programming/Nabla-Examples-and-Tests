#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_UNIFORM_SPHERE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_UNIFORM_SPHERE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/uniform_spheres.hlsl>
#include "jacobian_test.hlsl"

using namespace nbl::hlsl;

struct UniformSphereInputValues
{
	float32_t2 u;
};

struct UniformSphereTestResults
{
	float32_t3 generated;
	float32_t pdf;
	float32_t2 inverted;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t forwardWeight;
	float32_t backwardWeight;
	float32_t2 roundtripError;
	float32_t jacobianProduct;
	float32_t inverseJacobianPdf;
};

struct UniformSphereTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(UniformSphereInputValues) input, NBL_REF_ARG(UniformSphereTestResults) output)
	{
		sampling::UniformSphere<float32_t> sampler;
		{
			sampling::UniformSphere<float32_t>::cache_type cache;
			output.generated = sampler.generate(input.u, cache);
			output.forwardPdf = sampler.forwardPdf(input.u, cache);
			output.forwardWeight = sampler.forwardWeight(input.u, cache);
		}

		{
			sampling::UniformSphere<float32_t>::cache_type cache;
			output.inverted = sampler.generateInverse(output.generated);
			output.backwardPdf = sampler.backwardPdf(output.generated);
			output.backwardWeight = sampler.backwardWeight(output.generated);
		}
		output.roundtripError = nbl::hlsl::abs(input.u - output.inverted);
		output.jacobianProduct = computeJacobianProduct<JACOBIAN_CONCENTRIC_UXFOLD>(sampler, input.u, 1e-3f, 1.0f);
		const float32_t usDiskR = nbl::hlsl::length((float32_t2)output.generated);
		const float32_t absZ    = nbl::hlsl::abs(output.generated.z);
		output.inverseJacobianPdf = (absZ < 0.1f || usDiskR < 0.1f)
			? JACOBIAN_SKIP_CODOMAIN_SINGULARITY
			: computeInverseJacobianPdf(sampler, output.generated, output.backwardPdf, 0.0f, 1e30f);
	}
};

#endif
