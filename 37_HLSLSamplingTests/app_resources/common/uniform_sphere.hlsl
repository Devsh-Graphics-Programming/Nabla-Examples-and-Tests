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
	float32_t2 inverted;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t forwardWeight;
	float32_t backwardWeight;
	float32_t2 roundtripError;
	float32_t jacobianProduct;
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
		output.jacobianProduct = (float32_t(1.0) / output.forwardPdf) * output.backwardPdf;
	}
};

#endif
