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
	float32_t cachedPdf;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t roundtripError;
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
			output.cachedPdf = cache.pdf;
			output.forwardPdf = sampler.forwardPdf(cache);
		}

		{
			sampling::UniformSphere<float32_t>::cache_type cache;
			output.inverted = sampler.generateInverse(output.generated, cache);
			output.backwardPdf = sampler.backwardPdf(output.generated);
		}
		float32_t2 diff = input.u - output.inverted;
		output.roundtripError = nbl::hlsl::length(diff);
		output.jacobianProduct = (float32_t(1.0) / output.forwardPdf) * output.backwardPdf;
	}
};

#endif
