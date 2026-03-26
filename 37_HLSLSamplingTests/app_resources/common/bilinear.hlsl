#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_BILINEAR_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_BILINEAR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/bilinear.hlsl>

using namespace nbl::hlsl;

struct BilinearInputValues
{
	float32_t4 bilinearCoeffs;
	float32_t2 u;
};

struct BilinearTestResults
{
	float32_t2 generated;
	float32_t backwardPdf;
	float32_t forwardPdf;
};

struct BilinearTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(BilinearInputValues) input, NBL_REF_ARG(BilinearTestResults) output)
	{
		sampling::Bilinear<float32_t> sampler = sampling::Bilinear<float32_t>::create(input.bilinearCoeffs);
		{
			sampling::Bilinear<float32_t>::cache_type cache;
			output.generated = sampler.generate(input.u, cache);
			output.forwardPdf = sampler.forwardPdf(cache);
		}

		{
			output.backwardPdf = sampler.backwardPdf(output.generated);
		}
	}
};

#endif
