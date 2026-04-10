#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>

using namespace nbl::hlsl;

struct ProjectedSphericalRectangleInputValues
{
	float32_t3 observer;
	float32_t3 rectOrigin;
	float32_t3 right;
	float32_t3 receiverNormal;
	float32_t3 up;
	float32_t2 u;
	uint32_t receiverWasBSDF;
};

struct ProjectedSphericalRectangleTestResults
{
	float32_t2 generated;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t forwardWeight;
	float32_t backwardWeight;
	float32_t backwardPdfAtGenerated;
	float32_t backwardWeightAtGenerated;
	float32_t2 extents;
};

struct ProjectedSphericalRectangleTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(ProjectedSphericalRectangleInputValues) input, NBL_REF_ARG(ProjectedSphericalRectangleTestResults) output)
	{
		shapes::CompressedSphericalRectangle<float32_t> compressed;
		compressed.origin = input.rectOrigin;
		compressed.right = input.right;
		compressed.up = input.up;

		shapes::SphericalRectangle<float32_t> rect = shapes::SphericalRectangle<float32_t>::create(compressed);
		sampling::ProjectedSphericalRectangle<float32_t> sampler = sampling::ProjectedSphericalRectangle<float32_t>::create(rect, input.observer, input.receiverNormal, input.receiverWasBSDF);

		output.extents = rect.extents;
		sampling::ProjectedSphericalRectangle<float32_t>::cache_type cache;
		{
			output.generated = sampler.generate(input.u, cache);
			output.forwardPdf = sampler.forwardPdf(input.u, cache);
			output.forwardWeight = sampler.forwardWeight(input.u, cache);
		}
		// Test backwardPdf/Weight at the rect center: a deterministic interior point
		// that avoids amplifying generate's FP errors through backward evaluation.
		const float32_t2 center = float32_t2(0.5, 0.5);
		output.backwardPdf = sampler.backwardPdf(center);
		output.backwardWeight = sampler.backwardWeight(center);
		// Use cache.warped (the [0,1]^2 input to the spherical rect warp) for consistency
		// checks, NOT generated/extents (the nonlinear warp output). The bilinear in
		// forwardPdf evaluates at cache.warped, so backwardPdf must too.
		output.backwardPdfAtGenerated = sampler.backwardPdf(cache.warped);
		output.backwardWeightAtGenerated = sampler.backwardWeight(cache.warped);
	}
};

#endif
