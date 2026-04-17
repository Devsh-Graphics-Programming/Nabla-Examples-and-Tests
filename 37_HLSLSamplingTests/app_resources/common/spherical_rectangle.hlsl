#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>

using namespace nbl::hlsl;

struct SphericalRectangleInputValues
{
	float32_t3 observer;
	float32_t3 rectOrigin;
	float32_t3 right;
	float32_t3 up;
	float32_t2 u;
};

struct SphericalRectangleTestResults
{
	float32_t3 generated;
	float32_t2 surfaceOffset;
	float32_t3 referenceDirection;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t forwardWeight;
	float32_t backwardWeight;
	float32_t2 extents;
};

struct SphericalRectangleTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(SphericalRectangleInputValues) input, NBL_REF_ARG(SphericalRectangleTestResults) output)
	{
		shapes::CompressedSphericalRectangle<float32_t> compressed;
		compressed.origin = input.rectOrigin;
		compressed.right = input.right;
		compressed.up = input.up;

		shapes::SphericalRectangle<float32_t> rect = shapes::SphericalRectangle<float32_t>::create(compressed);
		sampling::SphericalRectangle<float32_t> sampler = sampling::SphericalRectangle<float32_t>::create(rect, input.observer);

		output.extents = rect.extents;
		{
			sampling::SphericalRectangle<float32_t>::cache_type cache;
			output.generated = sampler.generate(input.u, cache);
			output.forwardPdf = sampler.forwardPdf(input.u, cache);
			output.forwardWeight = sampler.forwardWeight(input.u, cache);
		}
		{
			sampling::SphericalRectangle<float32_t>::cache_type cache;
			output.surfaceOffset = sampler.generateSurfaceOffset(input.u, cache);
		}
		// reference direction: reconstruct local 3D point from surfaceOffset and normalize
		{
			const float32_t3 localPoint = sampler.r0 + float32_t3(output.surfaceOffset.x, output.surfaceOffset.y, float32_t(0));
			output.referenceDirection = nbl::hlsl::normalize(localPoint);
		}
		output.backwardPdf = sampler.backwardPdf(output.generated);
		output.backwardWeight = sampler.backwardWeight(output.generated);
	}
};

#endif
