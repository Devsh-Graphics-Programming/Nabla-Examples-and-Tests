#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>
#include "jacobian_test.hlsl"

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
	float32_t jacobianProduct;
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
		float32_t2 absXY;
		{
			sampling::SphericalRectangle<float32_t>::cache_type cache;
			absXY = sampler.generateLocalBasisXY(input.u, cache);
			output.surfaceOffset = absXY - float32_t2(sampler.r0.x, sampler.r0.y);
		}
		{
			const float32_t3 localDir = nbl::hlsl::normalize(float32_t3(absXY.x, absXY.y, sampler.r0.z));
			output.referenceDirection = sampler.basis[0] * localDir[0]
			                          + sampler.basis[1] * localDir[1]
			                          + sampler.basis[2] * localDir[2];
		}
		output.backwardPdf = sampler.backwardPdf(output.generated);
		output.backwardWeight = sampler.backwardWeight(output.generated);
		// marginFactor = 3: __generate's sin_au denominator goes through catastrophic cancellation
		// for u.x within ~2*eps of 0 or 1 (au near n*pi), leaving ~0.5% residual at factor 3.
		output.jacobianProduct = computeJacobianProduct<JACOBIAN_PLAIN>(sampler, input.u, 1e-3f, 3.0f);
	}
};

#endif
