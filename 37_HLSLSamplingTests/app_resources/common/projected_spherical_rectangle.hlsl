#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>
#include "jacobian_test.hlsl"

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
	float32_t3 generated;
	float32_t2 surfaceOffset;
	float32_t3 referenceDirection;
	float32_t forwardPdf;
	float32_t forwardWeight;
	float32_t backwardWeight;
	float32_t2 extents;
	float32_t jacobianProduct;
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
		output.generated = sampler.generate(input.u, cache);
		output.forwardPdf = sampler.forwardPdf(input.u, cache);
		output.forwardWeight = sampler.forwardWeight(input.u, cache);
		// backwardWeight now takes a 3D direction; evaluate at generated L.
		output.backwardWeight = sampler.backwardWeight(output.generated);

		float32_t2 absXY;
		{
			typename sampling::Bilinear<float32_t>::cache_type bc;
			const float32_t2 warped = sampler.bilinearPatch.generate(input.u, bc);
			typename sampling::SphericalRectangle<float32_t>::cache_type sphrectCache;
			absXY = sampler.sphrect.generateLocalBasisXY(warped, sphrectCache);
			output.surfaceOffset = absXY - float32_t2(sampler.sphrect.r0.x, sampler.sphrect.r0.y);
		}
		{
			const float32_t3 localPoint = float32_t3(absXY.x, absXY.y, sampler.sphrect.r0.z);
			const float32_t3 localDir = nbl::hlsl::normalize(localPoint);
			output.referenceDirection = sampler.sphrect.basis[0] * localDir[0]
			                          + sampler.sphrect.basis[1] * localDir[1]
			                          + sampler.sphrect.basis[2] * localDir[2];
		}

		output.jacobianProduct = computeJacobianProduct<JACOBIAN_PLAIN>(sampler, input.u, 1e-3f, 10.0f);
	}
};

#endif
