#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>
#include "jacobian_test.hlsl"

using namespace nbl::hlsl;

struct ProjectedSphericalTriangleInputValues
{
	float32_t3 vertex0;
	float32_t3 vertex1;
	float32_t3 vertex2;
	float32_t3 receiverNormal;
	uint32_t receiverWasBSDF;
	float32_t2 u;
};

struct ProjectedSphericalTriangleTestResults
{
	float32_t3 generated;
	float32_t forwardPdf;
	float32_t forwardWeight;
	float32_t backwardWeight;
	float32_t backwardWeightAtGenerated;
	float32_t jacobianProduct;
};

struct ProjectedSphericalTriangleTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(ProjectedSphericalTriangleInputValues) input, NBL_REF_ARG(ProjectedSphericalTriangleTestResults) output)
	{
		const float32_t3 verts[3] = { input.vertex0, input.vertex1, input.vertex2 };
		shapes::SphericalTriangle<float32_t> shape = shapes::SphericalTriangle<float32_t>::createFromUnitSphereVertices(verts);

		sampling::ProjectedSphericalTriangle<float32_t> sampler = sampling::ProjectedSphericalTriangle<float32_t>::create(shape, input.receiverNormal, (bool)input.receiverWasBSDF);

		{
			sampling::ProjectedSphericalTriangle<float32_t>::cache_type cache;
			output.generated = sampler.generate(input.u, cache);
			output.forwardPdf = sampler.forwardPdf(input.u, cache);
			output.forwardWeight = sampler.forwardWeight(input.u, cache);
		}
		const float32_t3 center = nbl::hlsl::normalize(input.vertex0 + input.vertex1 + input.vertex2);
		output.backwardWeight = sampler.backwardWeight(center);
		output.backwardWeightAtGenerated = sampler.backwardWeight(output.generated);
		// Check the bilinear-warped (inner) u directly: for skinny triangles with a strongly biased
		// receiver normal, outer u well inside [0,1] can still warp to inner u <~ 0.02 where Arvo's
		// sqrt(sinZ) noise dominates. Pre-skip on the inner u instead of padding an outer marginFactor.
		sampling::Bilinear<float32_t>::cache_type bc;
		const float32_t2 innerU = sampler.bilinearPatch.generate(input.u, bc);
		const float32_t innerMargin = 0.02f;
		const bool innerNearEdge = innerU.x < innerMargin || innerU.x > (1.0f - innerMargin)
		                        || innerU.y < innerMargin || innerU.y > (1.0f - innerMargin);
		output.jacobianProduct = innerNearEdge
			? JACOBIAN_SKIP_U_DOMAIN
			: computeJacobianProduct<JACOBIAN_PLAIN>(sampler, input.u, 1e-3f, 1.0f);
	}
};

#endif
