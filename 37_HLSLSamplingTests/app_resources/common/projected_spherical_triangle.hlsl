#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>

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
	float32_t backwardPdf;
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
			output.forwardPdf = sampler.forwardPdf(cache);
		}
		// Test backwardPdf at the triangle centroid: a deterministic interior point computed
		// from only basic arithmetic + sqrt (IEEE 754 exact), so CPU and GPU agree bit-exactly.
		// Using output.generated would amplify generate's transcendental FP errors through
		// generateInverse's acos, producing ~0.005-0.01 CPU/GPU divergence.
		const float32_t3 center = nbl::hlsl::normalize(input.vertex0 + input.vertex1 + input.vertex2);
		output.backwardPdf = sampler.backwardPdf(center);
	}
};

#endif
