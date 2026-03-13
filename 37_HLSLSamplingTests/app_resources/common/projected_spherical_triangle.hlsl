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
	float32_t cachedPdf;
	float32_t forwardPdf;
	float32_t backwardPdf;
};

struct ProjectedSphericalTriangleTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(ProjectedSphericalTriangleInputValues) input, NBL_REF_ARG(ProjectedSphericalTriangleTestResults) output)
	{
		shapes::SphericalTriangle<float32_t> shape;
		shape.vertices[0] = input.vertex0;
		shape.vertices[1] = input.vertex1;
		shape.vertices[2] = input.vertex2;
		shape.cos_sides = float32_t3(
			nbl::hlsl::dot(input.vertex1, input.vertex2),
			nbl::hlsl::dot(input.vertex2, input.vertex0),
			nbl::hlsl::dot(input.vertex0, input.vertex1));
		float32_t3 csc_sides2 = float32_t3(1.0, 1.0, 1.0) - shape.cos_sides * shape.cos_sides;
		shape.csc_sides = float32_t3(
			nbl::hlsl::rsqrt(csc_sides2.x),
			nbl::hlsl::rsqrt(csc_sides2.y),
			nbl::hlsl::rsqrt(csc_sides2.z));

		sampling::SphericalTriangle<float32_t> sphtri = sampling::SphericalTriangle<float32_t>::create(shape);

		sampling::ProjectedSphericalTriangle<float32_t> sampler;
		sampler.sphtri = sphtri;
		sampler.receiverNormal = input.receiverNormal;
		sampler.receiverWasBSDF = (bool)input.receiverWasBSDF;

		{
			sampling::ProjectedSphericalTriangle<float32_t>::cache_type cache;
			output.generated = sampler.generate(input.u, cache);
			output.cachedPdf = cache.pdf;
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
