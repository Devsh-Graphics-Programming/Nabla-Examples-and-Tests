#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_SPHERICAL_TRIANGLE_JACOBIAN_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_SPHERICAL_TRIANGLE_JACOBIAN_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_triangle.hlsl>

using namespace nbl::hlsl;

struct SphericalTriangleJacobianInputValues
{
	float32_t3 vertex0;
	float32_t3 vertex1;
	float32_t3 vertex2;
	float32_t2 u;
};

struct SphericalTriangleJacobianTestResults
{
	float32_t3 generated;
	float32_t forwardRcpPdf;
	float32_t2 inverted;
	float32_t inversePdf;
	float32_t roundtripError;
	float32_t jacobianProduct;
	// Minimum signed distance to a triangle edge (sin of angular distance to nearest great circle).
	// Positive = inside, negative = outside. Allows tolerance at boundaries.
	float32_t generatedInside;
	// Minimum margin to the [0,1]^2 boundary: min(u.x, 1-u.x, u.y, 1-u.y).
	// Positive = inside, negative = outside.
	float32_t invertedInDomain;
};

struct SphericalTriangleJacobianTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(SphericalTriangleJacobianInputValues) input, NBL_REF_ARG(SphericalTriangleJacobianTestResults) output)
	{
		shapes::SphericalTriangle<float32_t> shape;
		shape.vertex0 = input.vertex0;
		shape.vertex1 = input.vertex1;
		shape.vertex2 = input.vertex2;
		shape.cos_sides = float32_t3(
			nbl::hlsl::dot(input.vertex1, input.vertex2),
			nbl::hlsl::dot(input.vertex2, input.vertex0),
			nbl::hlsl::dot(input.vertex0, input.vertex1));
		float32_t3 csc_sides2 = float32_t3(1.0, 1.0, 1.0) - shape.cos_sides * shape.cos_sides;
		shape.csc_sides = float32_t3(
			nbl::hlsl::rsqrt(csc_sides2.x),
			nbl::hlsl::rsqrt(csc_sides2.y),
			nbl::hlsl::rsqrt(csc_sides2.z));

		sampling::SphericalTriangle<float32_t> sampler = sampling::SphericalTriangle<float32_t>::create(shape);

		// Forward: u -> v
		output.generated = sampler.generate(output.forwardRcpPdf, input.u);

		// Inverse: v -> u'
		output.inverted = sampler.generateInverse(output.inversePdf, output.generated);

		// Roundtrip error: ||u - u'||
		float32_t2 diff = input.u - output.inverted;
		output.roundtripError = nbl::hlsl::length(diff);

		// Jacobian product: rcpPdf * pdf should equal 1 for bijective samplers
		output.jacobianProduct = output.forwardRcpPdf * output.inversePdf;

		// Domain preservation:
		// A point is inside the spherical triangle iff it is on the "inside" half-plane
		// of every edge. The orientation of the triangle (CCW vs CW) is given by the
		// sign of the scalar triple product dot(v0, cross(v1, v2)).
		float32_t3 e01 = nbl::hlsl::cross(input.vertex0, input.vertex1);
		float32_t3 e12 = nbl::hlsl::cross(input.vertex1, input.vertex2);
		float32_t3 e20 = nbl::hlsl::cross(input.vertex2, input.vertex0);
		// Normalize by edge lengths so the value is the sine of the angular distance
		// to the nearest great-circle edge (positive = inside, negative = outside).
		float32_t orientation = nbl::hlsl::dot(input.vertex0, e12);
		float32_t sinDist01 = nbl::hlsl::dot(output.generated, e01) * orientation * shape.csc_sides.z;
		float32_t sinDist12 = nbl::hlsl::dot(output.generated, e12) * orientation * shape.csc_sides.x;
		float32_t sinDist20 = nbl::hlsl::dot(output.generated, e20) * orientation * shape.csc_sides.y;
		output.generatedInside = nbl::hlsl::min(nbl::hlsl::min(sinDist01, sinDist12), sinDist20);

		float32_t2 u = output.inverted;
		output.invertedInDomain = nbl::hlsl::min(nbl::hlsl::min(u.x, float32_t(1.0) - u.x), nbl::hlsl::min(u.y, float32_t(1.0) - u.y));
	}
};

#endif
