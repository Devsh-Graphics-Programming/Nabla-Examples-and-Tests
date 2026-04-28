//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BILINEAR_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BILINEAR_HLSL_INCLUDED_
#include <nbl/builtin/hlsl/sampling/bilinear.hlsl>

// ============================================================================
// Bilinear Approximation Sampling (closed-form, faster than biquadratic)
// ============================================================================
//
struct BilinearSampler
{
	nbl::hlsl::sampling::Bilinear<float32_t> sampler;

	float32_t rcpRectArea;

	// Precompute bilinear sampler from pyramid
	static BilinearSampler create(NBL_CONST_REF_ARG(SphericalPyramid) pyramid)
	{
		BilinearSampler self;

		// 4 corner positions on the rectangle
		const float32_t x0 = pyramid.rectR0.x;
		const float32_t x1 = x0 + pyramid.rectExtents.x;
		const float32_t y0 = pyramid.rectR0.y;
		const float32_t y1 = y0 + pyramid.rectExtents.y;

		// dSA(x,y) = 1 / (x^2 + y^2 + 1)^(3/2)  [z = 1.0 in local frame]
		const float32_t xx0 = x0 * x0, xx1 = x1 * x1;
		const float32_t yy0 = y0 * y0, yy1 = y1 * y1;

		// d^{-3/2} = rsqrt(d)^3: 1 rsqrt + 2 mul instead of 1 rsqrt + 1 div
		float32_t r;
		r = rsqrt(xx0 + yy0 + 1.0f);
		const float32_t v00 = r * r * r; // x0y0
		r = rsqrt(xx1 + yy0 + 1.0f);
		const float32_t v10 = r * r * r; // x1y0
		r = rsqrt(xx0 + yy1 + 1.0f);
		const float32_t v01 = r * r * r; // x0y1
		r = rsqrt(xx1 + yy1 + 1.0f);
		const float32_t v11 = r * r * r; // x1y1

		// Bilinear layout: (x0y0, x0y1, x1y0, x1y1)
		self.sampler = nbl::hlsl::sampling::Bilinear<float32_t>::create(float32_t4(v00, v01, v10, v11));
		self.rcpRectArea = rcp(max(pyramid.rectExtents.x * pyramid.rectExtents.y, 1e-20f));

		return self;
	}

	// Sample a direction on the spherical pyramid using bilinear importance sampling.
	// Returns the world-space direction; outputs pdf in solid-angle space and validity flag.
	float32_t3 sample(NBL_CONST_REF_ARG(SphericalPyramid) pyramid, NBL_CONST_REF_ARG(SilEdgeNormals) silEdgeNormals, float32_t2 xi, out float32_t pdf, out bool valid)
	{
		nbl::hlsl::sampling::Bilinear<float32_t>::cache_type cache;
		float32_t2 uv = sampler.generate(xi, cache);

		const float32_t localX = pyramid.rectR0.x + uv.x * pyramid.rectExtents.x;
		const float32_t localY = pyramid.rectR0.y + uv.y * pyramid.rectExtents.y;

		const float32_t dist2 = localX * localX + localY * localY + 1.0f;
		const float32_t rcpLen = rsqrt(dist2);
		float32_t3 direction = (localX * pyramid.axis1 +
								localY * pyramid.axis2 +
								pyramid.getAxis3()) * rcpLen;

		valid = direction.z > 0.0f && silEdgeNormals.isInsideLocal(localX, localY);

		// PDF in solid angle space: pdfBilinear * dist2^{3/2} * rcpRectArea
		pdf = sampler.forwardPdf(xi, cache) * dist2 * dist2 * rcpLen * rcpRectArea;

		return direction;
	}
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BILINEAR_HLSL_INCLUDED_
