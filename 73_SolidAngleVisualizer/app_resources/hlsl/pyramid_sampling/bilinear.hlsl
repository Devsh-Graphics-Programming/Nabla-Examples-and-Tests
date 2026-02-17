//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
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

	float32_t rcpTotalIntegral;
	float32_t rectArea;

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

		float32_t d;
		d = xx0 + yy0 + 1.0f;
		const float32_t v00 = rsqrt(d) / d; // x0y0
		d = xx1 + yy0 + 1.0f;
		const float32_t v10 = rsqrt(d) / d; // x1y0
		d = xx0 + yy1 + 1.0f;
		const float32_t v01 = rsqrt(d) / d; // x0y1
		d = xx1 + yy1 + 1.0f;
		const float32_t v11 = rsqrt(d) / d; // x1y1

		// Bilinear layout: (x0y0, x0y1, x1y0, x1y1)
		self.sampler = nbl::hlsl::sampling::Bilinear<float32_t>::create(float32_t4(v00, v01, v10, v11));

		// Total integral = average of 4 corners (bilinear integral over unit square)
		const float32_t totalIntegral = (v00 + v10 + v01 + v11) * 0.25f;
		self.rcpTotalIntegral = 1.0f / max(totalIntegral, 1e-20f);
		self.rectArea = pyramid.rectExtents.x * pyramid.rectExtents.y;

		return self;
	}

	// Sample a direction on the spherical pyramid using bilinear importance sampling.
	// Returns the world-space direction; outputs pdf in solid-angle space and validity flag.
	float32_t3 sample(NBL_CONST_REF_ARG(SphericalPyramid) pyramid, NBL_CONST_REF_ARG(SilEdgeNormals) silhouette, float32_t2 xi, out float32_t pdf, out bool valid)
	{
		// Step 1: Sample UV from bilinear distribution (closed-form via quadratic formula)
		float32_t rcpPdf;
		float32_t2 uv = sampler.generate(rcpPdf, xi);

		// Step 2: UV to direction
		// Bilinear sampler convention: u.y = first-sampled axis (X), u.x = second-sampled axis (Y)
		const float32_t localX = pyramid.rectR0.x + uv.y * pyramid.rectExtents.x;
		const float32_t localY = pyramid.rectR0.y + uv.x * pyramid.rectExtents.y;

		// Compute dist2 and rcpLen once, reuse for both normalization and dSA
		const float32_t dist2 = localX * localX + localY * localY + 1.0f;
		const float32_t rcpLen = rsqrt(dist2);
		float32_t3 direction = (localX * pyramid.axis1 +
								localY * pyramid.axis2 +
								pyramid.axis3) * rcpLen;

		valid = direction.z > 0.0f && silhouette.isInside(direction);

		// PDF in solid angle space: 1 / (rcpPdf * dSA * rectArea)
		// rcpPdf already = 1/pdfUV from Bilinear::generate, avoid redundant reciprocal
		const float32_t dsa = rcpLen / dist2;
		pdf = 1.0f / max(rcpPdf * dsa * rectArea, 1e-7f);

		return direction;
	}
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BILINEAR_HLSL_INCLUDED_
