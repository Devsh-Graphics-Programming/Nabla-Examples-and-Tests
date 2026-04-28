//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BIQUADRATIC_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BIQUADRATIC_HLSL_INCLUDED_
#include <nbl/builtin/hlsl/sampling/bilinear.hlsl> // reuse basic structure

// ============================================================================
// Biquadratic Approximation Sampling (cheap solid-angle approximation)
// ============================================================================
struct BiquadraticSampler
{
    nbl::hlsl::sampling::Bilinear<float32_t> baseSampler; // underlying bilinear generator

    float32_t rcpRectArea;

    // Precompute biquadratic sampler from pyramid
    static BiquadraticSampler create(NBL_CONST_REF_ARG(SphericalPyramid) pyramid)
    {
        BiquadraticSampler self;

        // 4 corner positions on the rectangle
        const float32_t x0 = pyramid.rectR0.x;
        const float32_t x1 = x0 + pyramid.rectExtents.x;
        const float32_t y0 = pyramid.rectR0.y;
        const float32_t y1 = y0 + pyramid.rectExtents.y;

        // Compute solid-angle weights at corners: d^{-3/2}
        const float32_t xx0 = x0 * x0, xx1 = x1 * x1;
        const float32_t yy0 = y0 * y0, yy1 = y1 * y1;

        // d^{-3/2} = rsqrt(d)^3
        float32_t r;
        r = rsqrt(xx0 + yy0 + 1.0f);
        const float32_t v00 = r * r * r;
        r = rsqrt(xx1 + yy0 + 1.0f);
        const float32_t v10 = r * r * r;
        r = rsqrt(xx0 + yy1 + 1.0f);
        const float32_t v01 = r * r * r;
        r = rsqrt(xx1 + yy1 + 1.0f);
        const float32_t v11 = r * r * r;

        self.baseSampler = nbl::hlsl::sampling::Bilinear<float32_t>::create(float32_t4(v00, v01, v10, v11));
        self.rcpRectArea = rcp(max(pyramid.rectExtents.x * pyramid.rectExtents.y, 1e-20f));

        return self;
    }

    // Sample a direction on the spherical pyramid using biquadratic importance sampling.
    // Applies a quadratic warp f(t) = t*(2-t) after bilinear sampling to redistribute
    // samples. The warp Jacobian f'(t) = 2*(1-t) is accounted for in the PDF.
    float32_t3 sample(NBL_CONST_REF_ARG(SphericalPyramid) pyramid, NBL_CONST_REF_ARG(SilEdgeNormals) silEdgeNormals, float32_t2 xi, out float32_t pdf, out bool valid)
    {
        nbl::hlsl::sampling::Bilinear<float32_t>::cache_type cache;
        float32_t2 uv = baseSampler.generate(xi, cache);

        // Quadratic warp: f(t) = t * (2 - t), f'(t) = 2 * (1 - t)
        const float32_t rcpWarpJacobian = rcp(4.0f * (1.0f - uv.x) * (1.0f - uv.y));
        uv = float32_t2(uv.x * (2.0f - uv.x), uv.y * (2.0f - uv.y));

        const float32_t localX = pyramid.rectR0.x + uv.y * pyramid.rectExtents.x;
        const float32_t localY = pyramid.rectR0.y + uv.x * pyramid.rectExtents.y;

        const float32_t dist2 = localX * localX + localY * localY + 1.0f;
        const float32_t rcpLen = rsqrt(dist2);
        float32_t3 direction = (localX * pyramid.axis1 +
                                localY * pyramid.axis2 +
                                pyramid.getAxis3()) *
                               rcpLen;

        valid = direction.z > 0.0f && silEdgeNormals.isInsideLocal(localX, localY);

        // PDF in solid-angle space, accounting for warp Jacobian
        pdf = baseSampler.forwardPdf(xi, cache) * dist2 * dist2 * rcpLen * rcpRectArea * rcpWarpJacobian;

        return direction;
    }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BIQUADRATIC_HLSL_INCLUDED_
