//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BIQUADRATIC_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BIQUADRATIC_HLSL_INCLUDED_

// ============================================================================
// Biquadratic Approximation Sampling (Hart et al. 2020)
// ============================================================================
//
// Precomputed biquadratic sampler for importance sampling solid angle density.
// Build once from a SphericalPyramid, then call sample() per random pair.

struct BiquadraticSampler
{
    // Column-major: cols[i] = (row0[i], row1[i], row2[i]) for fast sliceAtY via dot
    float32_t3x3 cols;

    // Precomputed marginal (Y) polynomial: f(y) = c0 + y*(c1 + y*c2)
    float32_t margC0, margC1, margC2, margIntegral;

    float32_t rcpTotalIntegral;
    float32_t rcpIntegralTimesRcpArea; // rcpTotalIntegral / rectArea (fused for PDF computation)

    // Newton-Raphson CDF inversion for a quadratic PDF (2 iterations)
    // Solves: c0*t + (c1/2)*t^2 + (c2/3)*t^3 = u * integral
    // Returns sampled t and the PDF value at t (avoids redundant recomputation by caller).
    // 2 iterations give ~4 decimal digits, should be sufficient for importance sampling with rejection?
    static float32_t sampleQuadraticCDF(float32_t u, float32_t c0, float32_t c1, float32_t c2, float32_t integral, out float32_t lastPdfVal)
    {
        const float32_t target = u * integral;
        const float32_t c1half = c1 * 0.5f;
        const float32_t c2third = c2 * (1.0f / 3.0f);
        float32_t t = u;

        // Iteration 1
        float32_t cdfVal = t * (c0 + t * (c1half + t * c2third));
        lastPdfVal = c0 + t * (c1 + t * c2);
        t = clamp(t - (cdfVal - target) / lastPdfVal, 0.0f, 1.0f);

        // Iteration 2
        cdfVal = t * (c0 + t * (c1half + t * c2third));
        lastPdfVal = c0 + t * (c1 + t * c2);
        t = clamp(t - (cdfVal - target) / lastPdfVal, 0.0f, 1.0f);

        return t;
    }

    // Precompute biquadratic sampler from pyramid (call ONCE, reuse for all samples)
    static BiquadraticSampler create(NBL_CONST_REF_ARG(SphericalPyramid) pyramid)
    {
        BiquadraticSampler self;

        // 3x3 grid positions on the rectangle
        const float32_t x0 = pyramid.rectR0.x;
        const float32_t x1 = x0 + 0.5f * pyramid.rectExtents.x;
        const float32_t x2 = x0 + pyramid.rectExtents.x;
        const float32_t y0 = pyramid.rectR0.y;
        const float32_t y1 = y0 + 0.5f * pyramid.rectExtents.y;
        const float32_t y2 = y0 + pyramid.rectExtents.y;

        // dSA(x,y) = rsqrt(x^2+y^2+1) / (x^2+y^2+1)  [z = rectR0.z = 1.0]
        const float32_t xx0 = x0 * x0, xx1 = x1 * x1, xx2 = x2 * x2;
        const float32_t yy0 = y0 * y0, yy1 = y1 * y1, yy2 = y2 * y2;

        float32_t3 row0, row1, row2;
        float32_t d;

        d = xx0 + yy0 + 1.0f;
        row0.x = rsqrt(d) / d;
        d = xx1 + yy0 + 1.0f;
        row0.y = rsqrt(d) / d;
        d = xx2 + yy0 + 1.0f;
        row0.z = rsqrt(d) / d;

        d = xx0 + yy1 + 1.0f;
        row1.x = rsqrt(d) / d;
        d = xx1 + yy1 + 1.0f;
        row1.y = rsqrt(d) / d;
        d = xx2 + yy1 + 1.0f;
        row1.z = rsqrt(d) / d;

        d = xx0 + yy2 + 1.0f;
        row2.x = rsqrt(d) / d;
        d = xx1 + yy2 + 1.0f;
        row2.y = rsqrt(d) / d;
        d = xx2 + yy2 + 1.0f;
        row2.z = rsqrt(d) / d;

        // Store column-major for sliceAtY: cols[i] = (row0[i], row1[i], row2[i])
        self.cols[0] = float32_t3(row0.x, row1.x, row2.x);
        self.cols[1] = float32_t3(row0.y, row1.y, row2.y);
        self.cols[2] = float32_t3(row0.z, row1.z, row2.z);

        // Marginal along Y: Simpson's rule integral of each row
        const float32_t3 marginal = float32_t3(
            (row0.x + 4.0f * row0.y + row0.z) / 6.0f,
            (row1.x + 4.0f * row1.y + row1.z) / 6.0f,
            (row2.x + 4.0f * row2.y + row2.z) / 6.0f);

        // Precompute marginal polynomial: f(y) = c0 + y*(c1 + y*c2)
        self.margC0 = marginal[0];
        self.margC1 = -3.0f * marginal[0] + 4.0f * marginal[1] - marginal[2];
        self.margC2 = 2.0f * (marginal[0] - 2.0f * marginal[1] + marginal[2]);
        self.margIntegral = (marginal[0] + 4.0f * marginal[1] + marginal[2]) / 6.0f;

        self.rcpTotalIntegral = 1.0f / max(self.margIntegral, 1e-20f);
        const float32_t rectArea = pyramid.rectExtents.x * pyramid.rectExtents.y;
        self.rcpIntegralTimesRcpArea = self.rcpTotalIntegral / max(rectArea, 1e-20f);

        return self;
    }

    // Sample a direction on the spherical pyramid using biquadratic importance sampling.
    // Returns the world-space direction; outputs pdf in solid-angle space and validity flag.
    float32_t3 sample(NBL_CONST_REF_ARG(SphericalPyramid) pyramid, NBL_CONST_REF_ARG(SilEdgeNormals) silhouette, float32_t2 xi, out float32_t pdf, out bool valid)
    {
        // Step 1: Sample Y from precomputed marginal polynomial
        float32_t margPdfAtY;
        const float32_t y = sampleQuadraticCDF(xi.y, margC0, margC1, margC2, margIntegral, margPdfAtY);

        // Step 2: Compute conditional X slice at sampled Y via Lagrange basis
        const float32_t y2 = y * y;
        const float32_t3 Ly = float32_t3(2.0f * y2 - 3.0f * y + 1.0f, -4.0f * y2 + 4.0f * y, 2.0f * y2 - y);
        const float32_t3 slice = float32_t3(dot(cols[0], Ly), dot(cols[1], Ly), dot(cols[2], Ly));

        // Step 3: Build conditional polynomial and sample X
        const float32_t condC0 = slice[0];
        const float32_t condC1 = -3.0f * slice[0] + 4.0f * slice[1] - slice[2];
        const float32_t condC2 = 2.0f * (slice[0] - 2.0f * slice[1] + slice[2]);
        const float32_t condIntegral = (slice[0] + 4.0f * slice[1] + slice[2]) / 6.0f;
        float32_t condPdfAtX;
        const float32_t x = sampleQuadraticCDF(xi.x, condC0, condC1, condC2, condIntegral, condPdfAtX);

        // Step 4: UV to direction
        const float32_t localX = pyramid.rectR0.x + x * pyramid.rectExtents.x;
        const float32_t localY = pyramid.rectR0.y + y * pyramid.rectExtents.y;

        // Compute dist2 and rcpLen once, reuse for both normalization and dSA
        const float32_t dist2 = localX * localX + localY * localY + 1.0f;
        const float32_t rcpLen = rsqrt(dist2);
        float32_t3 direction = (localX * pyramid.axis1 +
                                localY * pyramid.axis2 +
                                pyramid.axis3) *
                               rcpLen;

        valid = direction.z > 0.0f && silhouette.isInside(direction);

        // Step 5: PDF in solid angle space = condPdfAtX / (totalIntegral * dSA * rectArea)
        // condPdfAtX is reused from the last Newton iteration
        const float32_t dsa = rcpLen / dist2;
        pdf = condPdfAtX * rcpIntegralTimesRcpArea / max(dsa, 1e-7f);

        return direction;
    }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BIQUADRATIC_HLSL_INCLUDED_
