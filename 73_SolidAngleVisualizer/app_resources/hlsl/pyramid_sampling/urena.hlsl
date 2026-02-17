//// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_URENA_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_URENA_HLSL_INCLUDED_

// ============================================================================
// Sampling using Urena 2003 (SphericalRectangle)
// ============================================================================

struct UrenaSampler
{
    float32_t solidAngle; // Solid angle of the bounding region (steradians)
    float32_t samplerK;   // = 2*pi - q (angle offset for horizontal sampling)
    float32_t samplerB0;  // = n_z[0] (normalized edge parameter)
    float32_t samplerB1;  // = n_z[2] (normalized edge parameter)

    // Precompute solid angle AND sampler intermediates in one pass
    // (solidAngleOfRectangle and generate() both compute n_z/cosGamma -- fuse them)
    static UrenaSampler create(NBL_CONST_REF_ARG(SphericalPyramid) pyramid)
    {
        UrenaSampler self;

        const float32_t4 denorm_n_z = float32_t4(-pyramid.rectR0.y, pyramid.rectR0.x + pyramid.rectExtents.x, pyramid.rectR0.y + pyramid.rectExtents.y, -pyramid.rectR0.x);
        const float32_t4 n_z = denorm_n_z / sqrt((float32_t4)(pyramid.rectR0.z * pyramid.rectR0.z) + denorm_n_z * denorm_n_z);
        const float32_t4 cosGamma = float32_t4(-n_z[0] * n_z[1], -n_z[1] * n_z[2],
                                               -n_z[2] * n_z[3], -n_z[3] * n_z[0]);

        nbl::hlsl::math::sincos_accumulator<float32_t> adder = nbl::hlsl::math::sincos_accumulator<float32_t>::create(cosGamma[0]);
        adder.addCosine(cosGamma[1]);
        const float32_t p = adder.getSumofArccos();
        adder = nbl::hlsl::math::sincos_accumulator<float32_t>::create(cosGamma[2]);
        adder.addCosine(cosGamma[3]);
        const float32_t q = adder.getSumofArccos();

        self.solidAngle = p + q - 2.0f * nbl::hlsl::numbers::pi<float>;
        self.samplerK = 2.0f * nbl::hlsl::numbers::pi<float> - q;
        self.samplerB0 = n_z[0];
        self.samplerB1 = n_z[2];

        return self;
    }

    float32_t3 sample(NBL_CONST_REF_ARG(SphericalPyramid) pyramid, NBL_CONST_REF_ARG(SilEdgeNormals) silhouette, float32_t2 xi, out float32_t pdf, out bool valid)
    {
        // Inlined Urena 2003 with algebraic simplifications:
        const float32_t r1x = pyramid.rectR0.x + pyramid.rectExtents.x;
        const float32_t r1y = pyramid.rectR0.y + pyramid.rectExtents.y;

        // Horizontal CDF inversion
        const float32_t au = xi.x * solidAngle + samplerK;
        float32_t sinAu, cosAu;
        sincos(au, sinAu, cosAu);
        const float32_t fu = (cosAu * samplerB0 - samplerB1) / sinAu;

        // cu = sign(fu)/sqrt(cu_2), xu = cu/sqrt(1-cu^2)
        // Fused: xu = sign(fu)/sqrt(cu_2 - 1)  [eliminates 2 sqrt + 2 div -> 1 rsqrt]
        const float32_t cu_2 = max(fu * fu + samplerB0 * samplerB0, 1.0f);
        const float32_t xu = clamp(
            (fu >= 0.0f ? 1.0f : -1.0f) * rsqrt(max(cu_2 - 1.0f, 1e-10f)),
            pyramid.rectR0.x, r1x);
        const float32_t d_2 = xu * xu + 1.0f;

        // Vertical sampling in h-space (div -> rsqrt + mul)
        const float32_t h0 = pyramid.rectR0.y * rsqrt(d_2 + pyramid.rectR0.y * pyramid.rectR0.y);
        const float32_t h1 = r1y * rsqrt(d_2 + r1y * r1y);
        const float32_t hv = h0 + xi.y * (h1 - h0);

        // Normalized direction via ||(xu,yv,1)||^2 = d_2/(1-hv^2):
        //   localDir.y = yv/||v|| = hv   (exact cancellation)
        //   localDir.xz = (xu, 1) * t    where t = sqrt(1-hv^2)/sqrt(d_2)
        // Eliminates: sqrt(d_2), yv computation, and normalize()
        const float32_t t = sqrt(max(1.0f - hv * hv, 0.0f)) * rsqrt(d_2);
        const float32_t3 localDir = float32_t3(xu * t, hv, t);

        float32_t3 direction = localDir.x * pyramid.axis1 +
                               localDir.y * pyramid.axis2 +
                               localDir.z * pyramid.axis3;

        valid = direction.z > 0.0f && silhouette.isInside(direction);
        pdf = 1.0f / max(solidAngle, 1e-7f);

        return direction;
    }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_URENA_HLSL_INCLUDED_
