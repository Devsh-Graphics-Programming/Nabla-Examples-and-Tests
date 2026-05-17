//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BILINEAR_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BILINEAR_HLSL_INCLUDED_
#include <nbl/builtin/hlsl/sampling/bilinear.hlsl>

// Bilinear gnomonic-rect sampler. Stores the pyramid's basis so generate()
// returns world-space dirs (matching SphericalRectangle's contract).
struct BilinearSampler
{
   using scalar_type    = float32_t;
   using vector2_type   = float32_t2;
   using vector3_type   = float32_t3;
   using matrix3x3_type = float32_t3x3;
   using domain_type    = vector2_type;
   using codomain_type  = vector3_type;
   using density_type   = scalar_type;
   using weight_type    = density_type;

   nbl::hlsl::sampling::Bilinear<float32_t> sampler;
   matrix3x3_type basis;
   float32_t2 rectR0;
   float32_t2 rectExtents;
   float32_t  rcpRectArea;

   struct cache_type
   {
      nbl::hlsl::sampling::Bilinear<float32_t>::cache_type bilinearCache;
      float32_t dist2;
      float32_t rcpLen;
   };

   static BilinearSampler create(matrix3x3_type basis, float32_t2 rectR0, float32_t2 rectExtents)
   {
      BilinearSampler self;
      self.basis = basis;

      // 4 corner positions on the rectangle
      const float32_t x0 = rectR0.x;
      const float32_t x1 = x0 + rectExtents.x;
      const float32_t y0 = rectR0.y;
      const float32_t y1 = y0 + rectExtents.y;

      // dSA(x,y) = 1 / (x^2 + y^2 + 1)^(3/2)  [z = 1.0 in local frame]
      const float32_t xx0 = x0 * x0, xx1 = x1 * x1;
      const float32_t yy0 = y0 * y0, yy1 = y1 * y1;

      // d^{-3/2} = rsqrt(d)^3: 1 rsqrt + 2 mul instead of 1 rsqrt + 1 div
      float32_t r;
      r = rsqrt(xx0 + yy0 + 1.0f);
      const float32_t v00 = r * r * r;
      r = rsqrt(xx1 + yy0 + 1.0f);
      const float32_t v10 = r * r * r;
      r = rsqrt(xx0 + yy1 + 1.0f);
      const float32_t v01 = r * r * r;
      r = rsqrt(xx1 + yy1 + 1.0f);
      const float32_t v11 = r * r * r;

      // Bilinear layout: (x0y0, x0y1, x1y0, x1y1)
      self.sampler     = nbl::hlsl::sampling::Bilinear<float32_t>::create(float32_t4(v00, v01, v10, v11));
      self.rectR0      = rectR0;
      self.rectExtents = rectExtents;
      self.rcpRectArea = rcp(max(rectExtents.x * rectExtents.y, 1e-20f));

      return self;
   }

   // Returns world-space unit direction; caches dist2 and rcpLen for forwardPdf.
   // Returns local-frame unit direction; caches dist2/rcpLen for forwardPdf.
   // hitDist == 1/rcpLen (the gnomonic ray length on the rect at z=1).
   codomain_type generateNormalizedLocal(domain_type u, NBL_REF_ARG(cache_type) cache, NBL_REF_ARG(scalar_type) hitDist)
   {
      const vector2_type uv     = sampler.generate(u, cache.bilinearCache);
      const scalar_type  localX = rectR0.x + uv.x * rectExtents.x;
      const scalar_type  localY = rectR0.y + uv.y * rectExtents.y;
      cache.dist2               = localX * localX + localY * localY + 1.0f;
      cache.rcpLen              = rsqrt(cache.dist2);
      hitDist                   = 1.0f / cache.rcpLen;
      return codomain_type(localX, localY, 1.0f) * cache.rcpLen;
   }

   codomain_type generate(domain_type u, NBL_REF_ARG(cache_type) cache)
   {
      scalar_type dummy;
      const vector3_type localDir = generateNormalizedLocal(u, cache, dummy);
      return basis[0] * localDir.x + basis[1] * localDir.y + basis[2] * localDir.z;
   }

   // Solid-angle-measure pdf: bilinearPdf * dist2^{3/2} * rcpRectArea.
   density_type forwardPdf(domain_type u, cache_type cache) NBL_CONST_MEMBER_FUNC
   {
      return sampler.forwardPdf(u, cache.bilinearCache) * cache.dist2 * cache.dist2 * cache.rcpLen * rcpRectArea;
   }

   weight_type forwardWeight(domain_type u, cache_type cache) NBL_CONST_MEMBER_FUNC
   {
      return forwardPdf(u, cache);
   }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_SAMPLING_BILINEAR_HLSL_INCLUDED_
