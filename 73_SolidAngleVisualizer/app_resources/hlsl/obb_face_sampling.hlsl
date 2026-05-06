//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_OBB_FACE_SAMPLING_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_OBB_FACE_SAMPLING_HLSL_INCLUDED_

#include "common.hlsl"
#include "silhouette.hlsl" // for the (silhouette, view) overload's signature

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/shapes/obb.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>

// Multi-face OBB sampler -- Matt's design with shared tip vertex T as origin
// and silhouette pipeline skipped entirely. NO horizon clipping (option A):
// samples below z=0 just get pdf=0, biased for OBBs near receiver horizon.
//
// This is the best OBB-faces variant we measured (~92 ps @ 1:1, ~22 ps @ 1:16,
// ~17 ps @ 1:128). Still slower than PYRAMID_RECTANGLE on this Ampere SM at
// every ratio. Kept around as a documented baseline for future experiments
// (e.g. Las Vegas resampling, different inner samplers, fp16 packing) where
// the no-clipping property might justify the per-sample overhead.
//
// See feedback memory: feedback_obb_faces_direct_loses.md
struct OBBFaceSampler
{
   using scalar_type   = float32_t;
   using vector2_type  = float32_t2;
   using vector3_type  = float32_t3;
   using domain_type   = vector2_type;
   using codomain_type = vector3_type;
   using density_type  = scalar_type;
   using weight_type   = density_type;

   struct cache_type
   {
      typename sampling::SphericalRectangle<float32_t>::cache_type inner;
      density_type pdf;
   };

   sampling::SphericalRectangle<float32_t> rects[3];
   uint32_t  numRects;
   float32_t cumSA0;
   float32_t cumSA1;
   float32_t totalSolidAngle;
   float32_t rcpTotalSolidAngle;

   // Build sphrect for face on `Axis`, using T as the shared world-space origin.
   // T_idx encodes which OBB cube corner T is (bits 0/1/2 = axis sides).
   // swap flips right/up for correct outward-normal direction; rule is
   // popcount(T_idx) even => swap.
   template<uint32_t Axis>
   static sampling::SphericalRectangle<float32_t> makeRectFromTip(shapes::OBBView<float32_t> view, float32_t3 T_pos, uint32_t T_idx, bool swap)
   {
      const uint32_t a1 = (Axis + 1u) % 3u;
      const uint32_t a2 = (Axis + 2u) % 3u;

      const float32_t s1 = ((T_idx & (1u << a1)) != 0u) ? -1.0f : 1.0f;
      const float32_t s2 = ((T_idx & (1u << a2)) != 0u) ? -1.0f : 1.0f;
      const float32_t3 rNatural = view.columns[a1] * s1;
      const float32_t3 uNatural = view.columns[a2] * s2;

      shapes::CompressedSphericalRectangle<float32_t> compressed;
      compressed.origin = T_pos;
      if (swap)
      {
         compressed.right = uNatural;
         compressed.up    = rNatural;
      }
      else
      {
         compressed.right = rNatural;
         compressed.up    = uNatural;
      }

      const shapes::SphericalRectangle<float32_t> shapeRect = shapes::SphericalRectangle<float32_t>::create(compressed);
      return sampling::SphericalRectangle<float32_t>::create(shapeRect, float32_t3(0.0f, 0.0f, 0.0f));
   }

   // create(view) -- region derived inline from view, no silhouette pipeline.
   static OBBFaceSampler create(shapes::OBBView<float32_t> view)
   {
      OBBFaceSampler self;

      // Region inline (mirrors silhouette.hlsl ClippedSilhouette::create).
      const float32_t3 sqScales = float32_t3(dot(view.columns[0], view.columns[0]), dot(view.columns[1], view.columns[1]), dot(view.columns[2], view.columns[2]));
      const float32_t3 proj     = -float32_t3(dot(view.columns[0], view.minCorner), dot(view.columns[1], view.minCorner), dot(view.columns[2], view.minCorner));
      const uint32_t3 below     = uint32_t3(proj < float32_t3(0, 0, 0));
      const uint32_t3 above     = uint32_t3(proj > sqScales);
      const uint32_t3 region    = uint32_t3(uint32_t3(1u, 1u, 1u) + below - above);

      const bool xVis = (region.x != 1u);
      const bool yVis = (region.y != 1u);
      const bool zVis = (region.z != 1u);
      self.numRects = uint32_t(xVis) + uint32_t(yVis) + uint32_t(zVis);

      // Tip T: bit i set iff observer past max on axis i (region[i] == 0).
      const uint32_t T_idx = (uint32_t(region.x == 0u) << 0)
                           | (uint32_t(region.y == 0u) << 1)
                           | (uint32_t(region.z == 0u) << 2);
      const float32_t3 T_pos = view.getVertex(T_idx);

      const bool swap = (countbits(T_idx) & 1u) == 0u;

      // Slot 0: first visible axis. Cascade keeps every rects[K] write at a
      // literal slot index, every makeRectFromTip<Axis> at literal Axis.
      if (xVis)
         self.rects[0] = makeRectFromTip<0>(view, T_pos, T_idx, swap);
      else if (yVis)
         self.rects[0] = makeRectFromTip<1>(view, T_pos, T_idx, swap);
      else
         self.rects[0] = makeRectFromTip<2>(view, T_pos, T_idx, swap);

      // Slot 1: second visible. xVis && yVis -> y; otherwise z.
      if (self.numRects >= 2u)
      {
         if (xVis && yVis)
            self.rects[1] = makeRectFromTip<1>(view, T_pos, T_idx, swap);
         else
            self.rects[1] = makeRectFromTip<2>(view, T_pos, T_idx, swap);
      }

      // Slot 2: only when all 3 visible -> axis z.
      if (self.numRects == 3u)
         self.rects[2] = makeRectFromTip<2>(view, T_pos, T_idx, swap);

      // CDF over face solid angles.
      self.cumSA0             = self.rects[0].solidAngle;
      self.cumSA1             = self.cumSA0 + ((self.numRects >= 2u) ? self.rects[1].solidAngle : 0.0f);
      self.totalSolidAngle    = self.cumSA1 + ((self.numRects == 3u) ? self.rects[2].solidAngle : 0.0f);
      self.rcpTotalSolidAngle = 1.0f / self.totalSolidAngle;

      return self;
   }

   // Uniform interface compatibility: ignores `silhouette` since region is
   // derived inline from view.
   static OBBFaceSampler create(NBL_CONST_REF_ARG(ClippedSilhouette) /*silhouette*/, shapes::OBBView<float32_t> view)
   {
      return create(view);
   }

   codomain_type generate(domain_type u, NBL_REF_ARG(cache_type) cache)
   {
      const float32_t target = u.x * totalSolidAngle;
      codomain_type dir;

      if (target < cumSA0)
      {
         const float32_t uPrime = target / cumSA0;
         dir = rects[0].generate(float32_t2(uPrime, u.y), cache.inner);
      }
      else if (numRects == 2u || target < cumSA1)
      {
         const float32_t faceSA = (numRects == 2u) ? (totalSolidAngle - cumSA0) : (cumSA1 - cumSA0);
         const float32_t uPrime = (target - cumSA0) / faceSA;
         dir = rects[1].generate(float32_t2(uPrime, u.y), cache.inner);
      }
      else // numRects == 3 and target >= cumSA1
      {
         const float32_t faceSA = totalSolidAngle - cumSA1;
         const float32_t uPrime = (target - cumSA1) / faceSA;
         dir = rects[2].generate(float32_t2(uPrime, u.y), cache.inner);
      }

      const bool valid = dir.z > 0.0f;
      cache.pdf = hlsl::select(valid, rcpTotalSolidAngle, 0.0f);
      return dir;
   }

   density_type forwardPdf(domain_type u, cache_type cache) NBL_CONST_MEMBER_FUNC { return cache.pdf; }
   weight_type  forwardWeight(domain_type u, cache_type cache) NBL_CONST_MEMBER_FUNC { return cache.pdf; }
   uint32_t     selectedIdx(cache_type cache) NBL_CONST_MEMBER_FUNC { return 0u; }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_OBB_FACE_SAMPLING_HLSL_INCLUDED_
