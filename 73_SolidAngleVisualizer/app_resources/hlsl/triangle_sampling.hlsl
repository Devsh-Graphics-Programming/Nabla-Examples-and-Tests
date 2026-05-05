//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_TRIANGLE_SAMPLING_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_TRIANGLE_SAMPLING_HLSL_INCLUDED_

// Include the spherical triangle utilities
#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/random/pcg.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>
#include "silhouette.hlsl"

using namespace nbl::hlsl;

// Maximum number of triangles we can have after clipping
// Without clipping, max 3 faces can be visible at once so 3 faces * 2 triangles = 6 edges, forming max 4 triangles
// With clipping, one more edge. 7 - 2 = 5 max triangles because fanning from one vertex
#define MAX_TRIANGLES 5

// ============================================================================
// TriangleFanSampler: importance-sampled fan triangulation of the clipped
// silhouette. create() takes only the silhouette and materializes verts
// internally, storing them as a member so sample() has random access without
// the caller threading verts through.
//
// All loops over silCount/triangle-count are cascade-unrolled (instead of
// `for + break`) so every `self.verts[K]` / `cdf[K]` / `triangleSolidAngles[K]`
// access has a literal slot index. This keeps the local arrays in registers
// (SROA-promoted) instead of spilling to addressable Function memory -- a
// single dynamic-index access would demote the whole array and tank every
// subsequent read.
// ============================================================================
template<bool Projected>
struct TriangleFanSampler
{
   using scalar_type   = float32_t;
   using vector2_type  = float32_t2;
   using vector3_type  = float32_t3;
   using domain_type   = vector2_type;
   using codomain_type = vector3_type;
   using density_type  = scalar_type;
   using weight_type   = density_type;

   // Cache for the TractableSampler concept. Stores the per-triangle pdf
   // (selectionProb * trianglePdf) so forwardPdf is an O(1) load, plus the
   // selected fan-triangle index (used by the visualization code path to
   // colour each triangle differently).
   struct cache_type
   {
      density_type pdf;
      uint32_t     selectedIdx;
   };

   uint32_t        count;       // Number of valid triangles
   float32_t       totalWeight; // Sum of all triangle weights (for PDF computation)
   float32_t3      faceNormal;  // Face normal (only used for projected mode)
   float32_t       cdf[MAX_TRIANGLES];                 // Normalized CDF: cdf[i] = sum(weight[0..i]) / totalWeight
   float32_t       triangleSolidAngles[MAX_TRIANGLES]; // Raw weight per triangle (for PDF after selection)
   uint32_t        triangleIndices[MAX_TRIANGLES];     // Vertex index i (forms triangle with v0, vi, vi+1)
   float32_t3 verts[MAX_SILHOUETTE_VERTICES];

   // Build fan triangulation, cache weights for triangle selection.
   // Materializes silhouette verts internally (using the view stored in
   // ClippedSilhouette) and keeps them as a member for sample-time access.
   static TriangleFanSampler<Projected> create(NBL_CONST_REF_ARG(ClippedSilhouette) silhouette, shapes::OBBView<float32_t> view)
   {
      TriangleFanSampler<Projected> self;
      self.totalWeight        = 0.0f;
      self.faceNormal         = float32_t3(0, 0, 0);
      const uint32_t silCount = silhouette.count;
      silhouette.materialize(view, self.verts);

      // Pre-zero the per-triangle arrays so unused slots are well-defined --
      // the cascade below populates exactly silCount-2 slots and we don't
      // want the tail to leak garbage into the CDF.
      NBL_UNROLL
      for (uint32_t z = 0; z < MAX_TRIANGLES; z++)
      {
         self.triangleSolidAngles[z] = 0.0f;
         self.triangleIndices[z]     = 0u;
         self.cdf[z]                 = 0.0f;
      }

      if (silCount < 3)
      {
         self.count = 0;
         return self;
      }

      const float32_t3 v0 = self.verts[0];

      // Compute face normal ONCE before the loop - silhouette is planar!
      if (Projected)
      {
         const float32_t3 v1 = self.verts[1];
         const float32_t3 v2 = self.verts[2];
         self.faceNormal     = normalize(cross(v1 - v0, v2 - v0));
      }

      // Fan triangulation: triangles (v0, self.verts[I], self.verts[I+1]) for I = 1..silCount-2.
      // Cascade-on-silCount so each call site has literal I.
      processFanTri<1>(v0, self.faceNormal, self);
      if (silCount > 3)
      {
         processFanTri<2>(v0, self.faceNormal, self);
         if (silCount > 4)
         {
            processFanTri<3>(v0, self.faceNormal, self);
            if (silCount > 5)
            {
               processFanTri<4>(v0, self.faceNormal, self);
               if (silCount > 6)
                  processFanTri<5>(v0, self.faceNormal, self);
            }
         }
      }
      // self.count = silCount - 2 (every triangle slot gets populated, possibly
      // with zero weight for degenerates -- they're handled cleanly by the CDF).
      self.count = silCount - 2u;

      // CDF build: cascade-on-count so cdf[K] / triangleSolidAngles[K] are
      // literal-index accesses; otherwise the whole sampler struct's arrays
      // would demote to Function memory.
      const float32_t rcpTotal   = (self.totalWeight > 0.0f) ? rcp(self.totalWeight) : 0.0f;
      float32_t       cumulative = 0.0f;

      cumulative += self.triangleSolidAngles[0];
      self.cdf[0] = cumulative * rcpTotal;
      if (self.count > 1)
      {
         cumulative += self.triangleSolidAngles[1];
         self.cdf[1] = cumulative * rcpTotal;
         if (self.count > 2)
         {
            cumulative += self.triangleSolidAngles[2];
            self.cdf[2] = cumulative * rcpTotal;
            if (self.count > 3)
            {
               cumulative += self.triangleSolidAngles[3];
               self.cdf[3] = cumulative * rcpTotal;
               if (self.count > 4)
               {
                  cumulative += self.triangleSolidAngles[4];
                  self.cdf[4] = cumulative * rcpTotal;
               }
            }
         }
      }

#if DEBUG_DATA
      // Debug-only closed-loop walk over silhouette edges. Released builds DCE
      // both the loop (recordTriangleFan is a no-op stub) and luneDetected.
      bool luneDetected = false;
      for (uint32_t i = 0; i < silCount; i++)
      {
         const uint32_t   j  = (i + 1u < silCount) ? i + 1u : 0u;
         const float32_t3 ni = nbl::hlsl::normalize(self.verts[i]);
         const float32_t3 nj = nbl::hlsl::normalize(self.verts[j]);
         if (dot(ni, nj) < -0.99f)
         {
            luneDetected = true;
            assert(false && "Spherical lune detected: antipodal silhouette edge");
         }
      }
      DebugRecorder::recordTriangleFan(luneDetected, self.count, self.totalWeight, self.triangleSolidAngles);
#else
      DebugRecorder::recordTriangleFan(false, self.count, self.totalWeight, self.triangleSolidAngles);
#endif

      return self;
   }

   // TractableSampler::generate. Picks a fan triangle by xi.x via the cached
   // CDF, samples within it, and registers (selectedIdx, pdf) in the cache so
   // forwardPdf is an O(1) load. Geometry is reconstructed on-demand from
   // `this->verts`. The CDF-select and triangle-reconstruct steps both use
   // literal-index cascades on count / vertexIdx -- a single dynamic-index
   // access into verts.v / cdf / triangleIndices would demote those arrays to
   // Function memory and slow every call.
   codomain_type generate(domain_type xi, NBL_REF_ARG(cache_type) cache)
   {
      // Handle empty or invalid data
      if (count == 0 || totalWeight <= 0.0f)
      {
         cache.pdf         = 0.0f;
         cache.selectedIdx = 0;
         return codomain_type(0, 0, 1);
      }

      // Use a local idx for all the cascade work; assign to the cache once at
      // the end so the cache field doesn't get pessimised by repeated stores.
      uint32_t    idx     = count - 1u; // fall-through default for numerical roundoff
      scalar_type prevCdf = 0.0f;
      if (xi.x <= cdf[0])
      {
         idx = 0;
      }
      else if (count > 1 && xi.x <= cdf[1])
      {
         idx     = 1;
         prevCdf = cdf[0];
      }
      else if (count > 2 && xi.x <= cdf[2])
      {
         idx     = 2;
         prevCdf = cdf[1];
      }
      else if (count > 3 && xi.x <= cdf[3])
      {
         idx     = 3;
         prevCdf = cdf[2];
      }
      else if (count > 4 && xi.x <= cdf[4])
      {
         idx     = 4;
         prevCdf = cdf[3];
      }
      else // fall-through to last valid triangle
      {
         if (count == 2)
            prevCdf = cdf[0];
         else if (count == 3)
            prevCdf = cdf[1];
         else if (count == 4)
            prevCdf = cdf[2];
         else if (count == 5)
            prevCdf = cdf[3];
      }
      cache.selectedIdx = idx;

      // cdf[idx] read also via cascade so the array stays SROA'd.
      scalar_type selectedCdf;
      if (idx == 0)
         selectedCdf = cdf[0];
      else if (idx == 1)
         selectedCdf = cdf[1];
      else if (idx == 2)
         selectedCdf = cdf[2];
      else if (idx == 3)
         selectedCdf = cdf[3];
      else
         selectedCdf = cdf[4];

      const scalar_type cdfWidth = selectedCdf - prevCdf;
      const scalar_type u        = (xi.x - prevCdf) / max(cdfWidth, 1e-7f);

      scalar_type triSolidAngle;
      if (idx == 0)
         triSolidAngle = triangleSolidAngles[0];
      else if (idx == 1)
         triSolidAngle = triangleSolidAngles[1];
      else if (idx == 2)
         triSolidAngle = triangleSolidAngles[2];
      else if (idx == 3)
         triSolidAngle = triangleSolidAngles[3];
      else
         triSolidAngle = triangleSolidAngles[4];

      uint32_t vertexIdx;
      if (idx == 0)
         vertexIdx = triangleIndices[0];
      else if (idx == 1)
         vertexIdx = triangleIndices[1];
      else if (idx == 2)
         vertexIdx = triangleIndices[2];
      else if (idx == 3)
         vertexIdx = triangleIndices[3];
      else
         vertexIdx = triangleIndices[4];

      // Reconstruct triangle geometry. vertexIdx is in [1, MAX_SILHOUETTE_VERTICES-2]
      // and is data-dependent on xi -- cascade so verts[vertexIdx] / verts[vertexIdx+1]
      // become literal-index reads. With our 7-vertex max, vertexIdx <= 5.
      const codomain_type v0 = verts[0];
      codomain_type       v1, v2;
      if (vertexIdx == 1)
      {
         v1 = verts[1];
         v2 = verts[2];
      }
      else if (vertexIdx == 2)
      {
         v1 = verts[2];
         v2 = verts[3];
      }
      else if (vertexIdx == 3)
      {
         v1 = verts[3];
         v2 = verts[4];
      }
      else if (vertexIdx == 4)
      {
         v1 = verts[4];
         v2 = verts[5];
      }
      else
      {
         v1 = verts[5];
         v2 = verts[6];
      } // vertexIdx == 5

      const codomain_type origin = codomain_type(0, 0, 0);

      const codomain_type                  triVerts[3] = {v0, v1, v2};
      shapes::SphericalTriangle<float32_t> shapeTri    = shapes::SphericalTriangle<float32_t>::create(triVerts, origin);

      // Sample based on mode
      codomain_type    direction;
      const domain_type u2 = domain_type(u, xi.y);

      if (Projected)
      {
         // faceNormal was precomputed during create(), silhouette is planar
         sampling::ProjectedSphericalTriangle<float32_t>             samplingTri = sampling::ProjectedSphericalTriangle<float32_t>::create(shapeTri, faceNormal, false);
         sampling::ProjectedSphericalTriangle<float32_t>::cache_type triCache;
         direction     = samplingTri.generate(u2, triCache);
         triSolidAngle = 1.0f / samplingTri.forwardPdf(u2, triCache);
      }
      else
      {
         sampling::SphericalTriangle<float32_t>             samplingTri = sampling::SphericalTriangle<float32_t>::create(shapeTri);
         sampling::SphericalTriangle<float32_t>::cache_type triCache;
         direction = samplingTri.generate(u2, triCache);
      }

      // Calculate PDF: trianglePdf * selectionProb where the per-triangle pdf
      // is 1/triSolidAngle (uniform over the spherical triangle) and the
      // selection probability is triSolidAngle / totalWeight.
      cache.pdf = (1.0f / triSolidAngle) * (triSolidAngle / totalWeight);

      return normalize(direction);
   }

   density_type forwardPdf(domain_type u, cache_type cache) NBL_CONST_MEMBER_FUNC { return cache.pdf; }
   weight_type  forwardWeight(domain_type u, cache_type cache) NBL_CONST_MEMBER_FUNC { return cache.pdf; }
   uint32_t     selectedIdx(cache_type cache) NBL_CONST_MEMBER_FUNC { return cache.selectedIdx; }

   // Process one fan triangle (v0, self.verts[I], self.verts[I+1]) at the cascade level.
   // I is a template constant so self.verts[I] / self.verts[I+1] / triangleSolidAngles[I-1]
   // / triangleIndices[I-1] are all literal-index accesses; the body's
   // append-to-slot-(I-1) only works because we treat degenerate triangles as
   // zero-weight rather than skipping them. This is a behavior change from the
   // old `count++ on non-degenerate` form: degenerate triangles now occupy a
   // slot with zero weight, which contributes nothing to the CDF and has
   // selection probability 0, so the sampling result is unchanged.
   template<uint32_t I>
   static void processFanTri(float32_t3 v0, float32_t3 faceNormal, NBL_REF_ARG(TriangleFanSampler<Projected>) self)
   {
      const float32_t3 v1 = self.verts[I];
      const float32_t3 v2 = self.verts[I + 1];

      const float32_t3                     origin      = float32_t3(0, 0, 0);
      const float32_t3                     triVerts[3] = {v0, v1, v2};
      shapes::SphericalTriangle<float32_t> shapeTri    = shapes::SphericalTriangle<float32_t>::create(triVerts, origin);

      // Compute solid angle (or projected) and clamp to >= 0; degenerate
      // triangles end up with zero weight and don't affect sampling.
      float32_t sa = Projected ? shapeTri.projectedSolidAngle(faceNormal) : shapeTri.solid_angle;
      sa = max(sa, 0.0f);

      self.triangleSolidAngles[I - 1u] = sa;
      self.triangleIndices[I - 1u]     = I;
      self.totalWeight += sa;
   }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_TRIANGLE_SAMPLING_HLSL_INCLUDED_
