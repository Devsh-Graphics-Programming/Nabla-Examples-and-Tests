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

struct TriangleFanSampler
{
   uint32_t count; // Number of valid triangles
   uint32_t samplingMode; // Mode used during build
   float32_t totalWeight; // Sum of all triangle weights (for PDF computation)
   float32_t3 faceNormal; // Face normal (only used for projected mode)
   float32_t cdf[MAX_TRIANGLES]; // Normalized CDF: cdf[i] = sum(weight[0..i]) / totalWeight
   float32_t triangleSolidAngles[MAX_TRIANGLES]; // Raw weight per triangle (for PDF after selection)
   uint32_t triangleIndices[MAX_TRIANGLES]; // Vertex index i (forms triangle with v0, vi, vi+1)

   // Build fan triangulation, cache weights for triangle selection
   static TriangleFanSampler create(ClippedSilhouette silhouette, uint32_t mode)
   {
      TriangleFanSampler self;
      self.count = 0;
      self.totalWeight = 0.0f;
      self.samplingMode = mode;
      self.faceNormal = float32_t3(0, 0, 0);

      if (silhouette.count < 3)
         return self;

      const float32_t3 v0 = silhouette.vertices[0];
      const float32_t3 origin = float32_t3(0, 0, 0);

      // Compute face normal ONCE before the loop - silhouette is planar!
      if (mode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
      {
         float32_t3 v1 = silhouette.vertices[1];
         float32_t3 v2 = silhouette.vertices[2];
         self.faceNormal = normalize(cross(v1 - v0, v2 - v0));
      }

      // Build fan triangulation from v0
      NBL_UNROLL
      for (uint32_t i = 1; i < silhouette.count - 1; i++)
      {
         float32_t3 v1 = silhouette.vertices[i];
         float32_t3 v2 = silhouette.vertices[i + 1];

         const float32_t3 triVerts[3] = {v0, v1, v2};
         shapes::SphericalTriangle<float32_t> shapeTri = shapes::SphericalTriangle<float32_t>::create(triVerts, origin);

         // Skip degenerate triangles
         if (shapeTri.solid_angle <= 0.0f)
            continue;

         // Calculate triangle solid angle
         float32_t solidAngle;
         if (mode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
            solidAngle = shapeTri.projectedSolidAngle(self.faceNormal);
         else
            solidAngle = shapeTri.solid_angle;

         if (solidAngle <= 0.0f)
            continue;

         // Store only what's needed for weighted selection
         self.triangleSolidAngles[self.count] = solidAngle;
         self.triangleIndices[self.count] = i;
         self.totalWeight += solidAngle;
         self.count++;
      }

      // Build normalized CDF from raw weights
      {
         float32_t rcpTotal = (self.totalWeight > 0.0f) ? (1.0f / self.totalWeight) : 0.0f;
         float32_t cumulative = 0.0f;
         for (uint32_t i = 0; i < self.count; i++)
         {
            cumulative += self.triangleSolidAngles[i];
            self.cdf[i] = cumulative * rcpTotal;
         }
      }

      bool luneDetected = false;
      for (uint32_t i = 0; i < silhouette.count; i++)
      {
         uint32_t j = (i + 1) % silhouette.count;
         float32_t3 n1 = normalize(silhouette.vertices[i]);
         float32_t3 n2 = normalize(silhouette.vertices[j]);
         if (dot(n1, n2) < -0.99f)
         {
            luneDetected = true;
            assert(false && "Spherical lune detected: antipodal silhouette edge");
         }
      }
      DebugRecorder::recordTriangleFan(luneDetected, self.count, self.totalWeight, self.triangleSolidAngles);

      return self;
   }

   // Sample using cached selection weights, recompute geometry on-demand
   float32_t3 sample(ClippedSilhouette silhouette, float32_t2 xi, out float32_t pdf, out uint32_t selectedIdx)
   {
      selectedIdx = 0;

      // Handle empty or invalid data
      if (count == 0 || totalWeight <= 0.0f)
      {
         pdf = 0.0f;
         return float32_t3(0, 0, 1);
      }

      // Select triangle via precomputed normalized CDF
      float32_t prevCdf = 0.0f;
      NBL_UNROLL
      for (uint32_t i = 0; i < count; i++)
      {
         if (xi.x <= cdf[i])
         {
            selectedIdx = i;
            break;
         }
         prevCdf = cdf[i];
      }

      // Remap xi.x to [0,1] within selected triangle's CDF interval
      float32_t cdfWidth = cdf[selectedIdx] - prevCdf;
      float32_t u = (xi.x - prevCdf) / max(cdfWidth, 1e-7f);
      float32_t triSolidAngle = triangleSolidAngles[selectedIdx];

      // Reconstruct the selected triangle geometry
      uint32_t vertexIdx = triangleIndices[selectedIdx];
      float32_t3 v0 = silhouette.vertices[0];
      float32_t3 v1 = silhouette.vertices[vertexIdx];
      float32_t3 v2 = silhouette.vertices[vertexIdx + 1];

      float32_t3 origin = float32_t3(0, 0, 0);

      const float32_t3 triVerts[3] = {v0, v1, v2};
      shapes::SphericalTriangle<float32_t> shapeTri = shapes::SphericalTriangle<float32_t>::create(triVerts, origin);

      // Sample based on mode
      float32_t3 direction;
      const float32_t2 u2 = float32_t2(u, xi.y);

      if (samplingMode == SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE)
      {
         // faceNormal was precomputed during create() -- silhouette is planar
         sampling::ProjectedSphericalTriangle<float32_t> samplingTri = sampling::ProjectedSphericalTriangle<float32_t>::create(shapeTri, faceNormal, false);
         sampling::ProjectedSphericalTriangle<float32_t>::cache_type cache;
         direction = samplingTri.generate(u2, cache);
         triSolidAngle = 1.0f / samplingTri.forwardPdf(u2, cache);
      }
      else
      {
         sampling::SphericalTriangle<float32_t> samplingTri = sampling::SphericalTriangle<float32_t>::create(shapeTri);
         sampling::SphericalTriangle<float32_t>::cache_type cache;
         direction = samplingTri.generate(u2, cache);
      }

      // Calculate PDF
      float32_t trianglePdf = 1.0f / triSolidAngle;
      float32_t selectionProb = triSolidAngle / totalWeight;
      pdf = trianglePdf * selectionProb;

      return normalize(direction);
   }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_TRIANGLE_SAMPLING_HLSL_INCLUDED_
