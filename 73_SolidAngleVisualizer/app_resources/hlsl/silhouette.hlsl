//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_SILHOUETTE_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_SILHOUETTE_HLSL_INCLUDED_

#include "common.hlsl"
#include "debug_vis.hlsl"
#include "utils.hlsl"
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/shapes/obb.hlsl>

using namespace nbl::hlsl;

// TODO: unused, remove later
// Vertices are ordered CCW relative to the camera view.
static const uint32_t silhouettes[27][7] = {
   {6, 1, 3, 2, 6, 4, 5}, // 0: Black
   {6, 2, 6, 4, 5, 7, 3}, // 1: White
   {6, 0, 4, 5, 7, 3, 2}, // 2: Gray
   {6, 1, 3, 7, 6, 4, 5}, // 3: Red
   {4, 4, 5, 7, 6, 0, 0}, // 4: Green
   {6, 0, 4, 5, 7, 6, 2}, // 5: Blue
   {6, 0, 1, 3, 7, 6, 4}, // 6: Yellow
   {6, 0, 1, 5, 7, 6, 4}, // 7: Magenta
   {6, 0, 1, 5, 7, 6, 2}, // 8: Cyan
   {6, 1, 3, 2, 6, 7, 5}, // 9: Orange
   {4, 2, 6, 7, 3, 0, 0}, // 10: Light Orange
   {6, 0, 4, 6, 7, 3, 2}, // 11: Dark Orange
   {4, 1, 3, 7, 5, 0, 0}, // 12: Pink
   {4, 0, 4, 6, 7, 3, 2}, // 13: Light Pink
   {4, 0, 4, 6, 2, 0, 0}, // 14: Deep Rose
   {6, 0, 1, 3, 7, 5, 4}, // 15: Purple
   {4, 0, 1, 5, 4, 0, 0}, // 16: Light Purple
   {6, 0, 1, 5, 4, 6, 2}, // 17: Indigo
   {6, 0, 2, 6, 7, 5, 1}, // 18: Dark Green
   {6, 0, 2, 6, 7, 3, 1}, // 19: Lime
   {6, 0, 4, 6, 7, 3, 1}, // 20: Forest Green
   {6, 0, 2, 3, 7, 5, 1}, // 21: Navy
   {4, 0, 2, 3, 1, 0, 0}, // 22: Sky Blue
   {6, 0, 4, 6, 2, 3, 1}, // 23: Teal
   {6, 0, 2, 3, 7, 5, 4}, // 24: Brown
   {6, 0, 2, 3, 1, 5, 4}, // 25: Tan/Beige
   {6, 1, 5, 4, 6, 2, 3} // 26: Dark Brown
};

// Binary packed silhouettes
static const uint32_t binSilhouettes[27] = {
   0b11000000000000101100110010011001,
   0b11000000000000011111101100110010,
   0b11000000000000010011111101100000,
   0b11000000000000101100110111011001,
   0b10000000000000000000110111101100,
   0b11000000000000010110111101100000,
   0b11000000000000100110111011001000,
   0b11000000000000100110111101001000,
   0b11000000000000010110111101001000,
   0b11000000000000101111110010011001,
   0b10000000000000000000011111110010,
   0b11000000000000010011111110100000,
   0b10000000000000000000101111011001,
   0b11000000000000010011111110100000,
   0b10000000000000000000010110100000,
   0b11000000000000100101111011001000,
   0b10000000000000000000100101001000,
   0b11000000000000010110100101001000,
   0b11000000000000001101111110010000,
   0b11000000000000001011111110010000,
   0b11000000000000001011111110100000,
   0b11000000000000001101111011010000,
   0b10000000000000000000001011010000,
   0b11000000000000001011010110100000,
   0b11000000000000100101111011010000,
   0b11000000000000100101001011010000,
   0b11000000000000011010110100101001,
};

struct BinSilhouette
{
   static BinSilhouette create(uint32_t configIndex)
   {
      BinSilhouette s = (BinSilhouette)0;
      s.data = binSilhouettes[configIndex];
      return s;
   }

   uint32_t getVertexIndex(uint32_t index) NBL_CONST_MEMBER_FUNC
   {
      return (data >> (3u * index)) & 0x7u;
   }

   // Get silhouette size
   uint32_t getSilhouetteSize() NBL_CONST_MEMBER_FUNC
   {
      return (data >> 29u) & 0x7u;
   }

   // Build a 12-bit mask of which cube edges are part of the silhouette.
   // Edge enumeration: for axis in {0,1,2}, for each corner with axis-bit
   // clear, edge = (corner, corner | (1<<axis)).
   //
   // Each silhouette edge (s0, s1) differs in exactly one bit (adjacent cube
   // corners). We recover axis = firstbitlow(s0 ^ s1), then compute the
   // compact 2-bit index by stripping out the axis bit from the lower corner.
   uint32_t computeEdgeMask() NBL_CONST_MEMBER_FUNC
   {
      uint32_t vertexCount = getSilhouetteSize();
      uint32_t mask = 0;
      for (uint32_t s = 0; s < vertexCount; s++)
      {
         uint32_t s0 = getVertexIndex(s);
         uint32_t s1 = getVertexIndex((s + 1 < vertexCount) ? s + 1 : 0);
         uint32_t diff = s0 ^ s1;
         uint32_t axis = firstbitlow(diff);
         uint32_t lowCorner = s0 & s1;
         // Strip out the axis bit to get a 2-bit index among the 4 edges for this axis
         uint32_t below = lowCorner & ((1u << axis) - 1u);
         uint32_t above = lowCorner >> (axis + 1u);
         uint32_t compact = (above << axis) | below;
         mask |= 1u << (axis * 4u + compact);
      }
      return mask;
   }

   void rotr(uint32_t shift, uint32_t size)
   {
      data = nbl::hlsl::rotr(data, shift, size);
   }

   void rotl(uint32_t shift, uint32_t size)
   {
      data = nbl::hlsl::rotl(data, shift, size);
   }

   uint32_t data;
};

struct ClippedSilhouette
{

   static ClippedSilhouette create(shapes::OBBView<float32_t> view)
   {
      uint32_t3 region;
      uint32_t configIndex, vertexCount;
      BinSilhouette sil = computeRegionAndConfig(view, region, configIndex, vertexCount);
      ClippedSilhouette s = (ClippedSilhouette)0;
      s.compute(view, vertexCount, sil);
      return s;
   }

   // only used by projected parallelogram
   void normalize()
   {
      vertices[0] = nbl::hlsl::normalize(vertices[0]);
      vertices[1] = nbl::hlsl::normalize(vertices[1]);
      vertices[2] = nbl::hlsl::normalize(vertices[2]);
      if (count > 3)
      {
         vertices[3] = nbl::hlsl::normalize(vertices[3]);
         if (count > 4)
         {
            vertices[4] = nbl::hlsl::normalize(vertices[4]);
            if (count > 5)
            {
               vertices[5] = nbl::hlsl::normalize(vertices[5]);
               if (count > 6)
               {
                  vertices[6] = nbl::hlsl::normalize(vertices[6]);
               }
            }
         }
      }
   }

   // Compute the silhouette centroid (average direction)
   // Returns unnormalized centroid (sum of vertices). The direction is what
   // matters for the adaptive axis3 blend, the magnitude cancels out after
   // normalize(center * tBlend + (0,0,1)). just as small optimization.
   float32_t3 getUnnormalizedCenter()
   {
      float32_t3 sum = float32_t3(0, 0, 0);

      NBL_UNROLL
      for (uint32_t i = 0; i < MAX_SILHOUETTE_VERTICES; i++)
      {
         if (i < count)
            sum += vertices[i];
      }

      return sum;
   }

   static BinSilhouette computeRegionAndConfig(shapes::OBBView<float32_t> view, out uint32_t3 region, out uint32_t configIndex, out uint32_t vertexCount)
   {
      // With [0,1]^3 local space, the observer's unnormalized OBB-local
      // coordinate along axis i is proj_i = -dot(col_i, minCorner).
      // Compare against 0 and |col_i|^2 (the unnormalized [0,1] bounds)
      // to classify into the 27-configuration LUT.
      float32_t3 sqScales = float32_t3(
         dot(view.columns[0], view.columns[0]),
         dot(view.columns[1], view.columns[1]),
         dot(view.columns[2], view.columns[2]));

      float32_t3 proj = -float32_t3(
         dot(view.columns[0], view.minCorner),
         dot(view.columns[1], view.minCorner),
         dot(view.columns[2], view.minCorner));

      region = uint32_t3(
         proj.x < 0 ? 2 : (proj.x > sqScales.x ? 0 : 1),
         proj.y < 0 ? 2 : (proj.y > sqScales.y ? 0 : 1),
         proj.z < 0 ? 2 : (proj.z > sqScales.z ? 0 : 1));

      configIndex = region.x + region.y * 3u + region.z * 9u;

      BinSilhouette sil = BinSilhouette::create(configIndex);
      vertexCount = sil.getSilhouetteSize();

      return sil;
   }

   void compute(shapes::OBBView<float32_t> view, uint32_t vertexCount, BinSilhouette sil)
   {

      // Build clip mask (z < 0)
      uint32_t clipMask = 0u;
      NBL_UNROLL
      for (uint32_t i = 0; i < 4; i++)
         clipMask |= (view.getVertexZ(sil.getVertexIndex(i)) < 0.0f ? 1u : 0u) << i;

      if (vertexCount == 6)
      {
         NBL_UNROLL
         for (uint32_t i = 4; i < 6; i++)
            clipMask |= (view.getVertexZ(sil.getVertexIndex(i)) < 0.0f ? 1u : 0u) << i;
      }

      uint32_t clipCount = countbits(clipMask);

      // Invert clip mask to find first positive vertex
      uint32_t invertedMask = ~clipMask & ((1u << vertexCount) - 1u);

      // Check if wrap-around is needed (first and last bits negative)
      bool wrapAround = ((clipMask & 1u) != 0u) && ((clipMask & (1u << (vertexCount - 1))) != 0u);

      // Compute rotation amount
      uint32_t rotateAmount = nbl::hlsl::select(wrapAround, firstbitlow(invertedMask), // first positive
         firstbithigh(clipMask) + 1); // first vertex after last negative

      // Rotate masks
      uint32_t rotatedClipMask = nbl::hlsl::rotr(clipMask, rotateAmount, vertexCount);
      sil.rotr(rotateAmount * 3, vertexCount * 3);
      uint32_t positiveCount = vertexCount - clipCount;

      // Compute all 4 clip endpoints up front , independent obbVertex calls
      // give the compiler maximum ILP alongside the positive-vertex loop.
      uint32_t lastPosIdx = positiveCount - 1;
      uint32_t firstNegIdx = positiveCount;

      float32_t3 vLastPos = view.getVertex(sil.getVertexIndex(lastPosIdx));
      float32_t3 vFirstNeg = view.getVertex(sil.getVertexIndex(firstNegIdx));
      float32_t t = vLastPos.z / (vLastPos.z - vFirstNeg.z);
      float32_t3 clipA = lerp(vLastPos, vFirstNeg, t);

      float32_t3 vLastNeg = view.getVertex(sil.getVertexIndex(vertexCount - 1));
      float32_t3 vFirstPos = view.getVertex(sil.getVertexIndex(0));
      t = vLastNeg.z / (vLastNeg.z - vFirstPos.z);
      float32_t3 clipB = lerp(vLastNeg, vFirstPos, t);

      count = 0;

      NBL_UNROLL
      for (uint32_t i = 0; i < positiveCount; i++)
      {
         float32_t3 v0 = view.getVertex(sil.getVertexIndex(i));
         DebugRecorder::recordClippedVertex(count, v0, (i + rotateAmount) % vertexCount);
         vertices[count++] = v0;
      }

      if (clipCount > 0 && clipCount < vertexCount)
      {
         DebugRecorder::recordClippedVertex(count, clipA, 23);
         vertices[count++] = clipA;

         DebugRecorder::recordClippedVertex(count, clipB, 24);
         vertices[count++] = clipB;
      }

      DebugRecorder::recordClipResult(count, clipMask, clipCount, rotatedClipMask,
         rotateAmount, positiveCount, wrapAround, sil.data);
   }

   float32_t3 vertices[MAX_SILHOUETTE_VERTICES]; // Max 7 vertices after clipping, unnormalized
   uint32_t count;
};

struct SilEdgeNormals
{
   // Better not use and calculate it while creating the sampler
   static SilEdgeNormals create(NBL_CONST_REF_ARG(ClippedSilhouette) sil)
   {
      SilEdgeNormals result = (SilEdgeNormals)0;

      float32_t3 v0 = sil.vertices[0];
      float32_t3 v1 = sil.vertices[1];
      float32_t3 v2 = sil.vertices[2];

      result.edgeNormals[0] = cross(v0, v1);
      result.edgeNormals[1] = cross(v1, v2);

      if (sil.count > 3)
      {
         float32_t3 v3 = sil.vertices[3];
         result.edgeNormals[2] = cross(v2, v3);

         if (sil.count > 4)
         {
            float32_t3 v4 = sil.vertices[4];
            result.edgeNormals[3] = cross(v3, v4);

            if (sil.count > 5)
            {
               float32_t3 v5 = sil.vertices[5];
               result.edgeNormals[4] = cross(v4, v5);

               if (sil.count > 6)
               {
                  float32_t3 v6 = sil.vertices[6];
                  result.edgeNormals[5] = cross(v5, v6);
                  result.edgeNormals[6] = cross(v6, v0);
               }
               else
               {
                  result.edgeNormals[5] = cross(v5, v0);
               }
            }
            else
            {
               result.edgeNormals[4] = cross(v4, v0);
            }
         }
         else
         {
            result.edgeNormals[3] = cross(v3, v0);
         }
      }
      else
      {
         result.edgeNormals[2] = cross(v2, v0);
      }

      return result;
   }

   bool isInside(float32_t3 dir)
   {
      float32_t maxDot = dot(dir, edgeNormals[0]);
      maxDot = max(maxDot, dot(dir, edgeNormals[1]));
      maxDot = max(maxDot, dot(dir, edgeNormals[2]));
      maxDot = max(maxDot, dot(dir, edgeNormals[3]));
      maxDot = max(maxDot, dot(dir, edgeNormals[4]));
      maxDot = max(maxDot, dot(dir, edgeNormals[5]));
      maxDot = max(maxDot, dot(dir, edgeNormals[6]));
      return maxDot <= 0.0f;
   }

   // Transform edge normals from world-space to the pyramid's local frame in-place.
   // After this, edgeNormals[i] = (dot(n, axis1), dot(n, axis2), dot(n, axis3))
   // and isInsideLocal() can do 2-FMA half-plane tests without extra storage.
   // NOTE: destroys world-space normals , isInside() will no longer work correctly.
   void transformToLocal(float32_t3 axis1, float32_t3 axis2, float32_t3 axis3)
   {
      NBL_UNROLL
      for (uint32_t i = 0; i < MAX_SILHOUETTE_VERTICES; i++)
      {
         float32_t3 n = edgeNormals[i];
         edgeNormals[i] = float32_t3(dot(n, axis1), dot(n, axis2), dot(n, axis3));
      }
   }

   // 2D gnomonic containment test after transformToLocal().
   //   dot(dir_unnorm, n_local) = localX * n.x + localY * n.y + n.z
   bool isInsideLocal(float32_t localX, float32_t localY)
   {
      float32_t maxDot = localX * edgeNormals[0].x + localY * edgeNormals[0].y + edgeNormals[0].z;
      maxDot = max(maxDot, localX * edgeNormals[1].x + localY * edgeNormals[1].y + edgeNormals[1].z);
      maxDot = max(maxDot, localX * edgeNormals[2].x + localY * edgeNormals[2].y + edgeNormals[2].z);
      maxDot = max(maxDot, localX * edgeNormals[3].x + localY * edgeNormals[3].y + edgeNormals[3].z);
      maxDot = max(maxDot, localX * edgeNormals[4].x + localY * edgeNormals[4].y + edgeNormals[4].z);
      maxDot = max(maxDot, localX * edgeNormals[5].x + localY * edgeNormals[5].y + edgeNormals[5].z);
      maxDot = max(maxDot, localX * edgeNormals[6].x + localY * edgeNormals[6].y + edgeNormals[6].z);
      return maxDot <= 0.0f;
   }

   float32_t3 edgeNormals[MAX_SILHOUETTE_VERTICES];
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_SILHOUETTE_HLSL_INCLUDED_
