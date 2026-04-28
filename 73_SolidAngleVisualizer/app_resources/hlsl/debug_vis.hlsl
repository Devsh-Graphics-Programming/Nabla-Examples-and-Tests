//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_DEBUG_VIS_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_DEBUG_VIS_HLSL_INCLUDED_

#include "common.hlsl"

#ifdef __HLSL_VERSION
[[vk::binding(0, 0)]] RWStructuredBuffer<nbl::hlsl::ResultData> DebugDataBuffer;
#endif

struct DebugRecorder
{
#if DEBUG_DATA
   static void recordClippedVertex(uint32_t slot, float32_t3 pos, uint32_t originalIndex)
   {
      DebugDataBuffer[0].silhouette.clippedVertices[slot] = pos;
      DebugDataBuffer[0].silhouette.clippedVertexIndices[slot] = originalIndex;
   }

   static void recordClipResult(uint32_t vertexCount, uint32_t clipMask, uint32_t clipCount, uint32_t rotatedClipMask, uint32_t rotateAmount, uint32_t positiveCount, bool wrapAround, uint32_t rotatedSil)
   {
      DebugDataBuffer[0].silhouette.clippedVertexCount = vertexCount;
      DebugDataBuffer[0].silhouette.clipMask = clipMask;
      DebugDataBuffer[0].silhouette.clipCount = clipCount;
      DebugDataBuffer[0].silhouette.rotatedClipMask = rotatedClipMask;
      DebugDataBuffer[0].silhouette.rotateAmount = rotateAmount;
      DebugDataBuffer[0].silhouette.positiveVertCount = positiveCount;
      DebugDataBuffer[0].silhouette.wrapAround = (uint32_t)wrapAround;
      DebugDataBuffer[0].silhouette.rotatedSil = rotatedSil;
   }

   static void recordTriangleFan(bool luneDetected, uint32_t count, float32_t totalWeight, float32_t solidAngles[5])
   {
      DebugDataBuffer[0].triangleFan.sphericalLuneDetected = (uint32_t)luneDetected;
      DebugDataBuffer[0].triangleFan.maxTrianglesExceeded = (count > 5);
      DebugDataBuffer[0].triangleFan.triangleCount = count;
      DebugDataBuffer[0].triangleFan.totalSolidAngles = totalWeight;
      for (uint32_t tri = 0; tri < count; tri++)
         DebugDataBuffer[0].triangleFan.solidAngles[tri] = solidAngles[tri];
   }

   static void recordParallelogram(float32_t area, uint32_t convexMask, uint32_t n3Mask, float32_t2 corner, float32_t2 axisDir, float32_t width, float32_t height)
   {
      DebugDataBuffer[0].parallelogram.area = area;

      // Store per-edge convex and N3 flags
      DebugDataBuffer[0].parallelogram.n3Mask = n3Mask;
      for (uint32_t i = 0; i < 4; i++)
         DebugDataBuffer[0].parallelogram.edgeIsConvex[i] = (convexMask >> i) & 1u;

      // Compute and store the 4 parallelogram corners in circle-space
      float32_t2 perpDir = float32_t2(-axisDir.y, axisDir.x);
      DebugDataBuffer[0].parallelogram.corners[0] = corner;
      DebugDataBuffer[0].parallelogram.corners[1] = corner + width * axisDir;
      DebugDataBuffer[0].parallelogram.corners[2] = corner + width * axisDir + height * perpDir;
      DebugDataBuffer[0].parallelogram.corners[3] = corner + height * perpDir;
   }

   static void recordPyramid(float32_t3 axis1, float32_t3 axis2, float32_t3 center, float32_t4 bounds, float32_t solidAngle, uint32_t bestEdge)
   {
      DebugDataBuffer[0].pyramid.axis1 = axis1;
      DebugDataBuffer[0].pyramid.axis2 = axis2;
      DebugDataBuffer[0].pyramid.center = normalize(center);
      DebugDataBuffer[0].pyramid.halfWidth1 = (atan(bounds.z) - atan(bounds.x)) * 0.5f;
      DebugDataBuffer[0].pyramid.halfWidth2 = (atan(bounds.w) - atan(bounds.y)) * 0.5f;
      DebugDataBuffer[0].pyramid.solidAngle = solidAngle;
      DebugDataBuffer[0].pyramid.bestEdge = bestEdge;
      DebugDataBuffer[0].pyramid.min1 = bounds.x;
      DebugDataBuffer[0].pyramid.max1 = bounds.z;
      DebugDataBuffer[0].pyramid.min2 = bounds.y;
      DebugDataBuffer[0].pyramid.max2 = bounds.w;
   }

   static void recordSampleCount(uint32_t count) { DebugDataBuffer[0].sampling.sampleCount = count; }
   static void recordRay(uint32_t i, float32_t3 dir, float32_t pdf) { DebugDataBuffer[0].sampling.rayData[i] = float32_t4(dir, pdf); }

   static void recordFrameEnd(uint32_t3 region, uint32_t configIndex, uint32_t silSize, uint32_t silData, uint32_t vertexIndices[6], uint32_t validSampleCount)
   {
      InterlockedAdd(DebugDataBuffer[0].sampling.validSampleCount, validSampleCount);
      InterlockedAdd(DebugDataBuffer[0].sampling.threadCount, 1u);
      DebugDataBuffer[0].silhouette.region = region;
      DebugDataBuffer[0].silhouette.silhouetteIndex = configIndex;
      DebugDataBuffer[0].silhouette.silhouetteVertexCount = silSize;
      for (uint32_t i = 0; i < 6; i++)
         DebugDataBuffer[0].silhouette.vertices[i] = vertexIndices[i];
      DebugDataBuffer[0].silhouette.silhouette = silData;
   }
#else
   static void recordClippedVertex(uint32_t slot, float32_t3 pos, uint32_t originalIndex) {}
   static void recordClipResult(uint32_t vertexCount, uint32_t clipMask, uint32_t clipCount, uint32_t rotatedClipMask, uint32_t rotateAmount, uint32_t positiveCount, bool wrapAround, uint32_t rotatedSil) {}
   static void recordTriangleFan(bool luneDetected, uint32_t count, float32_t totalWeight, float32_t solidAngles[5]) {}
   static void recordParallelogram(float32_t area, uint32_t convexMask, uint32_t n3Mask, float32_t2 corner, float32_t2 axisDir, float32_t width, float32_t height) {}
   static void recordPyramid(float32_t3 axis1, float32_t3 axis2, float32_t3 center, float32_t4 bounds, float32_t solidAngle, uint32_t bestEdge) {}
   static void recordSampleCount(uint32_t count) {}
   static void recordRay(uint32_t i, float32_t3 dir, float32_t pdf) {}
   static void recordFrameEnd(uint32_t3 region, uint32_t configIndex, uint32_t silSize,
      uint32_t silData, uint32_t vertexIndices[6], uint32_t validSampleCount) {}
#endif
};

// Module-scope visualization state (per-thread in fragment shaders)
#if VISUALIZE_SAMPLES
static float32_t2 g_visNdc;
static float32_t3 g_visSpherePos;
static float32_t g_visAaWidth;
static float32_t4 g_visColor;
#endif

struct VisContext
{
#if VISUALIZE_SAMPLES
   static void begin(float32_t2 ndc, float32_t3 spherePos, float32_t _aaWidth)
   {
      g_visNdc = ndc;
      g_visSpherePos = spherePos;
      g_visAaWidth = _aaWidth;
      g_visColor = float32_t4(0, 0, 0, 0);
   }

   static void add(float32_t4 c) { g_visColor += c; }
   static float32_t4 flush() { return g_visColor; }

   static float32_t2 ndc() { return g_visNdc; }
   static float32_t3 spherePos() { return g_visSpherePos; }
   static float32_t aaWidth() { return g_visAaWidth; }
   static bool enabled() { return true; }
#else
   static void begin(nbl::hlsl::float32_t2 ndc, nbl::hlsl::float32_t3 spherePos, nbl::hlsl::float32_t aaWidth) {}
   static void add(nbl::hlsl::float32_t4 c) {}
   static nbl::hlsl::float32_t4 flush() { return nbl::hlsl::float32_t4(0, 0, 0, 0); }

   static nbl::hlsl::float32_t2 ndc() { return nbl::hlsl::float32_t2(0, 0); }
   static nbl::hlsl::float32_t3 spherePos() { return nbl::hlsl::float32_t3(0, 0, 0); }
   static nbl::hlsl::float32_t aaWidth() { return 0; }
   static bool enabled() { return false; }
#endif
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_DEBUG_VIS_HLSL_INCLUDED_
