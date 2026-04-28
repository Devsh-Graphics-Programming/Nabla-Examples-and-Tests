//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_COMMON_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_COMMON_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#define MAX_SILHOUETTE_VERTICES 7

namespace nbl
{
namespace hlsl
{
    
// Sampling mode enum
enum SAMPLING_MODE : uint32_t
{
   TRIANGLE_SOLID_ANGLE,
   TRIANGLE_PROJECTED_SOLID_ANGLE,
   PROJECTED_PARALLELOGRAM_SOLID_ANGLE,
   SYMMETRIC_PYRAMID_SOLID_ANGLE_RECTANGLE,
   SYMMETRIC_PYRAMID_SOLID_ANGLE_BIQUADRATIC,
   SYMMETRIC_PYRAMID_SOLID_ANGLE_BILINEAR,
   SYMMETRIC_PYRAMID_PROJECTED_SOLID_ANGLE_RECTANGLE,
   SILHOUETTE_CREATION_ONLY,
   PYRAMID_CREATION_ONLY,
   Count
};

struct ResultData
{
   struct SilhouetteData
   {
      uint32_t3 region;
      uint32_t silhouetteIndex;
      uint32_t silhouetteVertexCount;
      uint32_t silhouette;
      uint32_t vertices[6];

      // Clipping
      uint32_t clipMask;
      uint32_t clipCount;
      uint32_t rotatedClipMask;
      uint32_t rotateAmount;
      uint32_t positiveVertCount;
      uint32_t wrapAround;
      uint32_t rotatedSil;
      uint32_t edgeVisibilityMismatch;

      // Clipped output (layout matches ClippedSilhouette: vertices[7] then count)
      float32_t3 clippedVertices[MAX_SILHOUETTE_VERTICES];
      uint32_t clippedVertexCount;
      uint32_t clippedVertexIndices[MAX_SILHOUETTE_VERTICES];
   } silhouette;

   struct TriangleFanData
   {
      uint32_t maxTrianglesExceeded;
      uint32_t sphericalLuneDetected;
      uint32_t triangleCount;
      float32_t solidAngles[5];
      float32_t totalSolidAngles;
   } triangleFan;

   struct ParallelogramData
   {
      float32_t2 corners[4];
      uint32_t edgeIsConvex[4];
      uint32_t n3Mask;
      uint32_t doesNotBound;
      uint32_t failedVertexIndex;
      uint32_t verticesInside;
      uint32_t edgesInside;
      float32_t area;
   } parallelogram;

   struct PyramidData
   {
      float32_t3 axis1;            // First caliper axis direction
      float32_t3 axis2;            // Second caliper axis direction
      float32_t3 center;           // Silhouette center direction
      float32_t halfWidth1;        // Half-width along axis1 (sin-space)
      float32_t halfWidth2;        // Half-width along axis2 (sin-space)
      float32_t offset1;           // Center offset along axis1
      float32_t offset2;           // Center offset along axis2
      float32_t solidAngle;        // Bounding region solid angle
      uint32_t bestEdge;           // Which edge produced best caliper
      float32_t min1;              // Min dot product along axis1
      float32_t max1;              // Max dot product along axis1
      float32_t min2;              // Min dot product along axis2
      float32_t max2;              // Max dot product along axis2
      uint32_t axis2BiggerThanAxis1;
   } pyramid;

   struct SamplingData
   {
      uint32_t sampleCount;
      uint32_t validSampleCount;
      uint32_t threadCount; // Per-fragment counter, used as divisor for validSampleCount
      float32_t4 rayData[512]; // xyz = direction, w = PDF
   } sampling;
};

struct PushConstants
{
   float32_t3x4 modelMatrix;
   float32_t4 viewport;
   uint32_t sampleCount;
   uint32_t frameIndex;
};

struct PushConstantRayVis
{
   float32_t4x4 viewProjMatrix;
   float32_t3x4 viewMatrix;
   float32_t3x4 modelMatrix;
   float32_t3x4 invModelMatrix;
   float32_t4 viewport;
   uint32_t frameIndex;
};

struct BenchmarkPushConstants
{
   float32_t3x4 modelMatrix;
   uint32_t sampleCount;        // total samples per thread (= creations * samplesPerCreation)
   uint32_t samplesPerCreation; // inner-loop count; outer-loop count = sampleCount / samplesPerCreation
};

static const float32_t3 colorLUT[27] = {
   float32_t3(0, 0, 0), float32_t3(0.5, 0.5, 0.5),
   float32_t3(1, 0, 0), float32_t3(0, 1, 0), float32_t3(0, 0, 1),
   float32_t3(1, 1, 0), float32_t3(1, 0, 1), float32_t3(0, 1, 1),
   float32_t3(1, 0.5, 0), float32_t3(1, 0.65, 0), float32_t3(0.8, 0.4, 0),
   float32_t3(1, 0.4, 0.7), float32_t3(1, 0.75, 0.8), float32_t3(0.7, 0.1, 0.3),
   float32_t3(0.5, 0, 0.5), float32_t3(0.6, 0.4, 0.8), float32_t3(0.3, 0, 0.5),
   float32_t3(0, 0.5, 0), float32_t3(0.5, 1, 0), float32_t3(0, 0.5, 0.25),
   float32_t3(0, 0, 0.5), float32_t3(0.3, 0.7, 1), float32_t3(0, 0.4, 0.6),
   float32_t3(0.6, 0.4, 0.2), float32_t3(0.8, 0.7, 0.3), float32_t3(0.4, 0.3, 0.1), float32_t3(1, 1, 1)};

#ifndef __HLSL_VERSION
static const char* colorNames[27] = {"Black", "Gray", "Red", "Green", "Blue", "Yellow", "Magenta", "Cyan",
   "Orange", "Light Orange", "Dark Orange", "Pink", "Light Pink", "Deep Rose", "Purple", "Light Purple",
   "Indigo", "Dark Green", "Lime", "Forest Green", "Navy", "Sky Blue", "Teal", "Brown",
   "Tan/Beige", "Dark Brown", "White"};
#endif // __HLSL_VERSION

} // namespace hlsl

} // namespace nbl

static const nbl::hlsl::float32_t CIRCLE_RADIUS = 0.5f;
static const nbl::hlsl::float32_t INV_CIRCLE_RADIUS = 1.0f / CIRCLE_RADIUS;

#endif // _SOLID_ANGLE_VIS_EXAMPLE_COMMON_HLSL_INCLUDED_
