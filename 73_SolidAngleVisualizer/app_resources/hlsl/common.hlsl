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
    
// Sampling mode enum -- bit-encoded: low byte is the dense ID (0..Count-1),
// high bits are family/variant flags so callers can do `mode & FLAG_X` instead
// of long `||` chains. Host C++ that needs a dense index wraps mode access
// with `(uint32_t(mode) & DENSE_ID_MASK)`.
enum SAMPLING_MODE_FLAGS : uint32_t
{
   // ---- family flags (which underlying geometry/sampler family) ----
   FLAG_PYRAMID       = 0x100,
   FLAG_TRIANGLE      = 0x200,
   FLAG_PARALLELOGRAM = 0x400,
   FLAG_SILHOUETTE    = 0x800,
   FLAG_OBB_FACE      = 0x10000,
   FLAG_OBB_AXES      = 0x20000,

   // ---- variant flags (modifiers on the family) ----
   FLAG_CALIPER     = 0x1000,
   FLAG_PROJECTED   = 0x2000,
   FLAG_BILINEAR    = 0x4000,
   FLAG_CREATE_ONLY = 0x8000,

   // ---- dense-ID extractor for host-side array indexing ----
   DENSE_ID_MASK = 0xFF,

   // ---- modes: dense ID in low byte | family/variant flags ----
   SPH_RECT_FROM_CALIPER_PYRAMID       = 0 | FLAG_PYRAMID | FLAG_CALIPER,
   SPH_RECT_FROM_PYRAMID               = 1 | FLAG_PYRAMID,
   PROJ_SPH_RECT_FROM_PYRAMID          = 2 | FLAG_PYRAMID | FLAG_PROJECTED,

   TRIANGLE_SOLID_ANGLE                = 3 | FLAG_TRIANGLE,
   TRIANGLE_PROJECTED_SOLID_ANGLE      = 4 | FLAG_TRIANGLE | FLAG_PROJECTED,

   PROJECTED_PARALLELOGRAM_SOLID_ANGLE = 5 | FLAG_PARALLELOGRAM,

   BILINEAR_FROM_PYRAMID               = 6 | FLAG_PYRAMID | FLAG_BILINEAR,

   OBB_FACE_DIRECT                     = 7 | FLAG_OBB_FACE,

   SILHOUETTE_CREATION_ONLY            = 8 | FLAG_SILHOUETTE | FLAG_CREATE_ONLY,
   PYRAMID_CREATION_ONLY               = 9 | FLAG_PYRAMID | FLAG_CREATE_ONLY,
   CALIPER_PYRAMID_CREATION_ONLY       = 10 | FLAG_PYRAMID | FLAG_CALIPER | FLAG_CREATE_ONLY,

   Count = 11,  // count of distinct dense IDs
   CountWithoutCreateOnly = Count - 3 // count of modes that aren't "creation only" (i.e. that produce samples)
};

#ifndef __HLSL_VERSION
// Host helpers: dense IDs for array indexing + a parallel array for combo/iteration.
inline uint32_t denseIdOf(SAMPLING_MODE_FLAGS m) { return uint32_t(m) & uint32_t(SAMPLING_MODE_FLAGS::DENSE_ID_MASK); }

constexpr SAMPLING_MODE_FLAGS kAllModes[SAMPLING_MODE_FLAGS::Count] = {
   SAMPLING_MODE_FLAGS::SPH_RECT_FROM_CALIPER_PYRAMID,        // dense 0
   SAMPLING_MODE_FLAGS::SPH_RECT_FROM_PYRAMID,                // dense 1
   SAMPLING_MODE_FLAGS::PROJ_SPH_RECT_FROM_PYRAMID,           // dense 2
   SAMPLING_MODE_FLAGS::TRIANGLE_SOLID_ANGLE,                 // dense 3
   SAMPLING_MODE_FLAGS::TRIANGLE_PROJECTED_SOLID_ANGLE,       // dense 4
   SAMPLING_MODE_FLAGS::PROJECTED_PARALLELOGRAM_SOLID_ANGLE,  // dense 5
   SAMPLING_MODE_FLAGS::BILINEAR_FROM_PYRAMID,                // dense 6
   SAMPLING_MODE_FLAGS::OBB_FACE_DIRECT,                      // dense 7
   SAMPLING_MODE_FLAGS::SILHOUETTE_CREATION_ONLY,             // dense 8
   SAMPLING_MODE_FLAGS::PYRAMID_CREATION_ONLY,                // dense 9
   SAMPLING_MODE_FLAGS::CALIPER_PYRAMID_CREATION_ONLY,        // dense 10
};
#endif

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

      // Clipped output: positions written via DebugRecorder::recordClippedVertex
      // by callers that materialize silhouette vertices; indices recorded in parallel.
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
   float32_t3 shadingPoint;
   uint32_t sampleCount;
   uint32_t frameIndex;
};

struct PushConstantRayVis
{
   float32_t4x4 viewProjMatrix;
   float32_t3x4 viewMatrix;
   float32_t3x4 modelMatrix;
   float32_t3x4 invModelMatrix;
   float32_t3 shadingPoint;
   float32_t4 viewport;
   uint32_t frameIndex;
};

struct BenchmarkPushConstants
{
   float32_t3x4 modelMatrix;
   float32_t3 shadingPoint;
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
