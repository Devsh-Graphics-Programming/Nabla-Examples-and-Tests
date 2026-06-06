//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_SILHOUETTE_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_SILHOUETTE_HLSL_INCLUDED_

// Thin shim over the builtin OBB silhouette. The builtin (in
// nbl/builtin/hlsl/shapes/obb_silhouette.hlsl) is the source of truth for
// ClippedSilhouette / BinSilhouette / SilEdgeNormals; this file re-exports
// them at example-global scope and adds debug-recording wrappers that re-derive
// the intermediates the builtin's debug-free create() doesn't expose.
#include "common.hlsl"
#include "debug_vis.hlsl"
#include "utils.hlsl"
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/shapes/obb.hlsl>
#include <nbl/builtin/hlsl/shapes/obb_silhouette.hlsl>

using namespace nbl;
using namespace nbl::hlsl;

// Re-export builtin types at example-global scope so existing callsites
// (ClippedSilhouette::create, BinSilhouette::data, ...) keep compiling.
using BinSilhouette     = nbl::hlsl::shapes::BinSilhouette;
using ClippedSilhouette = nbl::hlsl::shapes::ClippedSilhouette;
using SilEdgeNormals    = nbl::hlsl::shapes::SilEdgeNormals;

// Debug-recording wrapper around ClippedSilhouette::create. Re-derives clipMask,
// rotateAmount, wrapAround, rotatedClipMask, rotatedSil by re-running the same
// classifier the builtin uses, then emits DebugRecorder::recordClipResult.
ClippedSilhouette createClippedSilhouetteDbg(shapes::OBBView<float32_t> view, float32_t3 shadingPoint)
{
   ClippedSilhouette result = ClippedSilhouette::create(view, shadingPoint);

   const float32_t3 toMin    = view.minCorner - shadingPoint;
   const float32_t3 sqScales = float32_t3(dot(view.columns[0], view.columns[0]), dot(view.columns[1], view.columns[1]), dot(view.columns[2], view.columns[2]));
   const float32_t3 proj     = -float32_t3(dot(view.columns[0], toMin), dot(view.columns[1], toMin), dot(view.columns[2], toMin));
   const uint32_t3  below    = uint32_t3(proj < float32_t3(0, 0, 0));
   const uint32_t3  above    = uint32_t3(proj > sqScales);
   const uint32_t3  region   = uint32_t3(uint32_t3(1u, 1u, 1u) + below - above);
   const uint32_t   configIndex = region.x + region.y * 3u + region.z * 9u;

   BinSilhouette  sil         = BinSilhouette::create(configIndex);
   const uint32_t vertexCount = sil.getVertexCount();
   const uint32_t validMask   = (1u << vertexCount) - 1u;
   uint32_t       clipMask    = 0u;
   NBL_UNROLL
   for (uint32_t i = 0; i < 6; i++)
      clipMask |= (hlsl::select(view.getVertexZ(sil.getVertexIndex(i)) < shadingPoint.z, 1u, 0u)) << i;
   clipMask &= validMask;
   const uint32_t clipCount    = countbits(clipMask);
   const uint32_t invertedMask = ~clipMask & validMask;
   const bool     wrapAround   = (clipMask & (clipMask >> (vertexCount - 1))) != 0u;
   const uint32_t rotateAmount = nbl::hlsl::select(wrapAround, firstbitlow(invertedMask), firstbithigh(clipMask) + 1);
   const uint32_t rotatedClipMask = nbl::hlsl::rotr(clipMask, rotateAmount, vertexCount);

   DebugRecorder::recordClipResult(result.count, clipMask, clipCount, rotatedClipMask, rotateAmount, result.positiveCount, wrapAround, sil.data);
   return result;
}

// Originals tagged with their cube corner index; clip verts use sentinels 23/24.
// Replaces the ClippedSilhouette::recordVertices member that was stripped from
// the builtin. recordClippedVertex is a no-op in release.
void recordClippedSilhouetteVertices(ClippedSilhouette silhouette, float32_t3 vertices[MAX_SILHOUETTE_VERTICES])
{
   for (uint32_t k = 0; k < silhouette.positiveCount; k++)
      DebugRecorder::recordClippedVertex(k, vertices[k], silhouette.cornerIndex(k));
   if (silhouette.count > silhouette.positiveCount)
   {
      DebugRecorder::recordClippedVertex(silhouette.positiveCount, vertices[silhouette.positiveCount], 23u);
      DebugRecorder::recordClippedVertex(silhouette.positiveCount + 1u, vertices[silhouette.positiveCount + 1u], 24u);
   }
}

#endif // _SOLID_ANGLE_VIS_EXAMPLE_SILHOUETTE_HLSL_INCLUDED_
