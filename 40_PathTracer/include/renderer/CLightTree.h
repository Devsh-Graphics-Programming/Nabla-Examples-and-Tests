// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_LIGHT_TREE_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_LIGHT_TREE_H_INCLUDED_


#include "nabla.h"
#include "nbl/builtin/hlsl/shapes/aabb.hlsl"
#include "renderer/shaders/light_tree.hlsl" // NonEmitterCustomIndex + SEmitterGPU layout + BDA accessors

#include <span>


namespace nbl::this_example
{

struct SLightTreeLeaf
{
   nbl::hlsl::shapes::AABB<3, float> worldAABB;
   nbl::hlsl::float32_t3             radiance;
   float                             power; // luma(radiance) * surfaceArea, used by tree descent weighting
   uint32_t                          emitterID;
};

struct SLightTreeNode
{
   nbl::hlsl::shapes::AABB<3, float> bbox;
   float                             power;
   uint32_t                          emitterID; // valid only for leaves; ~0u for internal and padding
};

// CWBVH-4 GPU records, aliases of the builtin library's canonical 32 B pack/unpack layout (shared with
// the GPU decoder + round-trip test). The CPU builder fills them via the library pack helpers.
using SLightTreeWideNode = nbl::hlsl::sampling::LightcutTreePackedWideNode;
using SLightTreeLeaf_GPU = nbl::hlsl::sampling::LightcutTreePackedLeaf;
static_assert(sizeof(SLightTreeWideNode) == 32, "Wide-node layout must be 32 B");
static_assert(sizeof(SLightTreeLeaf_GPU) == 32, "Leaf record must be 32 B");

struct SLightTree
{
   // CPU scratch tree (float bbox/power) used during build; the GPU buffers below derive from it.
   nbl::core::vector<SLightTreeNode> nodes;

   // GPU buffers (in CPU memory until uploaded).
   nbl::core::vector<SLightTreeWideNode> wideNodes; // (numLeavesPadded - 1) / 3 entries; empty for single-leaf tree
   nbl::core::vector<SLightTreeLeaf_GPU> leaves; // numLeavesPadded entries (incl. padding sentinels)

   nbl::core::vector<uint32_t> aliasEntries; // packA<Log2N>-packed words, size = aliasTableSize
   nbl::core::vector<float>    aliasPdf; // per-bin pdf, size = aliasTableSize
   uint32_t                    aliasTableSize = 0; // may be userN or userN+1 (PoT-dodge from AliasTableBuilder)

   // Per-internal-node power-weighted alias tables (one per wide-node W) over the leaves in W's
   // subtree, for the descent's early-stop: when per-level child weights stop discriminating, draw one
   // leaf from W's table in O(1). Observer-independent (power = luma * surfaceArea). Flat layout:
   // subtreeAliasOffsets[W] is W's first entry (offsets[firstLeafIndex] = total); subtreeLeafBases[W]
   // is the leftmost leaf, so entry k maps to heap index firstLeafIndex + subtreeLeafBases[W] + k.
   nbl::core::vector<uint32_t> subtreeAliasOffsets; // size = numInternalNodes + 1
   nbl::core::vector<uint32_t> subtreeLeafBases; // size = numInternalNodes
   nbl::core::vector<uint32_t> subtreeAliasEntries; // size = subtreeAliasOffsets.back()
   nbl::core::vector<float>    subtreeAliasPdfs; // size = subtreeAliasOffsets.back()

   // emitterToLeafIdx[emitterID] = heap index of that emitter's leaf in `nodes`.
   // Used by the backward pdf walk; the leaf-array position is `heapIdx - firstLeafIndex`.
   nbl::core::vector<uint32_t> emitterToLeafIdx;
   // Per-emitter quantization quality: max-axis ratio of quantized / precise leaf extent (1.0 = exact,
   // >1.0 = the inflated box the descent's weight evaluator sees). Indexed by emitterID.
   nbl::core::vector<float> quantQuality;
   uint32_t                 numLeavesActual = 0;
   uint32_t                 numLeavesPadded = 0;
   uint32_t                 firstLeafIndex  = 0;
};

SLightTree buildLightTreeCPU(std::span<const SLightTreeLeaf> leaves);

// CPU-side per-emitter backward NEE pdf at a fixed probe (point + normal), mirroring
// StochasticLightcutTreeSampler::backwardPdf over `tree.nodes` so the debug viz matches the shader
// without a per-pixel descent. `out` size >= tree.emitterToLeafIdx.size(); out[emitterID] = that
// emitter's backward pdf.
void computePerEmitterBackwardPdfCPU(const SLightTree& tree, const nbl::hlsl::float32_t3& probePoint, const nbl::hlsl::float32_t3& probeNormal, std::span<float> out);

// Top-down descent mirroring StochasticLightcutTreeSampler::generate with a fixed `u`; returns the
// HEAP index of the leaf it lands on (~0u if a level has no live child). Debug viz marks this path.
uint32_t computeDeterministicDescentLeafCPU(const SLightTree& tree, const nbl::hlsl::float32_t3& probePoint, const nbl::hlsl::float32_t3& probeNormal, float u);

// Per-HEAP-NODE cumulative descent probability at a fixed probe (root == 1, child = parent *
// childWeight/siblingSum; for a leaf this is its backward NEE pdf), so the debug viz can tint a whole
// cluster. `out` size >= tree.nodes.size(); unreachable nodes get 0.
void computeNodePdfsCPU(const SLightTree& tree, const nbl::hlsl::float32_t3& probePoint, const nbl::hlsl::float32_t3& probeNormal, std::span<float> out);

using blas_cache_t = nbl::core::unordered_map<const nbl::asset::ICPUGeometryCollection*, nbl::core::smart_refctd_ptr<nbl::asset::ICPUBottomLevelAccelerationStructure>>;

struct SEmitterSelectionDiagnostics
{
   uint32_t instancesTotal      = 0;
   uint32_t skippedNonStatic    = 0;
   uint32_t skippedNoCollection = 0; // BLAS -> collection lookup miss
   uint32_t skippedEmptyAABB    = 0; // collection produced no usable model AABB
   uint32_t eligible            = 0; // passed all the above filters
   uint32_t pickedByRng         = 0; // emitters chosen by the density roll
   uint32_t forcedPick          = 0; // forced when density rolled zero on a non-empty eligible set

   // Per-leaf bbox-extent histogram of the picked leaves; p95/median > ~100 (or bboxes spanning much of
   // the scene) flags inflated AABBs, e.g. an emitter BLAS that includes housing, not just the emitter.
   float leafBboxMaxExtentMin    = 0.f;
   float leafBboxMaxExtentMedian = 0.f;
   float leafBboxMaxExtentP95    = 0.f;
   float leafBboxMaxExtentMax    = 0.f;
   float leafSurfaceAreaMin      = 0.f;
   float leafSurfaceAreaMedian   = 0.f;
   float leafSurfaceAreaP95      = 0.f;
   float leafSurfaceAreaMax      = 0.f;
   float sceneOverallMaxExtent   = 0.f; // union AABB of every PICKED leaf's longest axis
};

// Picks emitters from the TLAS instance list by seeded RNG and computes each emitter's world AABB from
// its BLAS geometry collection. Sets each instance's instanceCustomIndex to a per-geometry BASE (prefix
// sum of geometry counts), NOT the emitter ID, and fills `outInstancedGeometryToEmitter` (keyed by
// instancedGeometryID = instanceCustomIndex + GeometryIndex(), NonEmitterCustomIndex for non-emissive),
// so the shader resolves a hit's emitter via this map rather than treating instanceCustomIndex as it.
// If emitterDensity > 0 and at least one candidate is eligible but the roll picks none, one is forced.
nbl::core::vector<SLightTreeLeaf> selectRandInstancesAsEmitterLeaves(std::span<nbl::asset::ICPUTopLevelAccelerationStructure::PolymorphicInstance> instances,
   const blas_cache_t&                                                                                                              blasCache,
   float                                                                                                                            emitterDensity,
   uint32_t                                                                                                                         rngSeed,
   nbl::core::vector<uint32_t>&                                                                                                     outInstancedGeometryToEmitter,
   SEmitterSelectionDiagnostics*                                                                                                    diagnostics = nullptr);

} // namespace nbl::this_example
#endif
