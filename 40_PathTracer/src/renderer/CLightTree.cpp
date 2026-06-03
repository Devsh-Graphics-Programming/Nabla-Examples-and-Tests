// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "renderer/CLightTree.h"

#include "nbl/builtin/hlsl/morton.hlsl"
#include "nbl/builtin/hlsl/sampling/alias_table_builder.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>


namespace nbl::this_example
{
namespace
{
using aabb_t   = nbl::hlsl::shapes::AABB<3, float>;
using point_t  = aabb_t::point_t;
using morton_t = nbl::hlsl::morton::code<false, 10, 3>;

bool isAABBValid(const aabb_t& a)
{ return a.minVx.x <= a.maxVx.x && a.minVx.y <= a.maxVx.y && a.minVx.z <= a.maxVx.z; }

nbl::hlsl::float32_t3x4 composeAffine(const nbl::hlsl::float32_t3x4& A, const nbl::hlsl::float32_t3x4& B)
{
   nbl::hlsl::float32_t3x4 R;
   for (int r = 0; r < 3; ++r)
   {
      for (int c = 0; c < 3; ++c)
         R[r][c] = A[r][0] * B[0][c] + A[r][1] * B[1][c] + A[r][2] * B[2][c];
      R[r][3] = A[r][0] * B[0][3] + A[r][1] * B[1][3] + A[r][2] * B[2][3] + A[r][3];
   }
   return R;
}

point_t transformPoint(const nbl::hlsl::float32_t3x4& M, const point_t& p)
{ return point_t(M[0][0] * p.x + M[0][1] * p.y + M[0][2] * p.z + M[0][3], M[1][0] * p.x + M[1][1] * p.y + M[1][2] * p.z + M[1][3], M[2][0] * p.x + M[2][1] * p.y + M[2][2] * p.z + M[2][3]); }

aabb_t storedModelAABB(const nbl::asset::ICPUPolygonGeometry* poly)
{
   aabb_t out = aabb_t::create();
   poly->visitAABB(
      [&out](const auto& aabb)
      {
         aabb_t cand;
         cand.minVx = point_t(float(aabb.minVx.x), float(aabb.minVx.y), float(aabb.minVx.z));
         cand.maxVx = point_t(float(aabb.maxVx.x), float(aabb.maxVx.y), float(aabb.maxVx.z));
         if (isAABBValid(cand))
         {
            out.addPoint(cand.minVx);
            out.addPoint(cand.maxVx);
         }
      });
   return out;
}

aabb_t walkedModelAABB(const nbl::asset::ICPUPolygonGeometry* poly)
{
   aabb_t      out     = aabb_t::create();
   const auto& posView = poly->getPositionView();
   if (!posView)
      return out;
   const auto count = posView.getElementCount();
   for (uint64_t i = 0; i < count; ++i)
   {
      nbl::hlsl::float32_t4 pt = { 0, 0, 0, 1 };
      if (posView.template decodeElement<nbl::hlsl::float32_t4, uint64_t>(i, pt))
         out.addPoint(point_t(pt.x, pt.y, pt.z));
   }
   return out;
}

void unionTransformedAABBInto(const aabb_t& src, const nbl::hlsl::float32_t3x4& M, aabb_t& out)
{
   if (!isAABBValid(src))
      return;
   for (int c = 0; c < 8; ++c)
   {
      const point_t p = point_t((c & 1) ? src.maxVx.x : src.minVx.x, (c & 2) ? src.maxVx.y : src.minVx.y, (c & 4) ? src.maxVx.z : src.minVx.z);
      out.addPoint(transformPoint(M, p));
   }
}

aabb_t worldSpaceAABB(const nbl::asset::ICPUGeometryCollection* coll, const nbl::hlsl::float32_t3x4& M)
{
   aabb_t out = aabb_t::create();
   if (!coll)
      return out;
   for (const auto& ref : coll->getGeometries())
   {
      const auto* geo = ref.geometry.get();
      if (!geo || geo->getPrimitiveType() != nbl::asset::IGeometryBase::EPrimitiveType::Polygon)
         continue;
      const auto* poly = static_cast<const nbl::asset::ICPUPolygonGeometry*>(geo);
      // world = M * ref.transform * pos (mesh positions are in the geometry-ref's local frame).
      const nbl::hlsl::float32_t3x4 effective = ref.hasTransform() ? composeAffine(M, ref.transform) : M;
      aabb_t                        model     = storedModelAABB(poly);
      if (!isAABBValid(model))
         model = walkedModelAABB(poly);
      unionTransformedAABBInto(model, effective, out);
   }
   return out;
}

SLightTreeNode mergeNodes(const SLightTreeNode& a, const SLightTreeNode& b, const SLightTreeNode& c, const SLightTreeNode& d)
{
   SLightTreeNode n;
   n.bbox.minVx = nbl::hlsl::min<point_t>(nbl::hlsl::min<point_t>(a.bbox.minVx, b.bbox.minVx), nbl::hlsl::min<point_t>(c.bbox.minVx, d.bbox.minVx));
   n.bbox.maxVx = nbl::hlsl::max<point_t>(nbl::hlsl::max<point_t>(a.bbox.maxVx, b.bbox.maxVx), nbl::hlsl::max<point_t>(c.bbox.maxVx, d.bbox.maxVx));
   n.power      = a.power + b.power + c.power + d.power;
   n.emitterID  = ~0u;
   return n;
}

uint32_t nextPowerOf4(uint32_t n)
{
   if (n <= 1)
      return 1;
   uint32_t p = std::bit_ceil(n);
   // bit_ceil gives next power of 2; if its bit position is odd, round up to next power of 4.
   if ((std::countr_zero(p) & 1u) != 0u)
      p <<= 1u;
   return p;
}

} // namespace


SLightTree buildLightTreeCPU(std::span<const SLightTreeLeaf> leaves)
{
   SLightTree tree;
   if (leaves.empty())
      return tree;

   const uint32_t numActual = uint32_t(leaves.size());
   // 4-ary heap: pad to power-of-4, total = (4*N - 1)/3, firstLeafIdx = (N - 1)/3.
   const uint32_t numPadded    = nextPowerOf4(numActual);
   const uint32_t totalNodes   = (4 * numPadded - 1) / 3;
   const uint32_t firstLeafIdx = totalNodes - numPadded;

   tree.nodes.resize(totalNodes);
   tree.numLeavesActual = numActual;
   tree.numLeavesPadded = numPadded;
   tree.firstLeafIndex  = firstLeafIdx;

   constexpr uint32_t          kSentinel = ~0u;
   nbl::core::vector<uint32_t> order(numPadded, kSentinel);

   // Morton Z-order in runs of 4: under occlusion, tight clusters collapse the K=16 RIS pool into one
   // cluster and drop shadow-ray survival, so Morton's looser clusters (keeping candidate diversity) win.
   aabb_t centroidBound = aabb_t::create();
   for (const auto& l : leaves)
      centroidBound.addPoint((l.worldAABB.minVx + l.worldAABB.maxVx) * 0.5f);

   // Avoid division by zero when all centroids coincide on some axis.
   const point_t extentSafe = nbl::hlsl::max<point_t>(centroidBound.getExtent(), nbl::hlsl::promote<point_t>(1e-12f));

   struct SCodedIndex
   {
      uint32_t code;
      uint32_t index;
   };
   nbl::core::vector<SCodedIndex> sorted;
   sorted.reserve(numActual);
   for (uint32_t i = 0; i < numActual; ++i)
   {
      const point_t              c         = (leaves[i].worldAABB.minVx + leaves[i].worldAABB.maxVx) * 0.5f;
      const point_t              norm      = (c - centroidBound.minVx) / extentSafe * 1024.f;
      const nbl::hlsl::uint16_t3 quantized = { uint16_t(std::clamp(norm.x, 0.f, 1023.f)), uint16_t(std::clamp(norm.y, 0.f, 1023.f)), uint16_t(std::clamp(norm.z, 0.f, 1023.f)) };
      sorted.push_back({ morton_t::create(quantized).value, i });
   }
   std::ranges::sort(sorted, [](const SCodedIndex& a, const SCodedIndex& b) { return a.code < b.code; });
   for (uint32_t i = 0; i < numActual; ++i)
      order[i] = sorted[i].index;

   tree.emitterToLeafIdx.resize(numActual);
   const aabb_t emptyAABB = aabb_t::create();
   for (uint32_t i = 0; i < numPadded; ++i)
   {
      SLightTreeNode& n   = tree.nodes[firstLeafIdx + i];
      const uint32_t  src = order[i];
      if (src != kSentinel)
      {
         const auto& leaf = leaves[src];
         n.bbox           = leaf.worldAABB;
         n.power          = leaf.power;
         n.emitterID      = leaf.emitterID;
         assert(leaf.emitterID < numActual);
         tree.emitterToLeafIdx[leaf.emitterID] = firstLeafIdx + i;
      }
      else
      {
         n.bbox      = emptyAABB;
         n.power     = 0.f;
         n.emitterID = ~0u;
      }
   }

   for (int32_t i = int32_t(firstLeafIdx) - 1; i >= 0; --i)
   {
      const uint32_t c0 = 4 * uint32_t(i) + 1;
      tree.nodes[i]     = mergeNodes(tree.nodes[c0 + 0], tree.nodes[c0 + 1], tree.nodes[c0 + 2], tree.nodes[c0 + 3]);
   }

   // ----- Emit GPU buffers (CWBVH-4 wide-nodes + precise leaf array) ---------------
   tree.wideNodes.resize(firstLeafIdx);
   tree.quantQuality.assign(numActual, 1.f);
   for (uint32_t W = 0; W < firstLeafIdx; ++W)
   {
      const auto&    parent = tree.nodes[W];
      const uint32_t c0     = 4 * W + 1;
      const point_t  origin = parent.bbox.minVx;
      const point_t  ext    = parent.bbox.getExtent();

      // Byte layout + quantization + power packing are owned by the library (sampling/stochastic_lightcut_tree).
      namespace pk         = nbl::hlsl::sampling;
      const uint32_t expS  = pk::lightcutTreePickBiasedExp(std::max({ ext.x, ext.y, ext.z }));
      const float    scale = pk::lightcutTreeBiasedExpToScale(expS);

      const float parentPowerSafe = parent.power > 0.f ? parent.power : 1.f; // avoid div by 0 in normalization
      uint32_t    childLeafMask   = 0u;
      uint32_t    childPacked[4]  = { 0u, 0u, 0u, 0u };
      for (uint32_t s = 0; s < 4; ++s)
      {
         const uint32_t childHeap = c0 + s;
         const auto&    ch        = tree.nodes[childHeap];
         if (childHeap >= firstLeafIdx)
            childLeafMask |= (1u << s);
         childPacked[s] = pk::lightcutTreePackChild(ch.bbox.minVx - origin, ch.bbox.maxVx - origin, scale, ch.power, parentPowerSafe);
      }

      SLightTreeWideNode& wn = tree.wideNodes[W];
      wn.origin              = origin;
      wn.powExpMask          = pk::lightcutTreePackPowExpMask(parent.power, expS, childLeafMask);
      wn.childPacked         = nbl::hlsl::uint32_t4(childPacked[0], childPacked[1], childPacked[2], childPacked[3]);

      // Decode the node we just packed (same bytes the GPU reads) instead of re-deriving the quantized
      // nibbles by hand. Only leaf children with a real emitter get a quality value.
      const pk::LightcutTreeWideNode<float> decoded = pk::lightcutTreeUnpackWideNode<float>(wn);
      for (uint32_t s = 0; s < 4; ++s)
      {
         const uint32_t childHeap = c0 + s;
         const auto&    ch        = tree.nodes[childHeap];
         if (childHeap >= firstLeafIdx && ch.emitterID != ~0u && ch.emitterID < numActual)
         {
            const auto&   dch               = decoded.children[s];
            const point_t pExt              = ch.bbox.getExtent();
            const float   eps               = 1e-6f;
            const float   rx                = (dch.bboxMax.x - dch.bboxMin.x) / std::max(pExt.x, eps);
            const float   ry                = (dch.bboxMax.y - dch.bboxMin.y) / std::max(pExt.y, eps);
            const float   rz                = (dch.bboxMax.z - dch.bboxMin.z) / std::max(pExt.z, eps);
            tree.quantQuality[ch.emitterID] = std::max({ rx, ry, rz });
         }
      }
   }

   // May contain one trailing zero-power bucket if numActual is a power of two (PoT-dodge from
   // AliasTableBuilder). AliasTableBuilder expects std::vector (std::allocator), not nbl::core::vector
   // (aligned_allocator), so use std::vector for the scratch then copy out.
   {
      std::vector<float> weights(numActual);
      for (uint32_t i = 0; i < numActual; ++i)
         weights[i] = leaves[i].power;
      std::vector<float>    prob;
      std::vector<uint32_t> alias;
      std::vector<float>    pdfs;
      const uint32_t        tableSize = nbl::hlsl::sampling::AliasTableBuilder<float>::build(std::span<const float>(weights.data(), weights.size()), prob, alias, pdfs);
      tree.aliasTableSize             = tableSize;
      tree.aliasEntries.resize(tableSize);
      // Log2N = NBL_LIGHTTREE_ALIAS_LOG2N (single source of truth in light_tree.hlsl): supports up to
      // 2^Log2N emitters, the rest of the word is stayProb precision. Must match the GPU AliasSampler.
      nbl::hlsl::sampling::AliasTableBuilder<float>::packA<NBL_LIGHTTREE_ALIAS_LOG2N>(std::span<const float>(prob.data(), prob.size()), std::span<const uint32_t>(alias.data(), alias.size()), tree.aliasEntries.data());
      tree.aliasPdf.resize(tableSize);
      for (uint32_t i = 0; i < tableSize; ++i)
         tree.aliasPdf[i] = pdfs[i];
   }

   // Per-internal-node alias tables for the descent's early-stop path. Padding leaves have power 0 and
   // never get picked; an all-padding subtree produces an empty table (offset[W] == offset[W+1]) and the
   // runtime bails to ~0u.
   {
      const uint32_t numInternalNodes = firstLeafIdx;
      tree.subtreeAliasOffsets.resize(numInternalNodes + 1);
      tree.subtreeLeafBases.resize(numInternalNodes);

      auto subtreeLeafRange = [&](uint32_t W) -> std::pair<uint32_t, uint32_t>
      {
         uint32_t left = W, right = W;
         while (left < firstLeafIdx)
            left = 4u * left + 1u;
         while (right < firstLeafIdx)
            right = 4u * right + 4u;
         return { left - firstLeafIdx, right - firstLeafIdx };
      };

      // Build into per-W scratch first; concat into flat buffers after offsets are known. Peak
      // memory ~= 2x final size; acceptable on CPU for our scene sizes.
      nbl::core::vector<nbl::core::vector<uint32_t>> perWEntries(numInternalNodes);
      nbl::core::vector<nbl::core::vector<float>>    perWPdfs(numInternalNodes);

      uint32_t totalEntries = 0;
      for (uint32_t W = 0; W < numInternalNodes; ++W)
      {
         const auto     range        = subtreeLeafRange(W);
         const uint32_t leafBase     = range.first;
         const uint32_t leafCount    = range.second - range.first + 1u;
         tree.subtreeLeafBases[W]    = leafBase;
         tree.subtreeAliasOffsets[W] = totalEntries;

         // power = luma * surfaceArea, the same metric as the global table.
         std::vector<float> weights(leafCount);
         float              wSum = 0.f;
         for (uint32_t k = 0; k < leafCount; ++k)
         {
            const auto& leafNode = tree.nodes[firstLeafIdx + leafBase + k];
            const float w        = (leafNode.emitterID == ~0u) ? 0.f : leafNode.power;
            weights[k]           = w;
            wSum += w;
         }

         if (wSum <= 0.f)
            continue; // all-padding subtree; leave entries/pdfs empty for this W.

         std::vector<float>    prob;
         std::vector<uint32_t> alias;
         std::vector<float>    pdfs;
         const uint32_t        tableSize = nbl::hlsl::sampling::AliasTableBuilder<float>::build(std::span<const float>(weights.data(), weights.size()), prob, alias, pdfs);

         perWEntries[W].resize(tableSize);
         nbl::hlsl::sampling::AliasTableBuilder<float>::packA<NBL_LIGHTTREE_ALIAS_LOG2N>(std::span<const float>(prob.data(), prob.size()), std::span<const uint32_t>(alias.data(), alias.size()), perWEntries[W].data());
         perWPdfs[W].resize(tableSize);
         for (uint32_t k = 0; k < tableSize; ++k)
            perWPdfs[W][k] = pdfs[k];

         totalEntries += tableSize;
      }
      tree.subtreeAliasOffsets[numInternalNodes] = totalEntries;

      tree.subtreeAliasEntries.resize(totalEntries);
      tree.subtreeAliasPdfs.resize(totalEntries);
      for (uint32_t W = 0; W < numInternalNodes; ++W)
      {
         const uint32_t off = tree.subtreeAliasOffsets[W];
         const uint32_t cnt = tree.subtreeAliasOffsets[W + 1] - off;
         for (uint32_t k = 0; k < cnt; ++k)
         {
            tree.subtreeAliasEntries[off + k] = perWEntries[W][k];
            tree.subtreeAliasPdfs[off + k]    = perWPdfs[W][k];
         }
      }
   }

   tree.leaves.resize(numPadded);
   for (uint32_t k = 0; k < numPadded; ++k)
   {
      const auto&         src = tree.nodes[firstLeafIdx + k];
      SLightTreeLeaf_GPU& dst = tree.leaves[k];
      dst.bboxMin             = src.bbox.minVx;
      dst.bboxMax             = src.bbox.maxVx;
      dst.emitterID           = (src.emitterID == ~0u) ? nbl::hlsl::sampling::LightcutTreePackedNoEmitter : src.emitterID;
      dst._pad                = 0u;
   }

   return tree;
}


nbl::core::vector<SLightTreeLeaf> selectRandInstancesAsEmitterLeaves(std::span<nbl::asset::ICPUTopLevelAccelerationStructure::PolymorphicInstance> instances,
   const blas_cache_t&                                                                                                                             blasCache,
   float                                                                                                                                           emitterDensity,
   uint32_t                                                                                                                                        rngSeed,
   nbl::core::vector<uint32_t>&                                                                                                                    outInstancedGeometryToEmitter,
   SEmitterSelectionDiagnostics*                                                                                                                   diagnostics)
{
   using namespace nbl::asset;

   nbl::core::unordered_map<const ICPUBottomLevelAccelerationStructure*, const ICPUGeometryCollection*> blasToCollection;
   blasToCollection.reserve(blasCache.size());
   for (const auto& [coll, blas] : blasCache)
      blasToCollection.emplace(blas.get(), coll);

   SEmitterSelectionDiagnostics diag;
   diag.instancesTotal = uint32_t(instances.size());

   nbl::core::vector<uint32_t> instanceToEmitterID(instances.size(), NonEmitterCustomIndex);

   struct SEligible
   {
      uint32_t instanceIdx;
      aabb_t   worldAABB;
   };
   nbl::core::vector<SEligible> eligible;
   eligible.reserve(instances.size());
   for (uint32_t i = 0; i < instances.size(); ++i)
   {
      auto& polyInst = instances[i];
      if (polyInst.getType() != ITopLevelAccelerationStructure::INSTANCE_TYPE::STATIC)
      {
         ++diag.skippedNonStatic;
         continue;
      }
      auto& staticInst = std::get<ICPUTopLevelAccelerationStructure::StaticInstance>(polyInst.instance);

      if (i == 0)
         continue;

      const auto it = blasToCollection.find(staticInst.base.blas.get());
      if (it == blasToCollection.end())
      {
         ++diag.skippedNoCollection;
         continue;
      }
      const aabb_t world = worldSpaceAABB(it->second, staticInst.transform);
      if (world.minVx.x > world.maxVx.x)
      {
         ++diag.skippedEmptyAABB;
         continue;
      }
      eligible.push_back({ i, world });
   }
   diag.eligible = uint32_t(eligible.size());

   std::mt19937                          rng(rngSeed);
   std::uniform_real_distribution<float> uni(0.f, 1.f);

   auto randomRadiance = [&]() -> nbl::hlsl::float32_t3
   {
      const float lo    = -5.0f;
      const float hi    = 4.0f;
      const float hue   = uni(rng);
      const float value = std::exp2(lo + (hi - lo) * uni(rng));
      const float h6    = hue * 6.f;
      const float x     = value * (1.f - std::abs(std::fmod(h6, 2.f) - 1.f));
      if (h6 < 1.f)
         return { value, x, 0.f };
      if (h6 < 2.f)
         return { x, value, 0.f };
      if (h6 < 3.f)
         return { 0.f, value, x };
      if (h6 < 4.f)
         return { 0.f, x, value };
      if (h6 < 5.f)
         return { x, 0.f, value };
      return { value, 0.f, x };
   };

   // A/B knob: true rescales radiance so luma*area is ~constant across emitters (strips the power
   // signal, forcing the descent onto orientation/distance only); false keeps natural luma*area power,
   // the regime where a power+geometry tree should beat the power-only alias table.
   constexpr bool equalizeEmitterPower = false;

   float meanArea = 0.f;
   for (const auto& e : eligible)
   {
      const auto ext = e.worldAABB.getExtent();
      meanArea += 2.f * (ext.x * ext.y + ext.y * ext.z + ext.z * ext.x);
   }
   if (!eligible.empty())
      meanArea /= float(eligible.size());

   nbl::core::vector<SLightTreeLeaf> leaves;
   leaves.reserve(eligible.size());
   auto pushAsLeaf = [&](const SEligible& e)
   {
      const uint32_t emitterID = uint32_t(leaves.size());
      assert(emitterID < NonEmitterCustomIndex);
      const auto  extent                 = e.worldAABB.getExtent();
      const float surfaceArea            = 2.f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
      const float radianceScale          = (equalizeEmitterPower && surfaceArea > 0.f) ? (meanArea / surfaceArea) : 1.f;
      const auto  radiance               = randomRadiance() * radianceScale;
      const float luma                   = 0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z;
      instanceToEmitterID[e.instanceIdx] = emitterID;
      leaves.push_back({ e.worldAABB, radiance, luma * surfaceArea, emitterID });
   };
   for (const auto& e : eligible)
   {
      if (uni(rng) >= emitterDensity)
         continue;
      ++diag.pickedByRng;
      pushAsLeaf(e);
   }

   // Guarantee at least one emitter so small scenes don't render to a dead tree.
   if (leaves.empty() && !eligible.empty() && emitterDensity > 0.f)
   {
      std::uniform_int_distribution<size_t> pickOne(0, eligible.size() - 1);
      pushAsLeaf(eligible[pickOne(rng)]);
      ++diag.forcedPick;
   }

   // Inflated AABBs (collection includes non-emitting geometry) show up as a large p95/median spread.
   if (!leaves.empty())
   {
      std::vector<float> maxExt;
      maxExt.reserve(leaves.size());
      std::vector<float> surfA;
      surfA.reserve(leaves.size());
      float sceneMaxExt = 0.f;
      for (const auto& l : leaves)
      {
         const auto  e = l.worldAABB.getExtent();
         const float m = std::max(e.x, std::max(e.y, e.z));
         const float s = 2.f * (e.x * e.y + e.y * e.z + e.z * e.x);
         maxExt.push_back(m);
         surfA.push_back(s);
         sceneMaxExt = std::max(sceneMaxExt, m);
      }
      std::sort(maxExt.begin(), maxExt.end());
      std::sort(surfA.begin(), surfA.end());
      auto pick = [&](const std::vector<float>& v, float q)
      {
         const size_t i = std::min(v.size() - 1, size_t(q * float(v.size())));
         return v[i];
      };
      diag.leafBboxMaxExtentMin    = maxExt.front();
      diag.leafBboxMaxExtentMedian = pick(maxExt, 0.5f);
      diag.leafBboxMaxExtentP95    = pick(maxExt, 0.95f);
      diag.leafBboxMaxExtentMax    = maxExt.back();
      diag.leafSurfaceAreaMin      = surfA.front();
      diag.leafSurfaceAreaMedian   = pick(surfA, 0.5f);
      diag.leafSurfaceAreaP95      = pick(surfA, 0.95f);
      diag.leafSurfaceAreaMax      = surfA.back();
      diag.sceneOverallMaxExtent   = sceneMaxExt;
   }

   outInstancedGeometryToEmitter.clear();
   for (uint32_t i = 0; i < uint32_t(instances.size()); ++i)
   {
      auto&          base      = instances[i].getBase();
      const auto     it        = blasToCollection.find(base.blas.get());
      const uint32_t geomCount = (it != blasToCollection.end()) ? uint32_t(it->second->getGeometries().size()) : 1u;
      const uint32_t emitterID = instanceToEmitterID[i];
      base.instanceCustomIndex = uint32_t(outInstancedGeometryToEmitter.size());
      for (uint32_t g = 0; g < geomCount; ++g)
         outInstancedGeometryToEmitter.push_back(emitterID);
   }

   if (diagnostics)
      *diagnostics = diag;
   return leaves;
}

// Deliberately a separate C++ copy of HLSL sampling::lightcutTreeChildWeight (mode 0), NOT a #include
// of the dual-compile header, so this stays a plain C++ translation unit with no DXC-isms.
static float lightcutTreeChildWeightCPU(const SLightTreeNode& c, const nbl::hlsl::float32_t3& x, const nbl::hlsl::float32_t3& n)
{
   if (!(c.power > 0.f))
      return 0.f;

   const point_t ext        = c.bbox.maxVx - c.bbox.minVx;
   const float   halfDiagSq = 0.25f * nbl::hlsl::dot<point_t>(ext, ext);

   // Importance distance to the NEAREST point of the bbox (conservative upper bound on 1/d^2), floored
   // at the cluster's half-diagonal^2 so x inside/near the box stays finite. Mirrors HLSL
   // lightcutTreeChildWeight (mode 0); keep in sync.
   const point_t center         = 0.5f * (c.bbox.minVx + c.bbox.maxVx);
   const point_t dToCentroid    = center - x;
   const float   centroidDistSq = nbl::hlsl::dot<point_t>(dToCentroid, dToCentroid);
   const point_t dNear          = nbl::hlsl::max<point_t>(nbl::hlsl::max<point_t>(c.bbox.minVx - x, x - c.bbox.maxVx), nbl::hlsl::promote<point_t>(0.f));
   const float   minDistSq      = nbl::hlsl::dot<point_t>(dNear, dNear);
   const float   distSq         = std::max(minDistSq, halfDiagSq);

   // Receiver-side cosine UPPER BOUND over the whole bbox: widen the
   // centroid-direction cosine by the bbox angular radius alpha
   // (sin(alpha) = halfDiag/distToCentroid) and take cos(max(phi - alpha, 0)).
   // orientFactor == 0 doubles as the below-horizon cull.
   const float distToCentroidSq = std::max(centroidDistSq, halfDiagSq);
   const float rcpDist          = 1.f / std::sqrt(distToCentroidSq);
   const float cosPhi           = nbl::hlsl::dot<point_t>(n, dToCentroid) * rcpDist;
   const float sinAlpha         = std::min(std::sqrt(halfDiagSq) * rcpDist, 1.f);
   const float cosAlpha         = std::sqrt(std::max(1.f - sinAlpha * sinAlpha, 0.f));
   const float sinPhi           = std::sqrt(std::max(1.f - cosPhi * cosPhi, 0.f));
   const float orientFactor     = (cosPhi >= cosAlpha) ? 1.f : std::max(cosPhi * cosAlpha + sinPhi * sinAlpha, 0.f);
   if (!(orientFactor > 0.f))
      return 0.f;

   return c.power * orientFactor / distSq;
}

void computePerEmitterBackwardPdfCPU(const SLightTree& tree, const nbl::hlsl::float32_t3& probePoint, const nbl::hlsl::float32_t3& probeNormal, std::span<float> out)
{
   const uint32_t numEmitters = uint32_t(tree.emitterToLeafIdx.size());
   assert(out.size() >= numEmitters);

   for (uint32_t e = 0u; e < numEmitters; ++e)
   {
      const uint32_t leafHeap = tree.emitterToLeafIdx[e];

      // Single-leaf tree: there are no internal nodes, leaf is the root.
      if (leafHeap == 0u)
      {
         out[e] = 1.f;
         continue;
      }
      if (tree.firstLeafIndex == 0u || tree.nodes.empty())
      {
         out[e] = 0.f;
         continue;
      }

      float    pdf  = 1.f;
      uint32_t heap = leafHeap;
      for (uint32_t step = 0u; step < 32u && heap != 0u; ++step)
      {
         const uint32_t parent   = (heap - 1u) / 4u;
         const uint32_t selfSlot = (heap - 1u) - 4u * parent;

         float w[4] = { 0.f, 0.f, 0.f, 0.f };
         for (uint32_t s = 0u; s < 4u; ++s)
         {
            const uint32_t childHeap = 4u * parent + 1u + s;
            w[s]                     = lightcutTreeChildWeightCPU(tree.nodes[childHeap], probePoint, probeNormal);
         }
         const float wSum = w[0] + w[1] + w[2] + w[3];
         if (!(wSum > 0.f))
         {
            pdf = 0.f;
            break;
         }
         pdf *= w[selfSlot] / wSum;
         heap = parent;
      }
      out[e] = pdf;
   }
}

uint32_t computeDeterministicDescentLeafCPU(const SLightTree& tree, const nbl::hlsl::float32_t3& probePoint, const nbl::hlsl::float32_t3& probeNormal, float u)
{
   // Single-leaf tree: the only leaf is the root.
   if (tree.firstLeafIndex == 0u || tree.nodes.empty())
      return 0u;

   uint32_t W  = 0u;
   float    xi = u;
   for (uint32_t step = 0u; step < 32u; ++step)
   {
      float w[4] = { 0.f, 0.f, 0.f, 0.f };
      for (uint32_t s = 0u; s < 4u; ++s)
      {
         const uint32_t childHeap = 4u * W + 1u + s;
         w[s]                     = lightcutTreeChildWeightCPU(tree.nodes[childHeap], probePoint, probeNormal);
      }
      const float wSum = w[0] + w[1] + w[2] + w[3];
      if (!(wSum > 0.f))
         return ~0u;

      // Matches the shader's branchless CDF selection.
      float    t    = xi * wSum;
      uint32_t slot = 3u;
      for (uint32_t s = 0u; s < 3u; ++s)
      {
         if (t < w[s])
         {
            slot = s;
            break;
         }
         t -= w[s];
      }
      xi = (w[slot] > 0.f) ? (t / w[slot]) : 0.f;

      const uint32_t childHeap = 4u * W + 1u + slot;
      if (childHeap >= tree.firstLeafIndex) // leaves occupy [firstLeafIndex, ...)
         return childHeap;
      W = childHeap;
   }
   return ~0u;
}

void computeNodePdfsCPU(const SLightTree& tree, const nbl::hlsl::float32_t3& probePoint, const nbl::hlsl::float32_t3& probeNormal, std::span<float> out)
{
   const uint32_t total = uint32_t(tree.nodes.size());
   assert(out.size() >= total);
   for (uint32_t i = 0u; i < total; ++i)
      out[i] = 0.f;
   if (total == 0u)
      return;

   // Root always carries the full descent mass. Internal nodes occupy heap
   // [0,firstLeafIndex); a parent's index is always < its children's, so a single
   // ascending pass fills children before they are themselves visited as parents.
   out[0] = 1.f;
   for (uint32_t W = 0u; W < tree.firstLeafIndex; ++W)
   {
      const float parentPdf = out[W];
      if (!(parentPdf > 0.f))
         continue; // unreachable subtree: leave descendants at 0

      float w[4] = { 0.f, 0.f, 0.f, 0.f };
      for (uint32_t s = 0u; s < 4u; ++s)
         w[s] = lightcutTreeChildWeightCPU(tree.nodes[4u * W + 1u + s], probePoint, probeNormal);
      const float wSum = w[0] + w[1] + w[2] + w[3];
      if (!(wSum > 0.f))
         continue;

      for (uint32_t s = 0u; s < 4u; ++s)
         out[4u * W + 1u + s] = parentPdf * w[s] / wSum;
   }
}

} // namespace nbl::this_example
