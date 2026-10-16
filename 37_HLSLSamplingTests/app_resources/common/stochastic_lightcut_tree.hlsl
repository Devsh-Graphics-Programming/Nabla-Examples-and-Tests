#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_STOCHASTIC_LIGHTCUT_TREE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_STOCHASTIC_LIGHTCUT_TREE_INCLUDED_

// The geometric test scenarios (DistFalloff / BelowPlane / Depth2) validate the full
// distance + orientation importance weighting (mode 0), which is now the library
// default -- so no #define is needed here. (A header-level define would be unreliable
// anyway: main.cpp includes the library before this header, locking the C++ TU's mode
// before the define could take effect.)

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/stochastic_lightcut_tree.hlsl>

using namespace nbl::hlsl;

NBL_CONSTEXPR uint32_t LightcutTestNumLeaves = 4u;

// One wide-node sits at heap idx 0; the 4 leaves occupy heap idx 1..4.
// firstLeafIdx = 1 for the multi-leaf executor, 0 for the single-leaf one.
NBL_CONSTEXPR uint32_t LightcutTestFirstLeafIdxMulti  = 1u;
NBL_CONSTEXPR uint32_t LightcutTestFirstLeafIdxSingle = 0u;

using LightcutTestWideNode = sampling::LightcutTreeWideNode<float32_t>;
using LightcutTestLeaf     = sampling::LightcutTreeLeaf<float32_t>;
using LightcutTestChild    = sampling::LightcutTreeChild<float32_t>;

// Single-wide-node accessor (we only ever need entry 0 for the synthetic tree).
// Field-wise copy avoids DXC's struct-of-array-of-struct copy quirks.
struct LightcutTestNodeAccessor
{
   using value_type = LightcutTestWideNode;

   template<typename V, typename I>
   void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
   {
      val.childLeafMask = data.childLeafMask;
      NBL_UNROLL
      for (uint32_t s = 0u; s < 4u; ++s)
      {
         val.children[s].bboxMin = data.children[s].bboxMin;
         val.children[s].bboxMax = data.children[s].bboxMax;
         val.children[s].power   = data.children[s].power;
      }
   }

   value_type data;
};

struct LightcutTestLeafAccessor
{
   using value_type = LightcutTestLeaf;

   template<typename V, typename I>
   void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
   {
      val.bboxMin   = data[i].bboxMin;
      val.bboxMax   = data[i].bboxMax;
      val.emitterID = data[i].emitterID;
   }

   value_type data[LightcutTestNumLeaves];
};

// 2-level tree topology for the depth-2 executor: root (heap 0) + 4 children
// (heap 1..4, all internal), each with 4 leaves (heap 5..20). Wide-node count
// = 5, leaf count = 16.
NBL_CONSTEXPR uint32_t LightcutTestNumWideNodesDepth2 = 5u;
NBL_CONSTEXPR uint32_t LightcutTestNumLeavesDepth2    = 16u;
NBL_CONSTEXPR uint32_t LightcutTestFirstLeafIdxDepth2 = 5u;

// Multi-node accessor for the depth-2 executor: indexes into a fixed-size
// table of 5 decoded wide-nodes.
struct LightcutTestNodeArrayAccessor
{
   using value_type = LightcutTestWideNode;

   template<typename V, typename I>
   void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
   {
      val.childLeafMask = data[i].childLeafMask;
      NBL_UNROLL
      for (uint32_t s = 0u; s < 4u; ++s)
      {
         val.children[s].bboxMin = data[i].children[s].bboxMin;
         val.children[s].bboxMax = data[i].children[s].bboxMax;
         val.children[s].power   = data[i].children[s].power;
      }
   }

   value_type data[LightcutTestNumWideNodesDepth2];
};

struct LightcutTestLeafAccessorDepth2
{
   using value_type = LightcutTestLeaf;

   template<typename V, typename I>
   void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
   {
      val.bboxMin   = data[i].bboxMin;
      val.bboxMax   = data[i].bboxMax;
      val.emitterID = data[i].emitterID;
   }

   value_type data[LightcutTestNumLeavesDepth2];
};

struct LightcutTreeInputValues
{
   float32_t u;
};

struct LightcutTreeTestResults
{
   uint32_t  generatedLeafHeap;
   uint32_t  generatedEmitterID;
   float32_t leafBboxMinX;
   float32_t leafBboxMinY;
   float32_t leafBboxMinZ;
   float32_t leafBboxMaxX;
   float32_t leafBboxMaxY;
   float32_t leafBboxMaxZ;
   float32_t forwardPdf;
   float32_t backwardPdf;
   float32_t forwardWeight;
   float32_t backwardWeight;
   float32_t jacobianProduct;
};

// Build a 4-leaf synthetic tree at +/-1 corners in the XZ plane. Powers vary
// so the weighted CDF isn't degenerate; the shading point is at the origin
// with normal +Y so every leaf passes the tangent-plane test. Reference bboxes
// are also returned via the cache so the AABB-tightness sanity assertions can
// compare them against the executor's known-good values.
template<bool SingleLeaf, uint32_t Mode>
struct LightcutTreeTestExecutorImpl
{
   void operator()(NBL_CONST_REF_ARG(LightcutTreeInputValues) input, NBL_REF_ARG(LightcutTreeTestResults) output)
   {
      // Per-leaf reference bboxes + power + emitter id. Picked so the four
      // children sit in distinct octants above the shading plane, keeping the
      // weight function well-conditioned.
      const float32_t3 mins[4]   = { float32_t3(1.0f, 1.0f, 1.0f), float32_t3(-2.0f, 1.0f, 1.0f), float32_t3(1.0f, 2.0f, -2.0f), float32_t3(-2.0f, 3.0f, -2.0f) };
      const float32_t3 maxs[4]   = { float32_t3(2.0f, 1.5f, 2.0f), float32_t3(-1.0f, 1.5f, 2.0f), float32_t3(2.0f, 2.5f, -1.0f), float32_t3(-1.0f, 3.5f, -1.0f) };
      const float32_t  powers[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
      const uint32_t   emIds[4]  = { 100u, 101u, 102u, 103u };

      // Build leaf accessor.
      LightcutTestLeafAccessor leafAcc;
      NBL_UNROLL
      for (uint32_t l = 0u; l < LightcutTestNumLeaves; ++l)
      {
         leafAcc.data[l].bboxMin   = mins[l];
         leafAcc.data[l].bboxMax   = maxs[l];
         leafAcc.data[l].emitterID = emIds[l];
      }

      // Build the single wide-node. For SingleLeaf=true we ignore the wide
      // node (firstLeafIdx=0 makes the sampler return leaf 0 immediately).
      LightcutTestNodeAccessor nodeAcc;
      nodeAcc.data.childLeafMask = 0xFu;
      NBL_UNROLL
      for (uint32_t s = 0u; s < 4u; ++s)
      {
         nodeAcc.data.children[s].bboxMin = mins[s];
         nodeAcc.data.children[s].bboxMax = maxs[s];
         nodeAcc.data.children[s].power   = powers[s];
      }

      const uint32_t firstLeafIdx = SingleLeaf ? LightcutTestFirstLeafIdxSingle : LightcutTestFirstLeafIdxMulti;

      const float32_t3 shadingPoint  = float32_t3(0.0f, 0.0f, 0.0f);
      const float32_t3 shadingNormal = float32_t3(0.0f, 1.0f, 0.0f);

      using SubAcc    = sampling::NoSubtreeAliasAccessor<float32_t, uint32_t>;
      using Sampler   = sampling::StochasticLightcutTreeSampler<float32_t, uint32_t, LightcutTestNodeAccessor, LightcutTestLeafAccessor, SubAcc, Mode>;
      Sampler sampler = Sampler::create(nodeAcc, leafAcc, SubAcc::create(), firstLeafIdx, shadingPoint, shadingNormal);

      typename Sampler::cache_type cache;
      output.generatedLeafHeap  = sampler.generate(input.u, cache);
      output.generatedEmitterID = cache.leaf.emitterID;
      output.leafBboxMinX       = cache.leaf.bboxMin.x;
      output.leafBboxMinY       = cache.leaf.bboxMin.y;
      output.leafBboxMinZ       = cache.leaf.bboxMin.z;
      output.leafBboxMaxX       = cache.leaf.bboxMax.x;
      output.leafBboxMaxY       = cache.leaf.bboxMax.y;
      output.leafBboxMaxZ       = cache.leaf.bboxMax.z;
      output.forwardPdf         = sampler.forwardPdf(input.u, cache);
      output.backwardPdf        = sampler.backwardPdf(output.generatedLeafHeap);
      output.forwardWeight      = sampler.forwardWeight(input.u, cache);
      output.backwardWeight     = sampler.backwardWeight(output.generatedLeafHeap);
      output.jacobianProduct    = (output.forwardPdf > 0.0f) ? ((1.0f / output.forwardPdf) * output.backwardPdf) : 0.0f;
   }
};

// multi/single are mode-agnostic CPU-vs-GPU consistency checks; pin them to the library default.
using LightcutTreeMultiLeafExecutor  = LightcutTreeTestExecutorImpl<false, NBL_LIGHTCUT_TREE_WEIGHT_MODE>;
using LightcutTreeSingleLeafExecutor = LightcutTreeTestExecutorImpl<true, NBL_LIGHTCUT_TREE_WEIGHT_MODE>;

// --- AABB-driven scenarios ---------------------------------------------------
// All four executors below share the multi-leaf wide-node layout (firstLeafIdx
// = 1) and differ only in the leaves' geometry / shading frame so each one
// exercises one branch of the weight function the default executor doesn't
// reach.

// All four leaves are placed entirely BELOW the shading tangent plane
// (normal +Y, shading point at origin). Every child should be killed by the
// `maxDotN <= 0` early-out in lightcutTreeChildWeight, the descent's wSum
// hits 0, and generate() returns the ~0u sentinel with pdf=0. This is the
// pdf=0 / sentinel-leaf path that the default scenario never exercises.
template<uint32_t Mode>
struct LightcutTreeBelowPlaneExecutor
{
   void operator()(NBL_CONST_REF_ARG(LightcutTreeInputValues) input, NBL_REF_ARG(LightcutTreeTestResults) output)
   {
      const float32_t3 mins[4]   = { float32_t3(1.0f, -2.5f, 1.0f), float32_t3(-2.0f, -2.0f, 1.0f), float32_t3(1.0f, -3.0f, -2.0f), float32_t3(-2.0f, -1.5f, -2.0f) };
      const float32_t3 maxs[4]   = { float32_t3(2.0f, -2.0f, 2.0f), float32_t3(-1.0f, -1.5f, 2.0f), float32_t3(2.0f, -2.5f, -1.0f), float32_t3(-1.0f, -1.0f, -1.0f) };
      const float32_t  powers[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
      const uint32_t   emIds[4]  = { 200u, 201u, 202u, 203u };

      LightcutTestLeafAccessor leafAcc;
      NBL_UNROLL
      for (uint32_t l = 0u; l < LightcutTestNumLeaves; ++l)
      {
         leafAcc.data[l].bboxMin   = mins[l];
         leafAcc.data[l].bboxMax   = maxs[l];
         leafAcc.data[l].emitterID = emIds[l];
      }

      LightcutTestNodeAccessor nodeAcc;
      nodeAcc.data.childLeafMask = 0xFu;
      NBL_UNROLL
      for (uint32_t s = 0u; s < 4u; ++s)
      {
         nodeAcc.data.children[s].bboxMin = mins[s];
         nodeAcc.data.children[s].bboxMax = maxs[s];
         nodeAcc.data.children[s].power   = powers[s];
      }

      const float32_t3 shadingPoint  = float32_t3(0.0f, 0.0f, 0.0f);
      const float32_t3 shadingNormal = float32_t3(0.0f, 1.0f, 0.0f);

      using SubAcc    = sampling::NoSubtreeAliasAccessor<float32_t, uint32_t>;
      using Sampler   = sampling::StochasticLightcutTreeSampler<float32_t, uint32_t, LightcutTestNodeAccessor, LightcutTestLeafAccessor, SubAcc, Mode>;
      Sampler sampler = Sampler::create(nodeAcc, leafAcc, SubAcc::create(), LightcutTestFirstLeafIdxMulti, shadingPoint, shadingNormal);

      typename Sampler::cache_type cache;
      output.generatedLeafHeap  = sampler.generate(input.u, cache);
      output.generatedEmitterID = cache.leaf.emitterID;
      // Leaf bbox isn't meaningful when generate() failed; surface zeros so the
      // tester's field comparisons stay deterministic across CPU and GPU.
      output.leafBboxMinX = output.leafBboxMinY = output.leafBboxMinZ = 0.0f;
      output.leafBboxMaxX = output.leafBboxMaxY = output.leafBboxMaxZ = 0.0f;
      output.forwardPdf                                               = sampler.forwardPdf(input.u, cache);
      output.backwardPdf                                              = sampler.backwardPdf(output.generatedLeafHeap);
      output.forwardWeight                                            = sampler.forwardWeight(input.u, cache);
      output.backwardWeight                                           = sampler.backwardWeight(output.generatedLeafHeap);
      output.jacobianProduct                                          = (output.forwardPdf > 0.0f) ? ((1.0f / output.forwardPdf) * output.backwardPdf) : 0.0f;
   }
};

// Two live leaves (equal power, same orientation) at distances d and 2d from
// the shading point; remaining two are zero-power padding. Inverse-square
// weighting should make the close leaf 4x more likely than the far one. We
// can't verify the ratio per-iteration (one pick per `u`) but the per-`u` pdfs
// must match the analytic CDF: pdf(close)=4/5, pdf(far)=1/5, and the boundary
// in `u` falls at 0.8. The tester adds those analytic checks on top of the
// generic consistency suite.
template<uint32_t Mode>
struct LightcutTreeDistanceFalloffExecutor
{
   void operator()(NBL_CONST_REF_ARG(LightcutTreeInputValues) input, NBL_REF_ARG(LightcutTreeTestResults) output)
   {
      // Slot 0: close, point-like leaf at (0, 1, 0) -> dist = 1
      // Slot 1: far,   point-like leaf at (0, 2, 0) -> dist = 2
      // Slots 2/3: zero-power padding well away.
      const float32_t  eps       = 1e-3f;
      const float32_t3 mins[4]   = { float32_t3(-eps, 1.0f - eps, -eps), float32_t3(-eps, 2.0f - eps, -eps), float32_t3(100.f, 100.f, 100.f), float32_t3(100.f, 100.f, 100.f) };
      const float32_t3 maxs[4]   = { float32_t3(eps, 1.0f + eps, eps), float32_t3(eps, 2.0f + eps, eps), float32_t3(100.f, 100.f, 100.f), float32_t3(100.f, 100.f, 100.f) };
      const float32_t  powers[4] = { 1.0f, 1.0f, 0.0f, 0.0f };
      const uint32_t   emIds[4]  = { 300u, 301u, 302u, 303u };

      LightcutTestLeafAccessor leafAcc;
      NBL_UNROLL
      for (uint32_t l = 0u; l < LightcutTestNumLeaves; ++l)
      {
         leafAcc.data[l].bboxMin   = mins[l];
         leafAcc.data[l].bboxMax   = maxs[l];
         leafAcc.data[l].emitterID = emIds[l];
      }

      LightcutTestNodeAccessor nodeAcc;
      nodeAcc.data.childLeafMask = 0xFu;
      NBL_UNROLL
      for (uint32_t s = 0u; s < 4u; ++s)
      {
         nodeAcc.data.children[s].bboxMin = mins[s];
         nodeAcc.data.children[s].bboxMax = maxs[s];
         nodeAcc.data.children[s].power   = powers[s];
      }

      const float32_t3 shadingPoint  = float32_t3(0.0f, 0.0f, 0.0f);
      const float32_t3 shadingNormal = float32_t3(0.0f, 1.0f, 0.0f);

      using SubAcc    = sampling::NoSubtreeAliasAccessor<float32_t, uint32_t>;
      using Sampler   = sampling::StochasticLightcutTreeSampler<float32_t, uint32_t, LightcutTestNodeAccessor, LightcutTestLeafAccessor, SubAcc, Mode>;
      Sampler sampler = Sampler::create(nodeAcc, leafAcc, SubAcc::create(), LightcutTestFirstLeafIdxMulti, shadingPoint, shadingNormal);

      typename Sampler::cache_type cache;
      output.generatedLeafHeap  = sampler.generate(input.u, cache);
      output.generatedEmitterID = cache.leaf.emitterID;
      output.leafBboxMinX       = cache.leaf.bboxMin.x;
      output.leafBboxMinY       = cache.leaf.bboxMin.y;
      output.leafBboxMinZ       = cache.leaf.bboxMin.z;
      output.leafBboxMaxX       = cache.leaf.bboxMax.x;
      output.leafBboxMaxY       = cache.leaf.bboxMax.y;
      output.leafBboxMaxZ       = cache.leaf.bboxMax.z;
      output.forwardPdf         = sampler.forwardPdf(input.u, cache);
      output.backwardPdf        = sampler.backwardPdf(output.generatedLeafHeap);
      output.forwardWeight      = sampler.forwardWeight(input.u, cache);
      output.backwardWeight     = sampler.backwardWeight(output.generatedLeafHeap);
      output.jacobianProduct    = (output.forwardPdf > 0.0f) ? ((1.0f / output.forwardPdf) * output.backwardPdf) : 0.0f;
   }
};

// Pathology test mirroring the user's real scene: one leaf with a TIGHT bbox
// and one with a huge inflated bbox spanning the shading point. Equal power.
// With the nearest-point distance + halfDiagSq floor (mode 0), the huge box's
// distSq floors to its own (large) half-diagonal^2 instead of collapsing to a
// tiny number, so it no longer steals weight from the tight near emitter -- the
// inverse of the old centroid-distance behaviour. The tester only checks
// consistency (fwd==bwd, jacobian==1), not which leaf wins, so it passes either
// way; this scenario guards that the floor keeps the inflated box well-conditioned.
template<uint32_t Mode>
struct LightcutTreeInflatedBboxExecutor
{
   void operator()(NBL_CONST_REF_ARG(LightcutTreeInputValues) input, NBL_REF_ARG(LightcutTreeTestResults) output)
   {
      // Slot 0: tight emitter at (0, 5, 0), half-extent 0.01
      // Slot 1: inflated bbox spanning [-50,50]^3 (origin inside!)
      // Slots 2/3: zero-power padding
      const float32_t3 mins[4]   = { float32_t3(-0.01f, 4.99f, -0.01f), float32_t3(-50.0f, -50.0f, -50.0f), float32_t3(100.f, 100.f, 100.f), float32_t3(100.f, 100.f, 100.f) };
      const float32_t3 maxs[4]   = { float32_t3(0.01f, 5.01f, 0.01f), float32_t3(50.0f, 50.0f, 50.0f), float32_t3(100.f, 100.f, 100.f), float32_t3(100.f, 100.f, 100.f) };
      const float32_t  powers[4] = { 1.0f, 1.0f, 0.0f, 0.0f };
      const uint32_t   emIds[4]  = { 400u, 401u, 402u, 403u };

      LightcutTestLeafAccessor leafAcc;
      NBL_UNROLL
      for (uint32_t l = 0u; l < LightcutTestNumLeaves; ++l)
      {
         leafAcc.data[l].bboxMin   = mins[l];
         leafAcc.data[l].bboxMax   = maxs[l];
         leafAcc.data[l].emitterID = emIds[l];
      }

      LightcutTestNodeAccessor nodeAcc;
      nodeAcc.data.childLeafMask = 0xFu;
      NBL_UNROLL
      for (uint32_t s = 0u; s < 4u; ++s)
      {
         nodeAcc.data.children[s].bboxMin = mins[s];
         nodeAcc.data.children[s].bboxMax = maxs[s];
         nodeAcc.data.children[s].power   = powers[s];
      }

      const float32_t3 shadingPoint  = float32_t3(0.0f, 0.0f, 0.0f);
      const float32_t3 shadingNormal = float32_t3(0.0f, 1.0f, 0.0f);

      using SubAcc    = sampling::NoSubtreeAliasAccessor<float32_t, uint32_t>;
      using Sampler   = sampling::StochasticLightcutTreeSampler<float32_t, uint32_t, LightcutTestNodeAccessor, LightcutTestLeafAccessor, SubAcc, Mode>;
      Sampler sampler = Sampler::create(nodeAcc, leafAcc, SubAcc::create(), LightcutTestFirstLeafIdxMulti, shadingPoint, shadingNormal);

      typename Sampler::cache_type cache;
      output.generatedLeafHeap  = sampler.generate(input.u, cache);
      output.generatedEmitterID = cache.leaf.emitterID;
      output.leafBboxMinX       = cache.leaf.bboxMin.x;
      output.leafBboxMinY       = cache.leaf.bboxMin.y;
      output.leafBboxMinZ       = cache.leaf.bboxMin.z;
      output.leafBboxMaxX       = cache.leaf.bboxMax.x;
      output.leafBboxMaxY       = cache.leaf.bboxMax.y;
      output.leafBboxMaxZ       = cache.leaf.bboxMax.z;
      output.forwardPdf         = sampler.forwardPdf(input.u, cache);
      output.backwardPdf        = sampler.backwardPdf(output.generatedLeafHeap);
      output.forwardWeight      = sampler.forwardWeight(input.u, cache);
      output.backwardWeight     = sampler.backwardWeight(output.generatedLeafHeap);
      output.jacobianProduct    = (output.forwardPdf > 0.0f) ? ((1.0f / output.forwardPdf) * output.backwardPdf) : 0.0f;
   }
};

// 2-level synthetic tree (16 leaves, depth 2). Forces the descent loop to take
// TWO `nodeAcc.get()` taps + multiply two conditional probabilities, exercising
// the per-level pdf product + heap-walk math that single-level scenarios skip.
// Leaves cluster into 4 groups of 4 at distinct +Y octant centroids; per-group
// extents are small so the parent wide-nodes' bboxes (used for descent at root)
// are well-conditioned. Power is uniform across leaves so the test is sensitive
// to the distance/orientation terms rather than power dominating.
template<uint32_t Mode>
struct LightcutTreeDepth2Executor
{
   void operator()(NBL_CONST_REF_ARG(LightcutTreeInputValues) input, NBL_REF_ARG(LightcutTreeTestResults) output)
   {
      // Group centroids (one per root child), all in front of normal +Y.
      const float32_t3 groupCentroids[4] = { float32_t3(2.0f, 1.5f, 2.0f), float32_t3(-2.0f, 1.5f, 2.0f), float32_t3(2.0f, 2.5f, -2.0f), float32_t3(-2.0f, 3.5f, -2.0f) };
      // Per-leaf offset inside its group's local cluster (4 corners of a tiny cube).
      const float32_t3 leafOffsets[4] = { float32_t3(0.10f, 0.0f, 0.10f), float32_t3(-0.10f, 0.0f, 0.10f), float32_t3(0.10f, 0.0f, -0.10f), float32_t3(-0.10f, 0.0f, -0.10f) };
      const float32_t  kHalfExt       = 0.02f;

      // Fill leaves: each group of 4 leaves around its centroid.
      LightcutTestLeafAccessorDepth2 leafAcc;
      NBL_UNROLL
      for (uint32_t g = 0u; g < 4u; ++g)
      {
         NBL_UNROLL
         for (uint32_t s = 0u; s < 4u; ++s)
         {
            const uint32_t   l        = g * 4u + s;
            const float32_t3 c        = groupCentroids[g] + leafOffsets[s];
            leafAcc.data[l].bboxMin   = c - float32_t3(kHalfExt, kHalfExt, kHalfExt);
            leafAcc.data[l].bboxMax   = c + float32_t3(kHalfExt, kHalfExt, kHalfExt);
            leafAcc.data[l].emitterID = 500u + l;
         }
      }

      // Build 5 wide-nodes:
      //  [0] root: 4 children (groups 0..3), all internal. childLeafMask = 0.
      //  [1..4]: leaf parents -- 4 children each, all leaves. childLeafMask = 0xF.
      LightcutTestNodeArrayAccessor nodeAcc;

      // Root: each child's bbox is the union of its group's 4 leaf bboxes,
      // power = 4 (uniform leaves of power 1.0).
      nodeAcc.data[0].childLeafMask = 0u;
      NBL_UNROLL
      for (uint32_t g = 0u; g < 4u; ++g)
      {
         float32_t3 mn = leafAcc.data[g * 4u].bboxMin;
         float32_t3 mx = leafAcc.data[g * 4u].bboxMax;
         NBL_UNROLL
         for (uint32_t s = 1u; s < 4u; ++s)
         {
            mn = min(mn, leafAcc.data[g * 4u + s].bboxMin);
            mx = max(mx, leafAcc.data[g * 4u + s].bboxMax);
         }
         nodeAcc.data[0].children[g].bboxMin = mn;
         nodeAcc.data[0].children[g].bboxMax = mx;
         nodeAcc.data[0].children[g].power   = 4.0f;
      }

      // Leaf parents (wide-nodes 1..4): copy leaf bboxes/powers into the
      // child records, mark all 4 slots as leaves.
      NBL_UNROLL
      for (uint32_t W = 1u; W <= 4u; ++W)
      {
         nodeAcc.data[W].childLeafMask = 0xFu;
         const uint32_t groupBase      = (W - 1u) * 4u;
         NBL_UNROLL
         for (uint32_t s = 0u; s < 4u; ++s)
         {
            nodeAcc.data[W].children[s].bboxMin = leafAcc.data[groupBase + s].bboxMin;
            nodeAcc.data[W].children[s].bboxMax = leafAcc.data[groupBase + s].bboxMax;
            nodeAcc.data[W].children[s].power   = 1.0f;
         }
      }

      const float32_t3 shadingPoint  = float32_t3(0.0f, 0.0f, 0.0f);
      const float32_t3 shadingNormal = float32_t3(0.0f, 1.0f, 0.0f);

      using SubAcc    = sampling::NoSubtreeAliasAccessor<float32_t, uint32_t>;
      using Sampler   = sampling::StochasticLightcutTreeSampler<float32_t, uint32_t, LightcutTestNodeArrayAccessor, LightcutTestLeafAccessorDepth2, SubAcc, Mode>;
      Sampler sampler = Sampler::create(nodeAcc, leafAcc, SubAcc::create(), LightcutTestFirstLeafIdxDepth2, shadingPoint, shadingNormal);

      typename Sampler::cache_type cache;
      output.generatedLeafHeap  = sampler.generate(input.u, cache);
      output.generatedEmitterID = cache.leaf.emitterID;
      output.leafBboxMinX       = cache.leaf.bboxMin.x;
      output.leafBboxMinY       = cache.leaf.bboxMin.y;
      output.leafBboxMinZ       = cache.leaf.bboxMin.z;
      output.leafBboxMaxX       = cache.leaf.bboxMax.x;
      output.leafBboxMaxY       = cache.leaf.bboxMax.y;
      output.leafBboxMaxZ       = cache.leaf.bboxMax.z;
      output.forwardPdf         = sampler.forwardPdf(input.u, cache);
      output.backwardPdf        = sampler.backwardPdf(output.generatedLeafHeap);
      output.forwardWeight      = sampler.forwardWeight(input.u, cache);
      output.backwardWeight     = sampler.backwardWeight(output.generatedLeafHeap);
      output.jacobianProduct    = (output.forwardPdf > 0.0f) ? ((1.0f / output.forwardPdf) * output.backwardPdf) : 0.0f;
   }
};

#endif
