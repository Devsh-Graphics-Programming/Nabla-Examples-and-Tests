#ifndef _NBL_THIS_EXAMPLE_LIGHT_TREE_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_LIGHT_TREE_HLSL_INCLUDED_


#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/sampling/alias_table.hlsl"
#include "renderer/shaders/bda_accessors.hlsl"


#include "nbl/builtin/hlsl/sampling/stochastic_lightcut_tree.hlsl"
#define NBL_LIGHTTREE_ALIAS_LOG2N 16u

namespace nbl
{
namespace this_example
{

NBL_CONSTEXPR_STATIC_INLINE uint32_t NonEmitterCustomIndex = 0xFFFFFFu;

struct SEmitterGPU
{
   hlsl::float32_t3 radiance;
   uint32_t         leafHeap;
   hlsl::float32_t3 bboxMin;
   hlsl::float32_t3 bboxMax;
   hlsl::float32_t2 _pad;
};
NBL_CONSTEXPR_STATIC_INLINE uint32_t EmitterRecordSize = sizeof(SEmitterGPU);

#ifdef __HLSL_VERSION

// BDA accessor satisfying StochasticLightcutTreeSampler's NodeAccessor concept.
// Loads + decodes one 32 B wide-node into the library's LightcutTreeWideNode<float>.
struct BDALightTreeNodeAccessor
{
   uint64_t base;

   static BDALightTreeNodeAccessor create(uint64_t _base)
   {
      BDALightTreeNodeAccessor r;
      r.base = _base;
      return r;
   }

   template<typename V, typename I> void get(I wideIdx, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
   {
      const uint64_t addr = base + uint64_t(wideIdx) * 32ull;

      // Two coalesced uint4 taps (32 B stride keeps every node 16-byte aligned), then the canonical
      // library unpack, one decode contract shared with the CPU builder + round-trip test.
      //   a (uint4 @ 0):  originX | originY | originZ | powExpMask
      //   b (uint4 @ 16): childPacked0..3
      const uint32_t4 a = vk::RawBufferLoad<uint32_t4>(addr + 0ull, 16u);
      const uint32_t4 b = vk::RawBufferLoad<uint32_t4>(addr + 16ull, 16u);

      nbl::hlsl::sampling::LightcutTreePackedWideNode packed;
      packed.origin      = asfloat(a.xyz);
      packed.powExpMask  = a.w;
      packed.childPacked = b;
      val                = nbl::hlsl::sampling::lightcutTreeUnpackWideNode<hlsl::float32_t>(packed);
   }
};

// BDA accessor satisfying StochasticLightcutTreeSampler's LeafAccessor concept.
// Loads one 32 B leaf record into the library's LightcutTreeLeaf<float>.
struct BDALightTreeLeafAccessor
{
   uint64_t base;

   static BDALightTreeLeafAccessor create(uint64_t _base)
   {
      BDALightTreeLeafAccessor r;
      r.base = _base;
      return r;
   }

   template<typename V, typename I> void get(I leafArrayIdx, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
   {
      const uint64_t addr = base + uint64_t(leafArrayIdx) * 32ull;
      // Two coalesced uint4 taps + the library unpack (shared decode contract).
      //   lo (uint4 @ 0):  bboxMin.xyz | bboxMax.x
      //   hi (uint4 @ 16): bboxMax.yz  | emitterID | _pad
      const uint32_t4 lo = vk::RawBufferLoad<uint32_t4>(addr + 0ull, 16u);
      const uint32_t4 hi = vk::RawBufferLoad<uint32_t4>(addr + 16ull, 16u);
      nbl::hlsl::sampling::LightcutTreePackedLeaf packed;
      packed.bboxMin   = asfloat(lo.xyz);
      packed.bboxMax   = hlsl::float32_t3(asfloat(lo.w), asfloat(hi.x), asfloat(hi.y));
      packed.emitterID = hi.z;
      val              = nbl::hlsl::sampling::lightcutTreeUnpackLeaf<hlsl::float32_t>(packed);
   }
};

// sample(W, u): O(3 RawBufferLoad + 1 alias pick), offsets[W], offsets[W+1], leafBases[W], then
// one bucket fetch via PackedAliasTableA on the sub-range.
struct BDASubtreeAliasAccessor
{
   uint64_t base;
   uint32_t numInternalNodes;
   uint32_t totalEntries;

   static BDASubtreeAliasAccessor create(uint64_t _base, uint32_t _numInternalNodes, uint32_t _totalEntries)
   {
      BDASubtreeAliasAccessor r;
      r.base             = _base;
      r.numInternalNodes = _numInternalNodes;
      r.totalEntries     = _totalEntries;
      return r;
   }

   using SubAlias = nbl::hlsl::sampling::PackedAliasTableA<hlsl::float32_t, hlsl::float32_t, uint32_t, BDAReadAccessor<uint32_t>, BDAReadAccessor<hlsl::float32_t>, NBL_LIGHTTREE_ALIAS_LOG2N>;

   // Resolved sub-range of W's per-subtree alias table within the concatenated buffer.
   // tableSize == 0 => W has no table (leafBase/alias addresses are then meaningless).
   struct SubRange
   {
      uint32_t tableSize;
      uint32_t leafBase; // leaf-array index of W's leftmost leaf
      uint64_t entriesAddr; // BDA of W's entries sub-array
      uint64_t pdfsAddr; // BDA of W's pdfs sub-array
   };

   // Shared address math for sample()/backwardPdf(): locate W's entries/pdfs/leafBase. The four
   // sections sit back-to-back (see the struct header comment). leafBase is only loaded when W has a table.
   SubRange __subRange(uint32_t W) NBL_CONST_MEMBER_FUNC
   {
      const uint64_t offsetsBase   = base;
      const uint64_t leafBasesBase = offsetsBase + uint64_t(numInternalNodes + 1u) * 4ull;
      const uint64_t entriesBase   = leafBasesBase + uint64_t(numInternalNodes) * 4ull;
      const uint64_t pdfsBase      = entriesBase + uint64_t(totalEntries) * 4ull;

      // offsets[W] and offsets[W+1] are adjacent (prefix-sum pair) -> one uint2 tap. The W*4 address
      // is only 4-byte aligned (odd W), so the load claims alignment 4.
      const uint32_t2 offs   = vk::RawBufferLoad<uint32_t2>(offsetsBase + uint64_t(W) * 4ull, 4u);
      const uint32_t  offW   = offs.x;
      const uint32_t  offWp1 = offs.y;

      SubRange r;
      r.tableSize   = offWp1 - offW;
      r.leafBase    = (r.tableSize == 0u) ? 0u : vk::RawBufferLoad<uint32_t>(leafBasesBase + uint64_t(W) * 4ull);
      r.entriesAddr = entriesBase + uint64_t(offW) * 4ull;
      r.pdfsAddr    = pdfsBase + uint64_t(offW) * 4ull;
      return r;
   }

   SubAlias __makeAlias(NBL_CONST_REF_ARG(SubRange) r) NBL_CONST_MEMBER_FUNC { return SubAlias::create(BDAReadAccessor<uint32_t>::create(r.entriesAddr), BDAReadAccessor<hlsl::float32_t>::create(r.pdfsAddr), r.tableSize); }

   // Outputs the picked leaf's LEAF-ARRAY index (caller adds firstLeafIdx for heap index).
   void sample(uint32_t W, hlsl::float32_t u, NBL_REF_ARG(uint32_t) outLeafArrayIdx, NBL_REF_ARG(hlsl::float32_t) outPdf) NBL_CONST_MEMBER_FUNC
   {
      const SubRange r = __subRange(W);
      if (r.tableSize == 0u)
      {
         outPdf          = hlsl::float32_t(0);
         outLeafArrayIdx = ~0u;
         return;
      }
      SubAlias             alias = __makeAlias(r);
      SubAlias::cache_type cache;
      const uint32_t       pickedEntry = alias.generate(u, cache);
      outPdf                           = cache.pdf;
      outLeafArrayIdx                  = r.leafBase + pickedEntry;
   }

   hlsl::float32_t backwardPdf(uint32_t W, uint32_t leafArrayIdx) NBL_CONST_MEMBER_FUNC
   {
      const SubRange r = __subRange(W);
      if (r.tableSize == 0u)
         return hlsl::float32_t(0);
      const uint32_t bin = leafArrayIdx - r.leafBase;
      if (bin >= r.tableSize)
         return hlsl::float32_t(0);
      SubAlias alias = __makeAlias(r);
      return alias.backwardPdf(bin);
   }
};

// Convenience aliases so callers don't repeat the full template path.
using LightTreeSampler = nbl::hlsl::sampling::StochasticLightcutTreeSampler<hlsl::float32_t, uint32_t, BDALightTreeNodeAccessor, BDALightTreeLeafAccessor, BDASubtreeAliasAccessor, NBL_LIGHTCUT_TREE_WEIGHT_MODE>;
using LightTreeLeaf    = nbl::hlsl::sampling::LightcutTreeLeaf<hlsl::float32_t>;

#endif

} // namespace this_example
} // namespace nbl
#endif
