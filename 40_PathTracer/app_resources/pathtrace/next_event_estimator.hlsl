#ifndef _PATHTRACER_40_NEXT_EVENT_ESTIMATOR_INCLUDED_
#define _PATHTRACER_40_NEXT_EVENT_ESTIMATOR_INCLUDED_

#include "nbl/builtin/hlsl/shapes/obb_silhouette.hlsl"
#include "nbl/builtin/hlsl/sampling/spherical_pyramid.hlsl"
#include "nbl/builtin/hlsl/sampling/alias_table.hlsl"
#include "nbl/builtin/hlsl/random/tea.hlsl"
#ifdef __HLSL_VERSION
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl" // spirv::executeCallable for the NBL_NEE_CALLABLE path
#endif

#include "common.hlsl"
#include "renderer/shaders/bda_accessors.hlsl"

// TEA round count for __iidUnit (the per-candidate IID sub-stream hash). 4 is the usual "good enough"
// decorrelation for RIS candidates; raise toward 8-16 for paranoia at the cost of a few ALU per draw.
#ifndef NBL_NEE_TEA_ROUNDS
#define NBL_NEE_TEA_ROUNDS 4u
#endif

#define NEE_RIS_CANDIDATES 4

// Light-selection RIS candidates: propose this many leaves ~ power, resample one ~ geometry.
// Geometry lives in the resample target (numerator), never in the selection pdf (denominator),
// so a wrong geometry estimate only mis-ranks a candidate instead of dividing into a firefly.
#define NEE_LIGHT_CANDIDATES 4

// A/B knob: include the 1/dist^2 term in the RIS resample target (__geomTarget).
//   1 = orientation/dist^2 (default). Pair with descent mode 3 (no distance) -> distance applied
//       ONCE, in the RIS numerator. This is config A.
//   0 = orientation only. Pair with descent mode 0 (power*orient/dist^2) -> distance applied ONCE,
//       in the descent.
#ifndef NEE_GEOMTARGET_DISTANCE
#define NEE_GEOMTARGET_DISTANCE 1
#endif

// Emitter-selection proposal: 1 = power alias table (O(1) lookup), 0 = stochastic light-cut tree descent
// (power + per-shading-point orientation/distance). Compile-time like NBL_MIS_MODE, not a push constant,
// so the unused path's sampler code is dropped from the shader entirely (no runtime branch / dead regs).
#ifndef NBL_NEE_USE_ALIAS
#define NBL_NEE_USE_ALIAS 1
#endif

#define NBL_MIS_MODE_NEE_ONLY 0
#define NBL_MIS_MODE_BXDF_ONLY 1
#define NBL_MIS_MODE_BOTH 2

#ifndef NBL_MIS_MODE
#define NBL_MIS_MODE NBL_MIS_MODE_BOTH
#endif

// Diagnostic knob: when 1, the center pixel captures the K NEE-light-selection candidates of
// sample 0 and beauty overlays them as a row of K cells right of center. Fill = per-emitter
// hashed hue, border = green if a deterministic-direction shadow ray reaches the emitter, red
// if occluded. Marker = single white pixel at the screen center. All non-probe pixels are
// forced to 0.05 gray. Pure diagnostic; ignores accumulation, no MIS, NEE-only assumed.
#ifndef NBL_NEE_PROPOSAL_PROBE
#define NBL_NEE_PROPOSAL_PROBE 0
#endif

// Experiment knob: when 1, forwardNEE and the emission MIS-deweight run through ray-tracing callables
// (neeCallable / emissionCallable in beauty.hlsl) instead of inlining, moving the silhouette + pyramid
// + spherical-rectangle footprint out of raygen's register / i-cache working set. The emission callable
// also runs the backward selection-pdf climb on a cache miss (raygen passes a negative sentinel), so
// that footprint leaves raygen too. Only gates the CALL SITE; the entry points + their SBT slots are
// unconditional (no shader<->C++ define to keep in sync, a mismatch would be a silent device-lost). The
// payload spills to the RT stack, so judge by occupancy + the callable's own reg count, not raygen's.
#ifndef NBL_NEE_CALLABLE
#define NBL_NEE_CALLABLE 0
#endif

namespace nbl
{
namespace this_example
{

struct SNeeCallableData
{
   // in
   float32_t3 hitPos;
   float32_t3 shadingNormal;
   float32_t3 V;
   float32_t3 throughput;
   float32_t3 randNEE;
   float32_t3 randNEE2;
   // in/out: estimator same-emitter MIS cache
   uint32_t  prevDescentNeeEmitterID;
   float32_t prevDescentNeePdf;
   // out
   float32_t3 pickedDir;
   float32_t3 contribution;
   uint32_t   pickedEmitterID;
   uint32_t   valid;
};


struct SEmissionCallableData
{
   // in
   float32_t3 currentHitPos;
   float32_t3 prevShadingHitPos;
   float32_t3 prevShadingNormal;
   uint32_t   emitterIdx;
   float32_t  emitterSelectBackPdf; // >= 0: cached selection pdf; < 0: cache miss, callable runs the climb
   float32_t  otherTechniqueHeuristic;
   // out
   float32_t deweight;
};

#ifdef __HLSL_VERSION
// NBL_LIGHTTREE_ALIAS_LOG2N is the single source of truth (renderer/shaders/light_tree.hlsl); the
// runtime table size is gScene.init.aliasTableSize.
using AliasSampler = nbl::hlsl::sampling::PackedAliasTableA<float32_t, float32_t, uint32_t, BDAReadAccessor<uint32_t>, BDAReadAccessor<float32_t>, NBL_LIGHTTREE_ALIAS_LOG2N>;
#endif

// Emitter selection (alias table OR light-tree descent), solid-angle sampling of the
// picked emitter's bounding box via a clipped spherical pyramid, and the MIS book-keeping
// to combine NEE with BSDF sampling. Also folds in emission-on-hit and env-map shading,
// since both are the backward (BSDF-side) half of the same MIS pair.
struct NextEventEstimator
{
   using bxdf_config_t           = nbl::hlsl::material_compiler3::backends::default_upt::BxDFConfig;
   using isotropic_interaction_t = bxdf_config_t::isotropic_interaction_type;
   using light_sample_t          = bxdf_config_t::sample_type;
   using spectral_type           = bxdf_config_t::spectral_type;
   using ray_dir_info_t          = light_sample_t::ray_dir_info_type;
   using value_weight_type       = nbl::hlsl::sampling::value_and_weight<spectral_type, float>;
   using brdf_t                  = nbl::hlsl::bxdf::reflection::SOrenNayar<bxdf_config_t>;

#ifndef NBL_NEE_PROJECTED_SPHRECT
#define NBL_NEE_PROJECTED_SPHRECT 1
#endif
#if NBL_NEE_PROJECTED_SPHRECT
   using pyramid_t = nbl::hlsl::sampling::SphericalPyramid<false, nbl::hlsl::sampling::ProjectedSphericalRectangle<float32_t, false> >;
#else
   using pyramid_t = nbl::hlsl::sampling::SphericalPyramid<false, nbl::hlsl::sampling::SphericalRectangle<float32_t> >;
#endif

   NBL_CONSTEXPR_STATIC_INLINE float32_t MISWeightThreshold = nbl::hlsl::numeric_limits<float32_t>::min;

   // Picked direction + the contribution to add if the shadow ray reaches the picked emitter.
   // contribution already folds in throughput, BSDF, emission, MIS weight and 1/pdf, the
   // caller only multiplies by binary visibility.
   struct SForwardSample
   {
      float32_t3    pickedDir;
      uint32_t      pickedEmitterID;
      spectral_type contribution;
      bool          valid;
      // Selection pdf of the picked emitter (winner.pProposal), for the caller to seed the next bounce's
      // BSDF-side MIS cache. Returned (not written to the estimator) so forwardNEE is pure -> callable-safe.
      // > 0 iff a winner was drawn this call (the caller updates the cache only then).
      float32_t pProposal;
   };

   static NextEventEstimator create()
   {
      NextEventEstimator nee;
      nee.prevDescentNeeEmitterID = ~0u;
      nee.prevDescentNeePdf       = 0.f;
      nee.prevShadingHitPos       = float32_t3(0, 0, 0);
      nee.prevShadingNormal       = float32_t3(0, 1, 0);
      return nee;
   }

   static float32_t __luma(const spectral_type c) { return hlsl::dot(c, spectral_type(nbl::hlsl::material_compiler3::backends::default_upt::LumaConversionCoeffs)); }

   // Independent uniform in [0,1) from a base random's bits + a candidate index, via TEA
   // (stream=baseBits, sequence=idx, NBL_NEE_TEA_ROUNDS rounds). Gives the IID candidates RIS needs.
   static float32_t __iidUnit(const uint32_t baseBits, const uint32_t idx)
   {
      const uint32_t2 h = nbl::hlsl::random::Tea::__call(baseBits, idx, NBL_NEE_TEA_ROUNDS);
      return float32_t(h.x) * hlsl::exp2(-32.f); // uint32 -> [0,1)
   }

   // Express the emitter's world-space bounding box as an oriented box in the shading
   // tangent frame and clip it against the upper hemisphere.
   static shapes::ClippedSilhouette __buildSilhouette(
      NBL_REF_ARG(shapes::OBBView<float32_t>) obbView, const float32_t3 hitPos, const float32_t3 frameT, const float32_t3 frameB, const float32_t3 normal, const float32_t3 bboxMin, const float32_t3 bboxMax)
   {
      const float32_t3 worldExt    = bboxMax - bboxMin;
      const float32_t3 worldMinRel = bboxMin - hitPos;

      obbView.minCorner  = float32_t3(hlsl::dot(worldMinRel, frameT), hlsl::dot(worldMinRel, frameB), hlsl::dot(worldMinRel, normal));
      obbView.columns[0] = float32_t3(frameT.x, frameB.x, normal.x) * worldExt.x;
      obbView.columns[1] = float32_t3(frameT.y, frameB.y, normal.y) * worldExt.y;
      obbView.columns[2] = float32_t3(frameT.z, frameB.z, normal.z) * worldExt.z;

      return shapes::ClippedSilhouette::create(obbView);
   }

   // Emitter's leaf bbox, read from the co-located emitter record (48 B: radiance | leafHeap |
   // bboxMin | bboxMax | pad). One direct load on emitterID, so no emitter -> leaf reverse-map ->
   // leaf-record dependent 2-load chain (which was the path tracer's worst LGSB stall line).
   static LightTreeLeaf __getLeaf(const uint32_t emitterIdx)
   {
      const uint64_t addr = gScene.init.pEmitters + uint64_t(emitterIdx) * uint64_t(EmitterRecordSize);
      // Two 16-byte-aligned uint4 taps over the bbox half of the record (the 48 B stride keeps the
      // record 16-aligned). b1 = bboxMin.xyz | bboxMax.x; b2 = bboxMax.yz | pad | pad.
      const uint32_t4                  b1 = vk::RawBufferLoad<uint32_t4>(addr + 16ull, 16u);
      const uint32_t4                  b2 = vk::RawBufferLoad<uint32_t4>(addr + 32ull, 16u);
      LightTreeLeaf leaf;
      leaf.bboxMin   = float32_t3(asfloat(b1.x), asfloat(b1.y), asfloat(b1.z));
      leaf.bboxMax   = float32_t3(asfloat(b1.w), asfloat(b2.x), asfloat(b2.y));
      leaf.emitterID = emitterIdx;
      return leaf;
   }

   // Geometry-only resampling target for one candidate leaf: receiver cosine upper bound (the
   // cone-vs-bbox form, == 1 when the box subtends the normal) over squared NEAREST-POINT distance.
   // Power is deliberately excluded here, it lives in the proposal pdf and cancels in the RIS
   // weight; only the geometry refines selection, in the numerator where its errors can't explode.
   //
   // 1/d^2 uses NEAREST-POINT distance, not centroid: a large emitter with a near face close to x but
   // a far centroid is an excellent target and must rank high (centroid distance under-ranks it). The
   // cone/orientation still uses centroid direction + angular radius (correct for a cone); the distance
   // is floored at halfDiagSq, matching the descent's distSq floor.
   static float32_t __geomTarget(const float32_t3 bboxMin, const float32_t3 bboxMax, const float32_t3 x, const float32_t3 n)
   {
      const float32_t3 ext            = bboxMax - bboxMin;
      const float32_t  halfDiagSq     = 0.25f * hlsl::dot(ext, ext);
      const float32_t3 dToCentroid    = 0.5f * (bboxMin + bboxMax) - x;
      const float32_t  centroidDistSq = hlsl::dot(dToCentroid, dToCentroid);
      const float32_t  rcpDist        = hlsl::rsqrt(hlsl::max(centroidDistSq, halfDiagSq));
      const float32_t  cosPhi         = hlsl::dot(n, dToCentroid) * rcpDist;
      const float32_t  sinAlpha       = hlsl::min(hlsl::sqrt(halfDiagSq) * rcpDist, 1.f);
      const float32_t  cosAlpha       = hlsl::sqrt(hlsl::max(1.f - sinAlpha * sinAlpha, 0.f));
      const float32_t  sinPhi         = hlsl::sqrt(hlsl::max(1.f - cosPhi * cosPhi, 0.f));
      const float32_t  orientFactor   = (cosPhi >= cosAlpha) ? 1.f : hlsl::max(cosPhi * cosAlpha + sinPhi * sinAlpha, 0.f);
#if NEE_GEOMTARGET_DISTANCE
      const float32_t3 dNear     = hlsl::max(hlsl::max(bboxMin - x, x - bboxMax), hlsl::promote<float32_t3>(0.f));
      const float32_t  minDistSq = hlsl::dot(dNear, dNear);
      return orientFactor / hlsl::max(minDistSq, halfDiagSq);
#else
      // Orientation only: distance is expected to come from the descent (mode 0) instead.
      return orientFactor;
#endif
   }

   struct SLightCandidate
   {
      uint32_t   emitterID;
      float32_t3 bboxMin;
      float32_t3 bboxMax;
      float32_t  pProposal; // normalized leaf-selection pdf of the power proposal
      float32_t  geomTarget; // resample weight (geometry only; power cancels)
   };

   // Draw one leaf from the power-proportional proposal (alias table, or a power-only tree descent
   // with NBL_LIGHTCUT_TREE_WEIGHT_MODE==1), and tag it with its geometry resample target.
   SLightCandidate __drawPowerCandidate(const float32_t u, const float32_t3 hitPos, const float32_t3 shadingNormal)
   {
      SLightCandidate c;
#if NBL_NEE_USE_ALIAS
      {
         AliasSampler             alias = AliasSampler::create(BDAReadAccessor<uint32_t>::create(gScene.init.pAliasEntries), BDAReadAccessor<float32_t>::create(gScene.init.pAliasPdf), gScene.init.aliasTableSize);
         AliasSampler::cache_type aliasCache;
         c.emitterID                                 = alias.generate(u, aliasCache);
         c.pProposal                                 = alias.forwardPdf(u, aliasCache);
         const LightTreeLeaf leaf = __getLeaf(c.emitterID);
         c.bboxMin                                   = leaf.bboxMin;
         c.bboxMax                                   = leaf.bboxMax;
      }
#else
      {
         LightTreeSampler             tree = LightTreeSampler::create(BDALightTreeNodeAccessor::create(gScene.init.pLightTreeNodes),
            BDALightTreeLeafAccessor::create(gScene.init.pLightTreeLeaves),
            BDASubtreeAliasAccessor::create(gScene.init.pSubtreeAlias, gScene.init.lightTreeFirstLeafIndex, gScene.init.subtreeAliasTotalEntries),
            gScene.init.lightTreeFirstLeafIndex,
            hitPos,
            shadingNormal);
         LightTreeSampler::cache_type treeCache;
         tree.generate(u, treeCache);
         c.emitterID = treeCache.leaf.emitterID;
         c.pProposal = treeCache.pdf;
         c.bboxMin   = treeCache.leaf.bboxMin;
         c.bboxMax   = treeCache.leaf.bboxMax;
      }
#endif
      c.geomTarget = (c.pProposal > 0.f && c.emitterID < NonEmitterCustomIndex) ? __geomTarget(c.bboxMin, c.bboxMax, hitPos, shadingNormal) : 0.f;
      return c;
   }

#if NBL_NEE_PROPOSAL_PROBE
   // Probe diagnostic: build candidate k as forwardNEE would for the K-sized RIS pool, then
   // sample one deterministic direction (silhouette midpoint, u=0.5) toward its bbox. The
   // caller fires a shadow ray with the returned direction. Returns emitterID = ~0u when the
   // candidate is degenerate (zero proposal, invalid silhouette, etc.) so the caller can mark
   // the cell as "no light".
   struct SProbeCandidate
   {
      uint32_t   emitterID;
      float32_t3 pickedDir;
   };
   SProbeCandidate __probeCandidate(uint32_t k, const float32_t3 hitPos, const float32_t3 shadingNormal, const float32_t3 randNEE)
   {
      SProbeCandidate r;
      r.emitterID = NonEmitterCustomIndex;
      r.pickedDir = float32_t3(0, 1, 0);

      const float32_t       u = __iidUnit(asuint(randNEE.x), k);
      const SLightCandidate c = __drawPowerCandidate(u, hitPos, shadingNormal);
      if (!(c.pProposal > 0.f) || c.emitterID >= NonEmitterCustomIndex)
         return r;

      float32_t3 frameT, frameB;
      math::frisvad<float32_t3>(shadingNormal, frameT, frameB);

      shapes::OBBView<float32_t>      obbView;
      const shapes::ClippedSilhouette silhouette = __buildSilhouette(obbView, hitPos, frameT, frameB, shadingNormal, c.bboxMin, c.bboxMax);
      if (silhouette.count == 0u)
         return r;

      pyramid_t pyramid = pyramid_t::create(silhouette, obbView);
      // Deterministic midpoint direction: u = (0.5, 0.5) samples the centroid of the silhouette.
      pyramid_t::cache_type pyrCache;
      const float32_t3      tangentDir = pyramid.generate(float32_t2(0.5f, 0.5f), pyrCache);
      r.pickedDir                      = hlsl::normalize(tangentDir.x * frameT + tangentDir.y * frameB + tangentDir.z * shadingNormal);
      r.emitterID                      = c.emitterID;
      return r;
   }
#endif

   // Backward probability that NEE would have selected this emitter from prev's shading point.
   float32_t __emitterSelectBackPdf(const uint32_t emitterIdx)
   {
#if NBL_NEE_USE_ALIAS
      AliasSampler aliasBwd = AliasSampler::create(BDAReadAccessor<uint32_t>::create(gScene.init.pAliasEntries), BDAReadAccessor<float32_t>::create(gScene.init.pAliasPdf), gScene.init.aliasTableSize);
      return aliasBwd.backwardPdf(emitterIdx);
#else
      // leafHeap is co-located in the emitter record (offset 12), so no reverse-map load.
      const uint32_t                      leafIdxBwd = vk::RawBufferLoad<uint32_t>(gScene.init.pEmitters + uint64_t(emitterIdx) * uint64_t(EmitterRecordSize) + 12ull);
      LightTreeSampler treeBwd    = LightTreeSampler::create(BDALightTreeNodeAccessor::create(gScene.init.pLightTreeNodes),
         BDALightTreeLeafAccessor::create(gScene.init.pLightTreeLeaves),
         BDASubtreeAliasAccessor::create(gScene.init.pSubtreeAlias, gScene.init.lightTreeFirstLeafIndex, gScene.init.subtreeAliasTotalEntries),
         gScene.init.lightTreeFirstLeafIndex,
         prevShadingHitPos,
         prevShadingNormal);
      return treeBwd.backwardPdf(leafIdxBwd);
#endif
   }

   // MIS deweight multiplier for emission on a BSDF hit: builds the picked emitter's clipped silhouette
   // + spherical pyramid and returns 1/(1+weightRatio^2) for the arrival direction, or 1 when the
   // silhouette is degenerate / rectProto<=0 (leave emission untouched). Reads prevShading* members.
   // Split out so the inline path and the emission callable (NBL_NEE_CALLABLE) share one implementation.
   float32_t __emissionDeweight(const uint32_t emitterIdx, const float32_t3 currentHitPos, const float32_t emitterSelectBackPdf, const float32_t otherTechniqueHeuristic)
   {
      const LightTreeLeaf leaf = __getLeaf(emitterIdx);
      float32_t3                             prevT, prevB;
      math::frisvad<float32_t3>(prevShadingNormal, prevT, prevB);

      shapes::OBBView<float32_t>      obbView;
      const shapes::ClippedSilhouette silhouette = __buildSilhouette(obbView, prevShadingHitPos, prevT, prevB, prevShadingNormal, leaf.bboxMin, leaf.bboxMax);
      if (silhouette.count == 0u)
         return 1.f;

      // obbView/silhouette live in the shading tangent frame; the arrival direction must be expressed
      // there too. backwardWeight is the analytic projected density cos/projSolidAngle. Same 1/2pi
      // hemisphere roll-off as forward; rectProto<=0 -> no deweight, rectProto=inf -> deweight 0 (no NaN).
      const float32_t3 dirWorld = hlsl::normalize(currentHitPos - prevShadingHitPos);
      const float32_t3 dirLocal = float32_t3(hlsl::dot(dirWorld, prevT), hlsl::dot(dirWorld, prevB), hlsl::dot(dirWorld, prevShadingNormal));

      pyramid_t       sampler    = pyramid_t::create(silhouette, obbView);
      const float32_t rectWeight = sampler.backwardWeight(dirLocal);
      const float32_t rectProto  = hlsl::max(rectWeight - 0.5f / numbers::pi<float32_t>, 0.f);
      if (!(rectProto > 0.f))
         return 1.f;
      const float32_t neePdf      = emitterSelectBackPdf * rectProto;
      const float32_t weightRatio = neePdf * otherTechniqueHeuristic; // neePdf / bsdfPdf
      return 1.f / (1.f + weightRatio * weightRatio);
   }

   // Emission on a BSDF-sampled hit, deweighted against the NEE technique via the power heuristic.
   // otherTechniqueHeuristic is 1/bsdfWeight from the previous bounce.
   spectral_type shadeEmission(const uint32_t emitterIdx, const float32_t3 currentHitPos, const float32_t otherTechniqueHeuristic, const spectral_type throughput)
   {
      if (!(emitterIdx < NonEmitterCustomIndex && gScene.init.pEmitters != 0))
         return spectral_type(0, 0, 0);

      float32_t3 emission = vk::RawBufferLoad<float32_t3>(gScene.init.pEmitters + uint64_t(emitterIdx) * uint64_t(EmitterRecordSize));

#if NBL_MIS_MODE == NBL_MIS_MODE_BOTH
      if (otherTechniqueHeuristic > MISWeightThreshold && gScene.init.pEmitterToLeafIdx != 0)
      {
         // Same-emitter cache hit (set at NEE forward time) supplies the selection pdf for free; a miss
         // passes a negative sentinel so the backward climb runs where it is cheapest. In the callable
         // build that is inside emissionCallable, keeping the climb's register / i-cache footprint out
         // of raygen; inline it runs here.
         const float32_t cachedBackPdf = (emitterIdx == prevDescentNeeEmitterID) ? prevDescentNeePdf : -1.f;

         // compute NEE MIS backward weight on the contribution color
         float32_t deweight;
#if NBL_NEE_CALLABLE
         [[vk::ext_storage_class(spv::StorageClassCallableDataKHR)]] SEmissionCallableData ec;
         ec.currentHitPos           = currentHitPos;
         ec.prevShadingHitPos       = prevShadingHitPos;
         ec.prevShadingNormal       = prevShadingNormal;
         ec.emitterIdx              = emitterIdx;
         ec.emitterSelectBackPdf    = cachedBackPdf; // < 0 => callable runs the climb itself
         ec.otherTechniqueHeuristic = otherTechniqueHeuristic;
         nbl::hlsl::spirv::executeCallable(1u, ec);
         deweight = ec.deweight;
#else
         const float32_t emitterSelectBackPdf = (cachedBackPdf < 0.f) ? __emitterSelectBackPdf(emitterIdx) : cachedBackPdf;
         deweight                             = (emitterSelectBackPdf > 0.f) ? __emissionDeweight(emitterIdx, currentHitPos, emitterSelectBackPdf, otherTechniqueHeuristic) : 1.f;
#endif
         assert(!hlsl::isinf(deweight));
         // apply emissive weight
         emission *= deweight;
      }
#endif

      return emission * throughput;
   }

   // Pick an emitter, sample a direction toward it (with optional RIS), and assemble the
   // contribution to add when the shadow ray confirms visibility. Caches the selection pdf
   // for the next bounce's emission-side MIS.
   SForwardSample forwardNEE(
      const float32_t3 hitPos, const float32_t3 shadingNormal, NBL_CONST_REF_ARG(isotropic_interaction_t) interaction, NBL_CONST_REF_ARG(brdf_t) diffuse, const spectral_type throughput, const float32_t3 randNEE, const float32_t3 randNEE2)
   {
      SForwardSample res;
      res.pickedDir       = float32_t3(0, 0, 0);
      res.pickedEmitterID = NonEmitterCustomIndex;
      res.contribution    = spectral_type(0, 0, 0);
      res.valid           = false;

      float32_t3 frameT, frameB;
      math::frisvad<float32_t3>(shadingNormal, frameT, frameB);

      // Light selection by RIS: propose NEE_LIGHT_CANDIDATES leaves ~ power (cancels against
      // contribution like the alias table), resample one ~ geometry (the numerator target). Candidates
      // must be IID: RIS is unbiased only for iid proposals, and stratifying one base uniform darkened
      // the image at M>1. Resample uses randNEE2.z.
      const uint32_t        selSeed          = asuint(randNEE.x);
      static const uint16_t kLightCandidates = uint16_t(NEE_LIGHT_CANDIDATES);
      float32_t             candG[NEE_LIGHT_CANDIDATES];
      float32_t             sumG = 0.f;
      for (uint16_t m = 0u; m < kLightCandidates; ++m)
      {
         const float32_t       u    = __iidUnit(selSeed, uint32_t(m));
         const SLightCandidate cand = __drawPowerCandidate(u, hitPos, shadingNormal);
         candG[m]                   = cand.geomTarget;
         sumG += candG[m];
      }
      // No candidate is above the horizon / has positive geometry: nothing NEE can usefully reach.
      if (!(sumG > 0.f))
         return res;

      const float32_t selThreshold = randNEE2.z * sumG;
      float32_t       selAccum     = 0.f;
      uint16_t        winnerIdx    = 0u;
      bool            selFound     = false;
      for (uint16_t m = 0u; m < kLightCandidates; ++m)
      {
         selAccum += candG[m];
         if (!selFound && candG[m] > 0.f && selAccum >= selThreshold)
         {
            winnerIdx = m;
            selFound  = true;
         }
      }
      // Redraw only the winner (keeps just one candidate's bbox/id live instead of all M).
      // Must use the SAME iid hash as candidate winnerIdx so the redraw reproduces that leaf.
      const float32_t       uWinner         = __iidUnit(selSeed, uint32_t(winnerIdx));
      const SLightCandidate winner          = __drawPowerCandidate(uWinner, hitPos, shadingNormal);
      const uint32_t        pickedEmitterID = winner.emitterID;

      // Selection-layer RIS unbiased contribution weight: (1/M) sum(t_i/p_i) / t_winner, with
      // t_i = power_i * geom_i and p_i = power_i / totalPower so power and totalPower cancel out. The
      // redraw reproduces winnerIdx's leaf via the same iid hash, so winner.geomTarget == candG[winnerIdx].
      const float32_t selWeight = (sumG / float32_t(kLightCandidates)) / (candG[winnerIdx] * winner.pProposal);

      prevDescentNeeEmitterID = pickedEmitterID;
      prevDescentNeePdf       = winner.pProposal;

      shapes::OBBView<float32_t>      obbView;
      const shapes::ClippedSilhouette silhouette = __buildSilhouette(obbView, hitPos, frameT, frameB, shadingNormal, winner.bboxMin, winner.bboxMax);

      const spectral_type emission = vk::RawBufferLoad<float32_t3>(gScene.init.pEmitters + uint64_t(pickedEmitterID) * uint64_t(EmitterRecordSize));

      static const uint16_t kRISCandidates = uint16_t(NEE_RIS_CANDIDATES);

      // Bail out on degenerate silhouette (observer inside OBB or fully horizon-clipped).
      if (silhouette.count == 0u)
         return res;

      pyramid_t pyramid = pyramid_t::create(silhouette, obbView);

      const uint32_t dirSeed = asuint(randNEE2.x) ^ (asuint(randNEE2.y) * 0x9E3779B9u);
      float32_t3     candDirs[NEE_RIS_CANDIDATES];
      float32_t      candTarget[NEE_RIS_CANDIDATES];
      float32_t      candPdf[NEE_RIS_CANDIDATES];
      float32_t      sumW = 0.f;
      for (uint16_t k = 0u; k < kRISCandidates; ++k)
      {
         const uint32_t2       hk = nbl::hlsl::random::Tea::__call(dirSeed, uint32_t(k), NBL_NEE_TEA_ROUNDS);
         const float32_t2      u  = float32_t2(float32_t(hk.x), float32_t(hk.y)) * hlsl::exp2(-32.f);
         pyramid_t::cache_type pyrCache;
         const float32_t3      dirLocal = pyramid.generate(u, pyrCache);
         const float32_t3      dirWorld = frameT * dirLocal.x + frameB * dirLocal.y + shadingNormal * dirLocal.z;
         candDirs[k]                    = dirWorld;

         float32_t target = 0.f;
         // pyrCache.pdf <= 0 means a degenerate/clipped sample: a zero-weight proposal.
         if (pyrCache.pdf > 0.f)
         {
            ray_dir_info_t tmp;
            tmp.setDirection(dirWorld);
            const light_sample_t    L  = light_sample_t::create(tmp, shadingNormal);
            const value_weight_type ev = diffuse.evalAndWeight(L, interaction);
            target                     = hlsl::max(__luma(throughput * ev.value() * emission), 0.f);
         }
         candTarget[k] = target;
         candPdf[k]    = pyrCache.pdf;
         // Projected sampling gives each candidate its own solid-angle density, so the RIS resampling
         // weight is target/pdf.
         sumW += (pyrCache.pdf > 0.f) ? (target / pyrCache.pdf) : 0.f;
      }

      // No candidate carries any unshadowed contribution (e.g. fully grazing): nothing to sample.
      if (!(sumW > 0.f))
         return res;

      // Categorical selection proportional to the resampling weight target/pdf. threshold < sumW
      // strictly for randNEE.y in [0,1), so the cumulative reaches a positive bucket: `found` is
      // guaranteed.
      const float32_t threshold = randNEE.y * sumW;
      float32_t3      pickedDir = float32_t3(0, 0, 0);
      float32_t       tWinner   = 0.f;
      float32_t       pdfWinner = 0.f;
      float32_t       accum     = 0.f;
      bool            found     = false;
      for (uint16_t k = 0u; k < kRISCandidates; ++k)
      {
         const float32_t w = (candPdf[k] > 0.f) ? (candTarget[k] / candPdf[k]) : 0.f;
         accum += w;
         if (!found && w > 0.f && accum >= threshold)
         {
            pickedDir = candDirs[k];
            tWinner   = candTarget[k];
            pdfWinner = candPdf[k];
            found     = true;
         }
      }

      ray_dir_info_t tmp;
      tmp.setDirection(pickedDir);
      const light_sample_t    L        = light_sample_t::create(tmp, shadingNormal);
      const value_weight_type bxdfEval = diffuse.evalAndWeight(L, interaction);

      const float32_t risWeight = (sumW / float32_t(kRISCandidates)) / tWinner;
#if NBL_MIS_MODE == NBL_MIS_MODE_NEE_ONLY
      // No competing BSDF technique: NEE carries the full direct-lighting estimate.
      const float32_t misWeight = 1.0f;
#else
      // pNee = selectionPdf * directionalPdf, both factors identical to the backward (shadeEmission)
      // side: selectionPdf = winner.pProposal (matches emitterSelectBackPdf), directionalPdf =
      // pyramid.backwardWeight (the same closed form, NOT pdfWinner, the bilinear forwardPdf used for
      // 1/pdf). The 1/2pi hemisphere rolloff is applied to the directional pdf as on the backward side.
      const float32_t3 pickedDirLocal = float32_t3(hlsl::dot(pickedDir, frameT), hlsl::dot(pickedDir, frameB), hlsl::dot(pickedDir, shadingNormal));
      const float32_t  neeDirProto    = hlsl::max(pyramid.backwardWeight(pickedDirLocal) - 0.5f / numbers::pi<float32_t>, 0.f);
      const float32_t  pNee           = winner.pProposal * neeDirProto;
      float32_t        misWeight      = 0.0f;
      if (pNee > 0.f)
      {
         const float32_t misRatio = bxdfEval.weight() / pNee;
         misWeight                = 1.0f / (1.f + misRatio * misRatio);
      }
#endif

      res.pickedDir       = pickedDir;
      res.pickedEmitterID = pickedEmitterID;
      // D_hat(winner) [direction RIS] * selWeight [selection RIS], the two layers compose:
      // the directional estimate is conditionally unbiased per leaf, selWeight estimates the sum.
      res.contribution = throughput * bxdfEval.value() * emission * risWeight * misWeight * selWeight;
      res.valid        = true;
      return res;
   }

   // Stash the BSDF-sampling vertex's frame so the next bounce's emission-on-hit can compute
   // the NEE pdf this technique would have assigned to the BSDF-sampled direction.
   void recordShadingVertex(const float32_t3 hitPos, const float32_t3 normal)
   {
      prevShadingHitPos = hitPos;
      prevShadingNormal = normal;
   }

   // Env-map radiance, deweighted against the sun-cone NEE technique via the power heuristic.
   static SEnvSample shadeEnvmap(const float32_t3 L, const float otherTechniqueHeuristic)
   {
      SEnvSample _sample = sampleEnv(L); // TODO: L might need to have a spread factor
#if NBL_MIS_MODE == NBL_MIS_MODE_BOTH
      if (otherTechniqueHeuristic > MISWeightThreshold)
      {
         const float neePdf      = (hlsl::dot(L, sunDir) > sunConeHalfAngleCos ? 1.f : 0.f) / (2.0 * numbers::pi<float32_t> * (1.0 - sunConeHalfAngleCos));
         const float weightRatio = neePdf * otherTechniqueHeuristic;
         _sample.color /= 1.f + weightRatio * weightRatio;
      }
#endif
      return _sample;
   }

   uint32_t  prevDescentNeeEmitterID;
   float32_t prevDescentNeePdf;
   // Prev-bounce shading frame for BSDF-side MIS against the tree-NEE technique.
   float32_t3 prevShadingHitPos;
   float32_t3 prevShadingNormal;
};

} // namespace this_example
} // namespace nbl

#endif
