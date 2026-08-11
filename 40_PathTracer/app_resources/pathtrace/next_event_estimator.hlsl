#ifndef _PATHTRACER_40_NEXT_EVENT_ESTIMATOR_INCLUDED_
#define _PATHTRACER_40_NEXT_EVENT_ESTIMATOR_INCLUDED_

#include "renderer/shaders/pt_config.hlsl"

#include "nbl/builtin/hlsl/sampling/alias_table.hlsl"
#if !NBL_NEE_DEFERRED
#if NBL_NEE_LEAF_MODE == 0
// OBB-silhouette method only: the triangle variants must not even compile pyramid/silhouette code.
#include "nbl/builtin/hlsl/shapes/obb_silhouette.hlsl"
#include "nbl/builtin/hlsl/sampling/spherical_pyramid.hlsl"
#else
#include "nbl/builtin/hlsl/shapes/spherical_triangle.hlsl"
#include "nbl/builtin/hlsl/sampling/spherical_triangle.hlsl"
#include "nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl"
#endif
#endif
#ifdef __HLSL_VERSION
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#endif

#include "common.hlsl"
#include "renderer/shaders/bda_accessors.hlsl"
#include "emitter_resolve.hlsl"

#if NBL_NEE_DEFERRED && NBL_MIS_MODE == NBL_MIS_MODE_BXDF_ONLY
#error "NBL_NEE_DEFERRED is pointless with NBL_MIS_MODE=1 (BxDF-only has no NEE)"
#endif

namespace nbl
{
namespace this_example
{

#ifdef __HLSL_VERSION
// NBL_LIGHTTREE_ALIAS_LOG2N is the single source of truth (renderer/shaders/light_tree.hlsl); the
// runtime table size is gScene.init.aliasTableSize.
using AliasSampler = nbl::hlsl::sampling::PackedAliasTableA<float32_t, float32_t, uint32_t, BDAReadAccessor<uint32_t>, BDAReadAccessor<float32_t>, NBL_LIGHTTREE_ALIAS_LOG2N>;

#if NBL_NEE_STATS
// SDebugProbe::neeStats counter indices
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeStatsCalls         = 0u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeStatsSelectionFail = 1u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeStatsSilhDegen     = 2u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeStatsDirDraws      = 3u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeStatsDirDegen      = 4u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeStatsDirZeroTarget = 5u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeStatsTraced        = 6u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeStatsConfirmed     = 7u;
// OBB path only; the triangle path folds this case into NeeStatsDirZeroTarget
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeStatsZeroContrib = 8u;
// per CALL, not per draw: catches pdf == inf flushing every target/pdf to zero
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeStatsNoUsable   = 9u;
NBL_CONSTEXPR_STATIC_INLINE uint64_t NeeStatsByteOffset = 32ull; // SDebugProbe::neeStats

inline void neeStatsAdd(const uint32_t counterIdx, const uint32_t v)
{
   if (gScene.init.pDebugProbe == 0 || v == 0u)
      return;
   nbl::hlsl::bda::__ptr<uint32_t> p = nbl::hlsl::bda::__ptr<uint32_t>::create(gScene.init.pDebugProbe + NeeStatsByteOffset + uint64_t(counterIdx) * 4ull);
   nbl::hlsl::glsl::atomicAdd(p.deref().ptr.value, v);
}
#endif
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

#if NBL_NEE_LEAF_MODE == 0 && !NBL_NEE_DEFERRED
#if NBL_NEE_PROJECTED_SPHRECT
   using pyramid_t = nbl::hlsl::sampling::SphericalPyramid<NBL_NEE_CALIPER != 0, nbl::hlsl::sampling::ProjectedSphericalRectangle<float32_t, false> >;
#else
   using pyramid_t = nbl::hlsl::sampling::SphericalPyramid<NBL_NEE_CALIPER != 0, nbl::hlsl::sampling::SphericalRectangle<float32_t>>;
#endif

   // Direction skips the inverse affine's translation column and is left un-renormalized, so ray 1's
   // committed t is already the world-space distance (reused as ray 2's tMax). Returns the gTLASes slot.
   static uint32_t __emitterModelRay(const uint32_t emitterID, const float32_t3 worldOrigin, const float32_t3 worldDir, NBL_REF_ARG(float32_t3) modelOrigin, NBL_REF_ARG(float32_t3) modelDir)
   {
      const uint64_t   addr = gScene.init.pEmitterRayQuery + uint64_t(emitterID) * uint64_t(EmitterRayQueryRecordSize);
      const float32_t4 r0   = vk::RawBufferLoad<float32_t4>(addr + 0ull, 16u);
      const float32_t4 r1   = vk::RawBufferLoad<float32_t4>(addr + 16ull, 16u);
      const float32_t4 r2   = vk::RawBufferLoad<float32_t4>(addr + 32ull, 16u);
      modelOrigin           = float32_t3(hlsl::dot(r0.xyz, worldOrigin) + r0.w, hlsl::dot(r1.xyz, worldOrigin) + r1.w, hlsl::dot(r2.xyz, worldOrigin) + r2.w);
      modelDir              = float32_t3(hlsl::dot(r0.xyz, worldDir), hlsl::dot(r1.xyz, worldDir), hlsl::dot(r2.xyz, worldDir));
      return vk::RawBufferLoad<uint32_t>(addr + 48ull, 4u);
   }
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

   // Cranley-Patterson rotation: candidate `idx` of `count` from one primary-sample uniform.
   static float32_t  __rotate1(const float32_t base, const uint32_t idx, const uint32_t count) { return hlsl::fract(base + float32_t(idx) / float32_t(count)); }
   static float32_t2 __rotate2(const float32_t2 base, const uint32_t idx) { return hlsl::fract(base + float32_t(idx) * float32_t2(0.7548776662466927f, 0.5698402909980532f)); }

#if !NBL_NEE_DEFERRED
   // ---- selection RIS, shared by ALL leaf modes (purely bbox-based) + the OBB-only silhouette -------
#if NBL_NEE_LEAF_MODE == 0
   static shapes::ClippedSilhouette __buildSilhouette(NBL_REF_ARG(shapes::OBBView<float32_t>) obbView,
      const float32_t3 hitPos,
      const float32_t3 frameT,
      const float32_t3 frameB,
      const float32_t3 normal,
      const uint32_t   emitterID)
   {

      const uint64_t   addr = gScene.init.pEmitterOBB + uint64_t(emitterID) * 48ull;
      const float32_t4 r0   = vk::RawBufferLoad<float32_t4>(addr + 0ull, 16u);
      const float32_t4 r1   = vk::RawBufferLoad<float32_t4>(addr + 16ull, 16u);
      const float32_t4 r2   = vk::RawBufferLoad<float32_t4>(addr + 32ull, 16u);
      const float32_t3 wc0       = float32_t3(r0.x, r1.x, r2.x);
      const float32_t3 wc1       = float32_t3(r0.y, r1.y, r2.y);
      const float32_t3 wc2       = float32_t3(r0.z, r1.z, r2.z);
      const float32_t3 originRel = float32_t3(r0.w, r1.w, r2.w) - hitPos;
      obbView.minCorner  = float32_t3(hlsl::dot(originRel, frameT), hlsl::dot(originRel, frameB), hlsl::dot(originRel, normal));
      obbView.columns[0] = float32_t3(hlsl::dot(wc0, frameT), hlsl::dot(wc0, frameB), hlsl::dot(wc0, normal));
      obbView.columns[1] = float32_t3(hlsl::dot(wc1, frameT), hlsl::dot(wc1, frameB), hlsl::dot(wc1, normal));
      obbView.columns[2] = float32_t3(hlsl::dot(wc2, frameT), hlsl::dot(wc2, frameB), hlsl::dot(wc2, normal));


      if (hlsl::dot(obbView.columns[0], obbView.columns[0]) < 1e-12f && hlsl::dot(hlsl::cross(obbView.columns[1], obbView.columns[2]), obbView.minCorner) < 0.f)
      {
         const float32_t3 tmp = obbView.columns[1];
         obbView.columns[1]   = obbView.columns[2];
         obbView.columns[2]   = tmp;
      }
      return shapes::ClippedSilhouette::create(obbView);
   }
#endif // NBL_NEE_LEAF_MODE == 0 (silhouette)

   // Emitter's leaf bbox, read from the co-located emitter record (48 B: radiance | leafHeap |
   // bboxMin | bboxMax | pad). One direct load on emitterID, so no emitter -> leaf reverse-map ->
   // leaf-record dependent 2-load chain (which was the path tracer's worst LGSB stall line).
   static LightTreeLeaf __getLeaf(const uint32_t emitterIdx)
   {
      const uint64_t addr = gScene.init.pEmitters + uint64_t(emitterIdx) * uint64_t(EmitterRecordSize);
      // Two 16-byte-aligned uint4 taps over the bbox half of the record (the 48 B stride keeps the
      // record 16-aligned). b1 = bboxMin.xyz | bboxMax.x; b2 = bboxMax.yz | pad | pad.
      const uint32_t4 b1 = vk::RawBufferLoad<uint32_t4>(addr + 16ull, 16u);
      const uint32_t4 b2 = vk::RawBufferLoad<uint32_t4>(addr + 32ull, 16u);
      LightTreeLeaf   leaf;
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
#if NEE_GEOMTARGET_DISTANCE == 2
      // Matches tree weight mode 4. Saturates near the cluster instead of exploding like 1/dist^2.
      return (1.f - cosAlpha) * orientFactor;
#elif NEE_GEOMTARGET_DISTANCE == 1
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
         AliasSampler alias = AliasSampler::create(BDAReadAccessor<uint32_t>::create(gScene.init.pAliasEntries), BDAReadAccessor<float32_t>::create(gScene.init.pAliasPdf), gScene.init.aliasTableSize);
         AliasSampler::cache_type aliasCache;
         c.emitterID              = alias.generate(u, aliasCache);
         c.pProposal              = alias.forwardPdf(u, aliasCache);
         const LightTreeLeaf leaf = __getLeaf(c.emitterID);
         c.bboxMin                = leaf.bboxMin;
         c.bboxMax                = leaf.bboxMax;
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
#endif // !NBL_NEE_DEFERRED (selection RIS)

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

      const float32_t       u = __rotate1(randNEE.x, k, uint32_t(NEE_LIGHT_CANDIDATES));
      const SLightCandidate c = __drawPowerCandidate(u, hitPos, shadingNormal);
      if (!(c.pProposal > 0.f) || c.emitterID >= NonEmitterCustomIndex)
         return r;

      float32_t3 frameT, frameB;
      math::frisvad<float32_t3>(shadingNormal, frameT, frameB);

      shapes::OBBView<float32_t>      obbView;
      const shapes::ClippedSilhouette silhouette = __buildSilhouette(obbView, hitPos, frameT, frameB, shadingNormal, c.emitterID);
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

#if !NBL_NEE_DEFERRED
   // Backward probability that NEE would have selected this emitter from prev's shading point.
   float32_t __emitterSelectBackPdf(const uint32_t emitterIdx)
   {
#if NBL_NEE_USE_ALIAS
      AliasSampler aliasBwd = AliasSampler::create(BDAReadAccessor<uint32_t>::create(gScene.init.pAliasEntries), BDAReadAccessor<float32_t>::create(gScene.init.pAliasPdf), gScene.init.aliasTableSize);
      return aliasBwd.backwardPdf(emitterIdx);
#else
      // leafHeap is co-located in the emitter record (offset 12), so no reverse-map load.
      const uint32_t   leafIdxBwd = vk::RawBufferLoad<uint32_t>(gScene.init.pEmitters + uint64_t(emitterIdx) * uint64_t(EmitterRecordSize) + 12ull);
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
#if NBL_NEE_LEAF_MODE != 0
   // ---- Single-triangle baseline helpers (prior art: triangles in the light tree, sampled directly) ----
   // World-space triangle verts from the dedicated buffer (NOT SEmitterGPU; keeps the OBB record 48 B).
   static void __getTriVerts(const uint32_t emitterID, NBL_REF_ARG(float32_t3) v0, NBL_REF_ARG(float32_t3) v1, NBL_REF_ARG(float32_t3) v2)
   {
      const uint64_t addr = gScene.init.pEmitterTriVerts + uint64_t(emitterID) * 36ull;
      v0                  = vk::RawBufferLoad<float32_t3>(addr + 0ull, 4u);
      v1                  = vk::RawBufferLoad<float32_t3>(addr + 12ull, 4u);
      v2                  = vk::RawBufferLoad<float32_t3>(addr + 24ull, 4u);
   }

   // dirPdf is the realized density (estimator's 1/pdf); dirWeight is the MIS weight and must be
   // evaluated identically to __triBackwardPdf or the partition stops summing to 1 and the result goes
   // dark. They coincide for uniform/Arvo but not for the projected warp.
   static bool __sampleTri(const float32_t3 origin,
      const float32_t3                      normal,
      const float32_t3                      vertex0,
      const float32_t3                      vertex1,
      const float32_t3                      vertex2,
      const float32_t2                      xi,
      NBL_REF_ARG(float32_t3) L,
      NBL_REF_ARG(float32_t) dirPdf,
      NBL_REF_ARG(float32_t) dirWeight)
   {
#if NBL_NEE_LEAF_MODE == 1 // uniform area (ex31 ShapeSampling<PST_TRIANGLE,PPM> area path)
      const float32_t3 edge0      = vertex1 - vertex0;
      const float32_t3 edge1      = vertex2 - vertex0;
      const float32_t  sqrtU      = hlsl::sqrt(xi.x);
      const float32_t3 pnt        = vertex0 + edge0 * (1.0 - sqrtU) + edge1 * sqrtU * xi.y;
      L                           = pnt - origin;
      const float32_t distanceSq  = hlsl::dot(L, L);
      const float32_t rcpDistance = 1.0 / hlsl::sqrt(distanceSq);
      L *= rcpDistance;
      dirPdf    = distanceSq / hlsl::abs(hlsl::dot(hlsl::cross(edge0, edge1) * 0.5f, L));
      dirWeight = dirPdf; // exact: backward == forward
      return dirPdf > numeric_limits<float32_t>::min && !hlsl::isinf(dirPdf);
#else // Arvo (2) / projected (3)
      const float32_t3                           tri_vertices[3] = { vertex0, vertex1, vertex2 };
      const shapes::SphericalTriangle<float32_t> st              = shapes::SphericalTriangle<float32_t>::create(tri_vertices, origin);
#if NBL_NEE_LEAF_MODE == 2 // ex31 ShapeSampling<PST_TRIANGLE,PPM_SOLID_ANGLE>
      sampling::SphericalTriangle<float32_t>             sst = sampling::SphericalTriangle<float32_t>::create(st);
      sampling::SphericalTriangle<float32_t>::cache_type cache;
      L         = sst.generate(xi, cache);
      dirPdf    = sst.forwardPdf(xi, cache);
      dirWeight = dirPdf; // exact: rcpSolidAngle both ways
#else // ex31 ShapeSampling<PST_TRIANGLE,PPM_APPROX_PROJECTED_SOLID_ANGLE>
      sampling::ProjectedSphericalTriangle<float32_t>             pst = sampling::ProjectedSphericalTriangle<float32_t>::create(st, normal, false);
      sampling::ProjectedSphericalTriangle<float32_t>::cache_type pstCache;
      L         = pst.generate(xi, pstCache);
      dirPdf    = pst.forwardPdf(xi, pstCache);
      dirWeight = pst.forwardWeight(xi, pstCache);
#endif
      return dirPdf > numeric_limits<float32_t>::min && !hlsl::isinf(dirPdf) && !hlsl::any(hlsl::isnan(L));
#endif
   }

   static float32_t __triBackwardPdf(
      const float32_t3 origin, const float32_t3 normal, const float32_t3 vertex0, const float32_t3 vertex1, const float32_t3 vertex2, const float32_t3 L, const float32_t dist)
   {
#if NBL_NEE_LEAF_MODE == 1 // Area triangle
      const float32_t3 normalTimesArea = hlsl::cross(vertex1 - vertex0, vertex2 - vertex0) * 0.5f;
      const float32_t  denom           = hlsl::abs(hlsl::dot(normalTimesArea, L));
      return (denom > numeric_limits<float32_t>::min) ? (dist * dist / denom) : 0.f;
#elif NBL_NEE_LEAF_MODE > 1 // is solid angle triangle
      const float32_t3                           tri_vertices[3] = { vertex0, vertex1, vertex2 };
      const shapes::SphericalTriangle<float32_t> triangleShape   = shapes::SphericalTriangle<float32_t>::create(tri_vertices, origin);
#if NBL_NEE_LEAF_MODE == 2 // Arvo
      sampling::SphericalTriangle<float32_t> triSampler = sampling::SphericalTriangle<float32_t>::create(triangleShape);
#elif NBL_NEE_LEAF_MODE == 3 // Projected Arvo
      sampling::ProjectedSphericalTriangle<float32_t> triSampler = sampling::ProjectedSphericalTriangle<float32_t>::create(triangleShape, normal, false);
#endif // NBL_NEE_LEAF_MODE > 1 // is solid angle triangle
      return triSampler.backwardWeight(L);
#elif NBL_NEE_LEAF_MODE == 0 || NBL_NEE_LEAF_MODE < 3
#error "Not a triangle!"
#endif // NBL_NEE_LEAF_MODE == 1 // Area triangle
   }

   // Triangle baseline backward MIS: same directional pdf as forwardNEE, evaluated at the arrival dir.
   float32_t __emissionDeweight(const uint32_t emitterIdx, const float32_t3 currentHitPos, const float32_t emitterSelectBackPdf, const float32_t otherTechniqueHeuristic)
   {
      float32_t3 v0, v1, v2;
      __getTriVerts(emitterIdx, v0, v1, v2);
      const float32_t3 d        = currentHitPos - prevShadingHitPos;
      const float32_t  dist     = hlsl::length(d);
      const float32_t3 L        = d / dist;
      const float32_t  dirProto = __triBackwardPdf(prevShadingHitPos, prevShadingNormal, v0, v1, v2, L, dist);
      if (!(dirProto > 0.f))
         return 1.f;
      const float32_t neePdf      = emitterSelectBackPdf * dirProto;
      const float32_t weightRatio = neePdf * otherTechniqueHeuristic;
      return 1.f / (1.f + weightRatio * weightRatio);
   }
#else
   float32_t __emissionDeweight(const uint32_t emitterIdx, const float32_t3 currentHitPos, const float32_t emitterSelectBackPdf, const float32_t otherTechniqueHeuristic)
   {
      const LightTreeLeaf leaf = __getLeaf(emitterIdx);
      float32_t3          prevT, prevB;
      math::frisvad<float32_t3>(prevShadingNormal, prevT, prevB);

      shapes::OBBView<float32_t>      obbView;
      const shapes::ClippedSilhouette silhouette = __buildSilhouette(obbView, prevShadingHitPos, prevT, prevB, prevShadingNormal, emitterIdx);
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
#endif // NBL_NEE_LEAF_MODE
#endif // !NBL_NEE_DEFERRED (__emitterSelectBackPdf + __emissionDeweight)

   // Emission on a BSDF-sampled hit, deweighted against the NEE technique via the power heuristic.
   // otherTechniqueHeuristic is 1/bsdfWeight from the previous bounce.
   spectral_type backwardNEE(const uint32_t emitterIdx, const float32_t3 currentHitPos, const float32_t otherTechniqueHeuristic, const spectral_type throughput)
   {
      if (!(emitterIdx < NonEmitterCustomIndex && gScene.init.pEmitters != 0))
         return spectral_type(0, 0, 0);

      float32_t3 emission = vk::RawBufferLoad<float32_t3>(gScene.init.pEmitters + uint64_t(emitterIdx) * uint64_t(EmitterRecordSize));

// Deferred raygen compiles the deweight machinery out and records the deweight-needing hits instead;
// the fused compute pass calls this full version.
#if NBL_MIS_MODE == NBL_MIS_MODE_BOTH && !NBL_NEE_DEFERRED
      if (otherTechniqueHeuristic > MISWeightThreshold && gScene.init.pEmitterToLeafIdx != 0)
      {
         // Same-emitter cache hit (set at NEE forward time) supplies the selection pdf for free; a miss
         // passes a negative sentinel so the backward climb runs where it is cheapest. In the callable
         // build that is inside emissionCallable, keeping the climb's register / i-cache footprint out
         // of raygen; inline it runs here.
         const float32_t cachedBackPdf = (emitterIdx == prevDescentNeeEmitterID) ? prevDescentNeePdf : -1.f;

         // compute NEE MIS backward weight on the contribution color
         float32_t deweight;

         const float32_t emitterSelectBackPdf = (cachedBackPdf < 0.f) ? __emitterSelectBackPdf(emitterIdx) : cachedBackPdf;
         deweight                             = (emitterSelectBackPdf > 0.f) ? __emissionDeweight(emitterIdx, currentHitPos, emitterSelectBackPdf, otherTechniqueHeuristic) : 1.f;
         assert(!hlsl::isinf(deweight));
         // apply emissive weight
         emission *= deweight;
      }
#endif

      return emission * throughput;
   }

#if !NBL_NEE_DEFERRED
   // Traces both shadow rays itself from shadowOrigin and assembles the contribution ONLY for visible
   // samples, so the caller just multiplies res.contribution by albedo when res.valid. Caches the
   // selection pdf for the next bounce's emission-side MIS.
   SForwardSample forwardNEE(const float32_t3 hitPos,
      const float32_t3                        shadowOrigin,
      const float32_t3                        shadingNormal,
      NBL_CONST_REF_ARG(isotropic_interaction_t) interaction,
      NBL_CONST_REF_ARG(brdf_t) diffuse,
      const spectral_type throughput,
      const float32_t3    randNEE,
      const float32_t3    randNEE2)
   {
      SForwardSample res;
      res.pickedDir       = float32_t3(0, 0, 0);
      res.pickedEmitterID = NonEmitterCustomIndex;
      res.contribution    = spectral_type(0, 0, 0);
      res.valid           = false;

#if NBL_NEE_STATS
      neeStatsAdd(NeeStatsCalls, 1u);
#endif // NBL_NEE_STATS
      // Candidates are correlated (one uniform rotated per index) but each stays marginally uniform, so
      // the proposal pdf is still exact, RIS only needs marginals.
      static const uint16_t kLightCandidates = uint16_t(NEE_LIGHT_CANDIDATES);
      float32_t             sumG             = 0.f;
      uint16_t              winnerIdx        = 0u;
      float32_t             winnerGeom       = 0.f;
      bool                  selFound         = false;
      float32_t             selPick          = randNEE2.z; // rescaled within the chosen branch each step to stay uniform
      for (uint16_t m = 0u; m < kLightCandidates; ++m)
      {
         const float32_t       u    = __rotate1(randNEE.x, uint32_t(m), uint32_t(kLightCandidates));
         const SLightCandidate cand = __drawPowerCandidate(u, hitPos, shadingNormal);
         const float32_t       g    = cand.geomTarget;
         if (g > 0.f)
         {
            sumG += g;
            const float32_t pReplace = g / sumG; // first valid candidate: pReplace == 1 -> always wins
            if (selPick < pReplace)
            {
               winnerIdx  = m;
               winnerGeom = g;
               selFound   = true;
               selPick    = selPick / pReplace;
            }
            else
               selPick = (selPick - pReplace) / (1.f - pReplace);
         }
      }
      if (!selFound) // no candidate above the horizon / with positive geometry
      {
#if NBL_NEE_STATS
         neeStatsAdd(NeeStatsSelectionFail, 1u);
#endif // NBL_NEE_STATS
         return res;
      }
      // Redraw only the winner; same rotation reproduces candidate winnerIdx's leaf.
      const float32_t       uWinner = __rotate1(randNEE.x, uint32_t(winnerIdx), uint32_t(kLightCandidates));
      const SLightCandidate winner  = __drawPowerCandidate(uWinner, hitPos, shadingNormal);
      // (1/M) sum(t_i/p_i) / t_winner with t_i = power_i * geom_i, p_i = power_i / totalPower.
      const float32_t selWeight = (sumG / float32_t(kLightCandidates)) / (winnerGeom * winner.pProposal);
      const uint32_t  emitterID = winner.emitterID;

      const spectral_type emission = vk::RawBufferLoad<float32_t3>(gScene.init.pEmitters + uint64_t(emitterID) * uint64_t(EmitterRecordSize));

      prevDescentNeeEmitterID = emitterID;
      prevDescentNeePdf       = winner.pProposal;

      // ---- directional sampler setup (leaf-mode-specific) -----------------------------------------
#if NBL_NEE_LEAF_MODE == 0
      float32_t3 frameT, frameB;
      math::frisvad<float32_t3>(shadingNormal, frameT, frameB);
      shapes::OBBView<float32_t>      obbView;
      const shapes::ClippedSilhouette silhouette = __buildSilhouette(obbView, hitPos, frameT, frameB, shadingNormal, emitterID);
      // Bail out on degenerate silhouette (observer inside OBB or fully horizon-clipped).
      if (silhouette.count == 0u)
      {
#if NBL_NEE_STATS
         neeStatsAdd(NeeStatsSilhDegen, 1u);
#endif // NBL_NEE_STATS
         return res;
      }
      pyramid_t pyramid = pyramid_t::create(silhouette, obbView);
#else // NBL_NEE_LEAF_MODE != 0
      float32_t3 v0, v1, v2;
      __getTriVerts(emitterID, v0, v1, v2);
#endif // NBL_NEE_LEAF_MODE == 0

      // direction RIS: correlated candidates (__rotate2, marginally uniform), online weighted reservoir.
      static const uint16_t kRISCandidates = uint16_t(NEE_RIS_CANDIDATES);
      float32_t             dirPick        = randNEE.y; // rescaled within the chosen branch each step to stay uniform
      float32_t3            pickedDir      = float32_t3(0, 0, 0);
      float32_t             tWinner        = 0.f;
      float32_t             sumW           = 0.f;
      bool                  found          = false;
      value_weight_type     winnerEv;
#if NBL_MIS_MODE != NBL_MIS_MODE_NEE_ONLY
      float32_t dirWeightWinner = 0.f; // winner's directional MIS weight (backwardWeight for OBB/projected, pdf for uniform/Arvo)
#endif // NBL_MIS_MODE != NEE_ONLY
#if NBL_NEE_STATS
      uint32_t statsDegen = 0u, statsZero = 0u;
#if NBL_NEE_LEAF_MODE == 0
      uint32_t statsZeroContrib = 0u;
#endif // NBL_NEE_LEAF_MODE == 0
#endif // NBL_NEE_STATS
      for (uint16_t k = 0u; k < kRISCandidates; ++k)
      {
         const float32_t2 u = __rotate2(randNEE2.xy, uint32_t(k));

         float32_t3 dir    = float32_t3(0, 1, 0);
         float32_t  dirPdf = 0.f;
#if NBL_NEE_LEAF_MODE == 0
         pyramid_t::cache_type pyrCache;
         const float32_t3      dirLocal = pyramid.generate(u, pyrCache);
         dir                            = frameT * dirLocal.x + frameB * dirLocal.y + shadingNormal * dirLocal.z;
         dirPdf                         = pyramid.forwardPdf(u, pyrCache); // <= 0 => degenerate/clipped sample (zero-weight proposal)
#if NBL_NEE_STATS
         if (!(dirPdf > 0.f))
         {
            if (dirLocal.z <= 0.f)
               statsZero++;
            else
               statsDegen++;
         }
#endif // NBL_NEE_STATS
#if NBL_MIS_MODE != NBL_MIS_MODE_NEE_ONLY
         // backwardWeight HERE (pyramid's last use) so the pyramid dies before the material eval below.
         const float32_t dirWeight = (dirPdf > 0.f) ? hlsl::max(pyramid.backwardWeight(dirLocal) - 0.5f / numbers::pi<float32_t>, 0.f) : 0.f;
#endif // NBL_MIS_MODE != NEE_ONLY
#else // NBL_NEE_LEAF_MODE != 0
         float32_t dirWeight = 0.f;
         if (!__sampleTri(hitPos, shadingNormal, v0, v1, v2, u, dir, dirPdf, dirWeight))
            dirPdf = 0.f;
#if NBL_NEE_STATS
         if (!(dirPdf > 0.f))
            statsDegen++;
#endif // NBL_NEE_STATS
#endif // NBL_NEE_LEAF_MODE == 0
         float32_t         target = 0.f;
         value_weight_type ev; // valid when dirPdf > 0
         if (dirPdf > 0.f)
         {
            ray_dir_info_t tmp;
            tmp.setDirection(dir);
            const light_sample_t Lk = light_sample_t::create(tmp, shadingNormal);
            ev                      = diffuse.evalAndWeight(Lk, interaction);
            target                  = hlsl::max(__luma(throughput * ev.value() * emission), 0.f);
         }
#if NBL_NEE_STATS
         if (dirPdf > 0.f && !(target > 0.f))
#if NBL_NEE_LEAF_MODE == 0
            statsZeroContrib++;
#else // NBL_NEE_LEAF_MODE != 0
            statsZero++;
#endif // NBL_NEE_LEAF_MODE == 0
#endif // NBL_NEE_STATS
         const float32_t w = (dirPdf > 0.f) ? (target / dirPdf) : 0.f;
         if (w > 0.f)
         {
            sumW += w;
            const float32_t pReplace = w / sumW; // first valid candidate: pReplace == 1 -> always wins
            if (dirPick < pReplace)
            {
               pickedDir = dir;
               tWinner   = target;
               winnerEv  = ev;
#if NBL_MIS_MODE != NBL_MIS_MODE_NEE_ONLY
               dirWeightWinner = dirWeight;
#endif // NBL_MIS_MODE != NEE_ONLY
               found   = true;
               dirPick = dirPick / pReplace;
            }
            else
               dirPick = (dirPick - pReplace) / (1.f - pReplace);
         }
      }
#if NBL_NEE_STATS
      neeStatsAdd(NeeStatsDirDraws, uint32_t(kRISCandidates));
      neeStatsAdd(NeeStatsDirDegen, statsDegen);
      neeStatsAdd(NeeStatsDirZeroTarget, statsZero);
#if NBL_NEE_LEAF_MODE == 0
      neeStatsAdd(NeeStatsZeroContrib, statsZeroContrib);
#endif // NBL_NEE_LEAF_MODE == 0
#endif // NBL_NEE_STATS
      if (!found)
      {
#if NBL_NEE_STATS
         neeStatsAdd(NeeStatsNoUsable, 1u);
#endif // NBL_NEE_STATS
         return res;
      }

#if NBL_MIS_MODE != NBL_MIS_MODE_NEE_ONLY
      const float32_t pNee = winner.pProposal * dirWeightWinner;
#endif // NBL_MIS_MODE != NEE_ONLY

#if NBL_NEE_SINGLE_RAY
#if NBL_NEE_STATS
      neeStatsAdd(NeeStatsTraced, 1u);
#endif // NBL_NEE_STATS
      {
         nbl::hlsl::spirv::RayQueryKHR q;
         nbl::hlsl::spirv::rayQueryInitializeKHR(q, gTLASes[0], spv::RayFlagsOpaqueKHRMask, 0xffu, shadowOrigin, 0.f, pickedDir, nbl::hlsl::numeric_limits<float32_t>::max);
         while (nbl::hlsl::spirv::rayQueryProceedKHR(q)) {}
         if (nbl::hlsl::spirv::rayQueryGetIntersectionTypeKHR(q, 1u) == 0u)
            return res; // hit nothing
         const uint32_t hitEmitterID = resolveHitEmitterID(nbl::hlsl::spirv::rayQueryGetIntersectionInstanceCustomIndexKHR(q, 1u),
            nbl::hlsl::spirv::rayQueryGetIntersectionGeometryIndexKHR(q, 1u),
            nbl::hlsl::spirv::rayQueryGetIntersectionPrimitiveIndexKHR(q, 1u));
         if (hitEmitterID != emitterID)
            return res; // a nearer occluder, or a different emitter, is in front
      }
#else // NBL_NEE_SINGLE_RAY
#if NBL_NEE_LEAF_MODE != 0
#error "two-ray visibility (NBL_NEE_SINGLE_RAY=0) is OBB-only; triangle leaf modes must use the single-ray path"
#endif // NBL_NEE_LEAF_MODE != 0
      // ray 1: closest-hit on the emitter's own geometry -> lit-point distance + rejection (does pickedDir reach it?).
      float32_t3                    modelOrigin, modelDir;
      const uint32_t                tlasSlot = __emitterModelRay(emitterID, shadowOrigin, pickedDir, modelOrigin, modelDir);
      nbl::hlsl::spirv::RayQueryKHR q1;
      // tlasSlot is divergent across the subgroup -> NonUniform required for the AS-array index.
      nbl::hlsl::spirv::rayQueryInitializeKHR(q1, gTLASes[NonUniformResourceIndex(tlasSlot)], spv::RayFlagsOpaqueKHRMask, 0xffu, modelOrigin, 0.f, modelDir, nbl::hlsl::numeric_limits<float32_t>::max);
      while (nbl::hlsl::spirv::rayQueryProceedKHR(q1)) {}
      if (nbl::hlsl::spirv::rayQueryGetIntersectionTypeKHR(q1, 1u) == 0u)
         return res; // rejection: the pyramid sampled a direction the geometry doesn't cover
      const float32_t shadowDist = nbl::hlsl::spirv::rayQueryGetIntersectionTKHR(q1, 1u);
#if NBL_NEE_STATS
      neeStatsAdd(NeeStatsTraced, 1u); // passed rejection -> a visibility ray is cast
#endif // NBL_NEE_STATS

      // ray 2 runs non-opaque so the picked emitter is skipped by identity: no tMax backoff, no self-occlusion.
      {
         nbl::hlsl::spirv::RayQueryKHR q2;
         nbl::hlsl::spirv::rayQueryInitializeKHR(q2, gTLASes[0], spv::RayFlagsNoOpaqueKHRMask | spv::RayFlagsTerminateOnFirstHitKHRMask, 0xffu, shadowOrigin, 0.f, pickedDir, shadowDist);
         while (nbl::hlsl::spirv::rayQueryProceedKHR(q2))
         {
            const uint32_t cEmitter = resolveHitEmitterID(nbl::hlsl::spirv::rayQueryGetIntersectionInstanceCustomIndexKHR(q2, 0u),
               nbl::hlsl::spirv::rayQueryGetIntersectionGeometryIndexKHR(q2, 0u),
               nbl::hlsl::spirv::rayQueryGetIntersectionPrimitiveIndexKHR(q2, 0u));
            if (cEmitter != emitterID)
               nbl::hlsl::spirv::rayQueryConfirmIntersectionKHR(q2); // genuine occluder -> commit & terminate
         }
         if (nbl::hlsl::spirv::rayQueryGetIntersectionTypeKHR(q2, 1u) != 0u)
            return res; // occluded
      }
#endif // NBL_NEE_SINGLE_RAY

      const value_weight_type bxdfEval = winnerEv; // winner's RIS-loop eval reused (keeps interaction off the ray frame)

      const float32_t risWeight = (sumW / float32_t(kRISCandidates)) / tWinner;
#if NBL_MIS_MODE == NBL_MIS_MODE_NEE_ONLY
      const float32_t misWeight = 1.0f;
#else // NBL_MIS_MODE != NBL_MIS_MODE_NEE_ONLY
      float32_t misWeight = 0.0f;
      if (pNee > 0.f)
      {
         const float32_t misRatio = bxdfEval.weight() / pNee;
         misWeight                = 1.0f / (1.f + misRatio * misRatio);
      }
#endif // NBL_MIS_MODE == NBL_MIS_MODE_NEE_ONLY

      res.pickedDir       = pickedDir;
      res.pickedEmitterID = emitterID;
      res.contribution    = throughput * bxdfEval.value() * emission * misWeight * risWeight * selWeight;
      res.valid           = true;
#if NBL_NEE_STATS
      neeStatsAdd(NeeStatsConfirmed, 1u); // visible -> contribution assembled
#endif // NBL_NEE_STATS
      return res;
   }
#endif // !NBL_NEE_DEFERRED (forwardNEE)

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
