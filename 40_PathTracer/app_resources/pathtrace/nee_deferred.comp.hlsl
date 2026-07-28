// Fused NEE+resolve pass for the batched deferral: one thread per pixel replays its samples' bounce
// slots with the exact inline random numbers and splats the result.
// - The ray query forces opaque (inline anyhit ran stochastic alpha); current scenes are opaque.
// - V is not in the record: OrenNayar A=0 is V-independent, the shading normal stands in.

#include "common.hlsl"
#include "renderer/shaders/bda_accessors.hlsl"
#include "next_event_estimator.hlsl"
#include "emitter_resolve.hlsl"
#include "accumulation.hlsl"
#include "renderer/shaders/pathtrace/nee_deferred_common.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"

[[vk::push_constant]] nbl::this_example::SBeautyPushConstants pc;

using namespace nbl::this_example;

[shader("compute")]
[numthreads(NeeDeferredWorkgroupSize, 1, 1)]
void neeDeferredMain(uint32_t3 groupId: SV_GroupID, uint32_t3 groupThreadId: SV_GroupThreadID)
{
   const SBeautyPushConstants::S16BitData unpacked16BitPC = pc.get16BitData();

   const uint32_t2 renderSize = uint32_t2(gSensor.renderSize);
   const uint32_t  S          = uint32_t(unpacked16BitPC.maxSppPerDispatch);

   // 8x8 tile swizzle (compute doesn't get the RT dispatch's hardware swizzle). The dispatch covers
   // one horizontal band: bandLocal addresses the records, coord is the actual pixel.
   const uint32_t  tilesX    = (renderSize.x + 7u) / 8u;
   const uint32_t3 bandLocal = uint32_t3((groupId.x % tilesX) * 8u + (groupThreadId.x & 7u), (groupId.x / tilesX) * 8u + (groupThreadId.x >> 3u), 0u);
   const uint32_t2 bandSize  = uint32_t2(renderSize.x, uint32_t(pc.tileHeight));
   const uint32_t3 coord     = bandLocal + uint32_t3(0u, uint32_t(pc.tileOffsetY), 0u);
   if (bandLocal.x >= renderSize.x || bandLocal.y >= bandSize.y || coord.y >= renderSize.y)
      return;
   gAccumulationCoord = uint16_t3(coord);

   const uint32_t bandPixels   = bandSize.x * bandSize.y;
   const uint32_t bandPixelIdx = bandLocal.y * bandSize.x + bandLocal.x;
   const uint32_t poolCount    = bandPixels * S;

   // advanceSampleCount already ran in raygen; read the post-advance count.
   const uint32_t  newSampleCount    = gSampleCount[uint16_t3(coord)];
   const float32_t rcpNewSampleCount = 1.f / float32_t(newSampleCount);

   using NEE = NextEventEstimator;
   NEE::brdf_t::SCreationParams cParams;
   cParams.A                 = 0.f;
   const NEE::brdf_t diffuse = NEE::brdf_t::create(cParams);

   float32_t3 referenceFrameSum = float32_t3(0, 0, 0);
   uint32_t   samplesTaken      = 0u;
   uint32_t   firstSample       = 0u;
   NBL_HLSL_LOOP
   for (uint32_t s = 0u; s < S; s++)
   {
      NeeDeferredRecords       records = NeeDeferredRecords::create(pc.pNeeRequests, s * bandPixels + bandPixelIdx, poolCount);
      const SNeeDeferredHeader header  = records.loadHeader();
      if (header.sampleWord == NeeDeferredNoSample)
         continue;
      const uint32_t sampleIndex = header.sampleWord & NeeDeferredSampleIdxMask;
      const uint32_t bounceCount = (header.sampleWord >> NeeDeferredBounceShift) & NeeDeferredBounceMask;

      float32_t3 color = header.pathColor;

      // Exact only because each dispatch is one sample: the state advances per fetch, so a multi-sample
      // dispatch would depend on history.
      randgen_t randgen;
      randgen.pSampleBuffer       = gScene.init.pSampleSequence;
      randgen.rng                 = scramble_state_t::construct(gScrambleKey[uint16_t3(uint16_t2(coord.xy) & uint16_t(511), 0)]);
      randgen.sequenceSamplesLog2 = gScene.init.sequenceSamplesLog2;
      const float32_t3 randVec    = randgen(0u, sampleIndex);

      const float32_t2  pixelSizeNDC = promote<float32_t2>(2.f) / float32_t2(renderSize);
      const float32_t2  NDC          = float32_t2(coord.xy) * pixelSizeNDC - promote<float32_t2>(1.f);
      const SPrimaryRay primary      = genPrimaryRay(pc.sensorDynamics, pixelSizeNDC, NDC, float16_t2(randVec.xy), 1u);
      float32_t3        prevRayOrigin = primary.ray.origin;

      // bounces in order so the same-emitter MIS cache and prev-shading vertex flow like inline
      NEE nee = NEE::create();
      SNeeDeferredBounce bounce;
      if (bounceCount != 0u)
         bounce = records.loadBounce(1u);
      NBL_HLSL_LOOP
      for (uint32_t d = 1u; d <= bounceCount; d++)
      {
         // software pipeline: the next bounce's taps load while this bounce's estimator + shadow ray run
         SNeeDeferredBounce nextBounce;
         if (d < bounceCount)
            nextBounce = records.loadBounce(d + 1u);

#if NBL_MIS_MODE == NBL_MIS_MODE_BOTH
         if (bounce.flags & NeeDeferredFlagEmission)
            color += nee.backwardNEE(bounce.flags & NeeDeferredEmitterMask, bounce.hitPos, bounce.otherTechniqueHeuristic, bounce.throughput);
#endif

         if (bounce.flags & NeeDeferredFlagNEE)
         {
            // inline drew randBRDF before its NEE pair, skip its two advances
            randgen.rng();
            randgen.rng();
            const uint16_t   sequenceProtoDim = _static_cast<uint16_t>((d - 1u) * RandDimTriplesPerDepth + PrimaryRayRandTripletsUsed);
            const float32_t3 randNEE          = randgen(sequenceProtoDim + uint16_t(1), sampleIndex);
            const float32_t3 randNEE2         = randgen(sequenceProtoDim + uint16_t(2), sampleIndex);

            // raygen's newRayOrigin recomputed exactly (relies on shadingNormal == geometricNormal);
            // the incoming origin chain: primary origin, then each bounce's shadow origin
            const float32_t3 originMagnitude = max(abs(bounce.hitPos), abs(prevRayOrigin));
            const float32_t  offsetMagnitude = hlsl::max(hlsl::max(hlsl::exp2(8.f), originMagnitude.x), hlsl::max(originMagnitude.y, originMagnitude.z)) * hlsl::exp2(-20.f);
            const float32_t3 shadowOrigin    = bounce.hitPos + bounce.shadingNormal * offsetMagnitude;
            prevRayOrigin                    = shadowOrigin;

            NEE::ray_dir_info_t V;
            V.setDirection(bounce.shadingNormal);
            NEE::isotropic_interaction_t interaction = NEE::isotropic_interaction_t::create(V, bounce.shadingNormal, bounce.throughput);

            // forwardNEE owns both shadow rays and only returns a valid sample when the emitter is visible.
            const NEE::SForwardSample fwd = nee.forwardNEE(bounce.hitPos, shadowOrigin, bounce.shadingNormal, interaction, diffuse, bounce.throughput, randNEE, randNEE2);
            if (fwd.valid)
               color += fwd.contribution * surfaceAlbedo();
            nee.recordShadingVertex(bounce.hitPos, bounce.shadingNormal);
         }
         bounce = nextBounce;
      }

      if (samplesTaken == 0u)
         firstSample = sampleIndex;
      samplesTaken++;
      referenceFrameSum += color;

      const bool                          doClear  = sampleIndex == 0u;
      rwmc::CascadeAccumulator<CCascades> colorAcc = rwmc::CascadeAccumulator<CCascades>::create(gSensor.splatting, doClear);
      colorAcc.addSample(_static_cast<uint16_t>(sampleIndex + 1u), accum_t(color));
   }
   if (samplesTaken == 0u)
      return;

   float32_t3 mean = (firstSample != 0u) ? gBeauty[uint16_t3(coord)].rgb : float32_t3(0, 0, 0);
   mean += (referenceFrameSum - mean * float32_t(samplesTaken)) * rcpNewSampleCount;
   gBeauty[uint16_t3(coord)] = float32_t4(mean, 1.0);
}
