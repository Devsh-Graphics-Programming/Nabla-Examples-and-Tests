// Wavefront sampler side: one thread per NEE-queue entry runs the estimator and splats finished paths.
// V is not in the record: OrenNayar A=0 is V-independent, the shading normal stands in.

#include "common.hlsl"
#include "renderer/shaders/bda_accessors.hlsl"
#include "next_event_estimator.hlsl"
#include "emitter_resolve.hlsl"
#include "accumulation.hlsl"
#include "renderer/shaders/pathtrace/wavefront_common.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"

[[vk::push_constant]] nbl::this_example::SBeautyPushConstants pc;

using namespace nbl::this_example;

[shader("compute")]
[numthreads(WavefrontWorkgroupSize, 1, 1)]
void waveNee(uint32_t3 dispatchId: SV_DispatchThreadID)
{
   const uint32_t2 renderSize = uint32_t2(gSensor.renderSize);
   const uint32_t  pixelCount = renderSize.x * renderSize.y;
   const uint32_t  neeCount   = vk::RawBufferLoad<uint32_t>(pc.pNeeRequests + WavefrontNeeCountOffset + 4ull * (uint32_t(pc.wavefrontBounce) & 1u), 4u);
   if (dispatchId.x >= neeCount)
      return;
   const uint32_t  pathId = vk::RawBufferLoad<uint32_t>(wavefrontNeeQueueAddress(pc.pNeeRequests, pixelCount) + uint64_t(dispatchId.x) * 4ull, 4u);
   const uint16_t3 coord  = uint16_t3(_static_cast<uint16_t>(pathId % renderSize.x), _static_cast<uint16_t>(pathId / renderSize.x), uint16_t(0));

   const uint32_t4 tap3 = vk::RawBufferLoad<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrColorRng, pathId), 16u);
   const uint32_t4 tap4 = vk::RawBufferLoad<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrPrevPos, pathId), 16u);
   const uint32_t4 tap5 = vk::RawBufferLoad<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrPrevNormal, pathId), 16u);
   const uint32_t4 tap6 = vk::RawBufferLoad<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrNeeHit, pathId), 16u);
   const uint32_t4 tap9 = vk::RawBufferLoad<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrNeeThroughput, pathId), 16u);

   const uint32_t   flags      = tap6.w;
   const float32_t3 hitPos     = asfloat(tap6.xyz);
   const float32_t3 throughput = asfloat(tap9.xyz);
   float32_t3       color      = asfloat(tap3.xyz);

   using NEE = NextEventEstimator;
   NEE nee                     = NEE::create();
   nee.prevShadingHitPos       = asfloat(tap4.xyz);
   nee.prevDescentNeePdf       = asfloat(tap4.w);
   nee.prevShadingNormal       = asfloat(tap5.xyz);
   nee.prevDescentNeeEmitterID = tap5.w;

#if NBL_MIS_MODE == NBL_MIS_MODE_BOTH
   if (flags & WavefrontFlagEmission)
      color += nee.backwardNEE(flags & WavefrontEmitterMask, hitPos, asfloat(tap9.w), throughput);
#endif

   if (flags & WavefrontFlagNEE)
   {
      const uint32_t4  tap7          = vk::RawBufferLoad<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrNeeShadow, pathId), 16u);
      const uint32_t4  tap8          = vk::RawBufferLoad<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrNeeNormal, pathId), 16u);
      const float32_t3 shadowOrigin  = asfloat(tap7.xyz);
      const float32_t3 shadingNormal = asfloat(tap8.xyz);
      const uint32_t   sampleIndex   = vk::RawBufferLoad<uint32_t>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrRayDir, pathId) + 12ull, 4u) & ~WavefrontPrimaryRayFlag;

      randgen_t randgen;
      randgen.pSampleBuffer       = gScene.init.pSampleSequence;
      randgen.rng                 = scramble_state_t::construct(uint32_t2(tap7.w, tap8.w));
      randgen.sequenceSamplesLog2 = gScene.init.sequenceSamplesLog2;
      const uint16_t   sequenceProtoDim = _static_cast<uint16_t>((uint32_t(pc.wavefrontBounce) - 1u) * RandDimTriplesPerDepth + PrimaryRayRandTripletsUsed);
      const float32_t3 randNEE          = randgen(sequenceProtoDim + uint16_t(1), sampleIndex);
      const float32_t3 randNEE2         = randgen(sequenceProtoDim + uint16_t(2), sampleIndex);

      NEE::ray_dir_info_t V;
      V.setDirection(shadingNormal);
      NEE::isotropic_interaction_t interaction = NEE::isotropic_interaction_t::create(V, shadingNormal, throughput);
      NEE::brdf_t::SCreationParams cParams;
      cParams.A                 = 0.f;
      const NEE::brdf_t diffuse = NEE::brdf_t::create(cParams);

      // forwardNEE owns both shadow rays and only returns a valid sample when the emitter is visible.
      const NEE::SForwardSample fwd = nee.forwardNEE(hitPos, shadowOrigin, shadingNormal, interaction, diffuse, throughput, randNEE, randNEE2);
      if (fwd.valid)
         color += fwd.contribution * surfaceAlbedo();
      vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrPrevPos, pathId), uint32_t4(asuint(hitPos), asuint(nee.prevDescentNeePdf)), 16u);
      vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrPrevNormal, pathId), uint32_t4(asuint(shadingNormal), nee.prevDescentNeeEmitterID), 16u);
   }

   vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrColorRng, pathId), uint32_t4(asuint(color), tap3.w), 16u);

   if (flags & WavefrontFlagFinalize)
   {
      const uint32_t sampleIndex = vk::RawBufferLoad<uint32_t>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrRayDir, pathId) + 12ull, 4u) & ~WavefrontPrimaryRayFlag;
      splatSampleAndMean(coord, sampleIndex, color);
   }
}
