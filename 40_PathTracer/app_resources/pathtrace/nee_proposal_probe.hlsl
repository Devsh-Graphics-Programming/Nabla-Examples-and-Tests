// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _PATHTRACER_40_NEE_PROPOSAL_PROBE_INCLUDED_
#define _PATHTRACER_40_NEE_PROPOSAL_PROBE_INCLUDED_

#if NBL_NEE_PROPOSAL_PROBE

// ===== Diagnostic-only takeover: visualize which emitter meshes are among the K NEE-
// candidate lights for the center pixel's sample 0.
//
//   * Pixels that don't hit an emitter -> dark gray (sky), mid gray (non-emitter surface).
//   * Pixels that hit an emitter NOT in the K -> very dark gray, so they fade away.
//   * Pixels that hit an emitter IN the K -> emitterColor(emitterID), full brightness
//     when the candidate's shadow ray would reach it, desaturated when occluded.
//   * Center pixel -> single white dot.
//
// Each pixel does its OWN primary ray to figure out what it sees. Only the emitter-hit
// pixels then re-trace the center pixel's primary + compute K candidates + possibly fire
// one shadow ray, so the redundant work is bounded by the area emitters cover on screen.
// Writes the visualization to gAlbedo; the caller (raygen) returns right after.
void runNeeProposalProbe(const uint16_t3 launchID, NBL_CONST_REF_ARG(SPixelSamplingInfo) samplingInfo)
{
   const uint32_t2 centerXY = uint32_t2(spirv::LaunchSizeKHR.x >> 1u, spirv::LaunchSizeKHR.y >> 1u);
   const bool      isCenter = (launchID.x == centerXY.x && launchID.y == centerXY.y);

   decltype(samplingInfo.randgen) probeRandgen = samplingInfo.randgen;
   const float32_t3 randVec0     = probeRandgen(uint16_t(0), 0u);
   const float32_t2 pixelSizeNDC = promote<float32_t2>(2.f) / float32_t2(spirv::LaunchSizeKHR.xy);

   //, Self primary: see what THIS pixel is looking at.
   const float32_t2  selfNDC     = float32_t2(launchID.xy) * pixelSizeNDC - promote<float32_t2>(1.f);
   const SPrimaryRay selfPrimary = genPrimaryRay(pc.sensorDynamics, pixelSizeNDC, selfNDC, float16_t2(randVec0.xy));
   spirv::HitObjectEXT                                                   selfHitObj;
   [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval selfPayload;
   selfPayload.init(randVec0.z, pc.sensorDynamics.tMax);
   spirv::hitObjectTraceRayEXT(selfHitObj, gTLASes[0], spv::RayFlagsMaskNone, 0xff, ESBTO_PATH, 0u, 0u, selfPrimary.ray.origin, selfPrimary.tMin, selfPrimary.ray.direction.getDirection(), pc.sensorDynamics.tMax, selfPayload);
   const bool     selfMissed         = spirv::hitObjectIsMissEXT(selfHitObj);
   // Resolved emitter ID (not the raw instance index) so it matches the candidate emitterIDs below.
   const uint32_t selfInstanceCustom = selfMissed ? 0xFFFFFFFFu : resolveEmitterID(uint32_t(spirv::hitObjectGetInstanceCustomIndexEXT(selfHitObj)), uint32_t(spirv::hitObjectGetGeometryIndexEXT(selfHitObj)));

   // Default backgrounds.
   float32_t3 col;
   if (selfMissed)
      col = float32_t3(0.03f, 0.03f, 0.04f);
   else if (selfInstanceCustom >= nbl::this_example::NonEmitterCustomIndex)
      col = float32_t3(0.08f, 0.08f, 0.09f); // non-emitter surface
   else
      col = float32_t3(0.04f, 0.04f, 0.04f); // emitter, not yet known if in the K, default dim

   // Emitter-hit pixels: re-trace center primary, compute the K candidates, look for a match.
   if (!selfMissed && selfInstanceCustom < nbl::this_example::NonEmitterCustomIndex)
   {
      const float32_t2  centerNDC     = float32_t2(centerXY) * pixelSizeNDC - promote<float32_t2>(1.f);
      const SPrimaryRay centerPrimary = genPrimaryRay(pc.sensorDynamics, pixelSizeNDC, centerNDC, float16_t2(randVec0.xy));
      spirv::HitObjectEXT                                                   centerHitObj;
      [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval centerPayload;
      centerPayload.init(randVec0.z, pc.sensorDynamics.tMax);
      spirv::hitObjectTraceRayEXT(centerHitObj, gTLASes[0], spv::RayFlagsMaskNone, 0xff, ESBTO_PATH, 0u, 0u, centerPrimary.ray.origin, centerPrimary.tMin, centerPrimary.ray.direction.getDirection(), pc.sensorDynamics.tMax, centerPayload);
      const bool centerHit = !spirv::hitObjectIsMissEXT(centerHitObj);

      if (centerHit)
      {
         SClosestHitRetval centerInfo = SClosestHitRetval::create(centerHitObj);
         const float32_t3 centerHitPos = centerInfo.hitPos;
         const float32_t3 centerNormal = centerInfo.geometricNormal;

         nbl::this_example::NextEventEstimator neeEst           = nbl::this_example::NextEventEstimator::create();
         const uint16_t                        sequenceProtoDim = PrimaryRayRandTripletsUsed;
         const float32_t3                      randNEE          = probeRandgen(sequenceProtoDim + uint16_t(1), 0u);

         // Search for a matching candidate.
         int        matchIdx     = -1;
         float32_t3 matchDir     = float32_t3(0, 1, 0);
         NBL_HLSL_LOOP for (uint32_t k = 0u; k < uint32_t(NEE_LIGHT_CANDIDATES); ++k)
         {
            nbl::this_example::NextEventEstimator::SProbeCandidate cand = neeEst.__probeCandidate(k, centerHitPos, centerNormal, randNEE);
            if (cand.emitterID == selfInstanceCustom && matchIdx < 0)
            {
               matchIdx = int(k);
               matchDir = cand.pickedDir;
            }
         }

         if (matchIdx >= 0)
         {
            // Read the emitter's own radiance and normalize to its chromaticity so the
            // cell color matches the emitter being shot at, regardless of intensity.
            float32_t3  emission = (gScene.init.pEmitters != 0) ? vk::RawBufferLoad<float32_t3>(gScene.init.pEmitters + uint64_t(selfInstanceCustom) * uint64_t(nbl::this_example::EmitterRecordSize)) : float32_t3(1.0f, 1.0f, 1.0f);
            const float peak     = hlsl::max(hlsl::max(emission.x, emission.y), emission.z);
            float32_t3  hue      = (peak > 0.f) ? (emission / peak) : float32_t3(1.0f, 1.0f, 1.0f);

            // Fire the shadow ray for THIS candidate, from CENTER's hit point in the
            // sampled direction. If it reaches the emitter, the cell shows full hue;
            // if occluded, it shows desaturated hue so the user can tell visibility.
            const float32_t3 originMagnitude = abs(centerHitPos);
            const float      offsetMagnitude = hlsl::max(hlsl::max(hlsl::exp2(8.f), originMagnitude.x), hlsl::max(originMagnitude.y, originMagnitude.z)) * hlsl::exp2(-20.f);
            const float32_t3 shadowOrigin    = centerHitPos + centerNormal * offsetMagnitude;
            const float32_t  tMin            = 0.f;

            bool shadowHit;
            [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval shadowPayload;
            shadowPayload.init(0.f, hlsl::numeric_limits<float32_t>::max);
            spirv::HitObjectEXT shadowHitObj;
            spirv::hitObjectTraceRayEXT(shadowHitObj, gTLASes[0], 0u, 0xff, ESBTO_PATH, 0u, ESBTO_PATH, shadowOrigin, tMin, matchDir, hlsl::numeric_limits<float32_t>::max, shadowPayload);
            shadowHit = !spirv::hitObjectIsMissEXT(shadowHitObj) && resolveEmitterID(uint32_t(spirv::hitObjectGetInstanceCustomIndexEXT(shadowHitObj)), uint32_t(spirv::hitObjectGetGeometryIndexEXT(shadowHitObj))) == selfInstanceCustom;

            if (shadowHit)
               col = hue;
            else
            {
               // Desaturated: blend toward gray
               const float lum = hlsl::dot(hue, float32_t3(0.3f, 0.6f, 0.1f));
               col = lerp(float32_t3(lum, lum, lum), hue, 0.25f) * 0.55f;
            }
         }
      }
   }

   // Center pixel marker overrides everything.
   if (isCenter)
      col = float32_t3(1.0f, 1.0f, 1.0f);

   gAlbedo[launchID] = float32_t4(col, 1.0f);
}

#endif // NBL_NEE_PROPOSAL_PROBE
#endif // _PATHTRACER_40_NEE_PROPOSAL_PROBE_INCLUDED_
