// Wavefront trace side: consumes the bounce's ray queue, shades emission/env, queues NEE/emission work
// and appends continuation rays. NBL_NEE_DEFERRED=1 keeps light-sampler code out of this kernel.
// Forced-opaque ray queries (no anyhit alpha), beauty only, single-layer sensors.
#define NBL_NEE_DEFERRED 1

#include "common.hlsl"
#include "renderer/shaders/bda_accessors.hlsl"
#include "next_event_estimator.hlsl"
#include "emitter_resolve.hlsl"
#include "accumulation.hlsl"
#include "renderer/shaders/pathtrace/wavefront_common.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"

[[vk::push_constant]] nbl::this_example::SBeautyPushConstants pc;

using namespace nbl::this_example;

[shader("compute")]
[numthreads(WavefrontWorkgroupSize, 1, 1)]
void waveTrace(uint32_t3 dispatchId: SV_DispatchThreadID)
{
   const uint32_t2 renderSize = uint32_t2(gSensor.renderSize);
   const uint32_t  pixelCount = renderSize.x * renderSize.y;
   const uint32_t  bounce     = uint32_t(pc.wavefrontBounce);
   const uint32_t  parityIn   = (bounce - 1u) & 1u;
   const uint32_t  rayCount   = vk::RawBufferLoad<uint32_t>(pc.pNeeRequests + WavefrontRayCountOffset + 4ull * parityIn, 4u);
   if (dispatchId.x >= rayCount)
      return;
   const uint32_t  pathId = vk::RawBufferLoad<uint32_t>(wavefrontRayQueueAddress(pc.pNeeRequests, pixelCount, parityIn) + uint64_t(dispatchId.x) * 4ull, 4u);
   const uint16_t3 coord  = uint16_t3(_static_cast<uint16_t>(pathId % renderSize.x), _static_cast<uint16_t>(pathId / renderSize.x), uint16_t(0));

   const uint32_t4 tap0 = vk::RawBufferLoad<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrRayOrigin, pathId), 16u);
   const uint32_t4 tap1 = vk::RawBufferLoad<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrRayDir, pathId), 16u);
   const uint32_t4 tap2 = vk::RawBufferLoad<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrThroughputRng, pathId), 16u);
   const uint32_t4 tap3 = vk::RawBufferLoad<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrColorRng, pathId), 16u);
   const float32_t  otherTechniqueHeuristic = asfloat(vk::RawBufferLoad<uint32_t>(wavefrontHeuristicAddress(pc.pNeeRequests, pixelCount, pathId), 4u));

   const float32_t3 rayOrigin   = asfloat(tap0.xyz);
   const float32_t3 rayDir      = asfloat(tap1.xyz);
   const uint32_t   sampleIndex = tap1.w & ~WavefrontPrimaryRayFlag;
   const float32_t  tMax        = (tap1.w & WavefrontPrimaryRayFlag) ? pc.sensorDynamics.tMax : hlsl::numeric_limits<float32_t>::max;
   float32_t3       throughput  = asfloat(tap2.xyz);
   float32_t3       color       = asfloat(tap3.xyz);

   using NEE = NextEventEstimator;
   NEE neeEstimator = NEE::create();

   nbl::hlsl::spirv::RayQueryKHR query;
   nbl::hlsl::spirv::rayQueryInitializeKHR(query, gTLASes[0], spv::RayFlagsOpaqueKHRMask, 0xffu, rayOrigin, asfloat(tap0.w), rayDir, tMax);
   while (nbl::hlsl::spirv::rayQueryProceedKHR(query))
   {
   }
   const uint32_t committed = 1u;
   if (nbl::hlsl::spirv::rayQueryGetIntersectionTypeKHR(query, committed) == 0u) // CommittedIntersectionNone
   {
      const SEnvSample _sample = NEE::shadeEnvmap(rayDir, otherTechniqueHeuristic);
      color += _sample.color * throughput;
      splatSampleAndMean(coord, sampleIndex, color);
      return;
   }

   const float32_t  hitT   = nbl::hlsl::spirv::rayQueryGetIntersectionTKHR(query, committed);
   const float32_t3 hitPos = rayOrigin + rayDir * hitT;

   float32_t3 geometricNormal;
   {
      const float32_t3 vertices[3] = nbl::hlsl::spirv::rayQueryGetIntersectionTriangleVertexPositionsKHR(query, committed);
      const float32_t3 objNormal   = hlsl::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
      const float32_t3x3 normalMatrix = hlsl::math::linalg::truncate<3, 3, 4, 3>(nbl::hlsl::spirv::rayQueryGetIntersectionWorldToObjectKHR(query, committed));
      geometricNormal                 = hlsl::normalize(hlsl::mul(normalMatrix, objNormal));
   }
   const float32_t3 V     = -rayDir;
   const float32_t  GdotV = hlsl::dot(V, geometricNormal);
   geometricNormal *= sign(GdotV);
   const float32_t3 shadingNormal = geometricNormal;

   const uint32_t emitterIdx = resolveHitEmitterID(
      nbl::hlsl::spirv::rayQueryGetIntersectionInstanceCustomIndexKHR(query, committed),
      nbl::hlsl::spirv::rayQueryGetIntersectionGeometryIndexKHR(query, committed),
      nbl::hlsl::spirv::rayQueryGetIntersectionPrimitiveIndexKHR(query, committed));

   uint32_t neeFlags = NonEmitterCustomIndex;
#if NBL_MIS_MODE == NBL_MIS_MODE_BOTH
   if (emitterIdx < NonEmitterCustomIndex && otherTechniqueHeuristic > NEE::MISWeightThreshold && gScene.init.pEmitterToLeafIdx != 0)
      neeFlags = emitterIdx | WavefrontFlagEmission;
   else
      color += neeEstimator.backwardNEE(emitterIdx, hitPos, otherTechniqueHeuristic, throughput);
#else
   color += neeEstimator.backwardNEE(emitterIdx, hitPos, otherTechniqueHeuristic, throughput);
#endif

#if NBL_MIS_MODE == NBL_MIS_MODE_NEE_ONLY
   bool pathEnds = true;
#else
   bool pathEnds = bounce == uint32_t(pc.sensorDynamics.lastPathDepth);
#endif

   // reconstruct the per-path random stream
   randgen_t randgen;
   randgen.pSampleBuffer       = gScene.init.pSampleSequence;
   randgen.rng                 = scramble_state_t::construct(uint32_t2(tap2.w, tap3.w));
   randgen.sequenceSamplesLog2 = gScene.init.sequenceSamplesLog2;

   if (!pathEnds || NBL_MIS_MODE == NBL_MIS_MODE_NEE_ONLY)
   {
      const uint16_t   sequenceProtoDim = _static_cast<uint16_t>((bounce - 1u) * RandDimTriplesPerDepth + PrimaryRayRandTripletsUsed);
      const float32_t3 randBRDF         = randgen(sequenceProtoDim, sampleIndex);

      const float32_t3 originMagnitude = max(abs(hitPos), abs(rayOrigin));
      const float32_t  offsetMagnitude = hlsl::max(hlsl::max(hlsl::exp2(8.f), originMagnitude.x), hlsl::max(originMagnitude.y, originMagnitude.z)) * hlsl::exp2(-20.f);
      const float32_t3 newRayOrigin    = hitPos + geometricNormal * offsetMagnitude;

      // queue this bounce's NEE work (the sampler pass re-derives randNEE from the snapshot)
      if (gScene.init.pLightTreeLeaves != 0 && gScene.init.pEmitters != 0)
      {
         const uint32_t2 rngState = randgen.rng.stateHolder.state;
         randgen.rng();
         randgen.rng();
         randgen.rng();
         randgen.rng();
         vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrNeeShadow, pathId), uint32_t4(asuint(newRayOrigin), rngState.x), 16u);
         vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrNeeNormal, pathId), uint32_t4(asuint(shadingNormal), rngState.y), 16u);
         vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrNeeThroughput, pathId), uint32_t4(asuint(throughput), asuint(otherTechniqueHeuristic)), 16u);
         neeFlags |= WavefrontFlagNEE;
      }

#if NBL_MIS_MODE != NBL_MIS_MODE_NEE_ONLY
      {
         using namespace nbl::hlsl::bxdf;
         using namespace nbl::hlsl::material_compiler3::backends::default_upt;
         using bxdf_config_t           = BxDFConfig;
         using isotropic_interaction_t = bxdf_config_t::isotropic_interaction_type;
         using light_sample_t          = bxdf_config_t::sample_type;
         using ray_dir_info_t          = light_sample_t::ray_dir_info_type;
         using quotient_weight_type    = sampling::quotient_and_weight<bxdf_config_t::spectral_type, float>;

         ray_dir_info_t Vinfo;
         Vinfo.setDirection(V);
         isotropic_interaction_t interaction = isotropic_interaction_t::create(Vinfo, shadingNormal, throughput);
         using brdf_t                        = reflection::SOrenNayar<bxdf_config_t>;
         brdf_t::SCreationParams cParams;
         cParams.A            = 0.f;
         const brdf_t diffuse = brdf_t::create(cParams);

         typename brdf_t::isocache_type cache;
         const light_sample_t           bxdfSample = diffuse.generate(interaction, randBRDF.xy, cache);
         const quotient_weight_type     qAw        = diffuse.quotientAndWeight(bxdfSample, interaction, cache);
         const float                    forwardWeight = qAw.weight();
         if (forwardWeight < 0.00000001f)
            pathEnds = true;
         else
         {
            throughput = throughput * qAw.quotient() * surfaceAlbedo();

            MaxContributionEstimator contribEstimator = MaxContributionEstimator::create(pc.get16BitData().rrThroughputWeights);
            float32_t                rrXi             = randBRDF.z;
            if (contribEstimator.surviveRussianRoulette(throughput, bounce <= uint32_t(pc.sensorDynamics.lastNoRussianRouletteDepth), rrXi))
            {
               // continuation
               vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrRayOrigin, pathId), uint32_t4(asuint(newRayOrigin), asuint(0.f)), 16u);
               vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrRayDir, pathId), uint32_t4(asuint(bxdfSample.getL().getDirection()), sampleIndex), 16u);
               vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrThroughputRng, pathId), uint32_t4(asuint(throughput), randgen.rng.stateHolder.state.x), 16u);
               vk::RawBufferStore<uint32_t>(wavefrontHeuristicAddress(pc.pNeeRequests, pixelCount, pathId), asuint(1.f / forwardWeight), 4u);

               const uint32_t slot = wavefrontAtomicAdd(pc.pNeeRequests + WavefrontRayCountOffset + 4ull * (1u - parityIn), 1u);
               vk::RawBufferStore<uint32_t>(wavefrontRayQueueAddress(pc.pNeeRequests, pixelCount, 1u - parityIn) + uint64_t(slot) * 4ull, pathId, 4u);
            }
            else
               pathEnds = true;
         }
      }
#endif
   }

   // color + the advanced rng state (the .w must stay current for any queued NEE snapshot consistency)
   vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrColorRng, pathId), uint32_t4(asuint(color), randgen.rng.stateHolder.state.y), 16u);

   const bool needQueue = (neeFlags & (WavefrontFlagEmission | WavefrontFlagNEE)) != 0u;
   if (needQueue)
   {
      if (pathEnds)
         neeFlags |= WavefrontFlagFinalize;
      // emission-only entries skipped the NEE block, so the throughput|heuristic attr still needs writing
      if (!(neeFlags & WavefrontFlagNEE))
         vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrNeeThroughput, pathId), uint32_t4(asuint(throughput), asuint(otherTechniqueHeuristic)), 16u);
      vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrNeeHit, pathId), uint32_t4(asuint(hitPos), neeFlags), 16u);
      const uint32_t slot = wavefrontAtomicAdd(pc.pNeeRequests + WavefrontNeeCountOffset + 4ull * (bounce & 1u), 1u);
      vk::RawBufferStore<uint32_t>(wavefrontNeeQueueAddress(pc.pNeeRequests, pixelCount) + uint64_t(slot) * 4ull, pathId, 4u);
   }
   else if (pathEnds)
      splatSampleAndMean(coord, sampleIndex, color);
}
