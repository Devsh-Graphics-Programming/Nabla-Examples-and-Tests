#include "common.hlsl"
#include "renderer/shaders/bda_accessors.hlsl"
#include "next_event_estimator.hlsl"
#include "emitter_resolve.hlsl"
#include "accumulation.hlsl"
#include "renderer/shaders/pathtrace/nee_deferred_common.hlsl"

// TODO: move this to material_compiler3
// There's actually a huge problem with doing any throughput or accumulation modification in AnyHit shaders, they run out of order (BVH order) and a hit behind your eventual closest hit can invoke the anyhit stage.
//
// Most examples which multiply alpha in anyhit are super misleading, because:
// - for shadow / anyhit rays you either eventually hit an opaque (leading to a mul/replacement of transparency by 0) or you hit all opaques along the ray
// - for NEE rays you often have a finite tMax and this stops you accumulating translucency behind the emitter
// - multiplicative operations are order independent, so accumulating the visibility function can happen out of order (basis of many OIT techniques) as long as you know tMax of the closest hit
// - stochastic transparency cancels out the alpha weighting on the throughput, so there's no multiplication to perform, the throughput stays constant no matter what you do.
//   Which means it doesn't matter if you perform the test for occluded transparent geometries, you will never know, the alpha on the opaque also cancels out (shouldn't use premultiplied to shade).
//
// However the minute you want to do stochastic RGB translucency the pdf no longer cancels out the RGB weight coefficients. While the application of `opacity/luma(opacity)` from a hit accepted as the closest,
// can be delayed until the closest hit if found, you'll start accumulating the wrong visibility from all the ignored hits. You literally have to use stochastic monochrome transparency.
//
// Furthermore the minute you wish to add emission to the accumulation in the payload you run into Order Dependent Transparency because it requires a blend over operator.
//
// The solutions are then as follows:
// 1. Only use Anyhit to employ stochastic transparency when the translucency weight is monochrome
// 2. Re-trace rays, find closest hit as with (1), then launch anyhit rays with known tMax - this only gets you correct RGB translucency
// 3. Use OIT techniques (A-Buffer, MLAB, WBOIT) to estimate the visibility function but without re-tracing need a robust technique which can handle "opaque transparents"
//    RGB translucency can be accumulated without sorting an A-Buffer in a O(1) pass over all intersections, also self-balancing tree and MLAB can throw out entries beyond current tMin.
//    Note that within a TLAS instances are likely to be traversed approximately in-order, and within a BLAS the primitives are too (see CWBVH8 paper with children visit order depending on ray direction signs).
//    Therefore a two tier linked list + insertion sort are a viable alternative to a self-balancing tree. To allow for emittance to be contributed by anyhit stage, it would need to be deferred to be performant,
//    the hit attributes would need to be stored alongside the translucency, so at least instance ID (possibly material ID or SBT offset), primitive ID, and the barycentrics.
// 4. Decompose the Complex Mixture Material into a Scalar Delta Transmission plus the rest of the BxDF. The motivation is simple, for monochrome materials we have
//         DeltaTransmission*(1-alpha) + alpha*(Rest of BxDF Nodes with their Weights)
//    Where the thing getting factored is a blackbox sum of contributors, but we can reformulate any BxDF as
//         DeltaTransmission*Factor + (Rest of BxDF Nodes with their Weights)
//    Then we can simply break down the transmissive part into a monochrome part and a coloured residual, if we're unwilling to get into negative weights only option is `Transparency = min_element(Factor[0],...)`
//         DeltaTransmission * Transparency + (DeltaTransmission * (Factor-Transparency) + Rest of BxDF Nodes with their Weights)
//    We can still use stochastic transparency! Its just that whenever we accept a hit, we need pass `transparency` at the point of acceptance to the closest hit shader as to compute this
//         (DeltaTransmission * (Factor-Transparency) + Rest of BxDF Nodes with their Weights)/(1-Transparency)
//    Since Transparency can be just an approximation of the `Factor` in a monochrome form (luma) or its minimum, already computed or fetched data could be passed in payload for accepted hit
//
// MOST IMPORTANT THING: AFTER ANYHIT ACCEPTS, ANOTHER MAY ACCEPT THATS CLOSER!
// This is very important to keep in mind when we do our Solid Angle Sampling.
//
// Anyhit needs to pass the transparency probability to any closest hit it accepts and which then becomes the final anyhit
struct[raypayload] SAnyHitRetval
{
   // before sending the ray by the caller
   inline void init(const float32_t _xi, float32_t tMax = hlsl::numeric_limits<float32_t>::max)
   {
      xi   = _xi;
      rayT = tMax;
   }
   // call in AnyHit instead of AcceptHit
   inline void acceptHit(const float16_t _transparency)
   {
      // need to read the spec if an anyhit is possible that the last anyhit to run and accept a hit candidate for a ray is not the last one to
      if (rayT > spirv::RayTmaxKHR)
      {
         rayT         = spirv::RayTmaxKHR;
         transparency = _transparency;
      }
      // Note that `spirv::terminateRayKHR` is NOT the correct instruction to call (it terminates ray prematurely without considering anything else)
   }
   //

   // opacity russian roulette requires this for Discrete Probability Sampling
   float32_t xi : read(anyhit) : write(caller, anyhit);
   // need to store the t value at which the anyhit was executed, so we know whether the current closest hit comes from a confirmed anyhit
   float32_t rayT : read(caller, anyhit) : write(caller, anyhit);
   // essentially the probability of transmission
   float16_t transparency : read(caller) : write(anyhit);
   // can use additional `float16` to store BxDF mixture weights or other things so they don't need recomputing/re-fetching during shading
};


// Because SER based on Material ID will probably greatly benefit us, the shading needs to happen in Raygen Shader or ClosestHit executed directly by Raygen
// Lets examine what happens in the 3 options of Shading with SER Hit Objects:
// 1. Fused hitObjectTraceReorderExecuteEXT -> shading in Closest Hit
//      Miss and Closest hit still called immediately, Shading happens in both of them, only need payload to store anyhit + random number state (depth and optionally the seed)
//      but `SClosestHitRetval` gets passed to a shading function. Use NO_NULL_MISS_SHADERS definitely, and NO_NULL_CLOSEST_HIT_SHADERS if there's no blackhole materials.
// 2. hitObjectTraceRayEXT && Shading in Closest Hit with hitObjectExecuteShaderEXT
//      Only Anyhit payload needed, separate `SClosestHitRetval` payload is made in raygen and passed to the hitObjectExecuteShaderEXT, miss shader is not used.
//      Can use NO_NULL_CLOSEST_HIT and NO_NULL_MISS_SHADERS and then never invoke an invalid ClosestHit
// 3. hitObjectTraceRayEXT && Shading in Raygen
//      Only Anyhit payload needed, separate `SClosestHitRetval` is made and passed to traceRay, no closest hit shaders at all.
//      Should use NO_NULL_CLOSEST_HIT and NO_NULL_MISS_SHADERS
struct SClosestHitRetval
{
   static inline SClosestHitRetval create(NBL_REF_ARG(spirv::HitObjectEXT) hitObject)
   {
      SClosestHitRetval retval;
      {
         [[vk::ext_storage_class(spv::StorageClassHitObjectAttributeEXT)]] float32_t2 tmp;
         spirv::hitObjectGetAttributesEXT(hitObject, tmp);
         retval.barycentrics = tmp;
      }
      // Which method of barycentric interpolation is more precise? Pick your poison!
#define POSITION_RECON_METHOD 0
#if POSITION_RECON_METHOD != 0
      // compute worldspace hit position
      const float32_t3 vertices[3] = spirv::hitObjectGetIntersectionTriangleVertexPositionsEXT(hitObject);
#if POSITION_RECON_METHOD != 2
      // This way at least we stay within the triangle, and compiler can do CSE with the geometric normal calculation
      const float32_t3 modelSpacePos = vertices[0] + (vertices[1] - vertices[0]) * retval.barycentrics[0] + (vertices[2] - vertices[0]) * retval.barycentrics[1];
#else
      // This way we get less catastrophic cancellation by adding and computing the edges, but can end up outside the triangle
      const float32_t3 modelSpacePos = vertices[0] * (1.f - retval.barycentrics[0] - retval.barycentrics[1]) + vertices[1] * retval.barycentrics[0] + vertices[2] * retval.barycentrics[1];
#endif
      retval.hitPos = math::linalg::promoted_mul(hlsl::transpose(spirv::hitObjectGetObjectToWorldEXT(hitObject)), modelSpacePos);
#else
      // the way that raytracers have done this before SPV_KHR_ray_tracing_position_fetch
      retval.hitPos = spirv::hitObjectGetWorldRayOriginEXT(hitObject) + spirv::hitObjectGetWorldRayDirectionEXT(hitObject) * spirv::hitObjectGetRayTMaxEXT(hitObject);
#endif
#undef POSITION_RECON_METHOD
      retval.instancedGeometryID = spirv::hitObjectGetInstanceCustomIndexEXT(hitObject) + spirv::hitObjectGetGeometryIndexEXT(hitObject);
      retval.primitiveID         = spirv::hitObjectGetPrimitiveIndexEXT(hitObject);
      retval.geometricNormal     = reconstructGeometricNormal(hitObject);
      return retval;
   }

   float32_t3 hitPos;
   // to interpolate our vertex attributes
   float32_t2 barycentrics;
   // to get our material and geometry data back
   uint32_t instancedGeometryID;
   // to get particular Triangle's indices
   uint32_t primitiveID;
   //
   float32_t3 geometricNormal;
};

// TODO: move to shared C++ and HLSL header
enum E_SBT_OFFSETS : uint16_t
{
   ESBTO_PATH,
   ESBTO_NEE
};

[[vk::push_constant]] SBeautyPushConstants pc;

// Diagnostic-only NEE-proposal probe takeover
#include "nee_proposal_probe.hlsl"

[shader("raygeneration")] 
void raygen()
{
#if NBL_NEE_DEFERRED
   // banded launch: LaunchIdKHR/LaunchSizeKHR are band-local and key the slot addressing
   const uint16_t3 bandLocalID = uint16_t3(spirv::LaunchIdKHR);
   const uint16_t3 launchID    = bandLocalID + uint16_t3(0, uint16_t(pc.tileOffsetY), 0);
#else
   const uint16_t3 launchID = uint16_t3(spirv::LaunchIdKHR);
#endif
   const SBeautyPushConstants::S16BitData unpacked16BitPC = pc.get16BitData();

   // Take n samples per frame
   // TODO: establish min/max - adaptive sampling
   SPixelSamplingInfo samplingInfo = advanceSampleCount(launchID, unpacked16BitPC.maxSppPerDispatch, uint16_t(pc.sensorDynamics.keepAccumulating), pc.sensorDynamics.maxSPP);
   // took max samples
   const uint32_t endSample        = samplingInfo.newSampleCount;
   const uint32_t samplesThisFrame = endSample - samplingInfo.firstSample;
   gAccumulationCoord              = launchID;
#if NBL_NEE_DEFERRED
   const uint32_t bandPixels    = spirv::LaunchSizeKHR.x * spirv::LaunchSizeKHR.y;
   const uint32_t bandPixelIdx  = uint32_t(bandLocalID.y) * spirv::LaunchSizeKHR.x + bandLocalID.x;
   const uint32_t neePoolCount  = bandPixels * uint32_t(unpacked16BitPC.maxSppPerDispatch);
   // records persist across dispatches, mark untaken slots' headers dead before any early return
   NBL_HLSL_LOOP
   for (uint32_t s = samplesThisFrame; s < uint32_t(unpacked16BitPC.maxSppPerDispatch); s++)
      nbl::this_example::NeeDeferredRecords::create(pc.pNeeRequests, s * bandPixels + bandPixelIdx, neePoolCount).markHeaderDead();
#endif
   if (samplesThisFrame == 0)
      return;

#if NBL_NEE_PROPOSAL_PROBE
   // Diagnostic-only takeover (body in nee_proposal_probe.hlsl): visualize the K NEE candidate
   // lights for the center pixel. Writes gAlbedo, then we return.
   runNeeProposalProbe(launchID, samplingInfo);
   return;
#endif

   // TODO: possible SER point if doing variable spp
   //spirv::reorderThreadWithHintEXT<uint32_t>(hlsl::min<uint32_t>(samplesThisFrame,1),1);

   // weight for non RWMC contribution
   const float16_t newSamplesOverTotal = _static_cast<float16_t>(_static_cast<float32_t>(samplesThisFrame) * samplingInfo.rcpNewSampleCount);
   const float16_t rcpSamplesThisFrame = float16_t(1) / _static_cast<float16_t>(samplesThisFrame);

   float16_t              transparency = 0.f;
   SArbitraryOutputValues aovs;
   aovs.clear();

   // some weird DXC and SPIR-V Tools Bug, lets try to move stuff out to temporaries and only use those
   decltype(samplingInfo.randgen) randgen          = samplingInfo.randgen;
   const bool                     keepAccumulating = samplingInfo.firstSample;
   // Held live across the path-tracing loop; summed per sample, written to gBeauty as an fp32 mean
   // after the loop (alongside the per-sample RWMC cascade splat).
   float32_t3 referenceFrameSum = float32_t3(0, 0, 0);
   NBL_HLSL_LOOP
   for (uint32_t sampleIndex = samplingInfo.firstSample; sampleIndex != endSample;)
   {
      // For RWMC to work, every sample must be splatted individually
      spectral_t color;
#if NBL_NEE_DEFERRED
      // highest bounce slot written this sample (0 = none), the header tells the fused pass how many to walk
      uint32_t bounceSlotsWritten = 0u;
      nbl::this_example::NeeDeferredRecords neeRecords =
         nbl::this_example::NeeDeferredRecords::create(pc.pNeeRequests, (sampleIndex - samplingInfo.firstSample) * bandPixels + bandPixelIdx, neePoolCount);
#endif

      using namespace nbl::hlsl::bxdf;
      using namespace nbl::hlsl::material_compiler3::backends::default_upt;
      using bxdf_config_t           = BxDFConfig;
      using isotropic_interaction_t = bxdf_config_t::isotropic_interaction_type;
      using light_sample_t          = bxdf_config_t::sample_type;
      using spectral_type           = bxdf_config_t::spectral_type;
      using ray_dir_info_t          = light_sample_t::ray_dir_info_type;
      using quotient_weight_type    = sampling::quotient_and_weight<spectral_type, float>;
      using value_weight_type       = sampling::value_and_weight<spectral_type, float>;
      // a little bit of persistent state
      spirv::HitObjectEXT hitObject;
      {
         // fetch random variable from memory
         const float32_t3 randVec = randgen(0u, sampleIndex);
         // TODO: motion blur and lens DOF triplet

         // get our NDC coordinates and ray
#if NBL_NEE_DEFERRED
         // LaunchSizeKHR is the band size under tiling; the NDC mapping needs the full render size.
         const float32_t2 pixelSizeNDC = promote<float32_t2>(2.f) / float32_t2(uint32_t2(gSensor.renderSize));
#else
         const float32_t2 pixelSizeNDC = promote<float32_t2>(2.f) / float32_t2(spirv::LaunchSizeKHR.xy);
#endif
         const float32_t2  NDC          = float32_t2(launchID.xy) * pixelSizeNDC - promote<float32_t2>(1.f);
         const SPrimaryRay primary      = genPrimaryRay(pc.sensorDynamics, pixelSizeNDC, NDC, float16_t2(randVec.xy), spirv::LaunchSizeKHR.z);
         const SRay        ray          = primary.ray;

         // TODO: possible SER point, sorting by ray direction
         //spirv::reorderThreadWithHintEXT<uint32_t>(,);

         [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval payload;
         const float                                                             tMax = pc.sensorDynamics.tMax;
         payload.init(randVec.z, tMax);
         spirv::hitObjectTraceRayEXT(hitObject, gTLASes[0], spv::RayFlagsMaskNone, 0xff, ESBTO_PATH, 0u, 0u, ray.origin, primary.tMin, ray.direction.getDirection(), tMax, payload);
         // TODO: do something with the payload's reported transparency
      }
      // TODO: Possible SER point
      const bool       primaryMissed = spirv::hitObjectIsMissEXT(hitObject);
      const float32_t3 primaryRayDir = spirv::hitObjectGetWorldRayDirectionEXT(hitObject);

      if (primaryMissed)
      {
         const SEnvSample _sample = nbl::this_example::NextEventEstimator::shadeEnvmap(primaryRayDir, 0.f);
         color                    = float16_t3(_sample.color);
         aovs                     = aovs + _sample.aov * rcpSamplesThisFrame;
         transparency += rcpSamplesThisFrame;
      }
      else // trace further rays
      {
         //
         MaxContributionEstimator contribEstimator           = MaxContributionEstimator::create(unpacked16BitPC.rrThroughputWeights);
         const uint16_t           lastPathDepth              = _static_cast<uint16_t>(pc.sensorDynamics.lastPathDepth);
         const uint16_t           lastNoRussianRouletteDepth = _static_cast<uint16_t>(pc.sensorDynamics.lastNoRussianRouletteDepth);
         //
         color                                                         = spectral_t(0, 0, 0);
         spectral_t                            throughput              = spectral_t(1, 1, 1);
         float32_t                             otherTechniqueHeuristic = 0.f;
         nbl::this_example::NextEventEstimator neeEstimator            = nbl::this_example::NextEventEstimator::create();
         SAOVThroughputs                       aovThroughput;
         aovThroughput.clear(rcpSamplesThisFrame);
         NBL_HLSL_LOOP
         for (uint16_t depth = 1; true; depth++) // ideally peel this loop once
         {
            // TODO: get the material ID and UVs

            // TODO: preserve spread metrics
            ray_dir_info_t V;
            // minus because of transmission
            V.setDirection(-spirv::hitObjectGetWorldRayDirectionEXT(hitObject));

            SClosestHitRetval closestInfo = SClosestHitRetval::create(hitObject);
            const float32_t   GdotV       = hlsl::dot(V.getDirection(), closestInfo.geometricNormal);
            // TODO: only for twosided materials
            closestInfo.geometricNormal *= sign(GdotV);

            float32_t3 shadingNormal = closestInfo.geometricNormal;

            // TODO: possible SER point based on NEE status, and material flags

            // TODO: get AoVs from material and emission
            SAOVThroughputs nextThroughput;
            nextThroughput.clear(0.f);
            SArbitraryOutputValues aovContrib;
            aovContrib.albedo = float16_t3(1, 1, 1);
            aovContrib.normal = float16_t3(shadingNormal);
            // obtain full next
            nextThroughput = aovThroughput * nextThroughput;
            // already premultiplied by next throughput complement
            aovs          = aovs + aovContrib * (aovThroughput - nextThroughput);
            aovThroughput = nextThroughput;

            // Emission shading: resolve the hit's emitter ID from the per-geometry aux map
            // (instanceCustomIndex is a base, not the emitter ID); NonEmitterCustomIndex if non-emissive.
#if NBL_NEE_DEFERRED
            uint32_t slotEmitterFlags = nbl::this_example::NonEmitterCustomIndex;
#endif
            {
               const uint32_t emitterIdx = resolveHitEmitterID(spirv::hitObjectGetInstanceCustomIndexEXT(hitObject), spirv::hitObjectGetGeometryIndexEXT(hitObject), closestInfo.primitiveID);
#if NBL_NEE_DEFERRED && NBL_MIS_MODE == NBL_MIS_MODE_BOTH
               // deweight-needing hits go to the fused pass; backwardNEE here compiles without deweight
               if (emitterIdx < nbl::this_example::NonEmitterCustomIndex && otherTechniqueHeuristic > nbl::this_example::NextEventEstimator::MISWeightThreshold && gScene.init.pEmitterToLeafIdx != 0)
                  slotEmitterFlags = emitterIdx | nbl::this_example::NeeDeferredFlagEmission;
               else
                  color += neeEstimator.backwardNEE(emitterIdx, closestInfo.hitPos, otherTechniqueHeuristic, throughput);
#else
               color += neeEstimator.backwardNEE(emitterIdx, closestInfo.hitPos, otherTechniqueHeuristic, throughput);
#endif
            }

            // TODO: SER point: Russian roulette / termination > Material Flags/Lengths > Material ID

            // to keep path depths equal for NEE and BxDF sampling, we can't continue and do NEE
            if (depth == lastPathDepth)
            {
#if NBL_NEE_DEFERRED
               // emission-only slot, no NEE happens at lastPathDepth
               if (slotEmitterFlags != nbl::this_example::NonEmitterCustomIndex)
               {
                  neeRecords.storeBounceEmission(depth, closestInfo.hitPos, slotEmitterFlags, float32_t3(throughput), otherTechniqueHeuristic);
                  bounceSlotsWritten = depth;
               }
#endif
               break;
            }

            // TODO: embed a bit in the material stream whether:
            // 1. anisotropic interaction is needed
            // 2. whether luma contribution hint is needed
            isotropic_interaction_t interaction = isotropic_interaction_t::create(V, shadingNormal, throughput);

            using brdf_t = reflection::SOrenNayar<bxdf_config_t>;
            brdf_t::SCreationParams cParams;
            cParams.A            = 0.f;
            const brdf_t diffuse = brdf_t::create(cParams);
            // the OrenNayar eval is cos/pi WITHOUT albedo; apply it to BOTH the NEE direct
            // contribution and the BSDF-continuation throughput
            const float32_t3 albedo = surfaceAlbedo();

            // get next random number, compensate for the triplets ray generation used
            const uint16_t sequenceProtoDim = (depth - uint16_t(1)) * RandDimTriplesPerDepth + PrimaryRayRandTripletsUsed;
            float32_t3     randBRDF         = randgen(sequenceProtoDim, sampleIndex);

            // TODO: start at 0 or numeric_limits::min?
            const float32_t tMin = 0.f;
            // should the offset be the same for NEE and Path Continuation?
            const float32_t3 originMagnitude = max(abs(closestInfo.hitPos), abs(spirv::hitObjectGetWorldRayOriginEXT(hitObject)));
            // TODO: should probably also take `tMax` of found hit into account
            const float      offsetMagnitude = hlsl::max(hlsl::max(hlsl::exp2(8.f), originMagnitude.x), hlsl::max(originMagnitude.y, originMagnitude.z)) * hlsl::exp2(-20.f);
            const float32_t3 newRayOrigin    = closestInfo.hitPos + closestInfo.geometricNormal * offsetMagnitude;

            // perform NEE
            const float32_t neeProb = 1.f;
#if NBL_MIS_MODE != NBL_MIS_MODE_BXDF_ONLY
            if (gScene.init.pLightTreeLeaves != 0 && gScene.init.pEmitters != 0)
            {
#if NBL_NEE_DEFERRED
               // The fused pass replays the whole random stream from gScrambleKey, valid at 1 spp per
               // dispatch, so no snapshot is stored. The 4 dummy advances keep this kernel's later
               // fetches aligned with the inline variant.
               randgen.rng();
               randgen.rng();
               randgen.rng();
               randgen.rng();
               neeRecords.storeBounceNEE(depth, closestInfo.hitPos, slotEmitterFlags | nbl::this_example::NeeDeferredFlagNEE, shadingNormal, float32_t3(throughput), otherTechniqueHeuristic);
               bounceSlotsWritten = depth;
#else
               const float32_t3 randNEE  = randgen(sequenceProtoDim + uint16_t(1), sampleIndex);
               const float32_t3 randNEE2 = randgen(sequenceProtoDim + uint16_t(2), sampleIndex);

               // forwardNEE owns both shadow rays and only returns a valid sample once the emitter is visible.
               const nbl::this_example::NextEventEstimator::SForwardSample nee = neeEstimator.forwardNEE(closestInfo.hitPos, newRayOrigin, shadingNormal, interaction, diffuse, throughput, randNEE, randNEE2);
               if (nee.valid)
                  color += nee.contribution * albedo;
#endif // NBL_NEE_DEFERRED
            }
#endif // NBL_MIS_MODE != NBL_MIS_MODE_BXDF_ONLY

#if NBL_MIS_MODE == NBL_MIS_MODE_NEE_ONLY
            // Direct-only: stop before BSDF sampling so only camera-visible emission + NEE contribute.
            break;
#endif

            // TODO: perform shading
            light_sample_t bxdfSample;
            {
               //
               typename brdf_t::isocache_type cache;
               bxdfSample = diffuse.generate(interaction, randBRDF.xy, cache);
               // Do I need to check `_sample.isValid()` myself before calling `forwardWeight`?
               const quotient_weight_type qAw           = diffuse.quotientAndWeight(bxdfSample, interaction, cache);
               const float                forwardWeight = qAw.weight();
               if (forwardWeight < 0.00000001f)
                  break;

               throughput = throughput * qAw.quotient() * albedo;

               // TODO: include neeProb here
               otherTechniqueHeuristic = 1.f / forwardWeight;
            }
            // Stash shading info so the next bounce's emission-on-hit can compute the
            // NEE pdf this technique would have assigned to the BSDF-sampled direction.
            neeEstimator.recordShadingVertex(closestInfo.hitPos, shadingNormal);

            // to keep path depths equal for NEE and BxDF sampling, we
            if (contribEstimator.surviveRussianRoulette(throughput, depth <= lastNoRussianRouletteDepth, randBRDF.z))
            {
               // continue the path
               {
                  const float32_t3                                                        L = bxdfSample.getL().getDirection();
                  [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]] SAnyHitRetval contPayload;
                  contPayload.init(randBRDF.z);
                  spirv::hitObjectTraceRayEXT(hitObject, gTLASes[0], spv::RayFlagsMaskNone, 0xff, ESBTO_PATH, 0u, 0u, newRayOrigin, tMin, L, hlsl::numeric_limits<float32_t>::max, contPayload);
                  const bool       bounceMissed = spirv::hitObjectIsMissEXT(hitObject);
                  const float32_t3 bounceRayDir = spirv::hitObjectGetWorldRayDirectionEXT(hitObject);
                  // TODO: do something with the payload's reported transparency
                  if (bounceMissed)
                  {
                     const SEnvSample _sample = neeEstimator.shadeEnvmap(bounceRayDir, otherTechniqueHeuristic);
                     color += float16_t3(_sample.color * throughput);
                     aovs = aovs + _sample.aov * aovThroughput;
                     transparency += aovThroughput.transparency;
                     break;
                  }
               }
            }
         }
      }
#if NBL_NEE_DEFERRED
      neeRecords.storeHeader(float32_t3(color), sampleIndex, bounceSlotsWritten);
      sampleIndex++;
#else
      // first sample clears the RWMC; can't use pc.keepAccumulating because of variable sampling
      referenceFrameSum += float32_t3(color);
      const bool doClear = (sampleIndex++) == 0;
      // don't precompute `rwmc::CascadeAccumulator<CCascades>::create(gSensor.splatting)` and keep it
      // as live state, it will spill anyway
      rwmc::CascadeAccumulator<CCascades> colorAcc = rwmc::CascadeAccumulator<CCascades>::create(gSensor.splatting, doClear);
      colorAcc.addSample(_static_cast<uint16_t>(sampleIndex), accum_t(color));
#endif
   }
#if !NBL_NEE_DEFERRED
   // Plain fp32 running mean across dispatches: gBeauty holds the mean over all
   // samples this pixel has accumulated. firstSample==0 means a fresh start.
   {
      float32_t3 mean = (samplingInfo.firstSample != 0u) ? gBeauty[launchID].rgb : float32_t3(0, 0, 0);
      mean += (referenceFrameSum - mean * float32_t(samplesThisFrame)) * samplingInfo.rcpNewSampleCount;
      gBeauty[launchID] = float32_t4(mean, 1.0);
   }
#endif
   // albedo
   Accumulator<ImageAccessor_gAlbedo> albedoAcc;
   albedoAcc.accumulate(launchID.xy, launchID.z, aovs.albedo, newSamplesOverTotal, keepAccumulating);
   // normal
   Accumulator<ImageAccessor_gNormal> normalAcc;
   normalAcc.accumulate(launchID.xy, launchID.z, correctSNorm10WhenStoringToUnorm(hlsl::normalize(aovs.normal)), newSamplesOverTotal, keepAccumulating);
   // TODO: motion
   // mask (TODO: do a separate pipeline for this with removed transparency calculations)
   if (gSensor.hideEnvironment)
   {
      Accumulator<ImageAccessor_gMask> maskAcc;
      vector<float16_t, 1>             opacity = float16_t(1) - transparency;
      maskAcc.accumulate(launchID.xy, launchID.z, opacity, newSamplesOverTotal, keepAccumulating);
   }
}

// TODO: Anyhit transparency will come from the Material Compiler
