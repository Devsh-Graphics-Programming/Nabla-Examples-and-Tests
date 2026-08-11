// Per-bounce indirect wavefront: spawns one 1-spp path per pixel per wave and appends it to ray queue 0.

#include "common.hlsl"
#include "renderer/shaders/pathtrace/wavefront_common.hlsl"

[[vk::push_constant]] nbl::this_example::SBeautyPushConstants pc;

using namespace nbl::this_example;

[shader("compute")]
[numthreads(WavefrontWorkgroupSize, 1, 1)]
void waveInit(uint32_t3 dispatchId: SV_DispatchThreadID)
{
   const uint32_t2 renderSize = uint32_t2(gSensor.renderSize);
   if (dispatchId.x >= renderSize.x * renderSize.y)
      return;
   const uint32_t  pathId = dispatchId.x;
   const uint16_t3 coord  = uint16_t3(_static_cast<uint16_t>(pathId % renderSize.x), _static_cast<uint16_t>(pathId / renderSize.x), uint16_t(0));

   // 1 spp per wave; waves past the first must not clear the accumulation they continue.
   const uint16_t     dontClear    = (pc.wavefrontWave == 0u) ? uint16_t(pc.sensorDynamics.keepAccumulating) : uint16_t(1);
   SPixelSamplingInfo samplingInfo = advanceSampleCount(coord, uint16_t(1), dontClear, pc.sensorDynamics.maxSPP);
   if (samplingInfo.newSampleCount == samplingInfo.firstSample)
      return;
   const uint32_t sampleIndex = samplingInfo.firstSample;

   decltype(samplingInfo.randgen) randgen = samplingInfo.randgen;
   const float32_t3               randVec = randgen(0u, sampleIndex);

   const float32_t2  pixelSizeNDC = promote<float32_t2>(2.f) / float32_t2(renderSize);
   const float32_t2  NDC          = float32_t2(coord.xy) * pixelSizeNDC - promote<float32_t2>(1.f);
   const SPrimaryRay primary      = genPrimaryRay(pc.sensorDynamics, pixelSizeNDC, NDC, float16_t2(randVec.xy), 1u);

   const uint32_t pixelCount = renderSize.x * renderSize.y;
   vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrRayOrigin, pathId), uint32_t4(asuint(primary.ray.origin), asuint(primary.tMin)), 16u);
   vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrRayDir, pathId), uint32_t4(asuint(primary.ray.direction.getDirection()), sampleIndex | WavefrontPrimaryRayFlag), 16u);
   vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrThroughputRng, pathId), uint32_t4(asuint(float32_t3(1, 1, 1)), randgen.rng.stateHolder.state.x), 16u);
   vk::RawBufferStore<uint32_t4>(wavefrontAttrAddress(pc.pNeeRequests, pixelCount, WfAttrColorRng, pathId), uint32_t4(asuint(float32_t3(0, 0, 0)), randgen.rng.stateHolder.state.y), 16u);
   vk::RawBufferStore<uint32_t>(wavefrontHeuristicAddress(pc.pNeeRequests, pixelCount, pathId), asuint(0.f), 4u);

   const uint32_t slot = wavefrontAtomicAdd(pc.pNeeRequests + WavefrontRayCountOffset, 1u);
   vk::RawBufferStore<uint32_t>(wavefrontRayQueueAddress(pc.pNeeRequests, pixelCount, 0u) + uint64_t(slot) * 4ull, pathId, 4u);
}
