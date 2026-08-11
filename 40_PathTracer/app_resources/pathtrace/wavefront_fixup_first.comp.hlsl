// Per-bounce indirect wavefront: turns waveInit's append counter into the first trace dispatch's args
// (all other counters were zeroed by the wave's fillBuffer).

#include "common.hlsl"
#include "renderer/shaders/pathtrace/wavefront_common.hlsl"

[[vk::push_constant]] nbl::this_example::SBeautyPushConstants pc;

using namespace nbl::this_example;

[shader("compute")]
[numthreads(1, 1, 1)]
void waveFixupFirst(uint32_t3 dispatchId: SV_DispatchThreadID)
{
   const uint32_t rayCount = vk::RawBufferLoad<uint32_t>(pc.pNeeRequests + WavefrontRayCountOffset, 4u);
   vk::RawBufferStore<uint32_t4>(pc.pNeeRequests + WavefrontRayArgsOffset, uint32_t4((rayCount + WavefrontWorkgroupSize - 1u) / WavefrontWorkgroupSize, 1u, 1u, 0u), 4u);
}
