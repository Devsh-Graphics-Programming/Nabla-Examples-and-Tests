// Per-bounce indirect wavefront: runs once between trace_d and nee_d. Both of trace_d's append
// counters are final, so one dispatch prepares bounce d's NEE args AND bounce d+1's trace args, and
// zeroes the counters d+1 appends to (nee_d still reads its own parity's NEE counter untouched).

#include "common.hlsl"
#include "renderer/shaders/pathtrace/wavefront_common.hlsl"

[[vk::push_constant]] nbl::this_example::SBeautyPushConstants pc;

using namespace nbl::this_example;

[shader("compute")]
[numthreads(1, 1, 1)]
void waveFixupBounce(uint32_t3 dispatchId: SV_DispatchThreadID)
{
   const uint32_t bounce    = uint32_t(pc.wavefrontBounce);
   const uint32_t parityOut = bounce & 1u; // trace_d appended continuations + NEE entries here

   const uint32_t neeCount = vk::RawBufferLoad<uint32_t>(pc.pNeeRequests + WavefrontNeeCountOffset + 4ull * parityOut, 4u);
   vk::RawBufferStore<uint32_t4>(pc.pNeeRequests + WavefrontNeeArgsOffset, uint32_t4((neeCount + WavefrontWorkgroupSize - 1u) / WavefrontWorkgroupSize, 1u, 1u, 0u), 4u);

   const uint32_t rayCount = vk::RawBufferLoad<uint32_t>(pc.pNeeRequests + WavefrontRayCountOffset + 4ull * parityOut, 4u);
   vk::RawBufferStore<uint32_t4>(pc.pNeeRequests + WavefrontRayArgsOffset, uint32_t4((rayCount + WavefrontWorkgroupSize - 1u) / WavefrontWorkgroupSize, 1u, 1u, 0u), 4u);

   // zero what trace_{d+1} appends to
   vk::RawBufferStore<uint32_t>(pc.pNeeRequests + WavefrontRayCountOffset + 4ull * (1u - parityOut), 0u, 4u);
   vk::RawBufferStore<uint32_t>(pc.pNeeRequests + WavefrontNeeCountOffset + 4ull * (1u - parityOut), 0u, 4u);
}
