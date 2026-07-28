#ifndef _PATHTRACER_40_WAVEFRONT_COMMON_INCLUDED_
#define _PATHTRACER_40_WAVEFRONT_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#ifdef __HLSL_VERSION
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#endif

// Per-bounce indirect wavefront: SoA path pool + path-ID queues + counters/args page in one buffer
// (pc.pNeeRequests). SoA so compaction-scrambled queue order still coalesces per-attribute loads.
namespace nbl
{
namespace this_example
{

// Attribute ownership: trace writes RayOrigin..ColorRng, NeeHit..NeeThroughput, Heuristic;
// nee RMWs ColorRng and writes PrevPos/PrevNormal.
NBL_CONSTEXPR_STATIC_INLINE uint32_t WfAttrRayOrigin     = 0u; // rayOrigin.xyz | tMin
NBL_CONSTEXPR_STATIC_INLINE uint32_t WfAttrRayDir        = 1u; // rayDir.xyz | sampleIndex (bit 31 = primary ray)
NBL_CONSTEXPR_STATIC_INLINE uint32_t WfAttrThroughputRng = 2u; // throughput.rgb | rngState.x
NBL_CONSTEXPR_STATIC_INLINE uint32_t WfAttrColorRng      = 3u; // color.rgb | rngState.y
NBL_CONSTEXPR_STATIC_INLINE uint32_t WfAttrPrevPos       = 4u; // prevShadingHitPos.xyz | prevNeePdf
NBL_CONSTEXPR_STATIC_INLINE uint32_t WfAttrPrevNormal    = 5u; // prevShadingNormal.xyz | prevNeeEmitterID
// valid between trace_d and nee_d
NBL_CONSTEXPR_STATIC_INLINE uint32_t WfAttrNeeHit        = 6u; // neeHitPos.xyz | emitterIdx (low 24) + flag bits
NBL_CONSTEXPR_STATIC_INLINE uint32_t WfAttrNeeShadow     = 7u; // neeShadowOrigin.xyz | nee rngState.x snapshot
NBL_CONSTEXPR_STATIC_INLINE uint32_t WfAttrNeeNormal     = 8u; // neeNormal.xyz | nee rngState.y snapshot
NBL_CONSTEXPR_STATIC_INLINE uint32_t WfAttrNeeThroughput = 9u; // neePreBsdfThroughput.rgb | neeHeuristic
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontVec4AttrCount = 10u;
// + a lone float array: otherTechniqueHeuristic
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontPathSize = WavefrontVec4AttrCount * 16u + 4u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontCountersPageSize = 64u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontWorkgroupSize    = 64u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontPrimaryRayFlag   = 0x80000000u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontEmitterMask      = 0x00FFFFFFu;
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontFlagEmission     = 0x20000000u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontFlagNEE          = 0x40000000u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontFlagFinalize     = 0x80000000u;

// counters ping-pong on bounce parity so the combined fixup after trace_d can zero d+1's append
// targets while nee_d still reads its own counter
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontRayCountOffset = 0u;  // uint[2]
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontNeeCountOffset = 8u;  // uint[2]
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontRayArgsOffset  = 16u; // uint3 VkDispatchIndirectCommand
NBL_CONSTEXPR_STATIC_INLINE uint32_t WavefrontNeeArgsOffset  = 32u; // uint3

// regions: counters page | vec4 attr arrays | heuristic float array | ray queue 0 | ray queue 1 | NEE queue
inline uint64_t wavefrontAttrAddress(const uint64_t base, const uint32_t pixelCount, const uint32_t attrIndex, const uint32_t pathId) { return base + WavefrontCountersPageSize + (uint64_t(attrIndex) * pixelCount + pathId) * 16ull; }
inline uint64_t wavefrontHeuristicAddress(const uint64_t base, const uint32_t pixelCount, const uint32_t pathId) { return base + WavefrontCountersPageSize + uint64_t(pixelCount) * (WavefrontVec4AttrCount * 16ull) + uint64_t(pathId) * 4ull; }
inline uint64_t wavefrontRayQueueAddress(const uint64_t base, const uint32_t pixelCount, const uint32_t parity) { return base + WavefrontCountersPageSize + uint64_t(pixelCount) * (WavefrontPathSize + 4u * parity); }
inline uint64_t wavefrontNeeQueueAddress(const uint64_t base, const uint32_t pixelCount) { return base + WavefrontCountersPageSize + uint64_t(pixelCount) * (WavefrontPathSize + 8u); }
inline uint64_t wavefrontBufferSize(const uint32_t pixelCount) { return WavefrontCountersPageSize + uint64_t(pixelCount) * (WavefrontPathSize + 12u); }

#ifdef __HLSL_VERSION
inline uint32_t wavefrontAtomicAdd(const uint64_t addr, const uint32_t value)
{
	hlsl::bda::__ptr<uint32_t> p = hlsl::bda::__ptr<uint32_t>::create(addr);
	return hlsl::glsl::atomicAdd(p.deref().ptr.value, value);
}
#endif

}
}
#endif
