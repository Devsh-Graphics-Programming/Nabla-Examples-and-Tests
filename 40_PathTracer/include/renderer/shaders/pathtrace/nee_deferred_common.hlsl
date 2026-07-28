#ifndef _PATHTRACER_40_NEE_DEFERRED_COMMON_INCLUDED_
#define _PATHTRACER_40_NEE_DEFERRED_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

// Batched deferred NEE record layout, SoA: one 16 B tap array per (slot, tap) across the band's pool of
// (pixel, sample) entries, so any tap's loads coalesce across a warp. AoS would put warp-neighbors
// (1+bounces)*64 B apart.
//
// Header (1 tap):
//   H0: pathColor.rgb (depth-1 emission + env, stays in raygen) | sampleWord
//       sampleWord: bits 0-14 sampleIndex, bits 16-23 bounce slot count, NeeDeferredNoSample = dead.
// Bounce slot d (d >= 1), 3 taps:
//   T0: hitPos.xyz       | emitterIdx (low 24, NonEmitter when none) + HasEmission/HasNEE flag bits
//   T1: shadingNormal.xyz| unused (emission-only slots skip this tap)
//   T2: throughput.rgb   | otherTechniqueHeuristic

namespace nbl
{
namespace this_example
{

NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeDeferredTapSize       = 16u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeDeferredHeaderTaps    = 1u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeDeferredBounceTaps    = 3u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeDeferredNoSample      = 0xFFFFFFFFu;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeDeferredSampleIdxMask = 0x7FFFu;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeDeferredBounceShift   = 16u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeDeferredBounceMask    = 0xFFu;
// bounce slot T0.w: low 24 bits emitter index (NonEmitterCustomIndex when no emission deferral)
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeDeferredEmitterMask   = 0x00FFFFFFu;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeDeferredFlagEmission  = 0x40000000u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeDeferredFlagNEE       = 0x80000000u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t NeeDeferredWorkgroupSize = 64u;

#ifdef __HLSL_VERSION
struct SNeeDeferredHeader
{
	float32_t3 pathColor;
	uint32_t   sampleWord;
};

struct SNeeDeferredBounce
{
	float32_t3 hitPos;
	uint32_t   flags; // emitterIdx low 24 + NeeDeferredFlag* bits
	float32_t3 shadingNormal;
	float32_t3 throughput;
	float32_t  otherTechniqueHeuristic;
};

// Typed access to the band's record pool; poolIdx = s * bandPixels + bandLocalPixel, poolCount =
// bandPixels * S. Bounce slots are 1-based.
struct NeeDeferredRecords
{
	static NeeDeferredRecords create(const uint64_t base, const uint32_t poolIdx, const uint32_t poolCount)
	{
		NeeDeferredRecords records;
		records.base      = base;
		records.poolIdx   = poolIdx;
		records.poolCount = poolCount;
		return records;
	}

	uint64_t __tapAddress(const uint32_t slotIdx, const uint32_t tapIdx)
	{
		const uint32_t linearTap = slotIdx == 0u ? tapIdx : (NeeDeferredHeaderTaps + (slotIdx - 1u) * NeeDeferredBounceTaps + tapIdx);
		return base + (uint64_t(linearTap) * poolCount + poolIdx) * NeeDeferredTapSize;
	}

	void storeHeader(const float32_t3 pathColor, const uint32_t sampleIndex, const uint32_t bounceCount)
	{
		vk::RawBufferStore<uint32_t4>(__tapAddress(0u, 0u), uint32_t4(asuint(pathColor), sampleIndex | (bounceCount << NeeDeferredBounceShift)), NeeDeferredTapSize);
	}

	void markHeaderDead() { vk::RawBufferStore<uint32_t>(__tapAddress(0u, 0u) + 12ull, NeeDeferredNoSample, 4u); }

	SNeeDeferredHeader loadHeader()
	{
		const uint32_t4    tap = vk::RawBufferLoad<uint32_t4>(__tapAddress(0u, 0u), NeeDeferredTapSize);
		SNeeDeferredHeader header;
		header.pathColor  = asfloat(tap.xyz);
		header.sampleWord = tap.w;
		return header;
	}

	void storeBounceNEE(const uint32_t d, const float32_t3 hitPos, const uint32_t flags, const float32_t3 shadingNormal, const float32_t3 throughput, const float32_t otherTechniqueHeuristic)
	{
		vk::RawBufferStore<uint32_t4>(__tapAddress(d, 0u), uint32_t4(asuint(hitPos), flags), NeeDeferredTapSize);
		vk::RawBufferStore<uint32_t4>(__tapAddress(d, 1u), uint32_t4(asuint(shadingNormal), 0u), NeeDeferredTapSize);
		vk::RawBufferStore<uint32_t4>(__tapAddress(d, 2u), uint32_t4(asuint(throughput), asuint(otherTechniqueHeuristic)), NeeDeferredTapSize);
	}

	// emission needs no shading normal, T1 stays unwritten
	void storeBounceEmission(const uint32_t d, const float32_t3 hitPos, const uint32_t flags, const float32_t3 throughput, const float32_t otherTechniqueHeuristic)
	{
		vk::RawBufferStore<uint32_t4>(__tapAddress(d, 0u), uint32_t4(asuint(hitPos), flags), NeeDeferredTapSize);
		vk::RawBufferStore<uint32_t4>(__tapAddress(d, 2u), uint32_t4(asuint(throughput), asuint(otherTechniqueHeuristic)), NeeDeferredTapSize);
	}

	// all taps loaded up front, independent of the flags, so the loads overlap instead of chaining
	SNeeDeferredBounce loadBounce(const uint32_t d)
	{
		const uint32_t4    tap0 = vk::RawBufferLoad<uint32_t4>(__tapAddress(d, 0u), NeeDeferredTapSize);
		const uint32_t4    tap1 = vk::RawBufferLoad<uint32_t4>(__tapAddress(d, 1u), NeeDeferredTapSize);
		const uint32_t4    tap2 = vk::RawBufferLoad<uint32_t4>(__tapAddress(d, 2u), NeeDeferredTapSize);
		SNeeDeferredBounce bounce;
		bounce.hitPos                  = asfloat(tap0.xyz);
		bounce.flags                   = tap0.w;
		bounce.shadingNormal           = asfloat(tap1.xyz);
		bounce.throughput              = asfloat(tap2.xyz);
		bounce.otherTechniqueHeuristic = asfloat(tap2.w);
		return bounce;
	}

	uint64_t base;
	uint32_t poolIdx;
	uint32_t poolCount;
};
#endif

}
}
#endif
