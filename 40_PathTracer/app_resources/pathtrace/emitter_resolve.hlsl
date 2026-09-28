#ifndef _PATHTRACER_40_EMITTER_RESOLVE_INCLUDED_
#define _PATHTRACER_40_EMITTER_RESOLVE_INCLUDED_

#include "common.hlsl"

// Shared by raygen (hit-emission + inline NEE shadow check) and the deferred NEE compute pass
// (ray-query shadow check), so both resolve a hit to the same emitter ID.

uint32_t resolveEmitterID(const uint32_t instanceCustomIndex, const uint32_t geometryIndex)
{
   if (gScene.init.pInstancedGeometryToEmitter == 0)
      return nbl::this_example::NonEmitterCustomIndex;
   return vk::RawBufferLoad<uint32_t>(gScene.init.pInstancedGeometryToEmitter + uint64_t(instanceCustomIndex + geometryIndex) * 4ull);
}

// Per-hit key, the value SEmitterGPU::hitKey stores for that emitter. OBB: the emitter ID, one per
// (instance, geometry). Triangle baseline: the map stores each geometry's BASE triangle index, so the
// hit's triangle is base + PrimitiveIndex(), in selection order.
uint32_t resolveHitKey(const uint32_t instanceCustomIndex, const uint32_t geometryIndex, const uint32_t primitiveID)
{
   const uint32_t base = resolveEmitterID(instanceCustomIndex, geometryIndex);
#if NBL_NEE_LEAF_MODE != 0
   return (base < nbl::this_example::NonEmitterCustomIndex) ? (base + primitiveID) : base;
#else
   return base;
#endif
}

// Per-hit emitter ID. Triangle baseline goes through pTriToEmitter since leaves were Morton-reordered.
uint32_t resolveHitEmitterID(const uint32_t instanceCustomIndex, const uint32_t geometryIndex, const uint32_t primitiveID)
{
   const uint32_t key = resolveHitKey(instanceCustomIndex, geometryIndex, primitiveID);
#if NBL_NEE_LEAF_MODE != 0
   if (key >= nbl::this_example::NonEmitterCustomIndex || gScene.init.pTriToEmitter == 0)
      return nbl::this_example::NonEmitterCustomIndex;
   return vk::RawBufferLoad<uint32_t>(gScene.init.pTriToEmitter + uint64_t(key) * 4ull);
#else
   return key;
#endif
}

#endif
