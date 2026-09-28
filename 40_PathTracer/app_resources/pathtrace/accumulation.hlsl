#ifndef _PATHTRACER_40_ACCUMULATION_INCLUDED_
#define _PATHTRACER_40_ACCUMULATION_INCLUDED_

#include "nbl/builtin/hlsl/rwmc/CascadeAccumulator.hlsl"

#include "common.hlsl"

// CascadeAccumulator default-constructs its cascades accessor, so this can't be a member. Raygen
// sets it from LaunchIdKHR, the deferred resolve pass from its own thread's pixel.
static uint16_t3 gAccumulationCoord;

// Every sample feeds BOTH outputs, the fp32 running mean in gBeauty and the fp16 RWMC cascade
// splat in gRWMCCascades, so one run yields both with no build-time toggle. The RWMC 16-bit
// per-cascade sample count wraps past 65535 spp, so very-high-spp references must read gBeauty.
struct CCascades
{
   using layer_type        = float16_t3;
   using sample_count_type = uint16_t;
   using weight_t          = float16_t;

   inline uint16_t getLastCascade() { return gSensor.lastCascadeIndex; }

   inline void clear()
   {
      for (uint16_t i = 0u; i <= getLastCascade(); ++i)
         gRWMCCascades[__getCoord(i)] = uint32_t2(0, 0);
   }

   inline void addSampleIntoCascadeEntry(const layer_type _sample, const uint16_t lowerCascadeIndex, const weight_t lowerCascadeLevelWeight, const weight_t higherCascadeLevelWeight, const sample_count_type sampleCount)
   {
      const weight_t reciprocalSampleCount = weight_t(1) / weight_t(sampleCount);
      uint16_t3      coord                 = __getCoord(lowerCascadeIndex);
      __splatToLayer(coord, _sample * lowerCascadeLevelWeight, sampleCount, reciprocalSampleCount);
      if (higherCascadeLevelWeight > weight_t(0))
      {
         coord.z++;
         __splatToLayer(coord, _sample * higherCascadeLevelWeight, sampleCount, reciprocalSampleCount);
      }
   }

   inline uint16_t3 __getCoord(const uint16_t cascadeIx)
   {
      uint16_t3 coord = gAccumulationCoord;
      coord.z         = coord.z * uint16_t(6) + cascadeIx;
      return coord;
   }

   inline void __splatToLayer(const uint16_t3 coord, const layer_type weightedSample, const sample_count_type sampleCount, const weight_t reciprocalSampleCount)
   {
      uint16_t4 data = uint16_t4(0, 0, 0, 0);
      if (sampleCount > 1)
         data = bit_cast<uint16_t4>(gRWMCCascades[coord]);
      layer_type              value          = bit_cast<layer_type>(data.xyz);
      const sample_count_type oldSampleCount = data.w;
#if NBL_RWMC_FP32_REWEIGHT
      float32_t3 v = float32_t3(value);
      v += (float32_t3(weightedSample) - v * float32_t(sampleCount - oldSampleCount)) / float32_t(sampleCount);
      value = layer_type(v);
#else
      value += (weightedSample - value * weight_t(sampleCount - oldSampleCount)) * reciprocalSampleCount;
#endif
      data                 = uint16_t4(bit_cast<uint16_t3>(value), sampleCount);
      gRWMCCascades[coord] = bit_cast<uint32_t2>(data);
   }
};

#endif
