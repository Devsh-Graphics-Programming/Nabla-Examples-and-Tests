// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_RENDERER_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_RENDERER_H_INCLUDED_


#include "nbl/examples/common/CCachedOwenScrambledSequence.hpp"

#include "renderer/CScene.h"
#include "renderer/CSession.h"

#include "renderer/shaders/pathtrace/push_constants.hlsl"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"


namespace nbl::this_example
{

class CRenderer : public core::IReferenceCounted, public core::InterfaceUnmovable
{
      friend struct SSubmitInfo;
    public:
      //
      constexpr static video::SPhysicalDeviceFeatures RequiredDeviceFeatures()
      {
         video::SPhysicalDeviceFeatures retval = {};
         retval.rayTracingPipeline = true;
         retval.accelerationStructure = true;
         // deferred NEE compute pass traces its shadow rays with ray queries
         retval.rayQuery = true;
         return retval;
      }
      //
      constexpr static video::SPhysicalDeviceFeatures PreferredDeviceFeatures()
      {
         auto retval = RequiredDeviceFeatures();
         retval.accelerationStructureHostCommands = true;
         return retval;
      }
#if 0 // see TODO in main.cpp
      constexpr static video::SPhysicalDeviceLimits RequiredDeviceLimits()
      {
         video::SPhysicalDeviceLimits retval = {};
         retval.shaderStorageImageReadWithoutFormat = true;
         return retval;
      }
#endif
      //
      template<core::StringLiteral ShaderKey>
      static inline core::smart_refctd_ptr<asset::IShader> loadPrecompiledShader(
         asset::IAssetManager* assMan, video::ILogicalDevice* device, system::logger_opt_ptr logger={}
      )
      {
         return loadPrecompiledShader_impl(assMan,builtin::build::get_spirv_key<ShaderKey>(device),logger);
      }

      struct SCachedCreationParams
      {
         inline operator bool() const
         {
            if (!graphicsQueue || !computeQueue || !uploadQueue)
               return false;
            if (!utilities)
               return false;
            if (graphicsQueue->getOriginDevice()!=utilities->getLogicalDevice())
               return false;
            if (computeQueue->getOriginDevice()!=utilities->getLogicalDevice())
               return false;
            if (uploadQueue->getOriginDevice()!=utilities->getLogicalDevice())
               return false;
            return true;
         }

         video::CThreadSafeQueueAdapter* graphicsQueue = nullptr;
         video::CThreadSafeQueueAdapter* computeQueue = nullptr;
         video::CThreadSafeQueueAdapter* uploadQueue = nullptr;
         //
         core::smart_refctd_ptr<video::IUtilities> utilities = nullptr;
         // can be null
         system::logger_opt_smart_ptr logger = nullptr;
      };
      struct SCreationParams : SCachedCreationParams
      {
         asset::IAssetManager* assMan;
         std::string sequenceCachePath;
      };
      static core::smart_refctd_ptr<CRenderer> create(SCreationParams&& _params);

      //
      inline const SCachedCreationParams& getCreationParams() const { return m_creation; }
   
      //
      inline system::logger_opt_ptr getLogger() const {return m_creation.logger.get().get();}

      //
      inline video::ILogicalDevice* getDevice() const {return m_creation.utilities->getLogicalDevice();}
      
      struct SCachedConstructionParams
      {
         constexpr static inline uint8_t FramesInFlight = 3;

         // TODO: Some Constant to Tell us how many dimensions each path vertex consumes
         inline auto getSequenceMaxPathDepth() const {return sequenceHeader.maxDimensions/3;}


         core::smart_refctd_ptr<video::ISemaphore> semaphore;

         // per pipeline UBO for other pipelines
         core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> uboDSLayout;
         // descriptor set for a scene shall contain sampled textures and compiled materials
         core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> sceneDSLayout;
         // descriptor set for sensors
         core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> sensorDSLayout;

         // temporary
         std::array<core::smart_refctd_ptr<asset::IShader>,uint8_t(CSession::RenderMode::Count)> shaders;
         std::array<core::smart_refctd_ptr<asset::IShader>,uint8_t(CSession::BeautyVariant::Count)> beautyVariantShaders;
         std::array<core::smart_refctd_ptr<video::IGPUPipelineLayout>,uint8_t(CSession::RenderMode::Count)> renderingLayouts;
         // Deferred (wavefront) NEE: shared compute pipeline layout (same DS layouts, ESS_COMPUTE push
         // constants) + one fused NEE+resolve pipeline per (light sampler, alias/tree, MIS mode). Flat
         // index = samplerSlot * 2 + (Both ? 1 : 0), where samplerSlot = (useAlias ? Count : 0) + sampler
         // (so 0..3 = tree, 4..7 = alias; slot 4 is the OBB-alias the GUI defaults to).
         constexpr static inline uint8_t NeeDeferredPipelineCount = uint8_t(CSession::LightSampler::Count)*2/*tree+alias*/*2/*MIS*/;
         core::smart_refctd_ptr<video::IGPUPipelineLayout> neeDeferredLayout;
         std::array<core::smart_refctd_ptr<video::IGPUComputePipeline>,NeeDeferredPipelineCount> neeDeferredPipelines;
         // Per-bounce indirect wavefront: trace index = (leafMode!=OBB)*2 + Both, NEE same flat scheme
         // as neeDeferredPipelines; init/fixups are define-independent (built from one trace blob).
         core::smart_refctd_ptr<video::IGPUComputePipeline> wavefrontInitPipeline;
         core::smart_refctd_ptr<video::IGPUComputePipeline> wavefrontFixupFirstPipeline;
         core::smart_refctd_ptr<video::IGPUComputePipeline> wavefrontFixupBouncePipeline;
         std::array<core::smart_refctd_ptr<video::IGPUComputePipeline>,4> wavefrontTracePipelines;
         std::array<core::smart_refctd_ptr<video::IGPUComputePipeline>,NeeDeferredPipelineCount> wavefrontNeePipelines;
         // TODO
//			std::array<core::smart_refctd_ptr<video::IGPURayTracingPipeline>,uint8_t(CSession::RenderMode::Count)> genericPipelines;

         //
         core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FramesInFlight];

         //
         core::smart_refctd_ptr<video::IGPUBuffer> sobolSequence;
         //! Brief guideline to good path depth limits
         // Want to see stuff with indirect lighting on the other side of a pane of glass
         // 5 = glass frontface->glass backface->diffuse surface->diffuse surface->light
         // Want to see through a glass box, vase, or office 
         // 7 = glass frontface->glass backface->glass frontface->glass backface->diffuse surface->diffuse surface->light
         // pick higher numbers for better GI and less bias
         // TODO: Upload only a subsection of the sample sequence to the GPU, so we can use more samples without trashing VRAM
         examples::CCachedOwenScrambledSequence::SCacheHeader sequenceHeader = {};
      };
      //
      inline const SCachedConstructionParams& getConstructionParams() const {return m_construction;}
      
      //
      core::smart_refctd_ptr<CScene> createScene(CScene::SCreationParams&& _params);

      inline void setEmitterDensity(float d) { m_emitterDensity = std::clamp(d,0.f,1.f); }
      inline float getEmitterDensity() const { return m_emitterDensity; }

      inline void setMaxSppPerDispatch(uint16_t spp) { m_maxSppPerDispatch = spp; }
      inline uint16_t getMaxSppPerDispatch() const { return m_maxSppPerDispatch; }

      inline void setUseAliasNEE(bool v) { m_useAliasNEE = v; }
      inline bool getUseAliasNEE() const { return m_useAliasNEE; }

      // Picks both the shader variant and the CPU leaf granularity, so switching to or from OBB needs a
      // scene/light-tree rebuild, like setEmitterDensity.
      inline void setLeafSampler(CSession::LightSampler s)
      {
         m_sampler  = s;
         m_leafMode = (s == CSession::LightSampler::OBB) ? ELightLeafMode::PerInstanceOBB : ELightLeafMode::PerTriangle;
      }
      inline CSession::LightSampler getLeafSampler() const { return m_sampler; }

      inline void setMisMode(CSession::MisMode v) { m_misMode = v; }
      inline CSession::MisMode getMisMode() const { return m_misMode; }

      // NEE architecture A/B: 0 = inline megakernel, 1 = batched deferral (per-bounce slots + one fused
      // compute pass), 2 = per-bounce indirect wavefront (path pool + ID queues, memory O(live paths)).
      enum class DeferredNEEMode : uint8_t
      {
         Inline,
         Batched,
         Wavefront,
         Count
      };
      inline void setDeferredNEEMode(DeferredNEEMode m) { m_deferredMode = m; }
      inline DeferredNEEMode getDeferredNEEMode() const { return m_deferredMode; }

      // Batched memory knob: request buffer capacity divides by bandCount*spp; image is identical at
      // any value, the floor is GPU saturation per band (~100-150k pixels).
      inline void setNeeBandCount(uint32_t c) { m_neeBandCount = std::clamp(c, 1u, 64u); }
      inline uint32_t getNeeBandCount() const { return m_neeBandCount; }

      // NEE sampler stats counts over the last 16-frame window (SDebugProbe::neeStats indices:
      // calls, selFail, silhDegen, draws, dirDegen, zeroTarget, traced, confirmed, zeroContrib).
      inline const std::array<uint32_t, 12>& getNeeStats() const { return m_statsSnapshot; }

      void setProbe(const hlsl::float32_t3& point, const hlsl::float32_t3& normal);
      inline hlsl::float32_t3 getProbePoint()  const { return m_probePoint; }
      inline hlsl::float32_t3 getProbeNormal() const { return m_probeNormal; }
      inline float getProbePdfSum() const { return m_probePdfSum; }

      //
      struct SSubmit final : core::Uncopyable
      {
         public:
            inline SSubmit() {}
            inline SSubmit(CRenderer* _renderer, video::IGPUCommandBuffer* _cb) : renderer(_renderer), cb(_cb) {assert(operator bool());}

            inline operator bool() const {return cb;}
            inline operator video::IGPUCommandBuffer*() const {return cb;}

            // returns semaphore signalled by submit
            video::IQueue::SSubmitInfo::SSemaphoreInfo operator()(std::span<const video::IQueue::SSubmitInfo::SSemaphoreInfo> extraWaits);

            asset::PIPELINE_STAGE_FLAGS stageMask = asset::PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT;
         private:
            CRenderer* renderer = nullptr;
            video::IGPUCommandBuffer* cb = nullptr;
      };
      // Optional GPU-side timing: when `timing.queryPool` is non-null
      struct STimingScope
      {
         video::IQueryPool* queryPool = nullptr;
         uint32_t startQueryIdx = 0;
         uint32_t endQueryIdx = 0;
         // When true, the renderer overrides `keepAccumulating=0`
         bool forceFreshFrame = false;
         // When true, the renderer forces `keepAccumulating=1`, so a timed bench
         // window accumulates regardless of the session's stored dynamics (which
         // the CLI bench path never advances via update()/a stationary camera).
         // Mutually exclusive with forceFreshFrame.
         bool forceAccumulate = false;
         // When non-zero, overrides the session's per-pixel sample cap for
         // this frame so long-running benches keep accumulating instead of
         // no-op'ing once the scene's modest `samplesNeeded` is reached.
         uint32_t maxSPPOverride = 0;
      };
      SSubmit render(CSession* session, const STimingScope& timing = {});

    protected:
      struct SConstructorParams : SCachedCreationParams, SCachedConstructionParams
      {
#if 0		
         // Resources used for envmap sampling
         nbl::ext::EnvmapImportanceSampling::EnvmapImportanceSampling m_envMapImportanceSampling;
#endif
      };
      inline CRenderer(SConstructorParams&& _params) : m_creation(std::move(_params)), m_construction(std::move(_params)),
         m_frameIx(m_construction.semaphore->getCounterValue()) {}
      virtual inline ~CRenderer() {}

      static core::smart_refctd_ptr<asset::IShader> loadPrecompiledShader_impl(asset::IAssetManager* assMan, const core::string& key, system::logger_opt_ptr logger);

      SCachedCreationParams m_creation;
      SCachedConstructionParams m_construction;
      uint64_t m_frameIx;
      float m_emitterDensity = 0.1f;
      uint16_t m_maxSppPerDispatch = 3;
      bool m_useAliasNEE = true;
      ELightLeafMode m_leafMode = ELightLeafMode::PerInstanceOBB;
      CSession::LightSampler m_sampler = CSession::LightSampler::OBB;
      CSession::MisMode m_misMode = CSession::MisMode::Both;
      DeferredNEEMode m_deferredMode = DeferredNEEMode::Batched;
      // Last frame's effective deferral state + requested mode, for the one-shot log in render().
      bool            m_neeDeferredActive      = false;
      DeferredNEEMode m_deferredModeRequestedLast = DeferredNEEMode::Batched;
      // Deferred NEE request buffer, lazily (re)created in render() when the needed size grows.
      core::smart_refctd_ptr<video::IGPUBuffer> m_neeRequests;
      uint32_t m_neeBandCount = 4;

      // Debug probe state. The scene owns the device buffer; we hold CPU state and
      // a typed pointer to the host-coherent mapping that the scene set up at createScene().
      hlsl::float32_t3 m_probePoint  = {0.f, 0.f, 0.f};
      hlsl::float32_t3 m_probeNormal = {0.f, 1.f, 0.f};
      SDebugProbe*     m_debugProbeMapped = nullptr;
      uint32_t         m_statsFrameCounter = 0u;
      std::array<uint32_t, 12> m_statsSnapshot = {};

      float*                                    m_probeDebugPdfsMapped = nullptr;
      uint32_t                                  m_probeDebugPdfsCount  = 0;
      // Per-heap-node cumulative descent pdf (host-coherent), indexed by heap index;
      // the debug viz tints cluster boxes by it. Refilled on every setProbe().
      float*                                    m_nodePdfsMapped       = nullptr;
      uint32_t                                  m_nodePdfsCount        = 0;
      const SLightTree*                         m_lightTreeForProbe    = nullptr; // borrowed; owned by CScene
      float                                     m_probePdfSum          = 0.f;

      // Recompute the probe-derived telemetry (pdf sum + deterministic u=0.5
      // descent leaf) from the freshly filled per-emitter pdf array and write it
      // into the host-coherent probe buffer for the debug shader. Cheap; called
      // whenever the probe moves and once at scene setup.
      void updateProbeDerived();
};

}
#endif
