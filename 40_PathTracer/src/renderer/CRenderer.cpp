// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "renderer/CRenderer.h"
#include "renderer/CLightTree.h"
#include "renderer/SAASequence.h"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include <array>
#include <thread>
#include <future>
#include <filesystem>


namespace nbl::this_example
{
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::system;
using namespace nbl::video;


smart_refctd_ptr<IShader> CRenderer::loadPrecompiledShader_impl(IAssetManager* assMan, const core::string& key, logger_opt_ptr logger)
{
   IAssetLoader::SAssetLoadParams lp = {};
   lp.logger                         = logger;
   lp.workingDirectory               = "app_resources"; // virtual root
   auto       assetBundle            = assMan->getAsset(key, lp);
   const auto assets                 = assetBundle.getContents();
   if (!assets.empty())
      if (auto shader = IAsset::castDown<IShader>(*assets.begin()); shader)
         return shader;

   logger.log("Failed to load precompiled shader %s", ILogger::ELL_ERROR, key.c_str());
   return nullptr;
}

//
smart_refctd_ptr<CRenderer> CRenderer::create(SCreationParams&& _params)
{
   if (!_params)
      return nullptr;

   // get started with the sequence ASAP
   auto* const assMan         = _params.assMan;
   auto        sequenceFuture = std::async(
      std::launch::async,
      [assMan](std::string&& cachePath) -> auto
      {
         // TODO: resize the Sample Sequence after every scene load, smaller sequence has better caching properties
         return nbl::examples::CCachedOwenScrambledSequence::create(
            { .cachePath = std::move(cachePath), .assMan = assMan, .header = { .maxSamplesLog2 = MaxSPPLog2, .maxDimensions = (RandDimTriplesPerDepth * ((0x1u << SSceneUniforms::SInit::MaxPathDepthLog2) - 1) + PrimaryRayRandTripletsUsed) * 3 } });
      },
      _params.sequenceCachePath);

   SConstructorParams params = { std::move(_params) };

   //
   if (!params.logger.get())
      params.logger = smart_refctd_ptr<ILogger>(params.utilities->getLogger());
   logger_opt_ptr logger = params.logger.get().get();

   //
   auto checkNullObject = [&params, logger](auto& obj, const std::string_view debugName) -> bool
   {
      if (!obj)
      {
         logger.log("Failed to Create %s Object!", ILogger::ELL_ERROR, debugName.data());
         return true;
      }
      obj->setObjectDebugName(debugName.data());
      return false;
   };

   //
   ILogicalDevice* device = params.utilities->getLogicalDevice();

   //
   params.semaphore = device->createSemaphore(0);
   if (checkNullObject(params.semaphore, "CRenderer Semaphore"))
      return nullptr;

   // basic samplers
   const auto samplerDefaultRepeat = device->createSampler({});

   using render_mode_e = CSession::RenderMode;
   // create the layouts
   {
      constexpr auto RTStages        = hlsl::ShaderStage::ESS_ALL_RAY_TRACING; // | hlsl::ShaderStage::ESS_COMPUTE;
      constexpr auto RenderingStages = RTStages | hlsl::ShaderStage::ESS_COMPUTE;
      // descriptor
      {
         using binding_create_flags_t                           = IDescriptorSetLayoutBase::SBindingBase::E_CREATE_FLAGS;
         constexpr IGPUDescriptorSetLayout::SBinding UBOBinding = { .binding = SensorDSBindings::UBO, .type = IDescriptor::E_TYPE::ET_UNIFORM_BUFFER, .createFlags = binding_create_flags_t::ECF_NONE, .stageFlags = RenderingStages, .count = 1 };
         // the generic single-UBO
         {
            params.uboDSLayout = device->createDescriptorSetLayout({ &UBOBinding, 1 });
            if (checkNullObject(params.uboDSLayout, "Generic Single UBO Layout"))
               return nullptr;
         }
         constexpr auto DescriptorIndexingFlags = binding_create_flags_t::ECF_UPDATE_AFTER_BIND_BIT | binding_create_flags_t::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT | binding_create_flags_t::ECF_PARTIALLY_BOUND_BIT;
         //
         auto singleStorageImage = [](const uint32_t binding) -> IGPUDescriptorSetLayout::SBinding
         {
            return { .binding = binding, .type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE, .createFlags = binding_create_flags_t::ECF_PARTIALLY_BOUND_BIT, .stageFlags = RenderingStages, .count = 1 };
         };
         // TODO: provide these two samplers from Envmap Importance sampling extension
         const auto samplerNearestRepeat = device->createSampler({ {
                                                                      .MinFilter         = ISampler::E_TEXTURE_FILTER::ETF_NEAREST,
                                                                      .MaxFilter         = ISampler::E_TEXTURE_FILTER::ETF_NEAREST,
                                                                      .MipmapMode        = ISampler::E_SAMPLER_MIPMAP_MODE::ESMM_NEAREST,
                                                                      .AnisotropicFilter = 0,
                                                                   },
            0.f,
            0.f,
            0.f });
         // bindless everything
         {
            // TODO: provide these two samplers from Envmap Importance sampling extension
            const auto                                                     samplerEnvmapPDF     = samplerNearestRepeat;
            const auto                                                     samplerEnvmapWarpmap = device->createSampler({ {
                                                                                                                             .MinFilter         = ISampler::E_TEXTURE_FILTER::ETF_LINEAR,
                                                                                                                             .MaxFilter         = ISampler::E_TEXTURE_FILTER::ETF_LINEAR,
                                                                                                                             .MipmapMode        = ISampler::E_SAMPLER_MIPMAP_MODE::ESMM_NEAREST,
                                                                                                                             .AnisotropicFilter = 0,
                                                                                                                          },
               0.f,
               0.f,
               0.f });
            std::initializer_list<const IGPUDescriptorSetLayout::SBinding> bindings             = { UBOBinding,
               { .binding = SceneDSBindings::Envmap, .type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE, .createFlags = binding_create_flags_t::ECF_NONE, .stageFlags = RTStages, .count = 1, .immutableSamplers = &samplerDefaultRepeat },
               { .binding = SceneDSBindings::TLASes, .type = IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE, .createFlags = DescriptorIndexingFlags, .stageFlags = RTStages, .count = SceneDSBindingCounts::TLASes },
               { .binding = SceneDSBindings::Samplers, .type = IDescriptor::E_TYPE::ET_SAMPLER, .createFlags = DescriptorIndexingFlags, .stageFlags = RTStages, .count = SceneDSBindingCounts::Samplers },
               { .binding = SceneDSBindings::SampledImages, .type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE, .createFlags = DescriptorIndexingFlags, .stageFlags = RTStages, .count = SceneDSBindingCounts::SampledImages },
               { .binding = SceneDSBindings::EnvmapPDF, .type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE, .createFlags = DescriptorIndexingFlags, .stageFlags = RTStages, .count = 1, .immutableSamplers = &samplerEnvmapPDF },
               { .binding = SceneDSBindings::EnvmapWarpMap, .type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE, .createFlags = DescriptorIndexingFlags, .stageFlags = RTStages, .count = 1, .immutableSamplers = &samplerEnvmapWarpmap } };
            params.sceneDSLayout                                                                = device->createDescriptorSetLayout(bindings);
            if (checkNullObject(params.sceneDSLayout, "Scene Descriptor Layout"))
               return nullptr;
         }
         // the sensor layout
         {
            constexpr auto                                                 ResolveAndPresentStages = hlsl::ShaderStage::ESS_COMPUTE | hlsl::ShaderStage::ESS_FRAGMENT;
            const auto                                                     defaultSampler          = device->createSampler({ { .AnisotropicFilter = 0 }, 0.f, 0.f, 0.f });
            std::initializer_list<const IGPUDescriptorSetLayout::SBinding> bindings                = { UBOBinding,
               singleStorageImage(SensorDSBindings::ScrambleKey),
               singleStorageImage(SensorDSBindings::SampleCount),
               singleStorageImage(SensorDSBindings::Beauty),
               singleStorageImage(SensorDSBindings::RWMCCascades),
               singleStorageImage(SensorDSBindings::Albedo),
               singleStorageImage(SensorDSBindings::Normal),
               singleStorageImage(SensorDSBindings::Motion),
               singleStorageImage(SensorDSBindings::Mask),
               { .binding            = SensorDSBindings::Samplers,
                  .type              = IDescriptor::E_TYPE::ET_SAMPLER,
                  .createFlags       = binding_create_flags_t::ECF_NONE,
                  .stageFlags        = ResolveAndPresentStages,
                  .count             = SensorDSBindingCounts::Samplers,
                  .immutableSamplers = &defaultSampler },
               { .binding      = SensorDSBindings::AsSampledImages,
                  .type        = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
                  .createFlags = binding_create_flags_t::ECF_PARTIALLY_BOUND_BIT,
                  .stageFlags  = ResolveAndPresentStages,
                  .count       = SensorDSBindingCounts::AsSampledImages } };
            params.sensorDSLayout                                                                  = device->createDescriptorSetLayout(bindings);
            if (checkNullObject(params.sensorDSLayout, "Sensor Descriptor Layout"))
               return nullptr;
         }
      }

      // but many push constant ranges
      SPushConstantRange pcRanges[uint8_t(render_mode_e::Count)];
      auto               setPCRange = [&pcRanges]<typename T>(const render_mode_e mode) -> void
      {
         pcRanges[uint8_t(mode)] = { .stageFlags = RTStages, .offset = 0, .size = sizeof(T) };
      };
      setPCRange.operator()<SPrevisPushConstants>(render_mode_e::Previs);
      setPCRange.operator()<SBeautyPushConstants>(render_mode_e::Beauty);
      setPCRange.operator()<SDebugPushConstants>(render_mode_e::Debug);
      for (uint8_t t = 0; t < uint8_t(render_mode_e::Count); t++)
      {
         params.renderingLayouts[t] = device->createPipelineLayout({ pcRanges + t, 1 }, params.sceneDSLayout, params.sensorDSLayout);
         string debugName           = to_string(static_cast<render_mode_e>(t)) + "Rendering Pipeline Layout";
         if (checkNullObject(params.renderingLayouts[t], debugName))
            return nullptr;
      }
   }

   // TODO: create the generic pipelines
   params.shaders[uint8_t(render_mode_e::Previs)] = loadPrecompiledShader<"pathtrace_previs">(_params.assMan, device, logger);
   params.shaders[uint8_t(render_mode_e::Beauty)] = loadPrecompiledShader<"pathtrace_beauty">(_params.assMan, device, logger);
   params.shaders[uint8_t(render_mode_e::Debug)]  = loadPrecompiledShader<"pathtrace_debug">(_params.assMan, device, logger);
   for (auto i = 0; i < params.shaders.size(); i++)
      if (!params.shaders[i])
      {
         logger.log("Failed to Load %s Shader!", ILogger::ELL_ERROR, system::to_string(static_cast<render_mode_e>(i)));
         return nullptr;
      }
   // Beauty (NBL_MIS_MODE, NBL_NEE_USE_ALIAS) variants (same source, distinct defines). The default
   // (Both + alias) is the regular pathtrace_beauty above; these cover the non-default combos so the UI
   // / bench can A/B them as separate pipelines. Indices match CSession::BeautyVariant.
   params.beautyVariantShaders[uint8_t(CSession::BeautyVariant::NEEOnly_Alias)] = loadPrecompiledShader<"pathtrace_beauty_nee_only">(_params.assMan, device, logger);
   params.beautyVariantShaders[uint8_t(CSession::BeautyVariant::BxDFOnly)]      = loadPrecompiledShader<"pathtrace_beauty_bxdf_only">(_params.assMan, device, logger);
   params.beautyVariantShaders[uint8_t(CSession::BeautyVariant::Both_Tree)]     = loadPrecompiledShader<"pathtrace_beauty_tree">(_params.assMan, device, logger);
   params.beautyVariantShaders[uint8_t(CSession::BeautyVariant::NEEOnly_Tree)]  = loadPrecompiledShader<"pathtrace_beauty_nee_only_tree">(_params.assMan, device, logger);
   for (auto i = 0; i < params.beautyVariantShaders.size(); i++)
      if (!params.beautyVariantShaders[i])
      {
         logger.log("Failed to Load Beauty MIS-mode variant Shader %d!", ILogger::ELL_ERROR, int(i));
         return nullptr;
      }

   // command buffers
   for (uint8_t i = 0; i < SConstructorParams::FramesInFlight; i++)
   {
      auto pool = device->createCommandPool(params.graphicsQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
      if (pool)
         pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1, params.commandBuffers + i, smart_refctd_ptr(params.logger.get()));
      if (checkNullObject(params.commandBuffers[i], "Graphics Command Buffer " + to_string(i)))
         return nullptr;
   }

   // upload quantized LDS sequence buffer
   {
      auto sequence            = sequenceFuture.get();
      params.sequenceHeader    = sequence->getHeader();
      auto* const seqBufferCPU = sequence->getBuffer();
      params.utilities->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo { .queue = params.graphicsQueue }, IGPUBuffer::SCreationParams { seqBufferCPU->getCreationParams() }, seqBufferCPU->getPointer()).move_into(params.sobolSequence);
      params.sobolSequence->setObjectDebugName("Low Discrepancy Sequence");
   }

   return core::smart_refctd_ptr<CRenderer>(new CRenderer(std::move(params)), core::dont_grab);
}

void CRenderer::setProbe(const hlsl::float32_t3& point, const hlsl::float32_t3& normal)
{
   m_probePoint  = point;
   m_probeNormal = normal;
   if (m_debugProbeMapped)
   {
      m_debugProbeMapped->shadingPoint = point;
      m_debugProbeMapped->normal       = normal;
   }
   // Recompute per-emitter backward pdfs CPU-side. Probe moves at gizmo speed
   // (handful of times per second tops), so this is cheap, pure scalar math
   // over emitterToLeafIdx.size() entries. Replaces the shader-side descent
   // that was spilling at high emitter density.
   if (m_probeDebugPdfsMapped && m_lightTreeForProbe && m_probeDebugPdfsCount > 0)
   {
      computePerEmitterBackwardPdfCPU(*m_lightTreeForProbe, point, normal, std::span<float>(m_probeDebugPdfsMapped, m_probeDebugPdfsCount));
      if (m_nodePdfsMapped && m_nodePdfsCount > 0)
         computeNodePdfsCPU(*m_lightTreeForProbe, point, normal, std::span<float>(m_nodePdfsMapped, m_nodePdfsCount));
      updateProbeDerived();
   }
}

void CRenderer::updateProbeDerived()
{
   m_probePdfSum = 0.f;
   if (m_probeDebugPdfsMapped && m_probeDebugPdfsCount > 0)
      for (uint32_t e = 0u; e < m_probeDebugPdfsCount; ++e)
         m_probePdfSum += m_probeDebugPdfsMapped[e];

   if (!m_debugProbeMapped)
      return;
   m_debugProbeMapped->pdfSum          = m_probePdfSum;
   m_debugProbeMapped->descentLeafHeap = m_lightTreeForProbe ? computeDeterministicDescentLeafCPU(*m_lightTreeForProbe, m_probePoint, m_probeNormal, 0.5f) : ~0u;
}

core::smart_refctd_ptr<CScene> CRenderer::createScene(CScene::SCreationParams&& _params)
{
   if (!_params)
      return nullptr;

   // Drop probe state borrowed from a previously-loaded scene: the host-coherent
   // buffers and the light tree it referenced are released when the old scene
   // dies. Repopulated below if this scene has emitters; left null otherwise so
   // setProbe() safely no-ops.
   m_debugProbeMapped     = nullptr;
   m_probeDebugPdfsMapped = nullptr;
   m_probeDebugPdfsCount  = 0;
   m_nodePdfsMapped       = nullptr;
   m_nodePdfsCount        = 0;
   m_lightTreeForProbe    = nullptr;

   auto* const device    = getDevice();
   auto        converter = core::smart_refctd_ptr<CAssetConverter>(_params.converter);

   CScene::SConstructorParams params = { std::move(_params) };
   //	params.sceneBound = ;
   params.sensors  = std::move(_params.load.sensors);
   params.renderer = smart_refctd_ptr<CRenderer>(this);
   {
      auto pool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT, { &m_construction.sceneDSLayout.get(), 1 });
      auto ds   = pool->createDescriptorSet(smart_refctd_ptr(m_construction.sceneDSLayout));
      if (!ds)
      {
         m_creation.logger.log("Failed to create a scene - failed descriptor set allocation!", ILogger::ELL_ERROR);
         return nullptr;
      }
      params.sceneDS = make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(ds));
   }

   auto* const cpuScene = _params.load.scene.get();

   constexpr auto RenderModeCount = uint8_t(CSession::RenderMode::Count);
   // Beauty (NBL_MIS_MODE, NBL_NEE_USE_ALIAS) variant pipelines ride extra SBT slots after the real
   // modes so they share the single asset-converter pass below; the default (Both+alias) is the regular
   // Beauty slot.
   constexpr uint8_t BeautyVariantCount = uint8_t(CSession::BeautyVariant::Count);
   constexpr uint8_t SBTCount           = RenderModeCount + BeautyVariantCount;
   // create the pipelines
   {
      IGPURayTracingPipeline::SCreationParams creationParams[RenderModeCount] = {};
      using creation_flags_e                                                  = IGPURayTracingPipeline::SCreationParams::FLAGS;
      IGPURayTracingPipeline::SShaderSpecInfo missShaders[RenderModeCount]    = {};
      IGPURayTracingPipeline::SHitGroup       hitShaders[RenderModeCount]     = {};
      // NEE callables (see NBL_NEE_CALLABLE): [0]=neeCallable (forwardNEE), [1]=emissionCallable
      // (shadeEmission MIS deweight). Only the Beauty module compiles these entry points; registered
      // unconditionally (raygen routes through them only when the shader define is set) so there is no
      // C++<->shader toggle that could mismatch into a device-lost. The array order IS the callable
      // SBT index used by spirv::executeCallable (0 and 1).
      IGPURayTracingPipeline::SShaderSpecInfo callableShaders[RenderModeCount][2] = {};
      {
         for (uint8_t m = 0; m < RenderModeCount; m++)
         {
            const bool        isBeauty = m == uint8_t(CSession::RenderMode::Beauty);
            const auto* const shader   = m_construction.shaders[m].get();
            missShaders[m]             = { .shader = shader, .entryPoint = "miss" };
            hitShaders[m].closestHit   = { .shader = shader, .entryPoint = "closestHit" };
            if (isBeauty)
            {
               callableShaders[m][0] = { .shader = shader, .entryPoint = "neeCallable" };
               callableShaders[m][1] = { .shader = shader, .entryPoint = "emissionCallable" };
            }
            // right now we don't do any procedular geometry
            core::bitflag<creation_flags_e> flags = creation_flags_e::NO_NULL_INTERSECTION_SHADERS;
            //flags |= creation_flags_e::SKIP_AABBS; this probably doesn't make anything faster since the `rayTraversalPrimitiveCulling` feature would need to be enabled
            // we always have a miss shader, or because of SER we never call it
            flags |= creation_flags_e::NO_NULL_MISS_SHADERS;
            if (isBeauty) // beauty doesn't call the closest hit shaders
               flags |= creation_flags_e::NO_NULL_CLOSEST_HIT_SHADERS;
            // we use `NO_NULL_ANY_HIT_SHADERS` to control opacity on a per-instance and per-geo level though, OPAQUE flags are only conservative culling

            // When pipelineExecutableInfo is enabled (currently driven by the
            // bench path), opt into stats + IR capture so PipelineStats can be
            // populated for the bench report.
            const bool captureStats = device->getEnabledFeatures().pipelineExecutableInfo;
            if (captureStats)
            {
               flags |= creation_flags_e::CAPTURE_STATISTICS;
               flags |= creation_flags_e::CAPTURE_INTERNAL_REPRESENTATIONS;
            }
            if (m == 0)
               m_creation.logger.log("RT pipeline create: pipelineExecutableInfo=%s, CAPTURE flags=%s", ILogger::ELL_INFO, captureStats ? "ENABLED" : "DISABLED", captureStats ? "applied" : "skipped");

            const size_t beautyMisses = 0ull;
            const size_t beautyHits   = 0ull;
            creationParams[m]         = { .layout = m_construction.renderingLayouts[m].get(),
               .shaderGroups              = { .raygen = { .shader = shader, .entryPoint = "raygen" },
                  .misses                = { missShaders + m, isBeauty ? beautyMisses : 1ull },
                  .hits                  = { hitShaders + m, isBeauty ? beautyHits : 1ull },
                  // NEE callables: 2 groups for Beauty (neeCallable, emissionCallable), 0 otherwise.
                  // TODO: use Material Compiler to get materials' callables for us
                  .callables = { callableShaders[m], isBeauty ? 2ull : 0ull } },
               // The path tracer loops in raygen and traces shadow/continuation rays from there,
               // so closest hit and miss never trace; one level of ray recursion is all that's used.
               .cached = { .flags = flags, .maxRecursionDepth = 1 } };
         }
      }
      if (!device->createRayTracingPipelines(nullptr, creationParams, params.pipelines))
      {
         m_creation.logger.log("Failed to create Path Tracing Pipelines", ILogger::ELL_ERROR);
         return nullptr;
      }

      // Beauty MIS-mode variants: identical layout/flags/group-structure to the Beauty pipeline, only
      // the shader (and its baked NBL_MIS_MODE) differs. Separate pipelines so each mode's unused code
      // is DCE'd and there's no shared-binary occupancy coupling between modes. Clone the Beauty
      // creationParams and swap the shader into every group.
      {
         constexpr uint8_t                       beautySlot                             = uint8_t(CSession::RenderMode::Beauty);
         IGPURayTracingPipeline::SCreationParams variantParams[BeautyVariantCount]      = {};
         IGPURayTracingPipeline::SShaderSpecInfo variantMiss[BeautyVariantCount]        = {};
         IGPURayTracingPipeline::SHitGroup       variantHit[BeautyVariantCount]         = {};
         IGPURayTracingPipeline::SShaderSpecInfo variantCallable[BeautyVariantCount][2] = {};
         for (uint8_t k = 0; k < BeautyVariantCount; k++)
         {
            const auto* const shader                = m_construction.beautyVariantShaders[k].get();
            variantMiss[k]                          = { .shader = shader, .entryPoint = "miss" };
            variantHit[k].closestHit                = { .shader = shader, .entryPoint = "closestHit" };
            variantCallable[k][0]                   = { .shader = shader, .entryPoint = "neeCallable" };
            variantCallable[k][1]                   = { .shader = shader, .entryPoint = "emissionCallable" };
            variantParams[k]                        = creationParams[beautySlot];
            variantParams[k].shaderGroups.raygen    = { .shader = shader, .entryPoint = "raygen" };
            variantParams[k].shaderGroups.misses    = { variantMiss + k, creationParams[beautySlot].shaderGroups.misses.size() };
            variantParams[k].shaderGroups.hits      = { variantHit + k, creationParams[beautySlot].shaderGroups.hits.size() };
            variantParams[k].shaderGroups.callables = { variantCallable[k], 2ull };
         }
         if (!device->createRayTracingPipelines(nullptr, variantParams, params.beautyVariantPipelines))
         {
            m_creation.logger.log("Failed to create Beauty MIS-mode variant Pipelines", ILogger::ELL_ERROR);
            return nullptr;
         }
      }
   }

   // TODO: make this configurable
   constexpr bool movableInstances = false;

   smart_refctd_ptr<IGPUBuffer>                                    ubo;
   core::vector<asset_cached_t<ICPUTopLevelAccelerationStructure>> TLASes;
   // kept alive across TLAS construction + buffer upload so the emitter table upload below
   // can read per-leaf radiance (radiance isn't retained on the heap-indexed tree nodes).
   core::vector<SLightTreeLeaf> emitterLeaves;
   // instancedGeometryID -> emitterID aux map; filled by selectRandInstancesAsEmitterLeaves, uploaded as a buffer below.
   core::vector<uint32_t> instancedGeoToEmitter;
   {
      // construct the TLASes
      core::vector<smart_refctd_ptr<ICPUTopLevelAccelerationStructure>> tmpTLASes;
      // main TLAS
      {
         using tlas_build_f           = ICPUTopLevelAccelerationStructure::BUILD_FLAGS;
         constexpr auto baseTLASFlags = tlas_build_f::PREFER_FAST_TRACE_BIT | tlas_build_f::ALLOW_COMPACTION_BIT;

         auto& main = tmpTLASes.emplace_back();
         main       = make_smart_refctd_ptr<ICPUTopLevelAccelerationStructure>();
         {
            auto& cpuInstances = cpuScene->getInstances();
            // need to convert weird topologies to lists
            for (auto i = 0u; i < cpuInstances.size(); i++)
               if (const auto* const targets = cpuInstances.getMorphTargets()[i].get(); targets)
                  for (auto& target : targets->getTargets())
                  {
                     auto* const collection = target.geoCollection.get();
                     for (auto& ref : *collection->getGeometries())
                     {
                        const auto* geo = ref.geometry.get();
                        if (geo->getPrimitiveType() != IGeometryBase::EPrimitiveType::Polygon)
                           continue;
                        ref.geometry = CPolygonGeometryManipulator::createTriangleListIndexing(static_cast<const ICPUPolygonGeometry*>(geo));
                     }
                  }
            ICPUScene::CDefaultTLASExporter exporter(cpuInstances);
            auto                            exported = exporter();
            if (!exported)
            {
               m_creation.logger.log("Failed to convert TLAS instances!", ILogger::ELL_ERROR);
               return nullptr;
            }
            if (!exported.allInstancesValid)
               m_creation.logger.log("Some instances in the scene are invisible!", ILogger::ELL_ERROR);
            for (auto& pair : exporter.m_blasCache)
            {
               using blas_build_f = ICPUBottomLevelAccelerationStructure::BUILD_FLAGS;
               auto flags         = pair.second->getBuildFlags();
               flags |= blas_build_f::ALLOW_DATA_ACCESS;
               pair.second->setBuildFlags(flags);
            }
            for (auto& instance : *exported.instances)
            {
               // TODO: for now, we need material compiler and knowing how we'll orgarnise our SBT
               instance.getBase().instanceShaderBindingTableRecordOffset = 0;
            }
            {
               // Fixed seed -> deterministic emitter membership per (scene, density) across runs.
               constexpr uint32_t           EmitterRngSeed = 0xb4017ee5u;
               SEmitterSelectionDiagnostics diag;
               emitterLeaves    = selectRandInstancesAsEmitterLeaves({ exported.instances->data(), exported.instances->size() }, exporter.m_blasCache, m_emitterDensity, EmitterRngSeed, instancedGeoToEmitter, &diag);
               params.lightTree = buildLightTreeCPU(emitterLeaves);
               m_creation.logger.log("Light tree: %u emitters of %u instances; %zu nodes (padded to %u leaves)"
                                     " [eligible=%u, skipped: nonStatic=%u noColl=%u emptyAABB=%u; pickedRng=%u forced=%u]",
                  ILogger::ELL_INFO,
                  uint32_t(emitterLeaves.size()),
                  diag.instancesTotal,
                  params.lightTree.nodes.size(),
                  params.lightTree.numLeavesPadded,
                  diag.eligible,
                  diag.skippedNonStatic,
                  diag.skippedNoCollection,
                  diag.skippedEmptyAABB,
                  diag.pickedByRng,
                  diag.forcedPick);
               // Per-leaf bbox-extent histogram. If p95/median is much greater
               // than ~10 OR if leaf max-extent approaches sceneOverallMaxExtent,
               // the AABBs likely include non-emitting BLAS geometry (e.g. lamp
               // housings) and the tree's importance weights will be wrong.
               m_creation.logger.log("Light tree leaf bbox stats: maxExtent[min=%g med=%g p95=%g max=%g; sceneMax=%g]  surfaceArea[min=%g med=%g p95=%g max=%g]",
                  ILogger::ELL_INFO,
                  diag.leafBboxMaxExtentMin,
                  diag.leafBboxMaxExtentMedian,
                  diag.leafBboxMaxExtentP95,
                  diag.leafBboxMaxExtentMax,
                  diag.sceneOverallMaxExtent,
                  diag.leafSurfaceAreaMin,
                  diag.leafSurfaceAreaMedian,
                  diag.leafSurfaceAreaP95,
                  diag.leafSurfaceAreaMax);
            }
            main->setInstances(std::move(exported.instances));
         }
         // de-instancing and welding BLASes
         {
            // TODO: de-instancing step, need AS memory budget and a heuristic of which instances to eliminate (out of the ones we don't want to be movable)
            // probably need OOBs of Geometry Collections, best heuristic would be "total surface of all intersections with other instances" divided by build memory.
            // In the meantime can do own OBB/AABB surface area divded by build size.
            // NOTE: can only "weld" BLASes which have the same build flags
         }
         {
            auto flags = baseTLASFlags;
            if (movableInstances)
               flags |= tlas_build_f::ALLOW_UPDATE_BIT;
            main->setBuildFlags(flags);
         }
      }

      struct Buffers final
      {
         using render_mode_e = CSession::RenderMode;
         // [ubo, sbts[real modes + beauty variants]], the variant SBTs ride the same converter pass.
         inline operator std::span<const ICPUBuffer* const>() const { return { &ubo.get(), 1 + SBTCount }; }

         smart_refctd_ptr<ICPUBuffer> ubo;
         smart_refctd_ptr<ICPUBuffer> sbts[SBTCount];
      } tmpBuffers;
      //
      using buffer_usage_e             = IGPUBuffer::E_USAGE_FLAGS;
      constexpr auto BasicBufferUsages = buffer_usage_e::EUF_SHADER_DEVICE_ADDRESS_BIT;
      // Upload the light tree before filling the UBO, since the UBO carries the buffer's BDA.
      // CWBVH-4: wide-node array (32 B each) + precise per-leaf array (16 B each).
      if (!params.lightTree.wideNodes.empty())
      {
         static_assert(sizeof(SLightTreeWideNode) == 32);
         IGPUBuffer::SCreationParams nodeBufferCreationParams = {};
         nodeBufferCreationParams.size                        = sizeof(SLightTreeWideNode) * params.lightTree.wideNodes.size();
         nodeBufferCreationParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT | buffer_usage_e::EUF_TRANSFER_DST_BIT;
         m_creation.utilities->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo { .queue = m_creation.graphicsQueue }, std::move(nodeBufferCreationParams), params.lightTree.wideNodes.data()).move_into(params.lightTreeNodes);
         if (params.lightTreeNodes)
            params.lightTreeNodes->setObjectDebugName("Light Tree Wide Nodes");
      }
      if (!params.lightTree.leaves.empty())
      {
         static_assert(sizeof(SLightTreeLeaf_GPU) == 32);
         IGPUBuffer::SCreationParams leafBufferCreationParams = {};
         leafBufferCreationParams.size                        = sizeof(SLightTreeLeaf_GPU) * params.lightTree.leaves.size();
         leafBufferCreationParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT | buffer_usage_e::EUF_TRANSFER_DST_BIT;
         m_creation.utilities->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo { .queue = m_creation.graphicsQueue }, std::move(leafBufferCreationParams), params.lightTree.leaves.data()).move_into(params.lightTreeLeaves);
         if (params.lightTreeLeaves)
            params.lightTreeLeaves->setObjectDebugName("Light Tree Leaves");
      }
      // Upload the emitter table (one entry per emitter, indexed by the dense emitterID, the value
      // the per-geometry aux map resolves a hit to, NOT instanceCustomIndex which is now a base).
      if (!emitterLeaves.empty())
      {
         static_assert(sizeof(SEmitterGPU) == 48);
         const bool                haveLeafMap = !params.lightTree.emitterToLeafIdx.empty();
         core::vector<SEmitterGPU> gpuEmitters(emitterLeaves.size());
         for (const auto& leaf : emitterLeaves)
         {
            assert(leaf.emitterID < gpuEmitters.size());
            SEmitterGPU& e = gpuEmitters[leaf.emitterID];
            e.radiance     = leaf.radiance;
            // Co-located leaf data so the shader skips the emitter -> leaf reverse-map chase.
            e.leafHeap = haveLeafMap ? params.lightTree.emitterToLeafIdx[leaf.emitterID] : 0u;
            e.bboxMin  = leaf.worldAABB.minVx;
            e.bboxMax  = leaf.worldAABB.maxVx;
            e._pad     = nbl::hlsl::float32_t2(0.f, 0.f);
         }
         IGPUBuffer::SCreationParams emitterBufferCreationParams = {};
         emitterBufferCreationParams.size                        = sizeof(SEmitterGPU) * gpuEmitters.size();
         emitterBufferCreationParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT | buffer_usage_e::EUF_TRANSFER_DST_BIT;
         m_creation.utilities->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo { .queue = m_creation.graphicsQueue }, std::move(emitterBufferCreationParams), gpuEmitters.data()).move_into(params.emitters);
         if (params.emitters)
            params.emitters->setObjectDebugName("Emitter Table");
      }
      // Upload the emitter -> heap-leaf-index reverse map (for backward NEE pdf walks).
      if (!params.lightTree.emitterToLeafIdx.empty())
      {
         IGPUBuffer::SCreationParams reverseMapCreationParams = {};
         reverseMapCreationParams.size                        = sizeof(uint32_t) * params.lightTree.emitterToLeafIdx.size();
         reverseMapCreationParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT | buffer_usage_e::EUF_TRANSFER_DST_BIT;
         m_creation.utilities->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo { .queue = m_creation.graphicsQueue }, std::move(reverseMapCreationParams), params.lightTree.emitterToLeafIdx.data()).move_into(params.emitterToLeafIdx);
         if (params.emitterToLeafIdx)
            params.emitterToLeafIdx->setObjectDebugName("Emitter to Leaf Index");
      }
      // Upload the instancedGeometryID -> emitterID aux map (hit-side emitter resolution).
      if (!instancedGeoToEmitter.empty())
      {
         IGPUBuffer::SCreationParams auxCreationParams = {};
         auxCreationParams.size                        = sizeof(uint32_t) * instancedGeoToEmitter.size();
         auxCreationParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT | buffer_usage_e::EUF_TRANSFER_DST_BIT;
         m_creation.utilities->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo { .queue = m_creation.graphicsQueue }, std::move(auxCreationParams), instancedGeoToEmitter.data()).move_into(params.instancedGeometryToEmitter);
         if (params.instancedGeometryToEmitter)
            params.instancedGeometryToEmitter->setObjectDebugName("InstancedGeometry to Emitter");
      }
      // Upload alias-table buffers (power-only global emitter sampler).
      if (params.lightTree.aliasTableSize > 0)
      {
         IGPUBuffer::SCreationParams aliasEntriesParams = {};
         aliasEntriesParams.size                        = sizeof(uint32_t) * params.lightTree.aliasEntries.size();
         aliasEntriesParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT | buffer_usage_e::EUF_TRANSFER_DST_BIT;
         m_creation.utilities->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo { .queue = m_creation.graphicsQueue }, std::move(aliasEntriesParams), params.lightTree.aliasEntries.data()).move_into(params.aliasEntries);
         if (params.aliasEntries)
            params.aliasEntries->setObjectDebugName("Alias Entries");

         IGPUBuffer::SCreationParams aliasPdfParams = {};
         aliasPdfParams.size                        = sizeof(float) * params.lightTree.aliasPdf.size();
         aliasPdfParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT | buffer_usage_e::EUF_TRANSFER_DST_BIT;
         m_creation.utilities->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo { .queue = m_creation.graphicsQueue }, std::move(aliasPdfParams), params.lightTree.aliasPdf.data()).move_into(params.aliasPdf);
         if (params.aliasPdf)
            params.aliasPdf->setObjectDebugName("Alias PDF");
      }
      // Upload per-internal-node alias tables (one contiguous buffer, four sections back-to-back).
      // Section sizes are derived in the shader from lightTreeFirstLeafIndex + subtreeAliasTotalEntries.
      if (params.lightTree.subtreeAliasOffsets.size() > 1u && !params.lightTree.subtreeAliasEntries.empty())
      {
         const uint32_t numInternal  = uint32_t(params.lightTree.subtreeLeafBases.size());
         const uint32_t totalEntries = uint32_t(params.lightTree.subtreeAliasEntries.size());
         assert(params.lightTree.subtreeAliasOffsets.size() == size_t(numInternal) + 1u);
         assert(params.lightTree.subtreeAliasPdfs.size() == totalEntries);

         const size_t          bytes = sizeof(uint32_t) * (size_t(numInternal + 1u) + size_t(numInternal) + size_t(totalEntries) + size_t(totalEntries));
         std::vector<uint32_t> concat(bytes / sizeof(uint32_t));
         uint32_t*             dst = concat.data();
         std::memcpy(dst, params.lightTree.subtreeAliasOffsets.data(), sizeof(uint32_t) * (numInternal + 1u));
         dst += numInternal + 1u;
         std::memcpy(dst, params.lightTree.subtreeLeafBases.data(), sizeof(uint32_t) * numInternal);
         dst += numInternal;
         std::memcpy(dst, params.lightTree.subtreeAliasEntries.data(), sizeof(uint32_t) * totalEntries);
         dst += totalEntries;
         std::memcpy(dst, params.lightTree.subtreeAliasPdfs.data(), sizeof(float) * totalEntries);

         IGPUBuffer::SCreationParams subtreeParams = {};
         subtreeParams.size                        = bytes;
         subtreeParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT | buffer_usage_e::EUF_TRANSFER_DST_BIT;
         m_creation.utilities->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo { .queue = m_creation.graphicsQueue }, std::move(subtreeParams), concat.data()).move_into(params.subtreeAlias);
         if (params.subtreeAlias)
            params.subtreeAlias->setObjectDebugName("Subtree Alias");
      }
      // Per-emitter quantization quality (float per emitter; debug viz only).
      if (!params.lightTree.quantQuality.empty())
      {
         IGPUBuffer::SCreationParams quantQualityParams = {};
         quantQualityParams.size                        = sizeof(float) * params.lightTree.quantQuality.size();
         quantQualityParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT | buffer_usage_e::EUF_TRANSFER_DST_BIT;
         m_creation.utilities->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo { .queue = m_creation.graphicsQueue }, std::move(quantQualityParams), params.lightTree.quantQuality.data()).move_into(params.quantQuality);
         if (params.quantQuality)
            params.quantQuality->setObjectDebugName("Quantization Quality");
      }
      // Host-visible debug-probe buffer (32 B = SDebugProbe). CPU writes via setProbe()
      // become visible to the GPU next frame; no explicit barrier needed thanks to coherent mapping.
      {
         IGPUBuffer::SCreationParams probeCreationParams = {};
         probeCreationParams.size                        = sizeof(SDebugProbe);
         probeCreationParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT;
         auto probeBuf                                   = device->createBuffer(std::move(probeCreationParams));
         if (probeBuf)
         {
            auto reqs = probeBuf->getMemoryReqs();
            reqs.memoryTypeBits &= device->getPhysicalDevice()->getDirectVRAMAccessMemoryTypeBits();
            auto alloc = device->allocate(reqs, probeBuf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
            if (alloc.memory)
            {
               alloc.memory->map({ .offset = 0, .length = reqs.size });
               m_debugProbeMapped                  = static_cast<SDebugProbe*>(alloc.memory->getMappedPointer());
               m_debugProbeMapped->shadingPoint    = m_probePoint;
               m_debugProbeMapped->pdfSum          = 0.f;
               m_debugProbeMapped->normal          = m_probeNormal;
               m_debugProbeMapped->descentLeafHeap = ~0u;
               params.debugProbe                   = std::move(probeBuf);
               if (params.debugProbe)
                  params.debugProbe->setObjectDebugName("Debug Probe");
            }
         }
      }
      // Host-coherent float[numEmittersActual] of per-emitter NEE backward pdfs
      // against the current probe. Recomputed by setProbe(). Lives alongside
      // the host-coherent debug-probe buffer so the gizmo refresh path doesn't
      // need a fence/barrier dance.
      const uint32_t numEmittersActual = uint32_t(params.lightTree.emitterToLeafIdx.size());
      if (numEmittersActual > 0)
      {
         IGPUBuffer::SCreationParams pdfsCreationParams = {};
         pdfsCreationParams.size                        = sizeof(float) * numEmittersActual;
         pdfsCreationParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT;
         auto pdfsBuf                                   = device->createBuffer(std::move(pdfsCreationParams));
         if (pdfsBuf)
         {
            auto reqs = pdfsBuf->getMemoryReqs();
            reqs.memoryTypeBits &= device->getPhysicalDevice()->getDirectVRAMAccessMemoryTypeBits();
            auto alloc = device->allocate(reqs, pdfsBuf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
            if (alloc.memory)
            {
               alloc.memory->map({ .offset = 0, .length = reqs.size });
               m_probeDebugPdfsMapped = static_cast<float*>(alloc.memory->getMappedPointer());
               m_probeDebugPdfsCount  = numEmittersActual;
               m_lightTreeForProbe    = &params.lightTree;
               // Initial fill against the current probe so the first frame already
               // has correct values.
               computePerEmitterBackwardPdfCPU(params.lightTree, m_probePoint, m_probeNormal, std::span<float>(m_probeDebugPdfsMapped, m_probeDebugPdfsCount));
               updateProbeDerived();
               params.probeDebugPdfs = std::move(pdfsBuf);
               if (params.probeDebugPdfs)
                  params.probeDebugPdfs->setObjectDebugName("Per-Emitter Probe PDFs");
            }
         }
      }
      // Host-coherent float[tree.nodes.size()] of per-heap-node cumulative descent
      // pdf against the current probe; the debug viz tints each cluster box by it.
      // Same coherent-mapping pattern as the per-emitter buffer above.
      const uint32_t numTreeNodes = uint32_t(params.lightTree.nodes.size());
      if (numTreeNodes > 0)
      {
         IGPUBuffer::SCreationParams nodePdfsCreationParams = {};
         nodePdfsCreationParams.size                        = sizeof(float) * numTreeNodes;
         nodePdfsCreationParams.usage                       = BasicBufferUsages | buffer_usage_e::EUF_STORAGE_BUFFER_BIT;
         auto nodePdfsBuf                                   = device->createBuffer(std::move(nodePdfsCreationParams));
         if (nodePdfsBuf)
         {
            auto reqs = nodePdfsBuf->getMemoryReqs();
            reqs.memoryTypeBits &= device->getPhysicalDevice()->getDirectVRAMAccessMemoryTypeBits();
            auto alloc = device->allocate(reqs, nodePdfsBuf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
            if (alloc.memory)
            {
               alloc.memory->map({ .offset = 0, .length = reqs.size });
               m_nodePdfsMapped = static_cast<float*>(alloc.memory->getMappedPointer());
               m_nodePdfsCount  = numTreeNodes;
               computeNodePdfsCPU(params.lightTree, m_probePoint, m_probeNormal, std::span<float>(m_nodePdfsMapped, m_nodePdfsCount));
               params.nodePdfs = std::move(nodePdfsBuf);
               if (params.nodePdfs)
                  params.nodePdfs->setObjectDebugName("Per-Node Probe PDFs");
            }
         }
      }
      {
         tmpBuffers.ubo                    = ICPUBuffer::create({ { .size = sizeof(SSceneUniforms), .usage = BasicBufferUsages | buffer_usage_e::EUF_UNIFORM_BUFFER_BIT }, nullptr });
         auto& uniforms                    = *reinterpret_cast<SSceneUniforms*>(tmpBuffers.ubo->getPointer());
         uniforms.init                     = {};
         uniforms.init.pSampleSequence     = m_construction.sobolSequence->getDeviceAddress();
         uniforms.init.sequenceSamplesLog2 = m_construction.sequenceHeader.maxSamplesLog2;
         // TODO: Some Constant to Tell us how many dimensions each path vertex consumes
         uniforms.init.lastSequencePathDepth       = m_construction.getSequenceMaxPathDepth();
         uniforms.init.pLightTreeNodes             = params.lightTreeNodes ? params.lightTreeNodes->getDeviceAddress() : 0;
         uniforms.init.pLightTreeLeaves            = params.lightTreeLeaves ? params.lightTreeLeaves->getDeviceAddress() : 0;
         uniforms.init.pEmitters                   = params.emitters ? params.emitters->getDeviceAddress() : 0;
         uniforms.init.pEmitterToLeafIdx           = params.emitterToLeafIdx ? params.emitterToLeafIdx->getDeviceAddress() : 0;
         uniforms.init.pInstancedGeometryToEmitter = params.instancedGeometryToEmitter ? params.instancedGeometryToEmitter->getDeviceAddress() : 0;
         uniforms.init.pAliasEntries               = params.aliasEntries ? params.aliasEntries->getDeviceAddress() : 0;
         uniforms.init.pAliasPdf                   = params.aliasPdf ? params.aliasPdf->getDeviceAddress() : 0;
         uniforms.init.aliasTableSize              = params.lightTree.aliasTableSize;
         uniforms.init.pSubtreeAlias               = params.subtreeAlias ? params.subtreeAlias->getDeviceAddress() : 0;
         uniforms.init.subtreeAliasTotalEntries    = uint32_t(params.lightTree.subtreeAliasEntries.size());
         uniforms.init.pDebugProbe                 = params.debugProbe ? params.debugProbe->getDeviceAddress() : 0;
         uniforms.init.pProbeDebugPdfs             = params.probeDebugPdfs ? params.probeDebugPdfs->getDeviceAddress() : 0;
         uniforms.init.pNodePdfs                   = params.nodePdfs ? params.nodePdfs->getDeviceAddress() : 0;
         uniforms.init.pQuantQuality               = params.quantQuality ? params.quantQuality->getDeviceAddress() : 0;
         uniforms.init.lightTreeFirstLeafIndex     = params.lightTree.firstLeafIndex;
         uniforms.init.lightTreeNumLeavesPadded    = params.lightTree.numLeavesPadded;
         tmpBuffers.ubo->setContentHash(tmpBuffers.ubo->computeContentHash());
      }
      // SBT
      const auto& limits = device->getPhysicalDevice()->getLimits();
      assert(limits.shaderGroupBaseAlignment >= limits.shaderGroupHandleAlignment);
      constexpr auto HandleSize        = SPhysicalDeviceLimits::ShaderGroupHandleSize;
      const auto     handleSizeAligned = nbl::core::alignUp(HandleSize, limits.shaderGroupHandleAlignment);
      for (uint8_t i = 0; i < SBTCount; i++)
      {
         // Slots [0,RenderModeCount) are the real render modes; the trailing BeautyVariantCount slots
         // are the Beauty MIS-mode variants. Both feed one converter pass via tmpBuffers.sbts.
         auto* const pipeline        = (i < RenderModeCount) ? params.pipelines[i].get() : params.beautyVariantPipelines[i - RenderModeCount].get();
         const auto  hitHandles      = pipeline->getHitHandles();
         const auto  missHandles     = pipeline->getMissHandles();
         const auto  callableHandles = pipeline->getCallableHandles();
         //
         {
            class CVectorBacked final : public core::refctd_memory_resource
            {
            public:
               inline CVectorBacked(const size_t reservation) { storage.reserve(reservation * HandleSize); }

               inline void* allocate(size_t bytes, size_t alignment) override
               {
                  assert(bytes == storage.size());
                  return storage.data();
               }
               inline void deallocate(void* p, size_t bytes, size_t alignment) override { storage = {}; }

               core::vector<uint8_t> storage;
            };
            auto memRsc = core::make_smart_refctd_ptr<CVectorBacked>(hitHandles.size() + missHandles.size() + callableHandles.size() + 1);
            {
               core::LinearAddressAllocatorST<uint32_t> allocator(nullptr, 0, 0, limits.shaderGroupBaseAlignment, 0x7fff0000u);
               // Without SER, the arrays for Miss and Closest Hit cannot be null, unless you can guarantee a Ray will never hit geometry or miss
               auto copyShaderHandles = [&](const std::span<const IGPURayTracingPipeline::SShaderGroupHandle> handles, const size_t minCount = 1) -> SBufferRange<const IGPUBuffer>
               {
                  SBufferRange<const IGPUBuffer> range = { .size = hlsl::max(handles.size(), minCount) * handleSizeAligned };
                  range.offset                         = allocator.alloc_addr(range.size, limits.shaderGroupBaseAlignment);
                  memRsc->storage.resize(core::alignUp(allocator.get_allocated_size(), limits.shaderGroupBaseAlignment));
                  uint8_t* out = memRsc->storage.data() + range.offset;
                  for (const auto& handle : handles)
                  {
                     memcpy(out, &handle, HandleSize);
                     out += handleSizeAligned;
                  }
                  for (auto i = minCount; i < handles.size(); i++)
                  {
                     memset(out, 0, HandleSize);
                     out += handleSizeAligned;
                  }
                  return range;
               };
               auto& sbt      = (i < RenderModeCount) ? params.sbts[i] : params.beautyVariantSbts[i - RenderModeCount];
               sbt.raygen     = copyShaderHandles({ &pipeline->getRaygen(), 1 });
               sbt.miss.range = copyShaderHandles(pipeline->getMissHandles());
               // TODO: the material compiler with an RT pipeline backend should give 3 or 4 hitgroups depending on opacity and other funny things
               // problem is that due to how TLAS instances and their Geometries call into hitgroups, we need to spam duplicates around the SBT
               // also de-dup stuff that has the same hash (array of hitgroups) so two instances can happily point at the same material
               sbt.hit.range = copyShaderHandles(pipeline->getHitHandles());
               // TODO: material compiler will give us callables and we need to turn those into materials
               sbt.callable.range = copyShaderHandles(pipeline->getCallableHandles(), 0);
               // TODO: futhermore different rays (NEE vs BxDF) should use different SBTs using big offsets so it becomes a really funny mess
               sbt.miss.stride = sbt.hit.stride = sbt.callable.stride = handleSizeAligned;
            }
            auto& sbtBuff = tmpBuffers.sbts[i];
            sbtBuff       = ICPUBuffer::create({ { .size = memRsc->storage.size(), .usage = BasicBufferUsages | buffer_usage_e::EUF_SHADER_BINDING_TABLE_BIT },
                                                  /*.data = */ memRsc->storage.data(),
                                                  /*.memoryResource = */ memRsc },
               core::adopt_memory);
            sbtBuff->setContentHash(sbtBuff->computeContentHash());
         }
      }

      // new cache if none provided
      if (!converter)
         converter = CAssetConverter::create({ .device = device, .optimizer = {} });

      // customized setup
      struct MyInputs : CAssetConverter::SInputs
      {
         // For the GPU Buffers to be directly writeable and so that we don't need a Transfer Queue submit at all
         inline uint32_t constrainMemoryTypeBits(const size_t groupCopyID, const IAsset* canonicalAsset, const blake3_hash_t& contentHash, const IDeviceMemoryBacked* memoryBacked) const override
         {
            assert(memoryBacked);
            return memoryBacked->getObjectType() != IDeviceMemoryBacked::EOT_BUFFER ? (~0u) : rebarMemoryTypes;
         }

         uint32_t rebarMemoryTypes;
      } inputs                = {};
      inputs.logger           = m_creation.logger.get().get();
      inputs.rebarMemoryTypes = device->getPhysicalDevice()->getDirectVRAMAccessMemoryTypeBits();
      // the allocator needs to be overriden to hand out memory ranges which have already been mapped so that the ReBAR fast-path can kick in
      // (multiple buffers can be bound to same memory, but memory can only be mapped once at one place, so Asset Converter can't do it)
      struct MyAllocator final : public IDeviceMemoryAllocator
      {
         ILogicalDevice* getDeviceForAllocations() const override { return device; }

         SAllocation allocate(const SAllocateInfo& info) override
         {
            auto retval = device->allocate(info);
            // map what is mappable by default so ReBAR checks succeed
            if (retval.isValid() && retval.memory->isMappable())
               retval.memory->map({ .offset = 0, .length = info.size });
            return retval;
         }

         ILogicalDevice* device;
      } myalloc;
      myalloc.device   = device;
      inputs.allocator = &myalloc;

      // assign inputs
      {
         std::get<CAssetConverter::SInputs::asset_span_t<ICPUBuffer>>(inputs.assets)                        = tmpBuffers;
         std::get<CAssetConverter::SInputs::asset_span_t<ICPUTopLevelAccelerationStructure>>(inputs.assets) = { &tmpTLASes.front().get(), tmpTLASes.size() };
      }
      CAssetConverter::SReserveResult reservation = converter->reserve(inputs);
      {
         bool success = true;
         auto check   = [&]<typename asset_type_t>(const CAssetConverter::SInputs::asset_span_t<asset_type_t> references) -> void
         {
            auto objects     = reservation.getGPUObjects<asset_type_t>();
            auto referenceIt = references.begin();
            for (auto& object : objects)
            {
               auto* reference = *(referenceIt++);
               if (!reference)
                  continue;

               success = bool(object.value);
               if (!success)
               {
                  inputs.logger.log("Failed to convert a CPU object to GPU of type %s!", ILogger::ELL_ERROR, system::to_string(reference->getAssetType()));
                  return;
               }
            }
         };
         check.template operator()<ICPUBuffer>(tmpBuffers);
         check.template operator()<ICPUTopLevelAccelerationStructure>({ &tmpTLASes.front().get(), tmpTLASes.size() });
         if (!success)
            return nullptr;
      }

      // convert
      {
         smart_refctd_ptr<CAssetConverter::SConvertParams::scratch_for_device_AS_build_t> scratchAlloc;
         {
            constexpr auto scratchUsages = IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | IGPUBuffer::EUF_STORAGE_BUFFER_BIT;

            constexpr uint16_t MaxAlignment      = 256;
            constexpr uint64_t MinAllocationSize = 1024;
            const auto         scratchSize       = core::alignUp(hlsl::max(reservation.getMaxASBuildScratchSize(false), MinAllocationSize), MaxAlignment);

            auto scratchBuffer = device->createBuffer({ { .size = scratchSize, .usage = scratchUsages } });

            auto reqs = scratchBuffer->getMemoryReqs();
            reqs.memoryTypeBits &= device->getPhysicalDevice()->getDirectVRAMAccessMemoryTypeBits();

            auto allocation = device->allocate(reqs, scratchBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
            allocation.memory->map({ .offset = 0, .length = reqs.size });

            scratchAlloc =
               make_smart_refctd_ptr<CAssetConverter::SConvertParams::scratch_for_device_AS_build_t>(SBufferRange<video::IGPUBuffer> { 0ull, scratchSize, std::move(scratchBuffer) }, core::allocator<uint8_t>(), MaxAlignment, MinAllocationSize);
         }

         constexpr auto CompBufferCount = 2;

         std::array<smart_refctd_ptr<IGPUCommandBuffer>, CompBufferCount>     compBufs     = {};
         std::array<IQueue::SSubmitInfo::SCommandBufferInfo, CompBufferCount> compBufInfos = {};
         {
            constexpr auto RequiredFlags = IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT;
            auto           pool          = device->createCommandPool(m_creation.computeQueue->getFamilyIndex(), RequiredFlags);
            if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, compBufs))
            {
               inputs.logger.log("Failed to create Command Buffers for the Compute Queue!", ILogger::ELL_ERROR);
               return nullptr;
            }
            compBufs.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            for (auto i = 0; i < CompBufferCount; i++)
               compBufInfos[i].cmdbuf = compBufs[i].get();
         }
         auto compSema = device->createSemaphore(0u);

         // TODO: `SIntendedSubmitInfo transfer` as well, because of images
         SIntendedSubmitInfo compute   = {};
         compute.queue                 = m_creation.computeQueue;
         compute.scratchCommandBuffers = compBufInfos;
         compute.scratchSemaphore = { .semaphore = compSema.get(), .value = 0u, .stageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT | PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_COPY_BIT | PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT };
         struct MyParams final : CAssetConverter::SConvertParams
         {
            inline uint32_t getFinalOwnerQueueFamily(const IGPUBuffer* buffer, const core::blake3_hash_t& createdFrom) override { return finalUser; }
            inline uint32_t getFinalOwnerQueueFamily(const IGPUAccelerationStructure* image, const core::blake3_hash_t& createdFrom) override { return finalUser; }

            uint8_t finalUser;
         } cvtParam                       = {};
         cvtParam.utilities               = m_creation.utilities.get();
         cvtParam.compute                 = &compute;
         cvtParam.scratchForDeviceASBuild = scratchAlloc.get();
         cvtParam.finalUser               = m_creation.graphicsQueue->getFamilyIndex();

         auto future = reservation.convert(cvtParam);
         // release the memory
         {
            for (auto& tmpTLAS : tmpTLASes)
            {
               IAsset* const asAsset = tmpTLAS.get();
               IPreHashed::discardDependantsContents({ &asAsset, 1 });
            }
            tmpTLASes.clear();
            tmpBuffers = {};
         }
         if (future.copy() != IQueue::RESULT::SUCCESS)
         {
            inputs.logger.log("Failed to await `CAssetConverter::SReserveResult::convert(...)` submission semaphore!", ILogger::ELL_ERROR);
            return nullptr;
         }


         const auto buffers = reservation.getGPUObjects<ICPUBuffer>();
         ubo                = buffers[0].value;
         for (uint8_t i = 0; i < SBTCount; i++)
         {
            const auto& buffer       = buffers[i + 1].value;
            auto        setSBTBuffer = [&buffer](SStridedRange<const IGPUBuffer>& stRange) -> void
            {
               stRange.range.buffer = stRange.range.size ? buffer : nullptr;
            };
            // Same slot mapping as the build loop: real modes first, then the Beauty MIS variants.
            auto& sbt         = (i < RenderModeCount) ? params.sbts[i] : params.beautyVariantSbts[i - RenderModeCount];
            sbt.raygen.buffer = buffer;
            setSBTBuffer(sbt.miss);
            setSBTBuffer(sbt.hit);
            setSBTBuffer(sbt.callable);
         }

         const bool success = reservation.moveGPUObjects<ICPUTopLevelAccelerationStructure>(TLASes);
         assert(success);
         params.TLAS = TLASes[0].value;
      }
   }

   // write into DS
   {
      vector<IGPUDescriptorSet::SDescriptorInfo>     infos;
      vector<IGPUDescriptorSet::SWriteDescriptorSet> writes;
      auto* const                                    ds       = params.sceneDS->getDescriptorSet();
      auto                                           addWrite = [&](const uint32_t binding) -> void
      {
         writes.emplace_back() = { .dstSet = ds, .binding = binding, .arrayElement = 0, .count = 1, .info = reinterpret_cast<const IGPUDescriptorSet::SDescriptorInfo*>(infos.size()) };
      };
      addWrite(SceneDSBindings::UBO);
      infos.push_back(SBufferRange<IGPUBuffer> { .offset = 0, .size = sizeof(SSceneUniforms), .buffer = std::move(ubo) });
      // TODO: Envmap
      {
         addWrite(SceneDSBindings::TLASes);
         infos.reserve(infos.size() + TLASes.size());
         for (auto& tlas : TLASes)
            infos.emplace_back().desc = tlas.value;
      }
      // TODO: Samplers
      // TODO: Sampled Images
      // TODO: Envmap PDF
      // TODO: Envmap Warp Map
      for (auto& write : writes)
         write.info = infos.data() + reinterpret_cast<const uint64_t&>(write.info);
      device->updateDescriptorSets(writes, {});
   }

#if 0
	float m_maxAreaLightLuma;
	// Resources used for envmap sampling
	nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_finalEnvmap;
#endif

   //
   if (!params)
   {
      m_creation.logger.log("Failed to create a scene!", ILogger::ELL_ERROR);
      return nullptr;
   }
   auto scene = core::smart_refctd_ptr<CScene>(new CScene(std::move(params)), core::dont_grab);
   // `params` (with its SLightTree) was just moved into the scene, so the probe's
   // borrowed tree pointer set during buffer setup now aims at a moved-from local.
   // Repoint it at the scene-owned tree (stable for the scene's lifetime), without
   // this, every setProbe() after load reads an empty moved-from tree and the
   // per-emitter pdf recompute silently no-ops (probe appears frozen).
   if (scene && (m_probeDebugPdfsMapped || m_nodePdfsMapped))
      m_lightTreeForProbe = &scene->getLightTree();
   return scene;
}


auto CRenderer::render(CSession* session, const STimingScope& timing) -> SSubmit
{
   if (!session || !session->isInitialized())
      return {};
   const auto& sessionParams = session->getConstructionParams();
   auto* const device        = getDevice();

   if (m_frameIx >= SCachedConstructionParams::FramesInFlight)
   {
      const ISemaphore::SWaitInfo cbDonePending[] = { { .semaphore = m_construction.semaphore.get(), .value = m_frameIx + 1 - SCachedConstructionParams::FramesInFlight } };
      if (device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
         return {};
   }
   const auto resourceIx = m_frameIx % SCachedConstructionParams::FramesInFlight;

   auto* const cb = m_construction.commandBuffers[resourceIx].get();
   cb->getPool()->reset();
   if (!cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
      return {};

   if (timing.queryPool)
   {
      // Vulkan requires queries to be reset before each write. Two single-slot
      // resets handle both adjacent and non-adjacent (start, end) indices.
      cb->resetQueryPool(timing.queryPool, timing.startQueryIdx, 1);
      cb->resetQueryPool(timing.queryPool, timing.endQueryIdx, 1);
      cb->writeTimestamp(PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, timing.queryPool, timing.startQueryIdx);
   }

   const auto* const scene            = session->getConstructionParams().scene.get();
   const auto        mode             = sessionParams.mode;
   const auto&       sessionResources = session->getActiveResources();
   // m_misMode + m_useAliasNEE only affect Beauty; other modes ignore them (getPipeline returns the
   // mode's pipeline). Both pick which compiled Beauty variant binds.
   const auto* const pipeline = scene->getPipeline(mode, m_misMode, m_useAliasNEE);

   bool success;
   // push constants
   {
      // Bench mode forces a fresh-start frame so the shader always does work,
      // independent of how much sample accumulation the session built up.
      SSensorDynamics dynForRender = sessionResources.currentSensorState;
      if (timing.forceFreshFrame)
         dynForRender.keepAccumulating = 0;
      else if (timing.forceAccumulate)
         dynForRender.keepAccumulating = 1;
      if (timing.maxSPPOverride != 0u)
         dynForRender.maxSPP = timing.maxSPPOverride;

      switch (mode)
      {
         case CSession::RenderMode::Debug:
            {
               SDebugPushConstants pc = { dynForRender };
               success                = cb->pushConstants(pipeline->getLayout(), hlsl::ShaderStage::ESS_ALL_RAY_TRACING, 0, sizeof(pc), &pc);
               break;
            }
         case CSession::RenderMode::Beauty:
            {
               SBeautyPushConstants pc = { .sensorDynamics = dynForRender };
               // TODOs
               pc.__16BitData.rrThroughputWeights = hlsl::promote<hlsl::float16_t3>(hlsl::numeric_limits<hlsl::float16_t>::max); // always pass RR, later LumaConversionCoeffs
               pc.__16BitData.maxSppPerDispatch   = m_maxSppPerDispatch;
               // alias-vs-tree is now a compiled Beauty variant (NBL_NEE_USE_ALIAS), picked via
               // getPipeline(mode, m_misMode, m_useAliasNEE) above, no longer a push constant.
               success = cb->pushConstants(pipeline->getLayout(), hlsl::ShaderStage::ESS_ALL_RAY_TRACING, 0, sizeof(pc), &pc);
               break;
            }
         default:
            getLogger().log("Unimplemented RenderMode::%s !", ILogger::ELL_ERROR, system::to_string(mode).c_str());
            return {};
      }
   }

   const auto& sessionImmutables = sessionResources.immutables;
   // bind pipelines
   success = success && cb->bindRayTracingPipeline(pipeline);
   {
      const IGPUDescriptorSet* sets[2] = { sessionParams.scene->getDescriptorSet(), sessionImmutables.ds.get() };
      success                          = success && cb->bindDescriptorSets(EPBP_RAY_TRACING, pipeline->getLayout(), 0, 2, sets);
   }

   // barrier against previous usages of accumulation targets (so that RMW cycles sync up properly)
   {
      constexpr auto raytracingStages = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT;
      using image_barrier_t           = IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t;
      core::vector<image_barrier_t> barr;
      {
         constexpr image_barrier_t base = {
            .barrier = {
               .dep = {
                  // Any of the images can be read by Debug/Presenter, ideally we should be aware of that and inject it here via a Command Graph
                  // but to keep code decoupled we'll have those subsystems use one more pipeline barrier after their own dispatch
                  .srcStageMask  = raytracingStages,
                  .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
                  .dstStageMask  = raytracingStages,
                  .dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS | ACCESS_FLAGS::SHADER_WRITE_BITS } },
            .subresourceRange = {}
         };
         barr.reserve(SensorDSBindingCounts::AsSampledImages);

         auto enqueueBarrier = [&barr, base](const CSession::SImageWithViews& img) -> void
         {
            auto& out            = barr.emplace_back(base);
            out.image            = img.image.get();
            out.subresourceRange = { .aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT, .levelCount = 1, .layerCount = out.image->getCreationParameters().arrayLayers };
         };
         enqueueBarrier(sessionImmutables.sampleCount);
         enqueueBarrier(sessionImmutables.rwmcCascades);
         // Beauty is a per-frame read-modify-write accumulation target in reference mode.
         enqueueBarrier(sessionImmutables.beauty);
         enqueueBarrier(sessionImmutables.albedo);
         enqueueBarrier(sessionImmutables.normal);
         enqueueBarrier(sessionImmutables.motion);
         enqueueBarrier(sessionImmutables.mask);
      }
      success = cb->pipelineBarrier(asset::EDF_NONE, { .imgBarriers = barr });
   }

   const auto renderSize = sessionParams.uniforms.renderSize;
   success               = success && cb->traceRays(scene->getSBT(mode, m_misMode, m_useAliasNEE), renderSize.x, renderSize.y, sessionParams.type != CSession::sensor_type_e::Env ? 1 : 6);

   if (timing.queryPool)
      cb->writeTimestamp(PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, timing.queryPool, timing.endQueryIdx);

   if (success)
   {
      session->onFrameRendered(m_maxSppPerDispatch);
      return SSubmit(this, cb);
   }
   else
      return {};
}

IQueue::SSubmitInfo::SSemaphoreInfo CRenderer::SSubmit::operator()(std::span<const video::IQueue::SSubmitInfo::SSemaphoreInfo> extraWaits)
{
   if (!cb || !cb->end())
      return {};

   const IQueue::SSubmitInfo::SSemaphoreInfo     rendered[]       = { { .semaphore = renderer->m_construction.semaphore.get(), .value = ++renderer->m_frameIx, .stageMask = stageMask } };
   const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] = { { .cmdbuf = cb } };
   const IQueue::SSubmitInfo                     infos[]          = { { .waitSemaphores = extraWaits, .commandBuffers = commandBuffers, .signalSemaphores = rendered } };
   if (renderer->getCreationParams().graphicsQueue->submit(infos) != IQueue::RESULT::SUCCESS)
   {
      renderer->m_frameIx--;
      return {};
   }
   return rendered[0];
}

} // namespace nbl::this_example