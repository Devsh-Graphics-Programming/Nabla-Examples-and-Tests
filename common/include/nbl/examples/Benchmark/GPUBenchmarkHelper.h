// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_COMMON_GPU_BENCHMARK_HELPER_INCLUDED_
#define _NBL_COMMON_GPU_BENCHMARK_HELPER_INCLUDED_

#include <nabla.h>
#include "nbl/examples/examples.hpp"
#include "nbl/examples/Benchmark/BenchmarkTypes.h"
#include "nbl/asset/utils/CCompilerSet.h"
#include "nbl/asset/utils/IShaderCompiler.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

class GPUBenchmarkHelper
{
public:
   struct InitData
   {
      nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> device;
      nbl::core::smart_refctd_ptr<nbl::system::ILogger>       logger;
      nbl::video::IPhysicalDevice*                            physicalDevice     = nullptr;
      uint32_t                                                computeFamilyIndex = 0;
      nbl::hlsl::uint32_t3                                    dispatchGroupCount = {0, 0, 0};
      uint64_t                                                samplesPerDispatch = 0;
   };

   // One shader source for a benchmark variant. Picks ONE of two paths:
   //   * Precompiled: `precompiledKey` is a SPIRV asset key from CMake-time
   //     NBL_CREATE_NSC_COMPILE_RULES. `defines` is ignored.
   //   * Runtime: `sourcePath` is an .hlsl file resolved against "app_resources",
   //     compiled at load time with `defines` as -D macros. Use this for fast
   //     variant iteration without reconfiguring CMake.
   struct ShaderVariant
   {
      // SMacroDefinition uses string_view; this struct owns the backing strings.
      struct Define
      {
         std::string identifier;
         std::string definition;
      };

      std::string                         sourcePath;
      std::string                         precompiledKey;
      std::vector<Define>                 defines;
      nbl::asset::IShader::E_SHADER_STAGE stage = nbl::asset::IShader::E_SHADER_STAGE::ESS_COMPUTE;

      static ShaderVariant Precompiled(std::string key)
      {
         ShaderVariant v;
         v.precompiledKey = std::move(key);
         return v;
      }
      static ShaderVariant FromSource(std::string path, std::vector<Define> defs = {}, nbl::asset::IShader::E_SHADER_STAGE stage = nbl::asset::IShader::E_SHADER_STAGE::ESS_COMPUTE)
      {
         ShaderVariant v;
         v.sourcePath = std::move(path);
         v.defines    = std::move(defs);
         v.stage      = stage;
         return v;
      }

      bool isRuntime() const { return !sourcePath.empty() && precompiledKey.empty(); }
      bool isPrecompiled() const { return !precompiledKey.empty(); }
   };

   // Logical layout: [warmup x dispatchOne][ts0][bench x dispatchOne][ts1][cooldown x dispatchOne]
   // Warmup/cooldown can be split into shorter submissions and the measured window stays intact.
   // Putting binds inside dispatchOne adds per-iteration cmdbuf overhead that
   // shows up in ps/sample on tight shaders.
   using DispatchFn = std::function<void(nbl::video::IGPUCommandBuffer*)>;

   // Input choice for createBindings(). Output is always implicit BDA.
   enum class InputBuffer : uint8_t
   {
      None,
      BDA,
      SSBO,
      UBO,
   };

   struct BindingsConfig
   {
      size_t      outputBytes       = 0;
      size_t      pushConstantBytes = 0;
      size_t      inputBytes        = 0;
      InputBuffer inputMode         = InputBuffer::None;
   };

   struct Bindings
   {
      nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer>         outputBuf;
      uint64_t                                                    outputAddress = 0;
      nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pipelineLayout;

      nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> inputBuf;
      uint64_t                                            inputAddress = 0; // BDA mode only

      nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> dsLayout;
      nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet>       ds;
   };

   struct PipelineEntry
   {
      nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
      nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout>  layout;
      PipelineStats                                                stats;
      std::string                                                  tag;
   };

   // Common bindOnce body: bind pipeline + upload push constants. Most benches
   // have nothing else in bindOnce; the few that bind descriptor sets too call
   // cb->bindDescriptorSets() before/after this.
   template<typename PC>
   static void defaultBindAndPush(nbl::video::IGPUCommandBuffer* cb, const PipelineEntry& pe, const PC& pc)
   {
      cb->bindComputePipeline(pe.pipeline.get());
      cb->pushConstants(pe.layout.get(), nbl::asset::IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(PC), &pc);
   }

   // Dispatch using m_dispatchGroupCount (the setup-time shape).
   void defaultDispatch(nbl::video::IGPUCommandBuffer* cb) const
   {
      cb->dispatch(m_dispatchGroupCount.x, m_dispatchGroupCount.y, m_dispatchGroupCount.z);
   }

   bool init(const InitData& data)
   {
      m_device             = data.device;
      m_logger             = data.logger;
      m_physicalDevice     = data.physicalDevice;
      m_queue              = m_device->getQueue(data.computeFamilyIndex, 0);
      m_dispatchGroupCount = data.dispatchGroupCount;
      m_samplesPerDispatch = data.samplesPerDispatch;

      m_cmdpool = m_device->createCommandPool(data.computeFamilyIndex,
         nbl::video::IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
      if (!m_cmdpool->createCommandBuffers(nbl::video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_cmdbuf))
      {
         benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_ERROR, "GPUBenchmarkHelper: failed to create cmdbuf");
         return false;
      }

      nbl::video::IQueryPool::SCreationParams qparams = {};
      qparams.queryType                               = nbl::video::IQueryPool::TYPE::TIMESTAMP;
      qparams.queryCount                              = 2;
      qparams.pipelineStatisticsFlags                 = nbl::video::IQueryPool::PIPELINE_STATISTICS_FLAGS::NONE;
      m_queryPool                                     = m_device->createQueryPool(qparams);
      if (!m_queryPool)
      {
         benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_ERROR, "GPUBenchmarkHelper: failed to create timestamp query pool");
         return false;
      }
      return true;
   }

   // Load (precompiled path) or load+compile (runtime path) a variant's SPIRV.
   nbl::core::smart_refctd_ptr<nbl::asset::IShader> loadShader(const ShaderVariant& variant, nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetMgr) const
   {
      using namespace nbl;
      if (!variant.isRuntime() && !variant.isPrecompiled())
      {
         benchLogFmt(m_logger.get(), system::ILogger::ELL_ERROR, "GPUBenchmarkHelper::loadShader: variant has neither sourcePath nor precompiledKey");
         return nullptr;
      }

      asset::IAssetLoader::SAssetLoadParams lp = {};
      lp.logger                                = m_logger.get();

      std::string key;
      if (variant.isPrecompiled())
      {
         lp.workingDirectory = "app_resources";
         key                 = variant.precompiledKey;
      }
      else
      {
         lp.workingDirectory = "";
         key                 = "app_resources/" + variant.sourcePath;
      }
      auto       bundle = assetMgr->getAsset(key, lp);
      const auto assets = bundle.getContents();
      if (assets.empty())
      {
         benchLogFmt(m_logger.get(), system::ILogger::ELL_ERROR, "GPUBenchmarkHelper::loadShader: failed to load '{}'", key);
         return nullptr;
      }
      auto source = asset::IAsset::castDown<asset::IShader>(assets[0]);
      if (!source)
      {
         benchLogFmt(m_logger.get(), system::ILogger::ELL_ERROR, "GPUBenchmarkHelper::loadShader: '{}' is not an IShader asset", key);
         return nullptr;
      }

      if (variant.isPrecompiled())
         return source;

      auto* compilerSet = assetMgr->getCompilerSet();
      auto  compiler    = compilerSet->getShaderCompiler(source->getContentType());
      if (!compiler)
      {
         benchLogFmt(m_logger.get(), system::ILogger::ELL_ERROR, "GPUBenchmarkHelper::loadShader: no compiler for content type of '{}'", variant.sourcePath);
         return nullptr;
      }

      std::vector<asset::IShaderCompiler::SMacroDefinition> wireDefines;
      wireDefines.reserve(variant.defines.size());
      for (const auto& d : variant.defines)
         wireDefines.push_back({d.identifier, d.definition});

      asset::IShaderCompiler::SCompilerOptions options = {};
      options.stage                                    = variant.stage;
      options.preprocessorOptions.targetSpirvVersion   = m_device->getPhysicalDevice()->getLimits().spirvVersion;
      options.preprocessorOptions.sourceIdentifier     = source->getFilepathHint();
      options.preprocessorOptions.logger               = m_logger.get();
      options.preprocessorOptions.includeFinder        = compiler->getDefaultIncludeFinder();
      options.preprocessorOptions.extraDefines         = {wireDefines.data(), wireDefines.size()};

      auto spirv = compilerSet->compileToSPIRV(source.get(), options);
      if (!spirv)
         benchLogFmt(m_logger.get(), system::ILogger::ELL_ERROR, "GPUBenchmarkHelper::loadShader: runtime compile failed for '{}'", variant.sourcePath);
      return spirv;
   }

   nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> allocateDeviceLocalBuffer(nbl::video::IGPUBuffer::SCreationParams bp, const char* label,
      nbl::video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS allocFlags = nbl::video::IDeviceMemoryAllocation::EMAF_NONE)
   {
      auto buf  = m_device->createBuffer(std::move(bp));
      auto reqs = buf->getMemoryReqs();
      reqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
      auto alloc = m_device->allocate(reqs, buf.get(), allocFlags);
      if (!alloc.isValid())
         benchLogFmt(m_logger.get(), nbl::system::ILogger::ELL_ERROR, "GPUBenchmarkHelper: failed to allocate {}", label);
      return buf;
   }

   struct SingleBindingDS
   {
      nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> layout;
      nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet>       set;
   };

   SingleBindingDS createSingleBindingDS(
      nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> buffer,
      nbl::asset::IDescriptor::E_TYPE                     type    = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
      uint32_t                                            binding = 0,
      nbl::hlsl::ShaderStage                              stages  = nbl::hlsl::ShaderStage::ESS_COMPUTE)
   {
      using namespace nbl;
      const size_t bufferBytes = buffer->getSize();

      video::IGPUDescriptorSetLayout::SBinding b = {
         .binding     = binding,
         .type        = type,
         .createFlags = video::IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
         .stageFlags  = stages,
         .count       = 1,
      };
      SingleBindingDS out;
      out.layout = m_device->createDescriptorSetLayout({&b, 1});
      auto pool  = m_device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, {&out.layout.get(), 1});
      out.set    = pool->createDescriptorSet(core::smart_refctd_ptr(out.layout));

      video::IGPUDescriptorSet::SDescriptorInfo info  = {};
      info.desc                                       = std::move(buffer);
      info.info.buffer                                = {.offset = 0, .size = bufferBytes};
      video::IGPUDescriptorSet::SWriteDescriptorSet w = {
         .dstSet       = out.set.get(),
         .binding      = binding,
         .arrayElement = 0,
         .count        = 1,
         .info         = &info,
      };
      m_device->updateDescriptorSets({&w, 1}, {});
      return out;
   }

   nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> createOutputBuffer(
      size_t                                                       bytes,
      nbl::core::bitflag<nbl::video::IGPUBuffer::E_USAGE_FLAGS>    extraUsage = nbl::video::IGPUBuffer::E_USAGE_FLAGS::EUF_NONE,
      nbl::video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS allocFlags = nbl::video::IDeviceMemoryAllocation::EMAF_NONE)
   {
      nbl::video::IGPUBuffer::SCreationParams bp = {};
      bp.size                                    = bytes;
      bp.usage                                   = nbl::core::bitflag(nbl::video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | extraUsage;
      return allocateDeviceLocalBuffer(std::move(bp), "output buffer", allocFlags);
   }

   // Buffer must have been created with EUF_TRANSFER_DST_BIT.
   void submitFillZero(nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> buf, size_t bytes) const
   {
      nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> initCmdbuf;
      m_cmdpool->createCommandBuffers(nbl::video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &initCmdbuf);
      initCmdbuf->begin(nbl::video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
      const nbl::asset::SBufferRange<nbl::video::IGPUBuffer> range = {.offset = 0, .size = bytes, .buffer = std::move(buf)};
      initCmdbuf->fillBuffer(range, 0u);
      initCmdbuf->end();

      const nbl::video::IQueue::SSubmitInfo::SCommandBufferInfo cmds[] = {{.cmdbuf = initCmdbuf.get()}};
      nbl::video::IQueue::SSubmitInfo                           submit = {};
      submit.commandBuffers                                            = cmds;
      m_queue->submit({&submit, 1u});
      m_device->waitIdle();
   }

   nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> createInputBufferZeroFilled(size_t bytes)
   {
      auto buf = createOutputBuffer(bytes, nbl::video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
      if (buf)
         submitFillZero(buf, bytes);
      return buf;
   }

   // BDA buffer staged into device-local VRAM via IUtilities.
   nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> createBdaBuffer(const void* srcData, size_t bytes)
   {
      using namespace nbl;
      if (!m_utils)
         m_utils = video::IUtilities::create(core::smart_refctd_ptr(m_device), core::smart_refctd_ptr(m_logger));

      video::IGPUBuffer::SCreationParams bp = {};
      bp.size                               = bytes;
      bp.usage                              = core::bitflag(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
      core::smart_refctd_ptr<video::IGPUBuffer> buf;
      auto                                      future = m_utils->createFilledDeviceLocalBufferOnDedMem(
         video::SIntendedSubmitInfo {.queue = m_queue}, std::move(bp), srcData);
      future.move_into(buf);
      return buf;
   }

   uint32_t createPipeline(const ShaderVariant&                        variant,
      nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager>           assetMgr,
      size_t                                                           pushConstantSize,
      std::string                                                      tag      = "",
      nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> dsLayout = nullptr)
   {
      using namespace nbl;
      const uint32_t idx = uint32_t(m_pipelines.size());
      m_pipelines.push_back({.tag = tag});
      PipelineEntry& slot = m_pipelines.back();

      const asset::SPushConstantRange pcRange = {
         .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
         .offset     = 0,
         .size       = uint32_t(pushConstantSize),
      };
      auto layout = dsLayout
         ? m_device->createPipelineLayout({&pcRange, 1}, core::smart_refctd_ptr(dsLayout))
         : m_device->createPipelineLayout({&pcRange, 1});
      if (!layout)
      {
         benchLogFmt(m_logger.get(), system::ILogger::ELL_ERROR, "createPipeline({}): pipeline layout creation failed", tag);
         return idx;
      }

      auto source = loadShader(variant, std::move(assetMgr));
      auto shader = source ? m_device->compileShader({.source = source.get()}) : nullptr;
      if (!shader)
      {
         benchLogFmt(m_logger.get(), system::ILogger::ELL_ERROR, "createPipeline({}): shader load/compile failed", tag);
         return idx;
      }

      video::IGPUComputePipeline::SCreationParams pp = {};
      pp.layout                                      = layout.get();
      pp.shader.shader                               = shader.get();
      pp.shader.entryPoint                           = "main";
      if (m_device->getEnabledFeatures().pipelineExecutableInfo)
         pp.flags |= video::IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS | video::IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;

      core::smart_refctd_ptr<video::IGPUComputePipeline> pipeline;
      if (!m_device->createComputePipelines(nullptr, {&pp, 1}, &pipeline) || !pipeline)
      {
         benchLogFmt(m_logger.get(), system::ILogger::ELL_ERROR, "createPipeline({}): createComputePipelines failed", tag);
         return idx;
      }

      if (m_device->getEnabledFeatures().pipelineExecutableInfo)
      {
         auto infos     = pipeline->getExecutableInfo();
         slot.stats.raw = nbl::system::to_string(infos);

         uint64_t vgpr = 0, sgpr = 0;
         for (const auto& info : infos)
         {
            if (info.subgroupSize)
               slot.stats.subgroupSize = std::max<uint32_t>(slot.stats.subgroupSize, info.subgroupSize);
            for (const auto& stat : info.structuredStatistics)
               matchStat(stat, slot.stats, vgpr, sgpr);
         }
         // AMD-style drivers expose VGPR/SGPR separately without a combined
         // register count, so fall back to the sum.
         if (slot.stats.registerCount == 0 && (vgpr || sgpr))
            slot.stats.registerCount = vgpr + sgpr;

         if (!slot.stats.raw.empty())
            benchLogFmt(m_logger.get(), system::ILogger::ELL_PERFORMANCE, "{} pipeline executable report:\n{}", tag, slot.stats.raw);
      }

      slot.layout   = std::move(layout);
      slot.pipeline = std::move(pipeline);
      return idx;
   }

   Bindings createBindings(const BindingsConfig& cfg)
   {
      using namespace nbl;
      Bindings out;

      out.outputBuf     = createOutputBuffer(cfg.outputBytes, video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT, video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
      out.outputAddress = out.outputBuf->getDeviceAddress();

      if (cfg.inputMode != InputBuffer::None && cfg.inputBytes > 0)
      {
         const bool useBDA  = cfg.inputMode == InputBuffer::BDA;
         const bool useUBO  = cfg.inputMode == InputBuffer::UBO;
         const bool useSSBO = cfg.inputMode == InputBuffer::SSBO;

         video::IGPUBuffer::SCreationParams bp = {};
         bp.size                               = cfg.inputBytes;
         bp.usage                              = core::bitflag(video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
         if (useBDA || useSSBO)
            bp.usage |= video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
         if (useBDA)
            bp.usage |= video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
         if (useUBO)
            bp.usage |= video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT;

         out.inputBuf = allocateDeviceLocalBuffer(std::move(bp), "input buffer",
            useBDA ? video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT : video::IDeviceMemoryAllocation::EMAF_NONE);

         if (useBDA)
            out.inputAddress = out.inputBuf->getDeviceAddress();

         submitFillZero(out.inputBuf, cfg.inputBytes);

         if (useSSBO || useUBO)
         {
            video::IGPUDescriptorSetLayout::SBinding b = {
               .binding     = 0,
               .type        = useSSBO ? asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER : asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
               .createFlags = video::IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
               .stageFlags  = nbl::hlsl::ShaderStage::ESS_COMPUTE,
               .count       = 1,
            };
            out.dsLayout = m_device->createDescriptorSetLayout({&b, 1});

            auto pool = m_device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, {&out.dsLayout.get(), 1});
            out.ds    = pool->createDescriptorSet(core::smart_refctd_ptr(out.dsLayout));

            video::IGPUDescriptorSet::SDescriptorInfo info  = {};
            info.desc                                       = core::smart_refctd_ptr(out.inputBuf);
            info.info.buffer                                = {.offset = 0, .size = cfg.inputBytes};
            video::IGPUDescriptorSet::SWriteDescriptorSet w = {
               .dstSet       = out.ds.get(),
               .binding      = 0,
               .arrayElement = 0,
               .count        = 1,
               .info         = &info,
            };
            m_device->updateDescriptorSets({&w, 1}, {});
         }
      }

      {
         const asset::SPushConstantRange pc = {
            .stageFlags = nbl::hlsl::ShaderStage::ESS_COMPUTE,
            .offset     = 0,
            .size       = uint32_t(cfg.pushConstantBytes),
         };
         std::span<const asset::SPushConstantRange> pcRange = cfg.pushConstantBytes > 0 ? std::span<const asset::SPushConstantRange>(&pc, 1) : std::span<const asset::SPushConstantRange> {};

         if (out.dsLayout)
            out.pipelineLayout = m_device->createPipelineLayout(pcRange, core::smart_refctd_ptr(out.dsLayout));
         else
            out.pipelineLayout = m_device->createPipelineLayout(pcRange);
      }

      return out;
   }

   struct BdaBuffer
   {
      nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> buf;
      uint64_t                                            address = 0;
   };

   BdaBuffer createBdaOutputBuffer(size_t bytes)
   {
      BdaBuffer out;
      out.buf     = createOutputBuffer(bytes, nbl::video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT, nbl::video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
      out.address = out.buf ? out.buf->getDeviceAddress() : 0;
      return out;
   }

   // Auto-sizes the dispatch count so the measured window covers ~targetBudgetMs
   // of GPU work. Pilots with a small N, then either scales to the budget or
   // doubles when the pilot is too noisy (sub-millisecond) to extrapolate.
   //
   // `samples` controls jitter robustness: values >1 take K independent
   // budget-sized timing windows and return the MEDIAN window, costing ~K *
   // targetBudgetMs of wall time. Median (not min) is used because GPU
   // measurement noise can be two-sided in practice. 
   TimingResult runTimedBudgeted(uint32_t warmupDispatches, uint64_t targetBudgetMs, const DispatchFn& bindOnce, const DispatchFn& dispatchOne, uint32_t samples)
   {
      const uint64_t     targetBudgetNs = targetBudgetMs * 1'000'000ull;
      constexpr uint32_t kPilotN        = 64;
      constexpr uint32_t kMaxN          = 1u << 24; // safety cap for ultra-fast shaders
      uint32_t           dispatchesPerSubmit = 1u;
      TimingResult       r                   = runTimed(warmupDispatches, kPilotN, bindOnce, dispatchOne, dispatchesPerSubmit);
      dispatchesPerSubmit                    = estimateDispatchesPerSubmit(r, kPilotN);
      uint32_t           lastN          = kPilotN;
      while (r.elapsed_ns > targetBudgetNs && lastN > 1u)
      {
         const double scale = double(targetBudgetNs) / r.elapsed_ns;
         uint32_t     nextN = uint32_t(std::max(1.0, std::floor(double(lastN) * scale)));
         if (nextN >= lastN)
            nextN = lastN - 1u;

         r                   = runTimed(warmupDispatches, nextN, bindOnce, dispatchOne, dispatchesPerSubmit);
         dispatchesPerSubmit = estimateDispatchesPerSubmit(r, nextN);
         lastN               = nextN;
      }

      while (r.elapsed_ns < targetBudgetNs && lastN < kMaxN)
      {
         uint32_t nextN;
         if (r.elapsed_ns > 1'000'000ull) // > 1 ms, stable enough to scale
         {
            const double scale = double(targetBudgetNs) / double(r.elapsed_ns);
            nextN              = uint32_t(std::min<double>(double(kMaxN), std::ceil(double(lastN) * scale)));
         }
         else
         {
            nextN = std::min(kMaxN, lastN * 2);
         }
         if (nextN <= lastN)
            break; // converged
         r                   = runTimed(warmupDispatches, nextN, bindOnce, dispatchOne, dispatchesPerSubmit);
         dispatchesPerSubmit = estimateDispatchesPerSubmit(r, nextN);
         lastN               = nextN;
      }

      if (samples <= 1)
         return r;

      // Reuse the convergence's final measurement as one of the K samples
      // (it's already a budget-sized window at lastN). Run K-1 more at the
      // same N. All windows measure the same dispatch count, so the per-window
      // elapsed_ns values are directly comparable.
      std::vector<double> ns;
      ns.reserve(samples);
      ns.push_back(r.elapsed_ns);
      for (uint32_t i = 1; i < samples; ++i)
      {
         const TimingResult ri = runTimed(warmupDispatches, lastN, bindOnce, dispatchOne, dispatchesPerSubmit);
         ns.push_back(ri.elapsed_ns);
      }
      std::sort(ns.begin(), ns.end());

      // Outlier rejection: GPU jitter is usually a one-sided spike
      const double median  = ns[ns.size() / 2];
      const double dLow    = median - ns.front();
      const double dHigh   = ns.back() - median;
      const double dCloser = std::min(dLow, dHigh);
      const double dFar    = std::max(dLow, dHigh);
      size_t       lo      = 0;
      size_t       hi      = ns.size();
      if (dCloser > 0.0 && dFar > 2.0 * dCloser)
      {
         if (dHigh > dLow)
            --hi; // top sample is the spike
         else
            ++lo; // bottom sample is the spike (rare on GPU but cheap to handle)
      }

      double sum = 0.0;
      for (size_t i = lo; i < hi; ++i)
         sum += ns[i];
      const double resultNs = sum / double(hi - lo);

      TimingResult m {};
      m.elapsed_ns     = resultNs;
      m.totalSamples   = uint64_t(lastN) * m_samplesPerDispatch;
      m.ps_per_sample  = m.totalSamples ? resultNs * 1e3 / double(m.totalSamples) : 0.0;
      m.gsamples_per_s = resultNs > 0.0 ? double(m.totalSamples) / resultNs : 0.0;
      m.ms_total       = resultNs * 1e-6;
      return m;
   }

   TimingResult runTimed(uint32_t warmupDispatches, uint32_t benchDispatches, const DispatchFn& bindOnce, const DispatchFn& dispatchOne, uint32_t maxDispatchesPerSubmit)
   {
      if (m_device->waitIdle() != nbl::video::IQueue::RESULT::SUCCESS)
         return {};

      const uint32_t cooldownDispatches = warmupDispatches;

      if (!runUntimedDispatches(warmupDispatches, bindOnce, dispatchOne, maxDispatchesPerSubmit))
         return {};

      double   elapsedNs = 0.0;
      uint32_t remaining = benchDispatches;
      while (remaining > 0u)
      {
         const uint32_t batch = std::min(remaining, std::max(1u, maxDispatchesPerSubmit));

         m_cmdbuf->reset(nbl::video::IGPUCommandBuffer::RESET_FLAGS::NONE);
         m_cmdbuf->begin(nbl::video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
         m_cmdbuf->resetQueryPool(m_queryPool.get(), 0, 2);

         if (bindOnce)
            bindOnce(m_cmdbuf.get());

         m_cmdbuf->writeTimestamp(nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, m_queryPool.get(), 0);
         for (uint32_t i = 0u; i < batch; ++i)
            dispatchOne(m_cmdbuf.get());
         m_cmdbuf->writeTimestamp(nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, m_queryPool.get(), 1);
         m_cmdbuf->end();

         if (!submitAndWait())
            return {};

         uint64_t   timestamps[2] = {};
         const auto flags         = nbl::core::bitflag(nbl::video::IQueryPool::RESULTS_FLAGS::_64_BIT) | nbl::core::bitflag(nbl::video::IQueryPool::RESULTS_FLAGS::WAIT_BIT);
         if (!m_device->getQueryPoolResults(m_queryPool.get(), 0, 2, timestamps, sizeof(uint64_t), flags))
            return {};

         const double timestampPeriod = double(m_physicalDevice->getLimits().timestampPeriodInNanoSeconds);
         elapsedNs += double(timestamps[1] - timestamps[0]) * timestampPeriod;
         remaining -= batch;
      }

      if (!runUntimedDispatches(cooldownDispatches, bindOnce, dispatchOne, maxDispatchesPerSubmit))
         return {};

      TimingResult r {};
      r.elapsed_ns                 = elapsedNs;
      r.totalSamples               = uint64_t(benchDispatches) * m_samplesPerDispatch;
      r.ps_per_sample              = r.totalSamples ? r.elapsed_ns * 1e3 / double(r.totalSamples) : 0.0;
      r.gsamples_per_s             = r.elapsed_ns > 0.0 ? double(r.totalSamples) / r.elapsed_ns : 0.0;
      r.ms_total                   = r.elapsed_ns * 1e-6;
      return r;
   }

protected:
   std::vector<PipelineEntry> m_pipelines;

private:
   // Soft target for one queue submit, estimated from timings on the current GPU.
   // Benchmark budgets still control measured work. This only chunks submits.
   static constexpr double SubmitChunkTargetNs = 250'000'000.0;

   static uint32_t estimateDispatchesPerSubmit(const TimingResult& r, uint32_t dispatches)
   {
      if (dispatches == 0u || r.elapsed_ns <= 0.0)
         return 1u;

      const double nsPerDispatch = r.elapsed_ns / double(dispatches);
      if (nsPerDispatch <= 0.0)
         return 1u;

      const double maxDispatches = std::floor(SubmitChunkTargetNs / nsPerDispatch);
      return uint32_t(std::clamp(maxDispatches, 1.0, double(std::numeric_limits<uint32_t>::max())));
   }

   bool submitAndWait()
   {
      auto semaphore = m_device->createSemaphore(0u);
      if (!semaphore)
         return false;

      const nbl::video::IQueue::SSubmitInfo::SCommandBufferInfo cmds[] = {{.cmdbuf = m_cmdbuf.get()}};
      const nbl::video::IQueue::SSubmitInfo::SSemaphoreInfo     done[] = {
         {.semaphore = semaphore.get(), .value = 1u, .stageMask = nbl::asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS}};
      nbl::video::IQueue::SSubmitInfo submit = {};
      submit.commandBuffers                  = cmds;
      submit.signalSemaphores                = done;
      if (m_queue->submit({&submit, 1u}) != nbl::video::IQueue::RESULT::SUCCESS)
         return false;

      const nbl::video::ISemaphore::SWaitInfo wait[] = {{.semaphore = semaphore.get(), .value = 1u}};
      return m_device->blockForSemaphores(wait) == nbl::video::ISemaphore::WAIT_RESULT::SUCCESS;
   }

   bool runUntimedDispatches(uint32_t dispatches, const DispatchFn& bindOnce, const DispatchFn& dispatchOne, uint32_t maxDispatchesPerSubmit)
   {
      while (dispatches > 0u)
      {
         const uint32_t batch = std::min(dispatches, std::max(1u, maxDispatchesPerSubmit));

         m_cmdbuf->reset(nbl::video::IGPUCommandBuffer::RESET_FLAGS::NONE);
         m_cmdbuf->begin(nbl::video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
         if (bindOnce)
            bindOnce(m_cmdbuf.get());
         for (uint32_t i = 0u; i < batch; ++i)
            dispatchOne(m_cmdbuf.get());
         m_cmdbuf->end();

         if (!submitAndWait())
            return false;
         dispatches -= batch;
      }
      return true;
   }

   static void matchStat(const nbl::video::IGPUPipelineBase::SExecutableStatistic& stat, PipelineStats& out, uint64_t& vgpr, uint64_t& sgpr)
   {
      const uint64_t v = stat.asUint();

      auto contains = [&](std::string_view kw)
      {
         const auto it = std::ranges::search(stat.name, kw,
            [&](char a, char b)
            { return std::tolower(a) == std::tolower(b); })
                            .begin();
         return it != stat.name.end();
      };

      // Order matters: more specific keys first.

      if (contains("subgroup size") || contains("subgroupsize") || contains("warp size") || contains("wave size"))
         out.subgroupSize = std::max<uint32_t>(out.subgroupSize, uint32_t(v));

      else if (contains("vgpr"))
         vgpr = std::max(vgpr, v);
      else if (contains("sgpr"))
         sgpr = std::max(sgpr, v);
      else if (contains("register"))
         out.registerCount = std::max(out.registerCount, v);

      else if (contains("binary size") || contains("binarysize") || contains("codesize") || contains("code size") || contains("isa size"))
         out.codeSizeBytes = std::max(out.codeSizeBytes, v);
      else if (contains("instructioncount") || contains("instruction count") || contains("numinstructions"))
         out.codeSizeBytes = std::max(out.codeSizeBytes, v); // proxy when no byte size

      else if (contains("shared memory") || contains("sharedmemory") || contains("groupshared") || contains("lds"))
         out.sharedMemBytes = std::max(out.sharedMemBytes, v);

      else if (contains("stack size") || contains("stacksize"))
         out.stackBytes = std::max(out.stackBytes, v);

      else if (contains("local memory") || contains("localmemory") || contains("scratch") || contains("private memory") || contains("privatememory") || contains("stack"))
         out.privateMemBytes = std::max(out.privateMemBytes, v);

      // Vendor-specific stats
      // get a structured copy so JSON round-trips the right numeric type.
      else
         out.unknowns.push_back(stat);
   }

   nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>    m_device;
   nbl::core::smart_refctd_ptr<nbl::system::ILogger>          m_logger;
   nbl::video::IPhysicalDevice*                               m_physicalDevice = nullptr;
   nbl::video::IQueue*                                        m_queue          = nullptr;
   nbl::hlsl::uint32_t3                                       m_dispatchGroupCount {};
   uint64_t                                                   m_samplesPerDispatch = 0;
   nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>   m_cmdpool;
   nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_cmdbuf;
   nbl::core::smart_refctd_ptr<nbl::video::IQueryPool>        m_queryPool;
   nbl::core::smart_refctd_ptr<nbl::video::IUtilities>        m_utils; // lazy, only built on first createBdaBuffer call
};

#endif
