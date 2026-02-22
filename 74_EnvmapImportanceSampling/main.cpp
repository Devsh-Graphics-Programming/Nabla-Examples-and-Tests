// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/this_example/builtin/build/spirv/keys.hpp"
#include "nbl/examples/examples.hpp"

#include "nbl/core/sampling/EnvmapSampler.h"
#include "nbl/core/hash/blake.h"

#include "nlohmann/json.hpp"
#include "argparse/argparse.hpp"

#include "app_resources/common.hlsl"

using json = nlohmann::json;

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;
using namespace nbl::examples;

namespace
{
	template<core::StringLiteral ShaderKey>
  smart_refctd_ptr<IShader> loadPrecompiledShader(ILogicalDevice* device, IAssetManager* assetManager, ILogger* logger)
  {
    IAssetLoader::SAssetLoadParams lp = {};
    lp.logger = logger;
    lp.workingDirectory = "app_resources";

    auto key = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(device);
    auto assetBundle = assetManager->getAsset(key.data(), lp);
    const auto assets = assetBundle.getContents();
    if (assets.empty())
      return nullptr;

    auto shader = IAsset::castDown<IShader>(assets[0]);
    return shader;
  };

  template<typename T>
  bool checkEq(T a, T b, float32_t eps = 1e-4)
  {
      T _a = hlsl::max(hlsl::abs(a), hlsl::promote<T>(1e-5));
      T _b = hlsl::max(hlsl::abs(b), hlsl::promote<T>(1e-5));
      return nbl::hlsl::all<hlsl::vector<bool, vector_traits<T>::Dimension> >(nbl::hlsl::max<T>(_a / _b, _b / _a) <= hlsl::promote<T>(1 + eps));
  }
}

class EnvmapImportanceSampleApp final : public application_templates::BasicMultiQueueApplication, public BuiltinResourcesApplication
{
		using device_base_t = application_templates::BasicMultiQueueApplication;
		using asset_base_t = BuiltinResourcesApplication;
		using clock_t = std::chrono::steady_clock;
		using perf_clock_resolution_t = std::chrono::milliseconds;

		constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);
		constexpr static inline std::string_view DefaultImagePathsFile = "../imagesTestList.txt";

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		inline EnvmapImportanceSampleApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}
		
		virtual bool isComputeOnly() const {return false;}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
      core::blake3_hasher hasher();
			argparse::ArgumentParser program("Envmap Importance Sampling Test");

      if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
        return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			// get custom input list of files to execute the program with
			system::path m_loadCWD = DefaultImagePathsFile;

			if (!m_testPathsFile.is_open())
				m_testPathsFile = std::ifstream(m_loadCWD);

			if (!m_testPathsFile.is_open())
				return logFail("Could not open the test paths file");

			m_logger->log("Connected \"%s\" input test list!", ILogger::ELL_INFO, m_loadCWD.string().c_str());
			m_loadCWD = m_loadCWD.parent_path();

			
			const auto* queue = getGraphicsQueue();

      {
        smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&m_cmdbuf,1}))
        {
          m_logger->log("Failed to create command buffer", ILogger::ELL_ERROR);
          return false;
        }
      }

      smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
      {
        auto defaultSampler = m_device->createSampler({
          .TextureWrapU = ETC_CLAMP_TO_EDGE,
          .TextureWrapV = ETC_CLAMP_TO_EDGE,
          .TextureWrapW = ETC_CLAMP_TO_EDGE,
          .MinFilter = ISampler::ETF_NEAREST,
          .MaxFilter = ISampler::ETF_NEAREST,
          .AnisotropicFilter = 0
        });

        const IGPUDescriptorSetLayout::SBinding bindings[] = {
          {
            .binding = 0,
            .type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
            .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
            .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
            .count = 1,
            .immutableSamplers = &defaultSampler
          },
          {
            .binding = 1,
            .type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
            .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
            .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
            .count = 1,
            .immutableSamplers = &defaultSampler
          },
        };
        dsLayout = m_device->createDescriptorSetLayout(bindings);
        if (!dsLayout)
        {
          m_logger->log("Failed to Create Descriptor Layout", ILogger::ELL_ERROR);
          return false;
        }
        asset::SPushConstantRange pcRange = {
          .stageFlags = hlsl::ESS_COMPUTE,
          .offset = 0,
          .size = sizeof(STestPushConstants)
        };
        const auto pipelineLayout = m_device->createPipelineLayout({ &pcRange, 1 }, dsLayout);

        const auto shader = loadPrecompiledShader<"test">(m_device.get(), m_assetMgr.get(), m_logger.get());

        video::IGPUComputePipeline::SCreationParams pipelineParams = {
          .layout = pipelineLayout.get(),
          .shader = {
            .shader = shader.get(),
            .entryPoint = "main",
          }
        };

        if (!m_device->createComputePipelines(nullptr, { &pipelineParams, 1 }, &m_pipeline))
        {
          m_logger->log("Fail to create test pipeline", ILogger::ELL_ERROR);
          return false;
        }

        const auto dsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, pipelineLayout->getDescriptorSetLayouts());
        
        m_descriptorSet = dsPool->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(pipelineLayout->getDescriptorSetLayouts()[0]));

        auto downStreamingBuffer = m_utils->getDefaultDownStreamingBuffer();
        std::chrono::steady_clock::time_point waitTill(std::chrono::years(45));
        uint32_t outputSize = sizeof(test_sample_t) * m_sampleCount;
        m_outputOffset = downStreamingBuffer->invalid_value;
        const auto& deviceLimits = m_device->getPhysicalDevice()->getLimits();
        const uint32_t alignment = core::max(deviceLimits.nonCoherentAtomSize,alignof(float));
        downStreamingBuffer->multi_allocate(waitTill, 1, &m_outputOffset, &outputSize, &alignment);

        m_scratchSemaphore = m_device->createSemaphore(0);
        if (!m_scratchSemaphore)
        {
          logFail("Could not create Scratch Semaphore");
          return false;
        }
        m_scratchSemaphore->setObjectDebugName("Scratch Semaphore");

        m_semaphore = m_device->createSemaphore(0);
        if (!m_semaphore)
        {
          logFail("Could not create Scratch Semaphore");
          return false;
        }
        m_semaphore->setObjectDebugName("Semaphore");
        m_timelineValue = 0;

        // now convert
        m_intendedSubmit.queue = getGraphicsQueue();
        // wait for nothing before upload
        m_intendedSubmit.waitSemaphores = {};
        m_intendedSubmit.prevCommandBuffers = {};
        // fill later
        m_intendedSubmit.scratchCommandBuffers = {};
        m_intendedSubmit.scratchSemaphore = {
          .semaphore = m_scratchSemaphore.get(),
          .value = 0,
          .stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
        };

        std::string nextPath;
        while (std::getline(m_testPathsFile,nextPath))
        {
          if (nextPath!="" && nextPath[0]!=';')
          {
            m_cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

            // load the image view
            system::path filename, extension;
            const core::smart_refctd_ptr<video::IGPUImageView> imgView = getImageView(nextPath, filename, extension, m_cmdbuf.get());
            
            {
              EnvmapSampler::SCreationParameters params;
              params.utilities = m_utils;
              params.assetManager = m_assetMgr;
              params.envMap = imgView;
              m_envmapImportanceSampling = EnvmapSampler::create(std::move(params));
              m_envmapImportanceSampling->computeWarpMap(getGraphicsQueue());
            }

            const auto lumaMap = m_envmapImportanceSampling->getLumaMapView();
            const auto warpMap = m_envmapImportanceSampling->getWarpMapView();

            auto downStreamingBuffer = m_utils->getDefaultDownStreamingBuffer();


            IGPUDescriptorSet::SDescriptorInfo lumaMapDescriptorInfo = {};
            lumaMapDescriptorInfo.desc = lumaMap;
            lumaMapDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

            IGPUDescriptorSet::SDescriptorInfo warpMapDescriptorInfo = {};
            warpMapDescriptorInfo.desc = warpMap;
            warpMapDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

            const IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
              {
                .dstSet = m_descriptorSet.get(), .binding = 0, .count = 1, .info = &lumaMapDescriptorInfo
              },
              {
                .dstSet = m_descriptorSet.get(), .binding = 1, .count = 1, .info = &warpMapDescriptorInfo
              },
            };

            m_utils->getLogicalDevice()->updateDescriptorSets(writes, {});

            const auto warpExtent = warpMap->getCreationParameters().image->getCreationParameters().extent;
            const STestPushConstants pc = {
              .eps = 1e-4,
              .outputAddress = downStreamingBuffer->getBuffer()->getDeviceAddress() + m_outputOffset,
              .warpResolution = uint32_t2(warpExtent.width, warpExtent.height),
              .avgLuma = m_envmapImportanceSampling->getAvgLuma(),
            };

            m_cmdbuf->bindComputePipeline(m_pipeline.get());
            m_cmdbuf->bindDescriptorSets(EPBP_COMPUTE, m_pipeline->getLayout(), 0, 1, &m_descriptorSet.get());
            m_cmdbuf->pushConstants(m_pipeline->getLayout(), ESS_COMPUTE, 0, sizeof(STestPushConstants), &pc);
            m_cmdbuf->dispatch(m_sampleCount / WorkgroupSize, 1, 1);

            m_cmdbuf->end();

            const IQueue::SSubmitInfo::SSemaphoreInfo signal[1] = {{.semaphore = m_semaphore.get(),.value=++m_timelineValue}};
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = {{.cmdbuf=m_cmdbuf.get()}};
            const IQueue::SSubmitInfo submits[1] = {{.commandBuffers=cmdbufs,.signalSemaphores=signal}};
            getGraphicsQueue()->submit(submits);
            const ISemaphore::SWaitInfo wait[1] = {{.semaphore=m_semaphore.get(),.value=m_timelineValue}};
            m_device->blockForSemaphores(wait);

            auto* gpuDownstreamingBuffer = downStreamingBuffer->getBuffer();
            if (downStreamingBuffer->needsManualFlushOrInvalidate())
            {
                const auto nonCoherentAtomSize = m_device->getPhysicalDevice()->getLimits().nonCoherentAtomSize;
                auto flushRange = ILogicalDevice::MappedMemoryRange(gpuDownstreamingBuffer->getBoundMemory().memory,m_outputOffset,m_sampleCount * sizeof(test_sample_t),ILogicalDevice::MappedMemoryRange::align_non_coherent_tag);
                m_device->invalidateMappedMemoryRanges(1u,&flushRange);
            }

            // Call the function
            const uint8_t* bufSrc = reinterpret_cast<uint8_t*>(downStreamingBuffer->getBufferPointer()) + m_outputOffset;
            const auto* testSamples = reinterpret_cast<const test_sample_t*>(bufSrc);

            for (uint32_t sample_i = 0; sample_i < m_sampleCount; sample_i++)
            {
              const auto& testSample = testSamples[sample_i];
              const auto& directOutput = testSample.directOutput;
              const auto& cachedOutput = testSample.cachedOutput;

              if (!checkEq(cachedOutput.L, directOutput.L) || !checkEq(cachedOutput.uv, directOutput.uv) || !checkEq(cachedOutput.pdf, directOutput.pdf) || !checkEq(cachedOutput.deferredPdf, directOutput.deferredPdf))
              {
                logFail("Failed similarity test between direct sampling and cached sampling. Direct Sampling = {uv = (%f, %f), L = (%f, %f %f), pdf = %f, deferredPdf = %f}, Cached Sampling = {uv = (%f, %f), L = (%f, %f %f), pdf = %f, deferredPdf = %f}", directOutput.uv.x, directOutput.uv.y, directOutput.L.x, directOutput.L.y, directOutput.L.z, directOutput.pdf, directOutput.deferredPdf, cachedOutput.uv.x, cachedOutput.uv.y, cachedOutput.L.x, cachedOutput.L.y, cachedOutput.L.z, cachedOutput.pdf, cachedOutput.pdf);
              }

              const auto& testOutput = directOutput;
              if (testOutput.jacobian < 1e-3) continue;
              if (const auto diff = abs(1.0f - (testOutput.jacobian * testOutput.pdf)); diff > 0.05)
              {
                m_logger->log("Failed similarity test of jacobian and pdf for image %s for sample number %d. xi = (%f, %f), uv = (%f, %f), Jacobian = %f, pdf = %f, difference = %f", ILogger::ELL_ERROR, "dummy", sample_i, testSample.xi.x, testSample.xi.y, testOutput.uv.x, testOutput.uv.y, testOutput.jacobian, testOutput.pdf, diff);
                continue;
              }
              
              if (const auto diff = abs(1.0f - (testOutput.jacobian * testOutput.deferredPdf)); diff > 0.05)
              {
                m_logger->log("Failed similarity test of jacobian and pdf for image %s for sample number %d. xi = (%f, %f), uv = (%f, %f), Jacobian = %f, deferredPdf = %f, difference = %f", ILogger::ELL_ERROR, "dummy", sample_i, testSample.xi.x, testSample.xi.y, testOutput.uv.x, testOutput.uv.y, testOutput.jacobian, testOutput.deferredPdf, diff);
              }
            }
          }
        }

        return true;
      }
		}

		inline void workLoopBody() override {}

		inline bool keepRunning() override { return false; }

		inline bool onAppTerminated() override
		{
      return true;
		}

	protected:

	private:
		smart_refctd_ptr<EnvmapSampler> m_envmapImportanceSampling;

		smart_refctd_ptr<IGPUImageView> getImageView(std::string inAssetPath, system::path& outFilename, system::path& outExtension, IGPUCommandBuffer* cmdbuf)
		{
			smart_refctd_ptr<ICPUImageView> cpuView;

			m_logger->log("Loading image from path %s", ILogger::ELL_DEBUG, inAssetPath.c_str());

			constexpr auto cachingFlags = static_cast<IAssetLoader::E_CACHING_FLAGS>(IAssetLoader::ECF_DONT_CACHE_REFERENCES & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
			const IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags, IAssetLoader::ELPF_NONE, m_logger.get(), m_loadCWD);
			
			auto bundle = m_assetMgr->getAsset(inAssetPath, loadParams);

			auto contents = bundle.getContents();
			if (contents.empty())
			{
				logFail("Failed to load image with path %s, skipping!", (m_loadCWD / inAssetPath).c_str());
				return nullptr;
			}

			core::splitFilename(inAssetPath.c_str(), nullptr, &outFilename, &outExtension);

			const auto& asset = contents[0];
			switch (asset->getAssetType())
			{
				case IAsset::ET_IMAGE:
				{
					auto image = smart_refctd_ptr_static_cast<ICPUImage>(asset);
					const auto format = image->getCreationParameters().format;

					ICPUImageView::SCreationParams viewParams = 
					{
						.flags = ICPUImageView::E_CREATE_FLAGS::ECF_NONE,
						.image = std::move(image),
						.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D,
						.format = format,
						.subresourceRange = {
							.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
							.baseMipLevel = 0u,
							.levelCount = ICPUImageView::remaining_mip_levels,
							.baseArrayLayer = 0u,
							.layerCount = ICPUImageView::remaining_array_layers
						}
					};

					cpuView = ICPUImageView::create(std::move(viewParams));
				} break;

				case IAsset::ET_IMAGE_VIEW:
					cpuView = smart_refctd_ptr_static_cast<ICPUImageView>(asset);
					break;
				default:
					logFail("Failed to load ICPUImage or ICPUImageView got some other Asset Type, skipping!");
					return nullptr;
			}

      auto converter = CAssetConverter::create({ .device = m_device.get() });

      // Test the provision of a custom patch this time
      CAssetConverter::patch_t<ICPUImageView> patch(cpuView.get(), IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT);

      // We don't want to generate mip-maps for these images (YET), to ensure that we must override the default callbacks.
      struct SInputs final : CAssetConverter::SInputs
      {
        inline uint8_t getMipLevelCount(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
        {
          return image->getCreationParameters().mipLevels;
        }
        inline uint16_t needToRecomputeMips(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
        {
          return 0b0u;
        }
      } inputs = {};
      std::get<CAssetConverter::SInputs::asset_span_t<ICPUImageView>>(inputs.assets) = { &cpuView.get(),1 };
      std::get<CAssetConverter::SInputs::patch_span_t<ICPUImageView>>(inputs.patches) = { &patch,1 };
      inputs.logger = m_logger.get();

      //
      auto reservation = converter->reserve(inputs);

      // get the created image view
      auto gpuView = reservation.getGPUObjects<ICPUImageView>().front().value;

      if (!gpuView)
        return nullptr;

      gpuView->getCreationParameters().image->setObjectDebugName(inAssetPath.c_str());

      // we should multi-buffer to not stall before renderpass recording but oh well
      IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = { cmdbuf };

      m_intendedSubmit.scratchCommandBuffers = { &cmdbufInfo,1 };
      CAssetConverter::SConvertParams params = {};
      params.transfer = &m_intendedSubmit;
      params.utilities = m_utils.get();
      auto result = reservation.convert(params);

      if (result.copy() != IQueue::RESULT::SUCCESS)
        return nullptr;

			return gpuView;

		}

		std::ifstream m_testPathsFile;
		system::path m_loadCWD;

		smart_refctd_ptr<ISemaphore> m_scratchSemaphore;
		smart_refctd_ptr<ISemaphore> m_semaphore;
    uint64_t m_timelineValue;

		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		SIntendedSubmitInfo m_intendedSubmit;

		smart_refctd_ptr<IGPUCommandBuffer> m_cmdbuf;
    core::smart_refctd_ptr<video::IGPUComputePipeline> m_pipeline;
    core::smart_refctd_ptr<video::IGPUDescriptorSet> m_descriptorSet;
    core::smart_refctd_ptr<video::IGPUBuffer> m_outputBuffer;


    uint32_t m_sampleCount = 10000;
    uint32_t m_outputOffset;

};

NBL_MAIN_FUNC(EnvmapImportanceSampleApp)
