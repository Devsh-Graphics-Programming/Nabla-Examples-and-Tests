// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/this_example/common.hpp"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

// TODO: Add a QueryPool for timestamping once its ready
class ComputeShaderPathtracer final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::SimpleWindowedApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
		using clock_t = std::chrono::steady_clock;

		enum E_LIGHT_GEOMETRY : uint8_t
		{
			ELG_SPHERE,
			ELG_TRIANGLE,
			ELG_RECTANGLE,
			ELG_COUNT
		};

		struct SBasicViewParametersAligned
		{
			SBasicViewParameters uboData;
		};

		constexpr static inline uint32_t2 WindowDimensions = { 1280, 720 };
		constexpr static inline uint32_t FramesInFlight = 5;
		constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);
		constexpr static inline E_LIGHT_GEOMETRY LightGeom = E_LIGHT_GEOMETRY::ELG_SPHERE;
		constexpr static inline uint32_t DefaultWorkGroupSize = 16u;
		constexpr static inline uint32_t MaxDescriptorCount = 256u;
		constexpr static inline uint32_t MaxDepthLog2 = 4u; // 5
		constexpr static inline uint32_t MaxSamplesLog2 = 10u; // 18
		constexpr static inline uint32_t MaxBufferDimensions = 3u << MaxDepthLog2;
		constexpr static inline uint32_t MaxBufferSamples = 1u << MaxSamplesLog2;
		constexpr static inline uint8_t MaxUITextureCount = 2u;
		constexpr static inline uint8_t SceneTextureIndex = 1u;
		static inline std::string DefaultImagePathsFile = "../../media/envmap/envmap_0.exr";
		static inline std::array<std::string, 3> ShaderPaths = { "app_resources/litBySphere.comp", "app_resources/litByTriangle.comp", "app_resources/litByRectangle.comp" };

	public:
		inline ComputeShaderPathtracer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {
			const auto cameraPos = core::vectorSIMDf(0, 5, -10);
			matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(
				core::radians(fov),
				static_cast<float32_t>(WindowDimensions.x) / static_cast<float32_t>(WindowDimensions.y),
				zNear,
				zFar
			);

			m_camera = Camera(cameraPos, core::vectorSIMDf(0, 0, 0), proj);
		}

		inline bool isComputeOnly() const override { return false; }

		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			if (!m_surface)
			{
				{
					auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
					IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
					params.width = WindowDimensions.x;
					params.height = WindowDimensions.y;
					params.x = 32;
					params.y = 32;
					params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
					params.windowCaption = "ComputeShaderPathtracer";
					params.callback = windowCallback;
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}

				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<nbl::video::CDefaultSwapchainFramebuffers>::create(std::move(surface));
			}

			if (m_surface)
				return { {m_surface->getSurface()/*,EQF_NONE*/} };

			return {};
		}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Init systems
			{
				m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

				// Remember to call the base class initialization!
				if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
					return false;
				if (!asset_base_t::onAppInitialized(std::move(system)))
					return false;

				m_uiSemaphore = m_device->createSemaphore(m_realFrameIx);
				if (!m_uiSemaphore)
					return logFail("Failed to create semaphore!");
			}

			// Create renderpass and init surface
			nbl::video::IGPURenderpass* renderpass;
			{
				ISwapchain::SCreationParams swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
				if (!swapchainParams.deduceFormat(m_physicalDevice))
					return logFail("Could not choose a Surface Format for the Swapchain!");

				const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] =
				{
					{
						.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
						.dstSubpass = 0,
						.memoryBarrier =
						{
							.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
							.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
							.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
							.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
						}
					},
					{
						.srcSubpass = 0,
						.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
						.memoryBarrier =
						{
							.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
							.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
						}
					},
					IGPURenderpass::SCreationParams::DependenciesEnd
				};

				auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(), swapchainParams.surfaceFormat.format, dependencies);
				renderpass = scResources->getRenderpass();

				if (!renderpass)
					return logFail("Failed to create Renderpass!");

				auto gQueue = getGraphicsQueue();
				if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
					return logFail("Could not create Window & Surface or initialize the Surface!");
			}

			// Compute no of frames in flight
			{
				m_maxFramesInFlight = m_surface->getMaxFramesInFlight();
				if (FramesInFlight < m_maxFramesInFlight)
				{
					m_logger->log("Lowering frames in flight!", ILogger::ELL_WARNING);
					m_maxFramesInFlight = FramesInFlight;
				}
			}

			// image upload utils
			{
				m_scratchSemaphore = m_device->createSemaphore(0);
				if (!m_scratchSemaphore)
					return logFail("Could not create Scratch Semaphore");
				m_scratchSemaphore->setObjectDebugName("Scratch Semaphore");
				// we don't want to overcomplicate the example with multi-queue
				m_intendedSubmit.queue = getGraphicsQueue();
				// wait for nothing before upload
				m_intendedSubmit.waitSemaphores = {};
				m_intendedSubmit.waitSemaphores = {};
				// fill later
				m_intendedSubmit.scratchCommandBuffers = {};
				m_intendedSubmit.scratchSemaphore = {
					.semaphore = m_scratchSemaphore.get(),
					.value = 0,
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
				};
			}

			// Create command pool and buffers
			{
				auto gQueue = getGraphicsQueue();
				m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");

				if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data(), 2 * m_maxFramesInFlight }))
					return logFail("Couldn't create Command Buffer!");
			}

			// Create descriptors and pipeline for the pathtracer
			{
				auto convertDSLayoutCPU2GPU = [&](smart_refctd_ptr<ICPUDescriptorSetLayout> cpuLayout) {
					auto converter = CAssetConverter::create({ .device = m_device.get() });
					CAssetConverter::SInputs inputs = {};
					inputs.readCache = converter.get();
					inputs.logger = m_logger.get();
					CAssetConverter::SConvertParams params = {};
					params.utilities = m_utils.get();

					std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSetLayout>>(inputs.assets) = { &cpuLayout.get(),1 };
					// don't need to assert that we don't need to provide patches since layouts are not patchable
					//assert(true);
					auto reservation = converter->reserve(inputs);
					// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
					auto gpuLayout = reservation.getGPUObjects<ICPUDescriptorSetLayout>().front().value;
					if (!gpuLayout) {
						m_logger->log("Failed to convert %s into an IGPUDescriptorSetLayout handle", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					return gpuLayout;
				};
				auto convertDSCPU2GPU = [&](smart_refctd_ptr<ICPUDescriptorSet> cpuDS) {
					auto converter = CAssetConverter::create({ .device = m_device.get() });
					CAssetConverter::SInputs inputs = {};
					inputs.readCache = converter.get();
					inputs.logger = m_logger.get();
					CAssetConverter::SConvertParams params = {};
					params.utilities = m_utils.get();

					std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSet>>(inputs.assets) = { &cpuDS.get(), 1 };
					// don't need to assert that we don't need to provide patches since layouts are not patchable
					//assert(true);
					auto reservation = converter->reserve(inputs);
					// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
					auto gpuDS = reservation.getGPUObjects<ICPUDescriptorSet>().front().value;
					if (!gpuDS) {
						m_logger->log("Failed to convert %s into an IGPUDescriptorSet handle", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					return gpuDS;
				};

				std::array<ICPUDescriptorSetLayout::SBinding, 1> descriptorSet0Bindings = {};
				std::array<ICPUDescriptorSetLayout::SBinding, 1> uboBindings = {};
				std::array<ICPUDescriptorSetLayout::SBinding, 3> descriptorSet3Bindings = {};

				descriptorSet0Bindings[0] = {
					.binding = 0u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};
				uboBindings[0] = {
					.binding = 0u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};
				descriptorSet3Bindings[0] = {
					.binding = 0u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};
				descriptorSet3Bindings[1] = {
					.binding = 1u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};
				descriptorSet3Bindings[2] = {
					.binding = 2u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};

				auto cpuDescriptorSetLayout0 = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(descriptorSet0Bindings);
				auto cpuDescriptorSetLayout1 = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(uboBindings);
				auto cpuDescriptorSetLayout2 = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(descriptorSet3Bindings);

				auto cpuDescriptorSet0 = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(cpuDescriptorSetLayout0));
				auto cpuDescriptorSet1 = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(cpuDescriptorSetLayout1));
				auto cpuDescriptorSet2 = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(cpuDescriptorSetLayout2));

				auto gpuDescriptorSetLayout0 = convertDSLayoutCPU2GPU(cpuDescriptorSetLayout0);
				auto gpuDescriptorSetLayout1 = convertDSLayoutCPU2GPU(cpuDescriptorSetLayout1);
				auto gpuDescriptorSetLayout2 = convertDSLayoutCPU2GPU(cpuDescriptorSetLayout2);

				m_descriptorSet0 = convertDSCPU2GPU(cpuDescriptorSet0);
				m_uboDescriptorSet1 = convertDSCPU2GPU(cpuDescriptorSet1);
				m_descriptorSet2 = convertDSCPU2GPU(cpuDescriptorSet2);

				auto createGpuResources = [&](std::string pathToShader, smart_refctd_ptr<IGPUComputePipeline>&& pipeline) -> bool
					{
						IAssetLoader::SAssetLoadParams lp = {};
						lp.logger = m_logger.get();
						lp.workingDirectory = ""; // virtual root
						auto assetBundle = m_assetMgr->getAsset(pathToShader, lp);
						const auto assets = assetBundle.getContents();
						if (assets.empty())
						{
							return logFail("Could not load shader!");
						}

						auto source = IAsset::castDown<ICPUShader>(assets[0]);
						// The down-cast should not fail!
						assert(source);

						// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
						auto shader = m_device->createShader(source.get());
						if (!shader)
						{
							return logFail("Shader creationed failed: %s!", pathToShader);
						}

						auto gpuPipelineLayout = m_device->createPipelineLayout({}, core::smart_refctd_ptr(gpuDescriptorSetLayout0), core::smart_refctd_ptr(gpuDescriptorSetLayout1), core::smart_refctd_ptr(gpuDescriptorSetLayout2), nullptr);
						if (!gpuPipelineLayout) {
							return logFail("Failed to create pipeline layout");
						}

						IGPUComputePipeline::SCreationParams params = {};
						params.layout = gpuPipelineLayout.get();
						params.shader.shader = shader.get();
						params.shader.entryPoint = "main";
						params.shader.entries = nullptr;
						params.shader.requireFullSubgroups = true;
						params.shader.requiredSubgroupSize = static_cast<IGPUShader::SSpecInfo::SUBGROUP_SIZE>(5);
						if (!m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline)) {
							return logFail("Failed to create compute pipeline!\n");
						}

						return true;
					};

				if (!createGpuResources(ShaderPaths[LightGeom], std::move(m_pipeline))) {
					return logFail("Pipeline creation failed!");
				}

				// Update Descriptors

				ISampler::SParams samplerParams0 = {
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETBC_FLOAT_OPAQUE_BLACK,
					ISampler::ETF_LINEAR,
					ISampler::ETF_LINEAR,
					ISampler::ESMM_LINEAR,
					0u,
					false,
					ECO_ALWAYS
				};
				auto sampler0 = m_device->createSampler(samplerParams0);
				ISampler::SParams samplerParams1 = {
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETC_CLAMP_TO_EDGE,
					ISampler::ETBC_INT_OPAQUE_BLACK,
					ISampler::ETF_NEAREST,
					ISampler::ETF_NEAREST,
					ISampler::ESMM_NEAREST,
					0u,
					false,
					ECO_ALWAYS
				};
				auto sampler1 = m_device->createSampler(samplerParams1);

				std::array<IGPUDescriptorSet::SDescriptorInfo, 5> writeDSInfos = {};
				writeDSInfos[0].desc = m_outImgView;
				writeDSInfos[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
				writeDSInfos[1].desc = m_ubo;
				writeDSInfos[1].info.buffer.offset = 0ull;
				writeDSInfos[1].info.buffer.size = sizeof(SBasicViewParametersAligned);
				writeDSInfos[2].desc = m_envMapView;
				// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
				writeDSInfos[2].info.combinedImageSampler.sampler = sampler0;
				writeDSInfos[2].info.combinedImageSampler.imageLayout = asset::IImage::LAYOUT::READ_ONLY_OPTIMAL;
				writeDSInfos[3].desc = m_sequenceBufferView;
				writeDSInfos[4].desc = m_scrambleView;
				// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
				writeDSInfos[4].info.combinedImageSampler.sampler = sampler1;
				writeDSInfos[4].info.combinedImageSampler.imageLayout = asset::IImage::LAYOUT::READ_ONLY_OPTIMAL;

				std::array<IGPUDescriptorSet::SWriteDescriptorSet, 5> writeDescriptorSets = {};
				writeDescriptorSets[0] = {
					.dstSet = m_descriptorSet0.get(),
					.binding = 0,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[0]
				};
				writeDescriptorSets[1] = {
					.dstSet = m_uboDescriptorSet1.get(),
					.binding = 0,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[1]
				};
				writeDescriptorSets[2] = {
					.dstSet = m_descriptorSet2.get(),
					.binding = 0,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[2]
				};
				writeDescriptorSets[3] = {
					.dstSet = m_descriptorSet2.get(),
					.binding = 1,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[3]
				};
				writeDescriptorSets[4] = {
					.dstSet = m_descriptorSet2.get(),
					.binding = 2,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[4]
				};

				m_device->updateDescriptorSets(writeDescriptorSets, {});
			}

			// load CPUImages and convert to GPUImages
			smart_refctd_ptr<IGPUImage> envMap, scrambleMap;
			{
				auto convertImgCPU2GPU = [&](smart_refctd_ptr<ICPUImage> cpuImg) {
					auto queue = getGraphicsQueue();
					auto cmdbuf = m_cmdBufs[0].get();
					cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
					std::array<IQueue::SSubmitInfo::SCommandBufferInfo, 1> commandBufferInfo = { cmdbuf };
					core::smart_refctd_ptr<ISemaphore> imgFillSemaphore = m_device->createSemaphore(0);
					imgFillSemaphore->setObjectDebugName("Image Fill Semaphore");

					auto converter = CAssetConverter::create({ .device = m_device.get() });
					// We don't want to generate mip-maps for these images, to ensure that we must override the default callbacks.
					struct SInputs final : CAssetConverter::SInputs
					{
						// we also need to override this to have concurrent sharing
						inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUImage* buffer, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
						{
							if (familyIndices.size() > 1)
								return familyIndices;
							return {};
						}

						inline uint8_t getMipLevelCount(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
						{
							return image->getCreationParameters().mipLevels;
						}
						inline uint16_t needToRecomputeMips(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
						{
							return 0b0u;
						}

						std::vector<uint32_t> familyIndices;
					} inputs = {};
					inputs.readCache = converter.get();
					inputs.logger = m_logger.get();
					{
						const core::set<uint32_t> uniqueFamilyIndices = { queue->getFamilyIndex(), queue->getFamilyIndex() };
						inputs.familyIndices = { uniqueFamilyIndices.begin(),uniqueFamilyIndices.end() };
					}
					// scratch command buffers for asset converter transfer commands
					SIntendedSubmitInfo transfer = {
						.queue = queue,
						.waitSemaphores = {},
						.prevCommandBuffers = {},
						.scratchCommandBuffers = commandBufferInfo,
						.scratchSemaphore = {
							.semaphore = imgFillSemaphore.get(),
							.value = 0,
							// because of layout transitions
							.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
						}
					};
					// as per the `SIntendedSubmitInfo` one commandbuffer must be begun
					cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
					// Normally we'd have to inherit and override the `getFinalOwnerQueueFamily` callback to ensure that the
					// compute queue becomes the owner of the buffers and images post-transfer, but in this example we use concurrent sharing
					CAssetConverter::SConvertParams params = {};
					params.transfer = &transfer;
					params.utilities = m_utils.get();

					std::get<CAssetConverter::SInputs::asset_span_t<ICPUImage>>(inputs.assets) = { &cpuImg.get(),1 };
					// assert that we don't need to provide patches
					assert(cpuImg->getImageUsageFlags().hasFlags(ICPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT));
					auto reservation = converter->reserve(inputs);
					// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
					auto gpuImg = reservation.getGPUObjects<ICPUImage>().front().value;
					if (!gpuImg) {
						m_logger->log("Failed to convert %s into an IGPUImage handle", ILogger::ELL_ERROR, DefaultImagePathsFile);
						std::exit(-1);
					}

					// we want our converter's submit to signal a semaphore that image contents are ready
					const IQueue::SSubmitInfo::SSemaphoreInfo signalSemaphore = {
							.semaphore = imgFillSemaphore.get(),
							.value = 1u,
							// cannot signal from COPY stage because there's a layout transition and a possible ownership transfer
							// and we need to wait for right after and they don't have an explicit stage
							.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
					};
					params.extraSignalSemaphores = { &signalSemaphore,1 };
					// and launch the conversions
					m_api->startCapture();
					auto result = reservation.convert(params);
					m_api->endCapture();
					if (!result.blocking() && result.copy() != IQueue::RESULT::SUCCESS) {
						m_logger->log("Failed to record or submit conversions", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					return gpuImg;
				};

				smart_refctd_ptr<ICPUImage> envMapCPU, scrambleMapCPU;
				{
					IAssetLoader::SAssetLoadParams lp;
					SAssetBundle bundle = m_assetMgr->getAsset(DefaultImagePathsFile, lp);
					if (bundle.getContents().empty()) {
						m_logger->log("Couldn't load an asset.", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					envMapCPU = IAsset::castDown<ICPUImage>(bundle.getContents()[0]);
					if (!envMapCPU) {
						m_logger->log("Couldn't load an asset.", ILogger::ELL_ERROR);
						std::exit(-1);
					}
				};
				{
					asset::ICPUImage::SCreationParams info;
					info.format = asset::E_FORMAT::EF_R32G32_UINT;
					info.type = asset::ICPUImage::ET_2D;
					auto extent = envMapCPU->getCreationParameters().extent;
					info.extent.width = extent.width;
					info.extent.height = extent.height;
					info.extent.depth = 1u;
					info.mipLevels = 1u;
					info.arrayLayers = 1u;
					info.samples = asset::ICPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
					info.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
					info.usage = asset::IImage::EUF_TRANSFER_SRC_BIT | asset::IImage::EUF_SAMPLED_BIT;

					scrambleMapCPU = ICPUImage::create(std::move(info));
					const uint32_t texelFormatByteSize = getTexelOrBlockBytesize(scrambleMapCPU->getCreationParameters().format);
					const uint32_t texelBufferSize = scrambleMapCPU->getImageDataSizeInBytes();
					auto texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(texelBufferSize);

					auto out = reinterpret_cast<uint8_t *>(texelBuffer->getPointer());
					for(auto index = 0u; index < texelBufferSize; index++)
						out[index] = 0;

					auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
					ICPUImage::SBufferCopy& region = regions->front();
					region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
					region.imageSubresource.mipLevel = 0u;
					region.imageSubresource.baseArrayLayer = 0u;
					region.imageSubresource.layerCount = 1u;
					region.bufferOffset = 0u;
					region.bufferRowLength = IImageAssetHandlerBase::calcPitchInBlocks(extent.width, texelFormatByteSize);
					region.bufferImageHeight = 0u;
					region.imageOffset = { 0u, 0u, 0u };
					region.imageExtent = scrambleMapCPU->getCreationParameters().extent;

					scrambleMapCPU->setBufferAndRegions(std::move(texelBuffer), regions);
				}

				envMap = convertImgCPU2GPU(envMapCPU);
				scrambleMap = convertImgCPU2GPU(scrambleMapCPU);
			}

			// create views for textures
			{
				auto createHDRIImage = [this](const asset::E_FORMAT colorFormat, const uint32_t width, const uint32_t height) -> smart_refctd_ptr<IGPUImage> {
					IGPUImage::SCreationParams imgInfo;
					imgInfo.format = colorFormat;
					imgInfo.type = IGPUImage::ET_2D;
					imgInfo.extent.width = width;
					imgInfo.extent.height = height;
					imgInfo.extent.depth = 1u;
					imgInfo.mipLevels = 1u;
					imgInfo.arrayLayers = 1u;
					imgInfo.samples = IGPUImage::ESCF_1_BIT;
					imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
					imgInfo.usage = asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT | asset::IImage::EUF_SAMPLED_BIT;

					auto image = m_device->createImage(std::move(imgInfo));
					auto imageMemReqs = image->getMemoryReqs();
					imageMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
					m_device->allocate(imageMemReqs, image.get());

					return image;
				};
				auto createHDRIImageView = [this](smart_refctd_ptr<IGPUImage> img) -> smart_refctd_ptr<IGPUImageView>
				{
					auto format = img->getCreationParameters().format;
					IGPUImageView::SCreationParams imgViewInfo;
					imgViewInfo.image = std::move(img);
					imgViewInfo.format = format;
					imgViewInfo.viewType = IGPUImageView::ET_2D;
					imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
					imgViewInfo.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
					imgViewInfo.subresourceRange.baseArrayLayer = 0u;
					imgViewInfo.subresourceRange.baseMipLevel = 0u;
					imgViewInfo.subresourceRange.layerCount = 1u;
					imgViewInfo.subresourceRange.levelCount = 1u;

					return m_device->createImageView(std::move(imgViewInfo));
				};

				auto params = envMap->getCreationParameters();
				auto extent = params.extent;
				envMap->setObjectDebugName("Env Map");
				m_envMapView = createHDRIImageView(envMap);
				m_envMapView->setObjectDebugName("Env Map View");
				scrambleMap->setObjectDebugName("Scramble Map");
				m_scrambleView = createHDRIImageView(scrambleMap);
				m_scrambleView->setObjectDebugName("Scramble Map View");
				auto outImg = createHDRIImage(asset::E_FORMAT::EF_R16G16B16A16_SFLOAT, WindowDimensions.x, WindowDimensions.y);
				outImg->setObjectDebugName("Output Image");
				m_outImgView = createHDRIImageView(outImg);
				m_outImgView->setObjectDebugName("Output Image View");
			}

			// create ubo and sequence buffer view
			{
				{
					IGPUBuffer::SCreationParams params = {};
					params.usage = IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
					params.size = sizeof(SBasicViewParametersAligned);

					m_ubo = m_device->createBuffer(std::move(params));
					m_ubo->setObjectDebugName("UBO");
					auto memReqs = m_ubo->getMemoryReqs();
					memReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
					m_device->allocate(memReqs, m_ubo.get());
				}

				{
					auto sampleSequence = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t) * MaxBufferDimensions * MaxBufferSamples);

					core::OwenSampler sampler(MaxBufferDimensions, 0xdeadbeefu);
					//core::SobolSampler sampler(MaxBufferDimensions);

					auto out = reinterpret_cast<uint32_t*>(sampleSequence->getPointer());
					for (auto dim = 0u; dim < MaxBufferDimensions; dim++)
						for (uint32_t i = 0; i < MaxBufferSamples; i++)
						{
							out[i * MaxBufferDimensions + dim] = sampler.sample(dim, i);
						}

					IGPUBuffer::SCreationParams params = {};
					params.usage = asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT;
					params.size = sampleSequence->getSize();

					// we don't want to overcomplicate the example with multi-queue
					auto queue = getGraphicsQueue();
					auto cmdbuf = m_cmdBufs[0].get();
					cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
					IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = { cmdbuf };
					m_intendedSubmit.scratchCommandBuffers = { &cmdbufInfo, 1 };

					cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
					m_api->startCapture();
					auto bufferFuture = m_utils->createFilledDeviceLocalBufferOnDedMem(
						m_intendedSubmit,
						std::move(params),
						sampleSequence->getPointer()
					);
					m_api->endCapture();
					bufferFuture.wait();
					auto buffer = bufferFuture.get();

					m_sequenceBufferView = m_device->createBufferView({ 0u, buffer->get()->getSize(), *buffer }, asset::E_FORMAT::EF_R32G32B32_UINT);
					m_sequenceBufferView->setObjectDebugName("Sequence Buffer");
				}
			}

#if 0
			// upload data
			{
				// upload scramble data
				{
					auto extent = m_envMapView->getCreationParameters().image->getCreationParameters().extent;

					IGPUImage::SBufferCopy region = {};
					region.bufferOffset = 0u;
					region.bufferRowLength = 0u;
					region.bufferImageHeight = 0u;
					region.imageExtent = extent;
					region.imageOffset = { 0u,0u,0u };
					region.imageSubresource.layerCount = 1u;
					region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;

					const std::span<const asset::IImage::SBufferCopy> regions = { &region, 1 };

					// we don't want to overcomplicate the example with multi-queue
					auto queue = getGraphicsQueue();
					auto cmdbuf = m_cmdBufs[0].get();
					cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
					IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = { cmdbuf };
					m_intendedSubmit.scratchCommandBuffers = { &cmdbufInfo, 1 };

					cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
					m_api->startCapture();
					// TODO: this silently failed for some reason instead of telling you its bad, add logging of errors in CommandBuffer and IUtilities.
					m_utils->updateImageViaStagingBufferAutoSubmit(
						m_intendedSubmit,
						random.data(),
						asset::E_FORMAT::EF_R32G32_UINT,
						m_scrambleView->getCreationParameters().image.get(),
						IGPUImage::LAYOUT::UNDEFINED,
						regions
					);
					m_api->endCapture();
				}
			}
#endif

			// Create ui descriptors
			{
				using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
				{
					IGPUSampler::SParams params;
					params.AnisotropicFilter = 1u;
					params.TextureWrapU = ISampler::ETC_REPEAT;
					params.TextureWrapV = ISampler::ETC_REPEAT;
					params.TextureWrapW = ISampler::ETC_REPEAT;

					m_ui.samplers.gui = m_device->createSampler(params);
					m_ui.samplers.gui->setObjectDebugName("Nabla IMGUI UI Sampler");
				}

				{
					IGPUSampler::SParams params;
					params.MinLod = 0.f;
					params.MaxLod = 0.f;
					params.TextureWrapU = ISampler::ETC_CLAMP_TO_EDGE;
					params.TextureWrapV = ISampler::ETC_CLAMP_TO_EDGE;
					params.TextureWrapW = ISampler::ETC_CLAMP_TO_EDGE;

					m_ui.samplers.scene = m_device->createSampler(params);
					m_ui.samplers.scene->setObjectDebugName("Nabla IMGUI Scene Sampler");
				}

				std::array<core::smart_refctd_ptr<IGPUSampler>, 69u> immutableSamplers;
				for (auto& it : immutableSamplers)
					it = smart_refctd_ptr(m_ui.samplers.scene);

				immutableSamplers[nbl::ext::imgui::UI::FontAtlasTexId] = smart_refctd_ptr(m_ui.samplers.gui);

				nbl::ext::imgui::UI::SCreationParameters params;

				params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
				params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
				params.assetManager = m_assetMgr;
				params.pipelineCache = nullptr;
				params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, MaxUITextureCount);
				params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
				params.streamingBuffer = nullptr;
				params.subpassIx = 0u;
				params.transfer = getTransferUpQueue();
				params.utilities = m_utils;
				{
					m_ui.manager = ext::imgui::UI::create(std::move(params));

					// note that we use default layout provided by our extension, but you are free to create your own by filling nbl::ext::imgui::UI::S_CREATION_PARAMETERS::resources
					const auto* descriptorSetLayout = m_ui.manager->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
					const auto& params = m_ui.manager->getCreationParameters();

					IDescriptorPool::SCreateInfo descriptorPoolInfo = {};
					descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLER)] = (uint32_t)nbl::ext::imgui::UI::DefaultSamplerIx::COUNT;
					descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)] = MaxUITextureCount;
					descriptorPoolInfo.maxSets = 1u;
					descriptorPoolInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT;

					m_guiDescriptorSetPool = m_device->createDescriptorPool(std::move(descriptorPoolInfo));
					assert(m_guiDescriptorSetPool);

					m_guiDescriptorSetPool->createDescriptorSets(1u, &descriptorSetLayout, &m_ui.descriptorSet);
					assert(m_ui.descriptorSet);
				}
			}
			m_ui.manager->registerListener(
				[this]() -> void {
					ImGuiIO& io = ImGui::GetIO();

					m_camera.setProjectionMatrix([&]()
					{
						static matrix4SIMD projection;

						projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y, zNear, zFar);

						return projection;
					}());

					ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
					ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

					// create a window and insert the inspector
					ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
					ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
					ImGui::Begin("Editor");

					ImGui::SameLine();

					ImGui::Text("Camera");

					ImGui::Checkbox("Enable camera movement", &move);
					ImGui::SliderFloat("Move speed", &moveSpeed, 0.1f, 10.f);

					ImGui::SliderFloat("Fov", &fov, 20.f, 150.f);

					ImGui::SliderFloat("zNear", &zNear, 0.1f, 100.f);
					ImGui::SliderFloat("zFar", &zFar, 110.f, 10000.f);

					ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);

					ImGui::Image(SceneTextureIndex, ImGui::GetContentRegionAvail());

					ImGui::End();
				}
			);

			m_winMgr->setWindowSize(m_window.get(), WindowDimensions.x, WindowDimensions.y);
			m_surface->recreateSwapchain();
			m_winMgr->show(m_window.get());
			m_oracle.reportBeginFrameRecord();
			m_camera.mapKeysToArrows();

			return true;
		}

		bool updateGUIDescriptorSet()
		{
			// texture atlas + our scene texture, note we don't create info & write pair for the font sampler because UI extension's is immutable and baked into DS layout
			static std::array<IGPUDescriptorSet::SDescriptorInfo, MaxUITextureCount> descriptorInfo;
			static IGPUDescriptorSet::SWriteDescriptorSet writes[MaxUITextureCount];

			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].desc = smart_refctd_ptr<IGPUImageView>(m_ui.manager->getFontAtlasView());

			descriptorInfo[SceneTextureIndex].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

			descriptorInfo[SceneTextureIndex].desc = m_outImgView;

			for (uint32_t i = 0; i < descriptorInfo.size(); ++i)
			{
				writes[i].dstSet = m_ui.descriptorSet.get();
				writes[i].binding = 0u;
				writes[i].arrayElement = i;
				writes[i].count = 1u;
			}
			writes[nbl::ext::imgui::UI::FontAtlasTexId].info = descriptorInfo.data() + nbl::ext::imgui::UI::FontAtlasTexId;
			writes[SceneTextureIndex].info = descriptorInfo.data() + SceneTextureIndex;

			return m_device->updateDescriptorSets(writes, {});
		}

		inline void workLoopBody() override
		{
			const auto resourceIx = m_realFrameIx % m_maxFramesInFlight;

			if (m_realFrameIx >= m_maxFramesInFlight)
			{
				const ISemaphore::SWaitInfo cbDonePending[] = 
				{
					{
						.semaphore = m_uiSemaphore.get(),
						.value = m_realFrameIx + 1 - m_maxFramesInFlight
					}
				};
				if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

			m_api->startCapture();

			// CPU events
			update();

			// render whole scene to offline frame buffer & submit
			{
				auto queue = getGraphicsQueue();
				auto& cmdbuf = m_cmdBufs[resourceIx];
				cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
				const auto viewMatrix = m_camera.getViewMatrix();
				const auto viewProjectionMatrix = matrix4SIMD();
				/*
				* Temporarily use identity matrix (Desktop only)
					matrix4SIMD::concatenateBFollowedByAPrecisely(
						video::ISurface::getSurfaceTransformationMatrix(swapchain->getPreTransform()),
						m_camera.getConcatenatedMatrix()
					);
				*/

				// safe to proceed
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				{
					auto mv = viewMatrix;
					auto mvp = viewProjectionMatrix;
					core::matrix3x4SIMD normalMat;
					mv.getSub3x3InverseTranspose(normalMat);

					SBasicViewParametersAligned viewParams;
					memcpy(viewParams.uboData.MV, mv.pointer(), sizeof(mv));
					memcpy(viewParams.uboData.MVP, mvp.pointer(), sizeof(mvp));
					memcpy(viewParams.uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));

					asset::SBufferRange<video::IGPUBuffer> range;
					range.buffer = m_ubo;
					range.offset = 0ull;
					range.size = sizeof(viewParams);
					
					IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = { cmdbuf.get() };
					m_intendedSubmit.scratchCommandBuffers = { &cmdbufInfo, 1 };
					
					m_utils->updateBufferRangeViaStagingBuffer(m_intendedSubmit, range, &viewParams);
					m_utils->autoSubmit(m_intendedSubmit, [&](SIntendedSubmitInfo& nextSubmit) -> bool { return true; });
				}

				// TRANSITION m_outImgView to GENERAL (because of descriptorSets0 -> ComputeShader Writes into the image)
				{
					const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers[] = {
						{
							.barrier = {
								.dep = {
									.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
									.srcAccessMask = ACCESS_FLAGS::NONE,
									.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
									.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
								}
							},
							.image = m_outImgView->getCreationParameters().image.get(),
							.subresourceRange = {
								.aspectMask = IImage::EAF_COLOR_BIT,
								.baseMipLevel = 0u,
								.levelCount = 1u,
								.baseArrayLayer = 0u,
								.layerCount = 1u
							},
							.oldLayout = IImage::LAYOUT::UNDEFINED,
							.newLayout = IImage::LAYOUT::GENERAL
						}
					};
					cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
					cmdbuf->beginDebugMarker("ComputeShaderPathtracer IMGUI Frame");
					cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers });
				}

				// cube envmap handle
				{
					cmdbuf->bindComputePipeline(m_pipeline.get());
					cmdbuf->bindDescriptorSets(EPBP_COMPUTE, m_pipeline->getLayout(), 0u, 1u, &m_descriptorSet0.get());
					cmdbuf->bindDescriptorSets(EPBP_COMPUTE, m_pipeline->getLayout(), 1u, 1u, &m_uboDescriptorSet1.get());
					cmdbuf->bindDescriptorSets(EPBP_COMPUTE, m_pipeline->getLayout(), 2u, 1u, &m_descriptorSet2.get());
					cmdbuf->dispatch(1 + (WindowDimensions.x - 1) / DefaultWorkGroupSize, 1 + (WindowDimensions.y - 1) / DefaultWorkGroupSize, 1u);
				}
				// TODO: tone mapping and stuff

				// Wait for offline compute render
				{
					constexpr SMemoryBarrier barriers[] = {
						{
							.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
							.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
						}
					};
					const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers[] = {
						{
							.barrier = {
								.dep = barriers[0]
							},
							.image = m_outImgView->getCreationParameters().image.get(),
							.subresourceRange = {
								.aspectMask = IImage::EAF_COLOR_BIT,
								.baseMipLevel = 0u,
								.levelCount = 1u,
								.baseArrayLayer = 0u,
								.layerCount = 1u
							}
						}
					};
					cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers });
				}
			}

			auto* const cb = m_cmdBufs.data()[resourceIx].get();
			auto* queue = getGraphicsQueue();

			asset::SViewport viewport;
			{
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = WindowDimensions.x;
				viewport.height = WindowDimensions.y;
			}
			cb->setViewport(0u, 1u, &viewport);

			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};

			// UI render pass
			{
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				const IGPUCommandBuffer::SRenderpassBeginInfo info = 
				{
					.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
					.colorClearValues = &clearColor,
					.depthStencilClearValues = nullptr,
					.renderArea = currentRenderArea
				};
				ISemaphore::SWaitInfo waitInfo = { .semaphore = m_uiSemaphore.get(), .value = m_realFrameIx + 1u };

				cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
				const auto uiParams = m_ui.manager->getCreationParameters();
				auto* pipeline = m_ui.manager->getPipeline();
				cb->bindGraphicsPipeline(pipeline);
				cb->bindDescriptorSets(EPBP_GRAPHICS, pipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get()); // note that we use default UI pipeline layout where uiParams.resources.textures.setIx == uiParams.resources.samplers.setIx
				m_ui.manager->render(cb, waitInfo);
				cb->endRenderPass();
			}
			cb->end();
			{
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] = 
				{ 
					{
						.semaphore = m_uiSemaphore.get(),
						.value = ++m_realFrameIx,
						.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
					} 
				};
				{
					{
						const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] = 
						{ 
							{ .cmdbuf = cb } 
						};

						const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = 
						{ 
							{
								.semaphore = m_currentImageAcquire.semaphore,
								.value = m_currentImageAcquire.acquireCount,
								.stageMask = PIPELINE_STAGE_FLAGS::NONE
							} 
						};
						const IQueue::SSubmitInfo infos[] = 
						{ 
							{
								.waitSemaphores = acquired,
								.commandBuffers = commandBuffers,
								.signalSemaphores = rendered
							} 
						};

						updateGUIDescriptorSet();

						if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
							m_realFrameIx--;
					}
				}

				m_window->setCaption("[Nabla Engine] Computer Path Tracer");
				m_surface->present(m_currentImageAcquire.imageIndex, rendered);
			}
			m_api->endCapture();
		}

		inline bool keepRunning() override
		{
			if (m_surface->irrecoverable())
				return false;

			return true;
		}

		inline bool onAppTerminated() override
		{
			return device_base_t::onAppTerminated();
		}

		inline void update()
		{
			m_camera.setMoveSpeed(moveSpeed);
			m_camera.setRotateSpeed(rotateSpeed);

			static std::chrono::microseconds previousEventTimestamp{};

			m_inputSystem->getDefaultMouse(&mouse);
			m_inputSystem->getDefaultKeyboard(&keyboard);

			auto updatePresentationTimestamp = [&]()
			{
				m_currentImageAcquire = m_surface->acquireNextImage();

				m_oracle.reportEndFrameRecord();
				const auto timestamp = m_oracle.getNextPresentationTimeStamp();
				m_oracle.reportBeginFrameRecord();

				return timestamp;
			};

			const auto nextPresentationTimestamp = updatePresentationTimestamp();

			struct
			{
				std::vector<SMouseEvent> mouse{};
				std::vector<SKeyboardEvent> keyboard{};
			} capturedEvents;

			if (move) m_camera.beginInputProcessing(nextPresentationTimestamp);
			{
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
				{
					if (move)
						m_camera.mouseProcess(events); // don't capture the events, only let camera handle them with its impl

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						capturedEvents.mouse.emplace_back(e);

						if (e.type == nbl::ui::SMouseEvent::EET_SCROLL)
							gcIndex = std::clamp<uint16_t>(int16_t(gcIndex) + int16_t(core::sign(e.scrollEvent.verticalScroll)), int64_t(0), int64_t(ELG_COUNT - (uint8_t)1u));
					}
				}, m_logger.get());

			keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
				{
					if (move)
						m_camera.keyboardProcess(events); // don't capture the events, only let camera handle them with its impl

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						capturedEvents.keyboard.emplace_back(e);
					}
				}, m_logger.get());
			}
			if (move) m_camera.endInputProcessing(nextPresentationTimestamp);

			const core::SRange<const nbl::ui::SMouseEvent> mouseEvents(capturedEvents.mouse.data(), capturedEvents.mouse.data() + capturedEvents.mouse.size());
			const core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents(capturedEvents.keyboard.data(), capturedEvents.keyboard.data() + capturedEvents.keyboard.size());
			const auto cursorPosition = m_window->getCursorControl()->getPosition();

			const ext::imgui::UI::SUpdateParameters params =
			{
				.mousePosition = float32_t2(cursorPosition.x, cursorPosition.y) - float32_t2(m_window->getX(), m_window->getY()),
				.displaySize = { m_window->getWidth(), m_window->getHeight() },
				.mouseEvents = mouseEvents,
				.keyboardEvents = keyboardEvents
			};

			m_ui.manager->update(params);
		}

	private:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;

		// gpu resources
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		smart_refctd_ptr<IGPUComputePipeline> m_pipeline;
		uint64_t m_realFrameIx : 59 = 0;
		uint64_t m_maxFramesInFlight : 5;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
		smart_refctd_ptr<IGPUDescriptorSet> m_descriptorSet0, m_uboDescriptorSet1, m_descriptorSet2;

		core::smart_refctd_ptr<IDescriptorPool> m_guiDescriptorSetPool;

		// system resources
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		// pathtracer resources
		smart_refctd_ptr<IGPUImageView> m_envMapView, m_scrambleView;
		smart_refctd_ptr<IGPUBufferView> m_sequenceBufferView;
		smart_refctd_ptr<IGPUBuffer> m_ubo;
		smart_refctd_ptr<IGPUImageView> m_outImgView;

		// sync
		smart_refctd_ptr<ISemaphore> m_uiSemaphore;

		// image upload resources
		smart_refctd_ptr<ISemaphore> m_scratchSemaphore;
		SIntendedSubmitInfo m_intendedSubmit;

		struct C_UI
		{
			nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> manager;

			struct
			{
				core::smart_refctd_ptr<video::IGPUSampler> gui, scene;
			} samplers;

			core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
		} m_ui;

		Camera m_camera;
		video::CDumbPresentationOracle m_oracle;

		uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed

		bool move = false;
		float fov = 60.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
		float viewWidth = 10.f;
		float camYAngle = 165.f / 180.f * 3.14159f;
		float camXAngle = 32.f / 180.f * 3.14159f;

		bool m_firstFrame = true;
		IGPUCommandBuffer::SClearColorValue clearColor = { .float32 = {0.f,0.f,0.f,1.f} };
};

NBL_MAIN_FUNC(ComputeShaderPathtracer)
