// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
#include "nbl/this_example/transform.hpp"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/builtin/hlsl/math/thin_lens_projection.hlsl"
#include "nbl/this_example/common.hpp"
#include "nbl/this_example/builtin/build/spirv/generated/PathTracerKeys.hpp"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"
#include "nbl/builtin/hlsl/sampling/quantized_sequence.hlsl"
#include "nbl/asset/utils/ISPIRVEntryPointTrimmer.h"
#include "nbl/system/ModuleLookupUtils.h"
#include "app_resources/hlsl/render_common.hlsl"
#include "app_resources/hlsl/render_rwmc_common.hlsl"
#include "app_resources/hlsl/resolve_common.hlsl"

#include <cstddef>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <optional>
#include <thread>

#include "nlohmann/json.hpp"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;
using namespace nbl::examples;

// TODO: Add a QueryPool for timestamping once its ready
// TODO: Do buffer creation using assConv
class HLSLComputePathtracer final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
		using device_base_t = SimpleWindowedApplication;
		using asset_base_t = BuiltinResourcesApplication;
		using clock_t = std::chrono::steady_clock;

		enum E_LIGHT_GEOMETRY : uint8_t
		{
			ELG_SPHERE,
			ELG_TRIANGLE,
			ELG_RECTANGLE,
			ELG_COUNT
		};

		enum E_POLYGON_METHOD : uint8_t
		{
			EPM_AREA,
			EPM_SOLID_ANGLE,
			EPM_PROJECTED_SOLID_ANGLE,
			EPM_COUNT
		};

		constexpr static inline uint32_t2 WindowDimensions = { 1280, 720 };
		constexpr static inline uint32_t MaxFramesInFlight = 5;
		static constexpr size_t BinaryToggleCount = 2ull;
		static constexpr std::string_view BuildConfigName = PATH_TRACER_BUILD_CONFIG_NAME;
		static constexpr std::string_view RuntimeConfigFilename = "path_tracer.runtime.json";
		static inline std::string DefaultImagePathsFile = "envmap/envmap_0.exr";
		static inline std::string OwenSamplerFilePath = "owen_sampler_buffer.bin";

		const char* shaderNames[E_LIGHT_GEOMETRY::ELG_COUNT] = {
			"ELG_SPHERE",
			"ELG_TRIANGLE",
			"ELG_RECTANGLE"
		};
		const char* polygonMethodNames[EPM_COUNT] = {
			"Area",
			"Solid Angle",
			"Projected Solid Angle"
		};

	public:
		inline HLSLComputePathtracer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		inline bool isComputeOnly() const override { return false; }

		inline core::bitflag<system::ILogger::E_LOG_LEVEL> getLogLevelMask() override
		{
			return core::bitflag(system::ILogger::ELL_INFO) | system::ILogger::ELL_WARNING | system::ILogger::ELL_PERFORMANCE | system::ILogger::ELL_ERROR;
		}

		inline video::SPhysicalDeviceLimits getRequiredDeviceLimits() const override
		{
			video::SPhysicalDeviceLimits retval = device_base_t::getRequiredDeviceLimits();
			retval.storagePushConstant16 = true;
			return retval;
		}

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
			m_startupBeganAt = clock_t::now();

			// Init systems
			{
				m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

				// Remember to call the base class initialization!
				if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
					return false;
				if (!asset_base_t::onAppInitialized(std::move(system)))
					return false;

				m_semaphore = m_device->createSemaphore(m_realFrameIx);

				if (!m_semaphore)
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
			// Create command pool and buffers
			{
				auto gQueue = getGraphicsQueue();
				m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");

				if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data(), MaxFramesInFlight }))
					return logFail("Couldn't create Command Buffer!");
			}
			{
				m_scratchSemaphore = m_device->createSemaphore(0);
				if (!m_scratchSemaphore)
					return logFail("Could not create Scratch Semaphore");
				m_scratchSemaphore->setObjectDebugName("Scratch Semaphore");
				m_intendedSubmit.queue = getGraphicsQueue();
				m_intendedSubmit.waitSemaphores = {};
				m_intendedSubmit.scratchCommandBuffers = {};
				m_intendedSubmit.scratchSemaphore = {
					.semaphore = m_scratchSemaphore.get(),
					.value = 0,
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
				};
			}
			initializePipelineCache();
			ISampler::SParams samplerParams = {
				.AnisotropicFilter = 0
			};
			auto defaultSampler = m_device->createSampler(samplerParams);

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

				std::array<ICPUDescriptorSetLayout::SBinding, 4> descriptorSetBindings = {};
				std::array<IGPUDescriptorSetLayout::SBinding, 1> presentDescriptorSetBindings;

				descriptorSetBindings[0] = {
					.binding = 0u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};
				descriptorSetBindings[1] = {
					.binding = 1u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};
				descriptorSetBindings[2] = {
					.binding = 2u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};
				descriptorSetBindings[3] = {
					.binding = 3u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};

				presentDescriptorSetBindings[0] = {
					.binding = 0u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
					.immutableSamplers = &defaultSampler
				};

				auto cpuDescriptorSetLayout = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(descriptorSetBindings);

				auto gpuDescriptorSetLayout = convertDSLayoutCPU2GPU(cpuDescriptorSetLayout);
				auto gpuPresentDescriptorSetLayout = m_device->createDescriptorSetLayout(presentDescriptorSetBindings);

				auto cpuDescriptorSet = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(cpuDescriptorSetLayout));

				m_descriptorSet = convertDSCPU2GPU(cpuDescriptorSet);

				smart_refctd_ptr<IDescriptorPool> presentDSPool;
				{
					const video::IGPUDescriptorSetLayout* const layouts[] = { gpuPresentDescriptorSetLayout.get() };
					const uint32_t setCounts[] = { 1u };
					presentDSPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
				}
				m_presentDescriptorSet = presentDSPool->createDescriptorSet(gpuPresentDescriptorSetLayout);

				const uint32_t deviceMinSubgroupSize = m_device->getPhysicalDevice()->getLimits().minSubgroupSize;
				m_requiredSubgroupSize = static_cast<IPipelineBase::SUBGROUP_SIZE>(hlsl::log2(float(deviceMinSubgroupSize)));

				{
					const nbl::asset::SPushConstantRange pcRange = {
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.offset = 0,
						.size = sizeof(RenderPushConstants)
					};
					m_renderPipelineLayout = m_device->createPipelineLayout(
						{ &pcRange, 1 },
						core::smart_refctd_ptr(gpuDescriptorSetLayout),
						nullptr,
						nullptr,
						nullptr
					);
					if (!m_renderPipelineLayout)
						return logFail("Failed to create Pathtracing pipeline layout");
				}

				{
					const nbl::asset::SPushConstantRange pcRange = {
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.offset = 0,
						.size = sizeof(RenderRWMCPushConstants)
					};
					m_rwmcRenderPipelineLayout = m_device->createPipelineLayout(
						{ &pcRange, 1 },
						core::smart_refctd_ptr(gpuDescriptorSetLayout),
						nullptr,
						nullptr,
						nullptr
					);
					if (!m_rwmcRenderPipelineLayout)
						return logFail("Failed to create RWMC Pathtracing pipeline layout");
				}

				{
					const nbl::asset::SPushConstantRange pcRange = {
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.offset = 0u,
						.size = sizeof(ResolvePushConstants)
					};
					m_resolvePipelineState.layout = m_device->createPipelineLayout(
						{ &pcRange, 1 },
						core::smart_refctd_ptr(gpuDescriptorSetLayout)
					);
					if (!m_resolvePipelineState.layout)
						return logFail("Failed to create resolve pipeline layout");
				}

				const auto ensureRenderShaderLoaded = [this](const E_LIGHT_GEOMETRY geometry, const bool persistentWorkGroups, const bool rwmc) -> bool
				{
					auto& shaderSlot = m_renderPipelines.getShaders(persistentWorkGroups, rwmc)[geometry];
					if (shaderSlot)
						return true;
					shaderSlot = loadRenderShader(geometry, persistentWorkGroups, rwmc);
					return static_cast<bool>(shaderSlot);
				};
				const auto ensureResolveShaderLoaded = [this]() -> bool
				{
					if (m_resolvePipelineState.shader)
						return true;
					m_resolvePipelineState.shader = loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.resolve")>();
					return static_cast<bool>(m_resolvePipelineState.shader);
				};

				const auto startupGeometry = static_cast<E_LIGHT_GEOMETRY>(guiControlled.PTPipeline);
				if (!ensureRenderShaderLoaded(startupGeometry, guiControlled.usePersistentWorkGroups, guiControlled.useRWMC))
					return logFail("Failed to load current precompiled compute shader variant");
				if (guiControlled.useRWMC && !ensureResolveShaderLoaded())
					return logFail("Failed to load precompiled resolve compute shader");

				ensureRenderPipeline(
					startupGeometry,
					guiControlled.usePersistentWorkGroups,
					guiControlled.useRWMC,
					static_cast<E_POLYGON_METHOD>(guiControlled.polygonMethod)
				);
				if (guiControlled.useRWMC)
					ensureResolvePipeline();

				for (auto geometry = 0u; geometry < ELG_COUNT; ++geometry)
				{
					for (const auto persistentWorkGroups : { false, true })
					{
						for (const auto rwmc : { false, true })
						{
							if (!ensureRenderShaderLoaded(static_cast<E_LIGHT_GEOMETRY>(geometry), persistentWorkGroups, rwmc))
								return logFail("Failed to load precompiled compute shader variant");
						}
					}
				}
				if (!ensureResolveShaderLoaded())
					return logFail("Failed to load precompiled resolve compute shader");

				// Create graphics pipeline
				{
					auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
					ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
					if (!fsTriProtoPPln)
						return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

					auto fragmentShader = loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.misc")>();
					if (!fragmentShader)
						return logFail("Failed to Load and Compile Fragment Shader: lumaMeterShader!");

					const IGPUPipelineBase::SShaderSpecInfo fragSpec = {
						.shader = fragmentShader.get(),
					    .entryPoint = "main"
					};

					auto presentLayout = m_device->createPipelineLayout(
						{},
						core::smart_refctd_ptr(gpuPresentDescriptorSetLayout),
						nullptr,
						nullptr,
						nullptr
					);
					m_presentPipeline = fsTriProtoPPln.createPipeline(fragSpec, presentLayout.get(), scRes->getRenderpass(), 0u, {}, hlsl::SurfaceTransform::FLAG_BITS::IDENTITY_BIT, m_pipelineCache.object.get());
					if (!m_presentPipeline)
						return logFail("Could not create Graphics Pipeline!");
					m_pipelineCache.dirty = true;

				}
			}

			// load CPUImages and convert to GPUImages
			smart_refctd_ptr<IGPUImage> envMap, scrambleMap;
			{
				auto convertImgCPU2GPU = [&](std::span<ICPUImage*> cpuImgs) {
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

					std::get<CAssetConverter::SInputs::asset_span_t<ICPUImage>>(inputs.assets) = cpuImgs;
					// assert that we don't need to provide patches
					assert(cpuImgs[0]->getImageUsageFlags().hasFlags(ICPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT));
					auto reservation = converter->reserve(inputs);
					// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
					auto gpuImgs = reservation.getGPUObjects<ICPUImage>();
					for (auto& gpuImg : gpuImgs) {
						if (!gpuImg) {
							m_logger->log("Failed to convert %s into an IGPUImage handle", ILogger::ELL_ERROR, DefaultImagePathsFile);
							std::exit(-1);
						}
					}

					// and launch the conversions
					m_api->startCapture();
					auto result = reservation.convert(params);
					m_api->endCapture();
					if (!result.blocking() && result.copy() != IQueue::RESULT::SUCCESS) {
						m_logger->log("Failed to record or submit conversions", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					envMap = gpuImgs[0].value;
					scrambleMap = gpuImgs[1].value;
				};

				smart_refctd_ptr<ICPUImage> envMapCPU, scrambleMapCPU;
				{
					IAssetLoader::SAssetLoadParams lp;
					lp.workingDirectory = this->sharedInputCWD;
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
					auto texelBuffer = ICPUBuffer::create({ texelBufferSize });

					core::RandomSampler rng(0xbadc0ffeu);
					auto out = reinterpret_cast<uint32_t*>(texelBuffer->getPointer());
					for (auto index = 0u; index < texelBufferSize / 4; index++) {
						out[index] = rng.nextSample();
					}

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

					// programmatically user-created IPreHashed need to have their hash computed (loaders do it while loading)
					scrambleMapCPU->setContentHash(scrambleMapCPU->computeContentHash());
				}

				std::array<ICPUImage*, 2> cpuImgs = { envMapCPU.get(), scrambleMapCPU.get() };
				convertImgCPU2GPU(cpuImgs);
			}

			// create views for textures
			{
				auto createHDRIImage = [this](const asset::E_FORMAT colorFormat, const uint32_t width, const uint32_t height, const bool useCascadeCreationParameters = false) -> smart_refctd_ptr<IGPUImage> {
					IGPUImage::SCreationParams imgInfo;
					imgInfo.format = colorFormat;
					imgInfo.type = IGPUImage::ET_2D;
					imgInfo.extent.width = width;
					imgInfo.extent.height = height;
					imgInfo.extent.depth = 1u;
					imgInfo.mipLevels = 1u;
					imgInfo.samples = IGPUImage::ESCF_1_BIT;
					imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

					if (!useCascadeCreationParameters)
					{
						imgInfo.arrayLayers = 1u;
						imgInfo.usage = asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT | asset::IImage::EUF_SAMPLED_BIT;
					}
					else
					{
						imgInfo.arrayLayers = CascadeCount;
						imgInfo.usage = asset::IImage::EUF_STORAGE_BIT;
					}

					auto image = m_device->createImage(std::move(imgInfo));
					auto imageMemReqs = image->getMemoryReqs();
					imageMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
					m_device->allocate(imageMemReqs, image.get());

					return image;
				};
				auto createHDRIImageView = [this](smart_refctd_ptr<IGPUImage> img, const uint32_t imageArraySize = 1u, const IGPUImageView::E_TYPE imageViewType = IGPUImageView::ET_2D) -> smart_refctd_ptr<IGPUImageView>
				{
					auto format = img->getCreationParameters().format;
					IGPUImageView::SCreationParams imgViewInfo;
					imgViewInfo.image = std::move(img);
					imgViewInfo.format = format;
					imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
					imgViewInfo.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
					imgViewInfo.subresourceRange.baseArrayLayer = 0u;
					imgViewInfo.subresourceRange.baseMipLevel = 0u;
					imgViewInfo.subresourceRange.levelCount = 1u;
					imgViewInfo.viewType = imageViewType;

					imgViewInfo.subresourceRange.layerCount = imageArraySize;

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
				m_outImgView = createHDRIImageView(outImg, 1, IGPUImageView::ET_2D_ARRAY);
				m_outImgView->setObjectDebugName("Output Image View");

				auto cascade = createHDRIImage(asset::E_FORMAT::EF_R16G16B16A16_SFLOAT, WindowDimensions.x, WindowDimensions.y, true);
				cascade->setObjectDebugName("Cascade");
				m_cascadeView = createHDRIImageView(cascade, CascadeCount, IGPUImageView::ET_2D_ARRAY);
				m_cascadeView->setObjectDebugName("Cascade View");
			}

			// create sequence buffer view
			{
				// TODO: do this better use asset manager to get the ICPUBuffer from `.bin`
				auto createBufferFromCacheFile = [this](
					const system::path& filePath,
					size_t byteSize,
					void* data,
					smart_refctd_ptr<ICPUBuffer>& buffer
				) -> bool
				{
					ISystem::future_t<smart_refctd_ptr<nbl::system::IFile>> owenSamplerFileFuture;
					ISystem::future_t<size_t> owenSamplerFileReadFuture;
					size_t owenSamplerFileBytesRead = 0ull;

					m_system->createFile(owenSamplerFileFuture, filePath, IFile::ECF_READ);
					smart_refctd_ptr<IFile> owenSamplerFile;

					if (owenSamplerFileFuture.wait())
					{
						owenSamplerFileFuture.acquire().move_into(owenSamplerFile);
						if (!owenSamplerFile)
							return false;

						owenSamplerFile->read(owenSamplerFileReadFuture, data, 0, byteSize);
						if (owenSamplerFileReadFuture.wait())
						{
							owenSamplerFileReadFuture.acquire().move_into(owenSamplerFileBytesRead);

							if (owenSamplerFileBytesRead < byteSize)
								return false;

							buffer = asset::ICPUBuffer::create({ { byteSize }, data });
							return true;
						}
					}

					return false;
				};
				auto writeBufferIntoCacheFile = [this](const system::path& filePath, size_t byteSize, const void* data)
				{
					std::filesystem::create_directories(filePath.parent_path());

					ISystem::future_t<smart_refctd_ptr<nbl::system::IFile>> owenSamplerFileFuture;
					ISystem::future_t<size_t> owenSamplerFileWriteFuture;
					size_t owenSamplerFileBytesWritten = 0ull;

					m_system->createFile(owenSamplerFileFuture, filePath, IFile::ECF_WRITE);
					if (!owenSamplerFileFuture.wait())
						return;

					smart_refctd_ptr<IFile> file;
					owenSamplerFileFuture.acquire().move_into(file);
					if (!file)
						return;

					file->write(owenSamplerFileWriteFuture, const_cast<void*>(data), 0, byteSize);
					if (owenSamplerFileWriteFuture.wait())
						owenSamplerFileWriteFuture.acquire().move_into(owenSamplerFileBytesWritten);
				};

				constexpr uint32_t quantizedDimensions = MaxBufferDimensions / 3u;
				using sequence_type = sampling::QuantizedSequence<uint32_t2, 3>;
				constexpr size_t sequenceCount = quantizedDimensions * MaxSamplesBuffer;
				constexpr size_t sequenceByteSize = sequenceCount * sizeof(sequence_type);
				std::array<sequence_type, sequenceCount> data = {};
				smart_refctd_ptr<ICPUBuffer> sampleSeq;

				const auto packagedOwenSamplerPath = sharedInputCWD / OwenSamplerFilePath;
				const auto generatedOwenSamplerPath = sharedOutputCWD / OwenSamplerFilePath;
				const bool cacheLoaded =
					createBufferFromCacheFile(packagedOwenSamplerPath, sequenceByteSize, data.data(), sampleSeq) ||
					createBufferFromCacheFile(generatedOwenSamplerPath, sequenceByteSize, data.data(), sampleSeq);
				if (!cacheLoaded)
				{
					core::OwenSampler sampler(MaxBufferDimensions, 0xdeadbeefu);

					ICPUBuffer::SCreationParams params = {};
					params.size = sequenceByteSize;
					sampleSeq = ICPUBuffer::create(std::move(params));

					auto out = reinterpret_cast<sequence_type*>(sampleSeq->getPointer());
					for (auto dim = 0u; dim < MaxBufferDimensions; dim++)
						for (uint32_t i = 0; i < MaxSamplesBuffer; i++)
						{
							const uint32_t quant_dim = dim / 3u;
							const uint32_t offset = dim % 3u;
							auto& seq = out[i * quantizedDimensions + quant_dim];
							const uint32_t sample = sampler.sample(dim, i);
							seq.set(offset, sample);
						}
					writeBufferIntoCacheFile(generatedOwenSamplerPath, sequenceByteSize, out);
				}

				IGPUBuffer::SCreationParams params = {};
				params.usage = asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
				params.size = sampleSeq->getSize();

				auto queue = getGraphicsQueue();
				auto cmdbuf = m_cmdBufs[0].get();
				cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
				IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = { cmdbuf };
				m_intendedSubmit.queue = queue;
				m_intendedSubmit.scratchCommandBuffers = { &cmdbufInfo, 1 };
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

				auto bufferFuture = m_utils->createFilledDeviceLocalBufferOnDedMem(
					m_intendedSubmit,
					std::move(params),
					sampleSeq->getPointer()
				);
				bufferFuture.wait();
				const auto uploadedBuffer = bufferFuture.get();
				if (!uploadedBuffer || !uploadedBuffer->get())
					return logFail("Failed to upload sequence buffer");
				m_sequenceBuffer = smart_refctd_ptr<IGPUBuffer>(*uploadedBuffer);

				m_sequenceBuffer->setObjectDebugName("Sequence buffer");
			}

			// Update Descriptors
			{
				ISampler::SParams samplerParams0 = {
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
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
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
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
				writeDSInfos[1].desc = m_cascadeView;
				writeDSInfos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
				writeDSInfos[2].desc = m_envMapView;
				// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
				writeDSInfos[2].info.combinedImageSampler.sampler = sampler0;
				writeDSInfos[2].info.combinedImageSampler.imageLayout = asset::IImage::LAYOUT::READ_ONLY_OPTIMAL;
				writeDSInfos[3].desc = m_scrambleView;
				// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
				writeDSInfos[3].info.combinedImageSampler.sampler = sampler1;
				writeDSInfos[3].info.combinedImageSampler.imageLayout = asset::IImage::LAYOUT::READ_ONLY_OPTIMAL;
				writeDSInfos[4].desc = m_outImgView;
				writeDSInfos[4].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

				std::array<IGPUDescriptorSet::SWriteDescriptorSet, 5> writeDescriptorSets = {};
				writeDescriptorSets[0] = {
					.dstSet = m_descriptorSet.get(),
					.binding = 2,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[0]
				};
				writeDescriptorSets[1] = {
					.dstSet = m_descriptorSet.get(),
					.binding = 3,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[1]
				};
				writeDescriptorSets[2] = {
					.dstSet = m_descriptorSet.get(),
					.binding = 0,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[2]
				};
				writeDescriptorSets[3] = {
					.dstSet = m_descriptorSet.get(),
					.binding = 1,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[3]
				};
				writeDescriptorSets[4] = {
					.dstSet = m_presentDescriptorSet.get(),
					.binding = 0,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[4]
				};

				m_device->updateDescriptorSets(writeDescriptorSets, {});
			}

			// Create ui descriptors
			{
				using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
				{
					IGPUSampler::SParams params;
					params.AnisotropicFilter = 1u;
					params.TextureWrapU = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
					params.TextureWrapV = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
					params.TextureWrapW = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;

					m_ui.samplers.gui = m_device->createSampler(params);
					m_ui.samplers.gui->setObjectDebugName("Nabla IMGUI UI Sampler");
				}

				std::array<core::smart_refctd_ptr<IGPUSampler>, 69u> immutableSamplers;
				for (auto& it : immutableSamplers)
					it = smart_refctd_ptr(m_ui.samplers.scene);

				immutableSamplers[nbl::ext::imgui::UI::FontAtlasTexId] = smart_refctd_ptr(m_ui.samplers.gui);

				nbl::ext::imgui::UI::SCreationParameters params;

				params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
				params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
				params.assetManager = m_assetMgr;
				params.pipelineCache = m_pipelineCache.object;
				params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, MaxUITextureCount);
				params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
				params.streamingBuffer = nullptr;
				params.subpassIx = 0u;
				params.transfer = getTransferUpQueue();
				params.utilities = m_utils;
				params.spirv = nbl::ext::imgui::UI::SCreationParameters::PrecompiledShaders{
					.vertex = loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.misc")>(),
					.fragment = loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.misc")>()
				};
				if (!params.spirv->vertex || !params.spirv->fragment)
					return logFail("Failed to load precompiled ImGui shaders");
				{
					m_ui.manager = ext::imgui::UI::create(std::move(params));
					if (m_ui.manager)
						m_pipelineCache.dirty = true;

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
					ImGuizmo::SetOrthographic(false);
					ImGuizmo::BeginFrame();
					ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, ImGui::GetWindowWidth(), ImGui::GetWindowHeight());

					const auto aspectRatio = io.DisplaySize.x / io.DisplaySize.y;
					m_camera.setProjectionMatrix(hlsl::math::thin_lens::rhPerspectiveFovMatrix<float>(hlsl::radians(guiControlled.fov), aspectRatio, guiControlled.zNear, guiControlled.zFar));

					const ImGuiViewport* viewport = ImGui::GetMainViewport();
					const ImVec2 viewportPos = viewport->Pos;
					const ImVec2 viewportSize = viewport->Size;
					const ImGuiStyle& style = ImGui::GetStyle();
					const float panelMargin = 10.f;
					const auto currentGeometry = static_cast<E_LIGHT_GEOMETRY>(guiControlled.PTPipeline);
					const auto requestedMethod = static_cast<E_POLYGON_METHOD>(guiControlled.polygonMethod);
					const auto currentVariant = getRenderVariantInfo(currentGeometry, guiControlled.usePersistentWorkGroups, requestedMethod);
					const size_t readyRenderPipelines = getReadyRenderPipelineCount();
					const size_t totalRenderPipelines = getKnownRenderPipelineCount();
					const size_t readyTotalPipelines = readyRenderPipelines + (m_resolvePipelineState.pipeline ? 1ull : 0ull);
					const size_t totalKnownPipelines = totalRenderPipelines + 1ull;
					const size_t runningPipelineBuilds = getRunningPipelineBuildCount();
					const size_t queuedPipelineBuilds = m_pipelineCache.warmup.queue.size();
					const bool warmupInProgress = m_hasPathtraceOutput && !m_pipelineCache.warmup.loggedComplete;
					const char* const effectiveEntryPoint = currentVariant.entryPoint;
					struct SFloatSliderRow
					{
						const char* label;
						float* value;
						float min;
						float max;
						const char* format;
					};
					struct SIntSliderRow
					{
						const char* label;
						int* value;
						int min;
						int max;
					};
					struct SCheckboxRow
					{
						const char* label;
						bool* value;
					};
					struct SComboRow
					{
						const char* label;
						int* value;
						const char* const* items;
						int count;
					};
					struct STextRow
					{
						const char* label;
						std::string value;
					};
					const auto calcMaxTextWidth = [](const auto& items, auto&& toText) -> float
					{
						float width = 0.f;
						for (const auto& item : items)
							width = std::max(width, ImGui::CalcTextSize(toText(item)).x);
						return width;
					};
					const auto makeReadyText = [](const size_t ready, const size_t total) -> std::string
					{
						return std::to_string(ready) + "/" + std::to_string(total);
					};
					const auto makeRunQueueText = [](const size_t running, const size_t queued) -> std::string
					{
						return std::to_string(running) + " / " + std::to_string(queued);
					};
					const std::string pipelineStatusText = !m_hasPathtraceOutput ?
						"Building pipeline..." :
						(warmupInProgress ?
							("Warmup " + std::to_string(readyTotalPipelines) + "/" + std::to_string(totalKnownPipelines)) :
							"All pipelines ready");
					const std::string cacheStateText = m_pipelineCache.loadedFromDisk ? "loaded from disk" : "cold start";
					const std::string trimCacheText = std::to_string(m_pipelineCache.trimmedShaders.loadedFromDiskCount + m_pipelineCache.trimmedShaders.generatedCount) + " ready";
					const std::string parallelismText = std::to_string(m_pipelineCache.warmup.budget);
					const std::string renderStateText = makeReadyText(readyTotalPipelines, totalKnownPipelines);
					const std::string warmupStateText = makeRunQueueText(runningPipelineBuilds, queuedPipelineBuilds);
					const std::string cursorText = "cursor " + std::to_string(static_cast<int>(io.MousePos.x)) + " " + std::to_string(static_cast<int>(io.MousePos.y));
					const SFloatSliderRow cameraFloatRows[] = {
						{ "move", &guiControlled.moveSpeed, 0.1f, 10.f, "%.2f" },
						{ "rotate", &guiControlled.rotateSpeed, 0.1f, 10.f, "%.2f" },
						{ "fov", &guiControlled.fov, 20.f, 150.f, "%.0f" },
						{ "zNear", &guiControlled.zNear, 0.1f, 100.f, "%.2f" },
						{ "zFar", &guiControlled.zFar, 110.f, 10000.f, "%.0f" },
					};
					const SComboRow renderComboRows[] = {
						{ "shader", &guiControlled.PTPipeline, shaderNames, E_LIGHT_GEOMETRY::ELG_COUNT },
						{ "method", &guiControlled.polygonMethod, polygonMethodNames, EPM_COUNT },
					};
					const SIntSliderRow renderIntRows[] = {
						{ "spp", &guiControlled.spp, 1, MaxSamplesBuffer },
						{ "depth", &guiControlled.depth, 1, MaxBufferDimensions / 4 },
					};
					const SCheckboxRow renderCheckboxRows[] = {
						{ "persistent WG", &guiControlled.usePersistentWorkGroups },
					};
					const SCheckboxRow rwmcCheckboxRows[] = {
						{ "enable", &guiControlled.useRWMC },
					};
					const SFloatSliderRow rwmcFloatRows[] = {
						{ "start", &guiControlled.rwmcParams.start, 1.0f, 32.0f, "%.3f" },
						{ "base", &guiControlled.rwmcParams.base, 1.0f, 32.0f, "%.3f" },
						{ "min rel.", &guiControlled.rwmcParams.minReliableLuma, 0.1f, 1024.0f, "%.3f" },
						{ "kappa", &guiControlled.rwmcParams.kappa, 0.1f, 1024.0f, "%.3f" },
					};
					const STextRow diagnosticsRows[] = {
						{ "geometry", shaderNames[currentGeometry] },
						{ "req. method", polygonMethodNames[requestedMethod] },
						{ "eff. method", polygonMethodNames[currentVariant.effectiveMethod] },
						{ "entrypoint", effectiveEntryPoint },
						{ "mode", PathTracerBuildModeName },
						{ "config", std::string(BuildConfigName) },
						{ "cache", cacheStateText },
						{ "trim cache", trimCacheText },
						{ "parallel", parallelismText },
						{ "render", renderStateText },
						{ "run/queue", warmupStateText },
					};
					const char* const standaloneTexts[] = {
						"PATH_TRACER",
						"Home camera  End light",
						pipelineStatusText.c_str(),
						cursorText.c_str(),
					};
					const char* const sliderPreviewTexts[] = {
						"10000.000",
						"1024.000",
						effectiveEntryPoint,
						PathTracerBuildModeName,
						BuildConfigName.data(),
						cacheStateText.c_str(),
						renderStateText.c_str(),
						warmupStateText.c_str(),
					};
					const float maxStandaloneTextWidth = calcMaxTextWidth(standaloneTexts, [](const char* text) { return text; });
					const float maxLabelTextWidth = std::max({
						calcMaxTextWidth(cameraFloatRows, [](const auto& row) { return row.label; }),
						calcMaxTextWidth(renderComboRows, [](const auto& row) { return row.label; }),
						calcMaxTextWidth(renderIntRows, [](const auto& row) { return row.label; }),
						calcMaxTextWidth(renderCheckboxRows, [](const auto& row) { return row.label; }),
						calcMaxTextWidth(rwmcCheckboxRows, [](const auto& row) { return row.label; }),
						calcMaxTextWidth(rwmcFloatRows, [](const auto& row) { return row.label; }),
						calcMaxTextWidth(diagnosticsRows, [](const auto& row) { return row.label; })
					});
					const float comboPreviewWidth = std::max(
						calcMaxTextWidth(shaderNames, [](const char* text) { return text; }),
						calcMaxTextWidth(polygonMethodNames, [](const char* text) { return text; })
					);
					const float sliderPreviewWidth = calcMaxTextWidth(sliderPreviewTexts, [](const char* text) { return text; });
					const float tableLabelColumnWidth = std::ceil(maxLabelTextWidth + style.FramePadding.x * 2.f + style.CellPadding.x * 2.f);
					const float tableValueColumnMinWidth =
						std::ceil(std::max(comboPreviewWidth, sliderPreviewWidth) + style.FramePadding.x * 2.f + style.ItemInnerSpacing.x + ImGui::GetFrameHeight() + 18.f);
					const float sectionTableWidth = tableLabelColumnWidth + tableValueColumnMinWidth + style.CellPadding.x * 4.f + style.ItemSpacing.x;
					const float contentWidth = std::max(maxStandaloneTextWidth, sectionTableWidth);
					const float panelWidth = std::min(
						std::ceil(contentWidth + style.WindowPadding.x * 2.f),
						std::max(0.f, viewportSize.x - panelMargin * 2.f)
					);
					const float panelMaxHeight = ImMax(300.0f, viewportSize.y * 0.84f);
					ImGui::SetNextWindowPos(ImVec2(viewportPos.x + panelMargin, viewportPos.y + panelMargin), ImGuiCond_Always);
					ImGui::SetNextWindowSizeConstraints(ImVec2(panelWidth, 0.0f), ImVec2(panelWidth, panelMaxHeight));
					ImGui::SetNextWindowBgAlpha(0.72f);
					ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.f, 5.f));
					ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.f);
					ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.f);
					ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 4.f);
					ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(5.f, 2.f));
					ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.10f, 0.13f, 0.88f));
					ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.32f, 0.39f, 0.47f, 0.65f));
					ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.18f, 0.28f, 0.36f, 0.92f));
					ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.24f, 0.36f, 0.46f, 0.96f));
					ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.28f, 0.42f, 0.54f, 1.0f));

					const ImGuiWindowFlags panelFlags =
						ImGuiWindowFlags_NoDecoration |
						ImGuiWindowFlags_NoMove |
						ImGuiWindowFlags_NoSavedSettings |
						ImGuiWindowFlags_NoNav |
						ImGuiWindowFlags_AlwaysAutoResize |
						ImGuiWindowFlags_NoResize;

					const auto beginSectionTable = [](const char* id) -> bool
					{
						return ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingFixedFit);
					};
					const auto setupSectionTable = [tableLabelColumnWidth]() -> void
					{
						ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, tableLabelColumnWidth);
						ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
					};
					const auto sliderFloatRow = [](const SFloatSliderRow& row) -> void
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::TextUnformatted(row.label);
						ImGui::TableSetColumnIndex(1);
						ImGui::SetNextItemWidth(-FLT_MIN);
						ImGui::PushID(row.label);
						ImGui::SliderFloat("##value", row.value, row.min, row.max, row.format, ImGuiSliderFlags_AlwaysClamp);
						ImGui::PopID();
					};
					const auto sliderIntRow = [](const SIntSliderRow& row) -> void
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::TextUnformatted(row.label);
						ImGui::TableSetColumnIndex(1);
						ImGui::SetNextItemWidth(-FLT_MIN);
						ImGui::PushID(row.label);
						ImGui::SliderInt("##value", row.value, row.min, row.max);
						ImGui::PopID();
					};
					const auto comboRow = [](const SComboRow& row) -> void
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::TextUnformatted(row.label);
						ImGui::TableSetColumnIndex(1);
						ImGui::SetNextItemWidth(-FLT_MIN);
						ImGui::PushID(row.label);
						ImGui::Combo("##value", row.value, row.items, row.count);
						ImGui::PopID();
					};
					const auto checkboxRow = [](const SCheckboxRow& row) -> void
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::TextUnformatted(row.label);
						ImGui::TableSetColumnIndex(1);
						ImGui::PushID(row.label);
						ImGui::Checkbox("##value", row.value);
						ImGui::PopID();
					};
					const auto textRow = [](const STextRow& row) -> void
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::TextUnformatted(row.label);
						ImGui::TableSetColumnIndex(1);
						ImGui::TextUnformatted(row.value.c_str());
					};

					if (ImGui::Begin("Path Tracer Controls", nullptr, panelFlags))
					{
						ImGui::TextUnformatted("PATH_TRACER");
						ImGui::Separator();
						ImGui::TextDisabled("Home camera  End light");
						if (!m_hasPathtraceOutput)
							ImGui::TextColored(ImVec4(0.83f, 0.86f, 0.90f, 1.0f), "Building pipeline...");
						else if (warmupInProgress)
							ImGui::TextColored(ImVec4(0.83f, 0.86f, 0.90f, 1.0f), "Warmup %zu/%zu", readyTotalPipelines, totalKnownPipelines);
						else
							ImGui::TextDisabled("All pipelines ready");
						ImGui::Dummy(ImVec2(0.f, 2.f));

						if (ImGui::CollapsingHeader("Camera"))
						{
							if (beginSectionTable("##camera_controls_table"))
							{
								setupSectionTable();
								for (const auto& row : cameraFloatRows)
									sliderFloatRow(row);
								ImGui::EndTable();
							}
						}

						if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen))
						{
							if (beginSectionTable("##render_controls_table"))
							{
								setupSectionTable();
								for (const auto& row : renderComboRows)
									comboRow(row);
								for (const auto& row : renderIntRows)
									sliderIntRow(row);
								for (const auto& row : renderCheckboxRows)
									checkboxRow(row);
								ImGui::EndTable();
							}
						}

						if (ImGui::CollapsingHeader("RWMC", ImGuiTreeNodeFlags_DefaultOpen))
						{
							if (beginSectionTable("##rwmc_controls_table"))
							{
								setupSectionTable();
								for (const auto& row : rwmcCheckboxRows)
									checkboxRow(row);
								for (const auto& row : rwmcFloatRows)
									sliderFloatRow(row);
								ImGui::EndTable();
							}
						}

						if (ImGui::CollapsingHeader("Diagnostics"))
						{
							if (beginSectionTable("##diagnostics_controls_table"))
							{
								setupSectionTable();
								for (const auto& row : diagnosticsRows)
									textRow(row);
								ImGui::EndTable();
							}
						}

						ImGui::Dummy(ImVec2(0.f, 2.f));
						ImGui::Separator();
						ImGui::TextDisabled("%s", cursorText.c_str());
					}
					ImGui::End();

					if (!m_hasPathtraceOutput || warmupInProgress)
					{
						ImGui::SetNextWindowPos(ImVec2(viewportPos.x + viewportSize.x - panelMargin, viewportPos.y + panelMargin), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
						ImGui::SetNextWindowBgAlpha(0.62f);
						ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.f, 10.f));
						ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.f);
						ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.07f, 0.09f, 0.12f, 0.90f));
						const ImGuiWindowFlags overlayFlags =
							ImGuiWindowFlags_NoDecoration |
							ImGuiWindowFlags_NoSavedSettings |
							ImGuiWindowFlags_NoMove |
							ImGuiWindowFlags_NoNav |
							ImGuiWindowFlags_AlwaysAutoResize |
							ImGuiWindowFlags_NoInputs;
						if (ImGui::Begin("##path_tracer_status_overlay", nullptr, overlayFlags))
						{
							ImGui::TextUnformatted(pipelineStatusText.c_str());
							ImGui::Text("Run %zu  Queue %zu", runningPipelineBuilds, queuedPipelineBuilds);
							ImGui::Text("Cache: %s", m_pipelineCache.loadedFromDisk ? "disk" : "cold");
						}
						ImGui::End();
						ImGui::PopStyleColor(1);
						ImGui::PopStyleVar(2);
					}
					ImGui::PopStyleColor(5);
					ImGui::PopStyleVar(5);
				}
			);

			m_ui.manager->registerListener(
				[this]() -> void {
					static struct
					{
						hlsl::float32_t4x4 view, projection;
					} imguizmoM16InOut;

					ImGuizmo::SetID(0u);

					imguizmoM16InOut.view = hlsl::transpose(math::linalg::promoted_mul(float32_t4x4(1.f), m_camera.getViewMatrix()));
					imguizmoM16InOut.projection = hlsl::transpose(m_camera.getProjectionMatrix());
					imguizmoM16InOut.projection[1][1] *= -1.f; // https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/	

					m_transformParams.editTransformDecomposition = true;
					m_transformParams.sceneTexDescIx = 1u;

					if (ImGui::IsKeyPressed(ImGuiKey_End))
					{
						m_lightModelMatrix = hlsl::float32_t4x4(
							0.3f, 0.0f, 0.0f, 0.0f,
							0.0f, 0.3f, 0.0f, 0.0f,
							0.0f, 0.0f, 0.3f, 0.0f,
							-1.0f, 1.5f, 0.0f, 1.0f
						);
					}

					if (E_LIGHT_GEOMETRY::ELG_SPHERE == guiControlled.PTPipeline)
					{
						m_transformParams.allowedOp = ImGuizmo::OPERATION::TRANSLATE | ImGuizmo::OPERATION::SCALEU;
						m_transformParams.isSphere = true;
					}
					else
					{
						m_transformParams.allowedOp = ImGuizmo::OPERATION::TRANSLATE | ImGuizmo::OPERATION::ROTATE | ImGuizmo::OPERATION::SCALE;
						m_transformParams.isSphere = false;
					}
					EditTransform(&imguizmoM16InOut.view[0][0], &imguizmoM16InOut.projection[0][0], &m_lightModelMatrix[0][0], m_transformParams);

					if (E_LIGHT_GEOMETRY::ELG_SPHERE == guiControlled.PTPipeline)
					{
						// keep uniform scale for sphere
						float32_t uniformScale = (m_lightModelMatrix[0][0] + m_lightModelMatrix[1][1] + m_lightModelMatrix[2][2]) / 3.0f;
						m_lightModelMatrix[0][0] = uniformScale;
						m_lightModelMatrix[1][1] = uniformScale; // Doesn't affect sphere but will affect rectangle/triangle if switching shapes
						m_lightModelMatrix[2][2] = uniformScale;
					}

				}
			);

			// Set Camera
			{
				core::vectorSIMDf cameraPosition(0, 5, -10);
				const auto proj = hlsl::math::thin_lens::rhPerspectiveFovMatrix<float>(hlsl::radians(guiControlled.fov), WindowDimensions.x / WindowDimensions.y, guiControlled.zNear, guiControlled.zFar);
				m_camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), proj);
			}
			m_showUI = true;

			m_winMgr->setWindowSize(m_window.get(), WindowDimensions.x, WindowDimensions.y);
			m_surface->recreateSwapchain();
			m_winMgr->show(m_window.get());
			m_oracle.reportBeginFrameRecord();
			m_camera.mapKeysToArrows();

			// set initial rwmc settings
			
			guiControlled.rwmcParams.start = hlsl::dot<float32_t3>(hlsl::transpose(colorspace::scRGBtoXYZ)[1], LightEminence);
			guiControlled.rwmcParams.base = 8.0f;
			guiControlled.rwmcParams.minReliableLuma = 1.0f;
			guiControlled.rwmcParams.kappa = 5.0f;
			return true;
		}

		bool updateGUIDescriptorSet()
		{
			// texture atlas, note we don't create info & write pair for the font sampler because UI extension's is immutable and baked into DS layout
			static std::array<IGPUDescriptorSet::SDescriptorInfo, MaxUITextureCount> descriptorInfo;
			static IGPUDescriptorSet::SWriteDescriptorSet writes[MaxUITextureCount];

			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].desc = smart_refctd_ptr<IGPUImageView>(m_ui.manager->getFontAtlasView());

			for (uint32_t i = 0; i < descriptorInfo.size(); ++i)
			{
				writes[i].dstSet = m_ui.descriptorSet.get();
				writes[i].binding = 0u;
				writes[i].arrayElement = i;
				writes[i].count = 1u;
			}
			writes[nbl::ext::imgui::UI::FontAtlasTexId].info = descriptorInfo.data() + nbl::ext::imgui::UI::FontAtlasTexId;

			return m_device->updateDescriptorSets(writes, {});
		}

		inline void workLoopBody() override
		{
			pollPendingPipelines();
			pumpPipelineWarmup();
			if (!m_loggedFirstFrameLoop)
			{
				logStartupEvent("first_frame_loop");
				m_loggedFirstFrameLoop = true;
			}

			// framesInFlight: ensuring safe execution of command buffers and acquires, `framesInFlight` only affect semaphore waits, don't use this to index your resources because it can change with swapchain recreation.
			const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
			// We block for semaphores for 2 reasons here:
				// A) Resource: Can't use resource like a command buffer BEFORE previous use is finished! [MaxFramesInFlight]
				// B) Acquire: Can't have more acquires in flight than a certain threshold returned by swapchain or your surface helper class. [MaxAcquiresInFlight]
			if (m_realFrameIx >= framesInFlight)
			{
				const ISemaphore::SWaitInfo cbDonePending[] = 
				{
					{
						.semaphore = m_semaphore.get(),
						.value = m_realFrameIx + 1 - framesInFlight
					}
				};
				if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}
			const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

			// CPU events
			update();

			auto queue = getGraphicsQueue();
			auto cmdbuf = m_cmdBufs[resourceIx].get();

			if (!keepRunning())
				return;

			cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);

			// safe to proceed
			// upload buffer data
			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->beginDebugMarker("ComputeShaderPathtracer IMGUI Frame");

			RenderRWMCPushConstants rwmcPushConstants;
			ResolvePushConstants resolvePushConstants;
			RenderPushConstants pc;
			auto updatePathtracerPushConstants = [&]() -> void {
				// disregard surface/swapchain transformation for now
				const float32_t4x4 viewProjectionMatrix = m_camera.getConcatenatedMatrix();
				const float32_t3x4 modelMatrix = hlsl::math::linalg::identity<hlsl::float32_t3x4>();

				const float32_t4x4 modelViewProjectionMatrix = nbl::hlsl::math::linalg::promoted_mul(viewProjectionMatrix, modelMatrix);
				const float32_t4x4 invMVP = hlsl::inverse(modelViewProjectionMatrix);

				if (guiControlled.useRWMC)
				{
					rwmcPushConstants.renderPushConstants.invMVP = invMVP;
					rwmcPushConstants.renderPushConstants.generalPurposeLightMatrix = hlsl::float32_t3x4(transpose(m_lightModelMatrix));
					rwmcPushConstants.renderPushConstants.depth = guiControlled.depth;
					rwmcPushConstants.renderPushConstants.sampleCount = guiControlled.rwmcParams.sampleCount = guiControlled.spp;
					rwmcPushConstants.renderPushConstants.polygonMethod = guiControlled.polygonMethod;
					rwmcPushConstants.renderPushConstants.pSampleSequence = m_sequenceBuffer->getDeviceAddress();
					rwmcPushConstants.splattingParameters = rwmc::SPackedSplattingParameters::create(guiControlled.rwmcParams.base, guiControlled.rwmcParams.start, CascadeCount);
				}
				else
				{
					pc.invMVP = invMVP;
					pc.generalPurposeLightMatrix = hlsl::float32_t3x4(transpose(m_lightModelMatrix));
					pc.sampleCount = guiControlled.spp;
					pc.depth = guiControlled.depth;
					pc.polygonMethod = guiControlled.polygonMethod;
					pc.pSampleSequence = m_sequenceBuffer->getDeviceAddress();
				}
			};
			updatePathtracerPushConstants();
			bool producedRenderableOutput = false;

			// TRANSITION m_outImgView to GENERAL (because of descriptorSets0 -> ComputeShader Writes into the image)
			{
				const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers[] = {
					{
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS,
								.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
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
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers });
			}

			// transit m_cascadeView layout to GENERAL, block until previous shader is done with reading from the cascade
			if(guiControlled.useRWMC)
			{
				const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> cascadeBarrier[] = {
						{
							.barrier = {
								.dep = {
									.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
									.srcAccessMask = ACCESS_FLAGS::NONE,
									.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
									.dstAccessMask = ACCESS_FLAGS::NONE
								}
							},
							.image = m_cascadeView->getCreationParameters().image.get(),
							.subresourceRange = {
								.aspectMask = IImage::EAF_COLOR_BIT,
								.baseMipLevel = 0u,
								.levelCount = 1u,
								.baseArrayLayer = 0u,
								.layerCount = CascadeCount
							},
							.oldLayout = IImage::LAYOUT::UNDEFINED,
							.newLayout = IImage::LAYOUT::GENERAL
						}
				};
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = cascadeBarrier });
			}

			{
				// TODO: shouldn't it be computed only at initialization stage and on window resize?
				const uint32_t dispatchSize = guiControlled.usePersistentWorkGroups ?
					m_physicalDevice->getLimits().computeOptimalPersistentWorkgroupDispatchSize(WindowDimensions.x * WindowDimensions.y, RenderWorkgroupSize) :
					1 + (WindowDimensions.x * WindowDimensions.y - 1) / RenderWorkgroupSize;

				IGPUComputePipeline* pipeline = pickPTPipeline();
				if (pipeline)
				{
					cmdbuf->bindComputePipeline(pipeline);
					cmdbuf->bindDescriptorSets(EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &m_descriptorSet.get());

					const uint32_t pushConstantsSize = guiControlled.useRWMC ? sizeof(RenderRWMCPushConstants) : sizeof(RenderPushConstants);
					const void* pushConstantsPtr = guiControlled.useRWMC ? reinterpret_cast<const void*>(&rwmcPushConstants) : reinterpret_cast<const void*>(&pc);
					cmdbuf->pushConstants(pipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, pushConstantsSize, pushConstantsPtr);

					cmdbuf->dispatch(dispatchSize, 1u, 1u);
					producedRenderableOutput = !guiControlled.useRWMC;
				}
			}

			// m_cascadeView synchronization - wait for previous compute shader to write into the cascade
			// TODO: create this and every other barrier once outside of the loop?
			if(guiControlled.useRWMC)
			{
				const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> cascadeBarrier[] = {
						{
							.barrier = {
								.dep = {
									.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
									.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
									.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
									.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
								}
							},
							.image = m_cascadeView->getCreationParameters().image.get(),
							.subresourceRange = {
								.aspectMask = IImage::EAF_COLOR_BIT,
								.baseMipLevel = 0u,
								.levelCount = 1u,
								.baseArrayLayer = 0u,
								.layerCount = CascadeCount
							}
						}
				};
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = cascadeBarrier });

				// resolve
				const uint32_t2 dispatchSize = uint32_t2(	// Round up division
					(m_window->getWidth() + ResolveWorkgroupSizeX - 1) / ResolveWorkgroupSizeX,
					(m_window->getHeight() + ResolveWorkgroupSizeY - 1) / ResolveWorkgroupSizeY
				);

				IGPUComputePipeline* pipeline = ensureResolvePipeline();
				if (pipeline)
				{
					resolvePushConstants.resolveParameters = rwmc::SResolveParameters::create(guiControlled.rwmcParams);

					cmdbuf->bindComputePipeline(pipeline);
					cmdbuf->bindDescriptorSets(EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &m_descriptorSet.get());
					cmdbuf->pushConstants(pipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(ResolvePushConstants), &resolvePushConstants);

					cmdbuf->dispatch(dispatchSize.x, dispatchSize.y, 1u);
					producedRenderableOutput = true;
				}
			}

			if (producedRenderableOutput)
			{
				m_hasPathtraceOutput = true;
				if (!m_loggedFirstRenderDispatch)
				{
					logStartupEvent("first_render_dispatch");
					m_loggedFirstRenderDispatch = true;
				}
			}

			// TRANSITION m_outImgView to READ (because of descriptorSets0 -> ComputeShader Writes into the image)
			{
				const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers[] = {
					{
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
								.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
								.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
								.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
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
						.oldLayout = IImage::LAYOUT::GENERAL,
						.newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL
					}
				};
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers });
			}

			// TODO: tone mapping and stuff

			asset::SViewport viewport;
			{
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = WindowDimensions.x;
				viewport.height = WindowDimensions.y;
			}
			cmdbuf->setViewport(0u, 1u, &viewport);


			VkRect2D defaultScisors[] = { {.offset = {(int32_t)viewport.x, (int32_t)viewport.y}, .extent = {(uint32_t)viewport.width, (uint32_t)viewport.height}} };
			cmdbuf->setScissor(defaultScisors);

			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};
			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());

			// Upload m_outImg to swapchain + UI
			{
				const IGPUCommandBuffer::SRenderpassBeginInfo info =
				{
					.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
					.colorClearValues = &clearColor,
					.depthStencilClearValues = nullptr,
					.renderArea = currentRenderArea
				};
				nbl::video::ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };

				cmdbuf->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

				if (m_hasPathtraceOutput)
				{
					cmdbuf->bindGraphicsPipeline(m_presentPipeline.get());
					cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, m_presentPipeline->getLayout(), 0, 1u, &m_presentDescriptorSet.get());
					ext::FullScreenTriangle::recordDrawCall(cmdbuf);
				}

				if (m_showUI)
				{
					const auto uiParams = m_ui.manager->getCreationParameters();
					auto* uiPipeline = m_ui.manager->getPipeline();
					cmdbuf->bindGraphicsPipeline(uiPipeline);
					cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, uiPipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get());
					m_ui.manager->render(cmdbuf, waitInfo);
				}

				cmdbuf->endRenderPass();
			}

			cmdbuf->end();
			{
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
				{
					{
						.semaphore = m_semaphore.get(),
						.value = ++m_realFrameIx,
						.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
					}
				};
				{
					{
						const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
						{
							{.cmdbuf = cmdbuf }
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

						m_api->startCapture();
						if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
							m_realFrameIx--;
						m_api->endCapture();
					}
				}

				if (producedRenderableOutput && !m_loggedFirstRenderSubmit)
				{
					logStartupEvent("first_render_submit");
					m_loggedFirstRenderSubmit = true;
				}
				if (m_hasPathtraceOutput && !m_pipelineCache.warmup.started)
				{
					kickoffPipelineWarmup();
				}
				maybeCheckpointPipelineCache();

				m_window->setCaption("[Nabla Engine] HLSL Compute Path Tracer");
				m_surface->present(m_currentImageAcquire.imageIndex, rendered);
			}
		}

		inline bool keepRunning() override
		{
			if (m_surface->irrecoverable())
				return false;

			return true;
		}

		inline bool onAppTerminated() override
		{
			waitForPendingPipelines();
			savePipelineCache();
			return device_base_t::onAppTerminated();
		}

		inline void update()
		{
			m_camera.setMoveSpeed(guiControlled.moveSpeed);
			m_camera.setRotateSpeed(guiControlled.rotateSpeed);

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

			m_camera.beginInputProcessing(nextPresentationTimestamp);
			{
				const auto& io = ImGui::GetIO();
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
					{
						if (!io.WantCaptureMouse)
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
						if (!io.WantCaptureKeyboard)
							m_camera.keyboardProcess(events); // don't capture the events, only let camera handle them with its impl

						for (const auto& e : events) // here capture
						{
							if (e.timeStamp < previousEventTimestamp)
								continue;

							if (e.keyCode == ui::EKC_H)
								if (e.action == ui::SKeyboardEvent::ECA_RELEASED)
									m_showUI = !m_showUI;

							previousEventTimestamp = e.timeStamp;
							capturedEvents.keyboard.emplace_back(e);
						}
					}, m_logger.get());
			}
			m_camera.endInputProcessing(nextPresentationTimestamp);

			const core::SRange<const nbl::ui::SMouseEvent> mouseEvents(capturedEvents.mouse.data(), capturedEvents.mouse.data() + capturedEvents.mouse.size());
			const core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents(capturedEvents.keyboard.data(), capturedEvents.keyboard.data() + capturedEvents.keyboard.size());
			const auto cursorPosition = m_window->getCursorControl()->getPosition();
			const auto mousePosition = float32_t2(cursorPosition.x, cursorPosition.y) - float32_t2(m_window->getX(), m_window->getY());

			const ext::imgui::UI::SUpdateParameters params =
			{
				.mousePosition = mousePosition,
				.displaySize = { m_window->getWidth(), m_window->getHeight() },
				.mouseEvents = mouseEvents,
				.keyboardEvents = keyboardEvents
			};

			if (m_showUI)
			    m_ui.manager->update(params);
		}
	
	private:
		template<core::StringLiteral ShaderKey>
		smart_refctd_ptr<IShader> loadPrecompiledShader()
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = "app_resources";

			const auto key = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_device.get());
			auto assetBundle = m_assetMgr->getAsset(key, lp);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
			{
				m_logger->log("Could not load precompiled shader: %s", ILogger::ELL_ERROR, key.c_str());
				return nullptr;
			}

			auto shader = IAsset::castDown<IShader>(assets[0]);
			if (!shader)
			{
				m_logger->log("Failed to cast %s asset to IShader!", ILogger::ELL_ERROR, key.c_str());
				return nullptr;
			}

			shader->setFilePathHint(std::string(std::string_view(ShaderKey.value)));
			return shader;
		}

		void logStartupEvent(const char* const eventName)
		{
			const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - m_startupBeganAt).count();
			m_logger->log("PATH_TRACER_STARTUP %s_ms=%lld", ILogger::ELL_INFO, eventName, static_cast<long long>(elapsedMs));
		}

		std::optional<path> tryGetPipelineCacheDirOverride() const
		{
			constexpr std::string_view prefix = "--pipeline-cache-dir=";
			for (size_t i = 1ull; i < argv.size(); ++i)
			{
				const std::string_view arg = argv[i];
				if (arg.rfind(prefix, 0ull) == 0ull)
				{
					const auto value = arg.substr(prefix.size());
					if (!value.empty())
						return path(std::string(value));
					return std::nullopt;
				}
				if (arg == "--pipeline-cache-dir")
				{
					if (i + 1ull < argv.size())
						return path(argv[i + 1ull]);
					return std::nullopt;
				}
			}
			return std::nullopt;
		}

		bool shouldClearPipelineCacheOnStartup() const
		{
			for (const auto& arg : argv)
			{
				if (arg == "--clear-pipeline-cache")
					return true;
			}
			return false;
		}

		static std::string hashToHex(const core::blake3_hash_t& hash)
		{
			static constexpr char digits[] = "0123456789abcdef";
			static constexpr size_t HexCharsPerByte = 2ull;
			static constexpr uint32_t HighNibbleBitOffset = 4u;
			static constexpr uint8_t NibbleMask = 0xfu;
			const auto hashByteCount = sizeof(hash.data);
			std::string retval;
			retval.resize(hashByteCount * HexCharsPerByte);
			for (size_t i = 0ull; i < hashByteCount; ++i)
			{
				const auto hexOffset = i * HexCharsPerByte;
				retval[hexOffset] = digits[(hash.data[i] >> HighNibbleBitOffset) & NibbleMask];
				retval[hexOffset + 1ull] = digits[hash.data[i] & NibbleMask];
			}
			return retval;
		}

		path getDefaultPipelineCacheDir() const
		{
			if (const auto* localAppData = std::getenv("LOCALAPPDATA"); localAppData && localAppData[0] != '\0')
				return path(localAppData) / "nabla/examples/31_HLSLPathTracer/pipeline/cache";
			return localOutputCWD / "pipeline/cache";
		}

		path getRuntimeConfigPath() const
		{
			return system::executableDirectory() / RuntimeConfigFilename;
		}

		std::optional<path> tryGetPipelineCacheDirFromRuntimeConfig() const
		{
			const auto configPath = getRuntimeConfigPath();
			if (!m_system->exists(configPath, IFile::ECF_READ))
				return std::nullopt;

			std::ifstream input(configPath);
			if (!input.is_open())
				return std::nullopt;

			nlohmann::json json;
			try
			{
				input >> json;
			}
			catch (const std::exception& e)
			{
				m_logger->log("Failed to parse PATH_TRACER runtime config %s: %s", ILogger::ELL_WARNING, configPath.string().c_str(), e.what());
				return std::nullopt;
			}

			const auto cacheRootIt = json.find("cache_root");
			if (cacheRootIt == json.end() || !cacheRootIt->is_string())
				return std::nullopt;

			const auto cacheRoot = cacheRootIt->get<std::string>();
			if (cacheRoot.empty())
				return std::nullopt;

			const path relativeRoot(cacheRoot);
			if (relativeRoot.is_absolute())
			{
				m_logger->log("Ignoring absolute cache_root in %s", ILogger::ELL_WARNING, configPath.string().c_str());
				return std::nullopt;
			}

			return (configPath.parent_path() / relativeRoot).lexically_normal();
		}

		path getPipelineCacheRootDir() const
		{
			if (const auto overrideDir = tryGetPipelineCacheDirOverride(); overrideDir.has_value())
				return overrideDir.value();
			if (const auto runtimeConfigDir = tryGetPipelineCacheDirFromRuntimeConfig(); runtimeConfigDir.has_value())
				return runtimeConfigDir.value();
			return getDefaultPipelineCacheDir();
		}

		path getPipelineCacheBlobPath() const
		{
			const auto key = m_device->getPipelineCacheKey();
			return getPipelineCacheRootDir() / "blob" / BuildConfigName / (std::string(key.deviceAndDriverUUID) + ".bin");
		}

		path getSpirvCacheDir() const
		{
			return getPipelineCacheRootDir() / "spirv" / BuildConfigName;
		}

		path getTrimmedShaderCachePath(const IShader* shader, const char* const entryPoint) const
		{
			core::blake3_hasher hasher;
			hasher << std::string_view(shader ? shader->getFilepathHint() : std::string_view{});
			hasher << std::string_view(entryPoint);
			return getSpirvCacheDir() / (hashToHex(static_cast<core::blake3_hash_t>(hasher)) + ".spv");
		}

		path getValidatedSpirvMarkerPath(const ICPUBuffer* spirvBuffer) const
		{
			auto contentHash = spirvBuffer->getContentHash();
			if (contentHash == ICPUBuffer::INVALID_HASH)
				contentHash = spirvBuffer->computeContentHash();
			return getSpirvCacheDir() / (hashToHex(contentHash) + ".hash");
		}

		size_t getBackgroundPipelineBuildBudget() const
		{
			static constexpr uint32_t ReservedForegroundThreadCount = 1u;
			const auto concurrency = std::thread::hardware_concurrency();
			if (concurrency > ReservedForegroundThreadCount)
				return static_cast<size_t>(concurrency - ReservedForegroundThreadCount);
			return ReservedForegroundThreadCount;
		}

		bool ensureCacheDirectoryExists(const path& dir, const char* const description)
		{
			if (dir.empty() || m_system->isDirectory(dir))
				return true;

			if (m_system->createDirectory(dir) || m_system->isDirectory(dir))
				return true;

			m_logger->log("Failed to create %s %s", ILogger::ELL_WARNING, description, dir.string().c_str());
			return false;
		}

		bool finalizeCacheFile(const path& tempPath, const path& finalPath, const char* const description)
		{
			m_system->deleteFile(finalPath);
			const auto ec = m_system->moveFileOrDirectory(tempPath, finalPath);
			if (!ec)
				return true;

			m_system->deleteFile(tempPath);
			m_logger->log("Failed to finalize %s %s", ILogger::ELL_WARNING, description, finalPath.string().c_str());
			return false;
		}

		void initializePipelineCache()
		{
			m_pipelineCache.blobPath = getPipelineCacheBlobPath();
			m_pipelineCache.trimmedShaders.rootDir = getSpirvCacheDir();
			m_pipelineCache.trimmedShaders.validationDir = getSpirvCacheDir();
			if (!m_pipelineCache.trimmedShaders.trimmer)
				m_pipelineCache.trimmedShaders.trimmer = core::make_smart_refctd_ptr<asset::ISPIRVEntryPointTrimmer>();
			const auto pipelineCacheRootDir = getPipelineCacheRootDir();
			std::error_code ec;
			m_pipelineCache.loadedBytes = 0ull;
			m_pipelineCache.loadedFromDisk = false;
			m_pipelineCache.clearedOnStartup = shouldClearPipelineCacheOnStartup();
			m_pipelineCache.newlyReadyPipelinesSinceLastSave = 0ull;
			m_pipelineCache.checkpointedAfterFirstSubmit = false;
			m_pipelineCache.lastSaveAt = clock_t::now();
			if (shouldClearPipelineCacheOnStartup())
			{
				if (m_system->isDirectory(pipelineCacheRootDir) && !m_system->deleteDirectory(pipelineCacheRootDir))
					m_logger->log("Failed to clear pipeline cache directory %s", ILogger::ELL_WARNING, pipelineCacheRootDir.string().c_str());
				else
					m_logger->log("PATH_TRACER_PIPELINE_CACHE clear root=%s", ILogger::ELL_INFO, pipelineCacheRootDir.string().c_str());
			}
			ensureCacheDirectoryExists(m_pipelineCache.blobPath.parent_path(), "pipeline cache directory");
			ensureCacheDirectoryExists(m_pipelineCache.trimmedShaders.rootDir, "trimmed shader cache directory");
			ensureCacheDirectoryExists(m_pipelineCache.trimmedShaders.validationDir, "validated shader cache directory");

			std::vector<uint8_t> initialData;
			{
				std::ifstream input(m_pipelineCache.blobPath, std::ios::binary | std::ios::ate);
				if (input.is_open())
				{
					const auto size = input.tellg();
					if (size > 0)
					{
						initialData.resize(static_cast<size_t>(size));
						input.seekg(0, std::ios::beg);
						input.read(reinterpret_cast<char*>(initialData.data()), static_cast<std::streamsize>(initialData.size()));
						if (!input)
							initialData.clear();
					}
				}
			}

			std::span<const uint8_t> initialDataSpan = {};
			if (!initialData.empty())
			{
				initialDataSpan = { initialData.data(), initialData.size() };
				m_pipelineCache.loadedBytes = initialData.size();
				m_pipelineCache.loadedFromDisk = true;
			}

			m_pipelineCache.object = m_device->createPipelineCache(initialDataSpan);
			if (!m_pipelineCache.object && !initialData.empty())
			{
				m_logger->log("Pipeline cache blob at %s was rejected. Falling back to empty cache.", ILogger::ELL_WARNING, m_pipelineCache.blobPath.string().c_str());
				m_pipelineCache.object = m_device->createPipelineCache(std::span<const uint8_t>{});
			}
			if (!m_pipelineCache.object)
			{
				m_logger->log("Failed to create PATH_TRACER pipeline cache.", ILogger::ELL_WARNING);
				return;
			}

			m_pipelineCache.object->setObjectDebugName("PATH_TRACER Pipeline Cache");
			m_logger->log("PATH_TRACER pipeline cache path: %s", ILogger::ELL_INFO, m_pipelineCache.blobPath.string().c_str());
			m_logger->log("PATH_TRACER trimmed shader cache path: %s", ILogger::ELL_INFO, m_pipelineCache.trimmedShaders.rootDir.string().c_str());
			m_logger->log("PATH_TRACER validated shader cache path: %s", ILogger::ELL_INFO, m_pipelineCache.trimmedShaders.validationDir.string().c_str());
			m_logger->log(
				"PATH_TRACER_PIPELINE_CACHE init clear=%u loaded_from_disk=%u loaded_bytes=%zu path=%s",
				ILogger::ELL_INFO,
				m_pipelineCache.clearedOnStartup ? 1u : 0u,
				m_pipelineCache.loadedFromDisk ? 1u : 0u,
				m_pipelineCache.loadedBytes,
				m_pipelineCache.blobPath.string().c_str()
			);
			if (!initialData.empty())
				m_logger->log("Loaded PATH_TRACER pipeline cache blob: %s", ILogger::ELL_INFO, m_pipelineCache.blobPath.string().c_str());
		}

		smart_refctd_ptr<IShader> tryLoadTrimmedShaderFromDisk(const IShader* sourceShader, const char* const entryPoint)
		{
			const auto cachePath = getTrimmedShaderCachePath(sourceShader, entryPoint);
			std::ifstream input(cachePath, std::ios::binary | std::ios::ate);
			if (!input.is_open())
				return nullptr;

			const auto size = input.tellg();
			if (size <= 0)
				return nullptr;

			std::vector<uint8_t> bytes(static_cast<size_t>(size));
			input.seekg(0, std::ios::beg);
			input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
			if (!input)
				return nullptr;

			auto buffer = ICPUBuffer::create({ { bytes.size() }, bytes.data() });
			if (!buffer)
				return nullptr;
			buffer->setContentHash(buffer->computeContentHash());
			{
				std::lock_guard lock(m_pipelineCache.trimmedShaders.mutex);
				m_pipelineCache.trimmedShaders.loadedBytes += bytes.size();
				++m_pipelineCache.trimmedShaders.loadedFromDiskCount;
			}
			m_logger->log(
				"PATH_TRACER_SHADER_CACHE load entrypoint=%s bytes=%zu path=%s",
				ILogger::ELL_INFO,
				entryPoint,
				bytes.size(),
				cachePath.string().c_str()
			);
			return core::make_smart_refctd_ptr<IShader>(std::move(buffer), IShader::E_CONTENT_TYPE::ECT_SPIRV, std::string(sourceShader->getFilepathHint()));
		}

		bool hasValidatedSpirvMarker(const ICPUBuffer* spirvBuffer) const
		{
			return m_system->exists(getValidatedSpirvMarkerPath(spirvBuffer), IFile::ECF_READ);
		}

		void saveValidatedSpirvMarker(const ICPUBuffer* spirvBuffer)
		{
			const auto markerPath = getValidatedSpirvMarkerPath(spirvBuffer);
			if (!ensureCacheDirectoryExists(markerPath.parent_path(), "validated shader cache directory"))
				return;

			auto tempPath = markerPath;
			tempPath += ".tmp";
			{
				std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
				if (!output.is_open())
				{
					m_logger->log("Failed to open validated shader marker temp file %s", ILogger::ELL_WARNING, tempPath.string().c_str());
					return;
				}
				output << "ok\n";
				output.flush();
				if (!output)
				{
					output.close();
					m_system->deleteFile(tempPath);
					m_logger->log("Failed to write validated shader marker %s", ILogger::ELL_WARNING, tempPath.string().c_str());
					return;
				}
			}

			finalizeCacheFile(tempPath, markerPath, "validated shader marker");
		}

		bool ensurePreparedShaderValidated(const smart_refctd_ptr<IShader>& preparedShader)
		{
			if (!preparedShader)
				return false;

			auto* const content = preparedShader->getContent();
			if (!content)
				return false;

			if (hasValidatedSpirvMarker(content))
				return true;

			if (!m_pipelineCache.trimmedShaders.trimmer->ensureValidated(content, m_logger.get()))
				return false;

			saveValidatedSpirvMarker(content);
			return true;
		}

		void saveTrimmedShaderToDisk(const IShader* shader, const char* const entryPoint, const path& cachePath)
		{
			const auto* content = shader->getContent();
			if (!content || !content->getPointer() || cachePath.empty())
				return;

			if (!ensureCacheDirectoryExists(cachePath.parent_path(), "trimmed shader cache directory"))
				return;

			auto tempPath = cachePath;
			tempPath += ".tmp";
			{
				std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
				if (!output.is_open())
				{
					m_logger->log("Failed to open trimmed shader cache temp file %s", ILogger::ELL_WARNING, tempPath.string().c_str());
					return;
				}
				output.write(reinterpret_cast<const char*>(content->getPointer()), static_cast<std::streamsize>(content->getSize()));
				output.flush();
				if (!output)
				{
					output.close();
					m_system->deleteFile(tempPath);
					m_logger->log("Failed to write trimmed shader cache blob to %s", ILogger::ELL_WARNING, tempPath.string().c_str());
					return;
				}
			}

			if (!finalizeCacheFile(tempPath, cachePath, "trimmed shader cache blob"))
				return;

			{
				std::lock_guard lock(m_pipelineCache.trimmedShaders.mutex);
				m_pipelineCache.trimmedShaders.savedBytes += content->getSize();
				++m_pipelineCache.trimmedShaders.savedToDiskCount;
			}
			m_logger->log(
				"PATH_TRACER_SHADER_CACHE save entrypoint=%s bytes=%zu path=%s",
				ILogger::ELL_INFO,
				entryPoint,
				content->getSize(),
				cachePath.string().c_str()
			);
		}

		smart_refctd_ptr<IShader> getPreparedShaderForEntryPoint(const smart_refctd_ptr<IShader>& shaderModule, const char* const entryPoint)
		{
			if (!shaderModule || shaderModule->getContentType() != IShader::E_CONTENT_TYPE::ECT_SPIRV)
				return shaderModule;

			const auto cachePath = getTrimmedShaderCachePath(shaderModule.get(), entryPoint);
			const auto cacheKey = cachePath.string();
			{
				std::lock_guard lock(m_pipelineCache.trimmedShaders.mutex);
				const auto found = m_pipelineCache.trimmedShaders.runtimeShaders.find(cacheKey);
				if (found != m_pipelineCache.trimmedShaders.runtimeShaders.end())
					return found->second;
			}

			const auto startedAt = clock_t::now();
			auto preparedShader = tryLoadTrimmedShaderFromDisk(shaderModule.get(), entryPoint);
			bool cameFromDisk = static_cast<bool>(preparedShader);
			bool wasTrimmed = false;
			if (!preparedShader)
			{
				const core::set entryPoints = { asset::ISPIRVEntryPointTrimmer::EntryPoint{ .name = entryPoint, .stage = hlsl::ShaderStage::ESS_COMPUTE } };
				const auto result = m_pipelineCache.trimmedShaders.trimmer->trim(shaderModule->getContent(), entryPoints, nullptr);
				if (!result)
				{
					m_logger->log("Failed to prepare trimmed PATH_TRACER shader for %s. Falling back to the original module.", ILogger::ELL_WARNING, entryPoint);
					return shaderModule;
				}
				if (result.spirv)
				{
					result.spirv->setContentHash(result.spirv->computeContentHash());
					preparedShader = core::make_smart_refctd_ptr<IShader>(core::smart_refctd_ptr(result.spirv), IShader::E_CONTENT_TYPE::ECT_SPIRV, std::string(shaderModule->getFilepathHint()));
				}
				else
					preparedShader = shaderModule;

				saveTrimmedShaderToDisk(preparedShader.get(), entryPoint, cachePath);
				{
					std::lock_guard lock(m_pipelineCache.trimmedShaders.mutex);
					++m_pipelineCache.trimmedShaders.generatedCount;
				}
				wasTrimmed = (preparedShader != shaderModule);
			}

			if (!ensurePreparedShaderValidated(preparedShader))
			{
				m_logger->log("Prepared PATH_TRACER shader for %s is not valid SPIR-V", ILogger::ELL_ERROR, entryPoint);
				return nullptr;
			}

			{
				std::lock_guard lock(m_pipelineCache.trimmedShaders.mutex);
				const auto [it, inserted] = m_pipelineCache.trimmedShaders.runtimeShaders.emplace(cacheKey, preparedShader);
				if (!inserted)
					preparedShader = it->second;
			}

			const auto wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - startedAt).count();
			m_logger->log(
				"PATH_TRACER_SHADER_CACHE ready entrypoint=%s wall_ms=%lld from_disk=%u trimmed=%u",
				ILogger::ELL_INFO,
				entryPoint,
				static_cast<long long>(wallMs),
				cameFromDisk ? 1u : 0u,
				wasTrimmed ? 1u : 0u
			);
			return preparedShader;
		}

		void savePipelineCache()
		{
			if (!m_pipelineCache.object || !m_pipelineCache.dirty || m_pipelineCache.blobPath.empty())
				return;

			const auto saveStartedAt = clock_t::now();
			auto cpuCache = m_pipelineCache.object->convertToCPUCache();
			if (!cpuCache)
				return;

			const auto& entries = cpuCache->getEntries();
			const auto found = entries.find(m_device->getPipelineCacheKey());
			if (found == entries.end() || !found->second.bin || found->second.bin->empty())
				return;

			if (!ensureCacheDirectoryExists(m_pipelineCache.blobPath.parent_path(), "pipeline cache directory"))
				return;

			auto tempPath = m_pipelineCache.blobPath;
			tempPath += ".tmp";
			{
				std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
				if (!output.is_open())
				{
					m_logger->log("Failed to open pipeline cache temp file %s", ILogger::ELL_WARNING, tempPath.string().c_str());
					return;
				}
				output.write(reinterpret_cast<const char*>(found->second.bin->data()), static_cast<std::streamsize>(found->second.bin->size()));
				output.flush();
				if (!output)
				{
					output.close();
					m_system->deleteFile(tempPath);
					m_logger->log("Failed to write pipeline cache blob to %s", ILogger::ELL_WARNING, tempPath.string().c_str());
					return;
				}
			}

			if (!finalizeCacheFile(tempPath, m_pipelineCache.blobPath, "pipeline cache blob"))
				return;

			m_pipelineCache.dirty = false;
			m_pipelineCache.savedBytes = found->second.bin->size();
			m_pipelineCache.newlyReadyPipelinesSinceLastSave = 0ull;
			m_pipelineCache.lastSaveAt = clock_t::now();
			const auto saveElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - saveStartedAt).count();
			m_logger->log(
				"PATH_TRACER_PIPELINE_CACHE save bytes=%zu wall_ms=%lld path=%s",
				ILogger::ELL_INFO,
				m_pipelineCache.savedBytes,
				static_cast<long long>(saveElapsedMs),
				m_pipelineCache.blobPath.string().c_str()
			);
			m_logger->log("Saved PATH_TRACER pipeline cache blob: %s", ILogger::ELL_INFO, m_pipelineCache.blobPath.string().c_str());
		}

		void maybeCheckpointPipelineCache()
		{
			if (!m_pipelineCache.object || !m_pipelineCache.dirty)
				return;

			if (m_loggedFirstRenderSubmit && !m_pipelineCache.checkpointedAfterFirstSubmit)
			{
				savePipelineCache();
				m_pipelineCache.checkpointedAfterFirstSubmit = true;
				return;
			}

			if (!m_pipelineCache.warmup.started || m_pipelineCache.warmup.loggedComplete)
				return;

			static constexpr size_t WarmupCheckpointThreshold = 4ull;
			if (m_pipelineCache.newlyReadyPipelinesSinceLastSave < WarmupCheckpointThreshold)
				return;

			const auto elapsedSinceLastSave = std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - m_pipelineCache.lastSaveAt).count();
			if (elapsedSinceLastSave < 1000ll)
				return;

			savePipelineCache();
		}

		smart_refctd_ptr<IShader> loadRenderShader(const E_LIGHT_GEOMETRY geometry, const bool persistentWorkGroups, const bool rwmc)
		{
			switch (geometry)
			{
			case ELG_SPHERE:
				if (rwmc)
					return loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.sphere.rwmc")>();
				return loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.sphere")>();
			case ELG_TRIANGLE:
#if defined(PATH_TRACER_BUILD_MODE_SPECIALIZED)
				if (rwmc)
					return persistentWorkGroups ?
						loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.triangle.rwmc.persistent")>() :
						loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.triangle.rwmc.linear")>();
				return persistentWorkGroups ?
					loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.triangle.persistent")>() :
					loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.triangle.linear")>();
#else
				if (rwmc)
					return loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.triangle.rwmc")>();
				return loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.triangle")>();
#endif
			case ELG_RECTANGLE:
#if defined(PATH_TRACER_BUILD_MODE_SPECIALIZED)
				if (rwmc)
					return persistentWorkGroups ?
						loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.rectangle.rwmc.persistent")>() :
						loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.rectangle.rwmc.linear")>();
#else
				if (rwmc)
					return loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.rectangle.rwmc")>();
#endif
				return loadPrecompiledShader<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("pt.compute.rectangle")>();
			default:
				return nullptr;
			}
		}

		using pipeline_future_t = std::future<smart_refctd_ptr<IGPUComputePipeline>>;
		using shader_array_t = std::array<smart_refctd_ptr<IShader>, E_LIGHT_GEOMETRY::ELG_COUNT>;
		using pipeline_method_array_t = std::array<smart_refctd_ptr<IGPUComputePipeline>, EPM_COUNT>;
		using pipeline_future_method_array_t = std::array<pipeline_future_t, EPM_COUNT>;
		using pipeline_array_t = std::array<pipeline_method_array_t, E_LIGHT_GEOMETRY::ELG_COUNT>;
		using pipeline_future_array_t = std::array<pipeline_future_method_array_t, E_LIGHT_GEOMETRY::ELG_COUNT>;
		struct SRenderPipelineStorage
		{
			std::array<std::array<shader_array_t, BinaryToggleCount>, BinaryToggleCount> shaders = {};
			std::array<std::array<pipeline_array_t, BinaryToggleCount>, BinaryToggleCount> pipelines = {};
			std::array<std::array<pipeline_future_array_t, BinaryToggleCount>, BinaryToggleCount> pendingPipelines = {};

			static constexpr size_t boolToIndex(const bool value)
			{
				return static_cast<size_t>(value);
			}

			shader_array_t& getShaders(const bool persistentWorkGroups, const bool rwmc)
			{
				return shaders[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
			}

			const shader_array_t& getShaders(const bool persistentWorkGroups, const bool rwmc) const
			{
				return shaders[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
			}

			pipeline_array_t& getPipelines(const bool persistentWorkGroups, const bool rwmc)
			{
				return pipelines[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
			}

			const pipeline_array_t& getPipelines(const bool persistentWorkGroups, const bool rwmc) const
			{
				return pipelines[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
			}

			pipeline_future_array_t& getPendingPipelines(const bool persistentWorkGroups, const bool rwmc)
			{
				return pendingPipelines[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
			}

			const pipeline_future_array_t& getPendingPipelines(const bool persistentWorkGroups, const bool rwmc) const
			{
				return pendingPipelines[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
			}
		};

		struct SResolvePipelineState
		{
			smart_refctd_ptr<IGPUPipelineLayout> layout;
			smart_refctd_ptr<IShader> shader;
			smart_refctd_ptr<IGPUComputePipeline> pipeline;
			pipeline_future_t pendingPipeline;
		};
		struct SWarmupJob
		{
			enum class E_TYPE : uint8_t
			{
				Render,
				Resolve
			};

			E_TYPE type = E_TYPE::Render;
			E_LIGHT_GEOMETRY geometry = ELG_SPHERE;
			bool persistentWorkGroups = false;
			bool rwmc = false;
			E_POLYGON_METHOD polygonMethod = EPM_PROJECTED_SOLID_ANGLE;
		};

		struct SPipelineCacheState
		{
			struct STrimmedShaderCache
			{
				smart_refctd_ptr<asset::ISPIRVEntryPointTrimmer> trimmer;
				path rootDir;
				path validationDir;
				size_t loadedFromDiskCount = 0ull;
				size_t generatedCount = 0ull;
				size_t savedToDiskCount = 0ull;
				size_t loadedBytes = 0ull;
				size_t savedBytes = 0ull;
				core::unordered_map<std::string, smart_refctd_ptr<IShader>> runtimeShaders;
				std::mutex mutex;
			} trimmedShaders;

			struct SWarmupState
			{
				bool started = false;
				bool loggedComplete = false;
				clock_t::time_point beganAt = clock_t::now();
				size_t budget = 1ull;
				size_t queuedJobs = 0ull;
				size_t launchedJobs = 0ull;
				size_t skippedJobs = 0ull;
				std::deque<SWarmupJob> queue;
			} warmup;

			smart_refctd_ptr<IGPUPipelineCache> object;
			path blobPath;
			bool dirty = false;
			bool loadedFromDisk = false;
			bool clearedOnStartup = false;
			size_t loadedBytes = 0ull;
			size_t savedBytes = 0ull;
			size_t newlyReadyPipelinesSinceLastSave = 0ull;
			bool checkpointedAfterFirstSubmit = false;
			clock_t::time_point lastSaveAt = clock_t::now();
		};

		static constexpr bool SpecializedBuildMode =
#if defined(PATH_TRACER_BUILD_MODE_SPECIALIZED)
			true;
#else
			false;
#endif

		static constexpr const char* PathTracerBuildModeName =
#if defined(PATH_TRACER_BUILD_MODE_SPECIALIZED)
			"SPECIALIZED";
#else
			"WALLTIME_OPTIMIZED";
#endif

		struct SRenderVariantInfo
		{
			E_POLYGON_METHOD effectiveMethod;
			E_POLYGON_METHOD pipelineMethod;
			const char* entryPoint;
		};

		static constexpr const char* getDefaultRenderEntryPointName(const bool persistentWorkGroups)
		{
			return persistentWorkGroups ? "mainPersistent" : "main";
		}

		static constexpr SRenderVariantInfo getRenderVariantInfo(const E_LIGHT_GEOMETRY geometry, const bool persistentWorkGroups, const E_POLYGON_METHOD requestedMethod)
		{
			const char* const defaultEntryPoint = getDefaultRenderEntryPointName(persistentWorkGroups);
			switch (geometry)
			{
			case ELG_SPHERE:
				return { EPM_SOLID_ANGLE, EPM_SOLID_ANGLE, defaultEntryPoint };
			case ELG_TRIANGLE:
				if (!SpecializedBuildMode)
					return { requestedMethod, EPM_PROJECTED_SOLID_ANGLE, defaultEntryPoint };
				switch (requestedMethod)
				{
				case EPM_AREA:
					return { EPM_AREA, EPM_AREA, persistentWorkGroups ? "mainPersistentArea" : "mainArea" };
				case EPM_SOLID_ANGLE:
					return { EPM_SOLID_ANGLE, EPM_SOLID_ANGLE, persistentWorkGroups ? "mainPersistentSolidAngle" : "mainSolidAngle" };
				case EPM_PROJECTED_SOLID_ANGLE:
				default:
					return { EPM_PROJECTED_SOLID_ANGLE, EPM_PROJECTED_SOLID_ANGLE, defaultEntryPoint };
				}
			case ELG_RECTANGLE:
				return { EPM_SOLID_ANGLE, EPM_SOLID_ANGLE, defaultEntryPoint };
			default:
				return { EPM_PROJECTED_SOLID_ANGLE, EPM_PROJECTED_SOLID_ANGLE, defaultEntryPoint };
			}
		}

		size_t getRunningPipelineBuildCount() const
		{
			size_t count = 0ull;
			const auto countPending = [&count](const pipeline_future_array_t& futures, const pipeline_array_t& pipelines) -> void
			{
				for (auto geometry = 0u; geometry < ELG_COUNT; ++geometry)
				{
					for (auto method = 0u; method < EPM_COUNT; ++method)
					{
						if (futures[geometry][method].valid() && !pipelines[geometry][method])
							++count;
					}
				}
			};
			for (const auto rwmc : { false, true })
			{
				for (const auto persistentWorkGroups : { false, true })
					countPending(m_renderPipelines.getPendingPipelines(persistentWorkGroups, rwmc), m_renderPipelines.getPipelines(persistentWorkGroups, rwmc));
			}
			if (m_resolvePipelineState.pendingPipeline.valid() && !m_resolvePipelineState.pipeline)
				++count;
			return count;
		}

		size_t getKnownRenderPipelineCount() const
		{
			size_t count = 0ull;
			bool seen[ELG_COUNT][BinaryToggleCount][BinaryToggleCount][EPM_COUNT] = {};
			for (auto geometry = 0u; geometry < ELG_COUNT; ++geometry)
			{
				for (auto persistentWorkGroups = 0u; persistentWorkGroups < BinaryToggleCount; ++persistentWorkGroups)
				{
					for (auto rwmc = 0u; rwmc < BinaryToggleCount; ++rwmc)
					{
						for (auto method = 0u; method < EPM_COUNT; ++method)
						{
							const auto pipelineMethod = static_cast<size_t>(getRenderVariantInfo(
								static_cast<E_LIGHT_GEOMETRY>(geometry),
								static_cast<bool>(persistentWorkGroups),
								static_cast<E_POLYGON_METHOD>(method)
							).pipelineMethod);
							if (seen[geometry][persistentWorkGroups][rwmc][pipelineMethod])
								continue;
							seen[geometry][persistentWorkGroups][rwmc][pipelineMethod] = true;
							++count;
						}
					}
				}
			}
			return count;
		}

		size_t getReadyRenderPipelineCount() const
		{
			size_t count = 0ull;
			const auto countReady = [&count](const pipeline_array_t& pipelines) -> void
			{
				for (const auto& perGeometry : pipelines)
				{
					for (const auto& pipeline : perGeometry)
					{
						if (pipeline)
							++count;
					}
				}
			};
			for (const auto rwmc : { false, true })
			{
				for (const auto persistentWorkGroups : { false, true })
					countReady(m_renderPipelines.getPipelines(persistentWorkGroups, rwmc));
			}
			return count;
		}

		void enqueueWarmupJob(const SWarmupJob& job)
		{
			for (const auto& existing : m_pipelineCache.warmup.queue)
			{
				if (existing.type != job.type)
					continue;
				if (existing.type == SWarmupJob::E_TYPE::Resolve)
					return;
				if (
					existing.geometry == job.geometry &&
					existing.persistentWorkGroups == job.persistentWorkGroups &&
					existing.rwmc == job.rwmc &&
					getRenderVariantInfo(existing.geometry, existing.persistentWorkGroups, existing.polygonMethod).pipelineMethod ==
					getRenderVariantInfo(job.geometry, job.persistentWorkGroups, job.polygonMethod).pipelineMethod
				)
					return;
			}
			m_pipelineCache.warmup.queue.push_back(job);
		}

		bool launchWarmupJobIfNeeded(const SWarmupJob& job)
		{
			if (job.type == SWarmupJob::E_TYPE::Resolve)
			{
				if (m_resolvePipelineState.pipeline || m_resolvePipelineState.pendingPipeline.valid())
					return false;
				ensureResolvePipeline();
				return m_resolvePipelineState.pendingPipeline.valid();
			}

			auto& pipelines = m_renderPipelines.getPipelines(job.persistentWorkGroups, job.rwmc);
			auto& pendingPipelines = m_renderPipelines.getPendingPipelines(job.persistentWorkGroups, job.rwmc);
			const auto methodIx = static_cast<size_t>(getRenderVariantInfo(job.geometry, job.persistentWorkGroups, job.polygonMethod).pipelineMethod);
			if (pipelines[job.geometry][methodIx] || pendingPipelines[job.geometry][methodIx].valid())
				return false;

			ensureRenderPipeline(job.geometry, job.persistentWorkGroups, job.rwmc, job.polygonMethod);
			return pendingPipelines[job.geometry][methodIx].valid();
		}

		void pumpPipelineWarmup()
		{
			if (!m_pipelineCache.warmup.started)
				return;

			while (!m_pipelineCache.warmup.queue.empty() && getRunningPipelineBuildCount() < m_pipelineCache.warmup.budget)
			{
				const auto job = m_pipelineCache.warmup.queue.front();
				m_pipelineCache.warmup.queue.pop_front();
				if (launchWarmupJobIfNeeded(job))
					++m_pipelineCache.warmup.launchedJobs;
				else
					++m_pipelineCache.warmup.skippedJobs;
			}

			if (!m_pipelineCache.warmup.loggedComplete && m_pipelineCache.warmup.queue.empty() && getRunningPipelineBuildCount() == 0ull)
			{
				m_pipelineCache.warmup.loggedComplete = true;
				const auto warmupElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - m_pipelineCache.warmup.beganAt).count();
				const auto readyRenderPipelines = getReadyRenderPipelineCount();
				const auto totalRenderPipelines = getKnownRenderPipelineCount();
				m_logger->log(
					"PATH_TRACER_PIPELINE_CACHE warmup_complete wall_ms=%lld queued_jobs=%zu launched_jobs=%zu skipped_jobs=%zu max_parallel=%zu ready_render=%zu total_render=%zu resolve_ready=%u",
					ILogger::ELL_INFO,
					static_cast<long long>(warmupElapsedMs),
					m_pipelineCache.warmup.queuedJobs,
					m_pipelineCache.warmup.launchedJobs,
					m_pipelineCache.warmup.skippedJobs,
					m_pipelineCache.warmup.budget,
					readyRenderPipelines,
					totalRenderPipelines,
					m_resolvePipelineState.pipeline ? 1u : 0u
				);
				logStartupEvent("pipeline_warmup_complete");
				savePipelineCache();
			}
		}

		pipeline_future_t requestComputePipelineBuild(smart_refctd_ptr<IShader> shaderModule, IGPUPipelineLayout* const pipelineLayout, const char* const entryPoint)
		{
			if (!shaderModule)
				return {};

			return std::async(
				std::launch::async,
				[
					this,
					device = m_device,
					pipelineCache = m_pipelineCache.object,
					shader = std::move(shaderModule),
					layout = smart_refctd_ptr<IGPUPipelineLayout>(pipelineLayout),
					requiredSubgroupSize = m_requiredSubgroupSize,
					logger = m_logger.get(),
					entryPointName = std::string(entryPoint),
					cacheLoadedFromDisk = m_pipelineCache.loadedFromDisk
				]() -> smart_refctd_ptr<IGPUComputePipeline>
				{
					const auto startedAt = clock_t::now();
					auto preparedShader = getPreparedShaderForEntryPoint(shader, entryPointName.c_str());
					if (!preparedShader)
						return nullptr;
					smart_refctd_ptr<IGPUComputePipeline> pipeline;
					IGPUComputePipeline::SCreationParams params = {};
					params.layout = layout.get();
					params.shader.shader = preparedShader.get();
					params.shader.entryPoint = entryPointName.c_str();
					params.shader.entries = nullptr;
					params.cached.requireFullSubgroups = true;
					params.shader.requiredSubgroupSize = requiredSubgroupSize;
					if (!device->createComputePipelines(pipelineCache.get(), { &params, 1 }, &pipeline))
					{
						if (logger)
							logger->log("Failed to create precompiled path tracing pipeline for %s", ILogger::ELL_ERROR, entryPointName.c_str());
						return nullptr;
					}
					if (logger)
					{
						const auto wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - startedAt).count();
						logger->log(
							"PATH_TRACER_PIPELINE_BUILD entrypoint=%s wall_ms=%lld cache_loaded_from_disk=%u",
							ILogger::ELL_INFO,
							entryPointName.c_str(),
							static_cast<long long>(wallMs),
							cacheLoadedFromDisk ? 1u : 0u
						);
					}
					return pipeline;
				}
			);
		}

		void pollPendingPipeline(pipeline_future_t& future, smart_refctd_ptr<IGPUComputePipeline>& pipeline)
		{
			if (!future.valid() || pipeline)
				return;
			if (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
				return;
			pipeline = future.get();
			if (pipeline)
			{
				m_pipelineCache.dirty = true;
				++m_pipelineCache.newlyReadyPipelinesSinceLastSave;
			}
		}

		void pollPendingPipelines()
		{
			for (const auto rwmc : { false, true })
			{
				for (const auto persistentWorkGroups : { false, true })
				{
					auto& pendingPipelines = m_renderPipelines.getPendingPipelines(persistentWorkGroups, rwmc);
					auto& pipelines = m_renderPipelines.getPipelines(persistentWorkGroups, rwmc);
					for (auto geometry = 0u; geometry < ELG_COUNT; ++geometry)
					{
						for (auto method = 0u; method < EPM_COUNT; ++method)
							pollPendingPipeline(pendingPipelines[geometry][method], pipelines[geometry][method]);
					}
				}
			}
			pollPendingPipeline(m_resolvePipelineState.pendingPipeline, m_resolvePipelineState.pipeline);
		}

		void waitForPendingPipelines()
		{
			auto waitAndStore = [](pipeline_future_t& future, smart_refctd_ptr<IGPUComputePipeline>& pipeline) -> void
			{
				if (!future.valid() || pipeline)
					return;
				future.wait();
				pipeline = future.get();
			};

			for (const auto rwmc : { false, true })
			{
				for (const auto persistentWorkGroups : { false, true })
				{
					auto& pendingPipelines = m_renderPipelines.getPendingPipelines(persistentWorkGroups, rwmc);
					auto& pipelines = m_renderPipelines.getPipelines(persistentWorkGroups, rwmc);
					for (auto geometry = 0u; geometry < ELG_COUNT; ++geometry)
					{
						for (auto method = 0u; method < EPM_COUNT; ++method)
						{
							const auto hadPipeline = static_cast<bool>(pipelines[geometry][method]);
							waitAndStore(pendingPipelines[geometry][method], pipelines[geometry][method]);
							const auto pipelineBecameReady = !hadPipeline && static_cast<bool>(pipelines[geometry][method]);
							m_pipelineCache.dirty = m_pipelineCache.dirty || pipelineBecameReady;
							m_pipelineCache.newlyReadyPipelinesSinceLastSave += pipelineBecameReady ? 1ull : 0ull;
						}
					}
				}
			}
			const auto hadResolvePipeline = static_cast<bool>(m_resolvePipelineState.pipeline);
			waitAndStore(m_resolvePipelineState.pendingPipeline, m_resolvePipelineState.pipeline);
			m_pipelineCache.dirty = m_pipelineCache.dirty || (!hadResolvePipeline && static_cast<bool>(m_resolvePipelineState.pipeline));
			if (!hadResolvePipeline && static_cast<bool>(m_resolvePipelineState.pipeline))
				++m_pipelineCache.newlyReadyPipelinesSinceLastSave;
		}

		IGPUComputePipeline* ensureRenderPipeline(const E_LIGHT_GEOMETRY geometry, const bool persistentWorkGroups, const bool rwmc, const E_POLYGON_METHOD polygonMethod)
		{
			auto& pipelines = m_renderPipelines.getPipelines(persistentWorkGroups, rwmc);
			auto& pendingPipelines = m_renderPipelines.getPendingPipelines(persistentWorkGroups, rwmc);
			const auto variantInfo = getRenderVariantInfo(geometry, persistentWorkGroups, polygonMethod);
			const auto methodIx = static_cast<size_t>(variantInfo.pipelineMethod);
			auto& pipeline = pipelines[geometry][methodIx];
			auto& future = pendingPipelines[geometry][methodIx];

			pollPendingPipeline(future, pipeline);
			if (pipeline)
				return pipeline.get();

			if (!future.valid())
			{
				const auto& shaders = m_renderPipelines.getShaders(persistentWorkGroups, rwmc);
				auto* const layout = rwmc ? m_rwmcRenderPipelineLayout.get() : m_renderPipelineLayout.get();
				future = requestComputePipelineBuild(shaders[geometry], layout, variantInfo.entryPoint);
			}

			return nullptr;
		}

		IGPUComputePipeline* ensureResolvePipeline()
		{
			pollPendingPipeline(m_resolvePipelineState.pendingPipeline, m_resolvePipelineState.pipeline);
			if (m_resolvePipelineState.pipeline)
				return m_resolvePipelineState.pipeline.get();

			if (!m_resolvePipelineState.pendingPipeline.valid())
				m_resolvePipelineState.pendingPipeline = requestComputePipelineBuild(m_resolvePipelineState.shader, m_resolvePipelineState.layout.get(), "resolve");

			return nullptr;
		}

		void kickoffPipelineWarmup()
		{
			m_pipelineCache.warmup.started = true;
			m_pipelineCache.warmup.queue.clear();
			m_pipelineCache.warmup.loggedComplete = false;
			m_pipelineCache.warmup.beganAt = clock_t::now();
			m_pipelineCache.warmup.budget = getBackgroundPipelineBuildBudget();
			m_pipelineCache.warmup.queuedJobs = 0ull;
			m_pipelineCache.warmup.launchedJobs = 0ull;
			m_pipelineCache.warmup.skippedJobs = 0ull;
			const auto currentGeometry = static_cast<E_LIGHT_GEOMETRY>(guiControlled.PTPipeline);
			const auto currentMethod = static_cast<E_POLYGON_METHOD>(guiControlled.polygonMethod);
			const auto enqueueRenderVariants = [this, currentGeometry](const E_LIGHT_GEOMETRY geometry, const E_POLYGON_METHOD preferredMethod) -> void
			{
				const auto enqueueForMethods = [this, geometry](const std::initializer_list<E_POLYGON_METHOD> methods, const bool preferPersistent, const bool preferRWMC) -> void
				{
					const bool persistentOrder[2] = { preferPersistent, !preferPersistent };
					const bool rwmcOrder[2] = { preferRWMC, !preferRWMC };
					for (const auto method : methods)
					{
						for (const auto persistentWorkGroups : persistentOrder)
						{
							for (const auto rwmc : rwmcOrder)
							{
								enqueueWarmupJob({
									.type = SWarmupJob::E_TYPE::Render,
									.geometry = geometry,
									.persistentWorkGroups = persistentWorkGroups,
									.rwmc = rwmc,
									.polygonMethod = method
								});
							}
						}
					}
				};

				const bool preferPersistent = geometry == currentGeometry ? guiControlled.usePersistentWorkGroups : false;
				const bool preferRWMC = geometry == currentGeometry ? guiControlled.useRWMC : false;
				switch (geometry)
				{
				case ELG_SPHERE:
					enqueueForMethods({ EPM_SOLID_ANGLE }, preferPersistent, preferRWMC);
					break;
				case ELG_TRIANGLE:
				{
					switch (preferredMethod)
					{
					case EPM_AREA:
						enqueueForMethods({ EPM_AREA, EPM_SOLID_ANGLE, EPM_PROJECTED_SOLID_ANGLE }, preferPersistent, preferRWMC);
						break;
					case EPM_SOLID_ANGLE:
						enqueueForMethods({ EPM_SOLID_ANGLE, EPM_AREA, EPM_PROJECTED_SOLID_ANGLE }, preferPersistent, preferRWMC);
						break;
					case EPM_PROJECTED_SOLID_ANGLE:
					default:
						enqueueForMethods({ EPM_PROJECTED_SOLID_ANGLE, EPM_AREA, EPM_SOLID_ANGLE }, preferPersistent, preferRWMC);
						break;
					}
					break;
				}
				case ELG_RECTANGLE:
					enqueueForMethods({ EPM_SOLID_ANGLE }, preferPersistent, preferRWMC);
					break;
				default:
					break;
				}
			};

			enqueueRenderVariants(currentGeometry, currentMethod);
			for (auto geometry = 0u; geometry < ELG_COUNT; ++geometry)
			{
				const auto geometryEnum = static_cast<E_LIGHT_GEOMETRY>(geometry);
				if (geometryEnum == currentGeometry)
					continue;
				enqueueRenderVariants(geometryEnum, currentMethod);
			}
			enqueueWarmupJob({ .type = SWarmupJob::E_TYPE::Resolve });
			m_pipelineCache.warmup.queuedJobs = m_pipelineCache.warmup.queue.size();
			const auto logicalConcurrency = std::thread::hardware_concurrency();
			m_logger->log(
				"PATH_TRACER_PIPELINE_CACHE warmup_start queued_jobs=%zu max_parallel=%zu logical_threads=%u current_geometry=%u current_method=%u",
				ILogger::ELL_INFO,
				m_pipelineCache.warmup.queuedJobs,
				m_pipelineCache.warmup.budget,
				logicalConcurrency,
				static_cast<uint32_t>(currentGeometry),
				static_cast<uint32_t>(currentMethod)
			);
			pumpPipelineWarmup();
		}

		IGPUComputePipeline* pickPTPipeline()
		{
			return ensureRenderPipeline(
				static_cast<E_LIGHT_GEOMETRY>(guiControlled.PTPipeline),
				guiControlled.usePersistentWorkGroups,
				guiControlled.useRWMC,
				static_cast<E_POLYGON_METHOD>(guiControlled.polygonMethod)
			);
		}

	private:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;

		// gpu resources
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		SRenderPipelineStorage m_renderPipelines;
		smart_refctd_ptr<IGPUPipelineLayout> m_renderPipelineLayout;
		smart_refctd_ptr<IGPUPipelineLayout> m_rwmcRenderPipelineLayout;
		SResolvePipelineState m_resolvePipelineState;
		smart_refctd_ptr<IGPUGraphicsPipeline> m_presentPipeline;
		IPipelineBase::SUBGROUP_SIZE m_requiredSubgroupSize = IPipelineBase::SUBGROUP_SIZE::UNKNOWN;
		uint64_t m_realFrameIx = 0;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
		smart_refctd_ptr<IGPUDescriptorSet> m_descriptorSet, m_presentDescriptorSet;

		core::smart_refctd_ptr<IDescriptorPool> m_guiDescriptorSetPool;

		// system resources
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
        InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		// pathtracer resources
		smart_refctd_ptr<IGPUImageView> m_envMapView, m_scrambleView;
		smart_refctd_ptr<IGPUBuffer> m_sequenceBuffer;
		smart_refctd_ptr<IGPUImageView> m_outImgView;
		smart_refctd_ptr<IGPUImageView> m_cascadeView;

		// sync
		smart_refctd_ptr<ISemaphore> m_semaphore;

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
		bool m_showUI;

		video::CDumbPresentationOracle m_oracle;

		uint16_t gcIndex = {};

		struct GUIControllables
		{
			float fov = 60.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
			float viewWidth = 10.f;
			float camYAngle = 165.f / 180.f * 3.14159f;
			float camXAngle = 32.f / 180.f * 3.14159f;
			int PTPipeline = E_LIGHT_GEOMETRY::ELG_SPHERE;
			int polygonMethod = EPM_PROJECTED_SOLID_ANGLE;
			int spp = 32;
			int depth = 3;
			rwmc::SResolveParameters::SCreateParams rwmcParams;
			bool usePersistentWorkGroups = false;
			bool useRWMC = false;
		};
		GUIControllables guiControlled;

		hlsl::float32_t4x4 m_lightModelMatrix = {
			0.3f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.3f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.3f, 0.0f,
			-1.0f, 1.5f, 0.0f, 1.0f,
		};
		TransformRequestParams m_transformParams;

		clock_t::time_point m_startupBeganAt = clock_t::now();
		bool m_hasPathtraceOutput = false;
		bool m_loggedFirstFrameLoop = false;
		bool m_loggedFirstRenderDispatch = false;
		bool m_loggedFirstRenderSubmit = false;
		SPipelineCacheState m_pipelineCache;
		IGPUCommandBuffer::SClearColorValue clearColor = { .float32 = {0.f,0.f,0.f,1.f} };
};

NBL_MAIN_FUNC(HLSLComputePathtracer)
