// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"

#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include "nbl/builtin/hlsl/luma_meter/common.hlsl"
#include "nbl/builtin/hlsl/workgroup2/arithmetic_config.hlsl"
#include "app_resources/common.hlsl"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;
using namespace nbl::examples;

class AutoexposureApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
	enum class MeteringMode {
		AVERAGE,
		MEDIAN
	};

	using device_base_t = SimpleWindowedApplication;
	using asset_base_t = BuiltinResourcesApplication;
	using clock_t = std::chrono::steady_clock;

	constexpr static inline uint32_t MaxFramesInFlight = 3u;

	static inline std::string DefaultImagePathsFile = "../../media/noises/spp_benchmark_4k_512.exr";
	static inline std::array<std::string, 5> ShaderPaths = {
		"app_resources/avg_luma_meter.comp.hlsl",
		"app_resources/avg_luma_tonemap.comp.hlsl",
		"app_resources/median_luma_meter.comp.hlsl",
		"app_resources/median_luma_tonemap.comp.hlsl",
		"app_resources/present.frag.hlsl"
	};
	constexpr static inline MeteringMode MeterMode = MeteringMode::AVERAGE;
	constexpr static inline uint32_t BinCount = 1024;
	constexpr static inline uint32_t2 Dimensions = { 1280, 720 };
	constexpr static inline float32_t2 MeteringMinUV = { 0.1f, 0.1f };
	constexpr static inline float32_t2 MeteringMaxUV = { 0.9f, 0.9f };
	constexpr static inline float32_t SamplingFactor = 2.f;
	constexpr static inline float32_t2 LumaRange = { 1.0f / 2048.0f, 65536.f };
	constexpr static inline float32_t2 PercentileRange = { 0.45f, 0.55f };
	constexpr static inline float32_t2 BaseExposureAdaptationFactorsLog2 = {-1.1f, -0.2f};

public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	inline AutoexposureApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	// Will get called mid-initialization, via `filterDevices` between when the API Connection is created and Physical Device is chosen
	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		// So let's create our Window and Surface then!
		if (!m_surface)
		{
			{
				IWindow::SCreationParams params = {};
				params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
				params.width = Dimensions[0];
				params.height = Dimensions[1];
				params.x = 32;
				params.y = 32;
				// Don't want to have a window lingering about before we're ready so create it hidden.
				// Only programmatic resize, not regular.
				params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
				params.windowCaption = "AutoexposureApp";
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
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		m_semaphore = m_device->createSemaphore(m_realFrameIx);

		// Create command pool and buffers
		{
			auto gQueue = getGraphicsQueue();
			m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!m_cmdPool)
				return logFail("Couldn't create Command Pool!");

			for (auto i = 0u; i < MaxFramesInFlight; i++)
			{
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");
				if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
					return logFail("Couldn't create Command Buffer!");
			}
		}

		// Create renderpass and init surface
		nbl::video::IGPURenderpass* renderpass;
		{
			ISwapchain::SCreationParams swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");

			// We actually need external dependencies to ensure ordering of the Implicit Layout Transitions relative to the semaphore signals
			constexpr IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
				// wipe-transition to ATTACHMENT_OPTIMAL
				{
					.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.dstSubpass = 0,
					.memoryBarrier = {
					// since we're uploading the image data we're about to draw
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
					.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
					.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					// because we clear and don't blend
					.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
				// leave view offsets and flags default
				},
				// ATTACHMENT_OPTIMAL to PRESENT_SRC
				{
					.srcSubpass = 0,
					.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.memoryBarrier = {
						.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
						// we can have NONE as the Destinations because the spec says so about presents
						}
				// leave view offsets and flags default
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

		// One asset converter to make the cache persist
		auto converter = CAssetConverter::create({ .device = m_device.get() });

		// Create descriptors and pipelines
		{
			// need to hoist
			CAssetConverter::SConvertParams params = {};
			params.utilities = m_utils.get();

			auto convertDSLayoutCPU2GPU = [&](std::span<ICPUDescriptorSetLayout *> cpuLayouts)
			{
				CAssetConverter::SInputs inputs = {};
				inputs.readCache = converter.get();
				inputs.logger = m_logger.get();

				std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSetLayout>>(inputs.assets) = cpuLayouts;
				// don't need to assert that we don't need to provide patches since layouts are not patchable
				//assert(true);
				auto reservation = converter->reserve(inputs);
				// even though it does nothing when none assets refer in any way (direct or indirect) to memory or need any device operations performed, still need to call to write the cache
				reservation.convert(params);
				// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
				auto gpuLayouts = reservation.getGPUObjects<ICPUDescriptorSetLayout>();
				std::vector<smart_refctd_ptr<IGPUDescriptorSetLayout>> result;
				result.reserve(cpuLayouts.size());

				for (auto& gpuLayout : gpuLayouts) {
					auto layout = gpuLayout.value;
					if (!layout) {
						m_logger->log("Failed to convert %s into an IGPUDescriptorSetLayout handle", ILogger::ELL_ERROR);
						std::exit(-1);
					}
					result.push_back(layout);
				}

				return result;
			};
			auto convertDSCPU2GPU = [&](std::span<ICPUDescriptorSet *> cpuDS)
			{
				CAssetConverter::SInputs inputs = {};
				inputs.readCache = converter.get();
				inputs.logger = m_logger.get();

				std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSet>>(inputs.assets) = cpuDS;
				// don't need to assert that we don't need to provide patches since layouts are not patchable
				//assert(true);
				auto reservation = converter->reserve(inputs);
				// even though it does nothing when none assets refer in any way (direct or indirect) to memory or need any device operations performed, still need to call to write the cache
				reservation.convert(params);
				// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
				auto gpuDS = reservation.getGPUObjects<ICPUDescriptorSet>();
				std::vector<smart_refctd_ptr<IGPUDescriptorSet>> result;
				result.reserve(cpuDS.size());

				for (auto& ds : gpuDS) {
					if (!ds.value) {
						m_logger->log("Failed to convert %s into an IGPUDescriptorSet handle", ILogger::ELL_ERROR);
						std::exit(-1);
					}
					result.push_back(ds.value);
				}

				return result;
			};

			ISampler::SParams samplerParams;
			samplerParams.AnisotropicFilter = 0;
			auto defaultSampler = make_smart_refctd_ptr<ICPUSampler>(samplerParams);

			std::array<ICPUDescriptorSetLayout::SBinding, 1> gpuImgbindings = {};
			std::array<ICPUDescriptorSetLayout::SBinding, 1> tonemappedImgRWbindings = {};
			std::array<ICPUDescriptorSetLayout::SBinding, 1> tonemappedImgSamplerbindings = {};

			gpuImgbindings[0] = {
				.binding = 0u,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
				.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
				.count = 1u,
				.immutableSamplers = &defaultSampler
			};
			tonemappedImgRWbindings[0] = {
				.binding = 0u,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
				.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
				.count = 1u,
				.immutableSamplers = nullptr
			};
			tonemappedImgSamplerbindings[0] = {
				.binding = 0u,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
				.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
				.count = 1u,
				.immutableSamplers = &defaultSampler
			};

			auto cpuGpuImgLayout = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(gpuImgbindings);
			auto cpuTonemappedImgRWLayout = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(tonemappedImgRWbindings);
			auto cpuTonemappedImgSamplerLayout = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(tonemappedImgSamplerbindings);

			std::array<ICPUDescriptorSetLayout*, 3> cpuLayouts = {
				cpuGpuImgLayout.get(),
				cpuTonemappedImgRWLayout.get(),
				cpuTonemappedImgSamplerLayout.get()
			};

			auto gpuLayouts = convertDSLayoutCPU2GPU(cpuLayouts);

			auto cpuGpuImgDS = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(cpuGpuImgLayout));
			auto cpuTonemappedImgRWDS = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(cpuTonemappedImgRWLayout));
			auto cpuTonemappedImgSamplerDS = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(cpuTonemappedImgSamplerLayout));

			std::array<ICPUDescriptorSet*, 3> cpuDS = {
				cpuGpuImgDS.get(),
				cpuTonemappedImgRWDS.get(),
				cpuTonemappedImgSamplerDS.get()
			};

			auto gpuDS = convertDSCPU2GPU(cpuDS);
			m_gpuImgDS = gpuDS[0];
			m_gpuImgDS->setObjectDebugName("m_gpuImgDS");
			m_tonemappedImgRWDS = gpuDS[1];
			m_tonemappedImgRWDS->setObjectDebugName("m_tonemappedImgRWDS");
			m_tonemappedImgSamplerDS = gpuDS[2];
			m_tonemappedImgSamplerDS->setObjectDebugName("m_tonemappedImgSamplerDS");

			// Create Shaders
			auto loadAndCompileShader = [&](std::string pathToShader) {
				IAssetLoader::SAssetLoadParams lp = {};
				auto assetBundle = m_assetMgr->getAsset(pathToShader, lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
				{
					m_logger->log("Could not load shader: ", ILogger::ELL_ERROR, pathToShader);
					std::exit(-1);
				}

				auto source = IAsset::castDown<IShader>(assets[0]);

				auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(m_system));
				CHLSLCompiler::SOptions options = {};
				options.stage = IShader::E_SHADER_STAGE::ESS_COMPUTE;
				options.preprocessorOptions.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
				options.spirvOptimizer = nullptr;
#ifndef _NBL_DEBUG
				ISPIRVOptimizer::E_OPTIMIZER_PASS optPasses = ISPIRVOptimizer::EOP_STRIP_DEBUG_INFO;
				auto opt = make_smart_refctd_ptr<ISPIRVOptimizer>(std::span<ISPIRVOptimizer::E_OPTIMIZER_PASS>(&optPasses, 1));
				options.spirvOptimizer = opt.get();
#else
				options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
#endif
				options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
				options.preprocessorOptions.logger = m_logger.get();

				auto* includeFinder = compiler->getDefaultIncludeFinder();
				options.preprocessorOptions.includeFinder = includeFinder;

				const uint32_t workgroupSize = m_physicalDevice->getLimits().maxComputeWorkGroupInvocations;
				m_subgroupSize = m_physicalDevice->getLimits().maxSubgroupSize;

				const uint32_t configItemsPerInvoc = MeterMode == MeteringMode::AVERAGE ? 1 : BinCount / workgroupSize;
				workgroup2::SArithmeticConfiguration wgConfig;
				wgConfig.init(hlsl::findMSB(workgroupSize), hlsl::log2(float(m_subgroupSize)), configItemsPerInvoc);

				struct MacroDefines
				{
					std::string identifier;
					std::string definition;
				};
				constexpr uint32_t NumBaseDefines = 3;
				constexpr uint32_t NumExtraDefines = 2;
				const MacroDefines definesBuf[NumBaseDefines+NumExtraDefines] = {
					{ "WORKGROUP_SIZE", std::to_string(workgroupSize) },
					{ "SUBGROUP_SIZE", std::to_string(m_subgroupSize) },
					{"WG_CONFIG_T", wgConfig.getConfigTemplateStructString()},
                    {"NATIVE_SUBGROUP_ARITHMETIC", "1"},
					{ "BIN_COUNT", std::to_string(BinCount) }
				};

				uint32_t defineCount = NumBaseDefines;
				if (m_physicalDevice->getLimits().shaderSubgroupArithmetic)
					defineCount++;
				if (MeterMode == MeteringMode::MEDIAN)
					defineCount++;
				std::vector<IShaderCompiler::SMacroDefinition> defines;
				for (uint32_t i = 0; i < defineCount; i++)
					defines.emplace_back(definesBuf[i].identifier, definesBuf[i].definition);
			    options.preprocessorOptions.extraDefines = defines;

				auto overriddenSource = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);
				if (!overriddenSource)
				{
					m_logger->log("Shader creationed failed: %s!", ILogger::ELL_ERROR, pathToShader.c_str());
					std::exit(-1);
				}

				return overriddenSource;
			};

			// Create compute pipelines
			{
				IGPUComputePipeline::SCreationParams params;
				auto shader = loadAndCompileShader((MeterMode == MeteringMode::AVERAGE) ? ShaderPaths[0] : ShaderPaths[2]);
				const nbl::asset::SPushConstantRange pcRange = {
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.offset = 0,
						.size = sizeof(AutoexposurePushData)
				};
				auto pipelineLayout = m_device->createPipelineLayout(
					{ &pcRange, 1 },
					smart_refctd_ptr(gpuLayouts[0]),
					nullptr,
					nullptr,
					nullptr
				);
				if (!pipelineLayout) {
					return logFail("Failed to create pipeline layout");
				}

				params.layout = pipelineLayout.get();
				params.shader.shader = shader.get();
				params.shader.entryPoint = "main";
				params.cached.requireFullSubgroups = true;
				params.shader.requiredSubgroupSize = static_cast<IPipelineBase::SUBGROUP_SIZE>(hlsl::findMSB(m_subgroupSize));

				if (!m_device->createComputePipelines(nullptr, { &params, 1 }, &m_meterPipeline)) {
					return logFail("Failed to create meter compute pipeline!\n");
				}
			}
			{
				IGPUComputePipeline::SCreationParams params;
				auto shader = loadAndCompileShader((MeterMode == MeteringMode::AVERAGE) ? ShaderPaths[1] : ShaderPaths[3]);
				const nbl::asset::SPushConstantRange pcRange = {
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.offset = 0,
						.size = sizeof(AutoexposurePushData)
				};
				auto pipelineLayout = m_device->createPipelineLayout(
					{ &pcRange, 1 },
					smart_refctd_ptr(gpuLayouts[0]),
					nullptr,
					nullptr,
					smart_refctd_ptr(gpuLayouts[1])
				);
				if (!pipelineLayout) {
					return logFail("Failed to create pipeline layout");
				}

				params.layout = pipelineLayout.get();
				params.shader.shader = shader.get();
				params.shader.entryPoint = "main";
				params.cached.requireFullSubgroups = true;
				params.shader.requiredSubgroupSize = static_cast<IPipelineBase::SUBGROUP_SIZE>(hlsl::findMSB(m_subgroupSize));

				if (!m_device->createComputePipelines(nullptr, { &params, 1 }, &m_tonemapPipeline)) {
					return logFail("Failed to create tonemap compute pipeline!\n");
				}
			}

			// Create graphics pipeline
			{
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
				if (!fsTriProtoPPln)
					return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

				// Load Fragment Shader
				auto fragmentShader = loadAndCompileShader(ShaderPaths[4]);
				if (!fragmentShader)
					return logFail("Failed to Load and Compile Fragment Shader: lumaMeterShader!");

				const IGPUPipelineBase::SShaderSpecInfo fragSpec = {
					.shader = fragmentShader.get(),
				    .entryPoint = "main"
				};

				auto presentLayout = m_device->createPipelineLayout(
					{},
					nullptr,
					nullptr,
					nullptr,
					smart_refctd_ptr(gpuLayouts[2])
				);
				m_presentPipeline = fsTriProtoPPln.createPipeline(fragSpec, presentLayout.get(), scRes->getRenderpass());
				if (!m_presentPipeline)
					return logFail("Could not create Graphics Pipeline!");
			}
		}

		// Load exr file into gpu
		smart_refctd_ptr<IGPUImage> gpuImg;
		{
			auto convertImgCPU2GPU = [&](ICPUImage* cpuImg)
			{
				auto queue = getGraphicsQueue();
				auto cmdbuf = m_cmdBufs[0].get();
				cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
				std::array<IQueue::SSubmitInfo::SCommandBufferInfo, 1> commandBufferInfo = { cmdbuf };
				core::smart_refctd_ptr<ISemaphore> imgFillSemaphore = m_device->createSemaphore(0);
				imgFillSemaphore->setObjectDebugName("Image Fill Semaphore");

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

				std::get<CAssetConverter::SInputs::asset_span_t<ICPUImage>>(inputs.assets) = { &cpuImg, 1 };
				// assert that we don't need to provide patches
				assert(cpuImg->getImageUsageFlags().hasFlags(ICPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT));
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

				return gpuImgs[0].value;
			};

			smart_refctd_ptr<ICPUImage> cpuImg;
			{
				IAssetLoader::SAssetLoadParams lp;
				SAssetBundle bundle = m_assetMgr->getAsset(DefaultImagePathsFile, lp);
				if (bundle.getContents().empty()) {
					m_logger->log("Couldn't load an asset.", ILogger::ELL_ERROR);
					std::exit(-1);
				}

				cpuImg = IAsset::castDown<ICPUImage>(bundle.getContents()[0]);
				if (!cpuImg) {
					m_logger->log("Couldn't load an asset.", ILogger::ELL_ERROR);
					std::exit(-1);
				}
			};

			gpuImg = convertImgCPU2GPU(cpuImg.get());
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
			auto createHDRIImageView = [this](smart_refctd_ptr<IGPUImage> img) -> smart_refctd_ptr<IGPUImageView> {
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

			auto params = gpuImg->getCreationParameters();
			auto extent = params.extent;
			gpuImg->setObjectDebugName("GPU Img");
			m_gpuImgView = createHDRIImageView(gpuImg);
			m_gpuImgView->setObjectDebugName("GPU Img View");
			auto outImg = createHDRIImage(asset::E_FORMAT::EF_R32G32B32A32_SFLOAT, Dimensions.x, Dimensions.y);
			outImg->setObjectDebugName("Tonemapped Image");
			m_tonemappedImgView = createHDRIImageView(outImg);
			m_tonemappedImgView->setObjectDebugName("Tonemapped Image View");
		}

		// Allocate and create buffer for Luma Gather
		{
			// Allocate memory
			m_gatherAllocation = {};
			m_histoAllocation = {};
			for (uint32_t i = 0; i < MaxFramesInFlight; i++)
			    m_lastLumaAllocations[i] = {};
			{
				auto build_buffer = [this](
					smart_refctd_ptr<ILogicalDevice> m_device,
					nbl::video::IDeviceMemoryAllocator::SAllocation *allocation,
					smart_refctd_ptr<IGPUBuffer> &buffer,
					size_t buffer_size,
					const char *label) {
					IGPUBuffer::SCreationParams params;
					params.size = buffer_size;
					params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
					buffer = m_device->createBuffer(std::move(params));
					if (!buffer)
						return logFail("Failed to create GPU buffer of size %d!\n", buffer_size);

					buffer->setObjectDebugName(label);

					auto reqs = buffer->getMemoryReqs();
					reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

					*allocation = m_device->allocate(reqs, buffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
					if (!allocation->isValid())
						return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

					assert(allocation->memory.get() == buffer->getBoundMemory().memory);
					return true;
				};

				build_buffer(
					m_device,
					&m_gatherAllocation,
					m_gatherBuffer,
					m_physicalDevice->getLimits().maxSubgroupSize * sizeof(float32_t),
 					"Luma Gather Buffer"
				);
				build_buffer(
					m_device,
					&m_histoAllocation,
					m_histoBuffer,
					BinCount * sizeof(uint32_t),
					"Luma Histogram Buffer"
				);
				for (uint32_t i = 0; i < MaxFramesInFlight; i++)
				    build_buffer(
					    m_device,
					    &m_lastLumaAllocations[i],
					    m_lastFrameEVBuffers[i],
					    sizeof(float32_t),
					    ("Last Luma EV Buffer " + std::to_string(i)).c_str()
				    );
			}

			m_gatherMemory = m_gatherAllocation.memory->map({ 0ull, m_gatherAllocation.memory->getAllocationSize() });
			m_histoMemory = m_histoAllocation.memory->map({ 0ull, m_histoAllocation.memory->getAllocationSize() });

			if (!m_gatherMemory || !m_histoMemory)
				return logFail("Failed to map the Device Memory!\n");

			for (uint32_t i = 0; i < MaxFramesInFlight; i++)
			{
				void* lastLumaMemory = m_lastLumaAllocations[i].memory->map({ 0ull, m_lastLumaAllocations[i].memory->getAllocationSize() });
				if (!lastLumaMemory)
					return logFail("Failed to map the Device Memory!\n");
				memset(lastLumaMemory, 0, m_lastFrameEVBuffers[i]->getSize());
			}
		}

		// transition m_tonemappedImgView to GENERAL
		{
			auto transitionSemaphore = m_device->createSemaphore(0);
			auto queue = getGraphicsQueue();
			auto cmdbuf = m_cmdBufs[0].get();
			cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);

			m_api->startCapture();

			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			// TRANSITION m_outImgView to GENERAL (because of descriptorSets0 -> ComputeShader Writes into the image)
			{
				const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers[] = {
					{
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
								.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
							}
						},
						.image = m_tonemappedImgView->getCreationParameters().image.get(),
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
			cmdbuf->end();

			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
			{
				{
					.semaphore = transitionSemaphore.get(),
					.value = 1,
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
				}
			};
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
			{
				{.cmdbuf = cmdbuf }
			};
			const IQueue::SSubmitInfo infos[] =
			{
				{
					.waitSemaphores = {},
					.commandBuffers = commandBuffers,
					.signalSemaphores = rendered
				}
			};
			queue->submit(infos);
			const ISemaphore::SWaitInfo waits[] = {
				{
					.semaphore = transitionSemaphore.get(),
					.value = 1
				}
			};
			m_device->blockForSemaphores(waits);
			m_api->endCapture();
		}

		// Update Descriptors
		{
			IGPUDescriptorSet::SDescriptorInfo infos[3];
			infos[0].info.combinedImageSampler.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			infos[0].desc = m_gpuImgView;
			infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
			infos[1].desc = m_tonemappedImgView;
			infos[2].info.combinedImageSampler.imageLayout = IImage::LAYOUT::GENERAL;
			infos[2].desc = m_tonemappedImgView;

			IGPUDescriptorSet::SWriteDescriptorSet writeDescriptors[] = {
				{
					.dstSet = m_gpuImgDS.get(),
					.binding = 0,
					.arrayElement = 0,
					.count = 1,
					.info = infos
				},
				{
					.dstSet = m_tonemappedImgRWDS.get(),
					.binding = 0,
					.arrayElement = 0,
					.count = 1,
					.info = infos + 1
				},
				{
					.dstSet = m_tonemappedImgSamplerDS.get(),
					.binding = 0,
					.arrayElement = 0,
					.count = 1,
					.info = infos + 2
				}
			};

			m_device->updateDescriptorSets(3, writeDescriptors, 0, nullptr);
		}

		m_winMgr->setWindowSize(m_window.get(), Dimensions.x, Dimensions.y);
		m_surface->recreateSwapchain();
		m_winMgr->show(m_window.get());
		oracle.reportBeginFrameRecord();

		m_lastPresentStamp = std::chrono::high_resolution_clock::now();

		return true;
	}

	// We do a very simple thing, display an image and wait `DisplayImageMs` to show it
	inline void workLoopBody() override
	{
		const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

		const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());

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

		auto updatePresentationTimestamp = [&]()
			{
				m_currentImageAcquire = m_surface->acquireNextImage();

				oracle.reportEndFrameRecord();
				const auto timestamp = oracle.getNextPresentationTimeStamp();
				oracle.reportBeginFrameRecord();

				return timestamp;
			};

		const auto nextPresentationTimestamp = updatePresentationTimestamp();

		if (!m_currentImageAcquire)
			return;

		memset(m_gatherMemory, 0, m_gatherBuffer->getSize());
		memset(m_histoMemory, 0, m_histoBuffer->getSize());

		auto gpuImgExtent = m_gpuImgView->getCreationParameters().image->getCreationParameters().extent;

		auto thisPresentStamp = std::chrono::high_resolution_clock::now();
		auto microsecondsElapsedBetweenPresents = std::chrono::duration_cast<std::chrono::microseconds>(thisPresentStamp - m_lastPresentStamp);
		m_lastPresentStamp = thisPresentStamp;

		auto pc = AutoexposurePushData
		{
			.lumaMin = LumaRange.x,
			.lumaMax = LumaRange.y,
			.viewportSize = Dimensions,
			.exposureAdaptationFactors = getAdaptationFactorFromFrameDelta(float(microsecondsElapsedBetweenPresents.count()) * 1e-6f),
			.pLumaMeterBuf = (MeterMode == MeteringMode::AVERAGE) ? m_gatherBuffer->getDeviceAddress() : m_histoBuffer->getDeviceAddress(),
			.pLastFrameEVBuf = m_lastFrameEVBuffers[resourceIx]->getDeviceAddress(),
		};
		pc.pCurrFrameEVBuf = m_lastFrameEVBuffers[(resourceIx+1)%MaxFramesInFlight]->getDeviceAddress();

		auto queue = getGraphicsQueue();
		auto cmdbuf = m_cmdBufs[resourceIx].get();
		cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cmdbuf->beginDebugMarker("Autoexposure Frame");

		// Luma Meter
		{
			auto ds = m_gpuImgDS.get();

			const float32_t2 meteringUVRange = MeteringMaxUV - MeteringMinUV;
			const uint32_t2 dispatchSize = uint32_t2(hlsl::ceil(float32_t2(gpuImgExtent.width, gpuImgExtent.height) * meteringUVRange / (m_subgroupSize * SamplingFactor)));

			pc.window = luma_meter::MeteringWindow::create(meteringUVRange / (float32_t2(dispatchSize) * static_cast<float>(m_subgroupSize)), MeteringMinUV);
			pc.rcpFirstPassWGCount = 1.f / float(dispatchSize.x * dispatchSize.y);

			uint32_t totalSampleCount = dispatchSize.x * m_subgroupSize * dispatchSize.y * m_subgroupSize;
			pc.lowerBoundPercentile = uint32_t(PercentileRange.x * totalSampleCount);
			pc.upperBoundPercentile = uint32_t(PercentileRange.y * totalSampleCount);

			cmdbuf->bindComputePipeline(m_meterPipeline.get());
			cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_meterPipeline->getLayout(), 0, 1, &ds); // also if you created DS Set with 3th index you need to respect it here - firstSet tells you the index of set and count tells you what range from this index it should update, useful if you had 2 DS with lets say set index 2,3, then you can bind both with single call setting firstSet to 2, count to 2 and last argument would be pointet to your DS pointers
			cmdbuf->pushConstants(m_meterPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(pc), &pc);
			cmdbuf->dispatch(dispatchSize.x, dispatchSize.y);
		}

		// Luma Gather and Tonemapping
		{
			{
				IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
				imageBarriers[0].barrier = {
					 .dep = {
						 .srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
						 .srcAccessMask = ACCESS_FLAGS::NONE,
						 .dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
						 .dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
					}
				};
				imageBarriers[0].image = m_tonemappedImgView->getCreationParameters().image.get();
				imageBarriers[0].subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = 1u,
					.baseArrayLayer = 0u,
					.layerCount = 1u
				};
				imageBarriers[0].oldLayout = IImage::LAYOUT::UNDEFINED;
				imageBarriers[0].newLayout = IImage::LAYOUT::GENERAL;
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
			}

			auto ds1 = m_gpuImgDS.get();
			auto ds2 = m_tonemappedImgRWDS.get();

			const uint32_t2 dispatchSize = {
				1 + ((Dimensions.x) - 1) / m_subgroupSize,
				1 + ((Dimensions.y) - 1) / m_subgroupSize
			};

			cmdbuf->bindComputePipeline(m_tonemapPipeline.get());
			cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_tonemapPipeline->getLayout(), 0, 1, &ds1);
			cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_tonemapPipeline->getLayout(), 3, 1, &ds2);
			cmdbuf->pushConstants(m_tonemapPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(pc), &pc);
			cmdbuf->dispatch(dispatchSize.x, dispatchSize.y);
		}

		// Render to swapchain
		{
			{
				IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
				imageBarriers[0].barrier = {
					 .dep = {
						 .srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
						 .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
						 .dstStageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS,
						 .dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
					}
				};
				imageBarriers[0].image = m_tonemappedImgView->getCreationParameters().image.get();
				imageBarriers[0].subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = 1u,
					.baseArrayLayer = 0u,
					.layerCount = 1u
				};
				imageBarriers[0].oldLayout = IImage::LAYOUT::GENERAL;
				imageBarriers[0].newLayout = IImage::LAYOUT::GENERAL;
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
			}

			auto ds = m_tonemappedImgSamplerDS.get();

			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = { m_window->getWidth(), m_window->getHeight() }
			};
			// set viewport
			{
				const asset::SViewport viewport =
				{
					.width = float32_t(m_window->getWidth()),
					.height = float32_t(m_window->getHeight())
				};
				cmdbuf->setViewport({ &viewport, 1 });
			}
			cmdbuf->setScissor({ &currentRenderArea, 1 });

			// begin the renderpass
			{
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
				const IGPUCommandBuffer::SRenderpassBeginInfo beginInfo =
				{
					.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = nullptr,
					.renderArea = currentRenderArea
				};

				cmdbuf->beginRenderPass(beginInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
			}

			cmdbuf->bindGraphicsPipeline(m_presentPipeline.get());
			cmdbuf->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS, m_presentPipeline->getLayout(), 3, 1, &ds);
			ext::FullScreenTriangle::recordDrawCall(cmdbuf);
			cmdbuf->endRenderPass();
		}

		cmdbuf->endDebugMarker();
		cmdbuf->end();

		{
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
			{
				{
					.semaphore = m_semaphore.get(),
					.value = ++m_realFrameIx,
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
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

					if (queue->submit(infos) == IQueue::RESULT::SUCCESS)
					{
						const nbl::video::ISemaphore::SWaitInfo waitInfos[] =
						{ {
							.semaphore = m_semaphore.get(),
							.value = m_realFrameIx
						} };

						m_device->blockForSemaphores(waitInfos); // this is not solution, quick wa to not throw validation errors
					}
					else
						--m_realFrameIx;
				}
			}

			std::string caption = "[Nabla Engine] Autoexposure Example";
			m_window->setCaption(caption);
			m_surface->present(m_currentImageAcquire.imageIndex, rendered);
		}
	}

	inline bool keepRunning() override
	{
		// Keep arunning as long as we have a surface to present to (usually this means, as long as the window is open)
		if (m_surface->irrecoverable())
			return false;

		return true;
	}

	inline bool onAppTerminated() override
	{
		return device_base_t::onAppTerminated();
	}

protected:
	float32_t2 getAdaptationFactorFromFrameDelta(float frameDeltaSeconds)
	{
		return hlsl::exp2(BaseExposureAdaptationFactorsLog2 * frameDeltaSeconds);
	}

	// window
	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;

	// Pipelines
	smart_refctd_ptr<IGPUComputePipeline> m_meterPipeline, m_tonemapPipeline;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_presentPipeline;

	// Descriptor Sets
	smart_refctd_ptr<IGPUDescriptorSet> m_gpuImgDS, m_tonemappedImgRWDS, m_tonemappedImgSamplerDS;

	// Command Buffers
	smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	smart_refctd_ptr<ISemaphore> m_semaphore;
	video::CDumbPresentationOracle oracle;

	// example resources
	uint32_t m_subgroupSize;
	uint32_t m_lastFrameEVIx = 0;
	smart_refctd_ptr<IGPUBuffer> m_gatherBuffer, m_histoBuffer;
	std::array<smart_refctd_ptr<IGPUBuffer>, MaxFramesInFlight> m_lastFrameEVBuffers;
	IDeviceMemoryAllocator::SAllocation m_gatherAllocation, m_histoAllocation;
	std::array<IDeviceMemoryAllocator::SAllocation, MaxFramesInFlight> m_lastLumaAllocations;
	void *m_gatherMemory, *m_histoMemory;
	smart_refctd_ptr<IGPUImageView> m_gpuImgView, m_tonemappedImgView;
	std::chrono::high_resolution_clock::time_point m_lastPresentStamp;
};

NBL_MAIN_FUNC(AutoexposureApp)
