// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "common.hpp"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/builtin/hlsl/indirect_commands.hlsl"

#include "nbl/examples/common/BuiltinResourcesApplication.hpp"


class RaytracingPipelineApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
	using device_base_t = SimpleWindowedApplication;
	using asset_base_t = BuiltinResourcesApplication;
	using clock_t = std::chrono::steady_clock;

	constexpr static inline uint32_t WIN_W = 1280, WIN_H = 720;
	constexpr static inline uint32_t MaxFramesInFlight = 3u;
	constexpr static inline uint8_t MaxUITextureCount = 1u;
	constexpr static inline uint32_t NumberOfProceduralGeometries = 5;

	static constexpr const char* s_lightTypeNames[E_LIGHT_TYPE::ELT_COUNT] = {
	  "Directional",
	  "Point",
	  "Spot"
	};

	struct ShaderBindingTable
	{
		SBufferRange<IGPUBuffer> raygenGroupRange;
		SBufferRange<IGPUBuffer> hitGroupsRange;
		uint32_t hitGroupsStride;
		SBufferRange<IGPUBuffer> missGroupsRange;
		uint32_t missGroupsStride;
		SBufferRange<IGPUBuffer> callableGroupsRange;
		uint32_t callableGroupsStride;
	};


public:
	inline RaytracingPipelineApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD)
	{
	}

	inline SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	{
		auto retval = device_base_t::getRequiredDeviceFeatures();
		retval.rayTracingPipeline = true;
		retval.accelerationStructure = true;
		retval.rayQuery = true;
		return retval;
	}

	inline SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
	{
		auto retval = device_base_t::getPreferredDeviceFeatures();
		retval.accelerationStructureHostCommands = true;
		return retval;
	}

	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		if (!m_surface)
		{
			{
				auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
				IWindow::SCreationParams params = {};
				params.callback = core::make_smart_refctd_ptr<ISimpleManagedSurface::ICallback>();
				params.width = WIN_W;
				params.height = WIN_H;
				params.x = 32;
				params.y = 32;
				params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
				params.windowCaption = "RaytracingPipelineApp";
				params.callback = windowCallback;
				const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
			}

			auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>::create(std::move(surface));
		}

		if (m_surface)
			return { {m_surface->getSurface()/*,EQF_NONE*/} };

		return {};
	}

	// so that we can use the same queue for asset converter and rendering
	inline core::vector<queue_req_t> getQueueRequirements() const override
	{
		auto reqs = device_base_t::getQueueRequirements();
		reqs.front().requiredFlags |= IQueue::FAMILY_FLAGS::COMPUTE_BIT;
		return reqs;
	}

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;

		if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;

		smart_refctd_ptr<IShaderCompiler::CCache> shaderReadCache = nullptr;
		smart_refctd_ptr<IShaderCompiler::CCache> shaderWriteCache = core::make_smart_refctd_ptr<IShaderCompiler::CCache>();
		auto shaderCachePath = localOutputCWD / "main_pipeline_shader_cache.bin";

		{
			core::smart_refctd_ptr<system::IFile> shaderReadCacheFile;
			{
				system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
				m_system->createFile(future, shaderCachePath.c_str(), system::IFile::ECF_READ);
				if (future.wait())
				{
					future.acquire().move_into(shaderReadCacheFile);
					if (shaderReadCacheFile)
					{
						const size_t size = shaderReadCacheFile->getSize();
						if (size > 0ull)
						{
							std::vector<uint8_t> contents(size);
							system::IFile::success_t succ;
							shaderReadCacheFile->read(succ, contents.data(), 0, size);
							if (succ)
								shaderReadCache = IShaderCompiler::CCache::deserialize(contents);
						}
					}
				}
				else
					m_logger->log("Failed Openning Shader Cache File.", ILogger::ELL_ERROR);
			}

		}

		// Load Custom Shader
		auto loadCompileAndCreateShader = [&](const std::string& relPath) -> smart_refctd_ptr<IShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = ""; // virtual root
				auto assetBundle = m_assetMgr->getAsset(relPath, lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
					return nullptr;

				// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
				auto sourceRaw = IAsset::castDown<IShader>(assets[0]);
				if (!sourceRaw)
					return nullptr;

				return m_device->compileShader({ sourceRaw.get(), nullptr, shaderReadCache.get(), shaderWriteCache.get() });
			};

		// load shaders
		const auto raygenShader = loadCompileAndCreateShader("app_resources/raytrace.rgen.hlsl");
		const auto closestHitShader = loadCompileAndCreateShader("app_resources/raytrace.rchit.hlsl");
		const auto proceduralClosestHitShader = loadCompileAndCreateShader("app_resources/raytrace_procedural.rchit.hlsl");
		const auto intersectionHitShader = loadCompileAndCreateShader("app_resources/raytrace.rint.hlsl");
		const auto anyHitShaderColorPayload = loadCompileAndCreateShader("app_resources/raytrace.rahit.hlsl");
		const auto anyHitShaderShadowPayload = loadCompileAndCreateShader("app_resources/raytrace_shadow.rahit.hlsl");
		const auto missShader = loadCompileAndCreateShader("app_resources/raytrace.rmiss.hlsl");
		const auto missShadowShader = loadCompileAndCreateShader("app_resources/raytrace_shadow.rmiss.hlsl");
		const auto directionalLightCallShader = loadCompileAndCreateShader("app_resources/light_directional.rcall.hlsl");
		const auto pointLightCallShader = loadCompileAndCreateShader("app_resources/light_point.rcall.hlsl");
		const auto spotLightCallShader = loadCompileAndCreateShader("app_resources/light_spot.rcall.hlsl");
		const auto fragmentShader = loadCompileAndCreateShader("app_resources/present.frag.hlsl");

		core::smart_refctd_ptr<system::IFile> shaderWriteCacheFile;
		{
			system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
			m_system->deleteFile(shaderCachePath); // temp solution instead of trimming, to make sure we won't have corrupted json
			m_system->createFile(future, shaderCachePath.c_str(), system::IFile::ECF_WRITE);
			if (future.wait())
			{
				future.acquire().move_into(shaderWriteCacheFile);
				if (shaderWriteCacheFile)
				{
					auto serializedCache = shaderWriteCache->serialize();
					if (shaderWriteCacheFile)
					{
						system::IFile::success_t succ;
						shaderWriteCacheFile->write(succ, serializedCache->getPointer(), 0, serializedCache->getSize());
						if (!succ)
							m_logger->log("Failed Writing To Shader Cache File.", ILogger::ELL_ERROR);
					}
				}
				else
					m_logger->log("Failed Creating Shader Cache File.", ILogger::ELL_ERROR);
			}
			else
				m_logger->log("Failed Creating Shader Cache File.", ILogger::ELL_ERROR);
		}

		m_semaphore = m_device->createSemaphore(m_realFrameIx);
		if (!m_semaphore)
			return logFail("Failed to Create a Semaphore!");

		auto gQueue = getGraphicsQueue();

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

			if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");
		}

		auto pool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

		m_converter = CAssetConverter::create({ .device = m_device.get(), .optimizer = {} });

		for (auto i = 0u; i < MaxFramesInFlight; i++)
		{
			if (!pool)
				return logFail("Couldn't create Command Pool!");
			if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
				return logFail("Couldn't create Command Buffer!");
		}

		m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
		m_surface->recreateSwapchain();


		// create output images
		m_hdrImage = m_device->createImage({
			{
			  .type = IGPUImage::ET_2D,
			  .samples = ICPUImage::ESCF_1_BIT,
			  .format = EF_R16G16B16A16_SFLOAT,
			  .extent = {WIN_W, WIN_H, 1},
			  .mipLevels = 1,
			  .arrayLayers = 1,
			  .flags = IImage::ECF_NONE,
			  .usage = bitflag(IImage::EUF_STORAGE_BIT) | IImage::EUF_TRANSFER_SRC_BIT | IImage::EUF_SAMPLED_BIT
			}
			});

		if (!m_hdrImage || !m_device->allocate(m_hdrImage->getMemoryReqs(), m_hdrImage.get()).isValid())
			return logFail("Could not create HDR Image");

		m_hdrImageView = m_device->createImageView({
		  .flags = IGPUImageView::ECF_NONE,
		  .subUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT | IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT,
		  .image = m_hdrImage,
		  .viewType = IGPUImageView::E_TYPE::ET_2D,
		  .format = asset::EF_R16G16B16A16_SFLOAT
			});



		// ray trace pipeline and descriptor set layout setup
		{
			const auto bindings = std::array<const ICPUDescriptorSetLayout::SBinding, 2>{
			  ICPUDescriptorSetLayout::SBinding{
          .binding = 0,
          .type = asset::IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE,
          .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
          .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_RAYGEN,
          .count = 1,
			  },
			  {
          .binding = 1,
          .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
          .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
          .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_RAYGEN,
          .count = 1,
			  }
			};
			auto cpuDescriptorSetLayout = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindings);

			const SPushConstantRange pcRange = {
			  .stageFlags = IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING,
			  .offset = 0u,
			  .size = sizeof(SPushConstants),
			};
			const auto cpuPipelineLayout = core::make_smart_refctd_ptr<ICPUPipelineLayout>(std::span<const asset::SPushConstantRange>({ pcRange }), std::move(cpuDescriptorSetLayout), nullptr, nullptr, nullptr);

			const auto pipeline = ICPURayTracingPipeline::create(cpuPipelineLayout.get());
			pipeline->getCachedCreationParams() = {
				.flags = IGPURayTracingPipeline::SCreationParams::FLAGS::NO_NULL_INTERSECTION_SHADERS,
				.maxRecursionDepth = 1,
				.dynamicStackSize = true,
			};

			pipeline->getSpecInfos(ESS_RAYGEN)[0] = {
				.shader = raygenShader,
				.entryPoint = "main",
			};

			pipeline->getSpecInfoVector(ESS_MISS)->resize(EMT_COUNT);
			const auto missGroups = pipeline->getSpecInfos(ESS_MISS);
			missGroups[EMT_PRIMARY] = { .shader = missShader, .entryPoint = "main" };
			missGroups[EMT_OCCLUSION] = { .shader = missShadowShader, .entryPoint = "main" };

			auto getHitGroupIndex = [](E_GEOM_TYPE geomType, E_RAY_TYPE rayType)
				{
					return geomType * ERT_COUNT + rayType;
				};

			const auto hitGroupCount = ERT_COUNT * EGT_COUNT;
			pipeline->getSpecInfoVector(ESS_CLOSEST_HIT)->resize(hitGroupCount);
			pipeline->getSpecInfoVector(ESS_ANY_HIT)->resize(hitGroupCount);
			pipeline->getSpecInfoVector(ESS_INTERSECTION)->resize(hitGroupCount);

			const auto closestHitSpecs = pipeline->getSpecInfos(ESS_CLOSEST_HIT);
			const auto anyHitSpecs = pipeline->getSpecInfos(ESS_ANY_HIT);
			const auto intersectionSpecs = pipeline->getSpecInfos(ESS_INTERSECTION);

			closestHitSpecs[getHitGroupIndex(EGT_TRIANGLES, ERT_PRIMARY)] = { .shader = closestHitShader, .entryPoint = "main" };
			anyHitSpecs[getHitGroupIndex(EGT_TRIANGLES, ERT_PRIMARY)] = {.shader = anyHitShaderColorPayload, .entryPoint = "main"};

			anyHitSpecs[getHitGroupIndex(EGT_TRIANGLES, ERT_OCCLUSION)] = { .shader = anyHitShaderShadowPayload, .entryPoint = "main" };

			closestHitSpecs[getHitGroupIndex(EGT_PROCEDURAL, ERT_PRIMARY)] = { .shader = proceduralClosestHitShader, .entryPoint = "main" };
			anyHitSpecs[getHitGroupIndex(EGT_PROCEDURAL, ERT_PRIMARY)] = { .shader = anyHitShaderColorPayload, .entryPoint = "main" };
			intersectionSpecs[getHitGroupIndex(EGT_PROCEDURAL, ERT_PRIMARY)] = { .shader = intersectionHitShader, .entryPoint = "main" };

			anyHitSpecs[getHitGroupIndex(EGT_PROCEDURAL, ERT_OCCLUSION)] = {.shader = anyHitShaderShadowPayload, .entryPoint = "main" };
			intersectionSpecs[getHitGroupIndex(EGT_PROCEDURAL, ERT_OCCLUSION)] = { .shader = intersectionHitShader, .entryPoint = "main" };

			pipeline->getSpecInfoVector(ESS_CALLABLE)->resize(ELT_COUNT);
			const auto callableGroups = pipeline->getSpecInfos(ESS_CALLABLE);
			callableGroups[ELT_DIRECTIONAL] = { .shader = directionalLightCallShader, .entryPoint = "main" };
			callableGroups[ELT_POINT] = { .shader = pointLightCallShader, .entryPoint = "main" };
			callableGroups[ELT_SPOT] = { .shader = spotLightCallShader, .entryPoint = "main" };

      smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = m_device.get(), .optimizer = {} });
		  CAssetConverter::SInputs inputs = {};
      inputs.logger = m_logger.get();

			const std::array cpuPipelines = { pipeline.get() };
			std::get<CAssetConverter::SInputs::asset_span_t<ICPURayTracingPipeline>>(inputs.assets) = cpuPipelines;

			CAssetConverter::SConvertParams params = {};
			params.utilities = m_utils.get();

      auto reservation = converter->reserve(inputs);
			auto future = reservation.convert(params);
			if (future.copy() != IQueue::RESULT::SUCCESS)
			{
				m_logger->log("Failed to await submission feature!", ILogger::ELL_ERROR);
				return false;
			}

			// assign gpu objects to output
			auto&& pipelines = reservation.getGPUObjects<ICPURayTracingPipeline>();
			m_rayTracingPipeline = pipelines[0].value;
			const auto* gpuDsLayout = m_rayTracingPipeline->getLayout()->getDescriptorSetLayouts()[0];

			const std::array<const IGPUDescriptorSetLayout*, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> dsLayoutPtrs = { gpuDsLayout };
      m_rayTracingDsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dsLayoutPtrs.begin(), dsLayoutPtrs.end()));
			m_rayTracingDs = m_rayTracingDsPool->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(gpuDsLayout));

			calculateRayTracingStackSize(m_rayTracingPipeline);

			if (!createShaderBindingTable(m_rayTracingPipeline))
				return logFail("Could not create shader binding table");

		}

		auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(system));

		if (!createIndirectBuffer())
			return logFail("Could not create indirect buffer");

		if (!createAccelerationStructuresFromGeometry())
			return logFail("Could not create acceleration structures from geometry creator");

		ISampler::SParams samplerParams = {
		  .AnisotropicFilter = 0
		};
		auto defaultSampler = m_device->createSampler(samplerParams);

		{
			const IGPUDescriptorSetLayout::SBinding bindings[] = {
			  {
				.binding = 0u,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
				.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
				.count = 1u,
				.immutableSamplers = &defaultSampler
			  }
			};
			auto gpuPresentDescriptorSetLayout = m_device->createDescriptorSetLayout(bindings);
			const video::IGPUDescriptorSetLayout* const layouts[] = { gpuPresentDescriptorSetLayout.get() };
			const uint32_t setCounts[] = { 1u };
			m_presentDsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
			m_presentDs = m_presentDsPool->createDescriptorSet(gpuPresentDescriptorSetLayout);

			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
			if (!fsTriProtoPPln)
				return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

			const IGPUPipelineBase::SShaderSpecInfo fragSpec = {
			  .shader = fragmentShader.get(),
			  .entryPoint = "main",
			};

			auto presentLayout = m_device->createPipelineLayout(
				{},
				core::smart_refctd_ptr(gpuPresentDescriptorSetLayout),
				nullptr,
				nullptr,
				nullptr
			);
			m_presentPipeline = fsTriProtoPPln.createPipeline(fragSpec, presentLayout.get(), scRes->getRenderpass());
			if (!m_presentPipeline)
				return logFail("Could not create Graphics Pipeline!");
		}

		// write descriptors
		IGPUDescriptorSet::SDescriptorInfo infos[3];
		infos[0].desc = m_gpuTlas;

		infos[1].desc = m_hdrImageView;
		if (!infos[1].desc)
			return logFail("Failed to create image view");
		infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;

		infos[2].desc = m_hdrImageView;
		infos[2].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

		IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
			{.dstSet = m_rayTracingDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
			{.dstSet = m_rayTracingDs.get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]},
			{.dstSet = m_presentDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[2] },
		};
		m_device->updateDescriptorSets(std::span(writes), {});

		// gui descriptor setup
		{
			using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
			{
				IGPUSampler::SParams params;
				params.AnisotropicFilter = 1u;
				params.TextureWrapU = ETC_REPEAT;
				params.TextureWrapV = ETC_REPEAT;
				params.TextureWrapW = ETC_REPEAT;

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
			params.pipelineCache = nullptr;
			params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, MaxUITextureCount);
			params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
			params.streamingBuffer = nullptr;
			params.subpassIx = 0u;
			params.transfer = getGraphicsQueue();
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

						projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(
							core::radians(m_cameraSetting.fov),
							io.DisplaySize.x / io.DisplaySize.y,
							m_cameraSetting.zNear,
							m_cameraSetting.zFar);

						return projection;
					}());

				ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
				ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

				// create a window and insert the inspector
				ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
				ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
				ImGui::Begin("Controls");

				ImGui::SameLine();

				ImGui::Text("Camera");

				ImGui::SliderFloat("Move speed", &m_cameraSetting.moveSpeed, 0.1f, 10.f);
				ImGui::SliderFloat("Rotate speed", &m_cameraSetting.rotateSpeed, 0.1f, 10.f);
				ImGui::SliderFloat("Fov", &m_cameraSetting.fov, 20.f, 150.f);
				ImGui::SliderFloat("zNear", &m_cameraSetting.zNear, 0.1f, 100.f);
				ImGui::SliderFloat("zFar", &m_cameraSetting.zFar, 110.f, 10000.f);
				Light m_oldLight = m_light;
				int light_type = m_light.type;
				ImGui::ListBox("LightType", &light_type, s_lightTypeNames, ELT_COUNT);
				m_light.type = static_cast<E_LIGHT_TYPE>(light_type);
				if (m_light.type == ELT_DIRECTIONAL)
				{
					ImGui::SliderFloat3("Light Direction", &m_light.direction.x, -1.f, 1.f);
				}
				else if (m_light.type == ELT_POINT)
				{
					ImGui::SliderFloat3("Light Position", &m_light.position.x, -20.f, 20.f);
				}
				else if (m_light.type == ELT_SPOT)
				{
					ImGui::SliderFloat3("Light Direction", &m_light.direction.x, -1.f, 1.f);
					ImGui::SliderFloat3("Light Position", &m_light.position.x, -20.f, 20.f);

					float32_t dOuterCutoff = hlsl::degrees(acos(m_light.outerCutoff));
					if (ImGui::SliderFloat("Light Outer Cutoff", &dOuterCutoff, 0.0f, 45.0f))
					{
						m_light.outerCutoff = cos(hlsl::radians(dOuterCutoff));
					}
				}
				ImGui::Checkbox("Use Indirect Command", &m_useIndirectCommand);
				if (m_light != m_oldLight)
				{
					m_frameAccumulationCounter = 0;
				}

				ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);

				ImGui::End();
			}
		);

		// Set Camera
		{
			core::vectorSIMDf cameraPosition(0, 5, -10);
			matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(
				core::radians(60.0f),
				WIN_W / WIN_H,
				0.01f,
				500.0f
			);
			m_camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), proj);
		}

		m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
		m_surface->recreateSwapchain();
		m_winMgr->show(m_window.get());
		m_oracle.reportBeginFrameRecord();
		m_camera.mapKeysToWASD();

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

		m_api->startCapture();

		update();

		auto queue = getGraphicsQueue();
		auto cmdbuf = m_cmdBufs[resourceIx].get();

		if (!keepRunning())
			return;

		cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cmdbuf->beginDebugMarker("RaytracingPipelineApp Frame");

		const auto viewMatrix = m_camera.getViewMatrix();
		const auto projectionMatrix = m_camera.getProjectionMatrix();
		const auto viewProjectionMatrix = m_camera.getConcatenatedMatrix();

		core::matrix3x4SIMD modelMatrix;
		modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
		modelMatrix.setRotation(quaternion(0, 0, 0));

		core::matrix4SIMD modelViewProjectionMatrix = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);
		if (m_cachedModelViewProjectionMatrix != modelViewProjectionMatrix)
		{
			m_frameAccumulationCounter = 0;
			m_cachedModelViewProjectionMatrix = modelViewProjectionMatrix;
		}
		core::matrix4SIMD invModelViewProjectionMatrix;
		modelViewProjectionMatrix.getInverseTransform(invModelViewProjectionMatrix);

		{
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
			imageBarriers[0].barrier = {
			   .dep = {
				 .srcStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT, // previous frame read from framgent shader
				 .srcAccessMask = ACCESS_FLAGS::SHADER_READ_BITS,
				 .dstStageMask = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT,
				 .dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
			  }
			};
			imageBarriers[0].image = m_hdrImage.get();
			imageBarriers[0].subresourceRange = {
			  .aspectMask = IImage::EAF_COLOR_BIT,
			  .baseMipLevel = 0u,
			  .levelCount = 1u,
			  .baseArrayLayer = 0u,
			  .layerCount = 1u
			};
			imageBarriers[0].oldLayout = m_frameAccumulationCounter == 0 ? IImage::LAYOUT::UNDEFINED : IImage::LAYOUT::READ_ONLY_OPTIMAL;
			imageBarriers[0].newLayout = IImage::LAYOUT::GENERAL;
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
		}

		// Trace Rays Pass
		{
			SPushConstants pc;
			pc.light = m_light;
			pc.proceduralGeomInfoBuffer = m_proceduralGeomInfoBuffer->getDeviceAddress();
			pc.triangleGeomInfoBuffer = m_triangleGeomInfoBuffer->getDeviceAddress();
			pc.frameCounter = m_frameAccumulationCounter;
			const core::vector3df camPos = m_camera.getPosition().getAsVector3df();
			pc.camPos = { camPos.X, camPos.Y, camPos.Z };
			memcpy(&pc.invMVP, invModelViewProjectionMatrix.pointer(), sizeof(pc.invMVP));

			cmdbuf->bindRayTracingPipeline(m_rayTracingPipeline.get());
			cmdbuf->setRayTracingPipelineStackSize(m_rayTracingStackSize);
			cmdbuf->pushConstants(m_rayTracingPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING, 0, sizeof(SPushConstants), &pc);
			cmdbuf->bindDescriptorSets(EPBP_RAY_TRACING, m_rayTracingPipeline->getLayout(), 0, 1, &m_rayTracingDs.get());
			if (m_useIndirectCommand)
			{
				cmdbuf->traceRaysIndirect(
					SBufferBinding<const IGPUBuffer>{
					.offset = 0,
						.buffer = m_indirectBuffer,
				});
			}
			else
			{
				cmdbuf->traceRays(
					m_shaderBindingTable.raygenGroupRange,
					m_shaderBindingTable.missGroupsRange, m_shaderBindingTable.missGroupsStride,
					m_shaderBindingTable.hitGroupsRange, m_shaderBindingTable.hitGroupsStride,
					m_shaderBindingTable.callableGroupsRange, m_shaderBindingTable.callableGroupsStride,
					WIN_W, WIN_H, 1);
			}
		}

		// pipeline barrier
		{
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
			imageBarriers[0].barrier = {
			  .dep = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT,
				.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
				.dstStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
				.dstAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
			  }
			};
			imageBarriers[0].image = m_hdrImage.get();
			imageBarriers[0].subresourceRange = {
			  .aspectMask = IImage::EAF_COLOR_BIT,
			  .baseMipLevel = 0u,
			  .levelCount = 1u,
			  .baseArrayLayer = 0u,
			  .layerCount = 1u
			};
			imageBarriers[0].oldLayout = IImage::LAYOUT::GENERAL;
			imageBarriers[0].newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
		}

		{
			asset::SViewport viewport;
			{
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = WIN_W;
				viewport.height = WIN_H;
			}
			cmdbuf->setViewport(0u, 1u, &viewport);


			VkRect2D defaultScisors[] = { {.offset = {(int32_t)viewport.x, (int32_t)viewport.y}, .extent = {(uint32_t)viewport.width, (uint32_t)viewport.height}} };
			cmdbuf->setScissor(defaultScisors);

			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			const VkRect2D currentRenderArea =
			{
			  .offset = {0,0},
			  .extent = {m_window->getWidth(),m_window->getHeight()}
			};
			const IGPUCommandBuffer::SClearColorValue clearColor = { .float32 = {0.f,0.f,0.f,1.f} };
			const IGPUCommandBuffer::SRenderpassBeginInfo info =
			{
			  .framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
			  .colorClearValues = &clearColor,
			  .depthStencilClearValues = nullptr,
			  .renderArea = currentRenderArea
			};
			nbl::video::ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };

			cmdbuf->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

			cmdbuf->bindGraphicsPipeline(m_presentPipeline.get());
			cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, m_presentPipeline->getLayout(), 0, 1u, &m_presentDs.get());
			ext::FullScreenTriangle::recordDrawCall(cmdbuf);

			const auto uiParams = m_ui.manager->getCreationParameters();
			auto* uiPipeline = m_ui.manager->getPipeline();
			cmdbuf->bindGraphicsPipeline(uiPipeline);
			cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, uiPipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get());
			m_ui.manager->render(cmdbuf, waitInfo);

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

					updateGUIDescriptorSet();

					if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
						m_realFrameIx--;
				}
			}

			m_window->setCaption("[Nabla Engine] Ray Tracing Pipeline");
			m_surface->present(m_currentImageAcquire.imageIndex, rendered);
		}
		m_api->endCapture();
		m_frameAccumulationCounter++;
	}

	inline void update()
	{
		m_camera.setMoveSpeed(m_cameraSetting.moveSpeed);
		m_camera.setRotateSpeed(m_cameraSetting.rotateSpeed);

		static std::chrono::microseconds previousEventTimestamp{};

		m_inputSystem->getDefaultMouse(&m_mouse);
		m_inputSystem->getDefaultKeyboard(&m_keyboard);

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
			m_mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
				{
					if (!io.WantCaptureMouse)
						m_camera.mouseProcess(events); // don't capture the events, only let camera handle them with its impl

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						capturedEvents.mouse.emplace_back(e);

					}
				}, m_logger.get());

			m_keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
				{
					if (!io.WantCaptureKeyboard)
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

		m_ui.manager->update(params);
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

private:
	uint32_t getWorkgroupCount(uint32_t dim, uint32_t size)
	{
		return (dim + size - 1) / size;
	}

	bool createIndirectBuffer()
	{
		const auto getBufferRangeAddress = [](const SBufferRange<IGPUBuffer>& range)
			{
				return range.buffer->getDeviceAddress() + range.offset;
			};
		const auto command = TraceRaysIndirectCommand_t{
		  .raygenShaderRecordAddress = getBufferRangeAddress(m_shaderBindingTable.raygenGroupRange),
		  .raygenShaderRecordSize = m_shaderBindingTable.raygenGroupRange.size,
		  .missShaderBindingTableAddress = getBufferRangeAddress(m_shaderBindingTable.missGroupsRange),
		  .missShaderBindingTableSize = m_shaderBindingTable.missGroupsRange.size,
		  .missShaderBindingTableStride = m_shaderBindingTable.missGroupsStride,
		  .hitShaderBindingTableAddress = getBufferRangeAddress(m_shaderBindingTable.hitGroupsRange),
		  .hitShaderBindingTableSize = m_shaderBindingTable.hitGroupsRange.size,
		  .hitShaderBindingTableStride = m_shaderBindingTable.hitGroupsStride,
		  .callableShaderBindingTableAddress = getBufferRangeAddress(m_shaderBindingTable.callableGroupsRange),
		  .callableShaderBindingTableSize = m_shaderBindingTable.callableGroupsRange.size,
		  .callableShaderBindingTableStride = m_shaderBindingTable.callableGroupsStride,
		  .width = WIN_W,
		  .height = WIN_H,
		  .depth = 1,
		};
		IGPUBuffer::SCreationParams params;
		params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INDIRECT_BUFFER_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
		params.size = sizeof(TraceRaysIndirectCommand_t);
		m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = getGraphicsQueue() }, std::move(params), &command).move_into(m_indirectBuffer);
		return true;
	}

	void calculateRayTracingStackSize(const smart_refctd_ptr<video::IGPURayTracingPipeline>& pipeline)
	{
		const auto raygenStackSize = pipeline->getRaygenStackSize();
		auto getMaxSize = [&](auto ranges, auto valProj) -> uint16_t
			{
				auto maxValue = 0;
				for (const auto& val : ranges)
				{
					maxValue = std::max<uint16_t>(maxValue, std::invoke(valProj, val));
				}
				return maxValue;
			};

		const auto closestHitStackMax = getMaxSize(pipeline->getHitStackSizes(), &IGPURayTracingPipeline::SHitGroupStackSize::closestHit);
		const auto anyHitStackMax = getMaxSize(pipeline->getHitStackSizes(), &IGPURayTracingPipeline::SHitGroupStackSize::anyHit);
		const auto intersectionStackMax = getMaxSize(pipeline->getHitStackSizes(), &IGPURayTracingPipeline::SHitGroupStackSize::intersection);
		const auto missStackMax = getMaxSize(pipeline->getMissStackSizes(), std::identity{});
		const auto callableStackMax = getMaxSize(pipeline->getCallableStackSizes(), std::identity{});
		auto firstDepthStackSizeMax = std::max(closestHitStackMax, missStackMax);
		firstDepthStackSizeMax = std::max<uint16_t>(firstDepthStackSizeMax, intersectionStackMax + anyHitStackMax);
		m_rayTracingStackSize = raygenStackSize + std::max(firstDepthStackSizeMax, callableStackMax);
	}

	bool createShaderBindingTable(const smart_refctd_ptr<video::IGPURayTracingPipeline>& pipeline)
	{
		const auto& limits = m_device->getPhysicalDevice()->getLimits();
		const auto handleSize = SPhysicalDeviceLimits::ShaderGroupHandleSize;
		const auto handleSizeAligned = nbl::core::alignUp(handleSize, limits.shaderGroupHandleAlignment);

		auto& raygenRange = m_shaderBindingTable.raygenGroupRange;

		auto& hitRange = m_shaderBindingTable.hitGroupsRange;
		const auto hitHandles = pipeline->getHitHandles();

		auto& missRange = m_shaderBindingTable.missGroupsRange;
		const auto missHandles = pipeline->getMissHandles();

		auto& callableRange = m_shaderBindingTable.callableGroupsRange;
		const auto callableHandles = pipeline->getCallableHandles();

		raygenRange = {
		  .offset = 0,
		  .size = core::alignUp(handleSizeAligned, limits.shaderGroupBaseAlignment)
		};

		missRange = {
		  .offset = raygenRange.size,
		  .size = core::alignUp(missHandles.size() * handleSizeAligned, limits.shaderGroupBaseAlignment),
		};
		m_shaderBindingTable.missGroupsStride = handleSizeAligned;

		hitRange = {
		  .offset = missRange.offset + missRange.size,
		  .size = core::alignUp(hitHandles.size() * handleSizeAligned, limits.shaderGroupBaseAlignment),
		};
		m_shaderBindingTable.hitGroupsStride = handleSizeAligned;

		callableRange = {
		  .offset = hitRange.offset + hitRange.size,
		  .size = core::alignUp(callableHandles.size() * handleSizeAligned, limits.shaderGroupBaseAlignment),
		};
		m_shaderBindingTable.callableGroupsStride = handleSizeAligned;

		const auto bufferSize = raygenRange.size + missRange.size + hitRange.size + callableRange.size;

		ICPUBuffer::SCreationParams cpuBufferParams;
		cpuBufferParams.size = bufferSize;
		auto cpuBuffer = ICPUBuffer::create(std::move(cpuBufferParams));
		uint8_t* pData = reinterpret_cast<uint8_t*>(cpuBuffer->getPointer());

		// copy raygen region
		memcpy(pData, &pipeline->getRaygen(), handleSize);

		// copy miss region
		uint8_t* pMissData = pData + missRange.offset;
		for (const auto& handle : missHandles)
		{
			memcpy(pMissData, &handle, handleSize);
			pMissData += m_shaderBindingTable.missGroupsStride;
		}

		// copy hit region
		uint8_t* pHitData = pData + hitRange.offset;
		for (const auto& handle : hitHandles)
		{
			memcpy(pHitData, &handle, handleSize);
			pHitData += m_shaderBindingTable.hitGroupsStride;
		}

		// copy callable region
		uint8_t* pCallableData = pData + callableRange.offset;
		for (const auto& handle : callableHandles)
		{
			memcpy(pCallableData, &handle, handleSize);
			pCallableData += m_shaderBindingTable.callableGroupsStride;
		}

		{
			IGPUBuffer::SCreationParams params;
			params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | IGPUBuffer::EUF_SHADER_BINDING_TABLE_BIT;
			params.size = bufferSize;
			m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = getGraphicsQueue() }, std::move(params), pData).move_into(raygenRange.buffer);
			missRange.buffer = core::smart_refctd_ptr(raygenRange.buffer);
			hitRange.buffer = core::smart_refctd_ptr(raygenRange.buffer);
			callableRange.buffer = core::smart_refctd_ptr(raygenRange.buffer);
		}

		return true;
	}

	bool createAccelerationStructuresFromGeometry()
	{
		auto queue = getGraphicsQueue();
		// get geometries into ICPUBuffers
		auto pool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		if (!pool)
			return logFail("Couldn't create Command Pool for geometry creation!");

		const auto defaultMaterial = Material{
		  .ambient = {0.2, 0.1, 0.1},
		  .diffuse = {0.8, 0.3, 0.3},
		  .specular = {0.8, 0.8, 0.8},
		  .shininess = 1.0f,
		  .alpha = 1.0f,
		};

		auto getTranslationMatrix = [](float32_t x, float32_t y, float32_t z)
			{
				core::matrix3x4SIMD transform;
				transform.setTranslation(nbl::core::vectorSIMDf(x, y, z, 0));
				return transform;
			};

		core::matrix3x4SIMD planeTransform;
		planeTransform.setRotation(quaternion::fromAngleAxis(core::radians(-90.0f), vector3df_SIMD{ 1, 0, 0 }));

		// triangles geometries
		auto geometryCreator = make_smart_refctd_ptr<CGeometryCreator>();

		const auto cpuObjects = std::array{
			scene::ReferenceObjectCpu {
				.data = geometryCreator->createRectangle({10, 10}),
				.material = defaultMaterial,
				.transform = planeTransform,
			},
			scene::ReferenceObjectCpu {
				.data = geometryCreator->createCube({1, 1, 1}),
				.material = defaultMaterial,
				.transform = getTranslationMatrix(0, 0.5f, 0),
			},
			scene::ReferenceObjectCpu {
				.data = geometryCreator->createCube({1.5, 1.5, 1.5}),
				.material = Material{
					.ambient = {0.1, 0.1, 0.2},
					.diffuse = {0.2, 0.2, 0.8},
					.specular = {0.8, 0.8, 0.8},
					.shininess = 1.0f,
					.alpha = 1.0f,
				},
				.transform = getTranslationMatrix(-5.0f, 1.0f, 0),
			},
			scene::ReferenceObjectCpu {
				.data = geometryCreator->createCube({1.5, 1.5, 1.5}),
				.material = Material{
					.ambient = {0.1, 0.2, 0.1},
					.diffuse = {0.2, 0.8, 0.2},
					.specular = {0.8, 0.8, 0.8},
					.shininess = 1.0f,
					.alpha = 0.2,
				},
				.transform = getTranslationMatrix(5.0f, 1.0f, 0),
			},
		};

		// procedural geometries
		using Aabb = IGPUBottomLevelAccelerationStructure::AABB_t;

		smart_refctd_ptr<ICPUBuffer> cpuProcBuffer;
		{
			ICPUBuffer::SCreationParams params;
			params.size = NumberOfProceduralGeometries * sizeof(Aabb);
			cpuProcBuffer = ICPUBuffer::create(std::move(params));
		}

		core::vector<SProceduralGeomInfo> proceduralGeoms;
		proceduralGeoms.reserve(NumberOfProceduralGeometries);
		auto proceduralGeometries = reinterpret_cast<Aabb*>(cpuProcBuffer->getPointer());
		for (int32_t i = 0; i < NumberOfProceduralGeometries; i++)
		{
			const auto middle_i = NumberOfProceduralGeometries / 2.0;
			SProceduralGeomInfo sphere = {
					.material = hlsl::_static_cast<MaterialPacked>(Material{
					.ambient = {0.1, 0.05 * i, 0.1},
					.diffuse = {0.3, 0.2 * i, 0.3},
					.specular = {0.8, 0.8, 0.8},
					.shininess = 1.0f,
				}),
				.center = float32_t3((i - middle_i) * 4.0, 2, 5.0),
				.radius = 1,
			};

			proceduralGeoms.push_back(sphere);
			const auto sphereMin = sphere.center - sphere.radius;
			const auto sphereMax = sphere.center + sphere.radius;
			proceduralGeometries[i] = {
				vector3d(sphereMin.x, sphereMin.y, sphereMin.z),
				vector3d(sphereMax.x, sphereMax.y, sphereMax.z)
			};
		}

		{
			IGPUBuffer::SCreationParams params;
			params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
			params.size = proceduralGeoms.size() * sizeof(SProceduralGeomInfo);
			m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = queue }, std::move(params), proceduralGeoms.data()).move_into(m_proceduralGeomInfoBuffer);
		}

		// get ICPUBuffers into ICPUBLAS
		// TODO use one BLAS and multiple triangles/aabbs in one
		const auto blasCount = std::size(cpuObjects) + 1;
		const auto proceduralBlasIdx = std::size(cpuObjects);

		std::array<smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>, std::size(cpuObjects)+1u> cpuBlasList;
		for (uint32_t i = 0; i < blasCount; i++)
		{
			auto& blas = cpuBlasList[i];
			blas = make_smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>();

			if (i == proceduralBlasIdx)
			{
				auto aabbs = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUBottomLevelAccelerationStructure::AABBs<ICPUBuffer>>>(1u);
				auto primitiveCounts = make_refctd_dynamic_array<smart_refctd_dynamic_array<uint32_t>>(1u);

				auto& aabb = aabbs->front();
				auto& primCount = primitiveCounts->front();
				
				primCount = NumberOfProceduralGeometries;
				aabb.data = { .offset = 0, .buffer = cpuProcBuffer };
				aabb.stride = sizeof(IGPUBottomLevelAccelerationStructure::AABB_t);
				aabb.geometryFlags = IGPUBottomLevelAccelerationStructure::GEOMETRY_FLAGS::OPAQUE_BIT; // only allow opaque for now

				blas->setGeometries(std::move(aabbs), std::move(primitiveCounts));
			}
			else
			{
				auto triangles = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUBottomLevelAccelerationStructure::Triangles<ICPUBuffer>>>(1u);
				auto primitiveCounts = make_refctd_dynamic_array<smart_refctd_dynamic_array<uint32_t>>(1u);

				auto& tri = triangles->front();

				auto& primCount = primitiveCounts->front();
				primCount = cpuObjects[i].data->getPrimitiveCount();

				tri = cpuObjects[i].data->exportForBLAS();
				tri.geometryFlags = cpuObjects[i].material.isTransparent() ?
					IGPUBottomLevelAccelerationStructure::GEOMETRY_FLAGS::NO_DUPLICATE_ANY_HIT_INVOCATION_BIT :
					IGPUBottomLevelAccelerationStructure::GEOMETRY_FLAGS::OPAQUE_BIT;

				blas->setGeometries(std::move(triangles), std::move(primitiveCounts));
			}

			auto blasFlags = bitflag(IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT) | IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::ALLOW_COMPACTION_BIT;
			if (i == proceduralBlasIdx)
				blasFlags |= IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT;

			blas->setBuildFlags(blasFlags);
			blas->setContentHash(blas->computeContentHash());
		}

		auto geomInfoBuffer = ICPUBuffer::create({ std::size(cpuObjects) * sizeof(STriangleGeomInfo) });
		STriangleGeomInfo* geomInfos = reinterpret_cast<STriangleGeomInfo*>(geomInfoBuffer->getPointer());

		// get ICPUBLAS into ICPUTLAS
		auto geomInstances = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUTopLevelAccelerationStructure::PolymorphicInstance>>(blasCount);
		{
			uint32_t i = 0;
			for (auto instance = geomInstances->begin(); instance != geomInstances->end(); instance++, i++)
			{
				const auto isProceduralInstance = i == proceduralBlasIdx;
				ICPUTopLevelAccelerationStructure::StaticInstance inst;
				inst.base.blas = cpuBlasList[i];
				inst.base.flags = static_cast<uint32_t>(IGPUTopLevelAccelerationStructure::INSTANCE_FLAGS::TRIANGLE_FACING_CULL_DISABLE_BIT);
				inst.base.instanceCustomIndex = i;
				inst.base.instanceShaderBindingTableRecordOffset = isProceduralInstance ? 2 : 0;
				inst.base.mask = 0xFF;
				inst.transform = isProceduralInstance ? matrix3x4SIMD() : cpuObjects[i].transform;

				instance->instance = inst;
			}
		}

		auto cpuTlas = make_smart_refctd_ptr<ICPUTopLevelAccelerationStructure>();
		cpuTlas->setInstances(std::move(geomInstances));
		cpuTlas->setBuildFlags(IGPUTopLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT);

		// convert with asset converter
		smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = m_device.get(), .optimizer = {} });
		struct MyInputs : CAssetConverter::SInputs
		{
			// For the GPU Buffers to be directly writeable and so that we don't need a Transfer Queue submit at all
			inline uint32_t constrainMemoryTypeBits(const size_t groupCopyID, const IAsset* canonicalAsset, const blake3_hash_t& contentHash, const IDeviceMemoryBacked* memoryBacked) const override
			{
				assert(memoryBacked);
				return memoryBacked->getObjectType() != IDeviceMemoryBacked::EOT_BUFFER ? (~0u) : rebarMemoryTypes;
			}

			uint32_t rebarMemoryTypes;
		} inputs = {};
		inputs.logger = m_logger.get();
		inputs.rebarMemoryTypes = m_physicalDevice->getDirectVRAMAccessMemoryTypeBits();
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
					retval.memory->map({ .offset = 0,.length = info.size });
				return retval;
			}

			ILogicalDevice* device;
		} myalloc;
		myalloc.device = m_device.get();
		inputs.allocator = &myalloc;

		std::array<ICPUTopLevelAccelerationStructure*, 1u> tmpTlas;
		std::array<ICPUBuffer*, 1> tmpBuffers;
		std::array<ICPUPolygonGeometry*, std::size(cpuObjects)> tmpGeometries;
		std::array<CAssetConverter::patch_t<asset::ICPUPolygonGeometry>, std::size(cpuObjects)> tmpGeometryPatches;
		{
			tmpTlas[0] = cpuTlas.get();
			tmpBuffers[0] = cpuProcBuffer.get();
			for (uint32_t i = 0; i < cpuObjects.size(); i++)
			{
				tmpGeometries[i] = cpuObjects[i].data.get();
				tmpGeometryPatches[i].indexBufferUsages= IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;
			}

			std::get<CAssetConverter::SInputs::asset_span_t<ICPUTopLevelAccelerationStructure>>(inputs.assets) = tmpTlas;
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUBuffer>>(inputs.assets) = tmpBuffers;
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = tmpGeometries;
			std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = tmpGeometryPatches;
		}

		auto reservation = converter->reserve(inputs);
		{
			auto prepass = [&]<typename asset_type_t>(const auto & references) -> bool
			{
				auto objects = reservation.getGPUObjects<asset_type_t>();
				uint32_t counter = {};
				for (auto& object : objects)
				{
					auto gpu = object.value;
					auto* reference = references[counter];

					if (reference)
					{
						if (!gpu)
						{
							m_logger->log("Failed to convert a CPU object to GPU!", ILogger::ELL_ERROR);
							return false;
						}
					}
					counter++;
				}
				return true;
			};

			prepass.template operator() < ICPUTopLevelAccelerationStructure > (tmpTlas);
			prepass.template operator() < ICPUBuffer > (tmpBuffers);
			prepass.template operator() < ICPUPolygonGeometry > (tmpGeometries);
		}

		constexpr auto CompBufferCount = 2;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, CompBufferCount> compBufs = {};
		std::array<IQueue::SSubmitInfo::SCommandBufferInfo, CompBufferCount> compBufInfos = {};
		{
			auto pool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
			pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, compBufs);
			compBufs.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			for (auto i = 0; i < CompBufferCount; i++)
				compBufInfos[i].cmdbuf = compBufs[i].get();
		}
		auto compSema = m_device->createSemaphore(0u);
		SIntendedSubmitInfo compute = {};
		compute.queue = queue;
		compute.scratchCommandBuffers = compBufInfos;
		compute.scratchSemaphore = {
			.semaphore = compSema.get(),
			.value = 0u,
			.stageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT | PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_COPY_BIT
		};
		// convert
		{
			smart_refctd_ptr<CAssetConverter::SConvertParams::scratch_for_device_AS_build_t> scratchAlloc;
			{
				constexpr auto MaxAlignment = 256;
				constexpr auto MinAllocationSize = 1024;
				const auto scratchSize = core::alignUp(reservation.getMaxASBuildScratchSize(false), MaxAlignment);


				IGPUBuffer::SCreationParams creationParams = {};
				creationParams.size = scratchSize;
				creationParams.usage = IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
				auto scratchBuffer = m_device->createBuffer(std::move(creationParams));

				auto reqs = scratchBuffer->getMemoryReqs();
				reqs.memoryTypeBits &= m_physicalDevice->getDirectVRAMAccessMemoryTypeBits();

				auto allocation = m_device->allocate(reqs, scratchBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
				allocation.memory->map({ .offset = 0,.length = reqs.size });

				scratchAlloc = make_smart_refctd_ptr<CAssetConverter::SConvertParams::scratch_for_device_AS_build_t>(
					SBufferRange<video::IGPUBuffer>{0ull, scratchSize, std::move(scratchBuffer)},
					core::allocator<uint8_t>(), MaxAlignment, MinAllocationSize
				);
			}

			struct MyParams final : CAssetConverter::SConvertParams
			{
				inline uint32_t getFinalOwnerQueueFamily(const IGPUBuffer* buffer, const core::blake3_hash_t& createdFrom) override
				{
					return finalUser;
				}
				inline uint32_t getFinalOwnerQueueFamily(const IGPUAccelerationStructure* image, const core::blake3_hash_t& createdFrom) override
				{
					return finalUser;
				}

				uint8_t finalUser;
			} params = {};
			params.utilities = m_utils.get();
			params.compute = &compute;
			params.scratchForDeviceASBuild = scratchAlloc.get();
			params.finalUser = queue->getFamilyIndex();

			auto future = reservation.convert(params);
			if (future.copy() != IQueue::RESULT::SUCCESS)
			{
				m_logger->log("Failed to await submission feature!", ILogger::ELL_ERROR);
				return false;
			}
			// 2 submits, BLAS build, TLAS build, DO NOT ADD COMPACTIONS IN THIS EXAMPLE!
			if (compute.getFutureScratchSemaphore().value>3)
				m_logger->log("Overflow submitted on Compute Queue despite using ReBAR (no transfer submits or usage of staging buffer) and providing a AS Build Scratch Buffer of correctly queried max size!",system::ILogger::ELL_ERROR);

			// assign gpu objects to output
			auto&& tlases = reservation.getGPUObjects<ICPUTopLevelAccelerationStructure>();
			m_gpuTlas = tlases[0].value;
			auto&& buffers = reservation.getGPUObjects<ICPUBuffer>();
			m_proceduralAabbBuffer = buffers[0].value;

			auto&& gpuPolygonGeometries = reservation.getGPUObjects<ICPUPolygonGeometry>();
			m_gpuPolygons.resize(gpuPolygonGeometries.size());

			for (uint32_t i = 0; i < gpuPolygonGeometries.size(); i++)
			{
				const auto& cpuObject = cpuObjects[i];
				const auto& gpuPolygon = gpuPolygonGeometries[i].value;
				const auto gpuTriangles = gpuPolygon->exportForBLAS();

				const auto& vertexBufferBinding = gpuTriangles.vertexData[0];
				const uint64_t vertexBufferAddress = vertexBufferBinding.buffer->getDeviceAddress() + vertexBufferBinding.offset;

				const auto& normalView = gpuPolygon->getNormalView();
				const uint64_t normalBufferAddress = normalView ? normalView.src.buffer->getDeviceAddress() + normalView.src.offset : 0;
        auto normalType = NT_R32G32B32_SFLOAT;
        if (normalView && normalView.composed.format == EF_R8G8B8A8_SNORM)
          normalType = NT_R8G8B8A8_SNORM;

				const auto& indexBufferBinding = gpuTriangles.indexData;
				auto& geomInfo = geomInfos[i];
				geomInfo = {
				  .material = hlsl::_static_cast<MaterialPacked>(cpuObject.material),
				  .vertexBufferAddress = vertexBufferAddress,
				  .indexBufferAddress = indexBufferBinding.buffer ? indexBufferBinding.buffer->getDeviceAddress() + indexBufferBinding.offset : vertexBufferAddress,
					.normalBufferAddress = normalBufferAddress,
					.normalType = normalType,
				  .indexType = gpuTriangles.indexType,
				};

				m_gpuPolygons[i] = gpuPolygon;
			}
		}

		{
			IGPUBuffer::SCreationParams params;
			params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
			params.size = geomInfoBuffer->getSize();
			m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = queue }, std::move(params), geomInfos).move_into(m_triangleGeomInfoBuffer);
		}

		return true;
	}

	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>> m_surface;
	smart_refctd_ptr<ISemaphore> m_semaphore;
	uint64_t m_realFrameIx = 0;
	uint32_t m_frameAccumulationCounter = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	core::smart_refctd_ptr<InputSystem> m_inputSystem;
	InputSystem::ChannelReader<IMouseEventChannel> m_mouse;
	InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;

	struct CameraSetting
	{
		float fov = 60.f;
		float zNear = 0.1f;
		float zFar = 10000.f;
		float moveSpeed = 1.f;
		float rotateSpeed = 1.f;
		float viewWidth = 10.f;
		float camYAngle = 165.f / 180.f * 3.14159f;
		float camXAngle = 32.f / 180.f * 3.14159f;

	} m_cameraSetting;
	Camera m_camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());

	Light m_light = {
	  .direction = {-1.0f, -1.0f, -0.4f},
	  .position = {10.0f, 15.0f, 8.0f},
	  .outerCutoff = 0.866025404f, // {cos(radians(30.0f))}, 
	  .type = ELT_DIRECTIONAL
	};

	video::CDumbPresentationOracle m_oracle;

	struct C_UI
	{
		nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> manager;

		struct
		{
			core::smart_refctd_ptr<video::IGPUSampler> gui, scene;
		} samplers;

		core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
	} m_ui;
	core::smart_refctd_ptr<IDescriptorPool> m_guiDescriptorSetPool;

	core::vector<SProceduralGeomInfo> m_gpuIntersectionSpheres;
	uint32_t m_intersectionHitGroupIdx;

	core::vector<smart_refctd_ptr<IGPUPolygonGeometry>> m_gpuPolygons;
	smart_refctd_ptr<IGPUTopLevelAccelerationStructure> m_gpuTlas;
	smart_refctd_ptr<IGPUBuffer> m_instanceBuffer;

	smart_refctd_ptr<IGPUBuffer> m_triangleGeomInfoBuffer;
	smart_refctd_ptr<IGPUBuffer> m_proceduralGeomInfoBuffer;
	smart_refctd_ptr<IGPUBuffer> m_proceduralAabbBuffer;
	smart_refctd_ptr<IGPUBuffer> m_indirectBuffer;

	smart_refctd_ptr<IGPUImage> m_hdrImage;
	smart_refctd_ptr<IGPUImageView> m_hdrImageView;

	smart_refctd_ptr<IDescriptorPool> m_rayTracingDsPool;
	smart_refctd_ptr<IGPUDescriptorSet> m_rayTracingDs;
	smart_refctd_ptr<IGPURayTracingPipeline> m_rayTracingPipeline;
	uint64_t m_rayTracingStackSize;
	ShaderBindingTable m_shaderBindingTable;

	smart_refctd_ptr<IGPUDescriptorSet> m_presentDs;
	smart_refctd_ptr<IDescriptorPool> m_presentDsPool;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_presentPipeline;

	smart_refctd_ptr<CAssetConverter> m_converter;


	core::matrix4SIMD m_cachedModelViewProjectionMatrix;
	bool m_useIndirectCommand = false;

};
NBL_MAIN_FUNC(RaytracingPipelineApp)