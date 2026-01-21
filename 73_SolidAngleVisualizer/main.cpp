// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "common.hpp"
#include "app_resources/hlsl/common.hlsl"
#include "app_resources/hlsl/benchmark/common.hlsl"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

/*
Renders scene texture to an offscreen framebuffer whose color attachment is then sampled into a imgui window.

Written with Nabla's UI extension and got integrated with ImGuizmo to handle scene's object translations.
*/
class SolidAngleVisualizer final : public MonoWindowApplication, public BuiltinResourcesApplication
{
	using device_base_t = MonoWindowApplication;
	using asset_base_t = BuiltinResourcesApplication;

	inline static std::string SolidAngleVisShaderPath = "app_resources/hlsl/SolidAngleVis.frag.hlsl";
	inline static std::string RayVisShaderPath = "app_resources/hlsl/RayVis.frag.hlsl";

public:
	inline SolidAngleVisualizer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
		device_base_t({ 2048, 1024 }, EF_UNKNOWN, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD)
	{
	}

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;

		interface.m_visualizer = this;

		m_semaphore = m_device->createSemaphore(m_realFrameIx);
		if (!m_semaphore)
			return logFail("Failed to Create a Semaphore!");

		auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		for (auto i = 0u; i < MaxFramesInFlight; i++)
		{
			if (!pool)
				return logFail("Couldn't create Command Pool!");
			if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
				return logFail("Couldn't create Command Buffer!");
		}

		const uint32_t addtionalBufferOwnershipFamilies[] = { getGraphicsQueue()->getFamilyIndex() };
		m_scene = CGeometryCreatorScene::create(
			{ .transferQueue = getTransferUpQueue(),
			 .utilities = m_utils.get(),
			 .logger = m_logger.get(),
			 .addtionalBufferOwnershipFamilies = addtionalBufferOwnershipFamilies },
			CSimpleDebugRenderer::DefaultPolygonGeometryPatch);

		// for the scene drawing pass
		{
			IGPURenderpass::SCreationParams params = {};
			const IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] = {
				{{{.format = sceneRenderDepthFormat,
				   .samples = IGPUImage::ESCF_1_BIT,
				   .mayAlias = false},
				   /*.loadOp =*/{IGPURenderpass::LOAD_OP::CLEAR},
				   /*.storeOp =*/{IGPURenderpass::STORE_OP::STORE},
				   /*.initialLayout =*/{IGPUImage::LAYOUT::UNDEFINED},
				   /*.finalLayout =*/{IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}}},
				 IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd };
			params.depthStencilAttachments = depthAttachments;
			const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
				{{
					{.format = finalSceneRenderFormat,
					 .samples = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
					 .mayAlias = false},
					 /*.loadOp =*/IGPURenderpass::LOAD_OP::CLEAR,
					 /*.storeOp =*/IGPURenderpass::STORE_OP::STORE,
					 /*.initialLayout =*/IGPUImage::LAYOUT::UNDEFINED,
					 /*.finalLayout =*/IGPUImage::LAYOUT::READ_ONLY_OPTIMAL // ImGUI shall read
				 }},
				 IGPURenderpass::SCreationParams::ColorAttachmentsEnd };
			params.colorAttachments = colorAttachments;
			IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
				{},
				IGPURenderpass::SCreationParams::SubpassesEnd };
			subpasses[0].depthStencilAttachment = { {.render = {.attachmentIndex = 0, .layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}} };
			subpasses[0].colorAttachments[0] = { .render = {.attachmentIndex = 0, .layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL} };
			params.subpasses = subpasses;

			const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
				// wipe-transition of Color to ATTACHMENT_OPTIMAL and depth
				{
					.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.dstSubpass = 0,
					.memoryBarrier = {
					// last place where the depth can get modified in previous frame, `COLOR_ATTACHMENT_OUTPUT_BIT` is implicitly later
					// while color is sampled by ImGUI
					.srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
					// don't want any writes to be available, as we are clearing both attachments
					.srcAccessMask = ACCESS_FLAGS::NONE,
					// destination needs to wait as early as possible
					// TODO: `COLOR_ATTACHMENT_OUTPUT_BIT` shouldn't be needed, because its a logically later stage, see TODO in `ECommonEnums.h`
					.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					// because depth and color get cleared first no read mask
					.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT}
					// leave view offsets and flags default
				},
				{
					.srcSubpass = 0, .dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External, .memoryBarrier = {// last place where the color can get modified, depth is implicitly earlier
																																	.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
																																	// only write ops, reads can't be made available, also won't be using depth so don't care about it being visible to anyone else
																																	.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
																																	// the ImGUI will sample the color, then next frame we overwrite both attachments
																																	.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT | PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
																																	// but we only care about the availability-visibility chain between renderpass and imgui
																																	.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT}
																																	// leave view offsets and flags default
																																},
																																IGPURenderpass::SCreationParams::DependenciesEnd };
			params.dependencies = dependencies;
			auto solidAngleRenderpassParams = params;
			m_mainRenderpass = m_device->createRenderpass(std::move(params));
			if (!m_mainRenderpass)
				return logFail("Failed to create Main Renderpass!");

			m_solidAngleRenderpass = m_device->createRenderpass(std::move(solidAngleRenderpassParams));
			if (!m_solidAngleRenderpass)
				return logFail("Failed to create Solid Angle Renderpass!");
		}

		const auto& geometries = m_scene->getInitParams().geometries;
		m_renderer = CSimpleDebugRenderer::create(m_assetMgr.get(), m_solidAngleRenderpass.get(), 0, { &geometries.front().get(), geometries.size() });
		// special case
		{
			const auto& pipelines = m_renderer->getInitParams().pipelines;
			auto ix = 0u;
			for (const auto& name : m_scene->getInitParams().geometryNames)
			{
				if (name == "Cone")
					m_renderer->getGeometry(ix).pipeline = pipelines[CSimpleDebugRenderer::SInitParams::PipelineType::Cone];
				ix++;
			}
		}
		// we'll only display one thing at a time
		m_renderer->m_instances.resize(1);

		// Create graphics pipeline
		{
			auto loadAndCompileHLSLShader = [&](const std::string& pathToShader, IShader::E_SHADER_STAGE stage, const std::string& defineMacro = "") -> smart_refctd_ptr<IShader>
				{
					IAssetLoader::SAssetLoadParams lp = {};
					lp.workingDirectory = localInputCWD;
					auto assetBundle = m_assetMgr->getAsset(pathToShader, lp);
					const auto assets = assetBundle.getContents();
					if (assets.empty())
					{
						m_logger->log("Could not load shader: ", ILogger::ELL_ERROR, pathToShader);
						std::exit(-1);
					}

					auto source = smart_refctd_ptr_static_cast<IShader>(assets[0]);
					// The down-cast should not fail!
					assert(source);

					auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(m_system));
					CHLSLCompiler::SOptions options = {};
					options.stage = stage;
					options.preprocessorOptions.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
					options.spirvOptimizer = nullptr;
#ifndef _NBL_DEBUG
					ISPIRVOptimizer::E_OPTIMIZER_PASS optPasses = ISPIRVOptimizer::EOP_STRIP_DEBUG_INFO;
					auto opt = make_smart_refctd_ptr<ISPIRVOptimizer>(std::span<ISPIRVOptimizer::E_OPTIMIZER_PASS>(&optPasses, 1));
					options.spirvOptimizer = opt.get();
#endif
					options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;// | IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_FILE_BIT | IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
					options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
					options.preprocessorOptions.logger = m_logger.get();
					options.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder();

					core::vector<IShaderCompiler::SMacroDefinition> defines;
					if (!defineMacro.empty())
						defines.push_back({ defineMacro, "" });

					options.preprocessorOptions.extraDefines = defines;

					source = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);

					auto shader = m_device->compileShader({ source.get(), nullptr, nullptr, nullptr });
					if (!shader)
					{
						m_logger->log("HLSL shader creationed failed: %s!", ILogger::ELL_ERROR, pathToShader);
						std::exit(-1);
					}

					return shader;
				};

			ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
			if (!fsTriProtoPPln)
				return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

			// Load Fragment Shader
			auto solidAngleVisFragShader = loadAndCompileHLSLShader(SolidAngleVisShaderPath, ESS_FRAGMENT);
			if (!solidAngleVisFragShader)
				return logFail("Failed to Load and Compile Fragment Shader: SolidAngleVis!");

			const IGPUPipelineBase::SShaderSpecInfo solidAngleFragSpec = {
				.shader = solidAngleVisFragShader.get(),
				.entryPoint = "main" };

			auto rayVisFragShader = loadAndCompileHLSLShader(RayVisShaderPath, ESS_FRAGMENT);
			if (!rayVisFragShader)
				return logFail("Failed to Load and Compile Fragment Shader: rayVis!");
			const IGPUPipelineBase::SShaderSpecInfo RayFragSpec = {
				.shader = rayVisFragShader.get(),
				.entryPoint = "main" };

			smart_refctd_ptr<IGPUPipelineLayout> solidAngleVisLayout, rayVisLayout;
			nbl::video::IGPUDescriptorSetLayout::SBinding bindings[1] = {
				{.binding = 0,
				 .type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
				 .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				 .stageFlags = ShaderStage::ESS_FRAGMENT,
				 .count = 1} };
			smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = m_device->createDescriptorSetLayout(bindings);

			const asset::SPushConstantRange saRanges[] = { {.stageFlags = hlsl::ShaderStage::ESS_FRAGMENT,
														   .offset = 0,
														   .size = sizeof(PushConstants)} };
			const asset::SPushConstantRange rayRanges[] = { {.stageFlags = hlsl::ShaderStage::ESS_FRAGMENT,
															.offset = 0,
															.size = sizeof(PushConstantRayVis)} };

			if (!dsLayout)
				logFail("Failed to create a Descriptor Layout!\n");

			solidAngleVisLayout = m_device->createPipelineLayout(saRanges, dsLayout);

			rayVisLayout = m_device->createPipelineLayout(rayRanges, dsLayout);

			{
				m_solidAngleVisPipeline = fsTriProtoPPln.createPipeline(solidAngleFragSpec, solidAngleVisLayout.get(), m_solidAngleRenderpass.get());
				if (!m_solidAngleVisPipeline)
					return logFail("Could not create Graphics Pipeline!");

				asset::SRasterizationParams rasterParams = ext::FullScreenTriangle::ProtoPipeline::DefaultRasterParams;
				rasterParams.depthWriteEnable = true;
				rasterParams.depthCompareOp = asset::E_COMPARE_OP::ECO_GREATER;

				m_rayVisualizationPipeline = fsTriProtoPPln.createPipeline(RayFragSpec, rayVisLayout.get(), m_mainRenderpass.get(), 0, {}, rasterParams);
				if (!m_rayVisualizationPipeline)
					return logFail("Could not create Graphics Pipeline!");
			}
			// Allocate the memory
			{
				constexpr size_t BufferSize = sizeof(ResultData);

				nbl::video::IGPUBuffer::SCreationParams params = {};
				params.size = BufferSize;
				params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
				m_outputStorageBuffer = m_device->createBuffer(std::move(params));
				if (!m_outputStorageBuffer)
					logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

				m_outputStorageBuffer->setObjectDebugName("ResultData output buffer");

				nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = m_outputStorageBuffer->getMemoryReqs();
				reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

				m_allocation = m_device->allocate(reqs, m_outputStorageBuffer.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
				if (!m_allocation.isValid())
					logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

				assert(m_outputStorageBuffer->getBoundMemory().memory == m_allocation.memory.get());
				smart_refctd_ptr<nbl::video::IDescriptorPool> pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &dsLayout.get(), 1 });

				m_ds = pool->createDescriptorSet(std::move(dsLayout));
				{
					IGPUDescriptorSet::SDescriptorInfo info[1];
					info[0].desc = smart_refctd_ptr(m_outputStorageBuffer);
					info[0].info.buffer = { .offset = 0, .size = BufferSize };
					IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
						{.dstSet = m_ds.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = info} };
					m_device->updateDescriptorSets(writes, {});
				}
			}

			if (!m_allocation.memory->map({ 0ull, m_allocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
				logFail("Failed to map the Device Memory!\n");

			// if the mapping is not coherent the range needs to be invalidated to pull in new data for the CPU's caches
			const ILogicalDevice::MappedMemoryRange memoryRange(m_allocation.memory.get(), 0ull, m_allocation.memory->getAllocationSize());
			if (!m_allocation.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
				m_device->invalidateMappedMemoryRanges(1, &memoryRange);
		}

		// Create ImGUI
		{
			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			ext::imgui::UI::SCreationParameters params = {};
			params.resources.texturesInfo = { .setIx = 0u, .bindingIx = TexturesImGUIBindingIndex };
			params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
			params.utilities = m_utils;
			params.transfer = getTransferUpQueue();
			params.pipelineLayout = ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, MaxImGUITextures);
			params.assetManager = make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(m_system));
			params.renderpass = smart_refctd_ptr<IGPURenderpass>(scRes->getRenderpass());
			params.subpassIx = 0u;
			params.pipelineCache = nullptr;
			interface.imGUI = ext::imgui::UI::create(std::move(params));
			if (!interface.imGUI)
				return logFail("Failed to create `nbl::ext::imgui::UI` class");
		}

		// create rest of User Interface
		{
			auto* imgui = interface.imGUI.get();
			// create the suballocated descriptor set
			{
				// note that we use default layout provided by our extension, but you are free to create your own by filling ext::imgui::UI::S_CREATION_PARAMETERS::resources
				const auto* layout = interface.imGUI->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
				auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT, { &layout, 1 });
				auto ds = pool->createDescriptorSet(smart_refctd_ptr<const IGPUDescriptorSetLayout>(layout));
				interface.subAllocDS = make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(ds));
				if (!interface.subAllocDS)
					return logFail("Failed to create the descriptor set");
				// make sure Texture Atlas slot is taken for eternity
				{
					auto dummy = SubAllocatedDescriptorSet::invalid_value;
					interface.subAllocDS->multi_allocate(0, 1, &dummy);
					assert(dummy == ext::imgui::UI::FontAtlasTexId);
				}
				// write constant descriptors, note we don't create info & write pair for the samplers because UI extension's are immutable and baked into DS layout
				IGPUDescriptorSet::SDescriptorInfo info = {};
				info.desc = smart_refctd_ptr<nbl::video::IGPUImageView>(interface.imGUI->getFontAtlasView());
				info.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
				const IGPUDescriptorSet::SWriteDescriptorSet write = {
					.dstSet = interface.subAllocDS->getDescriptorSet(),
					.binding = TexturesImGUIBindingIndex,
					.arrayElement = ext::imgui::UI::FontAtlasTexId,
					.count = 1,
					.info = &info };
				if (!m_device->updateDescriptorSets({ &write, 1 }, {}))
					return logFail("Failed to write the descriptor set");
			}
			imgui->registerListener([this]()
				{ interface(); });
		}

		interface.camera.mapKeysToWASD();

		onAppInitializedFinish();
		return true;
	}

	//
	virtual inline bool onAppTerminated()
	{
		SubAllocatedDescriptorSet::value_type fontAtlasDescIx = ext::imgui::UI::FontAtlasTexId;
		IGPUDescriptorSet::SDropDescriptorSet dummy[1];
		interface.subAllocDS->multi_deallocate(dummy, TexturesImGUIBindingIndex, 1, &fontAtlasDescIx);
		return device_base_t::onAppTerminated();
	}

	inline IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override
	{
		// CPU events
		update(nextPresentationTimestamp);

		{
			const auto& virtualSolidAngleWindowRes = interface.solidAngleViewTransformReturnInfo.sceneResolution;
			const auto& virtualMainWindowRes = interface.mainViewTransformReturnInfo.sceneResolution;
			if (!m_solidAngleViewFramebuffer || m_solidAngleViewFramebuffer->getCreationParameters().width != virtualSolidAngleWindowRes[0] || m_solidAngleViewFramebuffer->getCreationParameters().height != virtualSolidAngleWindowRes[1] ||
				!m_mainViewFramebuffer || m_mainViewFramebuffer->getCreationParameters().width != virtualMainWindowRes[0] || m_mainViewFramebuffer->getCreationParameters().height != virtualMainWindowRes[1])
				recreateFramebuffers();
		}

		//
		const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

		auto* const cb = m_cmdBufs.data()[resourceIx].get();
		cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

		if (m_solidAngleViewFramebuffer)
		{
			asset::SBufferRange<IGPUBuffer> range{
				.offset = 0,
				.size = m_outputStorageBuffer->getSize(),
				.buffer = m_outputStorageBuffer };
			cb->fillBuffer(range, 0u);
			{

				const auto& creationParams = m_solidAngleViewFramebuffer->getCreationParameters();
				cb->beginDebugMarker("Draw Circle View Frame");
				{
					const IGPUCommandBuffer::SClearDepthStencilValue farValue = { .depth = 0.f };
					const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f, 0.f, 0.f, 1.f} };
					const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo =
					{
						.framebuffer = m_solidAngleViewFramebuffer.get(),
						.colorClearValues = &clearValue,
						.depthStencilClearValues = &farValue,
						.renderArea = {
							.offset = {0, 0},
							.extent = {creationParams.width, creationParams.height}} };
					beginRenderpass(cb, renderpassInfo);
				}
				// draw scene
				{
					static uint32_t lastFrameSeed = 0u;
					lastFrameSeed = m_frameSeeding ? static_cast<uint32_t>(m_realFrameIx) : lastFrameSeed;
					PushConstants pc{
						.modelMatrix = hlsl::float32_t3x4(hlsl::transpose(interface.m_OBBModelMatrix)),
						.viewport = {0.f, 0.f, static_cast<float>(creationParams.width), static_cast<float>(creationParams.height)},
						.samplingMode = m_samplingMode,
						.sampleCount = static_cast<uint32_t>(m_SampleCount),
						.frameIndex = lastFrameSeed };
					auto pipeline = m_solidAngleVisPipeline;
					cb->bindGraphicsPipeline(pipeline.get());
					cb->pushConstants(pipeline->getLayout(), hlsl::ShaderStage::ESS_FRAGMENT, 0, sizeof(pc), &pc);
					cb->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS, pipeline->getLayout(), 0, 1, &m_ds.get());
					ext::FullScreenTriangle::recordDrawCall(cb);
				}
				cb->endRenderPass();
				cb->endDebugMarker();
			}
#if DEBUG_DATA
			m_device->waitIdle();
			std::memcpy(&m_GPUOutResulData, static_cast<ResultData*>(m_allocation.memory->getMappedPointer()), sizeof(ResultData));
			m_device->waitIdle();
#endif
		}
		// draw main view
		if (m_mainViewFramebuffer)
		{
			{
				auto creationParams = m_mainViewFramebuffer->getCreationParameters();
				const IGPUCommandBuffer::SClearDepthStencilValue farValue = { .depth = 0.f };
				const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.1f, 0.1f, 0.1f, 1.f} };
				const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo =
				{
					.framebuffer = m_mainViewFramebuffer.get(),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = &farValue,
					.renderArea = {
						.offset = {0, 0},
						.extent = {creationParams.width, creationParams.height}} };
				beginRenderpass(cb, renderpassInfo);
			}
			{ // draw rays visualization
				auto creationParams = m_mainViewFramebuffer->getCreationParameters();

				cb->beginDebugMarker("Draw Rays visualization");
				// draw scene
				{
					float32_t4x4 viewProj = *reinterpret_cast<const float32_t4x4*>(&interface.camera.getConcatenatedMatrix());
					float32_t3x4 view = *reinterpret_cast<const float32_t3x4*>(&interface.camera.getViewMatrix());
					PushConstantRayVis pc{
						.viewProjMatrix = viewProj,
						.viewMatrix = view,
						.modelMatrix = hlsl::float32_t3x4(hlsl::transpose(interface.m_OBBModelMatrix)),
						.viewport = {0.f, 0.f, static_cast<float>(creationParams.width), static_cast<float>(creationParams.height)},
						.frameIndex = m_frameSeeding ? static_cast<uint32_t>(m_realFrameIx) : 0u };
					auto pipeline = m_rayVisualizationPipeline;
					cb->bindGraphicsPipeline(pipeline.get());
					cb->pushConstants(pipeline->getLayout(), hlsl::ShaderStage::ESS_FRAGMENT, 0, sizeof(pc), &pc);
					cb->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS, pipeline->getLayout(), 0, 1, &m_ds.get());
					ext::FullScreenTriangle::recordDrawCall(cb);
				}
				cb->endDebugMarker();
			}
			// draw scene
			{
				cb->beginDebugMarker("Main Scene Frame");

				float32_t3x4 viewMatrix;
				float32_t4x4 viewProjMatrix;
				// TODO: get rid of legacy matrices
				{
					const auto& camera = interface.camera;
					memcpy(&viewMatrix, camera.getViewMatrix().pointer(), sizeof(viewMatrix));
					memcpy(&viewProjMatrix, camera.getConcatenatedMatrix().pointer(), sizeof(viewProjMatrix));
				}
				const auto viewParams = CSimpleDebugRenderer::SViewParams(viewMatrix, viewProjMatrix);

				// tear down scene every frame
				auto& instance = m_renderer->m_instances[0];
				instance.world = float32_t3x4(hlsl::transpose(interface.m_OBBModelMatrix));
				instance.packedGeo = m_renderer->getGeometries().data(); // cube // +interface.gcIndex;
				m_renderer->render(cb, viewParams);						 // draw the cube/OBB

				instance.world = float32_t3x4(1.0f);
				instance.packedGeo = m_renderer->getGeometries().data() + 2; // disk
				m_renderer->render(cb, viewParams);
			}

			cb->endDebugMarker();
			cb->endRenderPass();
		}

		{
			cb->beginDebugMarker("SolidAngleVisualizer IMGUI Frame");
			{
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f, 0.f, 0.f, 1.f} };
				const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo =
				{
					.framebuffer = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = nullptr,
					.renderArea = {
						.offset = {0, 0},
						.extent = {m_window->getWidth(), m_window->getHeight()}} };
				beginRenderpass(cb, renderpassInfo);
			}
			// draw ImGUI
			{
				auto* imgui = interface.imGUI.get();
				auto* pipeline = imgui->getPipeline();
				cb->bindGraphicsPipeline(pipeline);
				// note that we use default UI pipeline layout where uiParams.resources.textures.setIx == uiParams.resources.samplers.setIx
				const auto* ds = interface.subAllocDS->getDescriptorSet();
				cb->bindDescriptorSets(EPBP_GRAPHICS, pipeline->getLayout(), imgui->getCreationParameters().resources.texturesInfo.setIx, 1u, &ds);
				// a timepoint in the future to release streaming resources for geometry
				const ISemaphore::SWaitInfo drawFinished = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };
				if (!imgui->render(cb, drawFinished))
				{
					m_logger->log("TODO: need to present acquired image before bailing because its already acquired.", ILogger::ELL_ERROR);
					return {};
				}
			}
			cb->endRenderPass();
			cb->endDebugMarker();
		}
		cb->end();

		IQueue::SSubmitInfo::SSemaphoreInfo retval =
		{
			.semaphore = m_semaphore.get(),
			.value = ++m_realFrameIx,
			.stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS };
		const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
		{
			{.cmdbuf = cb} };
		const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = {
			{.semaphore = device_base_t::getCurrentAcquire().semaphore,
			 .value = device_base_t::getCurrentAcquire().acquireCount,
			 .stageMask = PIPELINE_STAGE_FLAGS::NONE} };
		const IQueue::SSubmitInfo infos[] =
		{
			{.waitSemaphores = acquired,
			 .commandBuffers = commandBuffers,
			 .signalSemaphores = {&retval, 1}} };

		if (getGraphicsQueue()->submit(infos) != IQueue::RESULT::SUCCESS)
		{
			retval.semaphore = nullptr; // so that we don't wait on semaphore that will never signal
			m_realFrameIx--;
		}

		m_window->setCaption("[Nabla Engine] UI App Test Demo");
		return retval;
	}

protected:
	const video::IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const override
	{
		// Subsequent submits don't wait for each other, but they wait for acquire and get waited on by present
		const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
			// don't want any writes to be available, we'll clear, only thing to worry about is the layout transition
			{
				.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.dstSubpass = 0,
				.memoryBarrier = {
					.srcStageMask = PIPELINE_STAGE_FLAGS::NONE, // should sync against the semaphore wait anyway
					.srcAccessMask = ACCESS_FLAGS::NONE,
					// layout transition needs to finish before the color write
					.dstStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					.dstAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT}
					// leave view offsets and flags default
				},
			// want layout transition to begin after all color output is done
			{
				.srcSubpass = 0, .dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External, .memoryBarrier = {
				// last place where the color can get modified, depth is implicitly earlier
				.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
				// only write ops, reads can't be made available
				.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
				// spec says nothing is needed when presentation is the destination
			}
			// leave view offsets and flags default
		},
		IGPURenderpass::SCreationParams::DependenciesEnd };
		return dependencies;
	}

private:
	inline void update(const std::chrono::microseconds nextPresentationTimestamp)
	{
		auto& camera = interface.camera;
		camera.setMoveSpeed(interface.moveSpeed);
		camera.setRotateSpeed(interface.rotateSpeed);

		m_inputSystem->getDefaultMouse(&mouse);
		m_inputSystem->getDefaultKeyboard(&keyboard);

		struct
		{
			std::vector<SMouseEvent> mouse{};
			std::vector<SKeyboardEvent> keyboard{};
		} uiEvents;

		// TODO: should be a member really
		static std::chrono::microseconds previousEventTimestamp{};

		// I think begin/end should always be called on camera, just events shouldn't be fed, why?
		// If you stop begin/end, whatever keys were up/down get their up/down values frozen leading to
		// `perActionDt` becoming obnoxiously large the first time the even processing resumes due to
		// `timeDiff` being computed since `lastVirtualUpTimeStamp`
		camera.beginInputProcessing(nextPresentationTimestamp);
		{
			mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
				{
					if (interface.move)
						camera.mouseProcess(events); // don't capture the events, only let camera handle them with its impl
					else
						camera.mouseKeysUp();

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						uiEvents.mouse.emplace_back(e);

						//if (e.type == nbl::ui::SMouseEvent::EET_SCROLL && m_renderer)
						//{
						//	interface.gcIndex += int16_t(core::sign(e.scrollEvent.verticalScroll));
						//	interface.gcIndex = core::clamp(interface.gcIndex, 0ull, m_renderer->getGeometries().size() - 1);
						//}
					} },
				m_logger.get());
			keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
				{
					if (interface.move)
						camera.keyboardProcess(events); // don't capture the events, only let camera handle them with its impl

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						uiEvents.keyboard.emplace_back(e);
					} },
				m_logger.get());
		}
		camera.endInputProcessing(nextPresentationTimestamp);

		const auto cursorPosition = m_window->getCursorControl()->getPosition();

		ext::imgui::UI::SUpdateParameters params =
		{
			.mousePosition = float32_t2(cursorPosition.x, cursorPosition.y) - float32_t2(m_window->getX(), m_window->getY()),
			.displaySize = {m_window->getWidth(), m_window->getHeight()},
			.mouseEvents = uiEvents.mouse,
			.keyboardEvents = uiEvents.keyboard };

		// interface.objectName = m_scene->getInitParams().geometryNames[interface.gcIndex];
		interface.imGUI->update(params);
	}

	void recreateFramebuffers()
	{

		auto createImageAndView = [&](const uint16_t2 resolution, E_FORMAT format) -> smart_refctd_ptr<IGPUImageView>
			{
				auto image = m_device->createImage({ {.type = IGPUImage::ET_2D,
													 .samples = IGPUImage::ESCF_1_BIT,
													 .format = format,
													 .extent = {resolution.x, resolution.y, 1},
													 .mipLevels = 1,
													 .arrayLayers = 1,
													 .usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_SAMPLED_BIT} });
				if (!m_device->allocate(image->getMemoryReqs(), image.get()).isValid())
					return nullptr;
				IGPUImageView::SCreationParams params = {
					.image = std::move(image),
					.viewType = IGPUImageView::ET_2D,
					.format = format };
				params.subresourceRange.aspectMask = isDepthOrStencilFormat(format) ? IGPUImage::EAF_DEPTH_BIT : IGPUImage::EAF_COLOR_BIT;
				return m_device->createImageView(std::move(params));
			};

		smart_refctd_ptr<IGPUImageView> solidAngleView;
		smart_refctd_ptr<IGPUImageView> mainView;
		const uint16_t2 solidAngleViewRes = interface.solidAngleViewTransformReturnInfo.sceneResolution;
		const uint16_t2 mainViewRes = interface.mainViewTransformReturnInfo.sceneResolution;

		// detect window minimization
		if (solidAngleViewRes.x < 0x4000 && solidAngleViewRes.y < 0x4000 ||
			mainViewRes.x < 0x4000 && mainViewRes.y < 0x4000)
		{
			solidAngleView = createImageAndView(solidAngleViewRes, finalSceneRenderFormat);
			auto solidAngleDepthView = createImageAndView(solidAngleViewRes, sceneRenderDepthFormat);
			m_solidAngleViewFramebuffer = m_device->createFramebuffer({ {.renderpass = m_solidAngleRenderpass,
																		.depthStencilAttachments = &solidAngleDepthView.get(),
																		.colorAttachments = &solidAngleView.get(),
																		.width = solidAngleViewRes.x,
																		.height = solidAngleViewRes.y} });

			mainView = createImageAndView(mainViewRes, finalSceneRenderFormat);
			auto mainDepthView = createImageAndView(mainViewRes, sceneRenderDepthFormat);
			m_mainViewFramebuffer = m_device->createFramebuffer({ {.renderpass = m_mainRenderpass,
																  .depthStencilAttachments = &mainDepthView.get(),
																  .colorAttachments = &mainView.get(),
																  .width = mainViewRes.x,
																  .height = mainViewRes.y} });
		}
		else
		{
			m_solidAngleViewFramebuffer = nullptr;
			m_mainViewFramebuffer = nullptr;
		}

		// release previous slot and its image
		interface.subAllocDS->multi_deallocate(0, static_cast<int>(CInterface::Count), interface.renderColorViewDescIndices, { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1 });
		//
		if (solidAngleView && mainView)
		{
			interface.subAllocDS->multi_allocate(0, static_cast<int>(CInterface::Count), interface.renderColorViewDescIndices);
			// update descriptor set
			IGPUDescriptorSet::SDescriptorInfo infos[static_cast<int>(CInterface::Count)] = {};
			infos[0].desc = mainView;
			infos[0].info.image.imageLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
			infos[1].desc = solidAngleView;
			infos[1].info.image.imageLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
			const IGPUDescriptorSet::SWriteDescriptorSet write[static_cast<int>(CInterface::Count)] = {
				{.dstSet = interface.subAllocDS->getDescriptorSet(),
				 .binding = TexturesImGUIBindingIndex,
				 .arrayElement = interface.renderColorViewDescIndices[static_cast<int>(CInterface::ERV_MAIN_VIEW)],
				 .count = 1,
				 .info = &infos[static_cast<int>(CInterface::ERV_MAIN_VIEW)]},
				{.dstSet = interface.subAllocDS->getDescriptorSet(),
				 .binding = TexturesImGUIBindingIndex,
				 .arrayElement = interface.renderColorViewDescIndices[static_cast<int>(CInterface::ERV_SOLID_ANGLE_VIEW)],
				 .count = 1,
				 .info = &infos[static_cast<int>(CInterface::ERV_SOLID_ANGLE_VIEW)]} };
			m_device->updateDescriptorSets({ write, static_cast<int>(CInterface::Count) }, {});
		}
		interface.transformParams.sceneTexDescIx = interface.renderColorViewDescIndices[CInterface::ERV_MAIN_VIEW];
	}

	inline void beginRenderpass(IGPUCommandBuffer* cb, const IGPUCommandBuffer::SRenderpassBeginInfo& info)
	{
		cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
		cb->setScissor(0, 1, &info.renderArea);
		const SViewport viewport = {
			.x = 0,
			.y = 0,
			.width = static_cast<float>(info.renderArea.extent.width),
			.height = static_cast<float>(info.renderArea.extent.height) };
		cb->setViewport(0u, 1u, &viewport);
	}

	~SolidAngleVisualizer() override
	{
		m_allocation.memory->unmap();
	}

	// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
	constexpr static inline uint32_t MaxFramesInFlight = 3u;
	constexpr static inline auto sceneRenderDepthFormat = EF_D32_SFLOAT;
	constexpr static inline auto finalSceneRenderFormat = EF_R8G8B8A8_SRGB;
	constexpr static inline auto TexturesImGUIBindingIndex = 0u;
	// we create the Descriptor Set with a few slots extra to spare, so we don't have to `waitIdle` the device whenever ImGUI virtual window resizes
	constexpr static inline auto MaxImGUITextures = 2u + MaxFramesInFlight;

	static inline SAMPLING_MODE m_samplingMode = SAMPLING_MODE::PROJECTED_PARALLELOGRAM_SOLID_ANGLE;
	static inline int m_SampleCount = 64;
	static inline bool m_frameSeeding = true;
	static inline ResultData m_GPUOutResulData;
	//
	smart_refctd_ptr<CGeometryCreatorScene> m_scene;
	smart_refctd_ptr<IGPURenderpass> m_solidAngleRenderpass;
	smart_refctd_ptr<IGPURenderpass> m_mainRenderpass;
	smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;
	smart_refctd_ptr<IGPUFramebuffer> m_solidAngleViewFramebuffer;
	smart_refctd_ptr<IGPUFramebuffer> m_mainViewFramebuffer;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_solidAngleVisPipeline;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_rayVisualizationPipeline;
	//
	nbl::video::IDeviceMemoryAllocator::SAllocation m_allocation = {};
	smart_refctd_ptr<IGPUBuffer> m_outputStorageBuffer;
	smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_ds = nullptr;
	smart_refctd_ptr<ISemaphore> m_semaphore;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	//
	InputSystem::ChannelReader<IMouseEventChannel> mouse;
	InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	// UI stuff
	struct CInterface
	{
		void operator()()
		{
			ImGuiIO& io = ImGui::GetIO();

			// TODO: why is this a lambda and not just an assignment in a scope ?
			camera.setProjectionMatrix([&]()
				{
					const auto& sceneRes = float16_t2(mainViewTransformReturnInfo.sceneResolution);

					matrix4SIMD projection;
					if (isPerspective)
						if (isLH)
							projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(fov), sceneRes.x / sceneRes.y, zNear, zFar);
						else
							projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(fov), sceneRes.x / sceneRes.y, zNear, zFar);
					else
					{
						float viewHeight = viewWidth * sceneRes.y / sceneRes.x;

						if (isLH)
							projection = matrix4SIMD::buildProjectionMatrixOrthoLH(viewWidth, viewHeight, zNear, zFar);
						else
							projection = matrix4SIMD::buildProjectionMatrixOrthoRH(viewWidth, viewHeight, zNear, zFar);
					}

					return projection; }());

			ImGuizmo::SetOrthographic(!isPerspective);
			ImGuizmo::BeginFrame();

			ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
			ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

			// create a window and insert the inspector
			ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
			ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
			ImGui::Begin("Editor");

			ImGui::Text("Benchmarking Solid Angle Visualizer");

			if (ImGui::Button("Run Benchmark"))
			{
				SolidAngleVisualizer::SamplingBenchmark benchmark(*m_visualizer);
				benchmark.run();
			}
			ImGui::Separator();

			ImGui::Text("Sampling Mode:");
			ImGui::SameLine();

			const char* samplingModes[] =
			{
				"Triangle Solid Angle",
				"Triangle Projected Solid Angle",
				"Parallelogram Projected Solid Angle"
			};

			int currentMode = static_cast<int>(m_samplingMode);

			if (ImGui::Combo("##SamplingMode", &currentMode, samplingModes, IM_ARRAYSIZE(samplingModes)))
			{
				m_samplingMode = static_cast<SAMPLING_MODE>(currentMode);
			}



			ImGui::Checkbox("Frame seeding", &m_frameSeeding);

			ImGui::SliderInt("Sample Count", &m_SampleCount, 0, 512);

			ImGui::Separator();

			ImGui::Text("Camera");

			if (ImGui::RadioButton("LH", isLH))
				isLH = true;

			ImGui::SameLine();

			if (ImGui::RadioButton("RH", !isLH))
				isLH = false;

			if (ImGui::RadioButton("Perspective", isPerspective))
				isPerspective = true;

			ImGui::SameLine();

			if (ImGui::RadioButton("Orthographic", !isPerspective))
				isPerspective = false;

			ImGui::Checkbox("Enable \"view manipulate\"", &transformParams.enableViewManipulate);
			// ImGui::Checkbox("Enable camera movement", &move);
			ImGui::SliderFloat("Move speed", &moveSpeed, 0.1f, 10.f);
			ImGui::SliderFloat("Rotate speed", &rotateSpeed, 0.1f, 10.f);

			// ImGui::Checkbox("Flip Gizmo's Y axis", &flipGizmoY); // let's not expose it to be changed in UI but keep the logic in case

			if (isPerspective)
				ImGui::SliderFloat("Fov", &fov, 20.f, 150.f);
			else
				ImGui::SliderFloat("Ortho width", &viewWidth, 1, 20);

			ImGui::SliderFloat("zNear", &zNear, 0.1f, 100.f);
			ImGui::SliderFloat("zFar", &zFar, 110.f, 10000.f);

			if (firstFrame)
			{
				camera.setPosition(cameraIntialPosition);
				camera.setTarget(cameraInitialTarget);
				camera.setUpVector(cameraInitialUp);

				camera.recomputeViewMatrix();
			}
			firstFrame = false;

			ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);
			if (ImGuizmo::IsUsing())
			{
				ImGui::Text("Using gizmo");
			}
			else
			{
				ImGui::Text(ImGuizmo::IsOver() ? "Over gizmo" : "");
				ImGui::SameLine();
				ImGui::Text(ImGuizmo::IsOver(ImGuizmo::TRANSLATE) ? "Over translate gizmo" : "");
				ImGui::SameLine();
				ImGui::Text(ImGuizmo::IsOver(ImGuizmo::ROTATE) ? "Over rotate gizmo" : "");
				ImGui::SameLine();
				ImGui::Text(ImGuizmo::IsOver(ImGuizmo::SCALE) ? "Over scale gizmo" : "");
			}
			ImGui::Separator();

			/*
			* ImGuizmo expects view & perspective matrix to be column major both with 4x4 layout
			* and Nabla uses row major matricies - 3x4 matrix for view & 4x4 for projection

			- VIEW:

				ImGuizmo

				|     X[0]          Y[0]          Z[0]         0.0f |
				|     X[1]          Y[1]          Z[1]         0.0f |
				|     X[2]          Y[2]          Z[2]         0.0f |
				| -Dot(X, eye)  -Dot(Y, eye)  -Dot(Z, eye)     1.0f |

				Nabla

				|     X[0]         X[1]           X[2]     -Dot(X, eye)  |
				|     Y[0]         Y[1]           Y[2]     -Dot(Y, eye)  |
				|     Z[0]         Z[1]           Z[2]     -Dot(Z, eye)  |

				<ImGuizmo View Matrix> = transpose(nbl::core::matrix4SIMD(<Nabla View Matrix>))

			- PERSPECTIVE [PROJECTION CASE]:

				ImGuizmo

				|      (temp / temp2)                 (0.0)                       (0.0)                   (0.0)  |
				|          (0.0)                  (temp / temp3)                  (0.0)                   (0.0)  |
				| ((right + left) / temp2)   ((top + bottom) / temp3)    ((-zfar - znear) / temp4)       (-1.0f) |
				|          (0.0)                      (0.0)               ((-temp * zfar) / temp4)        (0.0)  |

				Nabla

				|            w                        (0.0)                       (0.0)                   (0.0)               |
				|          (0.0)                       -h                         (0.0)                   (0.0)               |
				|          (0.0)                      (0.0)               (-zFar/(zFar-zNear))     (-zNear*zFar/(zFar-zNear)) |
				|          (0.0)                      (0.0)                      (-1.0)                   (0.0)               |

				<ImGuizmo Projection Matrix> = transpose(<Nabla Projection Matrix>)

			*
			* the ViewManipulate final call (inside EditTransform) returns world space column major matrix for an object,
			* note it also modifies input view matrix but projection matrix is immutable
			*/

			// No need because camera already has this functionality
			// if (ImGui::IsKeyPressed(ImGuiKey_Home))
			// {
			// 	cameraToHome();
			// }

			if (ImGui::IsKeyPressed(ImGuiKey_End))
			{
				m_TRS = TRS{};
			}

			{
				static struct
				{
					float32_t4x4 view, projection, model;
				} imguizmoM16InOut;

				ImGuizmo::SetID(0u);

				// TODO: camera will return hlsl::float32_tMxN
				auto view = *reinterpret_cast<const float32_t3x4*>(camera.getViewMatrix().pointer());
				imguizmoM16InOut.view = hlsl::transpose(getMatrix3x4As4x4(view));

				// TODO: camera will return hlsl::float32_tMxN
				imguizmoM16InOut.projection = hlsl::transpose(*reinterpret_cast<const float32_t4x4*>(camera.getProjectionMatrix().pointer()));
				ImGuizmo::RecomposeMatrixFromComponents(&m_TRS.translation.x, &m_TRS.rotation.x, &m_TRS.scale.x, &imguizmoM16InOut.model[0][0]);

				if (flipGizmoY)								   // note we allow to flip gizmo just to match our coordinates
					imguizmoM16InOut.projection[1][1] *= -1.f; // https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/

				transformParams.editTransformDecomposition = true;
				mainViewTransformReturnInfo = EditTransform(&imguizmoM16InOut.view[0][0], &imguizmoM16InOut.projection[0][0], &imguizmoM16InOut.model[0][0], transformParams);
				move = mainViewTransformReturnInfo.allowCameraMovement;

				ImGuizmo::DecomposeMatrixToComponents(&imguizmoM16InOut.model[0][0], &m_TRS.translation.x, &m_TRS.rotation.x, &m_TRS.scale.x);
				ImGuizmo::RecomposeMatrixFromComponents(&m_TRS.translation.x, &m_TRS.rotation.x, &m_TRS.scale.x, &imguizmoM16InOut.model[0][0]);
			}
			// object meta display
			//{
			//	ImGui::Begin("Object");
			//	ImGui::Text("type: \"%s\"", objectName.data());
			//	ImGui::End();
			//}

			// solid angle view window
			{
				ImGui::SetNextWindowSize(ImVec2(800, 800), ImGuiCond_Appearing);
				ImGui::SetNextWindowPos(ImVec2(1240, 20), ImGuiCond_Appearing);
				static bool isOpen = true;
				ImGui::Begin("Projected Solid Angle View", &isOpen, 0);

				ImVec2 contentRegionSize = ImGui::GetContentRegionAvail();
				solidAngleViewTransformReturnInfo.sceneResolution = uint16_t2(static_cast<uint16_t>(contentRegionSize.x), static_cast<uint16_t>(contentRegionSize.y));
				solidAngleViewTransformReturnInfo.allowCameraMovement = false; // not used in this view
				ImGui::Image({ renderColorViewDescIndices[ERV_SOLID_ANGLE_VIEW] }, contentRegionSize);
				ImGui::End();
			}

			// Show data coming from GPU
#if DEBUG_DATA
			{
				if (ImGui::Begin("Result Data"))
				{
					auto drawColorField = [&](const char* fieldName, uint32_t index)
						{
							ImGui::Text("%s: %u", fieldName, index);

							if (index >= 27)
							{
								ImGui::SameLine();
								ImGui::Text("<invalid>");
								return;
							}

							const auto& c = colorLUT[index]; // uses the combined LUT we made earlier

							ImGui::SameLine();

							// Color preview button
							ImGui::ColorButton(
								fieldName,
								ImVec4(c.r, c.g, c.b, 1.0f),
								0,
								ImVec2(20, 20));

							ImGui::SameLine();
							ImGui::Text("%s", colorNames[index]);
						};

					// Vertices
					if (ImGui::CollapsingHeader("Vertices", ImGuiTreeNodeFlags_DefaultOpen))
					{
						for (uint32_t i = 0; i < 6; ++i)
						{
							if (i < m_GPUOutResulData.silhouetteVertexCount)
							{
								ImGui::Text("corners[%u]", i);
								ImGui::SameLine();
								drawColorField(":", m_GPUOutResulData.vertices[i]);
								ImGui::SameLine();
								static const float32_t3 constCorners[8] = {
									float32_t3(-1, -1, -1), float32_t3(1, -1, -1), float32_t3(-1, 1, -1), float32_t3(1, 1, -1),
									float32_t3(-1, -1, 1), float32_t3(1, -1, 1), float32_t3(-1, 1, 1), float32_t3(1, 1, 1) };
								float32_t3 vertexLocation = constCorners[m_GPUOutResulData.vertices[i]];
								ImGui::Text(" : (%.3f, %.3f, %.3f", vertexLocation.x, vertexLocation.y, vertexLocation.z);
							}
							else
							{
								ImGui::Text("corners[%u] ::  ", i);
								ImGui::SameLine();
								ImGui::ColorButton(
									"<unused>",
									ImVec4(0.0f, 0.0f, 0.0f, 0.0f),
									0,
									ImVec2(20, 20));
								ImGui::SameLine();
								ImGui::Text("<unused>");
							}
						}
					}

					if (ImGui::CollapsingHeader("Color LUT Map"))
					{
						for (int i = 0; i < 27; i++)
							drawColorField(" ", i);
					}

					ImGui::Separator();

					// Silhouette info
					drawColorField("silhouetteIndex", m_GPUOutResulData.silhouetteIndex);

					ImGui::Text("silhouette Vertex Count: %u", m_GPUOutResulData.silhouetteVertexCount);
					ImGui::Text("silhouette Positive VertexCount: %u", m_GPUOutResulData.positiveVertCount);
					ImGui::Text("Silhouette Mismatch: %s", m_GPUOutResulData.edgeVisibilityMismatch ? "true" : "false");
					ImGui::Separator();
					ImGui::Text("Max triangles exceeded: %s", m_GPUOutResulData.maxTrianglesExceeded ? "true" : "false");
					ImGui::Text("spherical lune detected: %s", m_GPUOutResulData.sphericalLuneDetected ? "true" : "false");
					ImGui::Separator();
					//ImGui::Text("Sampling outside the silhouette: %s", m_GPUOutResulData.sampleOutsideSilhouette ? "true" : "false");
					ImGui::Text("Parallelogram does not bound: %s", m_GPUOutResulData.parallelogramDoesNotBound ? "true" : "false");
					ImGui::Text("Parallelogram vertices inside: %s", m_GPUOutResulData.parallelogramVerticesInside ? "true" : "false");
					ImGui::Text("Parallelogram edges inside: %s", m_GPUOutResulData.parallelogramEdgesInside ? "true" : "false");
					ImGui::Text("Parallelogram area: %.3f", m_GPUOutResulData.parallelogramArea);
					ImGui::Text("Failed vertex index: %u", m_GPUOutResulData.failedVertexIndex);
					ImGui::Text("Failed vertex UV: (%.3f, %.3f)", m_GPUOutResulData.failedVertexUV.x, m_GPUOutResulData.failedVertexUV.y);
					ImGui::Text("Failed edge index: %u", m_GPUOutResulData.failedEdgeIndex);
					ImGui::Text("Failed edge sample: %u", m_GPUOutResulData.failedEdgeSample);
					ImGui::Text("Failed edge UV: (%.3f, %.3f)", m_GPUOutResulData.failedEdgeUV.x, m_GPUOutResulData.failedEdgeUV.y);
					ImGui::Text("Failed point 3D: (%.3f, %.3f, %.3f)", m_GPUOutResulData.failedPoint.x, m_GPUOutResulData.failedPoint.y, m_GPUOutResulData.failedPoint.z);
					for (uint32_t i = 0; i < 8; i++)
						ImGui::Text("edge is convex: %s", m_GPUOutResulData.edgeIsConvex[i] ? "true" : "false");
					ImGui::Separator();

					{
						float32_t3 xAxis = m_OBBModelMatrix[0].xyz;
						float32_t3 yAxis = m_OBBModelMatrix[1].xyz;
						float32_t3 zAxis = m_OBBModelMatrix[2].xyz;

						float32_t3 nx = normalize(xAxis);
						float32_t3 ny = normalize(yAxis);
						float32_t3 nz = normalize(zAxis);

						const float epsilon = 1e-4;
						bool hasSkew = false;
						if (abs(dot(nx, ny)) > epsilon || abs(dot(nx, nz)) > epsilon || abs(dot(ny, nz)) > epsilon)
							hasSkew = true;
						ImGui::Text("Matrix Has Skew: %s", hasSkew ? "true" : "false");
					}

					static bool modalShown = false;
					static bool modalDismissed = false;
					static uint32_t lastSilhouetteIndex = ~0u;

					// Reset modal flags if silhouette configuration changed
					if (m_GPUOutResulData.silhouetteIndex != lastSilhouetteIndex)
					{
						modalShown = false;
						modalDismissed = false; // Allow modal to show again for new configuration
						lastSilhouetteIndex = m_GPUOutResulData.silhouetteIndex;
					}

					// Reset flags when mismatch is cleared
					if (!m_GPUOutResulData.edgeVisibilityMismatch && !m_GPUOutResulData.maxTrianglesExceeded && !m_GPUOutResulData.sphericalLuneDetected)
					{
						modalShown = false;
						modalDismissed = false;
					}

					// Open modal only if not already shown/dismissed
					if ((m_GPUOutResulData.edgeVisibilityMismatch || m_GPUOutResulData.maxTrianglesExceeded || m_GPUOutResulData.sphericalLuneDetected) && m_GPUOutResulData.silhouetteIndex != 13 && !modalShown && !modalDismissed) // Don't reopen if user dismissed it
					{
						ImGui::OpenPopup("Edge Visibility Mismatch Warning");
						modalShown = true;
					}

					// Modal popup
					if (ImGui::BeginPopupModal("Edge Visibility Mismatch Warning", NULL, ImGuiWindowFlags_AlwaysAutoResize))
					{
						ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Warning: Edge Visibility Mismatch Detected!");
						ImGui::Separator();
						ImGui::Text("The silhouette lookup table (LUT) does not match the computed edge visibility.");
						ImGui::Text("This indicates the pre-computed silhouette data may be incorrect.");
						ImGui::Spacing();
						ImGui::TextWrapped("Configuration Index: %u", m_GPUOutResulData.silhouetteIndex);
						ImGui::TextWrapped("Region: (%u, %u, %u)", m_GPUOutResulData.region.x, m_GPUOutResulData.region.y, m_GPUOutResulData.region.z);
						ImGui::Spacing();
						ImGui::Text("Mismatched Vertices (bitmask): 0x%08X", m_GPUOutResulData.edgeVisibilityMismatch);
						ImGui::Text("Vertices involved in mismatched edges:");
						ImGui::Indent();
						for (int i = 0; i < 8; i++)
						{
							if (m_GPUOutResulData.edgeVisibilityMismatch & (1u << i))
							{
								ImGui::BulletText("Vertex %d", i);
							}
						}
						ImGui::Unindent();
						ImGui::Spacing();
						if (ImGui::Button("OK", ImVec2(120, 0)))
						{
							ImGui::CloseCurrentPopup();
							modalShown = false;
							modalDismissed = true; // Mark as dismissed to prevent reopening
						}
						ImGui::EndPopup();
					}

					ImGui::Separator();

					// Region (uint32_t3)
					ImGui::Text("region: (%u, %u, %u)",
						m_GPUOutResulData.region.x, m_GPUOutResulData.region.y, m_GPUOutResulData.region.z);

					// print solidAngles for each triangle
					{
						ImGui::Text("Solid Angles per Triangle:");
						ImGui::BeginTable("SolidAnglesTable", 2);
						ImGui::TableSetupColumn("Triangle Index");
						ImGui::TableSetupColumn("Solid Angle");
						ImGui::TableHeadersRow();
						for (uint32_t i = 0; i < m_GPUOutResulData.triangleCount; ++i)
						{
							ImGui::TableNextRow();
							ImGui::TableSetColumnIndex(0);
							ImGui::Text("%u", i);
							ImGui::TableSetColumnIndex(1);
							ImGui::Text("%.6f", m_GPUOutResulData.solidAngles[i]);
						}
						ImGui::Text("Total: %.6f", m_GPUOutResulData.totalSolidAngles);
						ImGui::EndTable();
					}

					ImGui::Separator();

					// Silhouette mask printed in binary

					auto printBin = [](uint32_t bin, const char* name)
						{
							char buf[33];
							for (int i = 0; i < 32; i++)
								buf[i] = (bin & (1u << (31 - i))) ? '1' : '0';
							buf[32] = '\0';
							ImGui::Text("%s: 0x%08X", name, bin);
							ImGui::Text("binary: 0b%s", buf);
							ImGui::Separator();
						};
					printBin(m_GPUOutResulData.silhouette, "Silhouette");
					printBin(m_GPUOutResulData.rotatedSil, "rotatedSilhouette");

					printBin(m_GPUOutResulData.clipCount, "clipCount");
					printBin(m_GPUOutResulData.clipMask, "clipMask");
					printBin(m_GPUOutResulData.rotatedClipMask, "rotatedClipMask");
					printBin(m_GPUOutResulData.rotateAmount, "rotateAmount");
					printBin(m_GPUOutResulData.wrapAround, "wrapAround");
				}
				ImGui::End();
			}
#endif
			// view matrices editor
			{
				ImGui::Begin("Matrices");

				auto addMatrixTable = [&](const char* topText, const char* tableName, const int rows, const int columns, const float* pointer, const bool withSeparator = true)
					{
						ImGui::Text(topText);
						if (ImGui::BeginTable(tableName, columns))
						{
							for (int y = 0; y < rows; ++y)
							{
								ImGui::TableNextRow();
								for (int x = 0; x < columns; ++x)
								{
									ImGui::TableSetColumnIndex(x);
									ImGui::Text("%.3f", *(pointer + (y * columns) + x));
								}
							}
							ImGui::EndTable();
						}

						if (withSeparator)
							ImGui::Separator();
					};

				static RandomSampler rng(0x45); // Initialize RNG with seed

				// Helper function to check if cube intersects unit sphere at origin
				auto isCubeOutsideUnitSphere = [](const float32_t3& translation, const float32_t3& scale) -> bool
					{
						float cubeRadius = glm::length(scale) * 0.5f;
						float distanceToCenter = glm::length(translation);
						return (distanceToCenter - cubeRadius) > 1.0f;
					};

				static TRS lastTRS = {};
				if (ImGui::Button("Randomize Translation"))
				{
					lastTRS = m_TRS; // Backup before randomizing
					int attempts = 0;
					do
					{
						m_TRS.translation = float32_t3(rng.nextFloat(-3.f, 3.f), rng.nextFloat(-3.f, 3.f), rng.nextFloat(-1.f, 3.f));
						attempts++;
					} while (!isCubeOutsideUnitSphere(m_TRS.translation, m_TRS.scale) && attempts < 100);
				}
				ImGui::SameLine();
				if (ImGui::Button("Randomize Rotation"))
				{
					lastTRS = m_TRS; // Backup before randomizing
					m_TRS.rotation = float32_t3(rng.nextFloat(-180.f, 180.f), rng.nextFloat(-180.f, 180.f), rng.nextFloat(-180.f, 180.f));
				}
				ImGui::SameLine();
				if (ImGui::Button("Randomize Scale"))
				{
					lastTRS = m_TRS; // Backup before randomizing
					int attempts = 0;
					do
					{
						m_TRS.scale = float32_t3(rng.nextFloat(0.5f, 2.0f), rng.nextFloat(0.5f, 2.0f), rng.nextFloat(0.5f, 2.0f));
						attempts++;
					} while (!isCubeOutsideUnitSphere(m_TRS.translation, m_TRS.scale) && attempts < 100);
				}
				// ImGui::SameLine();
				if (ImGui::Button("Randomize All"))
				{
					lastTRS = m_TRS; // Backup before randomizing
					int attempts = 0;
					do
					{
						m_TRS.translation = float32_t3(rng.nextFloat(-3.f, 3.f), rng.nextFloat(-3.f, 3.f), rng.nextFloat(-1.f, 3.f));
						m_TRS.rotation = float32_t3(rng.nextFloat(-180.f, 180.f), rng.nextFloat(-180.f, 180.f), rng.nextFloat(-180.f, 180.f));
						m_TRS.scale = float32_t3(rng.nextFloat(0.5f, 2.0f), rng.nextFloat(0.5f, 2.0f), rng.nextFloat(0.5f, 2.0f));
						attempts++;
					} while (!isCubeOutsideUnitSphere(m_TRS.translation, m_TRS.scale) && attempts < 100);
				}
				ImGui::SameLine();
				if (ImGui::Button("Revert to Last"))
				{
					m_TRS = lastTRS; // Restore backed-up TRS
				}

				addMatrixTable("Model Matrix", "ModelMatrixTable", 4, 4, &m_OBBModelMatrix[0][0]);
				addMatrixTable("Camera View Matrix", "ViewMatrixTable", 3, 4, camera.getViewMatrix().pointer());
				addMatrixTable("Camera View Projection Matrix", "ViewProjectionMatrixTable", 4, 4, camera.getProjectionMatrix().pointer(), false);

				ImGui::End();
			}

			// Nabla Imgui backend MDI buffer info
			// To be 100% accurate and not overly conservative we'd have to explicitly `cull_frees` and defragment each time,
			// so unless you do that, don't use this basic info to optimize the size of your IMGUI buffer.
			{
				auto* streaminingBuffer = imGUI->getStreamingBuffer();

				const size_t total = streaminingBuffer->get_total_size();						  // total memory range size for which allocation can be requested
				const size_t freeSize = streaminingBuffer->getAddressAllocator().get_free_size(); // max total free bloock memory size we can still allocate from total memory available
				const size_t consumedMemory = total - freeSize;									  // memory currently consumed by streaming buffer

				float freePercentage = 100.0f * (float)(freeSize) / (float)total;
				float allocatedPercentage = (float)(consumedMemory) / (float)total;

				ImVec2 barSize = ImVec2(400, 30);
				float windowPadding = 10.0f;
				float verticalPadding = ImGui::GetStyle().FramePadding.y;

				ImGui::SetNextWindowSize(ImVec2(barSize.x + 2 * windowPadding, 110 + verticalPadding), ImGuiCond_Always);
				ImGui::Begin("Nabla Imgui MDI Buffer Info", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar);

				ImGui::Text("Total Allocated Size: %zu bytes", total);
				ImGui::Text("In use: %zu bytes", consumedMemory);
				ImGui::Text("Buffer Usage:");

				ImGui::SetCursorPosX(windowPadding);

				if (freePercentage > 70.0f)
					ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.0f, 1.0f, 0.0f, 0.4f)); // Green
				else if (freePercentage > 30.0f)
					ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 1.0f, 0.0f, 0.4f)); // Yellow
				else
					ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 0.0f, 0.0f, 0.4f)); // Red

				ImGui::ProgressBar(allocatedPercentage, barSize, "");

				ImGui::PopStyleColor();

				ImDrawList* drawList = ImGui::GetWindowDrawList();

				ImVec2 progressBarPos = ImGui::GetItemRectMin();
				ImVec2 progressBarSize = ImGui::GetItemRectSize();

				const char* text = "%.2f%% free";
				char textBuffer[64];
				snprintf(textBuffer, sizeof(textBuffer), text, freePercentage);

				ImVec2 textSize = ImGui::CalcTextSize(textBuffer);
				ImVec2 textPos = ImVec2(
					progressBarPos.x + (progressBarSize.x - textSize.x) * 0.5f,
					progressBarPos.y + (progressBarSize.y - textSize.y) * 0.5f);

				ImVec4 bgColor = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
				drawList->AddRectFilled(
					ImVec2(textPos.x - 5, textPos.y - 2),
					ImVec2(textPos.x + textSize.x + 5, textPos.y + textSize.y + 2),
					ImGui::GetColorU32(bgColor));

				ImGui::SetCursorScreenPos(textPos);
				ImGui::Text("%s", textBuffer);

				ImGui::Dummy(ImVec2(0.0f, verticalPadding));

				ImGui::End();
			}
			ImGui::End();

			ImGuizmo::RecomposeMatrixFromComponents(&m_TRS.translation.x, &m_TRS.rotation.x, &m_TRS.scale.x, &m_OBBModelMatrix[0][0]);
		}

		smart_refctd_ptr<ext::imgui::UI> imGUI;

		// descriptor set
		smart_refctd_ptr<SubAllocatedDescriptorSet> subAllocDS;
		enum E_RENDER_VIEWS : uint8_t
		{
			ERV_MAIN_VIEW,
			ERV_SOLID_ANGLE_VIEW,
			Count
		};
		SubAllocatedDescriptorSet::value_type renderColorViewDescIndices[E_RENDER_VIEWS::Count] = { SubAllocatedDescriptorSet::invalid_value, SubAllocatedDescriptorSet::invalid_value };
		//
		Camera camera = Camera(cameraIntialPosition, cameraInitialTarget, core::matrix4SIMD(), 1, 1, nbl::core::vectorSIMDf(0.0f, 0.0f, 1.0f));
		// mutables
		struct TRS // Source of truth
		{
			float32_t3 translation{ 0.0f, 0.0f, 1.5f };
			float32_t3 rotation{ 0.0f }; // MUST stay orthonormal
			float32_t3 scale{ 1.0f };
		} m_TRS;
		float32_t4x4 m_OBBModelMatrix; // always overwritten from TRS

		// std::string_view objectName;
		TransformRequestParams transformParams;
		TransformReturnInfo mainViewTransformReturnInfo;
		TransformReturnInfo solidAngleViewTransformReturnInfo;

		const static inline core::vectorSIMDf cameraIntialPosition{ -3.0f, 6.0f, 3.0f };
		const static inline core::vectorSIMDf cameraInitialTarget{ 0.f, 0.0f, 3.f };
		const static inline core::vectorSIMDf cameraInitialUp{ 0.f, 0.f, 1.f };

		float fov = 90.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
		float viewWidth = 10.f;
		// uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed
		bool isPerspective = true, isLH = true, flipGizmoY = true, move = true;
		bool firstFrame = true;

		SolidAngleVisualizer* m_visualizer;
	} interface;

	class SamplingBenchmark final
	{
	public:
		SamplingBenchmark(SolidAngleVisualizer& base)
			: m_api(base.m_api), m_device(base.m_device), m_logger(base.m_logger), m_visualizer(&base)
		{

			// setting up pipeline in the constructor
			m_queueFamily = base.getComputeQueue()->getFamilyIndex();
			m_cmdpool = base.m_device->createCommandPool(m_queueFamily, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			// core::smart_refctd_ptr<IGPUCommandBuffer>* cmdBuffs[] = { &m_cmdbuf, &m_timestampBeforeCmdBuff, &m_timestampAfterCmdBuff };
			if (!m_cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_cmdbuf))
				base.logFail("Failed to create Command Buffers!\n");
			if (!m_cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_timestampBeforeCmdBuff))
				base.logFail("Failed to create Command Buffers!\n");
			if (!m_cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_timestampAfterCmdBuff))
				base.logFail("Failed to create Command Buffers!\n");

			// Load shaders, set up pipeline
			{
				smart_refctd_ptr<IShader> shader;
				{
					IAssetLoader::SAssetLoadParams lp = {};
					lp.logger = base.m_logger.get();
					lp.workingDirectory = "app_resources"; // virtual root
					// this time we load a shader directly from a file
					auto key = nbl::this_example::builtin::build::get_spirv_key<"benchmark">(m_device.get());
					auto assetBundle = base.m_assetMgr->getAsset(key.data(), lp);
					const auto assets = assetBundle.getContents();
					if (assets.empty())
					{
						base.logFail("Could not load shader!");
						assert(0);
					}

					// It would be super weird if loading a shader from a file produced more than 1 asset
					assert(assets.size() == 1);
					shader = IAsset::castDown<IShader>(assets[0]);
				}

				if (!shader)
					base.logFail("Failed to load precompiled \"benchmark\" shader!\n");

				nbl::video::IGPUDescriptorSetLayout::SBinding bindings[1] = {
					{.binding = 0,
					 .type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					 .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					 .stageFlags = ShaderStage::ESS_COMPUTE,
					 .count = 1} };
				smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = base.m_device->createDescriptorSetLayout(bindings);
				if (!dsLayout)
					base.logFail("Failed to create a Descriptor Layout!\n");

				SPushConstantRange pushConstantRanges[] = {
					{.stageFlags = ShaderStage::ESS_COMPUTE,
					 .offset = 0,
					 .size = sizeof(BenchmarkPushConstants)} };
				m_pplnLayout = base.m_device->createPipelineLayout(pushConstantRanges, smart_refctd_ptr(dsLayout));
				if (!m_pplnLayout)
					base.logFail("Failed to create a Pipeline Layout!\n");

				{
					IGPUComputePipeline::SCreationParams params = {};
					params.layout = m_pplnLayout.get();
					params.shader.entryPoint = "main";
					params.shader.shader = shader.get();
					if (!base.m_device->createComputePipelines(nullptr, { &params, 1 }, &m_pipeline))
						base.logFail("Failed to create pipelines (compile & link shaders)!\n");
				}

				// Allocate the memory
				{
					constexpr size_t BufferSize = BENCHMARK_WORKGROUP_COUNT * BENCHMARK_WORKGROUP_DIMENSION_SIZE_X *
						BENCHMARK_WORKGROUP_DIMENSION_SIZE_Y * BENCHMARK_WORKGROUP_DIMENSION_SIZE_Z * sizeof(uint32_t);

					nbl::video::IGPUBuffer::SCreationParams params = {};
					params.size = BufferSize;
					params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
					smart_refctd_ptr<IGPUBuffer> dummyBuff = base.m_device->createBuffer(std::move(params));
					if (!dummyBuff)
						base.logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

					dummyBuff->setObjectDebugName("benchmark buffer");

					nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = dummyBuff->getMemoryReqs();

					m_allocation = base.m_device->allocate(reqs, dummyBuff.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
					if (!m_allocation.isValid())
						base.logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

					assert(dummyBuff->getBoundMemory().memory == m_allocation.memory.get());
					smart_refctd_ptr<nbl::video::IDescriptorPool> pool = base.m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &dsLayout.get(), 1 });

					m_ds = pool->createDescriptorSet(std::move(dsLayout));
					{
						IGPUDescriptorSet::SDescriptorInfo info[1];
						info[0].desc = smart_refctd_ptr(dummyBuff);
						info[0].info.buffer = { .offset = 0, .size = BufferSize };
						IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
							{.dstSet = m_ds.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = info} };
						base.m_device->updateDescriptorSets(writes, {});
					}
				}
			}

			IQueryPool::SCreationParams queryPoolCreationParams{};
			queryPoolCreationParams.queryType = IQueryPool::TYPE::TIMESTAMP;
			queryPoolCreationParams.queryCount = 2;
			queryPoolCreationParams.pipelineStatisticsFlags = IQueryPool::PIPELINE_STATISTICS_FLAGS::NONE;
			m_queryPool = m_device->createQueryPool(queryPoolCreationParams);

			m_computeQueue = m_device->getQueue(m_queueFamily, 0);
		}

		void run()
		{
			m_logger->log("\n\nsampling benchmark result:", ILogger::ELL_PERFORMANCE);

			m_logger->log("sampling benchmark, parallelogram projected solid angle result:", ILogger::ELL_PERFORMANCE);
			performBenchmark(SAMPLING_MODE::PROJECTED_PARALLELOGRAM_SOLID_ANGLE);

			m_logger->log("sampling benchmark, triangle solid angle result:", ILogger::ELL_PERFORMANCE);
			performBenchmark(SAMPLING_MODE::TRIANGLE_SOLID_ANGLE);

			//m_logger->log("sampling benchmark, triangle projected solid angle result:", ILogger::ELL_PERFORMANCE);
			//performBenchmark(SAMPLING_MODE::TRIANGLE_PROJECTED_SOLID_ANGLE);

		}

	private:
		void performBenchmark(SAMPLING_MODE mode)
		{
			m_device->waitIdle();

			recordTimestampQueryCmdBuffers();

			uint64_t semaphoreCounter = 0;
			smart_refctd_ptr<ISemaphore> semaphore = m_device->createSemaphore(semaphoreCounter);

			IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = semaphore.get(), .value = 0u, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
			IQueue::SSubmitInfo::SSemaphoreInfo waits[] = { {.semaphore = semaphore.get(), .value = 0u, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };

			IQueue::SSubmitInfo beforeTimestapSubmitInfo[1] = {};
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufsBegin[] = { {.cmdbuf = m_timestampBeforeCmdBuff.get()} };
			beforeTimestapSubmitInfo[0].commandBuffers = cmdbufsBegin;
			beforeTimestapSubmitInfo[0].signalSemaphores = signals;
			beforeTimestapSubmitInfo[0].waitSemaphores = waits;

			IQueue::SSubmitInfo afterTimestapSubmitInfo[1] = {};
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufsEnd[] = { {.cmdbuf = m_timestampAfterCmdBuff.get()} };
			afterTimestapSubmitInfo[0].commandBuffers = cmdbufsEnd;
			afterTimestapSubmitInfo[0].signalSemaphores = signals;
			afterTimestapSubmitInfo[0].waitSemaphores = waits;

			IQueue::SSubmitInfo benchmarkSubmitInfos[1] = {};
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = m_cmdbuf.get()} };
			benchmarkSubmitInfos[0].commandBuffers = cmdbufs;
			benchmarkSubmitInfos[0].signalSemaphores = signals;
			benchmarkSubmitInfos[0].waitSemaphores = waits;

			m_pushConstants.benchmarkMode = mode;
			m_pushConstants.modelMatrix = float32_t3x4(transpose(m_visualizer->interface.m_OBBModelMatrix));
			recordCmdBuff();

			// warmup runs
			for (int i = 0; i < WarmupIterations; ++i)
			{

				if (i == 0)
					m_api->startCapture();
				waits[0].value = semaphoreCounter;
				signals[0].value = ++semaphoreCounter;
				m_computeQueue->submit(benchmarkSubmitInfos);
				if (i == 0)
					m_api->endCapture();
			}

			waits[0].value = semaphoreCounter;
			signals[0].value = ++semaphoreCounter;
			m_computeQueue->submit(beforeTimestapSubmitInfo);

			// actual benchmark runs
			for (int i = 0; i < Iterations; ++i)
			{
				waits[0].value = semaphoreCounter;
				signals[0].value = ++semaphoreCounter;
				m_computeQueue->submit(benchmarkSubmitInfos);
			}

			waits[0].value = semaphoreCounter;
			signals[0].value = ++semaphoreCounter;
			m_computeQueue->submit(afterTimestapSubmitInfo);

			m_device->waitIdle();

			const uint64_t nativeBenchmarkTimeElapsedNanoseconds = calcTimeElapsed();
			const float nativeBenchmarkTimeElapsedSeconds = double(nativeBenchmarkTimeElapsedNanoseconds) / 1000000000.0;

			m_logger->log("%llu ns, %f s", ILogger::ELL_PERFORMANCE, nativeBenchmarkTimeElapsedNanoseconds, nativeBenchmarkTimeElapsedSeconds);
		}

		void recordCmdBuff()
		{
			m_cmdbuf->begin(IGPUCommandBuffer::USAGE::SIMULTANEOUS_USE_BIT);
			m_cmdbuf->beginDebugMarker("sampling compute dispatch", vectorSIMDf(0, 1, 0, 1));
			m_cmdbuf->bindComputePipeline(m_pipeline.get());
			m_cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_pplnLayout.get(), 0, 1, &m_ds.get());
			m_cmdbuf->pushConstants(m_pplnLayout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(BenchmarkPushConstants), &m_pushConstants);
			m_cmdbuf->dispatch(BENCHMARK_WORKGROUP_COUNT, 1, 1);
			m_cmdbuf->endDebugMarker();
			m_cmdbuf->end();
		}

		void recordTimestampQueryCmdBuffers()
		{
			static bool firstInvocation = true;

			if (!firstInvocation)
			{
				m_timestampBeforeCmdBuff->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
				m_timestampBeforeCmdBuff->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
			}

			m_timestampBeforeCmdBuff->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			m_timestampBeforeCmdBuff->resetQueryPool(m_queryPool.get(), 0, 2);
			m_timestampBeforeCmdBuff->writeTimestamp(PIPELINE_STAGE_FLAGS::NONE, m_queryPool.get(), 0);
			m_timestampBeforeCmdBuff->end();

			m_timestampAfterCmdBuff->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			m_timestampAfterCmdBuff->writeTimestamp(PIPELINE_STAGE_FLAGS::NONE, m_queryPool.get(), 1);
			m_timestampAfterCmdBuff->end();

			firstInvocation = false;
		}

		uint64_t calcTimeElapsed()
		{
			uint64_t timestamps[2];
			const core::bitflag flags = core::bitflag(IQueryPool::RESULTS_FLAGS::_64_BIT) | core::bitflag(IQueryPool::RESULTS_FLAGS::WAIT_BIT);
			m_device->getQueryPoolResults(m_queryPool.get(), 0, 2, &timestamps, sizeof(uint64_t), flags);
			return timestamps[1] - timestamps[0];
		}

	private:
		core::smart_refctd_ptr<video::CVulkanConnection> m_api;
		smart_refctd_ptr<ILogicalDevice> m_device;
		smart_refctd_ptr<ILogger> m_logger;
		SolidAngleVisualizer* m_visualizer;

		nbl::video::IDeviceMemoryAllocator::SAllocation m_allocation = {};
		smart_refctd_ptr<nbl::video::IGPUCommandPool> m_cmdpool = nullptr;
		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_cmdbuf = nullptr;
		smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_ds = nullptr;
		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_pplnLayout = nullptr;
		BenchmarkPushConstants m_pushConstants;
		smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_pipeline;

		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_timestampBeforeCmdBuff = nullptr;
		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_timestampAfterCmdBuff = nullptr;
		smart_refctd_ptr<nbl::video::IQueryPool> m_queryPool = nullptr;

		uint32_t m_queueFamily;
		IQueue* m_computeQueue;
		static constexpr int WarmupIterations = 50;
		static constexpr int Iterations = 1;
	};

	template <typename... Args>
	inline bool logFail(const char* msg, Args &&...args)
	{
		m_logger->log(msg, ILogger::ELL_ERROR, std::forward<Args>(args)...);
		return false;
	}

	std::ofstream m_logFile;
};

NBL_MAIN_FUNC(SolidAngleVisualizer)