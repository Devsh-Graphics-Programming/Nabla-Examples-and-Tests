// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "common.hpp"
#include "app_resources/hlsl/common.hlsl"

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
public:
	inline SolidAngleVisualizer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
		device_base_t({ 2048,1024 }, EF_UNKNOWN, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {
	}

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;

		m_semaphore = m_device->createSemaphore(m_realFrameIx);
		if (!m_semaphore)
			return logFail("Failed to Create a Semaphore!");

		auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		for (auto i = 0u; i < MaxFramesInFlight; i++)
		{
			if (!pool)
				return logFail("Couldn't create Command Pool!");
			if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i,1 }))
				return logFail("Couldn't create Command Buffer!");
		}

		const uint32_t addtionalBufferOwnershipFamilies[] = { getGraphicsQueue()->getFamilyIndex() };
		m_scene = CGeometryCreatorScene::create(
			{
				.transferQueue = getTransferUpQueue(),
				.utilities = m_utils.get(),
				.logger = m_logger.get(),
				.addtionalBufferOwnershipFamilies = addtionalBufferOwnershipFamilies
			},
			CSimpleDebugRenderer::DefaultPolygonGeometryPatch
		);

		// for the scene drawing pass
		{
			IGPURenderpass::SCreationParams params = {};
			const IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] = {
				{{
					{
						.format = sceneRenderDepthFormat,
						.samples = IGPUImage::ESCF_1_BIT,
						.mayAlias = false
					},
				/*.loadOp =*/ {IGPURenderpass::LOAD_OP::CLEAR},
				/*.storeOp =*/ {IGPURenderpass::STORE_OP::STORE},
				/*.initialLayout =*/ {IGPUImage::LAYOUT::UNDEFINED},
				/*.finalLayout =*/ {IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}
			}},
			IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd
			};
			params.depthStencilAttachments = depthAttachments;
			const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
				{{
					{
						.format = finalSceneRenderFormat,
						.samples = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
						.mayAlias = false
					},
				/*.loadOp =*/ IGPURenderpass::LOAD_OP::CLEAR,
				/*.storeOp =*/ IGPURenderpass::STORE_OP::STORE,
				/*.initialLayout =*/ IGPUImage::LAYOUT::UNDEFINED,
				/*.finalLayout =*/ IGPUImage::LAYOUT::READ_ONLY_OPTIMAL // ImGUI shall read
			}},
			IGPURenderpass::SCreationParams::ColorAttachmentsEnd
			};
			params.colorAttachments = colorAttachments;
			IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
				{},
				IGPURenderpass::SCreationParams::SubpassesEnd
			};
			subpasses[0].depthStencilAttachment = { {.render = {.attachmentIndex = 0,.layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}} };
			subpasses[0].colorAttachments[0] = { .render = {.attachmentIndex = 0,.layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL} };
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
					.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
				}
				// leave view offsets and flags default
			},
			{
				.srcSubpass = 0,
				.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.memoryBarrier = {
					// last place where the color can get modified, depth is implicitly earlier
					.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					// only write ops, reads can't be made available, also won't be using depth so don't care about it being visible to anyone else
					.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
					// the ImGUI will sample the color, then next frame we overwrite both attachments
					.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT | PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
					// but we only care about the availability-visibility chain between renderpass and imgui 
					.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT
				}
				// leave view offsets and flags default
			},
			IGPURenderpass::SCreationParams::DependenciesEnd
			};
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
		m_renderer = CSimpleDebugRenderer::create(m_assetMgr.get(), m_solidAngleRenderpass.get(), 0, { &geometries.front().get(),geometries.size() });
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
			auto loadAndCompileHLSLShader = [&](const std::string& pathToShader, const std::string& defineMacro = "") -> smart_refctd_ptr<IShader>
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
					options.stage = IShader::E_SHADER_STAGE::ESS_FRAGMENT;
					options.preprocessorOptions.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
					options.spirvOptimizer = nullptr;
#ifndef _NBL_DEBUG
					ISPIRVOptimizer::E_OPTIMIZER_PASS optPasses = ISPIRVOptimizer::EOP_STRIP_DEBUG_INFO;
					auto opt = make_smart_refctd_ptr<ISPIRVOptimizer>(std::span<ISPIRVOptimizer::E_OPTIMIZER_PASS>(&optPasses, 1));
					options.spirvOptimizer = opt.get();
#endif
					options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
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

			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
			if (!fsTriProtoPPln)
				return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

			// Load Fragment Shader
			auto fragmentShader = loadAndCompileHLSLShader(SolidAngleVisShaderPath);
			if (!fragmentShader)
				return logFail("Failed to Load and Compile Fragment Shader: lumaMeterShader!");

			const IGPUPipelineBase::SShaderSpecInfo fragSpec = {
				.shader = fragmentShader.get(),
				.entryPoint = "main"
			};

			const asset::SPushConstantRange ranges[] = { {
				.stageFlags = hlsl::ShaderStage::ESS_FRAGMENT,
				.offset = 0,
				.size = sizeof(PushConstants)
			} };

			auto visualizationLayout = m_device->createPipelineLayout(
				ranges,
				nullptr,
				nullptr,
				nullptr,
				nullptr
			);
			m_visualizationPipeline = fsTriProtoPPln.createPipeline(fragSpec, visualizationLayout.get(), m_solidAngleRenderpass.get());
			if (!m_visualizationPipeline)
				return logFail("Could not create Graphics Pipeline!");

		}

		// Create ImGUI
		{
			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			ext::imgui::UI::SCreationParameters params = {};
			params.resources.texturesInfo = { .setIx = 0u,.bindingIx = TexturesImGUIBindingIndex };
			params.resources.samplersInfo = { .setIx = 0u,.bindingIx = 1u };
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
				auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT, { &layout,1 });
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
					.info = &info
				};
				if (!m_device->updateDescriptorSets({ &write,1 }, {}))
					return logFail("Failed to write the descriptor set");
			}
			imgui->registerListener([this]() {interface(); });
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

		const auto& virtualWindowRes = interface.transformReturnInfo.sceneResolution;
		// TODO: check main frame buffer too
		if (!m_solidAngleViewFramebuffer || m_solidAngleViewFramebuffer->getCreationParameters().width != virtualWindowRes[0] || m_solidAngleViewFramebuffer->getCreationParameters().height != virtualWindowRes[1])
			recreateFramebuffer(virtualWindowRes);

		//
		const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

		auto* const cb = m_cmdBufs.data()[resourceIx].get();
		cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		// clear to black for both things
		const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
		if (m_solidAngleViewFramebuffer)
		{
			cb->beginDebugMarker("Draw Circle View Frame");
			{
				const IGPUCommandBuffer::SClearDepthStencilValue farValue = { .depth = 0.f };
				const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo =
				{
					.framebuffer = m_solidAngleViewFramebuffer.get(),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = &farValue,
					.renderArea = {
						.offset = {0,0},
						.extent = {virtualWindowRes[0],virtualWindowRes[1]}
					}
				};
				beginRenderpass(cb, renderpassInfo);
			}
			// draw scene
			{
				PushConstants pc{
					.modelMatrix = hlsl::float32_t3x4(hlsl::transpose(interface.m_OBBModelMatrix)),
					.viewport = { 0.f,0.f,static_cast<float>(virtualWindowRes[0]),static_cast<float>(virtualWindowRes[1]) }
				};
				auto pipeline = m_visualizationPipeline;
				cb->bindGraphicsPipeline(pipeline.get());
				cb->pushConstants(pipeline->getLayout(), hlsl::ShaderStage::ESS_FRAGMENT, 0, sizeof(PushConstants), &pc);
				//cb->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS, pipeline->getLayout(), 3, 1, &ds);
				ext::FullScreenTriangle::recordDrawCall(cb);
			}
			cb->endRenderPass();
			cb->endDebugMarker();
		}
		// draw main view
		if (m_mainViewFramebuffer)
		{
			cb->beginDebugMarker("Main Scene Frame");
			{
				const IGPUCommandBuffer::SClearDepthStencilValue farValue = { .depth = 0.f };
				const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo =
				{
					.framebuffer = m_mainViewFramebuffer.get(),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = &farValue,
					.renderArea = {
						.offset = {0,0},
						.extent = {virtualWindowRes[0],virtualWindowRes[1]}
					}
				};
				beginRenderpass(cb, renderpassInfo);
			}
			// draw scene
			{
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
				auto transposed = hlsl::transpose(interface.m_OBBModelMatrix);
				memcpy(&instance.world, &transposed, sizeof(instance.world));
				instance.packedGeo = m_renderer->getGeometries().data();// +interface.gcIndex;
				m_renderer->render(cb, viewParams); // draw the cube/OBB


				// TODO: a better way to get identity matrix
				float32_t3x4 origin = {
					0.2f,0.0f,0.0f,0.0f,
					0.0f,0.2f,0.0f,0.0f,
					0.0f,0.0f,0.2f,0.0f
				};
				memcpy(&instance.world, &origin, sizeof(instance.world));
				instance.packedGeo = m_renderer->getGeometries().data() + 3; // sphere
				m_renderer->render(cb, viewParams);
			}
			cb->endRenderPass();
			cb->endDebugMarker();
		}
		{
			cb->beginDebugMarker("SolidAngleVisualizer IMGUI Frame");
			{
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo =
				{
					.framebuffer = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = nullptr,
					.renderArea = {
						.offset = {0,0},
						.extent = {m_window->getWidth(),m_window->getHeight()}
					}
				};
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
				const ISemaphore::SWaitInfo drawFinished = { .semaphore = m_semaphore.get(),.value = m_realFrameIx + 1u };
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
			.stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS
		};
		const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
		{
			{.cmdbuf = cb }
		};
		const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = {
			{
				.semaphore = device_base_t::getCurrentAcquire().semaphore,
				.value = device_base_t::getCurrentAcquire().acquireCount,
				.stageMask = PIPELINE_STAGE_FLAGS::NONE
			}
		};
		const IQueue::SSubmitInfo infos[] =
		{
			{
				.waitSemaphores = acquired,
				.commandBuffers = commandBuffers,
				.signalSemaphores = {&retval,1}
			}
		};

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
					.dstAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
				}
			// leave view offsets and flags default
			},
			// want layout transition to begin after all color output is done
			{
				.srcSubpass = 0,
				.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.memoryBarrier = {
				// last place where the color can get modified, depth is implicitly earlier
				.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
				// only write ops, reads can't be made available
				.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
				// spec says nothing is needed when presentation is the destination
			}
			// leave view offsets and flags default
		},
		IGPURenderpass::SCreationParams::DependenciesEnd
		};
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
					}
				},
				m_logger.get()
			);
			keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
				{
					//if (interface.move)
						camera.keyboardProcess(events); // don't capture the events, only let camera handle them with its impl

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						uiEvents.keyboard.emplace_back(e);
					}
				},
				m_logger.get()
			);
		}
		camera.endInputProcessing(nextPresentationTimestamp);

		const auto cursorPosition = m_window->getCursorControl()->getPosition();

		ext::imgui::UI::SUpdateParameters params =
		{
			.mousePosition = float32_t2(cursorPosition.x,cursorPosition.y) - float32_t2(m_window->getX(),m_window->getY()),
			.displaySize = {m_window->getWidth(),m_window->getHeight()},
			.mouseEvents = uiEvents.mouse,
			.keyboardEvents = uiEvents.keyboard
		};

		//interface.objectName = m_scene->getInitParams().geometryNames[interface.gcIndex];
		interface.imGUI->update(params);
	}

	void recreateFramebuffer(const uint16_t2 resolution)
	{
		auto createImageAndView = [&](E_FORMAT format)->smart_refctd_ptr<IGPUImageView>
			{
				auto image = m_device->createImage({ {
					.type = IGPUImage::ET_2D,
					.samples = IGPUImage::ESCF_1_BIT,
					.format = format,
					.extent = {resolution.x,resolution.y,1},
					.mipLevels = 1,
					.arrayLayers = 1,
					.usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_SAMPLED_BIT
				} });
				if (!m_device->allocate(image->getMemoryReqs(), image.get()).isValid())
					return nullptr;
				IGPUImageView::SCreationParams params = {
					.image = std::move(image),
					.viewType = IGPUImageView::ET_2D,
					.format = format
				};
				params.subresourceRange.aspectMask = isDepthOrStencilFormat(format) ? IGPUImage::EAF_DEPTH_BIT : IGPUImage::EAF_COLOR_BIT;
				return m_device->createImageView(std::move(params));
			};

		smart_refctd_ptr<IGPUImageView> solidAngleView;
		smart_refctd_ptr<IGPUImageView> mainView;
		// detect window minimization
		if (resolution.x < 0x4000 && resolution.y < 0x4000)
		{
			solidAngleView = createImageAndView(finalSceneRenderFormat);
			auto solidAngleDepthView = createImageAndView(sceneRenderDepthFormat);
			m_solidAngleViewFramebuffer = m_device->createFramebuffer({ {
				.renderpass = m_solidAngleRenderpass,
				.depthStencilAttachments = &solidAngleDepthView.get(),
				.colorAttachments = &solidAngleView.get(),
				.width = resolution.x,
				.height = resolution.y
			} });

			mainView = createImageAndView(finalSceneRenderFormat);
			auto mainDepthView = createImageAndView(sceneRenderDepthFormat);
			m_mainViewFramebuffer = m_device->createFramebuffer({ {
					.renderpass = m_mainRenderpass,
					.depthStencilAttachments = &mainDepthView.get(),
					.colorAttachments = &mainView.get(),
					.width = resolution.x,
					.height = resolution.y
				} });

		}
		else
		{
			m_solidAngleViewFramebuffer = nullptr;
			m_mainViewFramebuffer = nullptr;
		}

		// release previous slot and its image
		interface.subAllocDS->multi_deallocate(0, static_cast<int>(CInterface::Count), interface.renderColorViewDescIndices, { .semaphore = m_semaphore.get(),.value = m_realFrameIx });
		//
		if (solidAngleView)
		{
			interface.subAllocDS->multi_allocate(0, static_cast<int>(CInterface::Count), interface.renderColorViewDescIndices);
			// update descriptor set
			IGPUDescriptorSet::SDescriptorInfo infos[static_cast<int>(CInterface::Count)] = {};
			infos[0].desc = solidAngleView;
			infos[0].info.image.imageLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
			infos[1].desc = mainView;
			infos[1].info.image.imageLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
			const IGPUDescriptorSet::SWriteDescriptorSet write[static_cast<int>(CInterface::Count)] = {
				{.dstSet = interface.subAllocDS->getDescriptorSet(),
				.binding = TexturesImGUIBindingIndex,
				.arrayElement = interface.renderColorViewDescIndices[static_cast<int>(CInterface::ERV_SOLID_ANGLE_VIEW)],
				.count = 1,
				.info = &infos[static_cast<int>(CInterface::ERV_MAIN_VIEW)]
				},
				{
				.dstSet = interface.subAllocDS->getDescriptorSet(),
				.binding = TexturesImGUIBindingIndex,
				.arrayElement = interface.renderColorViewDescIndices[static_cast<int>(CInterface::ERV_MAIN_VIEW)],
				.count = 1,
				.info = &infos[1]
				}
			};
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
			.height = static_cast<float>(info.renderArea.extent.height)
		};
		cb->setViewport(0u, 1u, &viewport);
	}

	// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
	constexpr static inline uint32_t MaxFramesInFlight = 3u;
	constexpr static inline auto sceneRenderDepthFormat = EF_D32_SFLOAT;
	constexpr static inline auto finalSceneRenderFormat = EF_R8G8B8A8_SRGB;
	constexpr static inline auto TexturesImGUIBindingIndex = 0u;
	// we create the Descriptor Set with a few slots extra to spare, so we don't have to `waitIdle` the device whenever ImGUI virtual window resizes
	constexpr static inline auto MaxImGUITextures = 2u + MaxFramesInFlight;

	//
	smart_refctd_ptr<CGeometryCreatorScene> m_scene;
	smart_refctd_ptr<IGPURenderpass> m_solidAngleRenderpass;
	smart_refctd_ptr<IGPURenderpass> m_mainRenderpass;
	smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;
	smart_refctd_ptr<IGPUFramebuffer> m_solidAngleViewFramebuffer;
	smart_refctd_ptr<IGPUFramebuffer> m_mainViewFramebuffer;
	smart_refctd_ptr<video::IGPUGraphicsPipeline> m_visualizationPipeline;
	//
	smart_refctd_ptr<ISemaphore> m_semaphore;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	//
	InputSystem::ChannelReader<IMouseEventChannel> mouse;
	InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	// UI stuff
	struct CInterface
	{
		void cameraToHome()
		{
			core::vectorSIMDf cameraPosition(-3.0f, 3.0f, 6.0f);
			core::vectorSIMDf cameraTarget(0.f, 0.f, 6.f);
			const static core::vectorSIMDf up(0.f, 1.f, 0.f);

			camera.setPosition(cameraPosition);
			camera.setTarget(cameraTarget);
			camera.setBackupUpVector(up);

			camera.recomputeViewMatrix();
		}

		void operator()()
		{
			ImGuiIO& io = ImGui::GetIO();

			// TODO: why is this a lambda and not just an assignment in a scope ?
			camera.setProjectionMatrix([&]()
				{
					matrix4SIMD projection;

					if (isPerspective)
						if (isLH)
							projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y, zNear, zFar);
						else
							projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y, zNear, zFar);
					else
					{
						float viewHeight = viewWidth * io.DisplaySize.y / io.DisplaySize.x;

						if (isLH)
							projection = matrix4SIMD::buildProjectionMatrixOrthoLH(viewWidth, viewHeight, zNear, zFar);
						else
							projection = matrix4SIMD::buildProjectionMatrixOrthoRH(viewWidth, viewHeight, zNear, zFar);
					}

					return projection;
				}());

			ImGuizmo::SetOrthographic(false);
			ImGuizmo::BeginFrame();

			ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
			ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

			// create a window and insert the inspector
			ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
			ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
			ImGui::Begin("Editor");

			//if (ImGui::RadioButton("Full view", !transformParams.useWindow))
			//	transformParams.useWindow = false;

			//ImGui::SameLine();

			//if (ImGui::RadioButton("Window", transformParams.useWindow))
			//	transformParams.useWindow = true;

			ImGui::Text("Camera");
			bool viewDirty = false;

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
			//ImGui::Checkbox("Enable camera movement", &move);
			ImGui::SliderFloat("Move speed", &moveSpeed, 0.1f, 10.f);
			ImGui::SliderFloat("Rotate speed", &rotateSpeed, 0.1f, 10.f);

			// ImGui::Checkbox("Flip Gizmo's Y axis", &flipGizmoY); // let's not expose it to be changed in UI but keep the logic in case

			if (isPerspective)
				ImGui::SliderFloat("Fov", &fov, 20.f, 150.f);
			else
				ImGui::SliderFloat("Ortho width", &viewWidth, 1, 20);

			ImGui::SliderFloat("zNear", &zNear, 0.1f, 100.f);
			ImGui::SliderFloat("zFar", &zFar, 110.f, 10000.f);

			viewDirty |= ImGui::SliderFloat("Distance", &transformParams.camDistance, 1.f, 69.f);

			if (viewDirty || firstFrame)
			{
				cameraToHome();
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

			if (ImGui::IsKeyPressed(ImGuiKey_Home))
			{
				cameraToHome();
			}

			if (ImGui::IsKeyPressed(ImGuiKey_End))
			{
				m_OBBModelMatrix = {
					1.0f, 0.0f, 0.0f, 0.0f,
					0.0f, 1.0f, 0.0f, 0.0f,
					0.0f, 0.0f, 1.0f, 0.0f,
					0.0f, 0.0f, 12.0f, 1.0f
				};
			}

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
			imguizmoM16InOut.model = m_OBBModelMatrix;

			{
				if (flipGizmoY) // note we allow to flip gizmo just to match our coordinates
					imguizmoM16InOut.projection[1][1] *= -1.f; // https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/	

				transformParams.editTransformDecomposition = true;
				transformReturnInfo = EditTransform(&imguizmoM16InOut.view[0][0], &imguizmoM16InOut.projection[0][0], &imguizmoM16InOut.model[0][0], transformParams);

				// TODO: camera stops when cursor hovers gizmo, but we also want to stop when gizmo is being used
				move = (ImGui::IsMouseDown(ImGuiMouseButton_Left) || transformReturnInfo.isGizmoWindowHovered) && (!transformReturnInfo.isGizmoBeingUsed);
			}

			// to Nabla + update camera & model matrices
			// TODO: make it more nicely, extract:
			// - Position by computing inverse of the view matrix and grabbing its translation
			// - Target from 3rd row without W component of view matrix multiplied by some arbitrary distance value (can be the length of position from origin) and adding the position
			// But then set the view matrix this way anyway, because up-vector may not be compatible
			//const auto& view = camera.getViewMatrix();
			//const_cast<core::matrix3x4SIMD&>(view) = core::transpose(imguizmoM16InOut.view).extractSub3x4(); // a hack, correct way would be to use inverse matrix and get position + target because now it will bring you back to last position & target when switching from gizmo move to manual move (but from manual to gizmo is ok)
			m_OBBModelMatrix = imguizmoM16InOut.model;

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
				ImGui::Begin("Solid angle view", &isOpen, 0);

				ImVec2 contentRegionSize = ImGui::GetContentRegionAvail();
				ImGui::Image({ renderColorViewDescIndices[ERV_SOLID_ANGLE_VIEW] }, contentRegionSize);
				ImGui::End();
			}

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

				const size_t total = streaminingBuffer->get_total_size();			// total memory range size for which allocation can be requested
				const size_t freeSize = streaminingBuffer->getAddressAllocator().get_free_size();		// max total free bloock memory size we can still allocate from total memory available
				const size_t consumedMemory = total - freeSize;			// memory currently consumed by streaming buffer

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
					ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.0f, 1.0f, 0.0f, 0.4f));  // Green
				else if (freePercentage > 30.0f)
					ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 1.0f, 0.0f, 0.4f));  // Yellow
				else
					ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 0.0f, 0.0f, 0.4f));  // Red

				ImGui::ProgressBar(allocatedPercentage, barSize, "");

				ImGui::PopStyleColor();

				ImDrawList* drawList = ImGui::GetWindowDrawList();

				ImVec2 progressBarPos = ImGui::GetItemRectMin();
				ImVec2 progressBarSize = ImGui::GetItemRectSize();

				const char* text = "%.2f%% free";
				char textBuffer[64];
				snprintf(textBuffer, sizeof(textBuffer), text, freePercentage);

				ImVec2 textSize = ImGui::CalcTextSize(textBuffer);
				ImVec2 textPos = ImVec2
				(
					progressBarPos.x + (progressBarSize.x - textSize.x) * 0.5f,
					progressBarPos.y + (progressBarSize.y - textSize.y) * 0.5f
				);

				ImVec4 bgColor = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
				drawList->AddRectFilled
				(
					ImVec2(textPos.x - 5, textPos.y - 2),
					ImVec2(textPos.x + textSize.x + 5, textPos.y + textSize.y + 2),
					ImGui::GetColorU32(bgColor)
				);

				ImGui::SetCursorScreenPos(textPos);
				ImGui::Text("%s", textBuffer);

				ImGui::Dummy(ImVec2(0.0f, verticalPadding));

				ImGui::End();
			}
			ImGui::End();
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
		Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
		// mutables
		float32_t4x4 m_OBBModelMatrix{
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 12.0f, 1.0f
		};

		//std::string_view objectName;
		TransformRequestParams transformParams;
		TransformReturnInfo transformReturnInfo;

		float fov = 90.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
		float viewWidth = 10.f;
		float camYAngle = 90.f / 180.f * 3.14159f;
		float camXAngle = 0.f / 180.f * 3.14159f;
		//uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed
		bool isPerspective = true, isLH = true, flipGizmoY = true, move = true;
		bool firstFrame = true;
	} interface;
};

NBL_MAIN_FUNC(SolidAngleVisualizer)