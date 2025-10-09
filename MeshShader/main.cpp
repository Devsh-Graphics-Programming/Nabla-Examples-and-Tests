// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"
#include "nbl/ui/ICursorControl.h"

struct MeshletPush {
	float32_t4x4 viewProj; //nbl::core::matrix4SIMD is 128bit??
	constexpr static uint8_t object_type_count_max = 16;//it can go up til this struct hits the limit for push size
	uint32_t objectCount[object_type_count_max]; 
};

/* 
Renders scene texture to an offscreen framebuffer whose color attachment is then sampled into a imgui window.

Written with Nabla's UI extension and got integrated with ImGuizmo to handle scene's object translations.
*/
class UISampleApp final : public MonoWindowApplication, public BuiltinResourcesApplication
{
		using device_base_t = MonoWindowApplication;
		using asset_base_t = BuiltinResourcesApplication;

	public:
		inline UISampleApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) 
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
			device_base_t({1280,720}, EF_UNKNOWN, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;

			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			for (auto i = 0u; i<MaxFramesInFlight; i++)
			{
				if (!pool)
					return logFail("Couldn't create Command Pool!");
				if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdBufs.data()+i,1}))
					return logFail("Couldn't create Command Buffer!");
			}
			
			const uint32_t addtionalBufferOwnershipFamilies[] = {getGraphicsQueue()->getFamilyIndex()};
			//auto creator = core::make_smart_refctd_ptr<CGeometryCreator>();
			//auto cube = creator->createCube({ 1.f,1.f,1.f });
			//id like to combine all the vertices into 1 buffer but given how it's set up, thats out of scope
			//cube->getPositionView();


			m_scene = CGeometryCreatorScene::create(
				{
					.transferQueue = getTransferUpQueue(),
					.utilities = m_utils.get(),
					.logger = m_logger.get(),
					.addtionalBufferOwnershipFamilies = addtionalBufferOwnershipFamilies
				},
				CSimpleDebugRenderer::DefaultPolygonGeometryPatch
			);
			for(uint8_t i = 0; i < m_scene->getInitParams().geometries.size(); i++){
				auto const& geom = m_scene->getInitParams().geometries[i];
				printf("%s - %zu - %zu\n", m_scene->getInitParams().geometryNames[i].c_str(), geom->getVertexReferenceCount(), geom->getIndexCount());				
			}
			
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
						/*.loadOp = */{IGPURenderpass::LOAD_OP::CLEAR},
						/*.storeOp = */{IGPURenderpass::STORE_OP::STORE},
						/*.initialLayout = */{IGPUImage::LAYOUT::UNDEFINED},
						/*.finalLayout = */{IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}
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
						/*.loadOp = */IGPURenderpass::LOAD_OP::CLEAR,
						/*.storeOp = */IGPURenderpass::STORE_OP::STORE,
						/*.initialLayout = */IGPUImage::LAYOUT::UNDEFINED,
						/*.finalLayout = */ IGPUImage::LAYOUT::READ_ONLY_OPTIMAL // ImGUI shall read
					}},
					IGPURenderpass::SCreationParams::ColorAttachmentsEnd
				};
				params.colorAttachments = colorAttachments;
				IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
					{},
					IGPURenderpass::SCreationParams::SubpassesEnd
				};
				subpasses[0].depthStencilAttachment = {{.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}}};
				subpasses[0].colorAttachments[0] = {.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};
				params.subpasses = subpasses;
				
				const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
					// wipe-transition of Color to ATTACHMENT_OPTIMAL and depth
					{
						.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
						.dstSubpass = 0,
						.memoryBarrier = {
							// last place where the depth can get modified in previous frame, `COLOR_ATTACHMENT_OUTPUT_BIT` is implicitly later
							// while color is sampled by ImGUI
							.srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT|PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
							// don't want any writes to be available, as we are clearing both attachments
							.srcAccessMask = ACCESS_FLAGS::NONE,
							// destination needs to wait as early as possible
							// TODO: `COLOR_ATTACHMENT_OUTPUT_BIT` shouldn't be needed, because its a logically later stage, see TODO in `ECommonEnums.h`
							.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT|PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
							// because depth and color get cleared first no read mask
							.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
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
							.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT|PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
							// but we only care about the availability-visibility chain between renderpass and imgui 
							.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT
						}
						// leave view offsets and flags default
					},
					IGPURenderpass::SCreationParams::DependenciesEnd
				};
				params.dependencies = {};
				m_renderpass = m_device->createRenderpass(std::move(params));
				if (!m_renderpass)
					return logFail("Failed to create Scene Renderpass!");
			}
			const auto& geometries = m_scene->getInitParams().geometries;
			m_renderer = CSimpleDebugRenderer::create(m_assetMgr.get(),m_renderpass.get(),0,{&geometries.front().get(),geometries.size()});
			// special case
			{
				//const auto& pipelines = m_renderer->getInitParams().pipelines;
				auto ix = 0u;
				for (const auto& name : m_scene->getInitParams().geometryNames)
				{
					if (name=="Cone")
						//m_renderer->getGeometry(ix).pipeline = pipelines[CSimpleDebugRenderer::SInitParams::PipelineType::Cone];
					ix++;
				}
			}
			// we'll only display one thing at a time
			//m_renderer->m_instances.resize(1);

			// Create ImGUI
			{
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				ext::imgui::UI::SCreationParameters params = {};
				params.resources.texturesInfo = {.setIx=0u,.bindingIx=TexturesImGUIBindingIndex};
				params.resources.samplersInfo = {.setIx=0u,.bindingIx=1u};
				params.utilities = m_utils;
				params.transfer = getTransferUpQueue();
				params.pipelineLayout = ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(),params.resources.texturesInfo,params.resources.samplersInfo,MaxImGUITextures);
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
					const auto* layout = imgui->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
					auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT,{&layout,1});
					auto ds = pool->createDescriptorSet(smart_refctd_ptr<const IGPUDescriptorSetLayout>(layout));
					interface.subAllocDS = make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(ds));
					if (!interface.subAllocDS)
						return logFail("Failed to create the descriptor set");
					// make sure Texture Atlas slot is taken for eternity
					{
						auto dummy = SubAllocatedDescriptorSet::invalid_value;
						interface.subAllocDS->multi_allocate(0,1,&dummy);
						assert(dummy==ext::imgui::UI::FontAtlasTexId);
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
					if (!m_device->updateDescriptorSets({&write,1},{}))
						return logFail("Failed to write the descriptor set");
				}
				imgui->registerListener([this](){interface();});
			}

			//create meshlet pipeline
			CreateMeshPipelines();

			interface.camera.mapKeysToArrows();

			onAppInitializedFinish();
			return true;
		}

		smart_refctd_ptr<IGPUDescriptorSetLayout> BuildMeshletDSLayout() const {
			smart_refctd_ptr<IGPUDescriptorSetLayout> ret;
			using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
			const IGPUDescriptorSetLayout::SBinding bindings[] =
			{
				{
					.binding = 0,
					.type = IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER,
					// need this trifecta of flags for `SubAllocatedDescriptorSet` to accept the binding as suballocatable
					.createFlags = binding_flags_t::ECF_UPDATE_AFTER_BIND_BIT | binding_flags_t::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT | binding_flags_t::ECF_PARTIALLY_BOUND_BIT,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_MESH,
					.count = UINT16_MAX
				},
				{
					.binding = 1,
					.type = IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
					.createFlags = binding_flags_t::ECF_UPDATE_AFTER_BIND_BIT | binding_flags_t::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_TASK | IShader::E_SHADER_STAGE::ESS_MESH,
					.count = 1
				}
			};
			ret = m_device->createDescriptorSetLayout(bindings);
			if (!ret) {
				m_logger->log("Could not create descriptor set layout!", ILogger::ELL_ERROR);
				return nullptr;
			}
			return ret;
		}

		std::array<const core::smart_refctd_ptr<nbl::asset::IShader>, 3> CreateTestShader() const {


			auto loadCompileAndCreateShader = [&](const std::string& relPath, hlsl::ShaderStage stage, std::span<const asset::IShaderCompiler::SMacroDefinition> extraDefines) -> smart_refctd_ptr<IShader>
				{
					IAssetLoader::SAssetLoadParams lp = {};
					lp.logger = m_logger.get();
					lp.workingDirectory = ""; // virtual root
					auto assetBundle = m_assetMgr->getAsset(relPath, lp);
					const auto assets = assetBundle.getContents();
					if (assets.empty()){
						printf("asset was empty - %s\n", relPath.c_str());
						return nullptr;
					}

					// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
					auto sourceRaw = IAsset::castDown<IShader>(assets[0]);
					if (!sourceRaw){
						printf("source raw was nullptr - %s\n", relPath.c_str());
						return nullptr;
					}

					nbl::video::ILogicalDevice::SShaderCreationParameters creationParams{
						.source = sourceRaw.get(),
						.optimizer = nullptr,
						.readCache = nullptr,
						.writeCache = nullptr,
						.extraDefines = extraDefines,
						.stage = stage
					};

					auto ret = m_device->compileShader(creationParams);
					if (ret.get() == nullptr) {
						printf("failed to compile shader - %s\n", relPath.c_str());
					}
					//m_assetMgr->removeAssetFromCache(assetBundle);
					//return nullptr;
					//i dont think that ^ was working
					return ret;
			};
			constexpr uint32_t WorkgroupSize = 64;
			constexpr uint32_t ObjectCount = WorkgroupSize;
			constexpr uint32_t InstanceCount = WorkgroupSize;
			const string WorkgroupSizeAsStr = std::to_string(WorkgroupSize);
			const string ObjectCountAsStr = std::to_string(ObjectCount);
			const string InstanceCountAsStr = std::to_string(InstanceCount);

			const IShaderCompiler::SMacroDefinition WorkgroupSizeDefine = { "WORKGROUP_SIZE",WorkgroupSizeAsStr };
			const IShaderCompiler::SMacroDefinition ObjectCountDefine = { "OBJECT_COUNT", ObjectCountAsStr };
			const IShaderCompiler::SMacroDefinition InstanceCountDefine = { "INSTANCE_COUNT", InstanceCountAsStr };

			const IShaderCompiler::SMacroDefinition meshArray[] = {WorkgroupSizeDefine, ObjectCountDefine, InstanceCountDefine};
			return {
				loadCompileAndCreateShader("app_resources/geom.task.hlsl", IShader::E_SHADER_STAGE::ESS_TASK, { meshArray }),
				loadCompileAndCreateShader("app_resources/geom.mesh.hlsl", IShader::E_SHADER_STAGE::ESS_MESH, { meshArray }),
				loadCompileAndCreateShader("app_resources/geom.frag.hlsl", IShader::E_SHADER_STAGE::ESS_FRAGMENT, {})
			};
		}
		
		bool CreateMeshPipelines() {
			//referencing example 10 for this
			//and referencing CSimpleDebugRenderer

			auto shaders = CreateTestShader();
			auto dsLayout = BuildMeshletDSLayout();
			{//descriptorset
				auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, { &dsLayout.get(),1 });
				auto ds = pool->createDescriptorSet(std::move(dsLayout));
				if (!ds) {
					m_logger->log("Could not descriptor set!", ILogger::ELL_ERROR);
					return false;
				}
				meshlet_subAllocDS = make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(ds));
			}

			{
				const SPushConstantRange ranges[] = { {
					.stageFlags = hlsl::ShaderStage::ESS_TASK | hlsl::ShaderStage::ESS_MESH,
					.offset = 0,
					.size = sizeof(MeshletPush),
				} }; 
				
				meshletLayout = m_device->createPipelineLayout(ranges, smart_refctd_ptr<const IGPUDescriptorSetLayout>(meshlet_subAllocDS->getDescriptorSet()->getLayout()));
				IGPUMeshPipeline::SCreationParams params = {};
				params.layout = meshletLayout.get();
				params.renderpass = m_renderpass.get();
				params.cached.subpassIx = 0;

				params.taskShader.shader = shaders[0].get();
				params.taskShader.entryPoint = "main";
				params.taskShader.entries = nullptr;
				params.taskShader.requiredSubgroupSize = static_cast<IPipelineBase::SUBGROUP_SIZE>(4); //ill need to adjust this probably


				params.meshShader.shader = shaders[1].get();
				params.meshShader.entryPoint = "main";
				params.meshShader.entries = nullptr;
				params.meshShader.requiredSubgroupSize = static_cast<IPipelineBase::SUBGROUP_SIZE>(5); //ill need to adjust this probably

				params.fragmentShader = { .shader = shaders[2].get(), .entryPoint = "main"};


				params.cached.requireFullSubgroups = true;
				params.cached.rasterization.faceCullingMode = EFCM_NONE; //maybe change this? i was a bit limited in example 61

				if (!m_device->createMeshPipelines(nullptr, { &params, 1 }, &meshletPipeline)) {
					logFail("Failed to create mesh pipeline!\n");
				}
			}


		}



		virtual inline bool onAppTerminated()
		{
			SubAllocatedDescriptorSet::value_type fontAtlasDescIx = ext::imgui::UI::FontAtlasTexId;
			IGPUDescriptorSet::SDropDescriptorSet dummy[1];
			interface.subAllocDS->multi_deallocate(dummy,TexturesImGUIBindingIndex,1,&fontAtlasDescIx);
			return device_base_t::onAppTerminated();
		}

		inline IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override
		{
			// CPU events
			update(nextPresentationTimestamp);

			const auto& virtualWindowRes = interface.sceneResolution;
			if (!m_framebuffer || m_framebuffer->getCreationParameters().width!=virtualWindowRes[0] || m_framebuffer->getCreationParameters().height!=virtualWindowRes[1])
				recreateFramebuffer(virtualWindowRes);

			const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

			auto* const cb = m_cmdBufs.data()[resourceIx].get();
			cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			// clear to black for both things
			const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
			if (m_framebuffer)
			{
				cb->beginDebugMarker("UISampleApp Scene Frame");
				{
					const IGPUCommandBuffer::SClearDepthStencilValue farValue = { .depth = 0.f };
					const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo =
					{
						.framebuffer = m_framebuffer.get(),
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
				UpdateScene(cb);
				cb->endRenderPass();
				cb->endDebugMarker();
			}
			{
				cb->beginDebugMarker("UISampleApp IMGUI Frame");
				{ //begin imgui subpass
					auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
					const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo = {
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
					cb->bindDescriptorSets(EPBP_GRAPHICS,pipeline->getLayout(),imgui->getCreationParameters().resources.texturesInfo.setIx,1u,&ds);
					// a timepoint in the future to release streaming resources for geometry
					const ISemaphore::SWaitInfo drawFinished = {.semaphore=m_semaphore.get(),.value=m_realFrameIx+1u};
					if (!imgui->render(cb,drawFinished))
					{
						m_logger->log("TODO: need to present acquired image before bailing because its already acquired.",ILogger::ELL_ERROR);
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


			m_window->setCaption("[Nabla Engine] Mesh Shader Demo");
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

		void UpdateScene(nbl::video::IGPUCommandBuffer* cb) {
			float32_t3x4 viewMatrix;
			float32_t4x4 viewProjMatrix;
			// TODO: get rid of legacy matrices //<-- camera.getViewMatrix returns matrix3x4SIMD
			{
				const auto& camera = interface.camera;
				memcpy(&viewMatrix, camera.getViewMatrix().pointer(), sizeof(viewMatrix));
				memcpy(&viewProjMatrix, camera.getConcatenatedMatrix().pointer(), sizeof(viewProjMatrix));
			}
			const auto viewParams = CSimpleDebugRenderer::SViewParams(viewMatrix, viewProjMatrix);

			// tear down scene every frame
			//auto& instance = m_renderer->m_instances[0];
			//memcpy(&instance.world, &interface.model, sizeof(instance.world));
			//instance.packedGeo = m_renderer->getGeometries().data() + interface.gcIndex;
			//m_renderer->render(cb, viewParams);

			//MeshPushConstant mPushConstant = { interface.camera.getConcatenatedMatrix(), cubeCount, coneCount, tubeCount };
		}


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

							if (e.type==nbl::ui::SMouseEvent::EET_SCROLL)// && m_renderer)
							{
								interface.gcIndex += int16_t(core::sign(e.scrollEvent.verticalScroll));
								//interface.gcIndex = core::clamp(interface.gcIndex,0ull,m_renderer->getGeometries().size()-1);
							}
						}
					},
					m_logger.get()
				);
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

			interface.objectName = m_scene->getInitParams().geometryNames[0];
			interface.imGUI->update(params);
		}

		void recreateFramebuffer(const uint16_t2 resolution)
		{
			auto createImageAndView = [&](E_FORMAT format)->smart_refctd_ptr<IGPUImageView>
			{
				auto image = m_device->createImage({{
					.type = IGPUImage::ET_2D,
					.samples = IGPUImage::ESCF_1_BIT,
					.format = format,
					.extent = {resolution.x,resolution.y,1},
					.mipLevels = 1,
					.arrayLayers = 1,
					.usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT|IGPUImage::EUF_SAMPLED_BIT
				}});
				if (!m_device->allocate(image->getMemoryReqs(),image.get()).isValid())
					return nullptr;
				IGPUImageView::SCreationParams params = {
					.image = std::move(image),
					.viewType = IGPUImageView::ET_2D,
					.format = format
				};
				params.subresourceRange.aspectMask = isDepthOrStencilFormat(format) ? IGPUImage::EAF_DEPTH_BIT:IGPUImage::EAF_COLOR_BIT;
				return m_device->createImageView(std::move(params));
			};
			
			smart_refctd_ptr<IGPUImageView> colorView;
			// detect window minimization
			if (resolution.x<0x4000 && resolution.y<0x4000)
			{
				colorView = createImageAndView(finalSceneRenderFormat);
				auto depthView = createImageAndView(sceneRenderDepthFormat);
				m_framebuffer = m_device->createFramebuffer({ {
					.renderpass = m_renderpass,
					.depthStencilAttachments = &depthView.get(),
					.colorAttachments = &colorView.get(),
					.width = resolution.x,
					.height = resolution.y
				}});
			}
			else
				m_framebuffer = nullptr;

			// release previous slot and its image
			interface.subAllocDS->multi_deallocate(0,1,&interface.renderColorViewDescIndex,{.semaphore=m_semaphore.get(),.value=m_realFrameIx});
			//
			if (colorView)
			{
				interface.subAllocDS->multi_allocate(0,1,&interface.renderColorViewDescIndex);
				// update descriptor set
				IGPUDescriptorSet::SDescriptorInfo info = {};
				info.desc = colorView;
				info.info.image.imageLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
				const IGPUDescriptorSet::SWriteDescriptorSet write = {
					.dstSet = interface.subAllocDS->getDescriptorSet(),
					.binding = TexturesImGUIBindingIndex,
					.arrayElement = interface.renderColorViewDescIndex,
					.count = 1,
					.info = &info
				};
				m_device->updateDescriptorSets({&write,1},{});
			}
			interface.transformParams.sceneTexDescIx = interface.renderColorViewDescIndex;
		}

		inline void beginRenderpass(IGPUCommandBuffer* cb, const IGPUCommandBuffer::SRenderpassBeginInfo& info)
		{
			cb->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
			cb->setScissor(0,1,&info.renderArea);
			const SViewport viewport = {
				.x = 0,
				.y = 0,
				.width = static_cast<float>(info.renderArea.extent.width),
				.height = static_cast<float>(info.renderArea.extent.height)
			};
			cb->setViewport(0u,1u,&viewport);
		}

		// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
		constexpr static inline uint32_t MaxFramesInFlight = 3u;
		constexpr static inline auto sceneRenderDepthFormat = EF_D32_SFLOAT;
		constexpr static inline auto finalSceneRenderFormat = EF_R8G8B8A8_SRGB;
		constexpr static inline auto TexturesImGUIBindingIndex = 0u;
		// we create the Descriptor Set with a few slots extra to spare, so we don't have to `waitIdle` the device whenever ImGUI virtual window resizes
		constexpr static inline auto MaxImGUITextures = 2u+MaxFramesInFlight;

		//
		smart_refctd_ptr<CGeometryCreatorScene> m_scene;
		smart_refctd_ptr<IGPURenderpass> m_renderpass;
		smart_refctd_ptr<IGPUFramebuffer> m_framebuffer;
		smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;
		//
		smart_refctd_ptr<ISemaphore> m_semaphore;
		uint64_t m_realFrameIx = 0;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,MaxFramesInFlight> m_cmdBufs;
		//
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		core::smart_refctd_ptr<video::SubAllocatedDescriptorSet> meshlet_subAllocDS;
		smart_refctd_ptr<IGPUPipelineLayout> meshletLayout;
		smart_refctd_ptr<IGPUMeshPipeline> meshletPipeline;
		// UI stuff
		struct CInterface
		{
			void operator()()
			{
				ImGuiIO& io = ImGui::GetIO();

				{
					matrix4SIMD projection;

					if (isPerspective)
						if(isLH)
							projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y, zNear, zFar);
						else
							projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y, zNear, zFar);
					else
					{
						float viewHeight = viewWidth * io.DisplaySize.y / io.DisplaySize.x;

						if(isLH)
							projection = matrix4SIMD::buildProjectionMatrixOrthoLH(viewWidth, viewHeight, zNear, zFar);
						else
							projection = matrix4SIMD::buildProjectionMatrixOrthoRH(viewWidth, viewHeight, zNear, zFar);
					}
					camera.setProjectionMatrix(projection);
				}

				ImGuizmo::SetOrthographic(false);
				ImGuizmo::BeginFrame();

				ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
				ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

				// create a window and insert the inspector
				ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
				ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
				ImGui::Begin("Editor");

				if (ImGui::Button("reload mesh shader")) {
					//printf("test shader result - %d\n", CreateTestShaderFuncPtr());
				}

				if (ImGui::RadioButton("Full view", !transformParams.useWindow))
					transformParams.useWindow = false;

				ImGui::SameLine();

				if (ImGui::RadioButton("Window", transformParams.useWindow))
					transformParams.useWindow = true;

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
				ImGui::Checkbox("Enable camera movement", &move);
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
					core::vectorSIMDf cameraPosition(cosf(camYAngle)* cosf(camXAngle)* transformParams.camDistance, sinf(camXAngle)* transformParams.camDistance, sinf(camYAngle)* cosf(camXAngle)* transformParams.camDistance);
					core::vectorSIMDf cameraTarget(0.f, 0.f, 0.f);
					const static core::vectorSIMDf up(0.f, 1.f, 0.f);

					camera.setPosition(cameraPosition);
					camera.setTarget(cameraTarget);
					camera.setBackupUpVector(up);

					camera.recomputeViewMatrix();
				}
				firstFrame = false;

				ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);
				if (ImGuizmo::IsUsing())
				{
					ImGui::Text("Using gizmo");
				}
				else {
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

// TODO: do all computation using `hlsl::matrix` and its `hlsl::float32_tNxM` aliases
				static struct
				{
					core::matrix4SIMD view, projection, model;
				} imguizmoM16InOut;

				ImGuizmo::SetID(0u);

				imguizmoM16InOut.view = core::transpose(matrix4SIMD(camera.getViewMatrix()));
				imguizmoM16InOut.projection = core::transpose(camera.getProjectionMatrix());
				if (currentTransform >= 0 && currentTransform < transforms.size()) {
					imguizmoM16InOut.model = core::transpose(matrix4SIMD(transforms[currentTransform]));
				}
				{
					transformParams.editTransformDecomposition = true;
					static TransformWidget transformWidget{};
					transformWidget.Update(imguizmoM16InOut.view.pointer(), imguizmoM16InOut.projection.pointer(), imguizmoM16InOut.model.pointer(), transformParams);
					sceneResolution = widgetBox.zw;
				}

				if (currentTransform >= 0 && currentTransform < transforms.size()) {
					transforms[currentTransform] = core::transpose(imguizmoM16InOut.model).extractSub3x4();
				}
				// to Nabla + update camera & model matrices
// TODO: make it more nicely, extract:
// - Position by computing inverse of the view matrix and grabbing its translation
// - Target from 3rd row without W component of view matrix multiplied by some arbitrary distance value (can be the length of position from origin) and adding the position
// But then set the view matrix this way anyway, because up-vector may not be compatible
				const auto& view = camera.getViewMatrix();
				const_cast<core::matrix3x4SIMD&>(view) = core::transpose(imguizmoM16InOut.view).extractSub3x4(); // a hack, correct way would be to use inverse matrix and get position + target because now it will bring you back to last position & target when switching from gizmo move to manual move (but from manual to gizmo is ok)
				// update concatanated matrix
				const auto& projection = camera.getProjectionMatrix();
				camera.setProjectionMatrix(projection);

				// object meta display
				{
					ImGui::Begin("Object Counts");
					ImGui::Text("object count - cube[%d] - cone[%d] - tube[%d]", cubeCount, coneCount, tubeCount);


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

					if (currentTransform >= 0 && currentTransform < transforms.size()) {
						addMatrixTable("Model Matrix", "ModelMatrixTable", 3, 4, transforms[currentTransform].pointer());
					}
					addMatrixTable("Camera View Matrix", "ViewMatrixTable", 3, 4, view.pointer());
					addMatrixTable("Camera View Projection Matrix", "ViewProjectionMatrixTable", 4, 4, projection.pointer(), false);

					ImGui::End();
				}
			}

			smart_refctd_ptr<ext::imgui::UI> imGUI;
			// descriptor set
			smart_refctd_ptr<SubAllocatedDescriptorSet> subAllocDS;
			SubAllocatedDescriptorSet::value_type renderColorViewDescIndex = SubAllocatedDescriptorSet::invalid_value;
			//
			Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
			// mutables
			int32_t currentTransform = -1;
			std::vector<core::matrix3x4SIMD> transforms;

			std::string_view objectName;
			TransformRequestParams transformParams;
			uint16_t2 sceneResolution = {1280,720};
			uint16_t4 widgetBox;
			float fov = 60.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
			float viewWidth = 10.f;
			float camYAngle = 165.f / 180.f * 3.14159f;
			float camXAngle = 32.f / 180.f * 3.14159f;
			uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed
			bool isPerspective = true, isLH = true, flipGizmoY = true, move = false;
			bool firstFrame = true;

			uint32_t cubeCount = 0;
			uint32_t coneCount = 0;
			uint32_t tubeCount = 0;
		} interface;
};

NBL_MAIN_FUNC(UISampleApp)