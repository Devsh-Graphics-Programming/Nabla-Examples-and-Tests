// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"
#include "keysmapping.hpp"
#include "camera/CCubeProjection.hpp"
#include "glm/glm/ext/matrix_clip_space.hpp" // TODO: TESTING

// FPS Camera, TESTS
using camera_t = CFPSCamera<matrix_precision_t>;
using projection_t = IProjection<matrix<matrix_precision_t, 4u, 4u>>; // TODO: temporary -> projections will own/reference cameras

constexpr IGPUImage::SSubresourceRange TripleBufferUsedSubresourceRange = 
{
	.aspectMask = IGPUImage::EAF_COLOR_BIT,
	.baseMipLevel = 0,
	.levelCount = 1,
	.baseArrayLayer = 0,
	.layerCount = 1
};

class CUIEventCallback : public nbl::video::ISmoothResizeSurface::ICallback // I cannot use common CEventCallback because I MUST inherit this callback in order to use smooth resize surface with window callback (for my input events)
{
public:
	CUIEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(m_inputSystem)), m_logger(std::move(logger)) {}
	CUIEventCallback() {}

	void setLogger(nbl::system::logger_opt_smart_ptr& logger)
	{
		m_logger = logger;
	}
	void setInputSystem(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem)
	{
		m_inputSystem = std::move(m_inputSystem);
	}
private:

	void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
	{
		m_logger.log("A mouse %p has been connected", nbl::system::ILogger::ELL_INFO, mch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_mouse, std::move(mch));
	}
	void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
	{
		m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse, mch);
	}
	void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
	{
		m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard, std::move(kbch));
	}
	void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
	{
		m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard, kbch);
	}

private:
	nbl::core::smart_refctd_ptr<InputSystem> m_inputSystem = nullptr;
	nbl::system::logger_opt_smart_ptr m_logger = nullptr;
};

class CSwapchainResources final : public ISmoothResizeSurface::ISwapchainResources
{
public:
	// Because we blit to the swapchain image asynchronously, we need a queue which can not only present but also perform graphics commands.
	// If we for example used a compute shader to tonemap and MSAA resolve, we'd request the COMPUTE_BIT here. 
	constexpr static inline IQueue::FAMILY_FLAGS RequiredQueueFlags = IQueue::FAMILY_FLAGS::GRAPHICS_BIT;

protected:
	// We can return `BLIT_BIT` here, because the Source Image will be already in the correct layout to be used for the present
	inline core::bitflag<asset::PIPELINE_STAGE_FLAGS> getTripleBufferPresentStages() const override { return asset::PIPELINE_STAGE_FLAGS::BLIT_BIT; }

	inline bool tripleBufferPresent(IGPUCommandBuffer* cmdbuf, const ISmoothResizeSurface::SPresentSource& source, const uint8_t imageIndex, const uint32_t qFamToAcquireSrcFrom) override
	{
		bool success = true;
		auto acquiredImage = getImage(imageIndex);

		// Ownership of the Source Blit Image, not the Swapchain Image
		const bool needToAcquireSrcOwnership = qFamToAcquireSrcFrom != IQueue::FamilyIgnored;
		// Should never get asked to transfer ownership if the source is concurrent sharing
		assert(!source.image->getCachedCreationParams().isConcurrentSharing() || !needToAcquireSrcOwnership);

		const auto blitDstLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo = {};

		// barrier before to transition the swapchain image layout
		using image_barrier_t = decltype(depInfo.imgBarriers)::element_type;
		const image_barrier_t preBarriers[2] = {
			{
				.barrier = {
					.dep = {
						.srcStageMask = asset::PIPELINE_STAGE_FLAGS::NONE, // acquire isn't a stage
						.srcAccessMask = asset::ACCESS_FLAGS::NONE, // performs no accesses
						.dstStageMask = asset::PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.dstAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT
					}
				},
				.image = acquiredImage,
				.subresourceRange = {
					.aspectMask = IGPUImage::EAF_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1
				},
				.oldLayout = IGPUImage::LAYOUT::UNDEFINED, // I do not care about previous contents of the swapchain
				.newLayout = blitDstLayout
			},
			{
				.barrier = {
					.dep = {
				// when acquiring ownership the source access masks don't matter
				.srcStageMask = asset::PIPELINE_STAGE_FLAGS::NONE,
				// Acquire must Happen-Before Semaphore wait, but neither has a true stage so NONE here
				// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
				// If no ownership acquire needed then this dep info won't be used at all
				.srcAccessMask = asset::ACCESS_FLAGS::NONE,
				.dstStageMask = asset::PIPELINE_STAGE_FLAGS::BLIT_BIT,
				.dstAccessMask = asset::ACCESS_FLAGS::TRANSFER_READ_BIT
			},
			.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
			.otherQueueFamilyIndex = qFamToAcquireSrcFrom
		},
		.image = source.image,
		.subresourceRange = TripleBufferUsedSubresourceRange
			// no layout transition, already in the correct layout for the blit
		}
		};
		// We only barrier the source image if we need to acquire ownership, otherwise thanks to Timeline Semaphores all sync is good
		depInfo.imgBarriers = { preBarriers,needToAcquireSrcOwnership ? 2ull : 1ull };
		success &= cmdbuf->pipelineBarrier(asset::EDF_NONE, depInfo);

		// TODO: Implement scaling modes other than a plain STRETCH, and allow for using subrectangles of the initial contents
		{
			const auto srcOffset = source.rect.offset;
			const auto srcExtent = source.rect.extent;
			const auto dstExtent = acquiredImage->getCreationParameters().extent;
			const IGPUCommandBuffer::SImageBlit regions[1] = { {
				.srcMinCoord = {static_cast<uint32_t>(srcOffset.x),static_cast<uint32_t>(srcOffset.y),0},
				.srcMaxCoord = {srcExtent.width,srcExtent.height,1},
				.dstMinCoord = {0,0,0},
				.dstMaxCoord = {dstExtent.width,dstExtent.height,1},
				.layerCount = acquiredImage->getCreationParameters().arrayLayers,
				.srcBaseLayer = 0,
				.dstBaseLayer = 0,
				.srcMipLevel = 0
			} };
			success &= cmdbuf->blitImage(source.image, IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL, acquiredImage, blitDstLayout, regions, IGPUSampler::ETF_LINEAR);
		}

		// Barrier after, note that I don't care about preserving the contents of the Triple Buffer when the Render queue starts writing to it again.
		// Therefore no ownership release, and no layout transition.
		const image_barrier_t postBarrier[1] = {
			{
				.barrier = {
				// When transitioning the image to VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR or VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, there is no need to delay subsequent processing,
				// or perform any visibility operations (as vkQueuePresentKHR performs automatic visibility operations).
				// To achieve this, the dstAccessMask member of the VkImageMemoryBarrier should be set to 0, and the dstStageMask parameter should be set to VK_PIPELINE_STAGE_2_NONE
				.dep = preBarriers[0].barrier.dep.nextBarrier(asset::PIPELINE_STAGE_FLAGS::NONE,asset::ACCESS_FLAGS::NONE)
			},
			.image = preBarriers[0].image,
			.subresourceRange = preBarriers[0].subresourceRange,
			.oldLayout = blitDstLayout,
			.newLayout = IGPUImage::LAYOUT::PRESENT_SRC
		}
		};
		depInfo.imgBarriers = postBarrier;
		success &= cmdbuf->pipelineBarrier(asset::EDF_NONE, depInfo);

		return success;
	}
};

/*
	Renders scene texture to an offline
	framebuffer which color attachment
	is then sampled into a imgui window.

	Written with Nabla, it's UI extension
	and got integrated with ImGuizmo to 
	handle scene's object translations.
*/

class UISampleApp final : public examples::SimpleWindowedApplication
{
	using base_t = examples::SimpleWindowedApplication;
	using clock_t = std::chrono::steady_clock;

	//_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_W = 1280, WIN_H = 720;

	constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

	public:
		using base_t::base_t;

		inline UISampleApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) 
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		// Will get called mid-initialization, via `filterDevices` between when the API Connection is created and Physical Device is chosen
		core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			// So let's create our Window and Surface then!
			if (!m_surface)
			{
				{
					const auto dpyInfo = m_winMgr->getPrimaryDisplayInfo();
					auto windowCallback = core::make_smart_refctd_ptr<CUIEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));

					IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<nbl::video::ISmoothResizeSurface::ICallback>();
					params.width = dpyInfo.resX;
					params.height = dpyInfo.resY;
					params.x = 32;
					params.y = 32;
					params.flags = IWindow::ECF_INPUT_FOCUS | IWindow::ECF_CAN_RESIZE | IWindow::ECF_CAN_MAXIMIZE | IWindow::ECF_CAN_MINIMIZE;
					params.windowCaption = "[Nabla Engine] UI App";
					params.callback = windowCallback;

					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}
				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSmoothResizeSurface<CSwapchainResources>::create(std::move(surface));
			}

			if (m_surface)
			{
				m_window->getManager()->maximize(m_window.get());
				return { {m_surface->getSurface()/*,EQF_NONE*/} };
			}
			
			return {};
		}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Create imput system
			m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			// Create asset manager
			m_assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(m_system));

			// First create the resources that don't depend on a swapchain
			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			// The nice thing about having a triple buffer is that you don't need to do acrobatics to account for the formats available to the surface.
			// You can transcode to the swapchain's format while copying, and I actually recommend to do surface rotation, tonemapping and OETF application there.
			const auto format = asset::EF_R8G8B8A8_SRGB;
			// Could be more clever and use the copy Triple Buffer to Swapchain as an opportunity to do a MSAA resolve or something
			const auto samples = IGPUImage::ESCF_1_BIT;

			// Create the renderpass
			{
				const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
					{{
						{
							.format = format,
							.samples = samples,
							.mayAlias = false
						},
					/*.loadOp = */IGPURenderpass::LOAD_OP::CLEAR,
					/*.storeOp = */IGPURenderpass::STORE_OP::STORE,
					/*.initialLayout = */IGPUImage::LAYOUT::UNDEFINED, // because we clear we don't care about contents when we grab the triple buffer img again
					/*.finalLayout = */IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL // put it already in the correct layout for the blit operation
				}},
				IGPURenderpass::SCreationParams::ColorAttachmentsEnd
				};
				IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
					{},
					IGPURenderpass::SCreationParams::SubpassesEnd
				};
				subpasses[0].colorAttachments[0] = { .render = {.attachmentIndex = 0,.layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL} };
				// We actually need external dependencies to ensure ordering of the Implicit Layout Transitions relative to the semaphore signals
				IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
					// wipe-transition to ATTACHMENT_OPTIMAL
					{
						.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
						.dstSubpass = 0,
						.memoryBarrier = {
						// we can have NONE as Sources because the semaphore wait is ALL_COMMANDS
						// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
						.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
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
							// we can have NONE as the Destinations because the semaphore signal is ALL_COMMANDS
							// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
						}
					// leave view offsets and flags default
				},
				IGPURenderpass::SCreationParams::DependenciesEnd
				};

				IGPURenderpass::SCreationParams params = {};
				params.colorAttachments = colorAttachments;
				params.subpasses = subpasses;
				params.dependencies = dependencies;
				m_renderpass = m_device->createRenderpass(params);
				if (!m_renderpass)
					return logFail("Failed to Create a Renderpass!");
			}

			// We just live life in easy mode and have the Swapchain Creation Parameters get deduced from the surface.
			// We don't need any control over the format of the swapchain because we'll be only using Renderpasses this time!
			// TODO: improve the queue allocation/choice and allocate a dedicated presentation queue to improve responsiveness and race to present.
			if (!m_surface || !m_surface->init(m_surface->pickQueue(m_device.get()), std::make_unique<CSwapchainResources>(), {}))
				return logFail("Failed to Create a Swapchain!");

			// Normally you'd want to recreate these images whenever the swapchain is resized in some increment, like 64 pixels or something.
			// But I'm super lazy here and will just create "worst case sized images" and waste all the VRAM I can get.
			const auto dpyInfo = m_winMgr->getPrimaryDisplayInfo();
			for (auto i = 0; i < MaxFramesInFlight; i++)
			{
				auto& image = m_tripleBuffers[i];
				{
					IGPUImage::SCreationParams params = {};
					params = asset::IImage::SCreationParams{
						.type = IGPUImage::ET_2D,
						.samples = samples,
						.format = format,
						.extent = {dpyInfo.resX,dpyInfo.resY,1},
						.mipLevels = 1,
						.arrayLayers = 1,
						.flags = IGPUImage::ECF_NONE,
						// in this example I'll be using a renderpass to clear the image, and then a blit to copy it to the swapchain
						.usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_TRANSFER_SRC_BIT
					};
					image = m_device->createImage(std::move(params));
					if (!image)
						return logFail("Failed to Create Triple Buffer Image!");

					// use dedicated allocations, we have plenty of allocations left, even on Win32
					if (!m_device->allocate(image->getMemoryReqs(), image.get()).isValid())
						return logFail("Failed to allocate Device Memory for Image %d", i);
				}
				image->setObjectDebugName(("Triple Buffer Image " + std::to_string(i)).c_str());

				// create framebuffers for the images
				{
					auto imageView = m_device->createImageView({
						.flags = IGPUImageView::ECF_NONE,
						// give it a Transfer SRC usage flag so we can transition to the Tranfer SRC layout with End Renderpass
						.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_TRANSFER_SRC_BIT,
						.image = core::smart_refctd_ptr(image),
						.viewType = IGPUImageView::ET_2D,
						.format = format
						});
					const auto& imageParams = image->getCreationParameters();
					IGPUFramebuffer::SCreationParams params = { {
						.renderpass = core::smart_refctd_ptr(m_renderpass),
						.depthStencilAttachments = nullptr,
						.colorAttachments = &imageView.get(),
						.width = imageParams.extent.width,
						.height = imageParams.extent.height,
						.layers = imageParams.arrayLayers
					} };
					m_framebuffers[i] = m_device->createFramebuffer(std::move(params));
					if (!m_framebuffers[i])
						return logFail("Failed to Create a Framebuffer for Image %d", i);
				}
			}

			// This time we'll create all CommandBuffers from one CommandPool, to keep life simple. However the Pool must support individually resettable CommandBuffers
			// because they cannot be pre-recorded because the fraembuffers/swapchain images they use will change when a swapchain recreates.
			auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data(),MaxFramesInFlight }, core::smart_refctd_ptr(m_logger)))
				return logFail("Failed to Create CommandBuffers!");

			// UI
			{
				{
					nbl::ext::imgui::UI::SCreationParameters params;
					params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
					params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
					params.assetManager = m_assetManager;
					params.pipelineCache = nullptr;
					params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, TotalUISampleTexturesAmount);
					params.renderpass = smart_refctd_ptr<IGPURenderpass>(m_renderpass);
					params.streamingBuffer = nullptr;
					params.subpassIx = 0u;
					params.transfer = getTransferUpQueue();
					params.utilities = m_utils;

					m_ui.manager = nbl::ext::imgui::UI::create(std::move(params));
				}

				if (!m_ui.manager)
					return false;

				// note that we use default layout provided by our extension, but you are free to create your own by filling nbl::ext::imgui::UI::S_CREATION_PARAMETERS::resources
				const auto* descriptorSetLayout = m_ui.manager->getPipeline()->getLayout()->getDescriptorSetLayout(0u);

				IDescriptorPool::SCreateInfo descriptorPoolInfo = {};
				descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLER)] = (uint32_t)nbl::ext::imgui::UI::DefaultSamplerIx::COUNT;
				descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)] = TotalUISampleTexturesAmount;
				descriptorPoolInfo.maxSets = 1u;
				descriptorPoolInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT;

				m_descriptorSetPool = m_device->createDescriptorPool(std::move(descriptorPoolInfo));
				assert(m_descriptorSetPool);

				m_descriptorSetPool->createDescriptorSets(1u, &descriptorSetLayout, &m_ui.descriptorSet);
				assert(m_ui.descriptorSet);

				m_ui.manager->registerListener([this]() -> void { imguiListen(); });
			}

			for (auto& projection : projections)
				projection = make_smart_refctd_ptr<projection_t>();

			// Geometry Creator Render Scenes
			{
				resources = ResourcesBundle::create(m_device.get(), m_logger.get(), getGraphicsQueue(), m_assetManager->getGeometryCreator());

				if (!resources)
				{
					m_logger->log("Could not create geometry creator gpu resources!", ILogger::ELL_ERROR);
					return false;
				}

				// TOOD: we should be able to load position & orientation from json file, support multiple cameraz
				const float32_t3 iPosition[CamerazCount] = { float32_t3{ -2.238f, 1.438f, -1.558f }, float32_t3{ -2.017f, 0.386f, 0.684f } };
				// order important for glm::quat, the ctor is GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q>::qua(T _w, T _x, T _y, T _z) but memory layout is x,y,z,w
				const glm::quat iOrientation[CamerazCount] = { glm::quat(0.888f, 0.253f, 0.368f, -0.105f), glm::quat(0.55f, 0.047f, 0.830f, -0.072f) };

				for (uint32_t i = 0u; i < cameraz.size(); ++i)
				{
					auto& camera = cameraz[i];

					// lets use key map presets to update the controller

					camera = make_smart_refctd_ptr<camera_t>(iPosition[i], iOrientation[i]);

					// init keyboard map
					camera->updateKeyboardMapping([&](auto& keys)
					{
						keys = camera->getKeyboardMappingPreset();
					});

					// init mouse map
					camera->updateMouseMapping([&](auto& keys)
					{
						keys = camera->getMouseMappingPreset();
					});

					// init imguizmo map
					camera->updateImguizmoMapping([&](auto& keys)
					{
						keys = camera->getImguizmoMappingPreset();
					});
				}

				for (uint32_t i = 0u; i < scenez.size(); ++i)
				{
					auto& scene = scenez[i];
					scene = CScene::create(smart_refctd_ptr(m_device), smart_refctd_ptr(m_logger), getGraphicsQueue(), smart_refctd_ptr(resources));

					if (!scene)
					{
						m_logger->log("Could not create geometry creator scene[%d]!", ILogger::ELL_ERROR, i);
						return false;
					}
				}
			}

			oracle.reportBeginFrameRecord();

			/*
				TESTS, TODO: remove all once finished work & integrate with the example properly
			*/

			const auto iAspectRatio = float(m_window->getWidth()) / float(m_window->getHeight());
			const auto iInvAspectRatio = float(m_window->getHeight()) / float(m_window->getWidth());

			for (uint32_t i = 0u; i < ProjectionsCount; ++i)
			{
				aspectRatio[i] = iAspectRatio;
				invAspectRatio[i] = iInvAspectRatio;
			}

			if (base_t::argv.size() >= 3 && argv[1] == "-timeout_seconds")
				timeout = std::chrono::seconds(std::atoi(argv[2].c_str()));
			start = clock_t::now();
			return true;
		}

		bool updateGUIDescriptorSet()
		{
			// UI texture atlas + our camera scene textures, note we don't create info & write pair for the font sampler because UI extension's is immutable and baked into DS layout
			static std::array<IGPUDescriptorSet::SDescriptorInfo, TotalUISampleTexturesAmount> descriptorInfo;
			static IGPUDescriptorSet::SWriteDescriptorSet writes[TotalUISampleTexturesAmount];

			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].desc = core::smart_refctd_ptr<nbl::video::IGPUImageView>(m_ui.manager->getFontAtlasView());

			descriptorInfo[OfflineSceneFirstCameraTextureIx].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			descriptorInfo[OfflineSceneFirstCameraTextureIx].desc = scenez[0]->getColorAttachment();

			descriptorInfo[OfflineSceneSecondCameraTextureIx].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			descriptorInfo[OfflineSceneSecondCameraTextureIx].desc = scenez[1]->getColorAttachment();

			for (uint32_t i = 0; i < descriptorInfo.size(); ++i)
			{
				writes[i].dstSet = m_ui.descriptorSet.get();
				writes[i].binding = 0u;
				writes[i].arrayElement = i;
				writes[i].count = 1u;
			}
			writes[nbl::ext::imgui::UI::FontAtlasTexId].info = descriptorInfo.data() + nbl::ext::imgui::UI::FontAtlasTexId;
			writes[OfflineSceneFirstCameraTextureIx].info = descriptorInfo.data() + OfflineSceneFirstCameraTextureIx;
			writes[OfflineSceneSecondCameraTextureIx].info = descriptorInfo.data() + OfflineSceneSecondCameraTextureIx;

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
				const ISemaphore::SWaitInfo cmdbufDonePending[] = {
					{
						.semaphore = m_semaphore.get(),
						.value = m_realFrameIx + 1 - framesInFlight
					}
				};
				if (m_device->blockForSemaphores(cmdbufDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

			// Predict size of next render, and bail if nothing to do
			const auto currentSwapchainExtent = m_surface->getCurrentExtent();
			if (currentSwapchainExtent.width * currentSwapchainExtent.height <= 0)
				return;
			// The extent of the swapchain might change between now and `present` but the blit should adapt nicely
			const VkRect2D currentRenderArea = { .offset = {0,0},.extent = currentSwapchainExtent };

			// You explicitly should not use `getAcquireCount()` see the comment on `m_realFrameIx`
			const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

			// We will be using this command buffer to produce the frame
			auto frame = m_tripleBuffers[resourceIx].get();
			auto cmdbuf = m_cmdBufs[resourceIx].get();

			// update CPU stuff - controllers, events, UI state
			update();

			bool willSubmit = true;
			{
				// render geometry creator scene to offline frame buffer & submit
				// TODO: OK with TRI buffer this thing is retarded now
				// (**) <- a note why bellow before submit

				for (auto scene : scenez)
				{
					scene->begin();
					{
						scene->update();
						scene->record();
						scene->end();
					}
					scene->submit(getGraphicsQueue());
				}

				willSubmit &= cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
				willSubmit &= cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				willSubmit &= cmdbuf->beginDebugMarker("UIApp Frame");
				
				const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
				const IGPUCommandBuffer::SRenderpassBeginInfo info = {
					.framebuffer = m_framebuffers[resourceIx].get(),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = nullptr,
					.renderArea = currentRenderArea
				};

				// UI renderpass
				willSubmit &= cmdbuf->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
				{
					asset::SViewport viewport;
					{
						viewport.minDepth = 1.f;
						viewport.maxDepth = 0.f;
						viewport.x = 0u;
						viewport.y = 0u;
						viewport.width = m_window->getWidth();
						viewport.height = m_window->getHeight();
					}

					willSubmit &= cmdbuf->setViewport(0u, 1u, &viewport);

					const VkRect2D currentRenderArea =
					{
						.offset = {0,0},
						.extent = {m_window->getWidth(),m_window->getHeight()}
					};

					IQueue::SSubmitInfo::SCommandBufferInfo commandBuffersInfo[] = { {.cmdbuf = cmdbuf } };

					const IGPUCommandBuffer::SRenderpassBeginInfo info =
					{
						.framebuffer = m_framebuffers[resourceIx].get(),
						.colorClearValues = &Traits::clearColor,
						.depthStencilClearValues = nullptr,
						.renderArea = currentRenderArea
					};

					nbl::video::ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };
					const auto uiParams = m_ui.manager->getCreationParameters();
					auto* pipeline = m_ui.manager->getPipeline();

					cmdbuf->bindGraphicsPipeline(pipeline);
					cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, pipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get()); // note that we use default UI pipeline layout where uiParams.resources.textures.setIx == uiParams.resources.samplers.setIx

					if (!keepRunning())
						return;

					willSubmit &= m_ui.manager->render(cmdbuf, waitInfo);
				}
				willSubmit &= cmdbuf->endRenderPass();

				// If the Rendering and Blit/Present Queues don't come from the same family we need to transfer ownership, because we need to preserve contents between them.
				auto blitQueueFamily = m_surface->getAssignedQueue()->getFamilyIndex();
				// Also should crash/error if concurrent sharing enabled but would-be-user-queue is not in the share set, but oh well.
				const bool needOwnershipRelease = cmdbuf->getQueueFamilyIndex() != blitQueueFamily && !frame->getCachedCreationParams().isConcurrentSharing();
				if (needOwnershipRelease)
				{
					const IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barrier[] = { {
						.barrier = {
							.dep = {
							// Normally I'd put `COLOR_ATTACHMENT` on the masks, but we want this to happen after Layout Transition :(
							// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
							.srcStageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
							.srcAccessMask = asset::ACCESS_FLAGS::MEMORY_READ_BITS | asset::ACCESS_FLAGS::MEMORY_WRITE_BITS,
							// For a Queue Family Ownership Release the destination access masks are irrelevant
							// and source stage mask can be NONE as long as the semaphore signals ALL_COMMANDS_BIT
							.dstStageMask = asset::PIPELINE_STAGE_FLAGS::NONE,
							.dstAccessMask = asset::ACCESS_FLAGS::NONE
						},
						.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
						.otherQueueFamilyIndex = blitQueueFamily
					},
					.image = frame,
					.subresourceRange = TripleBufferUsedSubresourceRange
						// there will be no layout transition, already done by the Renderpass End
					} };
					const IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo = { .imgBarriers = barrier };
					willSubmit &= cmdbuf->pipelineBarrier(asset::EDF_NONE, depInfo);
				}
			}
			willSubmit &= cmdbuf->end();

			// submit and present under a mutex ASAP
			if (willSubmit)
			{
				// We will signal a semaphore in the rendering queue, and await it with the presentation/blit queue
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered = 
				{
					.semaphore = m_semaphore.get(),
					.value = m_realFrameIx + 1,
					// Normally I'd put `COLOR_ATTACHMENT` on the masks, but we want to signal after Layout Transitions and optional Ownership Release
					// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
					.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
				};
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = 
				{ {
					.cmdbuf = cmdbuf
				} };
				// We need to wait on previous triple buffer blits/presents from our source image to complete
				auto* pBlitWaitValue = m_blitWaitValues.data() + resourceIx;
				auto swapchainLock = m_surface->pseudoAcquire(pBlitWaitValue);
				const IQueue::SSubmitInfo::SSemaphoreInfo blitted = 
				{
					.semaphore = m_surface->getPresentSemaphore(),
					.value = pBlitWaitValue->load(),
					// Normally I'd put `BLIT` on the masks, but we want to wait before Implicit Layout Transitions and optional Implicit Ownership Acquire
					// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
					.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
				};
				const IQueue::SSubmitInfo submitInfos[1] = 
				{
					{
						.waitSemaphores = {&blitted,1},
						.commandBuffers = cmdbufs,
						.signalSemaphores = {&rendered,1}
					}
				};

				// (**) -> wait on offline framebuffers
				{
					const nbl::video::ISemaphore::SWaitInfo waitInfos[] =
					{ 
						// wait for first camera scene view fb
						{
							.semaphore = scenez[0]->semaphore.progress.get(),
							.value = scenez[0]->semaphore.finishedValue
						},
						// and second one too
						{
							.semaphore = scenez[1]->semaphore.progress.get(),
							.value = scenez[1]->semaphore.finishedValue
						},
					};

					m_device->blockForSemaphores(waitInfos);
					updateGUIDescriptorSet();
				}

				if (getGraphicsQueue()->submit(submitInfos) != IQueue::RESULT::SUCCESS)
					return;

				m_realFrameIx++;

				// only present if there's successful content to show
				const ISmoothResizeSurface::SPresentInfo presentInfo = {
					{
						.source = {.image = frame,.rect = currentRenderArea},
						.waitSemaphore = rendered.semaphore,
						.waitValue = rendered.value,
						.pPresentSemaphoreWaitValue = pBlitWaitValue,
					},
					// The Graphics Queue will be the the most recent owner just before it releases ownership
					cmdbuf->getQueueFamilyIndex()
				};
				m_surface->present(std::move(swapchainLock), presentInfo);
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
			return base_t::onAppTerminated();
		}

		inline void update()
		{
			m_inputSystem->getDefaultMouse(&mouse);
			m_inputSystem->getDefaultKeyboard(&keyboard);

			auto updatePresentationTimestamp = [&]()
			{
				oracle.reportEndFrameRecord();
				const auto timestamp = oracle.getNextPresentationTimeStamp();
				oracle.reportBeginFrameRecord();

				return timestamp;
			};

			m_nextPresentationTimestamp = updatePresentationTimestamp();

			struct
			{
				std::vector<SMouseEvent> mouse {};
				std::vector<SKeyboardEvent> keyboard {};
			} capturedEvents;

			mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
			{
				for (const auto& e : events)
				{
					capturedEvents.mouse.emplace_back(e);

					if (e.type == nbl::ui::SMouseEvent::EET_SCROLL)
						gcIndex = std::clamp<uint16_t>(int16_t(gcIndex) + int16_t(core::sign(e.scrollEvent.verticalScroll)), int64_t(0), int64_t(OT_COUNT - (uint8_t)1u));
				}
			}, m_logger.get());

			keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
			{
				for (const auto& e : events)
				{
					capturedEvents.keyboard.emplace_back(e);
				}
			}, m_logger.get());

			const auto cursorPosition = m_window->getCursorControl()->getPosition();

			nbl::ext::imgui::UI::SUpdateParameters params =
			{
				.mousePosition = nbl::hlsl::float32_t2(cursorPosition.x, cursorPosition.y) - nbl::hlsl::float32_t2(m_window->getX(), m_window->getY()),
				.displaySize = { m_window->getWidth(), m_window->getHeight() },
				.mouseEvents = { capturedEvents.mouse.data(), capturedEvents.mouse.size() },
				.keyboardEvents = { capturedEvents.keyboard.data(), capturedEvents.keyboard.size() }
			};

			if (move)
			{
				// TODO: testing
				auto& camera = cameraz.front().get();

				static std::vector<CVirtualGimbalEvent> virtualEvents(0x45);
				uint32_t vCount;

				camera->beginInputProcessing(m_nextPresentationTimestamp);
				{
					camera->process(nullptr, vCount);

					if (virtualEvents.size() < vCount)
						virtualEvents.resize(vCount);

					camera->process(virtualEvents.data(), vCount, { params.keyboardEvents, params.mouseEvents });
				}
				camera->endInputProcessing();
				camera->manipulate({ virtualEvents.data(), vCount }, ICamera<matrix_precision_t>::Local);
			}

			m_ui.manager->update(params);
		}

	private:
		inline void imguiListen()
		{
			ImGuiIO& io = ImGui::GetIO();
			
			ImGuizmo::SetOrthographic(false);
			ImGuizmo::BeginFrame();

			ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
			ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

			// create a window and insert the inspector
			ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
			ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
			ImGui::Begin("Editor");

			//if (ImGui::RadioButton("Full view", !useWindow))
			//	useWindow = false;
			
			// TODO: I need this logic per viewport we will render a scene from a point of view of its bound camera
			{

				for (uint32_t i = 0u; i < projections.size(); ++i)
				{
					auto& projection = projections[i];

					if (isPerspective)
					{
						if (isLH)
							projection->setMatrix(buildProjectionMatrixPerspectiveFovLH<matrix_precision_t>(glm::radians(fov), aspectRatio[i], zNear, zFar));
						else
							projection->setMatrix(buildProjectionMatrixPerspectiveFovRH<matrix_precision_t>(glm::radians(fov), aspectRatio[i], zNear, zFar));
					}
					else
					{
						float viewHeight = viewWidth * invAspectRatio[i];

						if (isLH)
							projection->setMatrix(buildProjectionMatrixOrthoLH<matrix_precision_t>(viewWidth, viewHeight, zNear, zFar));
						else
							projection->setMatrix(buildProjectionMatrixOrthoRH<matrix_precision_t>(viewWidth, viewHeight, zNear, zFar));
					}
				}

				
			}

			ImGui::SameLine();

			//if (ImGui::RadioButton("Window", useWindow))
			//	useWindow = true;

			ImGui::Text("Camera");
			
			if (ImGui::RadioButton("LH", isLH))
				isLH = true;

			ImGui::SameLine();

			if (ImGui::RadioButton("RH", !isLH))
				isLH = false;

			//if (ImGui::RadioButton("Perspective", isPerspective))
			//	isPerspective = true;

			//ImGui::SameLine();

			//if (ImGui::RadioButton("Orthographic", !isPerspective))
			//	isPerspective = false;

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
			*/

			static struct
			{
				float32_t4x4 view[CamerazCount], projection[ProjectionsCount], inModel[2u], outModel[2u], outDeltaTRS[2u];
			} imguizmoM16InOut;

			//const auto& projectionMatrix = projection->getMatrix();

			const auto& firstcamera = cameraz.front();
			const auto& secondcamera = cameraz.back();

			ImGuizmo::SetID(0u);

			imguizmoM16InOut.view[0u] = transpose(getMatrix3x4As4x4(firstcamera->getGimbal().getView().matrix));
			imguizmoM16InOut.view[1u] = transpose(getMatrix3x4As4x4(secondcamera->getGimbal().getView().matrix));

			for(uint32_t i = 0u; i < ProjectionsCount; ++i)
				imguizmoM16InOut.projection[i] = transpose(projections[i]->getMatrix());

			// wth its kinda column major but seems to interprete RS part as row major?
			// will think of it later and check what its doing under the hood, now it
			// seems to match my local base orientation correctly
			auto toRetardedImguizmoTRSInput = [&](const float32_t3x4& nablaTrs) -> float32_t4x4
			{
				// *do not transpose whole matrix*, only the translate part
				float32_t4x4 trs = getMatrix3x4As4x4(nablaTrs);
				trs[3] = float32_t4(nablaTrs[0][3], nablaTrs[1][3], nablaTrs[2][3], 1.f);
				trs[0][3] = 0.f;
				trs[1][3] = 0.f;
				trs[2][3] = 0.f;

				return trs;
			};

			// we will transform a scene object's model
			imguizmoM16InOut.inModel[0] = toRetardedImguizmoTRSInput(m_model);
			// and second camera's model too
			imguizmoM16InOut.inModel[1] = toRetardedImguizmoTRSInput(secondcamera->getGimbal()());
			{
				float* views[]
				{
					&imguizmoM16InOut.view[0u][0][0], // first camera
					&imguizmoM16InOut.view[1u][0][0] // second camera
				};

				float* projections[]
				{
					&imguizmoM16InOut.projection[0u][0][0], // full screen, tmp off
					&imguizmoM16InOut.projection[1u][0][0], // first camera
					&imguizmoM16InOut.projection[2u][0][0] // second camera
				};

				if (flipGizmoY) // note we allow to flip gizmo just to match our coordinates
					for(uint32_t i = 0u; i < ProjectionsCount; ++i)
						imguizmoM16InOut.projection[i][1][1] *= -1.f; // https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/	

				float* lastUsedModel = &imguizmoM16InOut.inModel[lastUsing][0][0];

				// ImGuizmo manipulations on the last used model matrix
				TransformEditor(lastUsedModel);
				uint32_t gizmoID = {};
				bool manipulatedFromAnyWindow = false;

				// we have 2 GUI windows we render into with FBOs
				for (uint32_t windowIndex = 0; windowIndex < 2u; ++windowIndex)
				{
					auto* cameraView = views[windowIndex];
					auto* cameraProjection = projections[1u + windowIndex];

					TransformStart(windowIndex);

					// but only 2 objects with matrices we will try to manipulate & we can do this from both windows!
					// however note it only makes sense to obsly assume we cannot manipulate 2 objects at the same time
					for (uint32_t matId = 0; matId < 2; matId++)
					{
						//if(matId == 0)
						//	ImGuizmo::AllowAxisFlip(true);
						//else
							ImGuizmo::AllowAxisFlip(false);

						// and because of imguizmo API usage to achive it we must work on copies & filter the output (unless we try enable/disable logic described bellow) 
						// -> in general again we need to be careful to not edit the same model twice
						auto model = imguizmoM16InOut.inModel[matId];
						float32_t4x4 deltaOutputTRS;

						// note we also need to take care of unique gizmo IDs, we have in total 4 gizmos even though we only want to manipulate 2 objects in total
						ImGuizmo::PushID(gizmoID);

						ImGuiIO& io = ImGui::GetIO();
						const bool success = ImGuizmo::Manipulate(cameraView, cameraProjection, mCurrentGizmoOperation, mCurrentGizmoMode, &model[0][0], &deltaOutputTRS[0][0], useSnap ? &snap[0] : NULL);

						// we manipulated a gizmo from a X-th window, now we update output matrices and assume no more gizmos can be manipulated at the same frame
						// here we could also extend the logic & keep track of the gizmo being handled and as long as its manipulated (hold by mouse) -> then given ID of this gizmo we would
						// enable it & disable all other gizmos till we finish the manipulation
						if (!manipulatedFromAnyWindow)
						{
							imguizmoM16InOut.outModel[matId] = model;
							imguizmoM16InOut.outDeltaTRS[matId] = deltaOutputTRS;
						}

						if (success)
							manipulatedFromAnyWindow = true;
						
						if (ImGuizmo::IsUsing())
							lastUsing = matId;

						ImGuizmo::PopID();
						++gizmoID;
					}

					TransformEnd();
				}
			}

			/*
				testing imguizmo controller, we use deltra TRS matrix to generate virtual events
			*/

			{
				static std::vector<CVirtualGimbalEvent> virtualEvents(0x45);
				uint32_t vCount;

				secondcamera->beginInputProcessing(m_nextPresentationTimestamp);
				{
					secondcamera->process(nullptr, vCount);

					if (virtualEvents.size() < vCount)
						virtualEvents.resize(vCount);

					IGimbalController::SUpdateParameters params;

					params.imguizmoEvents = { { imguizmoM16InOut.outDeltaTRS[1u] } };

					secondcamera->process(virtualEvents.data(), vCount, params);
				}
				secondcamera->endInputProcessing();

				// TOOD: this needs debugging
				secondcamera->manipulate({ virtualEvents.data(), vCount }, ICamera<matrix_precision_t>::Local);
				/*
				if(vCount)
					switch (mCurrentGizmoMode)
					{
						case ImGuizmo::MODE::LOCAL:
							secondcamera->manipulate({ virtualEvents.data(), vCount }, ICamera<matrix_precision_t>::Local); break;
						case ImGuizmo::MODE::WORLD:
							secondcamera->manipulate({ virtualEvents.data(), vCount }, ICamera<matrix_precision_t>::World); break;
						default: break;
					}
					*/
			}

			// update scenes data
			// to Nabla + update camera & model matrices
		
			// TODO: make it more nicely
			std::string_view mName;
			const_cast<float32_t3x4&>(firstcamera->getGimbal().getView().matrix) = float32_t3x4(transpose(imguizmoM16InOut.view[0u])); // a hack, correct way would be to use inverse matrix and get position + target because now it will bring you back to last position & target when switching from gizmo move to manual move (but from manual to gizmo is ok)
			{
				m_model = float32_t3x4(transpose(imguizmoM16InOut.outModel[0]));

				const float32_t3x4* views[] = 
				{
					&firstcamera->getGimbal().getView().matrix,
					&secondcamera->getGimbal().getView().matrix
				};

				const auto& references = resources->objects;
				const auto type = static_cast<ObjectType>(gcIndex);
				const auto& [gpu, meta] = references[type];

				for (uint32_t i = 0u; i < cameraz.size(); ++i)
				{
					auto& scene = scenez[i];
					auto& hook = scene->object;

					hook.meta.type = type;
					mName = hook.meta.name = meta.name;
					{
						float32_t3x4 modelView, normal;
						float32_t4x4 modelViewProjection;

						const auto& viewMatrix = *views[i];
						modelView = concatenateBFollowedByA<float>(viewMatrix, m_model);

						// TODO
						//modelView.getSub3x3InverseTranspose(normal);

						auto concatMatrix = mul(projections[i]->getMatrix(), getMatrix3x4As4x4(viewMatrix));
						modelViewProjection = mul(concatMatrix, getMatrix3x4As4x4(m_model));

						memcpy(hook.viewParameters.MVP, &modelViewProjection[0][0], sizeof(hook.viewParameters.MVP));
						memcpy(hook.viewParameters.MV, &modelView[0][0], sizeof(hook.viewParameters.MV));
						memcpy(hook.viewParameters.NormalMat, &normal[0][0], sizeof(hook.viewParameters.NormalMat));
					}
				}
			}

			{
				auto addMatrixTable = [&](const char* topText, const char* tableName, int rows, int columns, const float* pointer, bool withSeparator = true)
				{
					ImGui::Text(topText);
					ImGui::PushStyleColor(ImGuiCol_TableRowBg, ImGui::GetStyleColorVec4(ImGuiCol_ChildBg));
					ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
					if (ImGui::BeginTable(tableName, columns, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame))
					{
						for (int y = 0; y < rows; ++y)
						{
							ImGui::TableNextRow();
							for (int x = 0; x < columns; ++x)
							{
								ImGui::TableSetColumnIndex(x);
								if (pointer)
									ImGui::Text("%.3f", *(pointer + (y * columns) + x));
								else
									ImGui::Text("-");
							}
						}
						ImGui::EndTable();
					}
					ImGui::PopStyleColor(2);
					if (withSeparator)
						ImGui::Separator();
				};

				// Scene Models
				{
					ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
					ImGui::Begin("Object Info and Model Matrix", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysVerticalScrollbar);

					ImGui::Text("Type: \"%s\"", mName.data());
					ImGui::Separator();

					ImGui::PushStyleColor(ImGuiCol_TableRowBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
					ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));

					addMatrixTable("Scene Object Model Matrix", "ModelMatrixTable", 3, 4, &m_model[0][0]);

					ImGui::PopStyleColor(2);
					ImGui::End();
				}

				// ImGuizmo inputs
				{
					ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
					ImGui::Begin("ImGuizmo model inputs", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysVerticalScrollbar);

					ImGui::PushStyleColor(ImGuiCol_TableRowBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
					ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));

					ImGui::Text("Type: GC Object");
					ImGui::Separator();

					addMatrixTable("Model", "GCIMGUIZMOTRS", 4, 4, &imguizmoM16InOut.inModel[0][0][0]);

					ImGui::Text("Type: Camera ID \"1\" object");
					ImGui::Separator();

					addMatrixTable("Model", "CAMERASECIMGUIZMOTRS", 4, 4, &imguizmoM16InOut.inModel[1][0][0]);

					ImGui::PopStyleColor(2);
					ImGui::End();
				}

				//imguizmoM16InOut.inModel

				// Cameraz
				{
					size_t cameraCount = cameraz.size();
					if (cameraCount > 0)
					{
						size_t columns = std::max<size_t>(1, static_cast<size_t>(std::sqrt(static_cast<double>(cameraCount))));
						size_t rows = (cameraCount + columns - 1) / columns;

						ImGui::Begin("Camera Matrices", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysVerticalScrollbar);

						ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 2.0f));

						for (size_t row = 0; row < rows; ++row)
						{
							for (size_t col = 0; col < columns; ++col)
							{
								size_t cameraIndex = row * columns + col;
								if (cameraIndex >= cameraCount)
									break;

								auto& camera = cameraz[cameraIndex];
								if (!camera)
									continue;

								auto& gimbal = camera->getGimbal();
								const auto& position = gimbal.getPosition();
								const auto& orientation = gimbal.getOrientation();
								const auto& viewMatrix = gimbal.getView().matrix;
								const auto trsMatrix = gimbal();

								ImGui::Text("ID: %zu", cameraIndex);
								ImGui::Separator();

								ImGui::Text("Type: %s", camera->getIdentifier().data());
								ImGui::Separator();

								ImGui::PushStyleColor(ImGuiCol_TableRowBg, ImGui::GetStyleColorVec4(ImGuiCol_ChildBg));
								ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));

								if (ImGui::BeginTable(("CameraTable" + std::to_string(cameraIndex)).c_str(), 1, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame))
								{
									ImGui::TableNextRow();
									ImGui::TableSetColumnIndex(0);

									const auto idxstr = std::to_string(cameraIndex);
									addMatrixTable("Position", ("PositionTable" + idxstr).c_str(), 1, 3, &position[0], false);
									addMatrixTable("Orientation (Quaternion)", ("OrientationTable" + idxstr).c_str(), 1, 4, &orientation[0], false);
									addMatrixTable("View Matrix", ("ViewMatrixTable" + idxstr).c_str(), 3, 4, &viewMatrix[0][0], false);
									addMatrixTable("TRS Matrix", ("TRSMatrixTable" + idxstr).c_str(), 3, 4, &trsMatrix[0][0], true);

									ImGui::EndTable();
								}

								ImGui::PopStyleColor(2);
							}
						}

						ImGui::PopStyleVar();
						ImGui::End();
					}
					else
						ImGui::Text("No camera properties to display.");
				}
			}

			
			// Nabla Imgui backend MDI buffer info
			// To be 100% accurate and not overly conservative we'd have to explicitly `cull_frees` and defragment each time,
			// so unless you do that, don't use this basic info to optimize the size of your IMGUI buffer.
			{
				auto* streaminingBuffer = m_ui.manager->getStreamingBuffer();

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

			displayKeyMappingsAndVirtualStates(cameraz.front().get());

			ImGui::End();
		}

		void TransformEditor(float* matrix)
		{
			static float bounds[] = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
			static float boundsSnap[] = { 0.1f, 0.1f, 0.1f };
			static bool boundSizing = false;
			static bool boundSizingSnap = false;

			if (ImGui::IsKeyPressed(ImGuiKey_T))
				mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
			if (ImGui::IsKeyPressed(ImGuiKey_E))
				mCurrentGizmoOperation = ImGuizmo::ROTATE;
			if (ImGui::IsKeyPressed(ImGuiKey_R)) // r Key
				mCurrentGizmoOperation = ImGuizmo::SCALE;
			if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
				mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
				mCurrentGizmoOperation = ImGuizmo::ROTATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
				mCurrentGizmoOperation = ImGuizmo::SCALE;
			float matrixTranslation[3], matrixRotation[3], matrixScale[3];
			ImGuizmo::DecomposeMatrixToComponents(matrix, matrixTranslation, matrixRotation, matrixScale);
			ImGui::InputFloat3("Tr", matrixTranslation);
			ImGui::InputFloat3("Rt", matrixRotation);
			ImGui::InputFloat3("Sc", matrixScale);
			ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, matrix);

			if (mCurrentGizmoOperation != ImGuizmo::SCALE)
			{
				if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
					mCurrentGizmoMode = ImGuizmo::LOCAL;
				ImGui::SameLine();
				if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
					mCurrentGizmoMode = ImGuizmo::WORLD;
			}

			if (ImGui::IsKeyPressed(ImGuiKey_S))
				useSnap = !useSnap;
			ImGui::Checkbox(" ", &useSnap);
			ImGui::SameLine();
			switch (mCurrentGizmoOperation)
			{
			case ImGuizmo::TRANSLATE:
				ImGui::InputFloat3("Snap", &snap[0]);
				break;
			case ImGuizmo::ROTATE:
				ImGui::InputFloat("Angle Snap", &snap[0]);
				break;
			case ImGuizmo::SCALE:
				ImGui::InputFloat("Scale Snap", &snap[0]);
				break;
			}
		}

		void TransformStart(uint32_t wCameraIndex)
		{
			ImGuiIO& io = ImGui::GetIO();
			static ImGuiWindowFlags gizmoWindowFlags = ImGuiWindowFlags_NoMove;

			SImResourceInfo info;
			switch (wCameraIndex) 
			{
				case 0:
					info.textureID = OfflineSceneFirstCameraTextureIx;
					break;
				case 1:
					info.textureID = OfflineSceneSecondCameraTextureIx;
					break;
				default:
					assert(false); // "wCameraIndex OOB!"
					break;
			}
			info.samplerIx = (uint16_t)nbl::ext::imgui::UI::DefaultSamplerIx::USER;

			aspectRatio[0] = io.DisplaySize.x / io.DisplaySize.y;
			invAspectRatio[0] = io.DisplaySize.y / io.DisplaySize.x;

			if (useWindow)
			{
				ImGui::SetNextWindowSize(ImVec2(800, 400), ImGuiCond_Appearing);
				ImGui::SetNextWindowPos(ImVec2(400, 20 + wCameraIndex * 420), ImGuiCond_Appearing);
				ImGui::PushStyleColor(ImGuiCol_WindowBg, (ImVec4)ImColor(0.35f, 0.3f, 0.3f));
				std::string ident = "Active Camera ID: " + std::to_string(wCameraIndex);
				ImGui::Begin(ident.data(), 0, gizmoWindowFlags);
				ImGuizmo::SetDrawlist();

				ImVec2 contentRegionSize, windowPos, cursorPos;
				contentRegionSize = ImGui::GetContentRegionAvail();
				cursorPos = ImGui::GetCursorScreenPos();
				windowPos = ImGui::GetWindowPos();

				aspectRatio[wCameraIndex + 1] = contentRegionSize.x / contentRegionSize.y;
				invAspectRatio[wCameraIndex + 1] = contentRegionSize.y / contentRegionSize.x;

				ImGui::Image(info, contentRegionSize);
				ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);
			}
			else
			{
				ImGui::SetNextWindowPos(ImVec2(0, 0));
				ImGui::SetNextWindowSize(io.DisplaySize);
				ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0)); // fully transparent fake window
				ImGui::Begin("FullScreenWindow", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs);

				ImVec2 contentRegionSize, windowPos, cursorPos;
				contentRegionSize = ImGui::GetContentRegionAvail();
				cursorPos = ImGui::GetCursorScreenPos();
				windowPos = ImGui::GetWindowPos();

				ImGui::Image(info, contentRegionSize);
				ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);
			}

			ImGuiWindow* window = ImGui::GetCurrentWindow();
			gizmoWindowFlags = ImGuiWindowFlags_NoMove;//(ImGuizmo::IsUsing() && ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(window->InnerRect.Min, window->InnerRect.Max) ? ImGuiWindowFlags_NoMove : 0);
		}

		void TransformEnd()
		{
			if (useWindow)
			{
				ImGui::End();
			}
			ImGui::PopStyleColor(1);
		}

		std::chrono::seconds timeout = std::chrono::seconds(0x7fffFFFFu);
		clock_t::time_point start;

		//! One window & surface
		smart_refctd_ptr<CSmoothResizeSurface<CSwapchainResources>> m_surface;
		smart_refctd_ptr<IWindow> m_window;
		// We can't use the same semaphore for acquire and present, because that would disable "Frames in Flight" by syncing previous present against next acquire.
		// At least two timelines must be used.
		smart_refctd_ptr<ISemaphore> m_semaphore;
		// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
		constexpr static inline uint32_t MaxFramesInFlight = 3u;
		// Use a separate counter to cycle through our resources because `getAcquireCount()` increases upon spontaneous resizes with immediate blit-presents 
		uint64_t m_realFrameIx = 0;
		// We'll write to the Triple Buffer with a Renderpass
		core::smart_refctd_ptr<IGPURenderpass> m_renderpass = {};
		// These are atomic counters where the Surface lets us know what's the latest Blit timeline semaphore value which will be signalled on the resource
		std::array<std::atomic_uint64_t, MaxFramesInFlight> m_blitWaitValues;
		// Enough Command Buffers and other resources for all frames in flight!
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
		// Our own persistent images that don't get recreated with the swapchain
		std::array<smart_refctd_ptr<IGPUImage>, MaxFramesInFlight> m_tripleBuffers;
		// Resources derived from the images
		std::array<core::smart_refctd_ptr<IGPUFramebuffer>, MaxFramesInFlight> m_framebuffers = {};
		// We will use it to get some asset stuff like geometry creator
		smart_refctd_ptr<nbl::asset::IAssetManager> m_assetManager;
		// Input system for capturing system events
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		// Handles mouse events
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		// Handles keyboard events
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
		//! next presentation timestamp
		std::chrono::microseconds m_nextPresentationTimestamp = {};

		// UI font atlas; first camera fb, second camera fb
		constexpr static inline auto TotalUISampleTexturesAmount = 3u;

		core::smart_refctd_ptr<IDescriptorPool> m_descriptorSetPool;

		struct CRenderUI
		{
			nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> manager;

			struct
			{
				core::smart_refctd_ptr<video::IGPUSampler> gui, scene;
			} samplers;

			core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
		};

		// one model object in the world, testing 2 cameraz for which view is rendered to separate frame buffers (so what they see) with new controller API including imguizmo
		nbl::hlsl::float32_t3x4 m_model = nbl::hlsl::float32_t3x4(1.f);
		static constexpr inline auto CamerazCount = 2u;
		std::array<nbl::core::smart_refctd_ptr<CScene>, CamerazCount> scenez;
		std::array<core::smart_refctd_ptr<ICamera<matrix_precision_t>>, CamerazCount> cameraz;
		nbl::core::smart_refctd_ptr<ResourcesBundle> resources;

		static constexpr inline auto OfflineSceneFirstCameraTextureIx = 1u;
		static constexpr inline auto OfflineSceneSecondCameraTextureIx = 2u;

		CRenderUI m_ui;
		video::CDumbPresentationOracle oracle;

		uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed

		static constexpr inline auto ProjectionsCount = 3u;
		std::array<smart_refctd_ptr<projection_t>, ProjectionsCount> projections; // TMP!
		bool isPerspective[ProjectionsCount] = { true, true, true }, isLH = true, flipGizmoY = true, move = false;
		float fov = 60.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
		float viewWidth = 10.f;
		float camYAngle = 165.f / 180.f * 3.14159f;
		float camXAngle = 32.f / 180.f * 3.14159f;

		float camDistance = 8.f, aspectRatio[ProjectionsCount] = {}, invAspectRatio[ProjectionsCount] = {};
		bool useWindow = true, useSnap = false;
		ImGuizmo::OPERATION mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
		ImGuizmo::MODE mCurrentGizmoMode = ImGuizmo::WORLD;
		float snap[3] = { 1.f, 1.f, 1.f };
		int lastUsing = 0;

		bool firstFrame = true;
};

NBL_MAIN_FUNC(UISampleApp)