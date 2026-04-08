#ifndef _NBL_THIS_EXAMPLE_APP_HPP_
#define _NBL_THIS_EXAMPLE_APP_HPP_

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include "argparse/argparse.hpp"

#include "common.hpp"
#include "keysmapping.hpp"
#include "app/AppTypes.hpp"
#include "camera/CCubeProjection.hpp"
#include "nbl/ext/Frustum/CDrawFrustum.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

class CUIEventCallback : public nbl::video::ISmoothResizeSurface::ICallback
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

	inline uint8_t getLastImageIndex() const { return m_lastImageIndex; }

protected:
	// We can return `BLIT_BIT` here, because the Source Image will be already in the correct layout to be used for the present
	inline core::bitflag<asset::PIPELINE_STAGE_FLAGS> getTripleBufferPresentStages() const override { return asset::PIPELINE_STAGE_FLAGS::BLIT_BIT; }

	inline bool tripleBufferPresent(IGPUCommandBuffer* cmdbuf, const ISmoothResizeSurface::SPresentSource& source, const uint8_t imageIndex, const uint32_t qFamToAcquireSrcFrom) override
	{
		bool success = true;
		auto acquiredImage = getImage(imageIndex);
		m_lastImageIndex = imageIndex;

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
				.oldLayout = IGPUImage::LAYOUT::UNDEFINED,
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

private:
	uint8_t m_lastImageIndex = 0u;
};

static smart_refctd_ptr<IGPUImageView> createAttachmentView(ILogicalDevice* device, E_FORMAT format, uint32_t width, uint32_t height, const char* debugName)
{
	if (!device)
		return nullptr;

	const bool isDepth = isDepthOrStencilFormat(format);
	auto usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT;
	if (!isDepth)
		usage |= IGPUImage::EUF_SAMPLED_BIT;

	auto image = device->createImage({{
		.type = IGPUImage::ET_2D,
		.samples = IGPUImage::ESCF_1_BIT,
		.format = format,
		.extent = { width, height, 1u },
		.mipLevels = 1u,
		.arrayLayers = 1u,
		.usage = usage
	}});
	if (!image)
		return nullptr;

	image->setObjectDebugName(debugName);

	if (!device->allocate(image->getMemoryReqs(), image.get()).isValid())
		return nullptr;

	IGPUImageView::SCreationParams params = {
		.subUsages = usage,
		.image = std::move(image),
		.viewType = IGPUImageView::ET_2D,
		.format = format
	};
	params.subresourceRange.aspectMask = isDepth ? IGPUImage::EAF_DEPTH_BIT : IGPUImage::EAF_COLOR_BIT;
	return device->createImageView(std::move(params));
}

static smart_refctd_ptr<IGPUFramebuffer> createSceneFramebuffer(ILogicalDevice* device, IGPURenderpass* renderpass, IGPUImageView* colorView, IGPUImageView* depthView)
{
	if (!device || !renderpass || !colorView || !depthView)
		return nullptr;

	const auto& imageParams = colorView->getCreationParameters().image->getCreationParameters();
	IGPUFramebuffer::SCreationParams params = { {
		.renderpass = core::smart_refctd_ptr<IGPURenderpass>(renderpass),
		.depthStencilAttachments = &depthView,
		.colorAttachments = &colorView,
		.width = imageParams.extent.width,
		.height = imageParams.extent.height,
		.layers = imageParams.arrayLayers
	} };
	return device->createFramebuffer(std::move(params));
}

class App final : public examples::SimpleWindowedApplication, public examples::BuiltinResourcesApplication
{
	using base_t = examples::SimpleWindowedApplication;
	using asset_base_t = examples::BuiltinResourcesApplication;
	using clock_t = std::chrono::steady_clock;

	constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);
	constexpr static inline auto sceneRenderDepthFormat = EF_D32_SFLOAT;
	constexpr static inline auto finalSceneRenderFormat = EF_R8G8B8A8_SRGB;
	constexpr static inline IGPUCommandBuffer::SClearColorValue SceneClearColor = { .float32 = {0.014f,0.018f,0.030f,1.f} };
	constexpr static inline IGPUCommandBuffer::SClearDepthStencilValue SceneClearDepth = { .depth = 0.f };
	struct SpaceEnvPushConstants
	{
		float32_t4x4 invProj = float32_t4x4(1.f);
		float32_t4x4 invViewRot = float32_t4x4(1.f);
		uint32_t orthoMode = 0u;
		uint32_t pad0 = 0u;
		uint32_t pad1 = 0u;
		uint32_t pad2 = 0u;
	};

	public:
		using base_t::base_t;

		inline App(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) 
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
				auto* cc = m_window->getCursorControl();
				cc->setVisible(true);

				return { {m_surface->getSurface()/*,EQF_NONE*/} };
			}
			
			return {};
		}

		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override;
		core::bitflag<nbl::system::ILogger::E_LOG_LEVEL> getLogLevelMask() override
		{
			return core::bitflag<nbl::system::ILogger::E_LOG_LEVEL>(nbl::system::ILogger::ELL_INFO) |
				nbl::system::ILogger::ELL_WARNING |
				nbl::system::ILogger::ELL_PERFORMANCE |
				nbl::system::ILogger::ELL_ERROR;
		}

		bool updateGUIDescriptorSet()
		{
			// UI texture atlas + our camera scene textures, note we don't create info & write pair for the font sampler because UI extension's is immutable and baked into DS layout
			static std::array<IGPUDescriptorSet::SDescriptorInfo, TotalUISampleTexturesAmount> descriptorInfo;
			static IGPUDescriptorSet::SWriteDescriptorSet writes[TotalUISampleTexturesAmount];

			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].desc = core::smart_refctd_ptr<nbl::video::IGPUImageView>(m_ui.manager->getFontAtlasView());
			writes[nbl::ext::imgui::UI::FontAtlasTexId].info = descriptorInfo.data() + nbl::ext::imgui::UI::FontAtlasTexId;

			for (uint32_t i = 0; i < windowBindings.size(); ++i)
			{
				const auto textureIx = i + 1u;

				descriptorInfo[textureIx].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
				descriptorInfo[textureIx].desc = windowBindings[i].sceneColorView;

				writes[textureIx].info = descriptorInfo.data() + textureIx;
				writes[textureIx].info = descriptorInfo.data() + textureIx;
			}

			for (uint32_t i = 0; i < descriptorInfo.size(); ++i)
			{
				writes[i].dstSet = m_ui.descriptorSet.get();
				writes[i].binding = 0u;
				writes[i].arrayElement = i;
				writes[i].count = 1u;
			}

			return m_device->updateDescriptorSets(writes, {});
		}

		void workLoopBody() override;

		inline void paceScriptedVisualDebugFrame()
		{
			if (!(m_scriptedInput.enabled && m_scriptedInput.visualDebug))
			{
				m_scriptedInput.framePacerInitialized = false;
				return;
			}

			if (m_scriptedInput.visualTargetFps <= 0.f)
				return;

			const auto frameDuration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
				std::chrono::duration<double>(1.0 / static_cast<double>(m_scriptedInput.visualTargetFps)));
			const auto now = std::chrono::steady_clock::now();

			if (!m_scriptedInput.framePacerInitialized)
			{
				m_scriptedInput.framePacerInitialized = true;
				m_scriptedInput.framePacerNext = now + frameDuration;
				return;
			}

			if (now < m_scriptedInput.framePacerNext)
				std::this_thread::sleep_until(m_scriptedInput.framePacerNext);

			auto postSleepNow = std::chrono::steady_clock::now();
			while (m_scriptedInput.framePacerNext < postSleepNow)
				m_scriptedInput.framePacerNext += frameDuration;
		}

		inline bool keepRunning() override
		{
			if (m_headlessCameraSmokeMode)
				return false;

			if (m_scriptedInput.enabled && m_scriptedInput.hardFail && m_scriptedInput.failed)
			{
				if (!m_ciMode || m_ciScreenshotDone)
					std::exit(EXIT_FAILURE);
			}
			if (m_ciMode && m_ciStartedAt != clock_t::time_point::min())
			{
				const auto elapsed = clock_t::now() - m_ciStartedAt;
				if (elapsed > CiMaxRuntime)
				{
					m_logger->log("[ci][fail] watchdog timeout after %.2f s.", ILogger::ELL_ERROR,
						std::chrono::duration<double>(elapsed).count());
					std::exit(EXIT_FAILURE);
				}
			}
			if (m_ciMode && m_ciScreenshotDone)
			{
				if (m_scriptedInput.enabled)
				{
					if (m_scriptedInput.nextCaptureIndex < m_scriptedInput.timeline.captureFrames.size())
						return true;
					if (m_scriptedInput.nextEventIndex < m_scriptedInput.timeline.events.size())
						return true;
					if (m_scriptedInput.checkRuntime.nextCheckIndex < m_scriptedInput.timeline.checks.size())
						return true;
				}
				return false;
			}
			if (m_surface->irrecoverable())
				return false;

			return true;
		}

		inline bool onAppTerminated() override
		{
			if (m_headlessCameraSmokeMode)
				return m_headlessCameraSmokePassed;

			return base_t::onAppTerminated();
		}

		void update();

		private:
		struct CUILogFormatter final : public nbl::system::ILogger
		{
			CUILogFormatter() : ILogger(ILogger::DefaultLogMask()) {}

			std::string format(E_LOG_LEVEL level, std::string_view fmt, ...)
			{
				va_list args;
				va_start(args, fmt);
				auto out = constructLogString(fmt, level, args);
				va_end(args);
				if (!out.empty() && out.back() == '\n')
					out.pop_back();
				return out;
			}

		protected:
			void log_impl(const std::string_view&, E_LOG_LEVEL, va_list) override {}
		};

		struct VirtualEventLogEntry
		{
			uint64_t frame = 0;
			CVirtualGimbalEvent::VirtualEventType type = CVirtualGimbalEvent::None;
			float64_t magnitude = 0.0;
			std::string source;
			std::string inputSource;
			std::string camera;
			uint32_t planarIx = 0u;
			std::string line;
		};

		using CameraPreset = CCameraPreset;
		using CameraKeyframe = CCameraKeyframe;
		using CameraKeyframeTrack = CCameraKeyframeTrack;

		using PresetFilterMode = EPresetApplyPresentationFilter;
		using PresetUiAnalysis = SCameraGoalApplyPresentation;
		using CaptureUiAnalysis = SCameraCapturePresentation;

		struct CameraPlaybackState : CCameraPlaybackCursor
		{
			bool overrideInput = true;
		};

		struct ApplyStatusBanner
		{
			std::string summary;
			bool succeeded = false;
			bool approximate = false;

			inline bool visible() const
			{
				return !summary.empty();
			}
		};

		enum class SceneManipulatedObjectKind : uint8_t
		{
			Model,
			FollowTarget,
			Camera
		};

		struct CameraControlSettings
		{
			bool mirrorInput = false;
			bool worldTranslate = false;
			float keyboardScale = 0.00625f;
			float mouseMoveScale = 1.0f;
			float mouseScrollScale = 1.0f;
			float translationScale = 1.0f;
			float rotationScale = 1.0f;
		};

		using CameraConstraintSettings = SCameraConstraintSettings;

		inline ICamera* getActiveCamera()
		{
			auto& binding = windowBindings[activeRenderWindowIx];
			auto& planar = m_planarProjections[binding.activePlanarIx];
			return planar ? planar->getCamera() : nullptr;
		}

		inline uint32_t getActivePlanarIx() const
		{
			return windowBindings[activeRenderWindowIx].activePlanarIx;
		}

		inline SCameraFollowConfig* getActiveFollowConfig()
		{
			const auto planarIx = getActivePlanarIx();
			if (planarIx >= m_planarFollowConfigs.size())
				return nullptr;
			return &m_planarFollowConfigs[planarIx];
		}

		inline const SCameraFollowConfig* getActiveFollowConfig() const
		{
			const auto planarIx = getActivePlanarIx();
			if (planarIx >= m_planarFollowConfigs.size())
				return nullptr;
			return &m_planarFollowConfigs[planarIx];
		}

		inline uint32_t getManipulableObjectCount() const
		{
			return 2u + static_cast<uint32_t>(m_planarProjections.size());
		}

		inline bool isManipulableObjectFollowTarget(const uint32_t objectIx) const
		{
			return objectIx == 1u;
		}

		inline std::optional<uint32_t> getManipulableObjectPlanarIx(const uint32_t objectIx) const
		{
			if (objectIx < 2u)
				return std::nullopt;
			const auto planarIx = objectIx - 2u;
			if (planarIx >= m_planarProjections.size())
				return std::nullopt;
			return planarIx;
		}

		inline uint32_t getManipulatedObjectIx() const
		{
			switch (m_manipulatedObjectKind)
			{
				case SceneManipulatedObjectKind::Model:
					return 0u;
				case SceneManipulatedObjectKind::FollowTarget:
					return 1u;
				case SceneManipulatedObjectKind::Camera:
				default:
					return boundPlanarCameraIxToManipulate.has_value() ? (boundPlanarCameraIxToManipulate.value() + 2u) : 0u;
			}
		}

		inline void bindManipulatedModel()
		{
			m_manipulatedObjectKind = SceneManipulatedObjectKind::Model;
			boundCameraToManipulate = nullptr;
			boundPlanarCameraIxToManipulate = std::nullopt;
		}

		inline void bindManipulatedFollowTarget()
		{
			m_manipulatedObjectKind = SceneManipulatedObjectKind::FollowTarget;
			boundCameraToManipulate = nullptr;
			boundPlanarCameraIxToManipulate = std::nullopt;
		}

		inline void bindManipulatedCamera(const uint32_t planarIx)
		{
			if (planarIx >= m_planarProjections.size())
			{
				bindManipulatedModel();
				return;
			}

			auto* camera = m_planarProjections[planarIx] ? m_planarProjections[planarIx]->getCamera() : nullptr;
			if (!camera)
			{
				bindManipulatedModel();
				return;
			}

			m_manipulatedObjectKind = SceneManipulatedObjectKind::Camera;
			boundPlanarCameraIxToManipulate = planarIx;
			boundCameraToManipulate = smart_refctd_ptr<ICamera>(camera);
		}

		inline void bindManipulatedObjectByIx(const uint32_t objectIx)
		{
			if (objectIx == 0u)
				return bindManipulatedModel();
			if (isManipulableObjectFollowTarget(objectIx))
				return bindManipulatedFollowTarget();
			if (const auto planarIx = getManipulableObjectPlanarIx(objectIx); planarIx.has_value())
				return bindManipulatedCamera(planarIx.value());
			bindManipulatedModel();
		}

		inline std::string getManipulableObjectLabel(const uint32_t objectIx) const
		{
			if (objectIx == 0u)
				return "Model";
			if (isManipulableObjectFollowTarget(objectIx))
				return m_followTarget.getIdentifier();
			if (const auto planarIx = getManipulableObjectPlanarIx(objectIx); planarIx.has_value())
			{
				auto* camera = m_planarProjections[planarIx.value()] ? m_planarProjections[planarIx.value()]->getCamera() : nullptr;
				if (!camera)
					return "Camera " + std::to_string(planarIx.value());
				return std::string(getCameraTypeLabel(camera)) + " Camera";
			}
			return "Unknown";
		}

		inline float32_t4x4 getManipulableObjectTransform(const uint32_t objectIx) const
		{
			if (objectIx == 0u)
				return hlsl::transpose(getMatrix3x4As4x4(m_model));
			if (isManipulableObjectFollowTarget(objectIx))
				return getCastedMatrix<float32_t>(m_followTarget.getGimbal().template operator()<float64_t4x4>());

			if (const auto planarIx = getManipulableObjectPlanarIx(objectIx); planarIx.has_value())
			{
				auto* camera = m_planarProjections[planarIx.value()] ? m_planarProjections[planarIx.value()]->getCamera() : nullptr;
				if (camera)
					return getCastedMatrix<float32_t>(camera->getGimbal().template operator()<float64_t4x4>());
			}

			return float32_t4x4(1.0f);
		}

		inline float32_t3 getManipulableObjectWorldPosition(const uint32_t objectIx) const
		{
			if (objectIx == 0u)
			{
				const auto modelPos = hlsl::transpose(getMatrix3x4As4x4(m_model))[3];
				return float32_t3(modelPos.x, modelPos.y, modelPos.z);
			}
			if (isManipulableObjectFollowTarget(objectIx))
				return getCastedVector<float32_t>(m_followTarget.getGimbal().getPosition());

			if (const auto planarIx = getManipulableObjectPlanarIx(objectIx); planarIx.has_value())
			{
				auto* camera = m_planarProjections[planarIx.value()] ? m_planarProjections[planarIx.value()]->getCamera() : nullptr;
				if (camera)
					return getCastedVector<float32_t>(camera->getGimbal().getPosition());
			}

			return float32_t3(0.0f);
		}

		inline float32_t3x4 computeFollowTargetMarkerWorld() const
		{
			const auto& targetGimbal = m_followTarget.getGimbal();
			const auto position = getCastedVector<float32_t>(targetGimbal.getPosition());
			const auto axisX = getCastedVector<float32_t>(targetGimbal.getXAxis());
			const auto axisY = getCastedVector<float32_t>(targetGimbal.getYAxis());
			const auto axisZ = getCastedVector<float32_t>(targetGimbal.getZAxis());
			const float markerScale = (m_scriptedInput.enabled && m_scriptedInput.visualDebug) ? 0.6f : 0.28f;
			return {
				float32_t4(axisX * markerScale, position.x),
				float32_t4(axisY * markerScale, position.y),
				float32_t4(axisZ * markerScale, position.z)
			};
		}

		inline void setFollowTargetTransform(const float64_t4x4& transform)
		{
			m_followTarget.trySetFromTransform(transform);
		}

		inline bool captureFollowOffsetsForPlanar(const uint32_t planarIx)
		{
			if (planarIx >= m_planarProjections.size() || planarIx >= m_planarFollowConfigs.size())
				return false;
			auto* camera = m_planarProjections[planarIx] ? m_planarProjections[planarIx]->getCamera() : nullptr;
			return nbl::core::captureFollowOffsetsFromCamera(m_cameraGoalSolver, camera, m_followTarget, m_planarFollowConfigs[planarIx]);
		}

		inline bool followConfigUsesCapturedOffset(const SCameraFollowConfig& config) const
		{
			return config.enabled && nbl::core::cameraFollowModeUsesCapturedOffset(config.mode);
		}

		inline void refreshFollowOffsetConfigForPlanar(const uint32_t planarIx)
		{
			if (planarIx >= m_planarProjections.size() || planarIx >= m_planarFollowConfigs.size())
				return;

			auto& config = m_planarFollowConfigs[planarIx];
			if (!followConfigUsesCapturedOffset(config))
				return;

			auto* camera = m_planarProjections[planarIx] ? m_planarProjections[planarIx]->getCamera() : nullptr;
			if (!camera)
				return;

			nbl::core::captureFollowOffsetsFromCamera(m_cameraGoalSolver, camera, m_followTarget, config);
		}

		inline void refreshFollowOffsetConfigsForCamera(ICamera* camera)
		{
			if (!camera)
				return;

			for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size() && planarIx < m_planarFollowConfigs.size(); ++planarIx)
			{
				auto* planarCamera = m_planarProjections[planarIx] ? m_planarProjections[planarIx]->getCamera() : nullptr;
				if (planarCamera != camera)
					continue;
				refreshFollowOffsetConfigForPlanar(planarIx);
			}
		}

		inline void refreshAllFollowOffsetConfigs()
		{
			for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size() && planarIx < m_planarFollowConfigs.size(); ++planarIx)
				refreshFollowOffsetConfigForPlanar(planarIx);
		}

		inline float64_t3 getDefaultFollowTargetPosition() const
		{
			return float64_t3(6.0, -4.5, 2.25);
		}

		inline camera_quaternion_t<float64_t> getDefaultFollowTargetOrientation() const
		{
			return makeIdentityQuaternion<float64_t>();
		}

		inline void resetFollowTargetToDefault()
		{
			m_followTarget.setPose(getDefaultFollowTargetPosition(), getDefaultFollowTargetOrientation());
		}

		inline void snapFollowTargetToModel()
		{
			const auto modelTransform = hlsl::transpose(getMatrix3x4As4x4(m_model));
			setFollowTargetTransform(getCastedMatrix<float64_t>(modelTransform));
		}

		inline SCameraFollowConfig makeDefaultFollowConfig(ICamera* camera)
		{
			SCameraFollowConfig config = {};
			if (!camera)
				return config;

			switch (camera->getKind())
			{
				case ICamera::CameraKind::Orbit:
				case ICamera::CameraKind::Arcball:
				case ICamera::CameraKind::Turntable:
				case ICamera::CameraKind::TopDown:
				case ICamera::CameraKind::Isometric:
				case ICamera::CameraKind::DollyZoom:
				case ICamera::CameraKind::Path:
					config.enabled = true;
					config.mode = ECameraFollowMode::OrbitTarget;
					break;
				case ICamera::CameraKind::Chase:
				case ICamera::CameraKind::Dolly:
					config.enabled = true;
					config.mode = ECameraFollowMode::KeepLocalOffset;
					break;
				default:
					break;
			}

			return config;
		}

		inline void applyFollowToConfiguredCameras(const bool allowDuringScriptedInput = false)
		{
			if (m_scriptedInput.enabled && !allowDuringScriptedInput)
				return;
			if (m_planarFollowConfigs.size() != m_planarProjections.size())
				return;

			for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size(); ++planarIx)
			{
				auto& planar = m_planarProjections[planarIx];
				auto* camera = planar ? planar->getCamera() : nullptr;
				if (!camera)
					continue;

				const auto& config = m_planarFollowConfigs[planarIx];
				if (!config.enabled || config.mode == ECameraFollowMode::Disabled)
					continue;

				const auto result = nbl::core::applyFollowToCamera(m_cameraGoalSolver, camera, m_followTarget, config);
				if (!result.succeeded())
					continue;

				for (auto& projection : planar->getPlanarProjections())
					nbl::core::syncDynamicPerspectiveProjection(camera, projection);
			}
		}

		inline bool isOrbitLikeCamera(ICamera* camera)
		{
			return camera && camera->hasCapability(ICamera::SphericalTarget);
		}

		inline void syncVisualDebugWindowBindings()
		{
			if (!m_scriptedInput.enabled)
				return;
			if (windowBindings.size() < 2u || m_planarProjections.empty())
				return;

			auto& perspectiveBinding = windowBindings[0u];
			if (perspectiveBinding.activePlanarIx >= m_planarProjections.size())
				return;
			auto& perspectivePlanar = m_planarProjections[perspectiveBinding.activePlanarIx];
			if (!perspectivePlanar)
				return;
			if (!perspectiveBinding.lastBoundPerspectivePresetProjectionIx.has_value())
				perspectiveBinding.pickDefaultProjections(perspectivePlanar->getPlanarProjections());
			if (perspectiveBinding.lastBoundPerspectivePresetProjectionIx.has_value())
				perspectiveBinding.boundProjectionIx = perspectiveBinding.lastBoundPerspectivePresetProjectionIx.value();

			auto& orthoBinding = windowBindings[1u];
			if (orthoBinding.activePlanarIx != perspectiveBinding.activePlanarIx)
			{
				orthoBinding.activePlanarIx = perspectiveBinding.activePlanarIx;
				auto& orthoPlanar = m_planarProjections[orthoBinding.activePlanarIx];
				if (!orthoPlanar)
					return;
				orthoBinding.pickDefaultProjections(orthoPlanar->getPlanarProjections());
			}
			if (orthoBinding.activePlanarIx >= m_planarProjections.size())
				return;
			auto& orthoPlanar = m_planarProjections[orthoBinding.activePlanarIx];
			if (!orthoPlanar)
				return;
			if (!orthoBinding.lastBoundOrthoPresetProjectionIx.has_value())
				orthoBinding.pickDefaultProjections(orthoPlanar->getPlanarProjections());
			if (orthoBinding.lastBoundOrthoPresetProjectionIx.has_value())
				orthoBinding.boundProjectionIx = orthoBinding.lastBoundOrthoPresetProjectionIx.value();
		}

		inline void drawScriptVisualDebugOverlay(const ImVec2& displaySize)
			{
			if (!(m_scriptedInput.enabled && m_scriptedInput.visualDebug))
				return;
			if (windowBindings.empty() || m_planarProjections.empty())
				return;
			if (activeRenderWindowIx >= windowBindings.size())
				return;

			const auto& binding = windowBindings[activeRenderWindowIx];
			if (binding.activePlanarIx >= m_planarProjections.size())
				return;

			auto& planar = m_planarProjections[binding.activePlanarIx];
			if (!planar)
				return;
			auto* camera = planar->getCamera();
			if (!camera)
				return;

			if (!m_scriptedInput.visualActivePlanarValid)
			{
				m_scriptedInput.visualActivePlanarValid = true;
				m_scriptedInput.visualActivePlanarIx = binding.activePlanarIx;
				m_scriptedInput.visualActivePlanarStartFrame = m_realFrameIx;
			}

			const uint64_t elapsedFrames = (m_realFrameIx >= m_scriptedInput.visualActivePlanarStartFrame) ?
				(m_realFrameIx - m_scriptedInput.visualActivePlanarStartFrame) : 0ull;
			const float fps = std::max(1.f, m_scriptedInput.visualTargetFps);
			const uint64_t holdFrames = static_cast<uint64_t>(std::round(std::max(0.f, m_scriptedInput.visualCameraHoldSeconds) * fps));
			const uint64_t progressFrames = holdFrames ? std::min(elapsedFrames, holdFrames) : elapsedFrames;

			nbl::ui::SCameraScriptVisualDebugStatus debugStatus = {};
			debugStatus.cameraLabel = getCameraTypeLabel(camera);
			debugStatus.cameraHint = getCameraTypeDescription(camera);
			debugStatus.cameraIndex = binding.activePlanarIx;
			debugStatus.cameraCount = static_cast<uint32_t>(m_planarProjections.size());
			debugStatus.planarIndex = binding.activePlanarIx;
			debugStatus.hasHoldFrames = holdFrames > 0u;
			debugStatus.progressFrames = progressFrames;
			debugStatus.holdFrames = holdFrames;
			debugStatus.targetFps = fps;
			debugStatus.absoluteFrame = m_realFrameIx;
			debugStatus.segmentLabel = m_scriptedInput.visualSegmentLabel;
			debugStatus.followActive = m_scriptedInput.visualFollowActive;
			debugStatus.followModeDescription = nbl::ui::getCameraFollowModeDescription(m_scriptedInput.visualFollowMode);
			debugStatus.followLockValid = m_scriptedInput.visualFollowLockValid;
			debugStatus.followLockAngleDeg = m_scriptedInput.visualFollowLockAngleDeg;
			debugStatus.followTargetDistance = m_scriptedInput.visualFollowTargetDistance;
			debugStatus.followTargetCenterNdcRadius = m_scriptedInput.visualFollowTargetCenterNdcRadius;

			float dynamicFov = 0.0f;
			if (camera && camera->tryGetDynamicPerspectiveFov(dynamicFov))
			{
				debugStatus.hasDynamicFov = true;
				debugStatus.dynamicFovDeg = dynamicFov;
			}

			nbl::ui::drawScriptVisualDebugOverlay(displaySize, nbl::ui::buildScriptVisualDebugOverlayData(debugStatus));
			}

		inline bool tryCaptureGoal(ICamera* camera, CCameraGoal& out) const
		{
			const auto capture = m_cameraGoalSolver.captureDetailed(camera);
			out = capture.goal;
			return capture.captured;
		}

		inline PresetUiAnalysis analyzePresetForUi(ICamera* camera, const CameraPreset& preset) const
		{
			return nbl::ui::analyzePresetPresentation(m_cameraGoalSolver, camera, preset);
		}

		inline CaptureUiAnalysis analyzeCameraCaptureForUi(ICamera* camera) const
		{
			return nbl::ui::analyzeCapturePresentation(m_cameraGoalSolver, camera);
		}

		inline CCameraGoalSolver::SCompatibilityResult analyzePresetCompatibility(ICamera* camera, const CameraPreset& preset) const
		{
			return nbl::core::analyzePresetApply(m_cameraGoalSolver, camera, preset).compatibility;
		}

		inline bool presetMatchesFilter(ICamera* camera, const CameraPreset& preset) const
		{
			return analyzePresetForUi(camera, preset).matchesFilter(m_presetFilterMode);
		}

		inline CCameraGoalSolver::SApplyResult applyPresetFromUi(ICamera* camera, const CameraPreset& preset)
		{
			const auto result = nbl::core::applyPresetDetailed(m_cameraGoalSolver, camera, preset);
			if (result.succeeded())
				refreshFollowOffsetConfigsForCamera(camera);
			const auto presetUi = analyzePresetForUi(camera, preset);
			storeApplyStatusBanner(m_manualPresetApplyBanner,
				describeApplyResult(result) + " | " + presetUi.compatibilityLabel,
				result.succeeded(),
				result.approximate());
			return result;
		}

		inline void storeApplyStatusBanner(ApplyStatusBanner& banner, std::string summary, const bool succeeded, const bool approximate)
		{
			banner.summary = std::move(summary);
			banner.succeeded = succeeded;
			banner.approximate = approximate;
		}

		inline void clearApplyStatusBanner(ApplyStatusBanner& banner)
		{
			banner.summary.clear();
			banner.succeeded = false;
			banner.approximate = false;
		}

		inline void storePlaybackApplySummary(const SCameraPresetApplySummary& summary)
		{
			storeApplyStatusBanner(m_playbackApplyBanner,
				nbl::ui::describePresetApplySummary(summary, m_playbackAffectsAll ? "Playback apply | no cameras available" : "Playback apply | no active camera"),
				summary.succeeded(),
				summary.approximate());
		}

		inline void appendVirtualEventLog(std::string_view source, std::string_view inputSource, uint32_t planarIx, ICamera* camera, const CVirtualGimbalEvent* events, uint32_t count)
		{
			m_uiVirtualEventsThisFrame += count;
			const std::string sourceStr(source);
			const std::string inputSourceStr(inputSource);
			const std::string cameraName = camera ? std::string(camera->getIdentifier()) : std::string("None");
			for (uint32_t i = 0u; i < count; ++i)
			{
				const auto* eventName = CVirtualGimbalEvent::virtualEventToString(events[i].type).data();
				auto line = m_logFormatter->format(ILogger::ELL_INFO,
					"virtual frame=%llu src=%s input=%s cam=%s planar=%u event=%s mag=%.6f",
					static_cast<unsigned long long>(m_realFrameIx),
					sourceStr.c_str(),
					inputSourceStr.c_str(),
					cameraName.c_str(),
					planarIx,
					eventName,
					events[i].magnitude);
				m_virtualEventLog.push_back({
					m_realFrameIx,
					events[i].type,
					events[i].magnitude,
					sourceStr,
					inputSourceStr,
					cameraName,
					planarIx,
					std::move(line)
				});
			}

			while (m_virtualEventLog.size() > m_virtualEventLogMax)
				m_virtualEventLog.pop_front();
		}

		inline SCameraPresetApplySummary applyPresetToTargets(const CameraPreset& preset)
		{
			SCameraPresetApplySummary summary = {};
			if (!m_playbackAffectsAll)
			{
				ICamera* activeCamera = getActiveCamera();
				summary = nbl::core::applyPresetToCameraRange(m_cameraGoalSolver, std::span<ICamera* const>(&activeCamera, activeCamera ? 1u : 0u), preset);
				if (summary.succeeded())
					refreshFollowOffsetConfigsForCamera(activeCamera);
				return summary;
			}

			std::vector<ICamera*> cameras;
			cameras.reserve(windowBindings.size());
			std::unordered_set<const ICamera*> visited;
			for (auto& binding : windowBindings)
			{
				auto& planar = m_planarProjections[binding.activePlanarIx];
				if (!planar)
					continue;
				auto* camera = planar->getCamera();
				if (!camera)
					continue;
				if (visited.insert(camera).second)
					cameras.push_back(camera);
			}

			summary = nbl::core::applyPresetToCameraRange(m_cameraGoalSolver, std::span<ICamera* const>(cameras.data(), cameras.size()), preset);
			if (summary.succeeded())
				refreshAllFollowOffsetConfigs();
			return summary;
		}

		inline bool tryBuildPlaybackPresetAtTime(const float time, CameraPreset& preset)
		{
			return nbl::core::tryBuildKeyframeTrackPresetAtTime(m_keyframeTrack, time, preset);
		}

		inline bool applyPlaybackAtTime(const float time)
		{
			CameraPreset preset;
			if (!tryBuildPlaybackPresetAtTime(time, preset))
			{
				clearApplyStatusBanner(m_playbackApplyBanner);
				return false;
			}

			storePlaybackApplySummary(applyPresetToTargets(preset));
			return true;
		}

		inline void sortKeyframesByTime()
		{
			nbl::core::sortKeyframeTrackByTime(m_keyframeTrack);
		}

		inline void clampPlaybackTimeToKeyframes()
		{
			nbl::core::clampPlaybackCursorToTrack(m_keyframeTrack, m_playback);
		}

		inline int selectKeyframeNearestTime(const float time)
		{
			return nbl::core::selectKeyframeTrackNearestTime(m_keyframeTrack, time);
		}

		inline void normalizeSelectedKeyframe()
		{
			nbl::core::normalizeSelectedKeyframeTrack(m_keyframeTrack);
		}

		inline CameraKeyframe* getSelectedKeyframe()
		{
			return nbl::core::getSelectedKeyframe(m_keyframeTrack);
		}

		inline const CameraKeyframe* getSelectedKeyframe() const
		{
			return nbl::core::getSelectedKeyframe(m_keyframeTrack);
		}

		inline bool replaceSelectedKeyframeFromCamera(ICamera* camera)
		{
			auto* selected = getSelectedKeyframe();
			if (!selected)
				return false;

			CameraPreset updatedPreset;
			const auto keyframeName = selected->preset.name.empty() ? std::string("Keyframe") : selected->preset.name;
			if (!nbl::core::tryCapturePreset(m_cameraGoalSolver, camera, keyframeName, updatedPreset))
				return false;

			return nbl::core::replaceSelectedKeyframePreset(m_keyframeTrack, std::move(updatedPreset));
		}

		inline void updatePlayback(double dtSec)
		{
			const auto advance = nbl::core::advancePlaybackCursor(m_playback, m_keyframeTrack, dtSec);
			if (!advance.hasTrack || !advance.changedTime)
				return;

			applyPlaybackAtTime(m_playback.time);
		}

		bool savePresetsToFile(const nbl::system::path& path);
		bool loadPresetsFromFile(const nbl::system::path& path);
		bool saveKeyframesToFile(const nbl::system::path& path);
		bool loadKeyframesFromFile(const nbl::system::path& path);

		void imguiListen();
		void drawWindowedViewportWindows(ImGuiIO& io, SImResourceInfo& info);
		void drawFullscreenViewportWindow(ImGuiIO& io, SImResourceInfo& info);
		void refreshViewportBindingMatrices();

		inline bool shouldCaptureOSCursor()
		{
			if (!enableActiveCameraMovement || !captureCursorInMoveMode)
				return false;
			if (m_ciMode || m_scriptedInput.enabled)
				return false;
			if (!m_window || !m_window->hasInputFocus() || !m_window->hasMouseFocus())
				return false;
			return true;
		}

		inline void UpdateBoundCameraMovement()
		{
			ImGuiIO& io = ImGui::GetIO();

			if (ImGui::IsKeyPressed(ImGuiKey_Space))
				enableActiveCameraMovement = !enableActiveCameraMovement;

			if (enableActiveCameraMovement)
			{
				io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
				io.MouseDrawCursor = false;
				io.WantCaptureMouse = false;

				if (shouldCaptureOSCursor())
				{
					ImVec2 viewportSize = io.DisplaySize;
					auto* cc = m_window->getCursorControl();
					if (cc)
					{
						int32_t posX = m_window->getX();
						int32_t posY = m_window->getY();

						if (resetCursorToCenter)
						{
							const ICursorControl::SPosition middle{ static_cast<int32_t>(viewportSize.x / 2 + posX), static_cast<int32_t>(viewportSize.y / 2 + posY) };
							cc->setPosition(middle);
						}
						else
						{
							auto currentCursorPos = cc->getPosition();
							ICursorControl::SPosition newPos{};
							newPos.x = std::clamp<int32_t>(currentCursorPos.x, posX, viewportSize.x + posX);
							newPos.y = std::clamp<int32_t>(currentCursorPos.y, posY, viewportSize.y + posY);
							cc->setPosition(newPos);
						}
					}
				}
			}
			else
			{
				io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
				io.MouseDrawCursor = false;
				io.WantCaptureMouse = true;
			}
		}

		inline void UpdateCursorVisibility()
		{
			auto* cc = m_window ? m_window->getCursorControl() : nullptr;
			if (!cc)
				return;
			cc->setVisible(!shouldCaptureOSCursor());
		}

		inline void UpdateUiMetrics()
		{
			m_uiLastFrameMs = static_cast<float>(m_frameDeltaSec * 1000.0);
			m_uiLastInputEvents = m_uiInputEventsThisFrame;
			m_uiLastVirtualEvents = m_uiVirtualEventsThisFrame;

			m_uiFrameMs[m_uiMetricIndex] = m_uiLastFrameMs;
			m_uiInputCounts[m_uiMetricIndex] = static_cast<float>(m_uiInputEventsThisFrame);
			m_uiVirtualCounts[m_uiMetricIndex] = static_cast<float>(m_uiVirtualEventsThisFrame);

			m_uiMetricIndex = (m_uiMetricIndex + 1u) % UiMetricSamples;
			m_uiInputEventsThisFrame = 0u;
			m_uiVirtualEventsThisFrame = 0u;
		}

		void DrawControlPanel();

		void TransformEditorContents();

		inline void addMatrixTable(const char* topText, const char* tableName, int rows, int columns, const float* pointer, bool withSeparator = true)
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
		// Input system for capturing system events
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		// Handles mouse events
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		// Handles keyboard events
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
		//! next presentation timestamp
		std::chrono::microseconds m_nextPresentationTimestamp = {};

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

		// Demo scene object rendered into the viewports alongside the tracked target and cameras.
		nbl::hlsl::float32_t3x4 m_model = nbl::hlsl::float32_t3x4(1.f);
		CTrackedTarget m_followTarget;
		std::vector<SCameraFollowConfig> m_planarFollowConfigs;
		bool m_followTargetVisible = true;
		std::optional<uint32_t> m_followTargetGeometryIx = std::nullopt;
		SceneManipulatedObjectKind m_manipulatedObjectKind = SceneManipulatedObjectKind::Model;

		nbl::core::smart_refctd_ptr<ICamera> boundCameraToManipulate = nullptr;
		std::optional<uint32_t> boundPlanarCameraIxToManipulate = std::nullopt;

		std::vector<nbl::core::smart_refctd_ptr<planar_projection_t>> m_planarProjections;

		bool enableActiveCameraMovement = false;
		bool captureCursorInMoveMode = false;

		bool resetCursorToCenter = true;

		inline void syncWindowInputBinding(SWindowControlBinding& binding)
		{
			if (!binding.boundProjectionIx.has_value())
				return;
			if (binding.activePlanarIx >= m_planarProjections.size())
				return;

			auto& planar = m_planarProjections[binding.activePlanarIx];
			if (!planar)
				return;

			const auto projectionIx = binding.boundProjectionIx.value();
			auto& projections = planar->getPlanarProjections();
			if (projectionIx >= projections.size())
				return;

			if (binding.inputBindingPlanarIx == binding.activePlanarIx && binding.inputBindingProjectionIx == projectionIx)
				return;

			binding.inputBinding.copyBindingLayoutFrom(projections[projectionIx].getInputBinding());
			binding.inputBindingPlanarIx = binding.activePlanarIx;
			binding.inputBindingProjectionIx = projectionIx;
		}

		inline void syncWindowInputBindingToProjection(SWindowControlBinding& binding)
		{
			if (!binding.boundProjectionIx.has_value())
				return;
			if (binding.activePlanarIx >= m_planarProjections.size())
				return;

			auto& planar = m_planarProjections[binding.activePlanarIx];
			if (!planar)
				return;

			const auto projectionIx = binding.boundProjectionIx.value();
			auto& projections = planar->getPlanarProjections();
			if (projectionIx >= projections.size())
				return;

			projections[projectionIx].getInputBinding().copyBindingLayoutFrom(binding.inputBinding);
			binding.inputBindingPlanarIx = binding.activePlanarIx;
			binding.inputBindingProjectionIx = projectionIx;
		}

		struct ScriptedInputState
		{
			bool enabled = false;
			bool log = false;
			bool exclusive = false;
			bool hardFail = false;
			bool visualDebug = false;
			float visualTargetFps = 0.f;
			float visualCameraHoldSeconds = 0.f;
			CCameraScriptedTimeline timeline = {};
			size_t nextEventIndex = 0;
			CCameraScriptedCheckRuntimeState checkRuntime = {};
			size_t nextCaptureIndex = 0;
			std::string capturePrefix = "script";
			nbl::system::path captureOutputDir;
			bool failed = false;
			bool summaryReported = false;
			bool visualActivePlanarValid = false;
			uint32_t visualActivePlanarIx = 0u;
			uint64_t visualActivePlanarStartFrame = 0u;
			std::string visualSegmentLabel;
			bool visualFollowActive = false;
			ECameraFollowMode visualFollowMode = ECameraFollowMode::Disabled;
			bool visualFollowLockValid = false;
			float visualFollowLockAngleDeg = 0.0f;
			float visualFollowTargetDistance = 0.0f;
			bool visualFollowProjectedValid = false;
			float visualFollowTargetCenterNdcX = 0.0f;
			float visualFollowTargetCenterNdcY = 0.0f;
			float visualFollowTargetCenterNdcRadius = 0.0f;
			bool scriptedLeftMouseDown = false;
			bool scriptedRightMouseDown = false;
			bool framePacerInitialized = false;
			std::chrono::steady_clock::time_point framePacerNext = {};
		};

		static constexpr inline auto MaxSceneFBOs = 2u;
		std::array<SWindowControlBinding, MaxSceneFBOs> windowBindings;
		uint32_t activeRenderWindowIx = 0u;

		// UI font atlas + viewport FBO color attachment textures
		constexpr static inline auto TotalUISampleTexturesAmount = 1u + MaxSceneFBOs;

		nbl::core::smart_refctd_ptr<CGeometryCreatorScene> m_scene;
		nbl::core::smart_refctd_ptr<IGPURenderpass> m_sceneRenderpass;
		nbl::core::smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;
		nbl::core::smart_refctd_ptr<nbl::ext::frustum::CDrawFrustum> m_drawFrustum;
		std::optional<uint32_t> m_gridGeometryIx = std::nullopt;
		core::smart_refctd_ptr<IGPUGraphicsPipeline> m_spaceEnvPipeline;
		core::smart_refctd_ptr<IGPUDescriptorSetLayout> m_spaceEnvDescriptorSetLayout;
		core::smart_refctd_ptr<IDescriptorPool> m_spaceEnvDescriptorPool;
		core::smart_refctd_ptr<IGPUDescriptorSet> m_spaceEnvDescriptorSet;
		core::smart_refctd_ptr<IGPUImage> m_spaceEnvImage;
		core::smart_refctd_ptr<IGPUImageView> m_spaceEnvImageView;
		core::smart_refctd_ptr<IGPUSampler> m_spaceEnvSampler;

		CRenderUI m_ui;
		video::CDumbPresentationOracle oracle;
		uint16_t gcIndex = {};

		static constexpr uint32_t CiFramesBeforeCapture = 10u;
		static constexpr auto CiMaxRuntime = std::chrono::minutes(2);
		bool m_ciMode = false;
		bool m_ciScreenshotDone = false;
		uint32_t m_ciFrameCounter = 0u;
		nbl::system::path m_ciScreenshotPath;
		clock_t::time_point m_ciStartedAt = clock_t::time_point::min();
		bool m_scriptVisualDebugCli = false;
		bool m_disableScreenshotsCli = false;
		bool m_headlessCameraSmokeMode = false;
		bool m_headlessCameraSmokePassed = false;
		ScriptedInputState m_scriptedInput;
		CameraControlSettings m_cameraControls;
		CameraConstraintSettings m_cameraConstraints;
		core::smart_refctd_ptr<CUILogFormatter> m_logFormatter;
		std::deque<VirtualEventLogEntry> m_virtualEventLog;
		size_t m_virtualEventLogMax = 128u;
		bool m_showHud = true;
		bool m_showEventLog = false;
		bool m_logAutoScroll = true;
		bool m_logWrap = true;
		std::vector<CameraPreset> m_presets;
		std::vector<CameraPreset> m_initialPlanarPresets;
		CameraKeyframeTrack m_keyframeTrack;
		CameraPlaybackState m_playback;
		CCameraGoalSolver m_cameraGoalSolver;
		ApplyStatusBanner m_manualPresetApplyBanner;
		ApplyStatusBanner m_playbackApplyBanner;
		PresetFilterMode m_presetFilterMode = PresetFilterMode::All;
		int m_selectedPresetIx = -1;
		bool m_playbackAffectsAll = false;
		float m_newKeyframeTime = 0.f;
		char m_presetName[64] = "Preset";
		char m_presetPath[260] = "camera_presets.json";
		char m_keyframePath[260] = "camera_keyframes.json";
		std::chrono::microseconds m_lastPresentationTimestamp = {};
		bool m_haveLastPresentationTimestamp = false;
		double m_frameDeltaSec = 0.0;
		static constexpr size_t UiMetricSamples = 96u;
		std::array<float, UiMetricSamples> m_uiFrameMs = {};
		std::array<float, UiMetricSamples> m_uiInputCounts = {};
		std::array<float, UiMetricSamples> m_uiVirtualCounts = {};
		uint32_t m_uiMetricIndex = 0u;
		uint32_t m_uiVirtualEventsThisFrame = 0u;
		uint32_t m_uiInputEventsThisFrame = 0u;
		uint32_t m_uiLastInputEvents = 0u;
		uint32_t m_uiLastVirtualEvents = 0u;
		float m_uiLastFrameMs = 0.0f;

		const bool flipGizmoY = true;

		float camYAngle = 165.f / 180.f * 3.14159f;
		float camXAngle = 32.f / 180.f * 3.14159f;
		float camDistance = 8.f;
		bool useWindow = true, useSnap = false;
		ImGuizmo::OPERATION mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
		ImGuizmo::MODE mCurrentGizmoMode = ImGuizmo::LOCAL;
		float snap[3] = { 1.f, 1.f, 1.f };

		bool firstFrame = true;
		const float32_t2 iPaddingOffset = float32_t2(10, 10);

		struct ImWindowInit
		{
			float32_t2 iPos, iSize;
		};

		struct
		{
			ImWindowInit trsEditor;
			ImWindowInit planars;
			std::array<ImWindowInit, MaxSceneFBOs> renderWindows;
		} wInit;
};


#endif // _NBL_THIS_EXAMPLE_APP_HPP_

