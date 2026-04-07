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
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include "nlohmann/json.hpp"
#include "argparse/argparse.hpp"
using nbl_json = nlohmann::json;

#include "common.hpp"
#include "keysmapping.hpp"
#include "app/AppTypes.hpp"
#include "camera/CCubeProjection.hpp"
#include "glm/glm/ext/matrix_clip_space.hpp"
#include "glm/gtc/quaternion.hpp"
#include "nbl/ext/Frustum/CDrawFrustum.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"
#if __has_include("nbl/this_example/builtin/CArchive.h")
#include "nbl/this_example/builtin/CArchive.h"
#endif
#if __has_include("nbl/this_example/builtin/build/CArchive.h")
#include "nbl/this_example/builtin/build/CArchive.h"
#endif

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

/*
	Renders scene texture to an offline
	framebuffer which color attachment
	is then sampled into a imgui window.

	Written with Nabla, it's UI extension
	and got integrated with ImGuizmo to 
	handle scene's object translations.
*/

class App final : public examples::SimpleWindowedApplication
{
	using base_t = examples::SimpleWindowedApplication;
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
		core::bitflag<system::ILogger::E_LOG_LEVEL> getLogLevelMask() override
		{
			return core::bitflag<system::ILogger::E_LOG_LEVEL>(system::ILogger::ELL_INFO) |
				system::ILogger::ELL_WARNING |
				system::ILogger::ELL_PERFORMANCE |
				system::ILogger::ELL_ERROR;
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
			std::string controller;
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
			return nbl::hlsl::captureFollowOffsetsFromCamera(m_cameraGoalSolver, camera, m_followTarget, m_planarFollowConfigs[planarIx]);
		}

		inline bool followConfigUsesCapturedOffset(const SCameraFollowConfig& config) const
		{
			return config.enabled && nbl::hlsl::cameraFollowModeUsesCapturedOffset(config.mode);
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

			nbl::hlsl::captureFollowOffsetsFromCamera(m_cameraGoalSolver, camera, m_followTarget, config);
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

		inline glm::quat getDefaultFollowTargetOrientation() const
		{
			return glm::quat(1.0, 0.0, 0.0, 0.0);
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

				const auto result = nbl::hlsl::applyFollowToCamera(m_cameraGoalSolver, camera, m_followTarget, config);
				if (!result.succeeded())
					continue;

				for (auto& projection : planar->getPlanarProjections())
					nbl::hlsl::syncDynamicPerspectiveProjection(camera, projection);
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

		inline bool projectWorldPointToViewport(
			const float32_t4x4& viewProjMatrix,
			const float32_t3& worldPoint,
			const ImVec2& viewportPos,
			const ImVec2& viewportSize,
			ImVec2& outScreen) const
		{
			if (viewportSize.x <= 1.0f || viewportSize.y <= 1.0f)
				return false;

			const auto clip = mul(viewProjMatrix, float32_t4(worldPoint.x, worldPoint.y, worldPoint.z, 1.0f));
			if (!std::isfinite(clip.x) || !std::isfinite(clip.y) || !std::isfinite(clip.z) || !std::isfinite(clip.w))
				return false;

			const float absW = std::abs(clip.w);
			if (absW < 1e-5f)
				return false;

			const float invW = 1.0f / clip.w;
			const float ndcX = clip.x * invW;
			const float ndcY = clip.y * invW;
			const float ndcZ = clip.z * invW;

			if (!std::isfinite(ndcX) || !std::isfinite(ndcY) || !std::isfinite(ndcZ))
				return false;
			if (std::abs(ndcX) > 100.0f || std::abs(ndcY) > 100.0f || std::abs(ndcZ) > 100.0f)
				return false;

			outScreen.x = viewportPos.x + (ndcX * 0.5f + 0.5f) * viewportSize.x;
			outScreen.y = viewportPos.y + (-ndcY * 0.5f + 0.5f) * viewportSize.y;
			return std::isfinite(outScreen.x) && std::isfinite(outScreen.y);
		}

		inline void drawFollowTargetViewportOverlay(
			const float32_t4x4& viewProjMatrix,
			const ImVec2& viewportPos,
			const ImVec2& viewportSize) const
		{
			if (!(m_scriptedInput.enabled && m_scriptedInput.visualDebug && m_scriptedInput.visualFollowActive))
				return;
			if (viewportSize.x <= 1.0f || viewportSize.y <= 1.0f)
				return;

			float ndcX = 0.0f;
			float ndcY = 0.0f;
			float ndcRadius = 0.0f;
			if (!nbl::hlsl::tryComputeProjectedFollowTargetMetrics(viewProjMatrix, m_followTarget, ndcX, ndcY, &ndcRadius))
				return;

			auto* drawList = ImGui::GetWindowDrawList();
			if (!drawList)
				return;

			const ImVec2 center(
				viewportPos.x + viewportSize.x * 0.5f,
				viewportPos.y + viewportSize.y * 0.5f);
			const ImVec2 target(
				viewportPos.x + (ndcX * 0.5f + 0.5f) * viewportSize.x,
				viewportPos.y + (-ndcY * 0.5f + 0.5f) * viewportSize.y);

			const bool centered = ndcRadius <= 0.03f;
			const ImU32 centerColor = IM_COL32(255, 170, 72, 235);
			const ImU32 targetColor = centered ? IM_COL32(64, 255, 164, 245) : IM_COL32(90, 220, 255, 245);
			const ImU32 targetFillColor = centered ? IM_COL32(24, 120, 76, 120) : IM_COL32(20, 92, 124, 120);
			const ImU32 lineColor = centered ? IM_COL32(96, 255, 186, 200) : IM_COL32(120, 220, 255, 200);
			const float centerRadius = 16.0f;
			const float targetRadius = centered ? 18.0f : 14.0f;

			drawList->AddCircle(center, centerRadius, centerColor, 32, 2.5f);
			drawList->AddLine(ImVec2(center.x - 22.0f, center.y), ImVec2(center.x + 22.0f, center.y), centerColor, 2.0f);
			drawList->AddLine(ImVec2(center.x, center.y - 22.0f), ImVec2(center.x, center.y + 22.0f), centerColor, 2.0f);

			drawList->AddLine(center, target, lineColor, 2.0f);
			drawList->AddCircleFilled(target, targetRadius, targetFillColor, 24);
			drawList->AddCircle(target, targetRadius, targetColor, 32, 2.5f);
			drawList->AddLine(ImVec2(target.x - 14.0f, target.y), ImVec2(target.x + 14.0f, target.y), targetColor, 2.0f);
			drawList->AddLine(ImVec2(target.x, target.y - 14.0f), ImVec2(target.x, target.y + 14.0f), targetColor, 2.0f);

			drawList->AddText(ImVec2(target.x + 16.0f, target.y - 28.0f), targetColor, "FOLLOW TARGET");
		}

		inline void drawWorldReferenceOverlay(
			const ImVec2& viewportPos,
			const ImVec2& viewportSize,
			const float32_t4x4& viewMatrix,
			const float32_t4x4& projectionMatrix,
			bool leftHandedProjection,
			float nearPlane,
			float farPlane)
		{
			if (!(m_scriptedInput.enabled && m_scriptedInput.visualDebug))
				return;
			if (viewportSize.x <= 1.0f || viewportSize.y <= 1.0f)
				return;

			auto* drawList = ImGui::GetWindowDrawList();
			if (!drawList)
				return;

			const float safeNear = std::max(nearPlane, 0.001f);
			const float safeFar = std::max(farPlane, safeNear + 0.001f);
			const auto depthOfViewPoint = [&](const float32_t4& viewPoint) -> float
			{
				return leftHandedProjection ? viewPoint.z : -viewPoint.z;
			};
			const auto ndcToViewport = [&](const ImVec2& ndc) -> ImVec2
			{
				return ImVec2(
					viewportPos.x + (ndc.x * 0.5f + 0.5f) * viewportSize.x,
					viewportPos.y + (-ndc.y * 0.5f + 0.5f) * viewportSize.y);
			};
			const auto clipSegmentByDepthRange = [&](float32_t4& viewA, float32_t4& viewB) -> bool
			{
				const float32_t4 a0 = viewA;
				const float32_t4 b0 = viewB;
				const float32_t4 delta = b0 - a0;
				const float depthA = depthOfViewPoint(a0);
				const float depthB = depthOfViewPoint(b0);

				float tEnter = 0.0f;
				float tExit = 1.0f;
				const auto clipByConstraint = [&](float fa, float fb) -> bool
				{
					if (fa < 0.0f && fb < 0.0f)
						return false;
					if (fa >= 0.0f && fb >= 0.0f)
						return true;

					const float denom = fa - fb;
					if (std::abs(denom) < 1e-6f)
						return false;
					const float t = std::clamp(fa / denom, 0.0f, 1.0f);

					if (fa < 0.0f)
						tEnter = std::max(tEnter, t);
					else
						tExit = std::min(tExit, t);

					return tEnter <= tExit;
				};

				if (!clipByConstraint(depthA - safeNear, depthB - safeNear))
					return false;
				if (!clipByConstraint(safeFar - depthA, safeFar - depthB))
					return false;

				viewA = a0 + delta * tEnter;
				viewB = a0 + delta * tExit;
				return true;
			};
			const auto projectViewPointToNdc = [&](const float32_t4& viewPoint, ImVec2& outNdc) -> bool
			{
				const auto clip = mul(projectionMatrix, viewPoint);
				if (!std::isfinite(clip.x) || !std::isfinite(clip.y) || !std::isfinite(clip.z) || !std::isfinite(clip.w))
					return false;

				const float absW = std::abs(clip.w);
				if (absW < 1e-6f)
					return false;

				const float invW = 1.0f / clip.w;
				const float ndcX = clip.x * invW;
				const float ndcY = clip.y * invW;
				const float ndcZ = clip.z * invW;
				if (!std::isfinite(ndcX) || !std::isfinite(ndcY) || !std::isfinite(ndcZ))
					return false;
				if (std::abs(ndcX) > 1e4f || std::abs(ndcY) > 1e4f || std::abs(ndcZ) > 1e4f)
					return false;

				outNdc = ImVec2(ndcX, ndcY);
				return true;
			};
			const auto clipNdcSegmentToViewport = [&](ImVec2& ndcA, ImVec2& ndcB) -> bool
			{
				float tEnter = 0.0f;
				float tExit = 1.0f;
				const float dx = ndcB.x - ndcA.x;
				const float dy = ndcB.y - ndcA.y;
				const auto clipTest = [&](float p, float q) -> bool
				{
					if (std::abs(p) < 1e-6f)
						return q >= 0.0f;

					const float r = q / p;
					if (p < 0.0f)
					{
						if (r > tExit)
							return false;
						tEnter = std::max(tEnter, r);
					}
					else
					{
						if (r < tEnter)
							return false;
						tExit = std::min(tExit, r);
					}
					return tEnter <= tExit;
				};

				if (!clipTest(-dx, ndcA.x + 1.0f))
					return false;
				if (!clipTest(dx, 1.0f - ndcA.x))
					return false;
				if (!clipTest(-dy, ndcA.y + 1.0f))
					return false;
				if (!clipTest(dy, 1.0f - ndcA.y))
					return false;

				const ImVec2 a0 = ndcA;
				ndcA = ImVec2(a0.x + dx * tEnter, a0.y + dy * tEnter);
				ndcB = ImVec2(a0.x + dx * tExit, a0.y + dy * tExit);
				return true;
			};
			const auto projectWorldPointToViewportClipped = [&](const float32_t3& worldPoint, ImVec2& outScreen) -> bool
			{
				const auto viewPoint = mul(viewMatrix, float32_t4(worldPoint.x, worldPoint.y, worldPoint.z, 1.0f));
				if (!std::isfinite(viewPoint.x) || !std::isfinite(viewPoint.y) || !std::isfinite(viewPoint.z) || !std::isfinite(viewPoint.w))
					return false;

				const float depth = depthOfViewPoint(viewPoint);
				if (depth < safeNear || depth > safeFar)
					return false;

				ImVec2 ndcPoint = {};
				if (!projectViewPointToNdc(viewPoint, ndcPoint))
					return false;
				if (ndcPoint.x < -1.0f || ndcPoint.x > 1.0f || ndcPoint.y < -1.0f || ndcPoint.y > 1.0f)
					return false;

				outScreen = ndcToViewport(ndcPoint);
				return std::isfinite(outScreen.x) && std::isfinite(outScreen.y);
			};

			const auto drawProjectedSegment = [&](const float32_t3& aWorld, const float32_t3& bWorld, ImU32 color, float thickness) -> void
			{
				float32_t4 viewA = mul(viewMatrix, float32_t4(aWorld.x, aWorld.y, aWorld.z, 1.0f));
				float32_t4 viewB = mul(viewMatrix, float32_t4(bWorld.x, bWorld.y, bWorld.z, 1.0f));
				if (!std::isfinite(viewA.x) || !std::isfinite(viewA.y) || !std::isfinite(viewA.z) || !std::isfinite(viewA.w) ||
					!std::isfinite(viewB.x) || !std::isfinite(viewB.y) || !std::isfinite(viewB.z) || !std::isfinite(viewB.w))
					return;
				if (!clipSegmentByDepthRange(viewA, viewB))
					return;

				ImVec2 ndcA = {};
				ImVec2 ndcB = {};
				if (!projectViewPointToNdc(viewA, ndcA))
					return;
				if (!projectViewPointToNdc(viewB, ndcB))
					return;
				if (!clipNdcSegmentToViewport(ndcA, ndcB))
					return;

				const ImVec2 screenA = ndcToViewport(ndcA);
				const ImVec2 screenB = ndcToViewport(ndcB);
				drawList->AddLine(screenA, screenB, color, thickness);
			};

			auto drawWorldLine = [&](const float32_t3& aWorld, const float32_t3& bWorld, ImU32 color, float thickness) -> void
			{
				drawProjectedSegment(aWorld, bWorld, color, thickness);
			};

			const float32_t3 origin = float32_t3(0.0f);
			ImVec2 originScreen = {};
			if (!projectWorldPointToViewportClipped(origin, originScreen))
				return;

			constexpr float axisLength = 5.0f;
			const float32_t3 xPos = float32_t3(axisLength, 0.0f, 0.0f);
			const float32_t3 yPos = float32_t3(0.0f, axisLength, 0.0f);
			const float32_t3 zPos = float32_t3(0.0f, 0.0f, axisLength);
			const float32_t3 xNeg = float32_t3(-axisLength * 0.4f, 0.0f, 0.0f);
			const float32_t3 yNeg = float32_t3(0.0f, -axisLength * 0.3f, 0.0f);
			const float32_t3 zNeg = float32_t3(0.0f, 0.0f, -axisLength * 0.4f);

			drawWorldLine(origin, xPos, IM_COL32(244, 92, 92, 245), 2.8f);
			drawWorldLine(origin, yPos, IM_COL32(124, 236, 132, 245), 2.8f);
			drawWorldLine(origin, zPos, IM_COL32(106, 166, 255, 245), 2.8f);
			drawWorldLine(origin, xNeg, IM_COL32(128, 74, 74, 170), 1.4f);
			drawWorldLine(origin, yNeg, IM_COL32(74, 128, 78, 170), 1.4f);
			drawWorldLine(origin, zNeg, IM_COL32(70, 88, 124, 170), 1.4f);

			const auto drawAxisArrowHead = [&](const float32_t3& tipWorld, const float32_t3& tailWorld, ImU32 color) -> void
			{
				ImVec2 tipScreen = {};
				ImVec2 tailScreen = {};
				if (!projectWorldPointToViewportClipped(tipWorld, tipScreen) || !projectWorldPointToViewportClipped(tailWorld, tailScreen))
					return;

				const ImVec2 dir = ImVec2(tipScreen.x - tailScreen.x, tipScreen.y - tailScreen.y);
				const float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
				if (len < 1e-3f)
					return;

				const ImVec2 n = ImVec2(dir.x / len, dir.y / len);
				const ImVec2 ortho = ImVec2(-n.y, n.x);
				const float headLength = 9.0f;
				const float headHalfWidth = 4.5f;

				const ImVec2 base = ImVec2(tipScreen.x - n.x * headLength, tipScreen.y - n.y * headLength);
				const ImVec2 left = ImVec2(base.x + ortho.x * headHalfWidth, base.y + ortho.y * headHalfWidth);
				const ImVec2 right = ImVec2(base.x - ortho.x * headHalfWidth, base.y - ortho.y * headHalfWidth);
				drawList->AddTriangleFilled(tipScreen, left, right, color);
			};

			drawAxisArrowHead(xPos, float32_t3(axisLength - 0.55f, 0.0f, 0.0f), IM_COL32(255, 162, 162, 255));
			drawAxisArrowHead(yPos, float32_t3(0.0f, axisLength - 0.55f, 0.0f), IM_COL32(186, 255, 192, 255));
			drawAxisArrowHead(zPos, float32_t3(0.0f, 0.0f, axisLength - 0.55f), IM_COL32(178, 216, 255, 255));

			auto drawAxisLabel = [&](const char* label, const float32_t3& worldPoint, ImU32 color) -> void
			{
				ImVec2 screenPos = {};
				if (!projectWorldPointToViewportClipped(worldPoint, screenPos))
					return;
				drawList->AddText(ImVec2(screenPos.x + 4.0f, screenPos.y + 3.0f), color, label);
			};

			drawList->AddCircleFilled(originScreen, 4.0f, IM_COL32(240, 248, 255, 220), 16);

			drawAxisLabel("X", xPos, IM_COL32(255, 152, 152, 255));
			drawAxisLabel("Y", yPos, IM_COL32(172, 255, 178, 255));
			drawAxisLabel("Z", zPos, IM_COL32(172, 210, 255, 255));
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

				const auto cameraLabel = getCameraTypeLabel(camera);
				const auto cameraHint = getCameraTypeDescription(camera);
				std::string lineTop = "SCRIPT VISUAL DEBUG";
				std::string lineMid = "Camera " + std::to_string(binding.activePlanarIx + 1u) + "/" + std::to_string(m_planarProjections.size()) + "  " + std::string(cameraLabel);

				char lineBottomBuffer[256] = {};
				if (holdFrames)
			{
				const double elapsedSeconds = static_cast<double>(progressFrames) / static_cast<double>(fps);
				const double holdSeconds = static_cast<double>(holdFrames) / static_cast<double>(fps);
				std::snprintf(
					lineBottomBuffer,
					sizeof(lineBottomBuffer),
					"Planar %u  Segment %.1f/%.1f s  Frame %llu/%llu",
					binding.activePlanarIx,
					elapsedSeconds,
					holdSeconds,
					static_cast<unsigned long long>(progressFrames),
					static_cast<unsigned long long>(holdFrames));
			}
			else
			{
				std::snprintf(
					lineBottomBuffer,
					sizeof(lineBottomBuffer),
					"Planar %u  Frame %llu",
					binding.activePlanarIx,
					static_cast<unsigned long long>(m_realFrameIx));
				}
				std::string lineBottom(lineBottomBuffer);
				if (!m_scriptedInput.visualSegmentLabel.empty())
					lineBottom += "  |  " + m_scriptedInput.visualSegmentLabel;
				std::string lineHint = std::string(cameraHint);
				float dynamicFov = 0.0f;
				if (camera && camera->tryGetDynamicPerspectiveFov(dynamicFov))
				{
					char fovBuffer[96] = {};
					std::snprintf(fovBuffer, sizeof(fovBuffer), "  |  Dynamic FOV %.2f deg", dynamicFov);
					lineHint += fovBuffer;
				}
				if (m_scriptedInput.visualFollowActive)
				{
					lineHint += "  |  " + std::string(nbl::hlsl::getCameraFollowModeDescription(m_scriptedInput.visualFollowMode));
					if (m_scriptedInput.visualFollowLockValid)
					{
						char followBuffer[192] = {};
						std::snprintf(
							followBuffer,
							sizeof(followBuffer),
							"  |  lock %.2f deg  |  target %.2f  |  center err %.3f",
							m_scriptedInput.visualFollowLockAngleDeg,
							m_scriptedInput.visualFollowTargetDistance,
							m_scriptedInput.visualFollowTargetCenterNdcRadius);
						lineHint += followBuffer;
					}
					else
					{
						lineHint += "  |  lock n/a  |  target n/a  |  center err n/a";
					}
				}
				else
				{
					lineHint += "  |  Follow off";
				}

				const float topSize = 50.f;
				const float midSize = 38.f;
				const float bottomSize = 28.f;
				const float hintSize = 24.f;
				const float marginTop = 18.f;
				const float padX = 24.f;
				const float padY = 16.f;
				const float gap = 6.f;

			ImFont* font = ImGui::GetFont();
			if (!font)
				return;

				const float textWrap = std::numeric_limits<float>::max();
				const ImVec2 topTextSize = font->CalcTextSizeA(topSize, textWrap, 0.0f, lineTop.c_str());
				const ImVec2 midTextSize = font->CalcTextSizeA(midSize, textWrap, 0.0f, lineMid.c_str());
				const ImVec2 bottomTextSize = font->CalcTextSizeA(bottomSize, textWrap, 0.0f, lineBottom.c_str());
				const ImVec2 hintTextSize = font->CalcTextSizeA(hintSize, textWrap, 0.0f, lineHint.c_str());
				const float panelWidth = std::max(std::max(topTextSize.x, midTextSize.x), std::max(bottomTextSize.x, hintTextSize.x)) + padX * 2.0f;
				const float panelHeight = topTextSize.y + midTextSize.y + bottomTextSize.y + hintTextSize.y + gap * 3.0f + padY * 2.0f;
				const ImVec2 panelMin((displaySize.x - panelWidth) * 0.5f, marginTop);
				const ImVec2 panelMax(panelMin.x + panelWidth, panelMin.y + panelHeight);

			auto* drawList = ImGui::GetForegroundDrawList();
			if (!drawList)
				return;

			drawList->AddRectFilled(panelMin, panelMax, IM_COL32(6, 8, 12, 232), 14.0f);
			drawList->AddRect(panelMin, panelMax, IM_COL32(255, 166, 64, 255), 14.0f, 0, 2.5f);

				const float topX = panelMin.x + (panelWidth - topTextSize.x) * 0.5f;
				const float midX = panelMin.x + (panelWidth - midTextSize.x) * 0.5f;
				const float bottomX = panelMin.x + (panelWidth - bottomTextSize.x) * 0.5f;
				const float hintX = panelMin.x + (panelWidth - hintTextSize.x) * 0.5f;
				const float topY = panelMin.y + padY;
				const float midY = topY + topTextSize.y + gap;
				const float bottomY = midY + midTextSize.y + gap;
				const float hintY = bottomY + bottomTextSize.y + gap;

				drawList->AddText(font, topSize, ImVec2(topX, topY), IM_COL32(255, 206, 120, 255), lineTop.c_str());
				drawList->AddText(font, midSize, ImVec2(midX, midY), IM_COL32(255, 244, 224, 255), lineMid.c_str());
				drawList->AddText(font, bottomSize, ImVec2(bottomX, bottomY), IM_COL32(202, 222, 255, 255), lineBottom.c_str());
				drawList->AddText(font, hintSize, ImVec2(hintX, hintY), IM_COL32(170, 204, 255, 255), lineHint.c_str());
			}

		inline bool tryCaptureGoal(ICamera* camera, CCameraGoal& out) const
		{
			const auto capture = m_cameraGoalSolver.captureDetailed(camera);
			out = capture.goal;
			return capture.captured;
		}

		inline PresetUiAnalysis analyzePresetForUi(ICamera* camera, const CameraPreset& preset) const
		{
			return nbl::hlsl::analyzePresetPresentation(m_cameraGoalSolver, camera, preset);
		}

		inline CaptureUiAnalysis analyzeCameraCaptureForUi(ICamera* camera) const
		{
			return nbl::hlsl::analyzeCapturePresentation(m_cameraGoalSolver, camera);
		}

		inline CCameraGoalSolver::SCompatibilityResult analyzePresetCompatibility(ICamera* camera, const CameraPreset& preset) const
		{
			return nbl::hlsl::analyzePresetApply(m_cameraGoalSolver, camera, preset).compatibility;
		}

		inline bool presetMatchesFilter(ICamera* camera, const CameraPreset& preset) const
		{
			return analyzePresetForUi(camera, preset).matchesFilter(m_presetFilterMode);
		}

		inline CCameraGoalSolver::SApplyResult applyPresetFromUi(ICamera* camera, const CameraPreset& preset)
		{
			const auto result = nbl::hlsl::applyPresetDetailed(m_cameraGoalSolver, camera, preset);
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
				nbl::hlsl::describePresetApplySummary(summary, m_playbackAffectsAll ? "Playback apply | no cameras available" : "Playback apply | no active camera"),
				summary.succeeded(),
				summary.approximate());
		}

		inline void appendVirtualEventLog(std::string_view source, std::string_view controller, uint32_t planarIx, ICamera* camera, const CVirtualGimbalEvent* events, uint32_t count)
		{
			m_uiVirtualEventsThisFrame += count;
			const std::string sourceStr(source);
			const std::string controllerStr(controller);
			const std::string cameraName = camera ? std::string(camera->getIdentifier()) : std::string("None");
			for (uint32_t i = 0u; i < count; ++i)
			{
				const auto* eventName = CVirtualGimbalEvent::virtualEventToString(events[i].type).data();
				auto line = m_logFormatter->format(ILogger::ELL_INFO,
					"virtual frame=%llu src=%s ctrl=%s cam=%s planar=%u event=%s mag=%.6f",
					static_cast<unsigned long long>(m_realFrameIx),
					sourceStr.c_str(),
					controllerStr.c_str(),
					cameraName.c_str(),
					planarIx,
					eventName,
					events[i].magnitude);
				m_virtualEventLog.push_back({
					m_realFrameIx,
					events[i].type,
					events[i].magnitude,
					sourceStr,
					controllerStr,
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
				summary = nbl::hlsl::applyPresetToCameraRange(m_cameraGoalSolver, std::span<ICamera* const>(&activeCamera, activeCamera ? 1u : 0u), preset);
				if (summary.succeeded())
					refreshFollowOffsetConfigsForCamera(activeCamera);
				return summary;
			}

			std::vector<ICamera*> cameras;
			cameras.reserve(windowBindings.size());
			std::unordered_set<uintptr_t> visited;
			for (auto& binding : windowBindings)
			{
				auto& planar = m_planarProjections[binding.activePlanarIx];
				if (!planar)
					continue;
				auto* camera = planar->getCamera();
				if (!camera)
					continue;
				const auto id = camera->getGimbal().getID();
				if (visited.insert(id).second)
					cameras.push_back(camera);
			}

			summary = nbl::hlsl::applyPresetToCameraRange(m_cameraGoalSolver, std::span<ICamera* const>(cameras.data(), cameras.size()), preset);
			if (summary.succeeded())
				refreshAllFollowOffsetConfigs();
			return summary;
		}

		inline bool tryBuildPlaybackPresetAtTime(const float time, CameraPreset& preset)
		{
			return nbl::hlsl::tryBuildKeyframeTrackPresetAtTime(m_keyframeTrack, time, preset);
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
			nbl::hlsl::sortKeyframeTrackByTime(m_keyframeTrack);
		}

		inline void clampPlaybackTimeToKeyframes()
		{
			nbl::hlsl::clampPlaybackCursorToTrack(m_keyframeTrack, m_playback);
		}

		inline int selectKeyframeNearestTime(const float time)
		{
			return nbl::hlsl::selectKeyframeTrackNearestTime(m_keyframeTrack, time);
		}

		inline void normalizeSelectedKeyframe()
		{
			nbl::hlsl::normalizeSelectedKeyframeTrack(m_keyframeTrack);
		}

		inline CameraKeyframe* getSelectedKeyframe()
		{
			return nbl::hlsl::getSelectedKeyframe(m_keyframeTrack);
		}

		inline const CameraKeyframe* getSelectedKeyframe() const
		{
			return nbl::hlsl::getSelectedKeyframe(m_keyframeTrack);
		}

		inline bool replaceSelectedKeyframeFromCamera(ICamera* camera)
		{
			auto* selected = getSelectedKeyframe();
			if (!selected)
				return false;

			CameraPreset updatedPreset;
			const auto keyframeName = selected->preset.name.empty() ? std::string("Keyframe") : selected->preset.name;
			if (!nbl::hlsl::tryCapturePreset(m_cameraGoalSolver, camera, keyframeName, updatedPreset))
				return false;

			return nbl::hlsl::replaceSelectedKeyframePreset(m_keyframeTrack, std::move(updatedPreset));
		}

		inline void updatePlayback(double dtSec)
		{
			const auto advance = nbl::hlsl::advancePlaybackCursor(m_playback, m_keyframeTrack, dtSec);
			if (!advance.hasTrack || !advance.changedTime)
				return;

			applyPlaybackAtTime(m_playback.time);
		}

		inline bool savePresetsToFile(const system::path& path)
		{
			return nbl::hlsl::savePresetCollectionToFile(path, std::span<const CameraPreset>(m_presets.data(), m_presets.size()));
		}

		inline bool loadPresetsFromFile(const system::path& path)
		{
			return nbl::hlsl::loadPresetCollectionFromFile(path, m_presets);
		}

		inline bool saveKeyframesToFile(const system::path& path)
		{
			return nbl::hlsl::saveKeyframeTrackToFile(path, m_keyframeTrack);
		}

		inline bool loadKeyframesFromFile(const system::path& path)
		{
			if (!nbl::hlsl::loadKeyframeTrackFromFile(path, m_keyframeTrack))
				return false;

			clampPlaybackTimeToKeyframes();
			if (m_keyframeTrack.keyframes.empty())
				clearApplyStatusBanner(m_playbackApplyBanner);
			return true;
		}

		void imguiListen();

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

		inline void DrawBadge(const char* label, const ImVec4& bg, const ImVec4& fg)
		{
			ImGui::PushStyleColor(ImGuiCol_Button, bg);
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bg);
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, bg);
			ImGui::PushStyleColor(ImGuiCol_Text, fg);
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 2.0f));
			ImGui::Button(label);
			ImGui::PopStyleVar();
			ImGui::PopStyleColor(4);
		}

		inline void DrawKeyHint(const char* label, const ImVec4& bg, const ImVec4& fg)
		{
			ImGui::PushStyleColor(ImGuiCol_Button, bg);
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bg);
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, bg);
			ImGui::PushStyleColor(ImGuiCol_Text, fg);
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 1.0f));
			ImGui::SmallButton(label);
			ImGui::PopStyleVar();
			ImGui::PopStyleColor(4);
		}

		inline void DrawHoverHint(const char* text)
		{
			if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
			{
				ImGui::BeginTooltip();
				ImGui::TextUnformatted(text);
				ImGui::EndTooltip();
			}
		}

		inline void DrawDot(const ImVec4& color)
		{
			ImVec2 p = ImGui::GetCursorScreenPos();
			const float radius = 3.5f;
			ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + radius, p.y + radius + 1.0f), radius, ImGui::ColorConvertFloat4ToU32(color));
			ImGui::Dummy(ImVec2(radius * 2.0f + 2.0f, radius * 2.0f));
			ImGui::SameLine(0, 6.0f);
		}

		inline void DrawSectionHeader(const char* id, const char* label, const ImVec4& accent)
		{
			ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
			ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.14f, 0.18f, 0.22f, 0.52f));
			if (ImGui::BeginChild(id, ImVec2(0, 20), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
			{
				ImVec2 p = ImGui::GetWindowPos();
				ImVec2 s = ImGui::GetWindowSize();
				ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + 2.0f, p.y + s.y), ImGui::ColorConvertFloat4ToU32(accent), 4.0f);
				ImGui::SetCursorPosX(8.0f);
				ImGui::AlignTextToFramePadding();
				ImGui::TextColored(accent, "%s", label);
			}
			ImGui::EndChild();
			ImGui::PopStyleColor();
			ImGui::PopStyleVar();
			ImGui::Spacing();
		}

		inline float CalcCardHeight(int rows) const
		{
			return ImGui::GetFrameHeightWithSpacing() * (static_cast<float>(rows) + 1.0f) + 10.0f;
		}

		inline bool BeginCard(const char* id, float height, const ImVec4& top, const ImVec4& bottom, const ImVec4& border)
		{
			ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 8.0f));
			ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
			const bool open = ImGui::BeginChild(id, ImVec2(0, height), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
			ImVec2 p = ImGui::GetWindowPos();
			ImVec2 s = ImGui::GetWindowSize();
			const ImU32 colTop = ImGui::ColorConvertFloat4ToU32(top);
			const ImU32 colBottom = ImGui::ColorConvertFloat4ToU32(bottom);
			ImGui::GetWindowDrawList()->AddRectFilledMultiColor(
				p,
				ImVec2(p.x + s.x, p.y + s.y),
				colTop,
				colTop,
				colBottom,
				colBottom
			);
			ImGui::GetWindowDrawList()->AddRect(p, ImVec2(p.x + s.x, p.y + s.y), ImGui::ColorConvertFloat4ToU32(border), 6.0f);
			return open;
		}

		inline void EndCard()
		{
			ImGui::EndChild();
			ImGui::PopStyleColor();
			ImGui::PopStyleVar(2);
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

		// one model object in the world, testing multiuple cameraz for which view is rendered to separate frame buffers (so what they see) with new controller API including imguizmo
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

		struct windowControlBinding
		{
			nbl::core::smart_refctd_ptr<IGPUFramebuffer> sceneFramebuffer;
			nbl::core::smart_refctd_ptr<IGPUImageView> sceneColorView;
			nbl::core::smart_refctd_ptr<IGPUImageView> sceneDepthView;
			float32_t3x4 viewMatrix = float32_t3x4(1.f);
			float32_t4x4 projectionMatrix = float32_t4x4(1.f);
			float32_t4x4 viewProjMatrix = float32_t4x4(1.f);

			uint32_t activePlanarIx = 0u;
			bool allowGizmoAxesToFlip = false;
			bool enableDebugGridDraw = true;
			bool isOrthographicProjection = false;
			float aspectRatio = 16.f / 9.f;
			bool leftHandedProjection = true;
			CGimbalInputBinder inputBinding;

			std::optional<uint32_t> boundProjectionIx = std::nullopt, lastBoundPerspectivePresetProjectionIx = std::nullopt, lastBoundOrthoPresetProjectionIx = std::nullopt;
			std::optional<uint32_t> inputBindingProjectionIx = std::nullopt;
			uint32_t inputBindingPlanarIx = std::numeric_limits<uint32_t>::max();

			inline void pickDefaultProjections(const planar_projections_range_t& projections)
			{
				auto init = [&](std::optional<uint32_t>& presetix, IPlanarProjection::CProjection::ProjectionType requestedType) -> void
				{
					for (uint32_t i = 0u; i < projections.size(); ++i)
					{
						const auto& params = projections[i].getParameters();
						if (params.m_type == requestedType)
						{
							presetix = i;
							break;
						}
					}

					assert(presetix.has_value());
				};

				init(lastBoundPerspectivePresetProjectionIx = std::nullopt, IPlanarProjection::CProjection::Perspective);
				init(lastBoundOrthoPresetProjectionIx = std::nullopt, IPlanarProjection::CProjection::Orthographic);
				boundProjectionIx = lastBoundPerspectivePresetProjectionIx.value();
				inputBindingProjectionIx = std::nullopt;
				inputBindingPlanarIx = std::numeric_limits<uint32_t>::max();
			}
		};

		inline void syncWindowInputBinding(windowControlBinding& binding)
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

		inline void syncWindowInputBindingToProjection(windowControlBinding& binding)
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
			system::path captureOutputDir;
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
		std::array<windowControlBinding, MaxSceneFBOs> windowBindings;
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
		system::path m_ciScreenshotPath;
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

