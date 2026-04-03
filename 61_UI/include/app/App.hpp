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
#include "glm/glm/ext/matrix_clip_space.hpp" // TODO: TESTING
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
					if (m_scriptedInput.nextCaptureIndex < m_scriptedInput.captureFrames.size())
						return true;
					if (m_scriptedInput.nextEventIndex < m_scriptedInput.events.size())
						return true;
					if (m_scriptedInput.nextCheckIndex < m_scriptedInput.checks.size())
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

		struct CameraPreset
		{
			std::string name;
			std::string identifier;
			float64_t3 position = float64_t3(0.0);
			glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
			float64_t3 targetPosition = float64_t3(0.0);
			bool hasTargetPosition = false;
			float distance = 0.f;
			bool hasDistance = false;
			double orbitU = 0.0;
			double orbitV = 0.0;
			float orbitDistance = 0.f;
			bool hasOrbitState = false;
			float dynamicBaseFov = 0.f;
			float dynamicReferenceDistance = 0.f;
			bool hasDynamicPerspectiveState = false;
		};

		struct CameraKeyframe
		{
			CameraPreset preset;
			float time = 0.f;
		};

		struct CameraPlaybackState
		{
			bool playing = false;
			bool loop = true;
			bool overrideInput = true;
			float speed = 1.f;
			float time = 0.f;
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

		struct CameraConstraintSettings
		{
			bool enabled = false;
			bool clampPitch = false;
			bool clampYaw = false;
			bool clampRoll = false;
			bool clampDistance = false;
			float pitchMinDeg = -80.f;
			float pitchMaxDeg = 80.f;
			float yawMinDeg = -180.f;
			float yawMaxDeg = 180.f;
			float rollMinDeg = -180.f;
			float rollMaxDeg = 180.f;
			float minDistance = 0.1f;
			float maxDistance = 1000.f;
		};

		inline ICamera* getActiveCamera()
		{
			auto& binding = windowBindings[activeRenderWindowIx];
			auto& planar = m_planarProjections[binding.activePlanarIx];
			return planar ? planar->getCamera() : nullptr;
		}

		inline bool isOrbitLikeCamera(ICamera* camera)
		{
			return camera && camera->hasCapability(ICamera::SphericalTarget);
		}

		inline std::string_view getCameraTypeLabel(const ICamera* camera) const
		{
			if (!camera)
				return "Unknown";

			switch (camera->getKind())
			{
				case ICamera::CameraKind::FPS: return "FPS";
				case ICamera::CameraKind::Free: return "Free";
				case ICamera::CameraKind::Orbit: return "Orbit";
				case ICamera::CameraKind::Arcball: return "Arcball";
				case ICamera::CameraKind::Turntable: return "Turntable";
				case ICamera::CameraKind::TopDown: return "TopDown";
				case ICamera::CameraKind::Isometric: return "Isometric";
				case ICamera::CameraKind::Chase: return "Chase";
				case ICamera::CameraKind::Dolly: return "Dolly";
				case ICamera::CameraKind::DollyZoom: return "Dolly Zoom";
				case ICamera::CameraKind::Path: return "Path";
				default: return "Unknown";
			}
		}

		inline std::string_view getCameraTypeDescription(const ICamera* camera) const
		{
			if (!camera)
				return "Unspecified camera behavior";

			switch (camera->getKind())
			{
				case ICamera::CameraKind::FPS: return "First-person WASD + mouse look";
				case ICamera::CameraKind::Free: return "Free-fly 6DOF with full rotation";
				case ICamera::CameraKind::Orbit: return "Orbit around target with dolly";
				case ICamera::CameraKind::Arcball: return "Arcball trackball around target";
				case ICamera::CameraKind::Turntable: return "Turntable yaw/pitch around target";
				case ICamera::CameraKind::TopDown: return "Fixed pitch top-down pan";
				case ICamera::CameraKind::Isometric: return "Fixed isometric view with pan";
				case ICamera::CameraKind::Chase: return "Target follow with chase controls";
				case ICamera::CameraKind::Dolly: return "Rig truck/dolly with look-at";
				case ICamera::CameraKind::DollyZoom: return "Orbit with dolly-zoom FOV";
				case ICamera::CameraKind::Path: return "Move along a target path";
				default: return "Unspecified camera behavior";
			}
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
				const std::string lineBottom(lineBottomBuffer);
				std::string lineHint = std::string(cameraHint);
				float dynamicFov = 0.0f;
				if (camera && camera->tryGetDynamicPerspectiveFov(dynamicFov))
				{
					char fovBuffer[96] = {};
					std::snprintf(fovBuffer, sizeof(fovBuffer), "  |  Dynamic FOV %.2f deg", dynamicFov);
					lineHint += fovBuffer;
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

		inline void applyDollyZoomProjection(ICamera* camera, IPlanarProjection::CProjection& projection)
		{
			if (!camera)
				return;
			const auto& params = projection.getParameters();
			if (params.m_type != IPlanarProjection::CProjection::Perspective)
				return;
			float dynamicFov = 0.0f;
			if (!camera->tryGetDynamicPerspectiveFov(dynamicFov))
				return;
			projection.setPerspective(params.m_zNear, params.m_zFar, dynamicFov);
		}

		inline void assignGoalToPreset(CameraPreset& preset, const CCameraGoal& goal) const
		{
			preset.position = goal.position;
			preset.orientation = goal.orientation;
			preset.targetPosition = goal.targetPosition;
			preset.hasTargetPosition = goal.hasTargetPosition;
			preset.distance = goal.distance;
			preset.hasDistance = goal.hasDistance;
			preset.orbitU = goal.orbitU;
			preset.orbitV = goal.orbitV;
			preset.orbitDistance = goal.orbitDistance;
			preset.hasOrbitState = goal.hasOrbitState;
			preset.dynamicBaseFov = goal.dynamicPerspectiveState.baseFov;
			preset.dynamicReferenceDistance = goal.dynamicPerspectiveState.referenceDistance;
			preset.hasDynamicPerspectiveState = goal.hasDynamicPerspectiveState;
		}

		inline CCameraGoal makeGoalFromPreset(const CameraPreset& preset) const
		{
			CCameraGoal target;
			target.position = preset.position;
			target.orientation = preset.orientation;
			target.hasTargetPosition = preset.hasTargetPosition;
			target.targetPosition = preset.targetPosition;
			target.hasDistance = preset.hasDistance;
			target.distance = preset.distance;
			target.hasOrbitState = preset.hasOrbitState;
			target.orbitU = preset.orbitU;
			target.orbitV = preset.orbitV;
			target.orbitDistance = preset.orbitDistance;
			target.hasDynamicPerspectiveState = preset.hasDynamicPerspectiveState;
			target.dynamicPerspectiveState = {
				.baseFov = preset.dynamicBaseFov,
				.referenceDistance = preset.dynamicReferenceDistance
			};
			return target;
		}

		inline CCameraGoal blendGoals(const CCameraGoal& a, const CCameraGoal& b, double alpha) const
		{
			CCameraGoal blended;
			blended.position = a.position + (b.position - a.position) * alpha;
			blended.orientation = glm::slerp(a.orientation, b.orientation, static_cast<float>(alpha));
			blended.hasTargetPosition = a.hasTargetPosition || b.hasTargetPosition;
			if (blended.hasTargetPosition)
			{
				const auto ta = a.hasTargetPosition ? a.targetPosition : b.targetPosition;
				const auto tb = b.hasTargetPosition ? b.targetPosition : a.targetPosition;
				blended.targetPosition = ta + (tb - ta) * alpha;
			}
			blended.hasDistance = a.hasDistance || b.hasDistance;
			if (blended.hasDistance)
			{
				const float da = a.hasDistance ? a.distance : b.distance;
				const float db = b.hasDistance ? b.distance : a.distance;
				blended.distance = da + (db - da) * static_cast<float>(alpha);
			}
			blended.hasOrbitState = a.hasOrbitState || b.hasOrbitState;
			if (blended.hasOrbitState)
			{
				const double ua = a.hasOrbitState ? a.orbitU : b.orbitU;
				const double ub = b.hasOrbitState ? b.orbitU : a.orbitU;
				const double va = a.hasOrbitState ? a.orbitV : b.orbitV;
				const double vb = b.hasOrbitState ? b.orbitV : a.orbitV;
				const float da = a.hasOrbitState ? a.orbitDistance : b.orbitDistance;
				const float db = b.hasOrbitState ? b.orbitDistance : a.orbitDistance;

				blended.orbitU = ua + (ub - ua) * alpha;
				blended.orbitV = va + (vb - va) * alpha;
				blended.orbitDistance = da + (db - da) * static_cast<float>(alpha);
			}
			blended.hasDynamicPerspectiveState = a.hasDynamicPerspectiveState || b.hasDynamicPerspectiveState;
			if (blended.hasDynamicPerspectiveState)
			{
				const auto dynamicA = a.hasDynamicPerspectiveState ? a.dynamicPerspectiveState : b.dynamicPerspectiveState;
				const auto dynamicB = b.hasDynamicPerspectiveState ? b.dynamicPerspectiveState : a.dynamicPerspectiveState;
				blended.dynamicPerspectiveState.baseFov = dynamicA.baseFov + (dynamicB.baseFov - dynamicA.baseFov) * static_cast<float>(alpha);
				blended.dynamicPerspectiveState.referenceDistance =
					dynamicA.referenceDistance + (dynamicB.referenceDistance - dynamicA.referenceDistance) * static_cast<float>(alpha);
			}
			return blended;
		}

		inline CameraPreset capturePreset(ICamera* camera, const std::string& name)
		{
			CameraPreset preset;
			preset.name = name;
			if (!camera)
				return preset;

			preset.identifier = std::string(camera->getIdentifier());
			CCameraGoal goal;
			if (m_cameraGoalSolver.capture(camera, goal))
				assignGoalToPreset(preset, goal);

			return preset;
		}

		inline CCameraGoalSolver::SApplyResult applyPresetToCameraDetailed(ICamera* camera, const CameraPreset& preset)
		{
			if (!camera)
				return {};

			return m_cameraGoalSolver.applyDetailed(camera, makeGoalFromPreset(preset));
		}

		inline bool applyPresetToCamera(ICamera* camera, const CameraPreset& preset)
		{
			return applyPresetToCameraDetailed(camera, preset).succeeded();
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

		inline void applyConstraintsToCamera(ICamera* camera)
		{
			if (!m_cameraConstraints.enabled || !camera)
				return;

			if (camera->hasCapability(ICamera::SphericalTarget))
			{
				if (m_cameraConstraints.clampDistance)
				{
					ICamera::SphericalTargetState sphericalState;
					if (camera->tryGetSphericalTargetState(sphericalState))
					{
						const float clamped = std::clamp<float>(sphericalState.distance, m_cameraConstraints.minDistance, m_cameraConstraints.maxDistance);
						camera->trySetSphericalDistance(clamped);
					}
				}
				return;
			}

			if (!(m_cameraConstraints.clampPitch || m_cameraConstraints.clampYaw || m_cameraConstraints.clampRoll))
				return;

			const auto& gimbal = camera->getGimbal();
			const auto pos = gimbal.getPosition();
			const auto eulerDeg = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));

			auto clamped = eulerDeg;
			if (m_cameraConstraints.clampPitch)
				clamped.x = std::clamp(clamped.x, m_cameraConstraints.pitchMinDeg, m_cameraConstraints.pitchMaxDeg);
			if (m_cameraConstraints.clampYaw)
				clamped.y = std::clamp(clamped.y, m_cameraConstraints.yawMinDeg, m_cameraConstraints.yawMaxDeg);
			if (m_cameraConstraints.clampRoll)
				clamped.z = std::clamp(clamped.z, m_cameraConstraints.rollMinDeg, m_cameraConstraints.rollMaxDeg);

			if (clamped.x == eulerDeg.x && clamped.y == eulerDeg.y && clamped.z == eulerDeg.z)
				return;

			CameraPreset preset;
			preset.position = pos;
			preset.orientation = glm::quat(hlsl::radians(clamped));
			applyPresetToCamera(camera, preset);
		}

		inline void applyVirtualEventScaling(std::vector<CVirtualGimbalEvent>& events, uint32_t count)
		{
			for (uint32_t i = 0u; i < count; ++i)
			{
				auto& ev = events[i];
				const auto type = ev.type;

				if (type == CVirtualGimbalEvent::MoveForward || type == CVirtualGimbalEvent::MoveBackward ||
					type == CVirtualGimbalEvent::MoveLeft || type == CVirtualGimbalEvent::MoveRight ||
					type == CVirtualGimbalEvent::MoveUp || type == CVirtualGimbalEvent::MoveDown)
				{
					ev.magnitude *= m_cameraControls.translationScale;
				}
				else if (type == CVirtualGimbalEvent::TiltUp || type == CVirtualGimbalEvent::TiltDown ||
					type == CVirtualGimbalEvent::PanLeft || type == CVirtualGimbalEvent::PanRight ||
					type == CVirtualGimbalEvent::RollLeft || type == CVirtualGimbalEvent::RollRight)
				{
					ev.magnitude *= m_cameraControls.rotationScale;
				}
			}
		}

		inline void remapTranslationToWorld(ICamera* camera, std::vector<CVirtualGimbalEvent>& events, uint32_t& count)
		{
			if (!camera)
				return;

			float64_t3 worldDelta = float64_t3(0.0);
			std::vector<CVirtualGimbalEvent> filtered;
			filtered.reserve(events.size());

			for (uint32_t i = 0u; i < count; ++i)
			{
				const auto& ev = events[i];
				switch (ev.type)
				{
					case CVirtualGimbalEvent::MoveRight: worldDelta.x += ev.magnitude; break;
					case CVirtualGimbalEvent::MoveLeft: worldDelta.x -= ev.magnitude; break;
					case CVirtualGimbalEvent::MoveUp: worldDelta.y += ev.magnitude; break;
					case CVirtualGimbalEvent::MoveDown: worldDelta.y -= ev.magnitude; break;
					case CVirtualGimbalEvent::MoveForward: worldDelta.z += ev.magnitude; break;
					case CVirtualGimbalEvent::MoveBackward: worldDelta.z -= ev.magnitude; break;
					default:
						filtered.emplace_back(ev);
						break;
				}
			}

			if (worldDelta.x == 0.0 && worldDelta.y == 0.0 && worldDelta.z == 0.0)
			{
				events = std::move(filtered);
				count = static_cast<uint32_t>(events.size());
				return;
			}

			const auto& gimbal = camera->getGimbal();
			const auto right = gimbal.getXAxis();
			const auto up = gimbal.getYAxis();
			const auto forward = gimbal.getZAxis();

			const float64_t3 localDelta = float64_t3(
				hlsl::dot(worldDelta, right),
				hlsl::dot(worldDelta, up),
				hlsl::dot(worldDelta, forward)
			);

			auto emitAxis = [&](double v, CVirtualGimbalEvent::VirtualEventType pos, CVirtualGimbalEvent::VirtualEventType neg)
			{
				if (v == 0.0)
					return;
				auto& ev = filtered.emplace_back();
				ev.type = (v > 0.0) ? pos : neg;
				ev.magnitude = std::abs(v);
			};

			emitAxis(localDelta.x, CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft);
			emitAxis(localDelta.y, CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown);
			emitAxis(localDelta.z, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);

			events = std::move(filtered);
			count = static_cast<uint32_t>(events.size());
		}

		inline void applyPresetToTargets(const CameraPreset& preset)
		{
			if (!m_playbackAffectsAll)
			{
				applyPresetToCamera(getActiveCamera(), preset);
				return;
			}

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
					applyPresetToCamera(camera, preset);
			}
		}

		inline void updatePlayback(double dtSec)
		{
			if (!m_playback.playing || m_keyframes.empty())
				return;

			m_playback.time += static_cast<float>(dtSec * m_playback.speed);

			const float duration = m_keyframes.back().time;
			if (duration <= 0.f)
			{
				applyPresetToTargets(m_keyframes.back().preset);
				return;
			}

			if (m_playback.loop)
			{
				while (m_playback.time > duration)
					m_playback.time -= duration;
			}
			else if (m_playback.time > duration)
			{
				m_playback.time = duration;
				m_playback.playing = false;
			}

			const auto time = m_playback.time;
			if (m_keyframes.size() == 1)
			{
				applyPresetToTargets(m_keyframes.front().preset);
				return;
			}

			size_t idx = 0u;
			while (idx + 1u < m_keyframes.size() && m_keyframes[idx + 1u].time < time)
				++idx;

			const auto& a = m_keyframes[idx];
			const auto& b = m_keyframes[std::min(idx + 1u, m_keyframes.size() - 1u)];

			if (b.time <= a.time)
			{
				applyPresetToTargets(a.preset);
				return;
			}

			const double alpha = static_cast<double>(time - a.time) / static_cast<double>(b.time - a.time);

			CameraPreset blended = a.preset;
			assignGoalToPreset(blended, blendGoals(makeGoalFromPreset(a.preset), makeGoalFromPreset(b.preset), alpha));

			applyPresetToTargets(blended);
		}

		inline bool savePresetsToFile(const system::path& path)
		{
			nbl_json root;
			root["presets"] = nbl_json::array();

			for (const auto& preset : m_presets)
			{
				nbl_json j;
				j["name"] = preset.name;
				j["identifier"] = preset.identifier;
				j["position"] = { preset.position.x, preset.position.y, preset.position.z };
				j["orientation"] = { preset.orientation.x, preset.orientation.y, preset.orientation.z, preset.orientation.w };
				if (preset.hasTargetPosition)
					j["target_position"] = { preset.targetPosition.x, preset.targetPosition.y, preset.targetPosition.z };
				if (preset.hasDistance)
					j["distance"] = preset.distance;
				if (preset.hasOrbitState)
				{
					j["orbit_u"] = preset.orbitU;
					j["orbit_v"] = preset.orbitV;
					j["orbit_distance"] = preset.orbitDistance;
				}
				if (preset.hasDynamicPerspectiveState)
				{
					j["dynamic_base_fov"] = preset.dynamicBaseFov;
					j["dynamic_reference_distance"] = preset.dynamicReferenceDistance;
				}
				root["presets"].push_back(std::move(j));
			}

			std::ofstream out(path.string(), std::ios::binary);
			if (!out)
				return false;
			out << root.dump(2);
			return true;
		}

		inline bool loadPresetsFromFile(const system::path& path)
		{
			std::ifstream in(path.string(), std::ios::binary);
			if (!in)
				return false;

			nbl_json root;
			in >> root;
			if (!root.contains("presets"))
				return false;

			m_presets.clear();
			for (const auto& entry : root["presets"])
			{
				CameraPreset preset;
				if (entry.contains("name"))
					preset.name = entry["name"].get<std::string>();
				if (entry.contains("identifier"))
					preset.identifier = entry["identifier"].get<std::string>();
				if (entry.contains("position") && entry["position"].is_array())
				{
					auto arr = entry["position"];
					preset.position = float64_t3(arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>());
				}
				if (entry.contains("orientation") && entry["orientation"].is_array())
				{
					auto arr = entry["orientation"];
					preset.orientation = glm::quat(
						arr[3].get<float>(),
						arr[0].get<float>(),
						arr[1].get<float>(),
						arr[2].get<float>()
					);
				}
				if (entry.contains("target_position") && entry["target_position"].is_array())
				{
					auto arr = entry["target_position"];
					preset.targetPosition = float64_t3(arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>());
					preset.hasTargetPosition = true;
				}
				if (entry.contains("distance"))
				{
					preset.distance = entry["distance"].get<float>();
					preset.hasDistance = true;
				}
				if (entry.contains("orbit_u"))
				{
					preset.orbitU = entry["orbit_u"].get<double>();
					preset.hasOrbitState = true;
				}
				if (entry.contains("orbit_v"))
				{
					preset.orbitV = entry["orbit_v"].get<double>();
					preset.hasOrbitState = true;
				}
				if (entry.contains("orbit_distance"))
				{
					preset.orbitDistance = entry["orbit_distance"].get<float>();
					preset.hasOrbitState = true;
				}
				if (entry.contains("dynamic_base_fov"))
				{
					preset.dynamicBaseFov = entry["dynamic_base_fov"].get<float>();
					preset.hasDynamicPerspectiveState = true;
				}
				if (entry.contains("dynamic_reference_distance"))
				{
					preset.dynamicReferenceDistance = entry["dynamic_reference_distance"].get<float>();
					preset.hasDynamicPerspectiveState = true;
				}
				m_presets.emplace_back(std::move(preset));
			}

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

		// if we had working IObjectTransform or something similar then it would be it instead, it is "last manipulated object" I need for TRS editor
		// in reality we should store range of those IObjectTransforem interface range & index to object representing last manipulated one
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

			binding.inputBinding.copyBindingLayoutFrom(projections[projectionIx]);
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

			binding.inputBinding.copyBindingLayoutTo(projections[projectionIx]);
			binding.inputBindingPlanarIx = binding.activePlanarIx;
			binding.inputBindingProjectionIx = projectionIx;
		}

		struct ScriptedInputEvent
		{
			enum class Type : uint8_t
			{
				Keyboard,
				Mouse,
				Imguizmo,
				Action
			};

			struct KeyboardData
			{
				ui::E_KEY_CODE key = ui::EKC_NONE;
				ui::SKeyboardEvent::E_KEY_ACTION action = ui::SKeyboardEvent::ECA_UNITIALIZED;
			};

			struct MouseData
			{
				ui::SMouseEvent::E_EVENT_TYPE type = ui::SMouseEvent::EET_UNITIALIZED;
				ui::E_MOUSE_BUTTON button = ui::EMB_LEFT_BUTTON;
				ui::SMouseEvent::SClickEvent::E_ACTION action = ui::SMouseEvent::SClickEvent::EA_UNITIALIZED;
				int16_t x = 0;
				int16_t y = 0;
				int16_t dx = 0;
				int16_t dy = 0;
				int16_t v = 0;
				int16_t h = 0;
			};

			struct ActionData
			{
				enum class Kind : uint8_t
				{
					SetActiveRenderWindow,
					SetActivePlanar,
					SetProjectionType,
					SetProjectionIndex,
					SetUseWindow,
					SetLeftHanded,
					ResetActiveCamera
				};

				Kind kind = Kind::SetActiveRenderWindow;
				int32_t value = 0;
			};

			uint64_t frame = 0;
			Type type = Type::Keyboard;
			KeyboardData keyboard;
			MouseData mouse;
			float32_t4x4 imguizmo = float32_t4x4(1.f);
			ActionData action;
		};

		struct ScriptedInputCheck
		{
			enum class Kind : uint8_t
			{
				Baseline,
				ImguizmoVirtual,
				GimbalNear,
				GimbalDelta,
				GimbalStep
			};

			struct ExpectedVirtualEvent
			{
				CVirtualGimbalEvent::VirtualEventType type = CVirtualGimbalEvent::None;
				float64_t magnitude = 0.0;
			};

			uint64_t frame = 0;
			Kind kind = Kind::Baseline;
			float tolerance = 1e-3f;
			std::vector<ExpectedVirtualEvent> expectedVirtualEvents;

			float32_t3 expectedPos = float32_t3(0.f);
			float32_t3 expectedEulerDeg = float32_t3(0.f);
			bool hasExpectedPos = false;
			bool hasExpectedEuler = false;
			float posTolerance = 0.05f;
			float eulerToleranceDeg = 1.0f;
			float minPosDelta = 0.0f;
			float minEulerDeltaDeg = 0.0f;
			bool hasPosDeltaConstraint = false;
			bool hasEulerDeltaConstraint = false;
		};

		struct ScriptedInputState
		{
			bool enabled = false;
			bool log = false;
			bool exclusive = false;
			bool hardFail = false;
			bool visualDebug = false;
			float visualTargetFps = 0.f;
			float visualCameraHoldSeconds = 0.f;
			std::vector<ScriptedInputEvent> events;
			size_t nextEventIndex = 0;
			std::vector<ScriptedInputCheck> checks;
			size_t nextCheckIndex = 0;
			std::vector<uint64_t> captureFrames;
			size_t nextCaptureIndex = 0;
			std::string capturePrefix = "script";
			system::path captureOutputDir;
			bool failed = false;
			bool summaryReported = false;
			bool baselineValid = false;
			float32_t3 baselinePos = float32_t3(0.f);
			float32_t3 baselineEulerDeg = float32_t3(0.f);
			bool stepValid = false;
			float32_t3 stepPos = float32_t3(0.f);
			float32_t3 stepEulerDeg = float32_t3(0.f);
			bool visualActivePlanarValid = false;
			uint32_t visualActivePlanarIx = 0u;
			uint64_t visualActivePlanarStartFrame = 0u;
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
		std::vector<CameraKeyframe> m_keyframes;
		CameraPlaybackState m_playback;
		CCameraGoalSolver m_cameraGoalSolver;
		bool m_playbackAffectsAll = false;
		float m_newKeyframeTime = 0.f;
		char m_presetName[64] = "Preset";
		char m_presetPath[260] = "camera_presets.json";
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

