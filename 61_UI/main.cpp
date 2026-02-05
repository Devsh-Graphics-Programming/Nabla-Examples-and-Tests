// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <string>
#include <unordered_set>
#include <utility>
#include "nlohmann/json.hpp"
#include "argparse/argparse.hpp"
using json = nlohmann::json;

#include "common.hpp"
#include "keysmapping.hpp"
#include "camera/CCubeProjection.hpp"
#include "glm/glm/ext/matrix_clip_space.hpp" // TODO: TESTING
#include "glm/gtc/quaternion.hpp"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"
#if __has_include("nbl/this_example/builtin/CArchive.h")
#include "nbl/this_example/builtin/CArchive.h"
#endif
#if __has_include("nbl/this_example/builtin/build/CArchive.h")
#include "nbl/this_example/builtin/build/CArchive.h"
#endif

using planar_projections_range_t = std::vector<IPlanarProjection::CProjection>;
using planar_projection_t = CPlanarProjection<planar_projections_range_t>;

// the only reason for those is to remind we must go with transpose & 4x4 matrices
struct ImGuizmoPlanarM16InOut
{
	float32_t4x4 view, projection;
};

struct ImGuizmoModelM16InOut
{
	float32_t4x4 inTRS, outTRS, outDeltaTRS;
};

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

class UISampleApp final : public examples::SimpleWindowedApplication
{
	using base_t = examples::SimpleWindowedApplication;
	using clock_t = std::chrono::steady_clock;

	constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);
	constexpr static inline auto sceneRenderDepthFormat = EF_D32_SFLOAT;
	constexpr static inline auto finalSceneRenderFormat = EF_R8G8B8A8_SRGB;
	constexpr static inline IGPUCommandBuffer::SClearColorValue SceneClearColor = { .float32 = {0.f,0.f,0.f,1.f} };
	constexpr static inline IGPUCommandBuffer::SClearDepthStencilValue SceneClearDepth = { .depth = 0.f };

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
				auto* cc = m_window->getCursorControl();
				cc->setVisible(false);

				return { {m_surface->getSurface()/*,EQF_NONE*/} };
			}
			
			return {};
		}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			argparse::ArgumentParser program("Virtual camera event system demo");

			program.add_argument<std::string>("--file")
				.help("Path to json file with camera inputs");
			program.add_argument("--ci")
				.help("Run in CI mode: capture a screenshot after a few frames and exit.")
				.default_value(false)
				.implicit_value(true);
			program.add_argument<std::string>("--script")
				.help("Path to json file with scripted input events");
			program.add_argument("--script-log")
				.help("Log scripted input and virtual events.")
				.default_value(false)
				.implicit_value(true);

			try
			{
				program.parse_args({ argv.data(), argv.data() + argv.size() });
			}
			catch (const std::exception& err)
			{
				std::cerr << err.what() << std::endl << program;
				return false;
			}

			m_ciMode = program.get<bool>("--ci");
			if (m_ciMode)
				m_ciScreenshotPath = localOutputCWD / "cameraz_ci.png";
			m_scriptedInput.log = program.get<bool>("--script-log");

			// Create imput system
			m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			{
				smart_refctd_ptr<system::IFileArchive> examplesHeaderArch, examplesSourceArch, examplesBuildArch, thisExampleArch, thisExampleBuildArch;
#ifdef NBL_EMBED_BUILTIN_RESOURCES
				examplesHeaderArch = core::make_smart_refctd_ptr<nbl::builtin::examples::include::CArchive>(smart_refctd_ptr(m_logger));
				examplesSourceArch = core::make_smart_refctd_ptr<nbl::builtin::examples::src::CArchive>(smart_refctd_ptr(m_logger));
				examplesBuildArch = core::make_smart_refctd_ptr<nbl::builtin::examples::build::CArchive>(smart_refctd_ptr(m_logger));

	#ifdef _NBL_THIS_EXAMPLE_BUILTIN_C_ARCHIVE_H_
				thisExampleArch = make_smart_refctd_ptr<nbl::this_example::builtin::CArchive>(smart_refctd_ptr(m_logger));
	#endif

	#ifdef _NBL_THIS_EXAMPLE_BUILTIN_BUILD_C_ARCHIVE_H_
				thisExampleBuildArch = make_smart_refctd_ptr<nbl::this_example::builtin::build::CArchive>(smart_refctd_ptr(m_logger));
	#endif
#else
				examplesHeaderArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(localInputCWD/"../common/include/nbl/examples", smart_refctd_ptr(m_logger), m_system.get());
				examplesSourceArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(localInputCWD/"../common/src/nbl/examples", smart_refctd_ptr(m_logger), m_system.get());
				examplesBuildArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(NBL_EXAMPLES_BUILD_MOUNT_POINT, smart_refctd_ptr(m_logger), m_system.get());
				thisExampleArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(localInputCWD/"app_resources", smart_refctd_ptr(m_logger), m_system.get());
	#ifdef NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT
				thisExampleBuildArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT, smart_refctd_ptr(m_logger), m_system.get());
	#endif
#endif
				m_system->mount(std::move(examplesHeaderArch),"nbl/examples");
				m_system->mount(std::move(examplesSourceArch),"nbl/examples");
				m_system->mount(std::move(examplesBuildArch),"nbl/examples");
				if (thisExampleArch)
					m_system->mount(std::move(thisExampleArch),"app_resources");
				if (thisExampleBuildArch)
					m_system->mount(std::move(thisExampleBuildArch),"app_resources");
			}

			{
				const std::optional<std::string> cameraJsonFile = program.is_used("--file") ? program.get<std::string>("--file") : std::optional<std::string>(std::nullopt);

				json j;
				auto loadDefaultConfig = [&]() -> bool
				{
#ifdef _NBL_THIS_EXAMPLE_BUILTIN_C_ARCHIVE_H_
					auto assets = make_smart_refctd_ptr<this_example::builtin::CArchive>(smart_refctd_ptr(m_logger));
					auto pFile = assets->getFile("cameras.json", IFile::ECF_READ, "");
					if (!pFile)
						return logFail("Could not open builtin cameras.json!");

					string config;
					IFile::success_t result;
					config.resize(pFile->getSize());
					pFile->read(result, config.data(), 0, pFile->getSize());
					j = json::parse(config);
					return true;
#else
					const auto fallbackPath = localInputCWD / "app_resources" / "cameras.json";
					std::ifstream fallbackFile(fallbackPath);
					if (!fallbackFile.is_open())
						return logFail("Cannot open default config \"%s\".", fallbackPath.string().c_str());
					fallbackFile >> j;
					return true;
#endif
				};

				auto file = cameraJsonFile.has_value() ? std::ifstream(cameraJsonFile.value()) : std::ifstream();
				if (!file.is_open())
				{
					if (cameraJsonFile.has_value())
						m_logger->log("Cannot open input \"%s\" json file. Switching to default config.", ILogger::ELL_WARNING, cameraJsonFile.value().c_str());
					else
						m_logger->log("No input json file provided. Switching to default config.", ILogger::ELL_WARNING);

					if (!loadDefaultConfig())
						return false;
				}
				else
				{
					file >> j;
				}

				auto loadScriptJson = [&](const std::string& path, json& out) -> bool
				{
					std::ifstream sfile(path);
					if (!sfile.is_open())
					{
						m_logger->log("Cannot open scripted input file \"%s\".", ILogger::ELL_ERROR, path.c_str());
						return false;
					}
					sfile >> out;
					return true;
				};

				auto parseScriptedInput = [&](const json& script) -> void
				{
					m_scriptedInput.events.clear();
					m_scriptedInput.checks.clear();
					m_scriptedInput.captureFrames.clear();
					m_scriptedInput.nextEventIndex = 0;
					m_scriptedInput.nextCheckIndex = 0;
					m_scriptedInput.nextCaptureIndex = 0;
					m_scriptedInput.failed = false;
					m_scriptedInput.summaryReported = false;
					m_scriptedInput.baselineValid = false;
					m_scriptedInput.exclusive = false;
					m_scriptedInput.hardFail = false;
					m_scriptedInput.capturePrefix = "script";
					m_scriptedInput.captureOutputDir = localOutputCWD;

					if (script.contains("enabled"))
						m_scriptedInput.enabled = script["enabled"].get<bool>();
					else
						m_scriptedInput.enabled = true;

					if (script.contains("log"))
						m_scriptedInput.log = script["log"].get<bool>() || m_scriptedInput.log;

					if (script.contains("hard_fail"))
						m_scriptedInput.hardFail = script["hard_fail"].get<bool>();

					if (script.contains("enableActiveCameraMovement"))
						enableActiveCameraMovement = script["enableActiveCameraMovement"].get<bool>();
					else if (m_scriptedInput.enabled)
						enableActiveCameraMovement = true;

					if (script.contains("exclusive_input"))
						m_scriptedInput.exclusive = script["exclusive_input"].get<bool>() || m_scriptedInput.exclusive;
					if (script.contains("exclusive"))
						m_scriptedInput.exclusive = script["exclusive"].get<bool>() || m_scriptedInput.exclusive;

					if (script.contains("capture_prefix"))
						m_scriptedInput.capturePrefix = script["capture_prefix"].get<std::string>();
					if (m_scriptedInput.capturePrefix.empty())
						m_scriptedInput.capturePrefix = "script";
					if (script.contains("capture_frames"))
						for (const auto& frame : script["capture_frames"])
							m_scriptedInput.captureFrames.emplace_back(frame.get<uint64_t>());

					if (script.contains("camera_controls"))
					{
						const auto& controls = script["camera_controls"];
						if (controls.contains("keyboard_scale"))
							m_cameraControls.keyboardScale = controls["keyboard_scale"].get<float>();
						if (controls.contains("mouse_move_scale"))
							m_cameraControls.mouseMoveScale = controls["mouse_move_scale"].get<float>();
						if (controls.contains("mouse_scroll_scale"))
							m_cameraControls.mouseScrollScale = controls["mouse_scroll_scale"].get<float>();
						if (controls.contains("translation_scale"))
							m_cameraControls.translationScale = controls["translation_scale"].get<float>();
						if (controls.contains("rotation_scale"))
							m_cameraControls.rotationScale = controls["rotation_scale"].get<float>();
					}

					if (script.contains("events"))
						for (const auto& ev : script["events"])
						{
						if (!ev.contains("frame") || !ev.contains("type"))
						{
							m_logger->log("Scripted input event missing \"frame\" or \"type\".", ILogger::ELL_WARNING);
							continue;
						}

						const auto frame = ev["frame"].get<uint64_t>();
						const auto type = ev["type"].get<std::string>();
						const bool captureFrame = ev.value("capture", false);

						if (type == "keyboard")
						{
							if (!ev.contains("key") || !ev.contains("action"))
							{
								m_logger->log("Scripted keyboard event missing \"key\" or \"action\".", ILogger::ELL_WARNING);
								continue;
							}

							const auto keyStr = ev["key"].get<std::string>();
							const auto actionStr = ev["action"].get<std::string>();
							const auto key = ui::stringToKeyCode(keyStr);
							if (key == ui::EKC_NONE)
							{
								m_logger->log("Scripted keyboard event has invalid key \"%s\".", ILogger::ELL_WARNING, keyStr.c_str());
								continue;
							}

							ui::SKeyboardEvent::E_KEY_ACTION action = ui::SKeyboardEvent::ECA_UNITIALIZED;
							if (actionStr == "pressed" || actionStr == "press")
								action = ui::SKeyboardEvent::ECA_PRESSED;
							else if (actionStr == "released" || actionStr == "release")
								action = ui::SKeyboardEvent::ECA_RELEASED;

							if (action == ui::SKeyboardEvent::ECA_UNITIALIZED)
							{
								m_logger->log("Scripted keyboard event has invalid action \"%s\".", ILogger::ELL_WARNING, actionStr.c_str());
								continue;
							}

							ScriptedInputEvent entry;
							entry.frame = frame;
							entry.type = ScriptedInputEvent::Type::Keyboard;
							entry.keyboard.key = key;
							entry.keyboard.action = action;
							m_scriptedInput.events.emplace_back(entry);
							if (captureFrame)
								m_scriptedInput.captureFrames.emplace_back(frame);
						}
						else if (type == "mouse")
						{
							if (!ev.contains("kind"))
							{
								m_logger->log("Scripted mouse event missing \"kind\".", ILogger::ELL_WARNING);
								continue;
							}

							const auto kind = ev["kind"].get<std::string>();
							ScriptedInputEvent entry;
							entry.frame = frame;
							entry.type = ScriptedInputEvent::Type::Mouse;

							if (kind == "move")
							{
								entry.mouse.type = ui::SMouseEvent::EET_MOVEMENT;
								entry.mouse.dx = ev.value("dx", 0);
								entry.mouse.dy = ev.value("dy", 0);
							}
							else if (kind == "scroll")
							{
								entry.mouse.type = ui::SMouseEvent::EET_SCROLL;
								entry.mouse.v = ev.value("v", 0);
								entry.mouse.h = ev.value("h", 0);
							}
							else if (kind == "click")
							{
								if (!ev.contains("button") || !ev.contains("action"))
								{
									m_logger->log("Scripted click event missing \"button\" or \"action\".", ILogger::ELL_WARNING);
									continue;
								}

								const auto buttonStr = ev["button"].get<std::string>();
								const auto actionStr = ev["action"].get<std::string>();

								ui::E_MOUSE_BUTTON button = ui::EMB_LEFT_BUTTON;
								if (buttonStr == "LEFT_BUTTON") button = ui::EMB_LEFT_BUTTON;
								else if (buttonStr == "RIGHT_BUTTON") button = ui::EMB_RIGHT_BUTTON;
								else if (buttonStr == "MIDDLE_BUTTON") button = ui::EMB_MIDDLE_BUTTON;
								else if (buttonStr == "BUTTON_4") button = ui::EMB_BUTTON_4;
								else if (buttonStr == "BUTTON_5") button = ui::EMB_BUTTON_5;
								else
								{
									m_logger->log("Scripted click event has invalid button \"%s\".", ILogger::ELL_WARNING, buttonStr.c_str());
									continue;
								}

								ui::SMouseEvent::SClickEvent::E_ACTION action = ui::SMouseEvent::SClickEvent::EA_UNITIALIZED;
								if (actionStr == "pressed" || actionStr == "press")
									action = ui::SMouseEvent::SClickEvent::EA_PRESSED;
								else if (actionStr == "released" || actionStr == "release")
									action = ui::SMouseEvent::SClickEvent::EA_RELEASED;

								if (action == ui::SMouseEvent::SClickEvent::EA_UNITIALIZED)
								{
									m_logger->log("Scripted click event has invalid action \"%s\".", ILogger::ELL_WARNING, actionStr.c_str());
									continue;
								}

								entry.mouse.type = ui::SMouseEvent::EET_CLICK;
								entry.mouse.button = button;
								entry.mouse.action = action;
								entry.mouse.x = ev.value("x", 0);
								entry.mouse.y = ev.value("y", 0);
							}
							else
							{
								m_logger->log("Scripted mouse event has invalid kind \"%s\".", ILogger::ELL_WARNING, kind.c_str());
								continue;
							}

							m_scriptedInput.events.emplace_back(entry);
							if (captureFrame)
								m_scriptedInput.captureFrames.emplace_back(frame);
						}
						else if (type == "imguizmo")
						{
							ScriptedInputEvent entry;
							entry.frame = frame;
							entry.type = ScriptedInputEvent::Type::Imguizmo;

							if (ev.contains("delta_trs"))
							{
								const auto arr = ev["delta_trs"].get<std::array<float, 16>>();
								float m16[16];
								for (size_t i = 0u; i < 16u; ++i)
									m16[i] = arr[i];
								entry.imguizmo = *reinterpret_cast<float32_t4x4*>(m16);
							}
							else
							{
								const auto t = ev.contains("translation") ? ev["translation"].get<std::array<float, 3>>() : std::array<float, 3>{0.f, 0.f, 0.f};
								const auto r = ev.contains("rotation_deg") ? ev["rotation_deg"].get<std::array<float, 3>>() : std::array<float, 3>{0.f, 0.f, 0.f};
								const auto s = ev.contains("scale") ? ev["scale"].get<std::array<float, 3>>() : std::array<float, 3>{1.f, 1.f, 1.f};

								float m16[16];
								float tr[3] = { t[0], t[1], t[2] };
								float rot[3] = { r[0], r[1], r[2] };
								float sc[3] = { s[0], s[1], s[2] };

								ImGuizmo::RecomposeMatrixFromComponents(tr, rot, sc, m16);
								entry.imguizmo = *reinterpret_cast<float32_t4x4*>(m16);
							}

							m_scriptedInput.events.emplace_back(entry);
							if (captureFrame)
								m_scriptedInput.captureFrames.emplace_back(frame);
						}
						else if (type == "action")
						{
							if (!ev.contains("action"))
							{
								m_logger->log("Scripted action event missing \"action\".", ILogger::ELL_WARNING);
								continue;
							}

							const auto actionStr = ev["action"].get<std::string>();
							ScriptedInputEvent entry;
							entry.frame = frame;
							entry.type = ScriptedInputEvent::Type::Action;

							auto getValueInt = [&]() -> int32_t
							{
								if (ev.contains("value"))
									return ev["value"].get<int32_t>();
								if (ev.contains("index"))
									return ev["index"].get<int32_t>();
								return 0;
							};

							if (actionStr == "set_active_render_window")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow;
								entry.action.value = getValueInt();
							}
							else if (actionStr == "set_active_planar")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetActivePlanar;
								entry.action.value = getValueInt();
							}
							else if (actionStr == "set_projection_type")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetProjectionType;
								if (ev.contains("value") && ev["value"].is_string())
								{
									const auto valueStr = ev["value"].get<std::string>();
									if (valueStr == "perspective")
										entry.action.value = static_cast<int32_t>(IPlanarProjection::CProjection::Perspective);
									else if (valueStr == "orthographic")
										entry.action.value = static_cast<int32_t>(IPlanarProjection::CProjection::Orthographic);
									else
									{
										m_logger->log("Scripted action projection type has invalid value \"%s\".", ILogger::ELL_WARNING, valueStr.c_str());
										continue;
									}
								}
								else
								{
									entry.action.value = getValueInt();
								}
							}
							else if (actionStr == "set_projection_index")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetProjectionIndex;
								entry.action.value = getValueInt();
							}
							else if (actionStr == "set_use_window")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetUseWindow;
								entry.action.value = ev.value("value", false) ? 1 : 0;
							}
							else if (actionStr == "set_left_handed")
							{
								entry.action.kind = ScriptedInputEvent::ActionData::Kind::SetLeftHanded;
								entry.action.value = ev.value("value", false) ? 1 : 0;
							}
							else
							{
								m_logger->log("Scripted action event has invalid action \"%s\".", ILogger::ELL_WARNING, actionStr.c_str());
								continue;
							}

							m_scriptedInput.events.emplace_back(entry);
							if (captureFrame)
								m_scriptedInput.captureFrames.emplace_back(frame);
						}
						else
						{
							m_logger->log("Scripted input event has invalid type \"%s\".", ILogger::ELL_WARNING, type.c_str());
						}
						}

					if (script.contains("checks"))
					{
						for (const auto& chk : script["checks"])
						{
							if (!chk.contains("frame") || !chk.contains("kind"))
							{
								m_logger->log("Scripted check missing \"frame\" or \"kind\".", ILogger::ELL_WARNING);
								continue;
							}

							const auto frame = chk["frame"].get<uint64_t>();
							const auto kind = chk["kind"].get<std::string>();

							ScriptedInputCheck entry;
							entry.frame = frame;

							if (kind == "baseline")
							{
								entry.kind = ScriptedInputCheck::Kind::Baseline;
							}
							else if (kind == "imguizmo_virtual")
							{
								entry.kind = ScriptedInputCheck::Kind::ImguizmoVirtual;
								entry.tolerance = chk.value("tolerance", entry.tolerance);

								if (!chk.contains("events"))
								{
									m_logger->log("Imguizmo virtual check missing \"events\".", ILogger::ELL_WARNING);
									continue;
								}

								for (const auto& ev : chk["events"])
								{
									if (!ev.contains("type") || !ev.contains("magnitude"))
									{
										m_logger->log("Imguizmo virtual check event missing \"type\" or \"magnitude\".", ILogger::ELL_WARNING);
										continue;
									}

									const auto typeStr = ev["type"].get<std::string>();
									const auto type = CVirtualGimbalEvent::stringToVirtualEvent(typeStr);
									if (type == CVirtualGimbalEvent::None)
									{
										m_logger->log("Imguizmo virtual check event has invalid type \"%s\".", ILogger::ELL_WARNING, typeStr.c_str());
										continue;
									}

									ScriptedInputCheck::ExpectedVirtualEvent expected;
									expected.type = type;
									expected.magnitude = ev["magnitude"].get<double>();
									entry.expectedVirtualEvents.emplace_back(expected);
								}
							}
							else if (kind == "gimbal_near")
							{
								entry.kind = ScriptedInputCheck::Kind::GimbalNear;
								entry.posTolerance = chk.value("pos_tolerance", entry.posTolerance);
								entry.eulerToleranceDeg = chk.value("euler_tolerance_deg", entry.eulerToleranceDeg);

								if (chk.contains("position"))
								{
									const auto pos = chk["position"].get<std::array<float, 3>>();
									entry.expectedPos = float32_t3(pos[0], pos[1], pos[2]);
									entry.hasExpectedPos = true;
								}
								if (chk.contains("euler_deg"))
								{
									const auto euler = chk["euler_deg"].get<std::array<float, 3>>();
									entry.expectedEulerDeg = float32_t3(euler[0], euler[1], euler[2]);
									entry.hasExpectedEuler = true;
								}
							}
							else if (kind == "gimbal_delta")
							{
								entry.kind = ScriptedInputCheck::Kind::GimbalDelta;
								entry.posTolerance = chk.value("pos_tolerance", entry.posTolerance);
								entry.eulerToleranceDeg = chk.value("euler_tolerance_deg", entry.eulerToleranceDeg);
							}
							else
							{
								m_logger->log("Scripted check has invalid kind \"%s\".", ILogger::ELL_WARNING, kind.c_str());
								continue;
							}

							m_scriptedInput.checks.emplace_back(entry);
						}
					}

					std::sort(m_scriptedInput.events.begin(), m_scriptedInput.events.end(),
						[](const ScriptedInputEvent& a, const ScriptedInputEvent& b) { return a.frame < b.frame; });
					std::sort(m_scriptedInput.checks.begin(), m_scriptedInput.checks.end(),
						[](const ScriptedInputCheck& a, const ScriptedInputCheck& b) { return a.frame < b.frame; });
					if (!m_scriptedInput.captureFrames.empty())
					{
						std::sort(m_scriptedInput.captureFrames.begin(), m_scriptedInput.captureFrames.end());
						m_scriptedInput.captureFrames.erase(std::unique(m_scriptedInput.captureFrames.begin(), m_scriptedInput.captureFrames.end()), m_scriptedInput.captureFrames.end());
					}
				};

				if (program.is_used("--script"))
				{
					system::path scriptPath = program.get<std::string>("--script");
					if (scriptPath.is_relative())
						scriptPath = localInputCWD / scriptPath;
					json scriptJson;
					if (!loadScriptJson(scriptPath.string(), scriptJson))
						return false;
					parseScriptedInput(scriptJson);
				}
				else if (j.contains("scripted_input"))
				{
					parseScriptedInput(j["scripted_input"]);
				}

				std::vector<smart_refctd_ptr<ICamera>> cameras;
				for (const auto& jCamera : j["cameras"])
				{
					if (jCamera.contains("type"))
					{
						if (!jCamera.contains("position"))
						{
							logFail("Expected \"position\" keyword for camera definition!");
							return false;
						}

						const bool withOrientation = jCamera.contains("orientation");

						auto position = [&]()
						{
							auto jret = jCamera["position"].get<std::array<float, 3>>();
							return float32_t3(jret[0], jret[1], jret[2]);
						}();

						auto getOrientation = [&]()
						{
							auto jret = jCamera["orientation"].get<std::array<float, 4>>();

							// order important for glm::quat,
							// the ctor is GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q>::qua(T _w, T _x, T _y, T _z)
							// but memory layout (and json) is x,y,z,w
							return glm::quat(jret[3], jret[0], jret[1], jret[2]);
						};

						auto getTarget = [&]()
						{
							auto jret = jCamera["target"].get<std::array<float, 3>>();
							return float32_t3(jret[0], jret[1], jret[2]);
						};

						constexpr float DefaultMoveScale = 0.01f;
						constexpr float DefaultRotateScale = 0.003f;
						constexpr float OrbitMoveScale = 0.5f;

						if (jCamera["type"] == "FPS")
						{
							if (!withOrientation)
							{
								logFail("Expected \"orientation\" keyword for FPS camera definition!");
								return false;
							}

							auto camera = make_smart_refctd_ptr<CFPSCamera>(position, getOrientation());
							camera->setMoveSpeedScale(DefaultMoveScale);
							camera->setRotationSpeedScale(DefaultRotateScale);
							cameras.emplace_back(std::move(camera));
						}
						else if (jCamera["type"] == "Free")
						{
							if (!withOrientation)
							{
								logFail("Expected \"orientation\" keyword for Free camera definition!");
								return false;
							}

							auto camera = make_smart_refctd_ptr<CFreeCamera>(position, getOrientation());
							camera->setMoveSpeedScale(DefaultMoveScale);
							camera->setRotationSpeedScale(DefaultRotateScale);
							cameras.emplace_back(std::move(camera));
						}
						else if (jCamera["type"] == "Orbit")
						{
							auto camera = make_smart_refctd_ptr<COrbitCamera>(position, getTarget());
							camera->setMoveSpeedScale(OrbitMoveScale);
							camera->setRotationSpeedScale(DefaultRotateScale);
							cameras.emplace_back(std::move(camera));
						}
						else if (jCamera["type"] == "Arcball")
						{
							auto camera = make_smart_refctd_ptr<CArcballCamera>(position, getTarget());
							camera->setMoveSpeedScale(OrbitMoveScale);
							camera->setRotationSpeedScale(DefaultRotateScale);
							cameras.emplace_back(std::move(camera));
						}
						else if (jCamera["type"] == "Turntable")
						{
							auto camera = make_smart_refctd_ptr<CTurntableCamera>(position, getTarget());
							camera->setMoveSpeedScale(OrbitMoveScale);
							camera->setRotationSpeedScale(DefaultRotateScale);
							cameras.emplace_back(std::move(camera));
						}
						else
						{
							logFail("Unsupported camera type!");
							return false;
						}
					}
					else
					{
						logFail("Expected \"type\" keyword for camera definition!");
						return false;
					}
				}

				std::vector<IPlanarProjection::CProjection> projections;
				for (const auto& jProjection : j["projections"])
				{
					if (jProjection.contains("type"))
					{
						float zNear, zFar;

						if (!jProjection.contains("zNear"))
						{
							logFail("Expected \"zNear\" keyword for planar projection definition!");
							return false;
						}

						if (!jProjection.contains("zFar"))
						{
							logFail("Expected \"zFar\" keyword for planar projection definition!");
							return false;
						}

						zNear = jProjection["zNear"].get<float>();
						zFar = jProjection["zFar"].get<float>();

						if (jProjection["type"] == "perspective")
						{
							if (!jProjection.contains("fov"))
							{
								logFail("Expected \"fov\" keyword for planar perspective projection definition!");
								return false;
							}

							float fov = jProjection["fov"].get<float>();
							projections.emplace_back(IPlanarProjection::CProjection::create<IPlanarProjection::CProjection::Perspective>(zNear, zFar, fov));
						}
						else if (jProjection["type"] == "orthographic")
						{
							if (!jProjection.contains("orthoWidth"))
							{
								logFail("Expected \"orthoWidth\" keyword for planar orthographic projection definition!");
								return false;
							}

							float orthoWidth = jProjection["orthoWidth"].get<float>();
							projections.emplace_back(IPlanarProjection::CProjection::create<IPlanarProjection::CProjection::Orthographic>(zNear, zFar, orthoWidth));
						}
						else
						{
							logFail("Unsupported projection!");
							return false;
						}
					}
				}

				struct
				{
					std::vector<IGimbalManipulateEncoder::keyboard_to_virtual_events_t> keyboard;
					std::vector<IGimbalManipulateEncoder::mouse_to_virtual_events_t> mouse;
				} controllers;

				if (j.contains("controllers"))
				{
					const auto& jControllers = j["controllers"];

					if (jControllers.contains("keyboard"))
					{
						for (const auto& jKeyboard : jControllers["keyboard"])
						{
							if (jKeyboard.contains("mappings"))
							{
								auto& controller = controllers.keyboard.emplace_back();
								for (const auto& [key, value] : jKeyboard["mappings"].items())
								{
									const auto nativeCode = stringToKeyCode(key.c_str());

									if (nativeCode == EKC_NONE)
									{
										logFail("Invalid native key \"%s\" code mapping for keyboard controller", key.c_str());
										return false;
									}

									controller[nativeCode] = CVirtualGimbalEvent::stringToVirtualEvent(value.get<std::string>());
								}
							}
							else
							{
								logFail("Expected \"mappings\" keyword for keyboard controller definition!");
								return false;
							}
						}
					}
					else
					{
						logFail("Expected \"keyboard\" keyword in controllers definition!");
						return false;
					}

					if (jControllers.contains("mouse"))
					{
						for (const auto& jMouse : jControllers["mouse"])
						{
							if (jMouse.contains("mappings"))
							{
								auto& controller = controllers.mouse.emplace_back();
								for (const auto& [key, value] : jMouse["mappings"].items())
								{
									const auto nativeCode = stringToMouseCode(key.c_str());

									if (nativeCode == EMC_NONE)
									{
										logFail("Invalid native key \"%s\" code mapping for mouse controller", key.c_str());
										return false;
									}

									controller[nativeCode] = CVirtualGimbalEvent::stringToVirtualEvent(value.get<std::string>());
								}
							}
							else
							{
								logFail("Expected \"mappings\" keyword for mouse controller definition!");
								return false;
							}
						}
					}
					else
					{
						logFail("Expected \"mouse\" keyword in controllers definition");
						return false;
					}
				}
				else
				{
					logFail("Expected \"controllers\" keyword in controllers JSON");
					return false;
				}

				if (j.contains("viewports") && j.contains("planars"))
				{
					for (const auto& jPlanar : j["planars"])
					{
						if (!jPlanar.contains("camera"))
						{
							logFail("Expected \"camera\" value in planar object");
							return false;
						}

						if (!jPlanar.contains("viewports"))
						{
							logFail("Expected \"viewports\" list in planar object");
							return false;
						}

						const auto cameraIx = jPlanar["camera"].get<uint32_t>();
						auto boundViewports = jPlanar["viewports"].get<std::vector<uint32_t>>();

						auto& planar = m_planarProjections.emplace_back() = planar_projection_t::create(smart_refctd_ptr(cameras[cameraIx]));
						for (const auto viewportIx : boundViewports)
						{
							auto& viewport = j["viewports"][viewportIx];
							if (!viewport.contains("projection") || !viewport.contains("controllers"))
							{
								logFail("\"projection\" or \"controllers\" missing in viewport object index %d", viewportIx);
								return false;
							}

							const auto projectionIx = viewport["projection"].get<uint32_t>();
							auto& projection = planar->getPlanarProjections().emplace_back(projections[projectionIx]);

							const bool hasKeyboardBound = viewport["controllers"].contains("keyboard");
							const bool hasMouseBound = viewport["controllers"].contains("mouse");

							if (hasKeyboardBound)
							{
								auto keyboardControllerIx = viewport["controllers"]["keyboard"].get<uint32_t>();
								projection.updateKeyboardMapping([&](auto& map) { map = controllers.keyboard[keyboardControllerIx]; });
							}
							else
								projection.updateKeyboardMapping([&](auto& map) { map = {}; }); // clean the map if not bound

							if (hasMouseBound)
							{
								auto mouseControllerIx = viewport["controllers"]["mouse"].get<uint32_t>();
								projection.updateMouseMapping([&](auto& map) { map = controllers.mouse[mouseControllerIx]; });
							}
							else
								projection.updateMouseMapping([&](auto& map) { map = {}; }); // clean the map if not bound
						}

						{
							auto* camera = planar->getCamera();
							{
								camera->updateKeyboardMapping([&](auto& map) { map = camera->getKeyboardMappingPreset(); });
								camera->updateMouseMapping([&](auto& map) { map = camera->getMouseMappingPreset(); });
								camera->updateImguizmoMapping([&](auto& map) { map = camera->getImguizmoMappingPreset(); });
							}
						}
					}
				}
				else
				{
					logFail("Expected \"viewports\" and \"planars\" lists in JSON");
					return false;
				}

				if (m_planarProjections.size() < windowBindings.size())
				{
					// TODO, temporary assuming it, I'm not going to implement each possible case now
					logFail("Expected at least %d planars", windowBindings.size());
					return false;
				}

				// init render window planar references - we make all render windows start with focus on first
				// planar but in a way that first window has the planar's perspective preset bound & second orthographic
				for (uint32_t i = 0u; i < windowBindings.size(); ++i)
				{
					auto& binding = windowBindings[i];

					auto& planar = m_planarProjections[binding.activePlanarIx = 0];
					binding.pickDefaultProjections(planar->getPlanarProjections());

					if (i)
						binding.boundProjectionIx = binding.lastBoundOrthoPresetProjectionIx.value();
					else
						binding.boundProjectionIx = binding.lastBoundPerspectivePresetProjectionIx.value();
				}
			}

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
			ISwapchain::SSharedCreationParams sharedParams = {};
			sharedParams.imageUsage |= IGPUImage::EUF_TRANSFER_SRC_BIT;
			auto swapchainResources = std::make_unique<CSwapchainResources>();
			if (!m_surface || !m_surface->init(m_surface->pickQueue(m_device.get()), std::move(swapchainResources), sharedParams))
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
					constexpr size_t ImGuiStreamingBufferSize = 32ull * 1024ull * 1024ull;
					auto createImGuiStreamingBuffer = [&](size_t size) -> smart_refctd_ptr<nbl::ext::imgui::UI::SCachedCreationParams::streaming_buffer_t>
					{
						constexpr uint32_t minStreamingBufferAllocationSize = 128u;
						constexpr uint32_t maxStreamingBufferAllocationAlignment = 4096u;

						auto getRequiredAccessFlags = [&](const bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>& properties)
						{
							bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> flags(IDeviceMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);

							if (properties.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT))
								flags |= IDeviceMemoryAllocation::EMCAF_READ;
							if (properties.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT))
								flags |= IDeviceMemoryAllocation::EMCAF_WRITE;

							return flags;
						};

						IGPUBuffer::SCreationParams mdiCreationParams = {};
						mdiCreationParams.usage = nbl::ext::imgui::UI::SCachedCreationParams::RequiredUsageFlags;
						mdiCreationParams.size = size;

						auto buffer = m_utils->getLogicalDevice()->createBuffer(std::move(mdiCreationParams));
						if (!buffer)
							return nullptr;

						buffer->setObjectDebugName("ImGui MDI Upstream Buffer");

						auto memoryReqs = buffer->getMemoryReqs();
						memoryReqs.memoryTypeBits &= m_utils->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

						auto allocation = m_utils->getLogicalDevice()->allocate(memoryReqs, buffer.get(), nbl::ext::imgui::UI::SCachedCreationParams::RequiredAllocateFlags);
						if (!allocation.isValid())
							return nullptr;

						auto memory = allocation.memory;

						if (!memory->map({ 0ull, memoryReqs.size }, getRequiredAccessFlags(memory->getMemoryPropertyFlags())))
						{
							m_logger->log("Could not map ImGui streaming buffer memory!", ILogger::ELL_ERROR);
							return nullptr;
						}

						return make_smart_refctd_ptr<nbl::ext::imgui::UI::SCachedCreationParams::streaming_buffer_t>(
							SBufferRange<IGPUBuffer>{0ull, mdiCreationParams.size, std::move(buffer)},
							maxStreamingBufferAllocationAlignment,
							minStreamingBufferAllocationSize);
					};

					auto imguiStreamingBuffer = createImGuiStreamingBuffer(ImGuiStreamingBufferSize);
					if (!imguiStreamingBuffer)
						return logFail("Failed to create ImGui streaming buffer.");

					nbl::ext::imgui::UI::SCreationParameters params;
					params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
					params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
					params.assetManager = m_assetManager;
					params.pipelineCache = nullptr;
					params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, TotalUISampleTexturesAmount);
					params.renderpass = smart_refctd_ptr<IGPURenderpass>(m_renderpass);
					params.streamingBuffer = std::move(imguiStreamingBuffer);
					params.subpassIx = 0u;
					params.transfer = getTransferUpQueue();
					params.utilities = m_utils;

					auto loadPrecompiledShader = [&](const std::string_view key) -> smart_refctd_ptr<IShader>
					{
						IAssetLoader::SAssetLoadParams loadParams = {};
						loadParams.logger = m_logger.get();
						loadParams.workingDirectory = "app_resources";
						auto bundle = m_assetManager->getAsset(key.data(), loadParams);
						const auto& contents = bundle.getContents();
						if (contents.empty())
							return nullptr;
						return IAsset::castDown<IShader>(contents[0]);
					};

					const auto vertexKey = nbl::this_example::builtin::build::get_spirv_key<"imgui_vertex">(m_device.get());
					const auto fragmentKey = nbl::this_example::builtin::build::get_spirv_key<"imgui_fragment">(m_device.get());
					auto vertexShader = loadPrecompiledShader(vertexKey.data());
					auto fragmentShader = loadPrecompiledShader(fragmentKey.data());
					if (!vertexShader || !fragmentShader)
						return logFail("Failed to load precompiled ImGui shaders.");

					params.spirv = nbl::ext::imgui::UI::SCreationParameters::PrecompiledShaders{
						.vertex = std::move(vertexShader),
						.fragment = std::move(fragmentShader)
					};

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
				{
					const auto ds = float32_t2{ m_window->getWidth(), m_window->getHeight() };

					wInit.trsEditor.iPos = iPaddingOffset;
					wInit.trsEditor.iSize = { 0.0f, ds.y - wInit.trsEditor.iPos.y * 2 };

					const float panelWidth = std::clamp(ds.x * 0.33f, 380.0f, ds.x * 0.48f);
					wInit.planars.iSize = { panelWidth, ds.y - iPaddingOffset.y * 2 };
					wInit.planars.iPos = { ds.x - wInit.planars.iSize.x - iPaddingOffset.x, 0 + iPaddingOffset.y };

					{
						const float renderPaddingX = 0.0f;
						const float renderPaddingY = 0.0f;
						const float splitGap = 4.0f;
						float leftX = renderPaddingX;
						float eachXSize = std::max(0.0f, ds.x - 2.0f * renderPaddingX);
						float eachYSize = (ds.y - 2.0f * renderPaddingY - (wInit.renderWindows.size() - 1) * splitGap) / wInit.renderWindows.size();
						
						for (size_t i = 0; i < wInit.renderWindows.size(); ++i)
						{
							auto& rw = wInit.renderWindows[i];
							rw.iPos = { leftX, renderPaddingY + i * (eachYSize + splitGap) };
							rw.iSize = { eachXSize, eachYSize };
						}
					}
				}
			}

			// Geometry Creator Render Scene FBOs
			{
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

				if (!m_scene)
					return logFail("Could not create geometry creator scene!");

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
							/*.finalLayout = */ IGPUImage::LAYOUT::READ_ONLY_OPTIMAL
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
						{
							.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
							.dstSubpass = 0,
							.memoryBarrier = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT|PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
								.srcAccessMask = ACCESS_FLAGS::NONE,
								.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT|PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
								.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
							}
						},
						{
							.srcSubpass = 0,
							.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
							.memoryBarrier = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
								.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
								.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT|PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
								.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT
							}
						},
						IGPURenderpass::SCreationParams::DependenciesEnd
					};
					params.dependencies = {};
					m_sceneRenderpass = m_device->createRenderpass(std::move(params));
					if (!m_sceneRenderpass)
						return logFail("Failed to create Scene Renderpass!");
				}

				const auto& geometries = m_scene->getInitParams().geometries;
				if (geometries.empty())
					return logFail("No geometries found for scene!");
				m_renderer = CSimpleDebugRenderer::create(m_assetManager.get(), m_sceneRenderpass.get(), 0, { &geometries.front().get(), geometries.size() });
				if (!m_renderer)
					return logFail("Failed to create debug renderer!");

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
				m_renderer->m_instances.resize(1);

				const auto dpyInfo = m_winMgr->getPrimaryDisplayInfo();
				for (uint32_t i = 0u; i < windowBindings.size(); ++i)
				{
					auto& binding = windowBindings[i];
					binding.sceneColorView = createAttachmentView(m_device.get(), finalSceneRenderFormat, dpyInfo.resX, dpyInfo.resY, "UI Scene Color Attachment");
					binding.sceneDepthView = createAttachmentView(m_device.get(), sceneRenderDepthFormat, dpyInfo.resX, dpyInfo.resY, "UI Scene Depth Attachment");
					binding.sceneFramebuffer = createSceneFramebuffer(m_device.get(), m_sceneRenderpass.get(), binding.sceneColorView.get(), binding.sceneDepthView.get());
					if (!binding.sceneFramebuffer)
						return logFail("Could not create geometry creator scene[%d]!", i);
				}
			}

			oracle.reportBeginFrameRecord();

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
				willSubmit &= cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
				willSubmit &= cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				willSubmit &= cmdbuf->beginDebugMarker("UIApp Frame");

				auto renderScene = [&](windowControlBinding& binding)
				{
					if (!binding.sceneFramebuffer)
						return;

					const auto& fbParams = binding.sceneFramebuffer->getCreationParameters();
					const VkRect2D renderArea = { .offset = {0,0}, .extent = {fbParams.width, fbParams.height} };
					const IGPUCommandBuffer::SRenderpassBeginInfo info = {
						.framebuffer = binding.sceneFramebuffer.get(),
						.colorClearValues = &SceneClearColor,
						.depthStencilClearValues = &SceneClearDepth,
						.renderArea = renderArea
					};

					willSubmit &= cmdbuf->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
					{
						asset::SViewport viewport = {};
						viewport.minDepth = 1.f;
						viewport.maxDepth = 0.f;
						viewport.x = 0u;
						viewport.y = 0u;
						viewport.width = fbParams.width;
						viewport.height = fbParams.height;

						willSubmit &= cmdbuf->setViewport(0u, 1u, &viewport);
						willSubmit &= cmdbuf->setScissor(0u, 1u, &renderArea);

						const auto viewParams = CSimpleDebugRenderer::SViewParams(binding.viewMatrix, binding.viewProjMatrix);
						m_renderer->render(cmdbuf, viewParams);

					}
					willSubmit &= cmdbuf->endRenderPass();
				};

				if (m_renderer && !m_renderer->m_instances.empty())
				{
					auto& instance = m_renderer->m_instances[0];
					instance.world = m_model;
					const auto geomCount = m_renderer->getGeometries().size();
					if (geomCount)
					{
						if (gcIndex >= geomCount)
							gcIndex = 0;
						instance.packedGeo = m_renderer->getGeometries().data() + gcIndex;
					}
				}

				if (useWindow)
					for (auto& binding : windowBindings)
						renderScene(binding);
				else
					renderScene(windowBindings[activeRenderWindowIx]);
				
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
						.colorClearValues = &clearValue,
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

				updateGUIDescriptorSet();

				if (getGraphicsQueue()->submit(submitInfos) != IQueue::RESULT::SUCCESS)
					return;

				m_realFrameIx++;

				const uint64_t renderedFrameIx = m_realFrameIx - 1u;
				auto captureScreenshot = [&](const system::path& outPath, const char* tag) -> void
				{
					if (!m_device || !m_assetManager || !m_surface)
						return;

					m_logger->log("%s screenshot capture start (frame %llu).", ILogger::ELL_INFO, tag, static_cast<unsigned long long>(renderedFrameIx));
					const ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx };
					if (m_device->blockForSemaphores({ &waitInfo, &waitInfo + 1 }) != ISemaphore::WAIT_RESULT::SUCCESS)
					{
						m_logger->log("%s screenshot failed: wait for render finished.", ILogger::ELL_ERROR, tag);
						return;
					}

					if (!frame)
					{
						m_logger->log("%s screenshot failed: missing frame image.", ILogger::ELL_ERROR, tag);
						return;
					}

					auto viewParams = IGPUImageView::SCreationParams{
						.subUsages = IGPUImage::EUF_TRANSFER_SRC_BIT,
						.image = core::smart_refctd_ptr<IGPUImage>(frame),
						.viewType = IGPUImageView::ET_2D,
						.format = frame->getCreationParameters().format
					};
					viewParams.subresourceRange.aspectMask = IGPUImage::EAF_COLOR_BIT;
					viewParams.subresourceRange.baseMipLevel = 0u;
					viewParams.subresourceRange.levelCount = 1u;
					viewParams.subresourceRange.baseArrayLayer = 0u;
					viewParams.subresourceRange.layerCount = 1u;
					auto frameView = m_device->createImageView(std::move(viewParams));
					if (!frameView)
					{
						m_logger->log("%s screenshot failed: could not create frame view.", ILogger::ELL_ERROR, tag);
						return;
					}

					m_logger->log("%s screenshot capture: calling createScreenShot.", ILogger::ELL_INFO, tag);
					const bool ok = ext::ScreenShot::createScreenShot(
						m_device.get(),
						getGraphicsQueue(),
						nullptr,
						frameView.get(),
						m_assetManager.get(),
						outPath,
						asset::IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
						asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT);

					if (ok)
						m_logger->log("%s screenshot saved to \"%s\".", ILogger::ELL_INFO, tag, outPath.string().c_str());
					else
						m_logger->log("%s screenshot failed to save.", ILogger::ELL_ERROR, tag);
				};

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
				if (m_ciMode && !m_ciScreenshotDone)
				{
					++m_ciFrameCounter;
					if (m_ciFrameCounter >= CiFramesBeforeCapture)
					{
						m_ciScreenshotDone = true;
						captureScreenshot(m_ciScreenshotPath, "CI");
					}
				}

				if (m_scriptedInput.enabled && !m_scriptedInput.captureFrames.empty())
				{
					while (m_scriptedInput.nextCaptureIndex < m_scriptedInput.captureFrames.size() &&
						m_scriptedInput.captureFrames[m_scriptedInput.nextCaptureIndex] == renderedFrameIx)
					{
						const auto outPath = m_scriptedInput.captureOutputDir /
							(m_scriptedInput.capturePrefix + "_" + std::to_string(renderedFrameIx) + ".png");
						captureScreenshot(outPath, "Script");
						++m_scriptedInput.nextCaptureIndex;
					}
				}

				m_surface->present(std::move(swapchainLock), presentInfo);
			}
			firstFrame = false;
		}

		inline bool keepRunning() override
		{
			if (m_scriptedInput.enabled && m_scriptedInput.hardFail && m_scriptedInput.failed)
			{
				if (!m_ciMode || m_ciScreenshotDone)
					std::exit(EXIT_FAILURE);
			}
			if (m_ciMode && m_ciScreenshotDone)
			{
				if (m_scriptedInput.enabled && m_scriptedInput.nextCaptureIndex < m_scriptedInput.captureFrames.size())
					return true;
				return false;
			}
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
			if (m_haveLastPresentationTimestamp)
			{
				const auto delta = m_nextPresentationTimestamp - m_lastPresentationTimestamp;
				if (delta.count() < 0)
					m_frameDeltaSec = 0.0;
				else
					m_frameDeltaSec = static_cast<double>(delta.count()) / 1000000.0;
			}
			m_lastPresentationTimestamp = m_nextPresentationTimestamp;
			m_haveLastPresentationTimestamp = true;

			updatePlayback(m_frameDeltaSec);
			const bool skipCameraInput = m_playback.playing && m_playback.overrideInput;

			struct
			{
				std::vector<SMouseEvent> mouse {};
				std::vector<SKeyboardEvent> keyboard {};
			} capturedEvents;
			{
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
				{
					if (m_window->hasInputFocus())
						for (const auto& e : events)
							capturedEvents.mouse.emplace_back(e);
				}, m_logger.get());

				keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
				{
					if (m_window->hasInputFocus())
						for (const auto& e : events)
							capturedEvents.keyboard.emplace_back(e);
				}, m_logger.get());
			}

			if (m_scriptedInput.enabled && m_scriptedInput.exclusive)
			{
				capturedEvents.mouse.clear();
				capturedEvents.keyboard.clear();
			}

			std::vector<SMouseEvent> scriptedMouse;
			std::vector<SKeyboardEvent> scriptedKeyboard;
			std::vector<float32_t4x4> scriptedImguizmo;
			std::vector<ScriptedInputEvent::ActionData> scriptedActions;
			const CVirtualGimbalEvent* scriptedImguizmoVirtual = nullptr;
			uint32_t scriptedImguizmoVirtualCount = 0u;

			if (m_scriptedInput.enabled && m_scriptedInput.nextEventIndex < m_scriptedInput.events.size())
			{
				const auto frame = m_realFrameIx;
				while (m_scriptedInput.nextEventIndex < m_scriptedInput.events.size() &&
					m_scriptedInput.events[m_scriptedInput.nextEventIndex].frame == frame)
				{
					const auto& ev = m_scriptedInput.events[m_scriptedInput.nextEventIndex];

					if (ev.type == ScriptedInputEvent::Type::Keyboard)
					{
						SKeyboardEvent e(m_nextPresentationTimestamp);
						e.keyCode = ev.keyboard.key;
						e.action = ev.keyboard.action;
						e.window = m_window.get();
						scriptedKeyboard.emplace_back(e);
					}
					else if (ev.type == ScriptedInputEvent::Type::Mouse)
					{
						SMouseEvent e(m_nextPresentationTimestamp);
						e.window = m_window.get();
						e.type = ev.mouse.type;
						if (ev.mouse.type == ui::SMouseEvent::EET_CLICK)
						{
							e.clickEvent.mouseButton = ev.mouse.button;
							e.clickEvent.action = ev.mouse.action;
							e.clickEvent.clickPosX = ev.mouse.x;
							e.clickEvent.clickPosY = ev.mouse.y;
						}
						else if (ev.mouse.type == ui::SMouseEvent::EET_SCROLL)
						{
							e.scrollEvent.verticalScroll = ev.mouse.v;
							e.scrollEvent.horizontalScroll = ev.mouse.h;
						}
						else if (ev.mouse.type == ui::SMouseEvent::EET_MOVEMENT)
						{
							e.movementEvent.relativeMovementX = ev.mouse.dx;
							e.movementEvent.relativeMovementY = ev.mouse.dy;
						}
						scriptedMouse.emplace_back(e);
					}
					else if (ev.type == ScriptedInputEvent::Type::Imguizmo)
					{
						scriptedImguizmo.emplace_back(ev.imguizmo);
					}
					else if (ev.type == ScriptedInputEvent::Type::Action)
					{
						scriptedActions.emplace_back(ev.action);
					}

					++m_scriptedInput.nextEventIndex;
				}
			}

			if (m_scriptedInput.enabled && scriptedActions.size())
			{
				auto applyAction = [&](const ScriptedInputEvent::ActionData& action) -> void
				{
					switch (action.kind)
					{
						case ScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow:
						{
							if (action.value < 0 || static_cast<size_t>(action.value) >= windowBindings.size())
							{
								m_logger->log("[script][warn] action set_active_render_window out of range: %d", ILogger::ELL_WARNING, action.value);
								return;
							}
							activeRenderWindowIx = static_cast<uint32_t>(action.value);
						} break;

						case ScriptedInputEvent::ActionData::Kind::SetActivePlanar:
						{
							if (action.value < 0 || static_cast<size_t>(action.value) >= m_planarProjections.size())
							{
								m_logger->log("[script][warn] action set_active_planar out of range: %d", ILogger::ELL_WARNING, action.value);
								return;
							}
							auto& binding = windowBindings[activeRenderWindowIx];
							binding.activePlanarIx = static_cast<uint32_t>(action.value);
							binding.pickDefaultProjections(m_planarProjections[binding.activePlanarIx]->getPlanarProjections());
						} break;

						case ScriptedInputEvent::ActionData::Kind::SetProjectionType:
						{
							auto& binding = windowBindings[activeRenderWindowIx];
							if (!binding.lastBoundPerspectivePresetProjectionIx.has_value() || !binding.lastBoundOrthoPresetProjectionIx.has_value())
								binding.pickDefaultProjections(m_planarProjections[binding.activePlanarIx]->getPlanarProjections());

							const auto type = static_cast<IPlanarProjection::CProjection::ProjectionType>(action.value);
							switch (type)
							{
								case IPlanarProjection::CProjection::Perspective:
									binding.boundProjectionIx = binding.lastBoundPerspectivePresetProjectionIx.value();
									break;
								case IPlanarProjection::CProjection::Orthographic:
									binding.boundProjectionIx = binding.lastBoundOrthoPresetProjectionIx.value();
									break;
								default:
									m_logger->log("[script][warn] action set_projection_type invalid value: %d", ILogger::ELL_WARNING, action.value);
									break;
							}
						} break;

						case ScriptedInputEvent::ActionData::Kind::SetProjectionIndex:
						{
							auto& binding = windowBindings[activeRenderWindowIx];
							auto& projections = m_planarProjections[binding.activePlanarIx]->getPlanarProjections();
							if (action.value < 0 || static_cast<size_t>(action.value) >= projections.size())
							{
								m_logger->log("[script][warn] action set_projection_index out of range: %d", ILogger::ELL_WARNING, action.value);
								return;
							}
							const auto ix = static_cast<uint32_t>(action.value);
							const auto type = projections[ix].getParameters().m_type;
							binding.boundProjectionIx = ix;
							if (type == IPlanarProjection::CProjection::Perspective)
								binding.lastBoundPerspectivePresetProjectionIx = ix;
							else if (type == IPlanarProjection::CProjection::Orthographic)
								binding.lastBoundOrthoPresetProjectionIx = ix;
						} break;

						case ScriptedInputEvent::ActionData::Kind::SetUseWindow:
						{
							useWindow = action.value != 0;
						} break;

						case ScriptedInputEvent::ActionData::Kind::SetLeftHanded:
						{
							auto& binding = windowBindings[activeRenderWindowIx];
							binding.leftHandedProjection = action.value != 0;
						} break;
					}
				};

				for (const auto& action : scriptedActions)
					if (action.kind == ScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow)
						applyAction(action);

				for (const auto& action : scriptedActions)
					if (action.kind != ScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow)
						applyAction(action);

				if (m_scriptedInput.log)
					m_logger->log("[script] frame %llu actions=%zu", ILogger::ELL_INFO, static_cast<unsigned long long>(m_realFrameIx), scriptedActions.size());
			}

			if (!scriptedMouse.empty())
				capturedEvents.mouse.insert(capturedEvents.mouse.end(), scriptedMouse.begin(), scriptedMouse.end());
			if (!scriptedKeyboard.empty())
				capturedEvents.keyboard.insert(capturedEvents.keyboard.end(), scriptedKeyboard.begin(), scriptedKeyboard.end());

			m_uiInputEventsThisFrame = static_cast<uint32_t>(capturedEvents.mouse.size() + capturedEvents.keyboard.size());

			auto cameraKeyboardEvents = capturedEvents.keyboard;
			auto cameraMouseEvents = capturedEvents.mouse;
			for (auto& ev : cameraMouseEvents)
			{
				if (ev.type == ui::SMouseEvent::EET_SCROLL)
				{
					ev.scrollEvent.verticalScroll *= m_cameraControls.mouseScrollScale;
					ev.scrollEvent.horizontalScroll *= m_cameraControls.mouseScrollScale;
				}
				else if (ev.type == ui::SMouseEvent::EET_MOVEMENT)
				{
					ev.movementEvent.relativeMovementX *= m_cameraControls.mouseMoveScale;
					ev.movementEvent.relativeMovementY *= m_cameraControls.mouseMoveScale;
				}
			}

			const auto cursorPosition = m_window->getCursorControl()->getPosition();

			nbl::ext::imgui::UI::SUpdateParameters params =
			{
				.mousePosition = nbl::hlsl::float32_t2(cursorPosition.x, cursorPosition.y) - nbl::hlsl::float32_t2(m_window->getX(), m_window->getY()),
				.displaySize = { m_window->getWidth(), m_window->getHeight() },
				.mouseEvents = { capturedEvents.mouse.data(), capturedEvents.mouse.size() },
				.keyboardEvents = { capturedEvents.keyboard.data(), capturedEvents.keyboard.size() }
			};

			if (m_scriptedInput.log && (scriptedKeyboard.size() || scriptedMouse.size() || scriptedImguizmo.size()))
			{
				m_logger->log("[script] frame %llu input kb=%zu mouse=%zu imguizmo=%zu", ILogger::ELL_INFO,
					static_cast<unsigned long long>(m_realFrameIx),
					scriptedKeyboard.size(),
					scriptedMouse.size(),
					scriptedImguizmo.size());
			}

			if (enableActiveCameraMovement && !skipCameraInput)
			{
				auto& binding = windowBindings[activeRenderWindowIx];
				auto& planar = m_planarProjections[binding.activePlanarIx];
				auto* camera = planar->getCamera();

				assert(binding.boundProjectionIx.has_value());
				auto& projection = planar->getPlanarProjections()[binding.boundProjectionIx.value()];

				static std::vector<CVirtualGimbalEvent> virtualEvents(0x45);
				uint32_t vCount = {};
				uint32_t vKeyboardEventsCount = {};
				uint32_t vMouseEventsCount = {};

				projection.beginInputProcessing(m_nextPresentationTimestamp);
				{
					projection.processKeyboard(nullptr, vKeyboardEventsCount, {});
					projection.processMouse(nullptr, vMouseEventsCount, {});

					const auto totalCount = vKeyboardEventsCount + vMouseEventsCount;
					if (virtualEvents.size() < totalCount)
						virtualEvents.resize(totalCount);

					auto* output = virtualEvents.data();
					projection.processKeyboard(output, vKeyboardEventsCount, { cameraKeyboardEvents.data(), cameraKeyboardEvents.size() });
					for (uint32_t i = 0u; i < vKeyboardEventsCount; ++i)
						output[i].magnitude *= m_cameraControls.keyboardScale;
					output += vKeyboardEventsCount;

					if (isOrbitLikeCamera(camera))
					{
						if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
							projection.processMouse(output, vMouseEventsCount, { cameraMouseEvents.data(), cameraMouseEvents.size() });
						else
							vMouseEventsCount = 0;
					}
					else
					{
						projection.processMouse(output, vMouseEventsCount, { cameraMouseEvents.data(), cameraMouseEvents.size() });
					}

					vCount = vKeyboardEventsCount + vMouseEventsCount;
				}
				projection.endInputProcessing();

				if (vCount)
				{
					applyVirtualEventScaling(virtualEvents, vCount);

					const char* controllerLabel = "Keyboard/Mouse";
					auto applyEventsToCamera = [&](ICamera* target, uint32_t planarIx)
					{
						if (!target)
							return;

						if (m_cameraControls.worldTranslate)
						{
							std::vector<CVirtualGimbalEvent> perCameraEvents(virtualEvents.begin(), virtualEvents.begin() + vCount);
							uint32_t perCount = vCount;
							remapTranslationToWorld(target, perCameraEvents, perCount);
							if (perCount)
								target->manipulate({ perCameraEvents.data(), perCount });
						}
						else
						{
							target->manipulate({ virtualEvents.data(), vCount });
						}

						applyConstraintsToCamera(target);
						appendVirtualEventLog("input", controllerLabel, planarIx, target, virtualEvents.data(), vCount);
					};

					if (m_cameraControls.mirrorInput)
					{
						std::unordered_set<uintptr_t> visited;
						for (size_t bindingIx = 0u; bindingIx < windowBindings.size(); ++bindingIx)
						{
							auto& bindingIt = windowBindings[bindingIx];
							auto& planarIt = m_planarProjections[bindingIt.activePlanarIx];
							if (!planarIt)
								continue;
							auto* target = planarIt->getCamera();
							if (!target)
								continue;
							const auto id = target->getGimbal().getID();
							if (!visited.insert(id).second)
								continue;
							applyEventsToCamera(target, bindingIt.activePlanarIx);
						}
					}
					else
					{
						applyEventsToCamera(camera, binding.activePlanarIx);
					}

					if (m_scriptedInput.log)
					{
						for (uint32_t i = 0u; i < vCount; ++i)
						{
							const auto& ev = virtualEvents[i];
							m_logger->log("[script] virtual %s magnitude=%.6f", ILogger::ELL_INFO, CVirtualGimbalEvent::virtualEventToString(ev.type).data(), ev.magnitude);
						}

						const auto& gimbal = camera->getGimbal();
						const auto pos = gimbal.getPosition();
						const auto euler = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));
						m_logger->log("[script] gimbal pos=(%.3f, %.3f, %.3f) euler_deg=(%.3f, %.3f, %.3f)", ILogger::ELL_INFO,
							pos.x, pos.y, pos.z, euler.x, euler.y, euler.z);
					}
				}
			}

			if (m_scriptedInput.enabled && scriptedImguizmo.size() && !skipCameraInput)
			{
				auto& binding = windowBindings[activeRenderWindowIx];
				auto& planar = m_planarProjections[binding.activePlanarIx];
				auto* camera = planar->getCamera();

				static std::vector<CVirtualGimbalEvent> imguizmoEvents(0x20);
				uint32_t vCount = 0u;

				camera->beginInputProcessing(m_nextPresentationTimestamp);
				{
					camera->processImguizmo(nullptr, vCount, {});
					if (imguizmoEvents.size() < vCount)
						imguizmoEvents.resize(vCount);

					camera->processImguizmo(imguizmoEvents.data(), vCount, { scriptedImguizmo.data(), scriptedImguizmo.size() });
				}
				camera->endInputProcessing();

				if (vCount)
				{
					camera->manipulate({ imguizmoEvents.data(), vCount });
					appendVirtualEventLog("imguizmo", "ImGuizmo", binding.activePlanarIx, camera, imguizmoEvents.data(), vCount);

					if (m_scriptedInput.log)
					{
						for (uint32_t i = 0u; i < vCount; ++i)
						{
							const auto& ev = imguizmoEvents[i];
							m_logger->log("[script] imguizmo virtual %s magnitude=%.6f", ILogger::ELL_INFO, CVirtualGimbalEvent::virtualEventToString(ev.type).data(), ev.magnitude);
						}

						const auto& gimbal = camera->getGimbal();
						const auto pos = gimbal.getPosition();
						const auto euler = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));
						m_logger->log("[script] imguizmo gimbal pos=(%.3f, %.3f, %.3f) euler_deg=(%.3f, %.3f, %.3f)", ILogger::ELL_INFO,
							pos.x, pos.y, pos.z, euler.x, euler.y, euler.z);
					}
				}

				scriptedImguizmoVirtual = vCount ? imguizmoEvents.data() : nullptr;
				scriptedImguizmoVirtualCount = vCount;
			}

			if (m_scriptedInput.enabled && m_scriptedInput.nextCheckIndex < m_scriptedInput.checks.size())
			{
				auto* camera = [&]() -> ICamera*
				{
					if (m_planarProjections.empty())
						return nullptr;
					auto& binding = windowBindings[activeRenderWindowIx];
					if (binding.activePlanarIx >= m_planarProjections.size())
						return nullptr;
					return m_planarProjections[binding.activePlanarIx]->getCamera();
				}();

				auto logFail = [&](const char* fmt, auto&&... args) -> void
				{
					m_scriptedInput.failed = true;
					m_logger->log(fmt, ILogger::ELL_ERROR, std::forward<decltype(args)>(args)...);
				};

				auto logPass = [&](const char* fmt, auto&&... args) -> void
				{
					if (!m_scriptedInput.log)
						return;
					m_logger->log(fmt, ILogger::ELL_INFO, std::forward<decltype(args)>(args)...);
				};

				auto angleDiffDeg = [](float a, float b) -> float
				{
					float d = std::fmod(a - b + 180.0f, 360.0f);
					if (d < 0.0f)
						d += 360.0f;
					return std::abs(d - 180.0f);
				};

				auto isFinite3 = [](const float32_t3& v) -> bool
				{
					return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
				};

				const auto frame = m_realFrameIx;
				while (m_scriptedInput.nextCheckIndex < m_scriptedInput.checks.size() &&
					m_scriptedInput.checks[m_scriptedInput.nextCheckIndex].frame == frame)
				{
					const auto& check = m_scriptedInput.checks[m_scriptedInput.nextCheckIndex];

					if (!camera)
					{
						logFail("[script][fail] check frame=%llu no active camera", static_cast<unsigned long long>(frame));
						++m_scriptedInput.nextCheckIndex;
						continue;
					}

					const auto& gimbal = camera->getGimbal();
					const auto pos = gimbal.getPosition();
					const auto euler = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));

					if (!isFinite3(pos) || !isFinite3(euler))
					{
						logFail("[script][fail] check frame=%llu non-finite gimbal state", static_cast<unsigned long long>(frame));
						++m_scriptedInput.nextCheckIndex;
						continue;
					}

					if (check.kind == ScriptedInputCheck::Kind::Baseline)
					{
						m_scriptedInput.baselineValid = true;
						m_scriptedInput.baselinePos = pos;
						m_scriptedInput.baselineEulerDeg = euler;
						logPass("[script][pass] baseline frame=%llu pos=(%.3f, %.3f, %.3f) euler_deg=(%.3f, %.3f, %.3f)",
							static_cast<unsigned long long>(frame),
							pos.x, pos.y, pos.z, euler.x, euler.y, euler.z);
					}
					else if (check.kind == ScriptedInputCheck::Kind::ImguizmoVirtual)
					{
						bool ok = true;
						if (!scriptedImguizmoVirtual || scriptedImguizmoVirtualCount == 0u)
						{
							ok = false;
						}
						else
						{
							for (const auto& expected : check.expectedVirtualEvents)
							{
								bool found = false;
								double actual = 0.0;
								for (uint32_t i = 0u; i < scriptedImguizmoVirtualCount; ++i)
								{
									if (scriptedImguizmoVirtual[i].type == expected.type)
									{
										found = true;
										actual = scriptedImguizmoVirtual[i].magnitude;
										break;
									}
								}
								if (!found || std::abs(actual - expected.magnitude) > check.tolerance)
								{
									ok = false;
									logFail("[script][fail] imguizmo_virtual frame=%llu type=%s expected=%.6f actual=%.6f tol=%.6f",
										static_cast<unsigned long long>(frame),
										CVirtualGimbalEvent::virtualEventToString(expected.type).data(),
										expected.magnitude,
										actual,
										check.tolerance);
								}
							}
						}

						if (ok)
							logPass("[script][pass] imguizmo_virtual frame=%llu events=%zu", static_cast<unsigned long long>(frame), check.expectedVirtualEvents.size());
					}
					else if (check.kind == ScriptedInputCheck::Kind::GimbalNear)
					{
						bool ok = true;
						if (check.hasExpectedPos)
						{
							const auto diff = float32_t3(pos.x - check.expectedPos.x, pos.y - check.expectedPos.y, pos.z - check.expectedPos.z);
							const auto d = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
							if (d > check.posTolerance)
							{
								ok = false;
								logFail("[script][fail] gimbal_near frame=%llu pos_diff=%.6f tol=%.6f",
									static_cast<unsigned long long>(frame), d, check.posTolerance);
							}
						}
						if (check.hasExpectedEuler)
						{
							const auto dx = angleDiffDeg(euler.x, check.expectedEulerDeg.x);
							const auto dy = angleDiffDeg(euler.y, check.expectedEulerDeg.y);
							const auto dz = angleDiffDeg(euler.z, check.expectedEulerDeg.z);
							const auto dmax = std::max(dx, std::max(dy, dz));
							if (dmax > check.eulerToleranceDeg)
							{
								ok = false;
								logFail("[script][fail] gimbal_near frame=%llu euler_diff=%.6f tol=%.6f",
									static_cast<unsigned long long>(frame), dmax, check.eulerToleranceDeg);
							}
						}

						if (ok)
							logPass("[script][pass] gimbal_near frame=%llu", static_cast<unsigned long long>(frame));
					}
					else if (check.kind == ScriptedInputCheck::Kind::GimbalDelta)
					{
						if (!m_scriptedInput.baselineValid)
						{
							logFail("[script][fail] gimbal_delta frame=%llu missing baseline", static_cast<unsigned long long>(frame));
						}
						else
						{
							const auto diff = float32_t3(pos.x - m_scriptedInput.baselinePos.x, pos.y - m_scriptedInput.baselinePos.y, pos.z - m_scriptedInput.baselinePos.z);
							const auto dpos = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
							const auto dx = angleDiffDeg(euler.x, m_scriptedInput.baselineEulerDeg.x);
							const auto dy = angleDiffDeg(euler.y, m_scriptedInput.baselineEulerDeg.y);
							const auto dz = angleDiffDeg(euler.z, m_scriptedInput.baselineEulerDeg.z);
							const auto dmax = std::max(dx, std::max(dy, dz));

							if (dpos > check.posTolerance || dmax > check.eulerToleranceDeg)
							{
								logFail("[script][fail] gimbal_delta frame=%llu pos_diff=%.6f tol=%.6f euler_diff=%.6f tol=%.6f",
									static_cast<unsigned long long>(frame),
									dpos, check.posTolerance,
									dmax, check.eulerToleranceDeg);
							}
							else
							{
								logPass("[script][pass] gimbal_delta frame=%llu pos_diff=%.6f euler_diff=%.6f",
									static_cast<unsigned long long>(frame), dpos, dmax);
							}
						}
					}

					++m_scriptedInput.nextCheckIndex;
				}

				if (!m_scriptedInput.summaryReported && m_scriptedInput.nextCheckIndex >= m_scriptedInput.checks.size())
				{
					m_scriptedInput.summaryReported = true;
					if (m_scriptedInput.failed)
						m_logger->log("[script] checks result: FAIL", ILogger::ELL_ERROR);
					else
						m_logger->log("[script] checks result: PASS", ILogger::ELL_INFO);
				}
			}

			UpdateUiMetrics();
			m_ui.manager->update(params);

			}

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
			float distance = 0.f;
			bool hasDistance = false;
			double orbitU = 0.0;
			double orbitV = 0.0;
			float orbitDistance = 0.f;
			bool hasOrbitState = false;
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
			return dynamic_cast<COrbitCamera*>(camera) || dynamic_cast<CArcballCamera*>(camera) || dynamic_cast<CTurntableCamera*>(camera);
		}

		template<typename Fn>
		inline bool withOrbitLikeCamera(ICamera* camera, Fn&& fn)
		{
			if (auto* orbit = dynamic_cast<COrbitCamera*>(camera))
			{
				fn(orbit);
				return true;
			}
			if (auto* arcball = dynamic_cast<CArcballCamera*>(camera))
			{
				fn(arcball);
				return true;
			}
			if (auto* turntable = dynamic_cast<CTurntableCamera*>(camera))
			{
				fn(turntable);
				return true;
			}
			return false;
		}

		inline std::string_view getCameraTypeLabel(const ICamera* camera) const
		{
			if (dynamic_cast<const CFPSCamera*>(camera))
				return "FPS";
			if (dynamic_cast<const CFreeCamera*>(camera))
				return "Free";
			if (dynamic_cast<const COrbitCamera*>(camera))
				return "Orbit";
			if (dynamic_cast<const CArcballCamera*>(camera))
				return "Arcball";
			if (dynamic_cast<const CTurntableCamera*>(camera))
				return "Turntable";
			return "Unknown";
		}

		inline std::string_view getCameraTypeDescription(const ICamera* camera) const
		{
			if (dynamic_cast<const CFPSCamera*>(camera))
				return "First-person WASD + mouse look";
			if (dynamic_cast<const CFreeCamera*>(camera))
				return "Free-fly 6DOF with full rotation";
			if (dynamic_cast<const COrbitCamera*>(camera))
				return "Orbit around target with dolly";
			if (dynamic_cast<const CArcballCamera*>(camera))
				return "Arcball trackball around target";
			if (dynamic_cast<const CTurntableCamera*>(camera))
				return "Turntable yaw/pitch around target";
			return "Unspecified camera behavior";
		}

		inline CameraPreset capturePreset(ICamera* camera, const std::string& name)
		{
			CameraPreset preset;
			preset.name = name;
			if (!camera)
				return preset;

			preset.identifier = std::string(camera->getIdentifier());
			const auto& gimbal = camera->getGimbal();
			preset.position = gimbal.getPosition();
			preset.orientation = gimbal.getOrientation();

			auto captureOrbit = [&](auto* orbit)
			{
				preset.distance = orbit->getDistance();
				preset.hasDistance = true;
				preset.orbitDistance = orbit->getDistance();
				preset.orbitU = orbit->getU();
				preset.orbitV = orbit->getV();
				preset.hasOrbitState = true;
			};

			withOrbitLikeCamera(camera, captureOrbit);

			return preset;
		}

		inline bool applyPresetToCamera(ICamera* camera, const CameraPreset& preset)
		{
			if (!camera)
				return false;

			CTargetPose target;
			target.position = preset.position;
			target.orientation = preset.orientation;
			target.hasDistance = preset.hasDistance;
			target.distance = preset.distance;
			target.hasOrbitState = preset.hasOrbitState;
			target.orbitU = preset.orbitU;
			target.orbitV = preset.orbitV;
			target.orbitDistance = preset.orbitDistance;

			return m_targetPoseController.apply(camera, target);
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
				auto line = m_logFormatter.format(ILogger::ELL_INFO,
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

			auto clampOrbitDistance = [&](auto* orbit)
			{
				if (m_cameraConstraints.clampDistance)
				{
					const float clamped = std::clamp<float>(orbit->getDistance(), m_cameraConstraints.minDistance, m_cameraConstraints.maxDistance);
					orbit->setDistance(clamped);
				}
			};

			if (withOrbitLikeCamera(camera, clampOrbitDistance))
				return;

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
			preset.orientation = glm::quat(glm::radians(clamped));
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
				glm::dot(worldDelta, right),
				glm::dot(worldDelta, up),
				glm::dot(worldDelta, forward)
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

			CameraPreset blended;
			blended.position = a.preset.position + (b.preset.position - a.preset.position) * alpha;
			blended.orientation = glm::slerp(a.preset.orientation, b.preset.orientation, static_cast<float>(alpha));
			blended.hasDistance = a.preset.hasDistance || b.preset.hasDistance;
			if (blended.hasDistance)
			{
				const float da = a.preset.hasDistance ? a.preset.distance : b.preset.distance;
				const float db = b.preset.hasDistance ? b.preset.distance : a.preset.distance;
				blended.distance = da + (db - da) * static_cast<float>(alpha);
			}
			blended.hasOrbitState = a.preset.hasOrbitState || b.preset.hasOrbitState;
			if (blended.hasOrbitState)
			{
				const double ua = a.preset.hasOrbitState ? a.preset.orbitU : b.preset.orbitU;
				const double ub = b.preset.hasOrbitState ? b.preset.orbitU : a.preset.orbitU;
				const double va = a.preset.hasOrbitState ? a.preset.orbitV : b.preset.orbitV;
				const double vb = b.preset.hasOrbitState ? b.preset.orbitV : a.preset.orbitV;
				const float da = a.preset.hasOrbitState ? a.preset.orbitDistance : b.preset.orbitDistance;
				const float db = b.preset.hasOrbitState ? b.preset.orbitDistance : a.preset.orbitDistance;

				blended.orbitU = ua + (ub - ua) * alpha;
				blended.orbitV = va + (vb - va) * alpha;
				blended.orbitDistance = da + (db - da) * static_cast<float>(alpha);
			}

			applyPresetToTargets(blended);
		}

		inline bool savePresetsToFile(const system::path& path)
		{
			json root;
			root["presets"] = json::array();

			for (const auto& preset : m_presets)
			{
				json j;
				j["name"] = preset.name;
				j["identifier"] = preset.identifier;
				j["position"] = { preset.position.x, preset.position.y, preset.position.z };
				j["orientation"] = { preset.orientation.x, preset.orientation.y, preset.orientation.z, preset.orientation.w };
				if (preset.hasDistance)
					j["distance"] = preset.distance;
				if (preset.hasOrbitState)
				{
					j["orbit_u"] = preset.orbitU;
					j["orbit_v"] = preset.orbitV;
					j["orbit_distance"] = preset.orbitDistance;
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

			json root;
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
				m_presets.emplace_back(std::move(preset));
			}

			return true;
		}

		inline void imguiListen()
		{
			ImGuiIO& io = ImGui::GetIO();
			if (m_ciMode)
			{
				io.IniFilename = nullptr;
				useWindow = true;
			}
			
			ImGuizmo::BeginFrame();
			{
				if (!m_ciMode)
				{
				}

				SImResourceInfo info;
				info.samplerIx = (uint16_t)nbl::ext::imgui::UI::DefaultSamplerIx::USER;

				// ORBIT CAMERA TEST
				{
					for (auto& planar : m_planarProjections)
					{
						auto* camera = planar->getCamera();
						withOrbitLikeCamera(camera, [&](auto* orbit)
						{
							auto targetPostion = hlsl::transpose(getMatrix3x4As4x4(m_model))[3];
							orbit->target(targetPostion);
							orbit->manipulate({}, {});
						});
					}
				}

				// render bound planar camera views onto GUI windows
				if (useWindow)
				{
					if(enableActiveCameraMovement)
						ImGuizmo::Enable(false);
					else
						ImGuizmo::Enable(true);

					size_t gizmoIx = {};
					size_t manipulationCounter = {};
					const std::optional<uint32_t> modelInUseIx = ImGuizmo::IsUsingAny() ? std::optional<uint32_t>(boundPlanarCameraIxToManipulate.has_value() ? 1u + boundPlanarCameraIxToManipulate.value() : 0u) : std::optional<uint32_t>(std::nullopt);

					for (uint32_t windowIx = 0; windowIx < windowBindings.size(); ++windowIx)
					{
						// setup
						{
							const auto& rw = wInit.renderWindows[windowIx];
							const ImGuiCond windowCond = m_ciMode ? ImGuiCond_Always : ImGuiCond_Appearing;
							ImGui::SetNextWindowPos({ rw.iPos.x, rw.iPos.y }, windowCond);
							ImGui::SetNextWindowSize({ rw.iSize.x, rw.iSize.y }, windowCond);
						}
						ImGui::SetNextWindowSizeConstraints(ImVec2(0x45, 0x45), ImVec2(7680, 4320));

						ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
						ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
						ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
						ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
						const std::string ident = "Render Window \"" + std::to_string(windowIx) + "\"";

						ImGui::Begin(ident.data(), 0, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);
						const ImVec2 contentRegionSize = ImGui::GetContentRegionAvail(), windowPos = ImGui::GetWindowPos(), cursorPos = ImGui::GetCursorScreenPos();

						ImGuiWindow* window = ImGui::GetCurrentWindow();
						{
							const auto mPos = ImGui::GetMousePos();

							if (mPos.x < cursorPos.x || mPos.y < cursorPos.y || mPos.x > cursorPos.x + contentRegionSize.x || mPos.y > cursorPos.y + contentRegionSize.y)
								window->Flags &= ~ImGuiWindowFlags_NoMove;
							else
								window->Flags |= ImGuiWindowFlags_NoMove;
						}

						// setup bound entities for the window like camera & projections
						auto& binding = windowBindings[windowIx];
						auto& planarBound = m_planarProjections[binding.activePlanarIx];
						assert(planarBound);

						binding.aspectRatio = contentRegionSize.x / contentRegionSize.y;
						auto* planarViewCameraBound = planarBound->getCamera();

						assert(planarViewCameraBound);
						assert(binding.boundProjectionIx.has_value());
						
						auto& projection = planarBound->getPlanarProjections()[binding.boundProjectionIx.value()];
						projection.update(binding.leftHandedProjection, binding.aspectRatio);

						// TODO: 
						// would be nice to normalize imguizmo visual vectors (possible with styles)

						// first 0th texture is for UI texture atlas, then there are our window textures
						auto fboImguiTextureID = windowIx + 1u;
						info.textureID = fboImguiTextureID;

						if(binding.allowGizmoAxesToFlip)
							ImGuizmo::AllowAxisFlip(true);
						else
							ImGuizmo::AllowAxisFlip(false);

						if(projection.getParameters().m_type == IPlanarProjection::CProjection::Orthographic)
							ImGuizmo::SetOrthographic(true);
						else
							ImGuizmo::SetOrthographic(false);

						ImGuizmo::SetDrawlist();
						ImGui::Image(info, contentRegionSize);
						ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);
						{
							const char* projLabel = projection.getParameters().m_type == IPlanarProjection::CProjection::Perspective ? "Persp" : "Ortho";
							const std::string overlayText = "Planar " + std::to_string(binding.activePlanarIx) + " | " + projLabel + " | W" + std::to_string(windowIx);
							const std::string cameraText = std::string(getCameraTypeLabel(planarViewCameraBound)) + ": " + std::string(getCameraTypeDescription(planarViewCameraBound));
							const ImVec2 textSize = ImGui::CalcTextSize(overlayText.c_str());
							const ImVec2 descSize = ImGui::CalcTextSize(cameraText.c_str());
							const ImVec2 pad = ImVec2(6.0f, 4.0f);
							const float lineGap = 2.0f;
							const float width = std::max(textSize.x, descSize.x);
							const float height = textSize.y + descSize.y + lineGap + pad.y * 2.0f;
							ImVec2 overlayPos = ImVec2(cursorPos.x + contentRegionSize.x - width - pad.x * 2.0f - 6.0f, cursorPos.y + 6.0f);
							overlayPos.x = std::max(overlayPos.x, cursorPos.x + 6.0f);
							ImVec2 overlayMax = ImVec2(overlayPos.x + width + pad.x * 2.0f, overlayPos.y + height);
							auto* drawList = ImGui::GetWindowDrawList();
							drawList->AddRectFilled(overlayPos, overlayMax, ImGui::ColorConvertFloat4ToU32(ImVec4(0.05f, 0.06f, 0.08f, 0.80f)), 6.0f);
							drawList->AddRect(overlayPos, overlayMax, ImGui::ColorConvertFloat4ToU32(ImVec4(0.60f, 0.66f, 0.76f, 0.80f)), 6.0f);
							drawList->AddText(ImVec2(overlayPos.x + pad.x, overlayPos.y + pad.y), ImGui::ColorConvertFloat4ToU32(ImVec4(0.96f, 0.98f, 1.0f, 1.0f)), overlayText.c_str());
							drawList->AddText(ImVec2(overlayPos.x + pad.x, overlayPos.y + pad.y + textSize.y + lineGap), ImGui::ColorConvertFloat4ToU32(ImVec4(0.78f, 0.82f, 0.90f, 1.0f)), cameraText.c_str());
						}

						// I will assume we need to focus a window to start manipulating objects from it
						if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows))
						{
							if (!(m_scriptedInput.enabled && m_scriptedInput.exclusive))
								activeRenderWindowIx = windowIx;
						}

						// we render a scene from view of a camera bound to planar window
						ImGuizmoPlanarM16InOut imguizmoPlanar;
						imguizmoPlanar.view = getCastedMatrix<float32_t>(hlsl::transpose(getMatrix3x4As4x4(planarViewCameraBound->getGimbal().getViewMatrix())));
						imguizmoPlanar.projection = getCastedMatrix<float32_t>(hlsl::transpose(projection.getProjectionMatrix()));

						if (flipGizmoY) // note we allow to flip gizmo just to match our coordinates
							imguizmoPlanar.projection[1][1] *= -1.f; // https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/	

						static constexpr float identityMatrix[] =
						{
							1.f, 0.f, 0.f, 0.f,
							0.f, 1.f, 0.f, 0.f,
							0.f, 0.f, 1.f, 0.f,
							0.f, 0.f, 0.f, 1.f 
						};

						if(binding.enableDebugGridDraw)
							ImGuizmo::DrawGrid(&imguizmoPlanar.view[0][0], &imguizmoPlanar.projection[0][0], identityMatrix, 100.f);

						for (uint32_t modelIx = 0; modelIx < 1u + m_planarProjections.size(); modelIx++)
						{
							ImGuizmo::PushID(gizmoIx); ++gizmoIx;

							const bool isCameraGimbalTarget = modelIx; // I assume scene demo model is 0th ix, left are planar cameras
							ICamera* const targetGimbalManipulationCamera = isCameraGimbalTarget ? m_planarProjections[modelIx - 1u]->getCamera() : nullptr;

							// if we try to manipulate a camera which appears to be the same camera we see scene from then obvsly it doesn't make sense to manipulate its gizmo so we skip it
							// EDIT: it actually makes some sense if you assume render planar view is rendered with ortho projection, but we would need to add imguizmo controller virtual map
							// to ban forward/backward in this mode if this condition is true
							if (targetGimbalManipulationCamera == planarViewCameraBound)
							{
								ImGuizmo::PopID();
								continue;
							}

							ImGuizmoModelM16InOut imguizmoModel;

							if (isCameraGimbalTarget)
							{
								assert(targetGimbalManipulationCamera);
								imguizmoModel.inTRS = getCastedMatrix<float32_t>(targetGimbalManipulationCamera->getGimbal().template operator() < float64_t4x4 > ());
							}
							else
								imguizmoModel.inTRS = hlsl::transpose(getMatrix3x4As4x4(m_model));

							float gizmoSizeClip = 0.1f;
							ImGuizmo::SetGizmoSizeClipSpace(gizmoSizeClip);

							imguizmoModel.outTRS = imguizmoModel.inTRS;
							{
								const bool success = ImGuizmo::Manipulate(&imguizmoPlanar.view[0][0], &imguizmoPlanar.projection[0][0], ImGuizmo::OPERATION::UNIVERSAL, mCurrentGizmoMode, &imguizmoModel.outTRS[0][0], &imguizmoModel.outDeltaTRS[0][0], useSnap ? &snap[0] : nullptr);

								if (success)
								{
									if (targetGimbalManipulationCamera)
									{
										const auto referenceFrame = getCastedMatrix<float64_t>(*reinterpret_cast<float32_t4x4*>(ImGuizmo::GetReferenceFrame()));

										boundCameraToManipulate = smart_refctd_ptr<ICamera>(targetGimbalManipulationCamera);
										boundPlanarCameraIxToManipulate = modelIx - 1u;

										// TODO: TO BE REMOVED, ONLY FOR TESTING ITS INCOMPLETE TYPE!
										const auto& imguizmoCtx = ImGuizmo::GetContext();

										struct
										{
											float32_t3 t, r, s;
										} out, delta;

										ImGuizmo::DecomposeMatrixToComponents(&imguizmoModel.outTRS[0][0], &out.t[0], &out.r[0], &out.s[0]);
										ImGuizmo::DecomposeMatrixToComponents(&imguizmoModel.outDeltaTRS[0][0], &delta.t[0], &delta.r[0], &delta.s[0]);
										{
											std::vector<CVirtualGimbalEvent> virtualEvents;
	
											auto requestMagnitudeUpdateWithScalar = [&](float signPivot, float dScalar, float dMagnitude, auto positive, auto negative)
											{
												if (dScalar != signPivot)
												{
													auto& ev = virtualEvents.emplace_back();
													auto code = (dScalar > signPivot) ? positive : negative;

													ev.type = code;
													ev.magnitude += dMagnitude;
												}
											};
		
											// TODO TESTING STUFF WITH MY IMGUIZMO UPDATES
											// IT WILL BE REMOVED ONCE ALL TESTS ARE DONE 
											// AND CONTROLLER API WILL BE USED INSTEAD

											// translations
											{
												ImGuizmo::OPERATION ioType;
												const auto dScalar = ImGuizmo::GetTranslationDeltaScalar(&ioType);

												if (dScalar)
												{
													switch (ioType)
													{
													case ImGuizmo::OPERATION::TRANSLATE_X:
													{
														requestMagnitudeUpdateWithScalar(0.f, dScalar, std::abs(dScalar), CVirtualGimbalEvent::VirtualEventType::MoveRight, CVirtualGimbalEvent::VirtualEventType::MoveLeft);
													} break;

													case ImGuizmo::OPERATION::TRANSLATE_Y:
													{
														requestMagnitudeUpdateWithScalar(0.f, dScalar, std::abs(dScalar), CVirtualGimbalEvent::VirtualEventType::MoveUp, CVirtualGimbalEvent::VirtualEventType::MoveDown);
													} break;

													case ImGuizmo::OPERATION::TRANSLATE_Z:
													{
														requestMagnitudeUpdateWithScalar(0.f, dScalar, std::abs(dScalar), CVirtualGimbalEvent::VirtualEventType::MoveForward, CVirtualGimbalEvent::VirtualEventType::MoveBackward);
													} break;

													default: break;
													}
												}
											}

											// TODO: ok becuase I have only one reference from imguizmo I must do it differently when 
											// I have local base && want to do rotation with respect to world instead; we almost there
												
											// rotations
											{
												ImGuizmo::OPERATION ioType;
												float dRadians = ImGuizmo::GetRotationDeltaRadians(&ioType);

												if (dRadians)
												{
													switch (ioType)
													{
													case ImGuizmo::OPERATION::ROTATE_X:
													{
														requestMagnitudeUpdateWithScalar(0.f, dRadians, std::abs(dRadians), CVirtualGimbalEvent::VirtualEventType::TiltUp, CVirtualGimbalEvent::VirtualEventType::TiltDown);
													} break;

													case ImGuizmo::OPERATION::ROTATE_Y:
													{
														requestMagnitudeUpdateWithScalar(0.f, dRadians, std::abs(dRadians), CVirtualGimbalEvent::VirtualEventType::PanRight, CVirtualGimbalEvent::VirtualEventType::PanLeft);
													} break;

													case ImGuizmo::OPERATION::ROTATE_Z:
													{
														requestMagnitudeUpdateWithScalar(0.f, dRadians, std::abs(dRadians), CVirtualGimbalEvent::VirtualEventType::RollRight, CVirtualGimbalEvent::VirtualEventType::RollLeft);
													} break;

													default:
														assert(false); break; // should never be hit
													}
												}
											}

											const auto vCount = virtualEvents.size();

											if (vCount)
											{
												const float pMoveSpeed = targetGimbalManipulationCamera->getMoveSpeedScale();
												const float pRotationSpeed = targetGimbalManipulationCamera->getRotationSpeedScale();

												// I start to think controller should be able to set sensitivity to scale magnitudes of generated events
												// in order for camera to not keep any magnitude scalars like move or rotation speed scales

												targetGimbalManipulationCamera->setMoveSpeedScale(1);
												targetGimbalManipulationCamera->setRotationSpeedScale(1);

												targetGimbalManipulationCamera->manipulate({ virtualEvents.data(), vCount }, &referenceFrame);

												targetGimbalManipulationCamera->setMoveSpeedScale(pMoveSpeed);
												targetGimbalManipulationCamera->setRotationSpeedScale(pRotationSpeed);
											}

										}
									}
									else
									{
										// again, for scene demo model full affine transformation without limits is assumed 
										m_model = float32_t3x4(hlsl::transpose(imguizmoModel.outTRS));
										boundCameraToManipulate = nullptr;
										boundPlanarCameraIxToManipulate = std::nullopt;
									}
								}

								if (ImGuizmo::IsOver() and not ImGuizmo::IsUsingAny() && not enableActiveCameraMovement)
								{
									if (targetGimbalManipulationCamera && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
									{
										const uint32_t newPlanarIx = modelIx - 1u;
										if (newPlanarIx < m_planarProjections.size())
										{
											binding.activePlanarIx = newPlanarIx;
											binding.pickDefaultProjections(m_planarProjections[binding.activePlanarIx]->getPlanarProjections());
											if (!(m_scriptedInput.enabled && m_scriptedInput.exclusive))
												activeRenderWindowIx = windowIx;
										}
									}

									ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.2f, 0.2f, 0.2f, 0.8f));
									ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
									ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);

									ImGuiIO& io = ImGui::GetIO();
									ImVec2 mousePos = io.MousePos;
									ImGui::SetNextWindowPos(ImVec2(mousePos.x + 10, mousePos.y + 10), ImGuiCond_Always);

									ImGui::Begin("InfoOverlay", nullptr,
										ImGuiWindowFlags_NoDecoration |
										ImGuiWindowFlags_AlwaysAutoResize |
										ImGuiWindowFlags_NoSavedSettings);

									std::string ident;

									if (targetGimbalManipulationCamera)
										ident = targetGimbalManipulationCamera->getIdentifier();
									else
										ident = "Geometry Creator Object";

									ImGui::Text("Identifier: %s", ident.c_str());
									ImGui::Text("Object Ix: %u", modelIx);
									if (targetGimbalManipulationCamera)
									{
										ImGui::Separator();
										ImGui::TextDisabled("RMB: switch view to this camera");
										ImGui::TextDisabled("LMB drag: manipulate gizmo");
										ImGui::TextDisabled("SPACE: toggle move mode");
									}

									ImGui::End();

									ImGui::PopStyleVar();
									ImGui::PopStyleColor(2);
								}
							}
							ImGuizmo::PopID();
						}

						ImGui::End();
						ImGui::PopStyleColor(1);
						ImGui::PopStyleVar(3);
					}
					if (windowBindings.size() > 1u)
					{
						const auto& topRw = wInit.renderWindows[0];
						const float splitY = topRw.iPos.y + topRw.iSize.y;
						const float gap = std::max(0.0f, wInit.renderWindows[1].iPos.y - splitY);
						ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
						ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
						ImGui::Begin("SplitOverlay", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus);
						auto* drawList = ImGui::GetWindowDrawList();
						if (gap >= 2.0f)
							drawList->AddRectFilled(ImVec2(0.0f, splitY), ImVec2(io.DisplaySize.x, splitY + gap), ImGui::ColorConvertFloat4ToU32(ImVec4(0.05f, 0.06f, 0.08f, 0.85f)));
						else
							drawList->AddLine(ImVec2(0.0f, splitY), ImVec2(io.DisplaySize.x, splitY), ImGui::ColorConvertFloat4ToU32(ImVec4(0.80f, 0.84f, 0.92f, 0.75f)), 2.0f);
						ImGui::End();
					}
					assert(manipulationCounter <= 1u);
				}
				// render selected camera view onto full screen
				else
				{
					info.textureID = 1u + activeRenderWindowIx;

					ImGui::SetNextWindowPos(ImVec2(0, 0));
					ImGui::SetNextWindowSize(io.DisplaySize);
					ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0)); // fully transparent fake window
					ImGui::Begin("FullScreenWindow", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs);
					const ImVec2 contentRegionSize = ImGui::GetContentRegionAvail(), windowPos = ImGui::GetWindowPos(), cursorPos = ImGui::GetCursorScreenPos();
					{
						auto& binding = windowBindings[activeRenderWindowIx];
						auto& planarBound = m_planarProjections[binding.activePlanarIx];
						assert(planarBound);

						binding.aspectRatio = contentRegionSize.x / contentRegionSize.y;
						auto* planarViewCameraBound = planarBound->getCamera();

						assert(planarViewCameraBound);
						assert(binding.boundProjectionIx.has_value());

						auto& projection = planarBound->getPlanarProjections()[binding.boundProjectionIx.value()];
						projection.update(binding.leftHandedProjection, binding.aspectRatio);
					}

					ImGui::Image(info, contentRegionSize);
					ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);

					ImGui::End();
					ImGui::PopStyleColor(1);
				}
			}

			DrawControlPanel();
			UpdateBoundCameraMovement();
			UpdateCursorVisibility();

			// update camera matrices for scene rendering
			{
				for (uint32_t i = 0u; i < windowBindings.size(); ++i)
				{
					auto& binding = windowBindings[i];

					auto& planarBound = m_planarProjections[binding.activePlanarIx];
					assert(planarBound);
					auto* boundPlanarCamera = planarBound->getCamera();

					assert(binding.boundProjectionIx.has_value());
					auto& projection = planarBound->getPlanarProjections()[binding.boundProjectionIx.value()];
					projection.update(binding.leftHandedProjection, binding.aspectRatio);

					auto viewMatrix = getCastedMatrix<float32_t>(boundPlanarCamera->getGimbal().getViewMatrix());
					auto viewProjMatrix = mul(getCastedMatrix<float32_t>(projection.getProjectionMatrix()), getMatrix3x4As4x4(viewMatrix));

					binding.viewMatrix = viewMatrix;
					binding.viewProjMatrix = viewProjMatrix;
				}
			}


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

				ImVec2 viewportSize = io.DisplaySize;
				auto* cc = m_window->getCursorControl();
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
			else
			{
				io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
				io.MouseDrawCursor = true;
				io.WantCaptureMouse = true;
			}
		}

		inline void UpdateCursorVisibility()
		{
			ImGuiIO& io = ImGui::GetIO();
			ImVec2 mousePos = ImGui::GetMousePos();
			ImVec2 viewportSize = io.DisplaySize;
			auto* cc = m_window->getCursorControl();

			if (mousePos.x < 0.0f || mousePos.y < 0.0f || mousePos.x > viewportSize.x || mousePos.y > viewportSize.y)
			{
				if (not enableActiveCameraMovement)
					cc->setVisible(true);
			}
			else
			{
				cc->setVisible(false);
			}
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


		inline void DrawControlPanel()
		{
			const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
			const float panelWidth = std::clamp(displaySize.x * 0.19f, 200.0f, displaySize.x * 0.25f);
			const float panelHeight = std::clamp(displaySize.y * 0.34f, 200.0f, displaySize.y * 0.50f);
			const ImVec2 panelSize = { panelWidth, panelHeight };
			const ImVec2 panelPos = { 0.0f, 0.0f };
			ImGui::SetNextWindowPos(panelPos, ImGuiCond_Always);
			ImGui::SetNextWindowSize(panelSize, ImGuiCond_Always);

			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.0f, 4.0f));
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 1.0f));
			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(3.0f, 2.0f));
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 4.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_TabRounding, 3.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 4.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(3.0f, 2.0f));

			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.06f, 0.08f, 0.0f));
			ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.12f, 0.16f, 0.44f));
			ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.64f, 0.72f, 0.84f, 0.55f));
			ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.16f, 0.19f, 0.24f, 0.54f));
			ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.26f, 0.32f, 0.40f, 0.64f));
			ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.30f, 0.36f, 0.45f, 0.70f));
			ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.14f, 0.18f, 0.24f, 0.60f));
			ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.24f, 0.30f, 0.40f, 0.70f));
			ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.28f, 0.36f, 0.46f, 0.78f));
			ImGui::PushStyleColor(ImGuiCol_Tab, ImVec4(0.14f, 0.18f, 0.24f, 0.60f));
			ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4(0.24f, 0.30f, 0.40f, 0.70f));
			ImGui::PushStyleColor(ImGuiCol_TabActive, ImVec4(0.20f, 0.26f, 0.36f, 0.78f));
			ImGui::PushStyleColor(ImGuiCol_TableRowBg, ImVec4(0.12f, 0.14f, 0.18f, 0.50f));
			ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ImVec4(0.16f, 0.18f, 0.22f, 0.50f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.98f, 0.99f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TextDisabled, ImVec4(0.82f, 0.86f, 0.90f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.54f, 0.60f, 0.70f, 0.80f));
			ImGui::PushStyleColor(ImGuiCol_SeparatorHovered, ImVec4(0.68f, 0.76f, 0.88f, 0.90f));
			ImGui::PushStyleColor(ImGuiCol_SeparatorActive, ImVec4(0.82f, 0.90f, 1.0f, 0.96f));

			ImGui::SetNextWindowCollapsed(false, ImGuiCond_Always);
			ImGui::SetNextWindowBgAlpha(0.0f);
			if (m_ciMode)
				ImGui::SetNextWindowFocus();
			ImGui::Begin("Control Panel", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);

			const ImVec4 accent = ImVec4(0.60f, 0.82f, 1.0f, 1.0f);
			const ImVec4 good = ImVec4(0.45f, 0.90f, 0.60f, 1.0f);
			const ImVec4 bad = ImVec4(1.0f, 0.50f, 0.45f, 1.0f);
			const ImVec4 warn = ImVec4(0.95f, 0.80f, 0.45f, 1.0f);
			const ImVec4 muted = ImVec4(0.92f, 0.93f, 0.95f, 1.0f);
			const ImVec4 badgeText = ImVec4(0.10f, 0.11f, 0.13f, 1.0f);
			const ImVec4 keyBg = ImVec4(0.20f, 0.22f, 0.25f, 1.0f);
			const ImVec4 keyFg = ImVec4(0.92f, 0.94f, 0.96f, 1.0f);
			const ImGuiTableFlags tableFlags = ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_PadOuterX;
			const ImVec4 panelBg = ImVec4(0.03f, 0.04f, 0.05f, 0.50f);
			const ImVec4 panelEdge = ImVec4(0.62f, 0.70f, 0.84f, 0.60f);
			const ImVec4 panelStripe = ImVec4(0.28f, 0.56f, 0.90f, 0.70f);
			const ImVec4 panelShadow = ImVec4(0.0f, 0.0f, 0.0f, 0.12f);

			{
				const ImVec2 panelPos = ImGui::GetWindowPos();
				const ImVec2 panelSize = ImGui::GetWindowSize();
				auto* drawList = ImGui::GetWindowDrawList();
				drawList->AddRectFilled(ImVec2(panelPos.x + 2.0f, panelPos.y + 3.0f), ImVec2(panelPos.x + panelSize.x + 4.0f, panelPos.y + panelSize.y + 5.0f), ImGui::ColorConvertFloat4ToU32(panelShadow), 8.0f);
				drawList->AddRectFilled(panelPos, ImVec2(panelPos.x + panelSize.x, panelPos.y + panelSize.y), ImGui::ColorConvertFloat4ToU32(panelBg), 6.0f);
				drawList->AddRect(panelPos, ImVec2(panelPos.x + panelSize.x, panelPos.y + panelSize.y), ImGui::ColorConvertFloat4ToU32(panelEdge), 6.0f);
				drawList->AddRectFilled(panelPos, ImVec2(panelPos.x + 4.0f, panelPos.y + panelSize.y), ImGui::ColorConvertFloat4ToU32(panelStripe), 6.0f);
			}

			auto row = [&](const char* label, auto&& drawValue)
			{
				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::TextUnformatted(label);
				ImGui::TableSetColumnIndex(1);
				drawValue();
			};

			auto metricMax = [&](const std::array<float, UiMetricSamples>& values, float minValue) -> float
			{
				float maxValue = minValue;
				for (const float v : values)
					maxValue = std::max(maxValue, v);
				return maxValue;
			};

			auto miniStat = [&](const char* id, const char* label, const ImVec4& color, const std::array<float, UiMetricSamples>& series, float minValue, auto&& drawValue)
			{
				ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.14f, 0.16f, 0.19f, 0.75f));
				ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
				if (ImGui::BeginChild(id, ImVec2(0, 56), true, ImGuiWindowFlags_NoScrollbar))
				{
					ImGui::TextDisabled("%s", label);
					ImGui::SetWindowFontScale(1.05f);
					drawValue();
					ImGui::SetWindowFontScale(1.0f);
					ImGui::PushStyleColor(ImGuiCol_PlotLines, color);
					const float maxValue = metricMax(series, minValue);
					ImGui::PlotLines("##plot", series.data(), static_cast<int>(UiMetricSamples), static_cast<int>(m_uiMetricIndex), nullptr, 0.0f, maxValue, ImVec2(0, 24));
					ImGui::PopStyleColor();
				}
				ImGui::EndChild();
				ImGui::PopStyleVar();
				ImGui::PopStyleColor();
			};

			auto calcPillWidth = [&](const char* label, const ImVec2& pad)
			{
				return ImGui::CalcTextSize(label).x + pad.x * 2.0f;
			};

			auto drawTogglePill = [&](const char* label, bool& value, const ImVec4& onCol, const ImVec4& offCol, const ImVec2& pad)
			{
				ImGui::PushStyleColor(ImGuiCol_Button, value ? onCol : offCol);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, value ? onCol : offCol);
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, value ? onCol : offCol);
				ImGui::PushStyleColor(ImGuiCol_Text, badgeText);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, pad);
				if (ImGui::Button(label))
					value = !value;
				ImGui::PopStyleVar();
				ImGui::PopStyleColor(4);
			};

			ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
			if (ImGui::BeginChild("PanelHeader", ImVec2(0, 64), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
			{
				ImGui::Dummy(ImVec2(0.0f, 1.0f));
				ImGui::SetWindowFontScale(1.08f);
				ImGui::TextColored(accent, "Control Panel");
				ImGui::SetWindowFontScale(1.0f);
				{
					const ImVec2 badgePad = ImVec2(6.0f, 2.0f);
					const float gap = ImGui::GetStyle().ItemSpacing.x;
					const char* badgeWindow = useWindow ? "WINDOW" : "FULL";
					const char* badgeMove = enableActiveCameraMovement ? "MOVE ON" : "MOVE OFF";
					const char* badgeScript = m_scriptedInput.enabled ? (m_scriptedInput.exclusive ? "SCRIPT EXCL" : "SCRIPT") : "SCRIPT OFF";
					const float badgeRowWidth = calcPillWidth(badgeWindow, badgePad)
						+ gap + calcPillWidth(badgeMove, badgePad)
						+ gap + calcPillWidth(badgeScript, badgePad)
						+ (m_ciMode ? (gap + calcPillWidth("CI", badgePad)) : 0.0f);
					ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - badgeRowWidth) * 0.5f));

					DrawBadge(badgeWindow, accent, badgeText);
					ImGui::SameLine(0.0f, gap);
					DrawBadge(badgeMove, enableActiveCameraMovement ? good : bad, badgeText);
					ImGui::SameLine(0.0f, gap);
					DrawBadge(badgeScript, m_scriptedInput.enabled ? accent : ImVec4(0.35f, 0.36f, 0.38f, 1.0f), badgeText);
					if (m_ciMode)
					{
						ImGui::SameLine(0.0f, gap);
						DrawBadge("CI", warn, badgeText);
					}
				}

				ImGui::Dummy(ImVec2(0.0f, 2.0f));
				{
					const ImVec2 keyPad = ImVec2(4.0f, 1.0f);
					const float gap = ImGui::GetStyle().ItemSpacing.x;
					const float groupGap = gap * 2.0f;
					const float moveWidth = ImGui::CalcTextSize("Move").x + gap
						+ calcPillWidth("W", keyPad) + gap
						+ calcPillWidth("A", keyPad) + gap
						+ calcPillWidth("S", keyPad) + gap
						+ calcPillWidth("D", keyPad);
					const float lookWidth = ImGui::CalcTextSize("Look").x + gap + calcPillWidth("RMB", keyPad);
					const float zoomWidth = ImGui::CalcTextSize("Zoom").x + gap + calcPillWidth("MW", keyPad);
					const float rowWidth = moveWidth + groupGap + lookWidth + groupGap + zoomWidth;
					ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - rowWidth) * 0.5f));

					ImGui::TextDisabled("Move");
					ImGui::SameLine();
					DrawKeyHint("W", keyBg, keyFg);
					ImGui::SameLine();
					DrawKeyHint("A", keyBg, keyFg);
					ImGui::SameLine();
					DrawKeyHint("S", keyBg, keyFg);
					ImGui::SameLine();
					DrawKeyHint("D", keyBg, keyFg);

					ImGui::SameLine(0.0f, groupGap);
					ImGui::TextDisabled("Look");
					ImGui::SameLine();
					DrawKeyHint("RMB", keyBg, keyFg);

					ImGui::SameLine(0.0f, groupGap);
					ImGui::TextDisabled("Zoom");
					ImGui::SameLine();
					DrawKeyHint("MW", keyBg, keyFg);
				}

				ImGui::Dummy(ImVec2(0.0f, 2.0f));
				if (ImGui::BeginTable("HeaderMetrics", 3, ImGuiTableFlags_SizingStretchProp))
				{
					const float frameMs = std::max(0.0f, m_uiLastFrameMs);
					const float fps = frameMs > 0.0f ? (1000.0f / frameMs) : 0.0f;

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					miniStat("FrameStat", "Frame", accent, m_uiFrameMs, 16.0f, [&]
					{
						ImGui::TextColored(accent, "%.1f ms  %.0f fps", frameMs, fps);
					});

					ImGui::TableSetColumnIndex(1);
					miniStat("InputStat", "Input", accent, m_uiInputCounts, 4.0f, [&]
					{
						ImGui::TextColored(accent, "%u ev", m_uiLastInputEvents);
					});

					ImGui::TableSetColumnIndex(2);
					miniStat("VirtualStat", "Virtual", accent, m_uiVirtualCounts, 4.0f, [&]
					{
						ImGui::TextColored(accent, "%u ev", m_uiLastVirtualEvents);
					});
					ImGui::EndTable();
				}
			}
			ImGui::EndChild();
			ImGui::PopStyleVar();

			ImGui::Spacing();

			{
				const ImVec2 togglePad = ImVec2(6.0f, 2.0f);
				const float gap = ImGui::GetStyle().ItemSpacing.x;
				const float rowWidth = calcPillWidth("WINDOW", togglePad)
					+ gap + calcPillWidth("STATUS", togglePad)
					+ gap + calcPillWidth("EVENT LOG", togglePad);
				ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - rowWidth) * 0.5f));
				drawTogglePill("WINDOW", useWindow, accent, ImVec4(0.35f, 0.36f, 0.38f, 1.0f), togglePad);
				DrawHoverHint("Toggle split render windows");
				ImGui::SameLine(0.0f, gap);
				drawTogglePill("STATUS", m_showHud, accent, ImVec4(0.35f, 0.36f, 0.38f, 1.0f), togglePad);
				DrawHoverHint("Show system and camera status panel");
				ImGui::SameLine(0.0f, gap);
				drawTogglePill("EVENT LOG", m_showEventLog, accent, ImVec4(0.35f, 0.36f, 0.38f, 1.0f), togglePad);
				DrawHoverHint("Show virtual event log");
			}

			ImGui::Separator();

			if (ImGui::BeginTabBar("ControlTabs"))
			{
				if (m_showHud && ImGui::BeginTabItem("Status"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("StatusPanel", ImVec2(0, 0), true))
					{
						ImGui::PushItemWidth(-1.0f);
						const ImVec4 cardTop = ImVec4(0.20f, 0.22f, 0.26f, 0.98f);
						const ImVec4 cardBottom = ImVec4(0.12f, 0.13f, 0.15f, 0.98f);
						const ImVec4 cardBorder = ImVec4(0.45f, 0.48f, 0.54f, 1.0f);

						DrawSectionHeader("SessionHeader", "Session", accent);
						if (BeginCard("SessionCard", CalcCardHeight(3), cardTop, cardBottom, cardBorder))
						{
							if (ImGui::BeginTable("SessionTable", 2, tableFlags))
							{
								ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 120.0f);
								ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
								row("Mode", [&] { DrawDot(accent); ImGui::TextColored(accent, "%s", useWindow ? "Window" : "Fullscreen"); });
								row("Active window", [&] { DrawDot(accent); ImGui::TextColored(accent, "%u", activeRenderWindowIx); });
								row("Movement", [&] { const ImVec4 c = enableActiveCameraMovement ? good : bad; DrawDot(c); ImGui::TextColored(c, "%s", enableActiveCameraMovement ? "Enabled" : "Disabled"); });
								ImGui::EndTable();
							}
						}
						EndCard();

						DrawSectionHeader("CameraHeader", "Camera", accent);

						auto* activeCamera = getActiveCamera();
						if (activeCamera)
						{
							const auto& gimbal = activeCamera->getGimbal();
							const auto pos = gimbal.getPosition();
							const auto euler = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));

							if (BeginCard("CameraCard", CalcCardHeight(5), cardTop, cardBottom, cardBorder))
							{
								if (ImGui::BeginTable("CameraTable", 2, tableFlags))
								{
									ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 120.0f);
									ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
									row("Name", [&] { DrawDot(accent); ImGui::TextColored(muted, "%s", activeCamera->getIdentifier().data()); });
									row("Position", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.2f %.2f %.2f", pos.x, pos.y, pos.z); });
									row("Euler", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.1f %.1f %.1f", euler.x, euler.y, euler.z); });
									row("Move scale", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.4f", activeCamera->getMoveSpeedScale()); });
									row("Rotate scale", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.4f", activeCamera->getRotationSpeedScale()); });
									ImGui::EndTable();
								}
							}
							EndCard();
						}
						else
						{
							if (BeginCard("CameraCard", CalcCardHeight(2), cardTop, cardBottom, cardBorder))
								ImGui::TextDisabled("No active camera");
							EndCard();
						}

						DrawSectionHeader("ProjectionHeader", "Projection", accent);

						auto& binding = windowBindings[activeRenderWindowIx];
						auto& planar = m_planarProjections[binding.activePlanarIx];
						if (planar && binding.boundProjectionIx.has_value())
						{
							auto& projection = planar->getPlanarProjections()[binding.boundProjectionIx.value()];
							const auto& params = projection.getParameters();
							if (BeginCard("ProjectionCard", CalcCardHeight(4), cardTop, cardBottom, cardBorder))
							{
								if (ImGui::BeginTable("ProjectionTable", 2, tableFlags))
								{
									ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 120.0f);
									ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
									row("Type", [&] { DrawDot(accent); ImGui::TextColored(muted, "%s", params.m_type == IPlanarProjection::CProjection::Perspective ? "Perspective" : "Orthographic"); });
									row("zNear", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.2f", params.m_zNear); });
									row("zFar", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.2f", params.m_zFar); });
									if (params.m_type == IPlanarProjection::CProjection::Perspective)
										row("Fov", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.1f", params.m_planar.perspective.fov); });
									else
										row("Ortho width", [&] { DrawDot(muted); ImGui::TextColored(muted, "%.1f", params.m_planar.orthographic.orthoWidth); });
									ImGui::EndTable();
								}
							}
							EndCard();
						}
						else
						{
							if (BeginCard("ProjectionCard", CalcCardHeight(2), cardTop, cardBottom, cardBorder))
								ImGui::TextDisabled("No projection bound");
							EndCard();
						}
						ImGui::PopItemWidth();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (ImGui::BeginTabItem("Projection"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("ProjectionPanel", ImVec2(0, 0), true))
					{
						ImGui::PushItemWidth(-1.0f);
						auto& active = windowBindings[activeRenderWindowIx];
						const auto activeRenderWindowIxString = std::to_string(activeRenderWindowIx);

						DrawSectionHeader("PlanarSelectHeader", "Planar Selection", accent);
						ImGui::Text("Active Render Window: %s", activeRenderWindowIxString.c_str());
						DrawHoverHint("Window that receives input and camera switching");
						{
							const size_t planarsCount = m_planarProjections.size();
							assert(planarsCount);

							std::vector<std::string> sbels(planarsCount);
							for (size_t i = 0; i < planarsCount; ++i)
								sbels[i] = "Planar " + std::to_string(i);

							std::vector<const char*> labels(planarsCount);
							for (size_t i = 0; i < planarsCount; ++i)
								labels[i] = sbels[i].c_str();

							int currentPlanarIx = static_cast<int>(active.activePlanarIx);
							if (ImGui::Combo("Active Planar", &currentPlanarIx, labels.data(), static_cast<int>(labels.size())))
							{
								active.activePlanarIx = static_cast<uint32_t>(currentPlanarIx);
								active.pickDefaultProjections(m_planarProjections[active.activePlanarIx]->getPlanarProjections());
							}
							DrawHoverHint("Select which camera the window renders");
						}

						assert(active.boundProjectionIx.has_value());
						assert(active.lastBoundPerspectivePresetProjectionIx.has_value());
						assert(active.lastBoundOrthoPresetProjectionIx.has_value());

						const auto activePlanarIxString = std::to_string(active.activePlanarIx);
						auto& planarBound = m_planarProjections[active.activePlanarIx];
						assert(planarBound);

						DrawSectionHeader("ProjectionParamsHeader", "Projection Parameters", accent);

						auto selectedProjectionType = planarBound->getPlanarProjections()[active.boundProjectionIx.value()].getParameters().m_type;
						{
							const char* labels[] = { "Perspective", "Orthographic" };
							int type = static_cast<int>(selectedProjectionType);

							if (ImGui::Combo("Projection Type", &type, labels, IM_ARRAYSIZE(labels)))
							{
								selectedProjectionType = static_cast<IPlanarProjection::CProjection::ProjectionType>(type);

								switch (selectedProjectionType)
								{
									case IPlanarProjection::CProjection::Perspective: active.boundProjectionIx = active.lastBoundPerspectivePresetProjectionIx.value(); break;
									case IPlanarProjection::CProjection::Orthographic: active.boundProjectionIx = active.lastBoundOrthoPresetProjectionIx.value(); break;
									default: active.boundProjectionIx = std::nullopt; assert(false); break;
								}
							}
							DrawHoverHint("Switch projection type for this planar");
						}

						auto getPresetName = [&](auto ix) -> std::string
						{
							switch (selectedProjectionType)
							{
								case IPlanarProjection::CProjection::Perspective: return "Perspective Projection Preset " + std::to_string(ix);
								case IPlanarProjection::CProjection::Orthographic: return "Orthographic Projection Preset " + std::to_string(ix);
								default: return "Unknown Projection Preset " + std::to_string(ix);
							}
						};

						bool updateBoundVirtualMaps = false;
						if (ImGui::BeginCombo("Projection Preset", getPresetName(active.boundProjectionIx.value()).c_str()))
						{
							auto& projections = planarBound->getPlanarProjections();

							for (uint32_t i = 0; i < projections.size(); ++i)
							{
								const auto& projection = projections[i];
								const auto& params = projection.getParameters();

								if (params.m_type != selectedProjectionType)
									continue;

								bool isSelected = (i == active.boundProjectionIx.value());

								if (ImGui::Selectable(getPresetName(i).c_str(), isSelected))
								{
									active.boundProjectionIx = i;
									updateBoundVirtualMaps |= true;

									switch (selectedProjectionType)
									{
										case IPlanarProjection::CProjection::Perspective: active.lastBoundPerspectivePresetProjectionIx = active.boundProjectionIx.value(); break;
										case IPlanarProjection::CProjection::Orthographic: active.lastBoundOrthoPresetProjectionIx = active.boundProjectionIx.value(); break;
										default: assert(false); break;
									}
								}

								if (isSelected)
									ImGui::SetItemDefaultFocus();
							}
							ImGui::EndCombo();
						}
						DrawHoverHint("Switch preset projection for this planar");

						auto* const boundCamera = planarBound->getCamera();
						auto& boundProjection = planarBound->getPlanarProjections()[active.boundProjectionIx.value()];
						assert(not boundProjection.isProjectionSingular());

						auto updateParameters = boundProjection.getParameters();

						if (useWindow)
							ImGui::Checkbox("Allow axes to flip##allowAxesToFlip", &active.allowGizmoAxesToFlip);
						DrawHoverHint("Allow ImGuizmo axes to flip based on view");

						if(useWindow)
							ImGui::Checkbox("Draw debug grid##drawDebugGrid", &active.enableDebugGridDraw);
						DrawHoverHint("Toggle debug grid in the render window");

						if (ImGui::RadioButton("LH", active.leftHandedProjection))
							active.leftHandedProjection = true;

						ImGui::SameLine();

						if (ImGui::RadioButton("RH", not active.leftHandedProjection))
							active.leftHandedProjection = false;
						DrawHoverHint("Toggle left or right handed projection");

						updateParameters.m_zNear = std::clamp(updateParameters.m_zNear, 0.1f, 100.f);
						updateParameters.m_zFar = std::clamp(updateParameters.m_zFar, 110.f, 10000.f);

						ImGui::SliderFloat("zNear", &updateParameters.m_zNear, 0.1f, 100.f, "%.2f", ImGuiSliderFlags_Logarithmic);
						DrawHoverHint("Near clip plane");
						ImGui::SliderFloat("zFar", &updateParameters.m_zFar, 110.f, 10000.f, "%.1f", ImGuiSliderFlags_Logarithmic);
						DrawHoverHint("Far clip plane");

						switch (selectedProjectionType)
						{
							case IPlanarProjection::CProjection::Perspective:
							{
								ImGui::SliderFloat("Fov", &updateParameters.m_planar.perspective.fov, 20.f, 150.f, "%.1f", ImGuiSliderFlags_Logarithmic);
								DrawHoverHint("Perspective field of view");
								boundProjection.setPerspective(updateParameters.m_zNear, updateParameters.m_zFar, updateParameters.m_planar.perspective.fov);
							} break;

							case IPlanarProjection::CProjection::Orthographic:
							{
								ImGui::SliderFloat("Ortho width", &updateParameters.m_planar.orthographic.orthoWidth, 1.f, 30.f, "%.1f", ImGuiSliderFlags_Logarithmic);
								DrawHoverHint("Orthographic width");
								boundProjection.setOrthographic(updateParameters.m_zNear, updateParameters.m_zFar, updateParameters.m_planar.orthographic.orthoWidth);
							} break;

							default: break;
						}

						DrawSectionHeader("CursorHeader", "Cursor Behaviour", accent);
						if (ImGui::TreeNodeEx("Cursor Behaviour"))
						{
							if (ImGui::RadioButton("Clamp to the window", !resetCursorToCenter))
								resetCursorToCenter = false;
							if (ImGui::RadioButton("Reset to the window center", resetCursorToCenter))
								resetCursorToCenter = true;
							ImGui::TreePop();
						}

						if (enableActiveCameraMovement)
							ImGui::TextColored(good, "Bound Camera Movement: Enabled");
						else
							ImGui::TextColored(bad, "Bound Camera Movement: Disabled");

						ImGui::Separator();

						DrawSectionHeader("BoundCameraHeader", "Bound Camera", accent);
						const auto flags = ImGuiTreeNodeFlags_DefaultOpen;
						if (ImGui::TreeNodeEx("Bound Camera", flags))
						{
							ImGui::Text("Type: %s", boundCamera->getIdentifier().data());
							ImGui::Text("Object Ix: %s", std::to_string(active.activePlanarIx + 1u).c_str());
							ImGui::Separator();
							{
								auto* orbit = dynamic_cast<COrbitCamera*>(boundCamera);
								auto* arcball = dynamic_cast<CArcballCamera*>(boundCamera);
								auto* turntable = dynamic_cast<CTurntableCamera*>(boundCamera);
								const bool isOrbitLike = orbit || arcball || turntable;

								float moveSpeed = boundCamera->getMoveSpeedScale();
								float rotationSpeed = boundCamera->getRotationSpeedScale();

								ImGui::SliderFloat("Move speed factor", &moveSpeed, 0.0001f, 10.f, "%.4f", ImGuiSliderFlags_Logarithmic);
								DrawHoverHint("Scale translation speed for this camera");

								if (not orbit)
									ImGui::SliderFloat("Rotate speed factor", &rotationSpeed, 0.0001f, 10.f, "%.4f", ImGuiSliderFlags_Logarithmic);
								DrawHoverHint("Scale rotation speed for this camera");

								boundCamera->setMoveSpeedScale(moveSpeed);
								boundCamera->setRotationSpeedScale(rotationSpeed);

								if (isOrbitLike)
								{
									auto applyDistance = [&](auto* cam)
									{
										float distance = cam->getDistance();
										ImGui::SliderFloat("Distance", &distance, cam->MinDistance, cam->MaxDistance, "%.4f", ImGuiSliderFlags_Logarithmic);
										DrawHoverHint("Current orbit distance");
										cam->setDistance(distance);
									};

									if (orbit)
										applyDistance(orbit);
									else if (arcball)
										applyDistance(arcball);
									else if (turntable)
										applyDistance(turntable);
								}
							}

							if (ImGui::TreeNodeEx("World Data", flags))
							{
								auto& gimbal = boundCamera->getGimbal();
								const auto position = getCastedVector<float32_t>(gimbal.getPosition());
								const auto& orientation = gimbal.getOrientation();
								const auto viewMatrix = getCastedMatrix<float32_t>(gimbal.getViewMatrix());

								addMatrixTable("Position", ("PositionTable_" + activePlanarIxString).c_str(), 1, 3, &position[0], false);
								addMatrixTable("Orientation (Quaternion)", ("OrientationTable_" + activePlanarIxString).c_str(), 1, 4, &orientation[0], false);
								addMatrixTable("View Matrix", ("ViewMatrixTable_" + activePlanarIxString).c_str(), 3, 4, &viewMatrix[0][0], false);
								ImGui::TreePop();
							}

							if (ImGui::TreeNodeEx("Virtual Event Mappings", flags))
							{
								displayKeyMappingsAndVirtualStatesInline(&boundProjection);
								ImGui::TreePop();
							}

							ImGui::TreePop();
						}
						ImGui::PopItemWidth();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (ImGui::BeginTabItem("Camera"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("CameraPanel", ImVec2(0, 0), true))
					{
						ImGui::PushItemWidth(-1.0f);
						DrawSectionHeader("CameraInputHeader", "Input", accent);
						ImGui::Checkbox("Mirror input to all cameras", &m_cameraControls.mirrorInput);
						DrawHoverHint("Apply keyboard and mouse input to every camera");
						ImGui::Checkbox("World translate", &m_cameraControls.worldTranslate);
						DrawHoverHint("Translate in world space instead of camera space");
						ImGui::SliderFloat("Keyboard scale", &m_cameraControls.keyboardScale, 0.01f, 10.f, "%.2f");
						DrawHoverHint("Scale keyboard movement magnitudes");
						ImGui::SliderFloat("Mouse move scale", &m_cameraControls.mouseMoveScale, 0.01f, 10.f, "%.2f");
						DrawHoverHint("Scale mouse move magnitudes");
						ImGui::SliderFloat("Mouse scroll scale", &m_cameraControls.mouseScrollScale, 0.01f, 10.f, "%.2f");
						DrawHoverHint("Scale mouse wheel magnitudes");
						ImGui::SliderFloat("Translate scale", &m_cameraControls.translationScale, 0.01f, 10.f, "%.2f");
						DrawHoverHint("Overall translation scale for virtual events");
						ImGui::SliderFloat("Rotate scale", &m_cameraControls.rotationScale, 0.01f, 10.f, "%.2f");
						DrawHoverHint("Overall rotation scale for virtual events");

						DrawSectionHeader("CameraConstraintsHeader", "Constraints", accent);
						ImGui::Checkbox("Enable constraints", &m_cameraConstraints.enabled);
						DrawHoverHint("Enable or disable all camera constraints");
						ImGui::Checkbox("Clamp distance", &m_cameraConstraints.clampDistance);
						DrawHoverHint("Clamp orbit distance to min/max");
						ImGui::SliderFloat("Min distance", &m_cameraConstraints.minDistance, 0.01f, 1000.f, "%.3f", ImGuiSliderFlags_Logarithmic);
						DrawHoverHint("Minimum orbit distance");
						ImGui::SliderFloat("Max distance", &m_cameraConstraints.maxDistance, 0.01f, 10000.f, "%.3f", ImGuiSliderFlags_Logarithmic);
						DrawHoverHint("Maximum orbit distance");
						ImGui::Separator();
						ImGui::Checkbox("Clamp pitch", &m_cameraConstraints.clampPitch);
						DrawHoverHint("Clamp pitch angle");
						ImGui::SliderFloat("Pitch min", &m_cameraConstraints.pitchMinDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Minimum pitch in degrees");
						ImGui::SliderFloat("Pitch max", &m_cameraConstraints.pitchMaxDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Maximum pitch in degrees");
						ImGui::Checkbox("Clamp yaw", &m_cameraConstraints.clampYaw);
						DrawHoverHint("Clamp yaw angle");
						ImGui::SliderFloat("Yaw min", &m_cameraConstraints.yawMinDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Minimum yaw in degrees");
						ImGui::SliderFloat("Yaw max", &m_cameraConstraints.yawMaxDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Maximum yaw in degrees");
						ImGui::Checkbox("Clamp roll", &m_cameraConstraints.clampRoll);
						DrawHoverHint("Clamp roll angle");
						ImGui::SliderFloat("Roll min", &m_cameraConstraints.rollMinDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Minimum roll in degrees");
						ImGui::SliderFloat("Roll max", &m_cameraConstraints.rollMaxDeg, -180.f, 180.f, "%.1f");
						DrawHoverHint("Maximum roll in degrees");

						DrawSectionHeader("OrbitHeader", "Orbit Target", accent);

						auto* activeCamera = getActiveCamera();
						const bool hasOrbitTarget = withOrbitLikeCamera(activeCamera, [&](auto* orbit)
						{
							auto target = getCastedVector<float32_t>(orbit->getTarget());
							if (ImGui::InputFloat3("Target", &target[0]))
								orbit->target(getCastedVector<float64_t>(target));

							if (ImGui::Button("Target model"))
							{
								auto targetPos = hlsl::transpose(getMatrix3x4As4x4(m_model))[3];
								orbit->target(targetPos);
							}
							DrawHoverHint("Set orbit target to the model position");
							ImGui::SameLine();
							if (ImGui::Button("Target origin"))
								orbit->target(float64_t3(0.0));
							DrawHoverHint("Set orbit target to world origin");
						});
						if (!hasOrbitTarget)
						{
							ImGui::TextDisabled("Active camera is not orbit.");
						}
						ImGui::PopItemWidth();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (ImGui::BeginTabItem("Presets"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("PresetsPanel", ImVec2(0, 0), true))
					{
						ImGui::PushItemWidth(-1.0f);
						DrawSectionHeader("PresetsHeader", "Presets", accent);
						ImGui::InputText("Preset name", m_presetName, IM_ARRAYSIZE(m_presetName));
						if (ImGui::Button("Add preset"))
						{
							auto* activeCamera = getActiveCamera();
							m_presets.emplace_back(capturePreset(activeCamera, m_presetName));
						}
						DrawHoverHint("Store current camera as a preset");
						ImGui::SameLine();
						if (ImGui::Button("Clear presets"))
							m_presets.clear();
						DrawHoverHint("Remove all presets");

						if (!m_presets.empty())
						{
							std::vector<const char*> names;
							names.reserve(m_presets.size());
							for (const auto& preset : m_presets)
								names.push_back(preset.name.c_str());

							static int selectedPreset = -1;
							ImGui::ListBox("Preset list", &selectedPreset, names.data(), static_cast<int>(names.size()), 6);

							if (selectedPreset >= 0 && static_cast<size_t>(selectedPreset) < m_presets.size())
							{
								if (ImGui::Button("Apply preset"))
									applyPresetToCamera(getActiveCamera(), m_presets[static_cast<size_t>(selectedPreset)]);
								DrawHoverHint("Apply selected preset to the active camera");
								ImGui::SameLine();
								if (ImGui::Button("Remove preset"))
									m_presets.erase(m_presets.begin() + selectedPreset);
								DrawHoverHint("Remove selected preset");
							}
						}

						DrawSectionHeader("PresetsStorageHeader", "Storage", accent);
						ImGui::InputText("Preset file", m_presetPath, IM_ARRAYSIZE(m_presetPath));
						if (ImGui::Button("Save presets"))
						{
							if (!savePresetsToFile(system::path(m_presetPath)))
								m_logger->log("Failed to save presets to \"%s\".", ILogger::ELL_ERROR, m_presetPath);
						}
						DrawHoverHint("Save presets to JSON file");
						ImGui::SameLine();
						if (ImGui::Button("Load presets"))
						{
							if (!loadPresetsFromFile(system::path(m_presetPath)))
								m_logger->log("Failed to load presets from \"%s\".", ILogger::ELL_ERROR, m_presetPath);
						}
						DrawHoverHint("Load presets from JSON file");
						ImGui::PopItemWidth();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (ImGui::BeginTabItem("Playback"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("PlaybackPanel", ImVec2(0, 0), true))
					{
						ImGui::PushItemWidth(-1.0f);
						DrawSectionHeader("PlaybackHeader", "Playback", accent);
						ImGui::Checkbox("Loop", &m_playback.loop);
						DrawHoverHint("Loop playback when it reaches the end");
						ImGui::Checkbox("Override input", &m_playback.overrideInput);
						DrawHoverHint("Ignore manual input during playback");
						ImGui::Checkbox("Affect all cameras", &m_playbackAffectsAll);
						DrawHoverHint("Apply playback to all cameras");
						ImGui::SliderFloat("Speed", &m_playback.speed, 0.1f, 4.f, "%.2f");
						DrawHoverHint("Playback speed multiplier");

						if (ImGui::Button(m_playback.playing ? "Pause" : "Play"))
							m_playback.playing = !m_playback.playing;
						DrawHoverHint("Start or pause playback");
						ImGui::SameLine();
						if (ImGui::Button("Stop"))
						{
							m_playback.playing = false;
							m_playback.time = 0.f;
						}
						DrawHoverHint("Stop playback and reset time");

						if (!m_keyframes.empty())
						{
							const float duration = m_keyframes.back().time;
							ImGui::SliderFloat("Time", &m_playback.time, 0.f, duration, "%.3f");
						}

						DrawSectionHeader("KeyframesHeader", "Keyframes", accent);
						ImGui::InputFloat("New keyframe time", &m_newKeyframeTime, 0.1f, 1.f, "%.3f");
						DrawHoverHint("Time value for new keyframe");
						if (ImGui::Button("Add keyframe"))
						{
							auto* activeCamera = getActiveCamera();
							CameraKeyframe keyframe;
							keyframe.time = m_newKeyframeTime;
							keyframe.preset = capturePreset(activeCamera, "Keyframe");
							m_keyframes.emplace_back(std::move(keyframe));
							std::sort(m_keyframes.begin(), m_keyframes.end(), [](const auto& a, const auto& b) { return a.time < b.time; });
						}
						DrawHoverHint("Add keyframe from current camera");
						ImGui::SameLine();
						if (ImGui::Button("Clear keyframes"))
							m_keyframes.clear();
						DrawHoverHint("Remove all keyframes");

						if (!m_keyframes.empty())
						{
							if (ImGui::BeginChild("KeyframeList", ImVec2(0, 120), true))
							{
								for (size_t i = 0; i < m_keyframes.size(); ++i)
								{
									ImGui::Text("[%zu] t=%.3f", i, m_keyframes[i].time);
								}
							}
							ImGui::EndChild();
						}
						ImGui::PopItemWidth();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (ImGui::BeginTabItem("Gizmo"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("GizmoPanel", ImVec2(0, 0), true))
					{
						DrawSectionHeader("GizmoHeader", "Gizmo", accent);
						TransformEditorContents();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				if (m_showEventLog && ImGui::BeginTabItem("Log"))
				{
					ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
					if (ImGui::BeginChild("LogPanel", ImVec2(0, 0), true))
					{
						DrawSectionHeader("LogHeader", "Virtual Events", accent);
						ImGui::Checkbox("Auto-scroll", &m_logAutoScroll);
						ImGui::SameLine();
						ImGui::Checkbox("Wrap", &m_logWrap);
						ImGui::Separator();

						ImGuiWindowFlags logFlags = m_logWrap ? ImGuiWindowFlags_None : ImGuiWindowFlags_HorizontalScrollbar;
						if (ImGui::BeginChild("LogList", ImVec2(0, 0), false, logFlags))
						{
							const float scrollY = ImGui::GetScrollY();
							const float scrollMax = ImGui::GetScrollMaxY();
							const bool wasAtBottom = scrollY >= scrollMax - 5.0f;
							const size_t start = m_virtualEventLog.size() > 200 ? m_virtualEventLog.size() - 200 : 0;
							if (m_logWrap)
								ImGui::PushTextWrapPos(0.0f);
							for (size_t i = start; i < m_virtualEventLog.size(); ++i)
							{
								const auto& entry = m_virtualEventLog[i];
								ImGui::TextUnformatted(entry.line.c_str());
							}
							if (m_logWrap)
								ImGui::PopTextWrapPos();
							if (m_logAutoScroll && wasAtBottom && !m_virtualEventLog.empty())
								ImGui::SetScrollHereY(1.0f);
						}
						ImGui::EndChild();
					}
					ImGui::EndChild();
					ImGui::PopStyleVar();
					ImGui::EndTabItem();
				}

				ImGui::EndTabBar();
			}

			ImGui::End();
			ImGui::PopStyleColor(19);
			ImGui::PopStyleVar(9);
		}

		inline void TransformEditorContents()
		{
			static float bounds[] = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
			static float boundsSnap[] = { 0.1f, 0.1f, 0.1f };
			static bool boundSizing = false;
			static bool boundSizingSnap = false;

			const size_t objectsCount = m_planarProjections.size() + 1u;
			assert(objectsCount);

			std::vector<std::string> sbels(objectsCount);
			for (size_t i = 0; i < objectsCount; ++i)
				sbels[i] = "Object " + std::to_string(i);

			std::vector<const char*> labels(objectsCount);
			for (size_t i = 0; i < objectsCount; ++i)
				labels[i] = sbels[i].c_str();

			int activeObject = boundCameraToManipulate ? static_cast<int>(boundPlanarCameraIxToManipulate.value() + 1u) : 0;
			if (ImGui::Combo("Active Object", &activeObject, labels.data(), static_cast<int>(labels.size())))
			{
				const auto newActiveObject = static_cast<uint32_t>(activeObject);

				if (newActiveObject) // camera
				{
					boundPlanarCameraIxToManipulate = newActiveObject - 1u;
					ICamera* const targetGimbalManipulationCamera = m_planarProjections[boundPlanarCameraIxToManipulate.value()]->getCamera();
					boundCameraToManipulate = smart_refctd_ptr<ICamera>(targetGimbalManipulationCamera);
				}
				else // gc model
				{
					boundPlanarCameraIxToManipulate = std::nullopt;
					boundCameraToManipulate = nullptr;
				}
			}

			ImGuizmoModelM16InOut imguizmoModel;

			if (boundCameraToManipulate)
				imguizmoModel.inTRS = getCastedMatrix<float32_t>(boundCameraToManipulate->getGimbal().template operator() < float64_t4x4 > ());
			else
				imguizmoModel.inTRS = hlsl::transpose(getMatrix3x4As4x4(m_model));

			imguizmoModel.outTRS = imguizmoModel.inTRS;
			float* m16TRSmatrix = &imguizmoModel.outTRS[0][0];

			std::string indent; 
			if (boundCameraToManipulate)
				indent = boundCameraToManipulate->getIdentifier();
			else
				indent = "Geometry Creator Object";

			ImGui::Text("Identifier: \"%s\"", indent.c_str());
			{
				if (ImGuizmo::IsUsingAny())
					ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Gizmo: In Use");
				else
					ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Gizmo: Idle");

				if (ImGui::IsItemHovered())
				{
					ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.2f, 0.2f, 0.2f, 0.8f));
					ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
					ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);

					ImVec2 mousePos = ImGui::GetMousePos();
					ImGui::SetNextWindowPos(ImVec2(mousePos.x + 10, mousePos.y + 10), ImGuiCond_Always);

					ImGui::Begin("HoverOverlay", nullptr,
						ImGuiWindowFlags_NoDecoration |
						ImGuiWindowFlags_AlwaysAutoResize |
						ImGuiWindowFlags_NoSavedSettings);

					ImGui::Text("Right-click and drag on the gizmo to manipulate the object.");

					ImGui::End();

					ImGui::PopStyleVar();
					ImGui::PopStyleColor(2);
				}
			}

			ImGui::Separator();

			if (!boundCameraToManipulate)
			{
				const auto& names = m_scene->getInitParams().geometryNames;
				if (!names.empty())
				{
					if (gcIndex >= names.size())
						gcIndex = 0;

					if (ImGui::BeginCombo("Object Type", names[gcIndex].c_str()))
					{
						for (uint32_t i = 0u; i < names.size(); ++i)
						{
							const bool isSelected = (gcIndex == i);
							if (ImGui::Selectable(names[i].c_str(), isSelected))
								gcIndex = static_cast<uint16_t>(i);

							if (isSelected)
								ImGui::SetItemDefaultFocus();
						}
						ImGui::EndCombo();
					}
				}

			}

			addMatrixTable("Model (TRS) Matrix", "ModelMatrixTable", 4, 4, m16TRSmatrix);

			if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
				mCurrentGizmoOperation = ImGuizmo::TRANSLATE;

			ImGui::SameLine();
			if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
				mCurrentGizmoOperation = ImGuizmo::ROTATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
				mCurrentGizmoOperation = ImGuizmo::SCALE;

			float32_t3 matrixTranslation, matrixRotation, matrixScale;
			IGimbalController::input_imguizmo_event_t decomposed, recomposed;
			imguizmoModel.outDeltaTRS = IGimbalController::input_imguizmo_event_t(1);

			ImGuizmo::DecomposeMatrixToComponents(m16TRSmatrix, &matrixTranslation[0], &matrixRotation[0], &matrixScale[0]);
			decomposed = *reinterpret_cast<float32_t4x4*>(m16TRSmatrix);
			{
				ImGuiInputTextFlags flags = 0;

				ImGui::InputFloat3("Tr", &matrixTranslation[0], "%.3f", flags);
				ImGui::InputFloat3("Rt", &matrixRotation[0], "%.3f", flags);
				ImGui::InputFloat3("Sc", &matrixScale[0], "%.3f", flags);
			}
			ImGuizmo::RecomposeMatrixFromComponents(&matrixTranslation[0], &matrixRotation[0], &matrixScale[0], m16TRSmatrix);
			recomposed = *reinterpret_cast<float32_t4x4*>(m16TRSmatrix);

			if (mCurrentGizmoOperation != ImGuizmo::SCALE)
			{
				if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
					mCurrentGizmoMode = ImGuizmo::LOCAL;
				ImGui::SameLine();
				if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
					mCurrentGizmoMode = ImGuizmo::WORLD;
			}

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

			// generate virtual events given delta TRS matrix
			if (boundCameraToManipulate)
			{
				const float pmSpeed = boundCameraToManipulate->getMoveSpeedScale();
				const float prSpeed = boundCameraToManipulate->getRotationSpeedScale();

				boundCameraToManipulate->setMoveSpeedScale(1);
				boundCameraToManipulate->setRotationSpeedScale(1);

				auto referenceFrame = getCastedMatrix<float64_t>(imguizmoModel.outTRS);
				boundCameraToManipulate->manipulate({}, &referenceFrame);

				boundCameraToManipulate->setMoveSpeedScale(pmSpeed);
				boundCameraToManipulate->setRotationSpeedScale(prSpeed);

				/*
				{
					static std::vector<CVirtualGimbalEvent> virtualEvents(0x45);

					if (not enableActiveCameraMovement)
					{
						uint32_t vCount = {};

						boundCameraToManipulate->beginInputProcessing(m_nextPresentationTimestamp);
						{
							boundCameraToManipulate->process(nullptr, vCount);

							if (virtualEvents.size() < vCount)
								virtualEvents.resize(vCount);

							IGimbalController::SUpdateParameters params;
							params.imguizmoEvents = { { imguizmoModel.outDeltaTRS } };
							boundCameraToManipulate->process(virtualEvents.data(), vCount, params);
						}
						boundCameraToManipulate->endInputProcessing();

						// I start to think controller should be able to set sensitivity to scale magnitudes of generated events
						// in order for camera to not keep any magnitude scalars like move or rotation speed scales

						if (vCount)
						{
							const float pmSpeed = boundCameraToManipulate->getMoveSpeedScale();
							const float prSpeed = boundCameraToManipulate->getRotationSpeedScale();

							boundCameraToManipulate->setMoveSpeedScale(1);
							boundCameraToManipulate->setRotationSpeedScale(1);

							auto referenceFrame = getCastedMatrix<float64_t>(imguizmoModel.outTRS);
							boundCameraToManipulate->manipulate({ virtualEvents.data(), vCount }, &referenceFrame);

							boundCameraToManipulate->setMoveSpeedScale(pmSpeed);
							boundCameraToManipulate->setRotationSpeedScale(prSpeed);
						}
					}
				}
				*/
			}
			else
			{
				// for scene demo model full affine transformation without limits is assumed 
				m_model = float32_t3x4(hlsl::transpose(imguizmoModel.outTRS));
			}
		}

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

		bool resetCursorToCenter = true;

		struct windowControlBinding
		{
			nbl::core::smart_refctd_ptr<IGPUFramebuffer> sceneFramebuffer;
			nbl::core::smart_refctd_ptr<IGPUImageView> sceneColorView;
			nbl::core::smart_refctd_ptr<IGPUImageView> sceneDepthView;
			float32_t3x4 viewMatrix = float32_t3x4(1.f);
			float32_t4x4 viewProjMatrix = float32_t4x4(1.f);

			uint32_t activePlanarIx = 0u;
			bool allowGizmoAxesToFlip = false;
			bool enableDebugGridDraw = true;
			float aspectRatio = 16.f / 9.f;
			bool leftHandedProjection = true;

			std::optional<uint32_t> boundProjectionIx = std::nullopt, lastBoundPerspectivePresetProjectionIx = std::nullopt, lastBoundOrthoPresetProjectionIx = std::nullopt;

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
			}
		};

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
					SetLeftHanded
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
				GimbalDelta
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
		};

		struct ScriptedInputState
		{
			bool enabled = false;
			bool log = false;
			bool exclusive = false;
			bool hardFail = false;
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
		};

		static constexpr inline auto MaxSceneFBOs = 2u;
		std::array<windowControlBinding, MaxSceneFBOs> windowBindings;
		uint32_t activeRenderWindowIx = 0u;

		// UI font atlas + viewport FBO color attachment textures
		constexpr static inline auto TotalUISampleTexturesAmount = 1u + MaxSceneFBOs;

		nbl::core::smart_refctd_ptr<CGeometryCreatorScene> m_scene;
		nbl::core::smart_refctd_ptr<IGPURenderpass> m_sceneRenderpass;
		nbl::core::smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;

		CRenderUI m_ui;
		video::CDumbPresentationOracle oracle;
		uint16_t gcIndex = {}; 

		static constexpr uint32_t CiFramesBeforeCapture = 10u;
		bool m_ciMode = false;
		bool m_ciScreenshotDone = false;
		uint32_t m_ciFrameCounter = 0u;
		system::path m_ciScreenshotPath;
		ScriptedInputState m_scriptedInput;
		CameraControlSettings m_cameraControls;
		CameraConstraintSettings m_cameraConstraints;
		CUILogFormatter m_logFormatter;
		std::deque<VirtualEventLogEntry> m_virtualEventLog;
		size_t m_virtualEventLogMax = 128u;
		bool m_showHud = true;
		bool m_showEventLog = false;
		bool m_logAutoScroll = true;
		bool m_logWrap = true;
		std::vector<CameraPreset> m_presets;
		std::vector<CameraKeyframe> m_keyframes;
		CameraPlaybackState m_playback;
		CTargetPoseController m_targetPoseController;
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

NBL_MAIN_FUNC(UISampleApp)
