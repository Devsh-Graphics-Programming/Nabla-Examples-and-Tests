// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nlohmann/json.hpp"
#include "argparse/argparse.hpp"
using json = nlohmann::json;

#include "common.hpp"
#include "keysmapping.hpp"
#include "camera/CCubeProjection.hpp"
#include "glm/glm/ext/matrix_clip_space.hpp" // TODO: TESTING
#include "nbl/ext/ScreenShot/ScreenShot.h"
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

						if (jCamera["type"] == "FPS")
						{
							if (!withOrientation)
							{
								logFail("Expected \"orientation\" keyword for FPS camera definition!");
								return false;
							}

							cameras.emplace_back() = make_smart_refctd_ptr<CFPSCamera>(position, getOrientation());
						}
						else if (jCamera["type"] == "Free")
						{
							if (!withOrientation)
							{
								logFail("Expected \"orientation\" keyword for Free camera definition!");
								return false;
							}

							cameras.emplace_back() = make_smart_refctd_ptr<CFreeCamera>(position, getOrientation());
						}
						else if (jCamera["type"] == "Orbit")
						{
							auto& camera = cameras.emplace_back() = make_smart_refctd_ptr<COrbitCamera>(position, getTarget());
							camera->setMoveSpeedScale(0.2);
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
				{
					const auto ds = float32_t2{ m_window->getWidth(), m_window->getHeight() };

					wInit.trsEditor.iPos = iPaddingOffset;
					wInit.trsEditor.iSize = { ds.x * 0.1, ds.y - wInit.trsEditor.iPos.y * 2 };

					wInit.planars.iSize = { ds.x * 0.2, ds.y - iPaddingOffset.y * 2 };
					wInit.planars.iPos = { ds.x - wInit.planars.iSize.x - iPaddingOffset.x, 0 + iPaddingOffset.y };

					{
						float leftX = wInit.trsEditor.iPos.x + wInit.trsEditor.iSize.x + iPaddingOffset.x;
						float eachXSize = wInit.planars.iPos.x - (wInit.trsEditor.iPos.x + wInit.trsEditor.iSize.x) - 2*iPaddingOffset.x;
						float eachYSize = (ds.y - 2 * iPaddingOffset.y - (wInit.renderWindows.size() - 1) * iPaddingOffset.y) / wInit.renderWindows.size();
						
						for (size_t i = 0; i < wInit.renderWindows.size(); ++i)
						{
							auto& rw = wInit.renderWindows[i];
							rw.iPos = { leftX, (1+i) * iPaddingOffset.y + i * eachYSize };
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
						if (!m_device || !m_assetManager || !m_surface)
							return;

						m_logger->log("CI screenshot capture start (frame %u).", ILogger::ELL_INFO, m_ciFrameCounter);
						const ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx };
						if (m_device->blockForSemaphores({ &waitInfo, &waitInfo + 1 }) != ISemaphore::WAIT_RESULT::SUCCESS)
						{
							m_logger->log("CI screenshot failed: wait for render finished.", ILogger::ELL_ERROR);
							return;
						}

						if (!frame)
						{
							m_logger->log("CI screenshot failed: missing frame image.", ILogger::ELL_ERROR);
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
							m_logger->log("CI screenshot failed: could not create frame view.", ILogger::ELL_ERROR);
							return;
						}

						m_logger->log("CI screenshot capture: calling createScreenShot.", ILogger::ELL_INFO);
						const bool ok = ext::ScreenShot::createScreenShot(
							m_device.get(),
							getGraphicsQueue(),
							nullptr,
							frameView.get(),
							m_assetManager.get(),
							m_ciScreenshotPath,
							asset::IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
							asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT);

						if (ok)
							m_logger->log("CI screenshot saved to \"%s\".", ILogger::ELL_INFO, m_ciScreenshotPath.string().c_str());
						else
							m_logger->log("CI screenshot failed to save.", ILogger::ELL_ERROR);
					}
				}

				m_surface->present(std::move(swapchainLock), presentInfo);
			}
			firstFrame = false;
		}

		inline bool keepRunning() override
		{
			if (m_ciMode && m_ciScreenshotDone)
				return false;
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

			const auto cursorPosition = m_window->getCursorControl()->getPosition();

			nbl::ext::imgui::UI::SUpdateParameters params =
			{
				.mousePosition = nbl::hlsl::float32_t2(cursorPosition.x, cursorPosition.y) - nbl::hlsl::float32_t2(m_window->getX(), m_window->getY()),
				.displaySize = { m_window->getWidth(), m_window->getHeight() },
				.mouseEvents = { capturedEvents.mouse.data(), capturedEvents.mouse.size() },
				.keyboardEvents = { capturedEvents.keyboard.data(), capturedEvents.keyboard.size() }
			};

			if (enableActiveCameraMovement)
			{
				auto& binding = windowBindings[activeRenderWindowIx];
				auto& planar = m_planarProjections[binding.activePlanarIx];
				auto* camera = planar->getCamera();
				
				assert(binding.boundProjectionIx.has_value());
				auto& projection = planar->getPlanarProjections()[binding.boundProjectionIx.value()];

				static std::vector<CVirtualGimbalEvent> virtualEvents(0x45);
				uint32_t vCount = {};

				projection.beginInputProcessing(m_nextPresentationTimestamp);
				{
					projection.process(nullptr, vCount);

					if (virtualEvents.size() < vCount)
						virtualEvents.resize(vCount);

					auto* orbit = dynamic_cast<COrbitCamera*>(camera);

					if (orbit)
					{
						uint32_t vKeyboardEventsCount = {}, vMouseEventsCount = {};

						projection.processKeyboard(nullptr, vKeyboardEventsCount, {});
						projection.processMouse(nullptr, vMouseEventsCount, {});

						auto* output = virtualEvents.data();

						projection.processKeyboard(output, vKeyboardEventsCount, params.keyboardEvents); 
						output += vKeyboardEventsCount;

						if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
							projection.processMouse(output, vMouseEventsCount, params.mouseEvents);
						else
							vMouseEventsCount = 0;

						vCount = vKeyboardEventsCount + vMouseEventsCount;
					}
					else
						projection.process(virtualEvents.data(), vCount, { params.keyboardEvents, params.mouseEvents });
				}
				projection.endInputProcessing();

				if (vCount)
					camera->manipulate({ virtualEvents.data(), vCount });
			}

			m_ui.manager->update(params);
		}

	private:
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
					nbl::hlsl::ShowDebugWindow();
					ImGuizmo::ShowDebugImguizmoWindow();
				}

				SImResourceInfo info;
				info.samplerIx = (uint16_t)nbl::ext::imgui::UI::DefaultSamplerIx::USER;

				// ORBIT CAMERA TEST
				{
					for (auto& planar : m_planarProjections)
					{
						auto* camera = planar->getCamera();

						auto* orbit = dynamic_cast<COrbitCamera*>(camera);

						if (orbit)
						{
							auto targetPostion = hlsl::transpose(getMatrix3x4As4x4(m_model))[3];
							orbit->target(targetPostion);
							orbit->manipulate({}, {});
						}
					}
				}

				// render bound planar camera views onto GUI windows
				if (useWindow)
				{
					// ABS TRS editor to manipulate bound object
					TransformEditor();

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

						ImGui::PushStyleColor(ImGuiCol_WindowBg, (ImVec4)ImColor(0.35f, 0.3f, 0.3f));
						const std::string ident = "Render Window \"" + std::to_string(windowIx) + "\"";

						ImGui::Begin(ident.data(), 0);
						const ImVec2 contentRegionSize = ImGui::GetContentRegionAvail(), windowPos = ImGui::GetWindowPos(), cursorPos = ImGui::GetCursorScreenPos();

						ImGuiWindow* window = ImGui::GetCurrentWindow();
						{
							const auto mPos = ImGui::GetMousePos();

							if (mPos.x < cursorPos.x || mPos.y < cursorPos.y || mPos.x > cursorPos.x + contentRegionSize.x || mPos.y > cursorPos.y + contentRegionSize.y)
								window->Flags = ImGuiWindowFlags_None;
							else
								window->Flags = ImGuiWindowFlags_NoMove;
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

						// I will assume we need to focus a window to start manipulating objects from it
						if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows))
							activeRenderWindowIx = windowIx;

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

									ImGui::End();

									ImGui::PopStyleVar();
									ImGui::PopStyleColor(2);
								}
							}
							ImGuizmo::PopID();
						}

						ImGui::End();
						ImGui::PopStyleColor(1);
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

			// Planars
			{
				// setup
				{					
					const ImGuiCond windowCond = m_ciMode ? ImGuiCond_Always : ImGuiCond_Appearing;
					ImGui::SetNextWindowPos({ wInit.planars.iPos.x, wInit.planars.iPos.y }, windowCond);
					ImGui::SetNextWindowSize({ wInit.planars.iSize.x, wInit.planars.iSize.y }, windowCond);
				}

				ImGui::Begin("Planar projection");
				ImGui::Checkbox("Window mode##useWindow", &useWindow);
				ImGui::Separator();

				auto& active = windowBindings[activeRenderWindowIx];
				const auto activeRenderWindowIxString = std::to_string(activeRenderWindowIx);

				ImGui::Text("Active Render Window: %s", activeRenderWindowIxString.c_str());
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
				}

				assert(active.boundProjectionIx.has_value());
				assert(active.lastBoundPerspectivePresetProjectionIx.has_value());
				assert(active.lastBoundOrthoPresetProjectionIx.has_value());

				const auto activePlanarIxString = std::to_string(active.activePlanarIx);
				auto& planarBound = m_planarProjections[active.activePlanarIx];
				assert(planarBound);

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

				auto* const boundCamera = planarBound->getCamera();
				auto& boundProjection = planarBound->getPlanarProjections()[active.boundProjectionIx.value()];
				assert(not boundProjection.isProjectionSingular());

				auto updateParameters = boundProjection.getParameters();

				if (useWindow)
					ImGui::Checkbox("Allow axes to flip##allowAxesToFlip", &active.allowGizmoAxesToFlip);

				if(useWindow)
					ImGui::Checkbox("Draw debug grid##drawDebugGrid", &active.enableDebugGridDraw);

				if (ImGui::RadioButton("LH", active.leftHandedProjection))
					active.leftHandedProjection = true;

				ImGui::SameLine();

				if (ImGui::RadioButton("RH", not active.leftHandedProjection))
					active.leftHandedProjection = false;

				updateParameters.m_zNear = std::clamp(updateParameters.m_zNear, 0.1f, 100.f);
				updateParameters.m_zFar = std::clamp(updateParameters.m_zFar, 110.f, 10000.f);

				ImGui::SliderFloat("zNear", &updateParameters.m_zNear, 0.1f, 100.f, "%.2f", ImGuiSliderFlags_Logarithmic);
				ImGui::SliderFloat("zFar", &updateParameters.m_zFar, 110.f, 10000.f, "%.1f", ImGuiSliderFlags_Logarithmic);

				switch (selectedProjectionType)
				{
					case IPlanarProjection::CProjection::Perspective:
					{
						ImGui::SliderFloat("Fov", &updateParameters.m_planar.perspective.fov, 20.f, 150.f, "%.1f", ImGuiSliderFlags_Logarithmic);
						boundProjection.setPerspective(updateParameters.m_zNear, updateParameters.m_zFar, updateParameters.m_planar.perspective.fov);
					} break;

					case IPlanarProjection::CProjection::Orthographic:
					{
						ImGui::SliderFloat("Ortho width", &updateParameters.m_planar.orthographic.orthoWidth, 1.f, 30.f, "%.1f", ImGuiSliderFlags_Logarithmic);
						boundProjection.setOrthographic(updateParameters.m_zNear, updateParameters.m_zFar, updateParameters.m_planar.orthographic.orthoWidth);
					} break;

					default: break;
				}

				{
					if (ImGui::TreeNodeEx("Cursor Behaviour"))
					{
						if (ImGui::RadioButton("Clamp to the window", !resetCursorToCenter))
							resetCursorToCenter = false;
						if (ImGui::RadioButton("Reset to the window center", resetCursorToCenter))
							resetCursorToCenter = true;
						ImGui::TreePop();
					}
				}

				{
					ImGuiIO& io = ImGui::GetIO();

					if (ImGui::IsKeyPressed(ImGuiKey_Space))
						enableActiveCameraMovement = !enableActiveCameraMovement;

					if (enableActiveCameraMovement)
					{
						ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Bound Camera Movement: Enabled");
						io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
						io.MouseDrawCursor = false;
						io.WantCaptureMouse = false;

						ImVec2 cursorPos = ImGui::GetMousePos();
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
						ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Bound Camera Movement: Disabled");
						io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
						io.MouseDrawCursor = true;
						io.WantCaptureMouse = true;
					}
		

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

						ImGui::Text("Press 'Space' to Enable/Disable bound planar camera movement");

						ImGui::End();

						ImGui::PopStyleVar();
						ImGui::PopStyleColor(2);
					}

					ImGui::Separator();

					const auto flags = ImGuiTreeNodeFlags_DefaultOpen;
					if (ImGui::TreeNodeEx("Bound Camera", flags))
					{
						ImGui::Text("Type: %s", boundCamera->getIdentifier().data());
						ImGui::Text("Object Ix: %s", std::to_string(active.activePlanarIx + 1u).c_str());
						ImGui::Separator();
						{
							auto* orbit = dynamic_cast<COrbitCamera*>(boundCamera);

							float moveSpeed = boundCamera->getMoveSpeedScale();
							float rotationSpeed = boundCamera->getRotationSpeedScale();

							ImGui::SliderFloat("Move speed factor", &moveSpeed, 0.0001f, 10.f, "%.4f", ImGuiSliderFlags_Logarithmic);

							if(not orbit)
								ImGui::SliderFloat("Rotate speed factor", &rotationSpeed, 0.0001f, 10.f, "%.4f", ImGuiSliderFlags_Logarithmic);

							boundCamera->setMoveSpeedScale(moveSpeed);
							boundCamera->setRotationSpeedScale(rotationSpeed);

							{
								if (orbit)
								{
									float distance = orbit->getDistance();
									ImGui::SliderFloat("Distance", &distance, COrbitCamera::MinDistance, COrbitCamera::MaxDistance, "%.4f", ImGuiSliderFlags_Logarithmic);
									orbit->setDistance(distance);
								}
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
				}

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

				ImGui::End();
			}
		}

		inline void TransformEditor()
		{
			static float bounds[] = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
			static float boundsSnap[] = { 0.1f, 0.1f, 0.1f };
			static bool boundSizing = false;
			static bool boundSizingSnap = false;

			ImGuiIO& io = ImGui::GetIO();

			// setup
			{
				const ImGuiCond windowCond = m_ciMode ? ImGuiCond_Always : ImGuiCond_Appearing;
				ImGui::SetNextWindowPos({ wInit.trsEditor.iPos.x, wInit.trsEditor.iPos.y }, windowCond);
				ImGui::SetNextWindowSize({ wInit.trsEditor.iSize.x, wInit.trsEditor.iSize.y }, windowCond);
			}

			ImGui::Begin("TRS Editor");
			{
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

			ImGui::End();
			{
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

		bool resetCursorToCenter = false;

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
