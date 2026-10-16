// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/this_example/builtin/build/spirv/keys.hpp"
#include "nbl/examples/examples.hpp"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include <nbl/system/CHttpDownloader.h>

#include <nbl/builtin/hlsl/math/thin_lens_projection.hlsl>

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;
using namespace nbl::examples;

#include "app_resources/push_constants.hlsl"

class HTTP_AssetLoading final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
	using device_base_t = SimpleWindowedApplication;
	using asset_base_t = BuiltinResourcesApplication;

	std::string ImageFileUrl = "https://raw.githubusercontent.com/Devsh-Graphics-Programming/Nabla-Example-And-Tests-Media/f48f2afe20fee6b356be4c1d922393c751d4dfa7/color_space_test/R8G8B8_1.jpg";
	std::string PlyFileUrl = "https://github.com/Devsh-Graphics-Programming/Nabla-Example-And-Tests-Media/raw/f48f2afe20fee6b356be4c1d922393c751d4dfa7/ply/Spanner-ply.ply";

	enum class EDisplayMode : uint8_t
	{
		Image,	// key 1
		Ply		// key 2
	};
	constexpr static inline auto DepthFormat = EF_D32_SFLOAT;

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		inline HTTP_AssetLoading(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}
		
		// Will get called mid-initialization, via `filterDevices` between when the API Connection is created and Physical Device is chosen
		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			// So let's create our Window and Surface then!
			if (!m_surface)
			{
				{
					IWindow::SCreationParams params = {};
					// forwards keyboard and mouse channels to the input system
					params.callback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem),smart_refctd_ptr(m_logger));
					params.width = 1280;
					params.height = 720;
					params.x = 32;
					params.y = 32;
					// Don't want to have a window lingering about before we're ready so create it hidden.
					params.flags = ui::IWindow::ECF_HIDDEN|IWindow::ECF_CAN_RESIZE|IWindow::ECF_CAN_MAXIMIZE|IWindow::ECF_CAN_MINIMIZE;
					params.windowCaption = "HTTP_AssetLoading";
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}
				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api),smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>::create(std::move(surface));
			}
			if (m_surface)
			{
				return {{m_surface->getSurface()/*,EQF_NONE*/}};
			}

			return {};
		}
		
		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// the window callback created inside `getSurfaces` needs the system, logger and input system to already exist
			if (!MonoSystemMonoLoggerApplication::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			// downloads run on their own thread, their callbacks fire from `m_downloader->update()` on this one
			m_downloader = std::make_unique<CHttpDownloader>(smart_refctd_ptr(m_logger));

			// Load FSTri Shader
			ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
			if (!fsTriProtoPPln)
				return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

			// Load Custom Shader
			auto loadPrecompiledShader = [&]<core::StringLiteral ShaderKey>() -> smart_refctd_ptr<IShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = "app_resources";

				auto key = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_device.get());
				auto assetBundle = m_assetMgr->getAsset(key.data(), lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
					return nullptr;

				auto shader = IAsset::castDown<IShader>(assets[0]);
				return shader;
			};
			auto fragmentShader = loadPrecompiledShader.operator()<"present">(); // "app_resources/present.frag.hlsl"
			if (!fragmentShader)
				return logFail("Failed to load precompiled fragment shader!");

			// Now surface indep resources
			m_semaphore = m_device->createSemaphore(m_submitIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			/*
			* We'll be using a combined image sampler for this example, which lets us assign both a sampled image and a sampler to the same binding.
			* In this example we provide a sampler at descriptor set creation time, via the SBinding struct below. This specifies that the sampler for this binding is immutable,
			* as evidenced by the name of the field in the SBinding.
			* Samplers for combined image samplers can also be mutable, which for a binding of a descriptor set is specified also at creation time by leaving the immutableSamplers
			* field set to its default (nullptr).
			*/
			smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
			{
				auto defaultSampler = m_device->createSampler({
					.AnisotropicFilter = 0
					});

				const IGPUDescriptorSetLayout::SBinding bindings[1] = { {
					.binding = 0,
					.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1,
					.immutableSamplers = &defaultSampler
				}
				};
				dsLayout = m_device->createDescriptorSetLayout(bindings);
				if (!dsLayout)
					return logFail("Failed to Create Descriptor Layout");

			}

			ISwapchain::SCreationParams swapchainParams = { .surface = m_surface->getSurface() };
			// Need to choose a surface format
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");
			// We actually need external dependencies to ensure ordering of the Implicit Layout Transitions relative to the semaphore signals
			const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
				// wipe-transition to ATTACHMENT_OPTIMAL
				{
					.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.dstSubpass = 0,
					.memoryBarrier = {
					// since we're uploading the image data we're about to draw
					// the depth attachment is reused by the next frame, so wait for its last writes too
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT|asset::PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,
					.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
					.dstStageMask = asset::PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT|asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					// because we clear and don't blend
					.dstAccessMask = asset::ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
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
						// we can have NONE as the Destinations because the spec says so about presents
					}
				// leave view offsets and flags default
			},
			IGPURenderpass::SCreationParams::DependenciesEnd
			};
			// the depth buffer is only needed by the mesh modes, the full screen triangle ignores it
			auto scResources = std::make_unique<CSwapchainFramebuffersAndDepth>(m_device.get(), DepthFormat, swapchainParams.surfaceFormat.format, dependencies);
			if (!scResources->getRenderpass())
				return logFail("Failed to create Renderpass!");

			// Now create the pipeline
			{
				const asset::SPushConstantRange range = {
					.stageFlags = ESS_FRAGMENT,
					.offset = 0,
					.size = sizeof(push_constants_t)
				};
				auto layout = m_device->createPipelineLayout({ &range,1 }, nullptr, nullptr, nullptr, core::smart_refctd_ptr(dsLayout));
				const IGPUPipelineBase::SShaderSpecInfo fragSpec = {
					.shader = fragmentShader.get(),
					.entryPoint = "main",
				};
				m_pipeline = fsTriProtoPPln.createPipeline(fragSpec, layout.get(), scResources->getRenderpass()/*,default is subpass 0*/);
				if (!m_pipeline)
					return logFail("Could not create Graphics Pipeline!");
			}

			auto queue = getGraphicsQueue();
			// Let's just use the same queue since there's no need for async present
			if (!m_surface || !m_surface->init(queue, std::move(scResources), swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");

			m_renderer = CSimpleDebugRenderer::create(m_assetMgr.get(), static_cast<CSwapchainFramebuffersAndDepth*>(m_surface->getSwapchainResources())->getRenderpass(), 0);
			if (!m_renderer)
				return logFail("Failed to create the mesh renderer!");

			// create the descriptor sets, 1 per FIF and with enough room for one image sampler
			{
				const uint32_t setCount = MaxFramesInFlight;
				auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, { &dsLayout.get(),1 }, &setCount);
				if (!pool)
					return logFail("Failed to Create Descriptor Pool");

				for (auto i = 0u; i < MaxFramesInFlight; i++)
				{
					m_descriptorSets[i] = pool->createDescriptorSet(core::smart_refctd_ptr(dsLayout));
					if (!m_descriptorSets[i])
						return logFail("Could not create Descriptor Set!");
				}
			}

			// need resetttable commandbuffers for the upload utility
			m_cmdPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			// create the commandbuffers
			for (auto i = 0u; i < MaxFramesInFlight; i++)
			{
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");
				if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i,1 }))
					return logFail("Couldn't create Command Buffer!");
			}

			// things for IUtilities
			m_scratchSemaphore = m_device->createSemaphore(0);
			if (!m_scratchSemaphore)
				return logFail("Could not create Scratch Semaphore");
			m_scratchSemaphore->setObjectDebugName("Scratch Semaphore");
			// we don't want to overcomplicate the example with multi-queue
			m_intendedSubmit.queue = queue;
			// wait for nothing before upload
			m_intendedSubmit.waitSemaphores = {};
			m_intendedSubmit.prevCommandBuffers = {};
			// fill later
			m_intendedSubmit.scratchCommandBuffers = {};
			m_intendedSubmit.scratchSemaphore = {
				.semaphore = m_scratchSemaphore.get(),
				.value = 0,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};

			m_window->setCaption("[Nabla Engine] HTTP Asset Load Test Demo");

			m_loadCWD = "..";

			requestImage();
			requestPly();

			m_winMgr->show(m_window.get());

			return true;
		}

	void processInputs()
	{
		// 1 = image, 2 = PLY
		{
			m_inputSystem->getDefaultKeyboard(&m_keyboard);
			std::optional<EDisplayMode> requested;
			m_keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events)->void
			{
				for (const auto& event : events)
				{
					if (event.action != SKeyboardEvent::ECA_RELEASED)
					{
						continue;
					}

					switch (event.keyCode)
					{
					case E_KEY_CODE::EKC_1: requested = EDisplayMode::Image; break;
					case E_KEY_CODE::EKC_2: requested = EDisplayMode::Ply; break;
					default: break;
					}
				}
			}, m_logger.get());

			if (requested)
			{
				switchTo(*requested);
			}
		}
	}

	bool render_and_present()
	{
		// framesInFlight: ensuring safe execution of command buffers and acquires, `framesInFlight` only affect semaphore waits, don't use this to index your resources because it can change with swapchain recreation.
		const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
		// We block for semaphores for 2 reasons here:
		// A) Resource: Can't use resource like a command buffer BEFORE previous use is finished! [MaxFramesInFlight]
		// B) Acquire: Can't have more acquires in flight than a certain threshold returned by swapchain or your surface helper class. [MaxAcquiresInFlight]
		if (m_submitIx >= framesInFlight)
		{
			const ISemaphore::SWaitInfo cbDonePending[] =
			{
				{
					.semaphore = m_semaphore.get(),
					.value = m_submitIx + 1 - framesInFlight
				}
			};
			if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
				return true;
		}

		auto acquire = m_surface->acquireNextImage();
		if (!acquire)
			return true;

		const auto resourceIx = m_submitIx%MaxFramesInFlight;

		auto queue = getGraphicsQueue();
		auto cmdbuf = m_cmdBufs[resourceIx].get();
		auto ds = m_descriptorSets[resourceIx].get();
		cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

		auto scRes = static_cast<CSwapchainFramebuffersAndDepth*>(m_surface->getSwapchainResources());
		const auto& framebufferParams = scRes->getFramebuffer(acquire.imageIndex)->getCreationParameters();
		const VkRect2D currentRenderArea =
		{
			.offset = {0,0},
			.extent = {framebufferParams.width,framebufferParams.height}
		};
		{
			const asset::SViewport viewport =
			{
				.width = float(framebufferParams.width),
				.height = float(framebufferParams.height)
			};
			cmdbuf->setViewport({&viewport,1});
		}
		cmdbuf->setScissor({&currentRenderArea,1});

		{
			IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {1.f,0.f,1.f,1.f} };
			if (m_mode==EDisplayMode::Ply)
				clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
			// reverse Z, the mesh pipelines use `GREATER`
			const IGPUCommandBuffer::SClearDepthStencilValue depthValue = { .depth = 0.f };
			const IGPUCommandBuffer::SRenderpassBeginInfo info = {
				.framebuffer = scRes->getFramebuffer(acquire.imageIndex),
				.colorClearValues = &clearValue,
				.depthStencilClearValues = &depthValue,
				.renderArea = currentRenderArea
			};
			cmdbuf->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
		}

		if (m_mode==EDisplayMode::Image)
		{
			// the descriptor sets only get written once the image has downloaded, until then just clear
			if (m_imageLoaded)
			{
				cmdbuf->bindGraphicsPipeline(m_pipeline.get());
				cmdbuf->pushConstants(m_pipeline->getLayout(),hlsl::ShaderStage::ESS_FRAGMENT,0,sizeof(push_constants_t),&m_pc);
				cmdbuf->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS,m_pipeline->getLayout(),3,1,&ds);
				ext::FullScreenTriangle::recordDrawCall(cmdbuf);
			}
		}
		else
		{
			const float aspectRatio = float(framebufferParams.width)/float(framebufferParams.height);
			recordMesh(cmdbuf,aspectRatio);
		}
		cmdbuf->endRenderPass();

		cmdbuf->end();

		const IQueue::SSubmitInfo::SSemaphoreInfo rendered[1] = {{
			.semaphore = m_semaphore.get(),
			.value = ++m_submitIx,
			// just as we've outputted all pixels, signal
			.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
		}};
		{
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = {{
				.cmdbuf = cmdbuf
			}};
			// we don't need to wait for the transfer semaphore, because we submit everything to the same queue
			const IQueue::SSubmitInfo::SSemaphoreInfo acquired[1] = {{
				.semaphore = acquire.semaphore,
				.value = acquire.acquireCount,
				.stageMask = PIPELINE_STAGE_FLAGS::NONE
			}};
			const IQueue::SSubmitInfo infos[1] = {{
				.waitSemaphores = acquired,
				.commandBuffers = commandBuffers,
				.signalSemaphores = rendered
			}};
			// we won't signal the semaphore if no success
			if (queue->submit(infos)!=IQueue::RESULT::SUCCESS)
			{
				m_submitIx--;
			}
		}

		return m_surface->present(acquire.imageIndex, rendered);
	}

		inline void workLoopBody() override
		{
			//Check for stuff that's downloaded. Triggers callbacks
			m_downloader->update();

			processInputs();
			render_and_present();
		}

		inline bool keepRunning() override
		{
			// Keep running as long as we have a surface to present to (usually this means, as long as the window is open)
			return !m_surface->irrecoverable();
		}

		inline bool onAppTerminated() override
		{
			m_downloader = nullptr;

			if (m_device)
			{
				m_device->waitIdle();
			}

			m_renderer = nullptr;
			m_inputSystem = nullptr;

			return device_base_t::onAppTerminated();
		}

	protected:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>> m_surface;
		smart_refctd_ptr<InputSystem> m_inputSystem;
		InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;
		// draws the PLY
		smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;
		EDisplayMode m_mode = EDisplayMode::Image;
		// what the user last asked for, may still be downloading
		EDisplayMode m_requestedMode = EDisplayMode::Image;
		bool m_imageLoaded = false;
		bool m_plyLoaded = false;
		hlsl::float64_t3 m_modelCenter = hlsl::float64_t3(0.0);
		double m_modelSize = 1.0;
		const std::chrono::steady_clock::time_point m_startTime = std::chrono::steady_clock::now();
		//
		system::path m_loadCWD;
		//

		std::unique_ptr<CHttpDownloader> m_downloader;

		smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;
		// We can't use the same semaphore for acquire and present, because that would disable "Frames in Flight" by syncing previous present against next acquire.
		smart_refctd_ptr<ISemaphore> m_semaphore;
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		// for image uploads
		smart_refctd_ptr<ISemaphore> m_scratchSemaphore;
		SIntendedSubmitInfo m_intendedSubmit;
		// Use a separate counter to cycle through our resources for clarity
		uint64_t m_submitIx = 0;
		// Maximum frames which can be simultaneously submitted,used to cycle through our per-frame resources like command buffers
		constexpr static inline uint32_t MaxFramesInFlight = 3u;

		// Enough Command Buffers and other resources for all frames in flight!
		std::array<smart_refctd_ptr<IGPUDescriptorSet>,MaxFramesInFlight> m_descriptorSets;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,MaxFramesInFlight> m_cmdBufs;
		push_constants_t m_pc = {};

	private:

		void setMode(const EDisplayMode mode)
		{
			m_mode = mode;
			m_window->setCaption(mode==EDisplayMode::Image ? "[Nabla Engine] HTTP Asset Load Test Demo":"[Nabla Engine] HTTP Asset Load Test Demo - mesh (.ply)");
		}

		void switchTo(const EDisplayMode mode)
		{
			m_requestedMode = mode;
			if (mode==m_mode)
				return;

			// the mesh is only downloaded in `onAppInitialized`, if it's still in flight `onPlyDownloaded` switches once it arrives, if it failed this stays on the image
			if (mode==EDisplayMode::Ply && !m_plyLoaded)
				return;

			setMode(mode);
		}

		// only called from `onAppInitialized`
		void requestPly()
		{
			m_logger->log("Loading mesh from url %s", ILogger::ELL_INFO, PlyFileUrl.c_str());
			m_downloader->downloadFile(PlyFileUrl,[this](bool success, const std::vector<uint8_t>& data)->void
			{
				onPlyDownloaded(success,data);
			});
		}

		void onPlyDownloaded(const bool success, const std::vector<uint8_t>& data)
		{
			if (!success)
			{
				m_logger->log("Failed to download %s", ILogger::ELL_ERROR, PlyFileUrl.c_str());
				return;
			}

			auto file = CHttpDownloader::createFile(path(PlyFileUrl).filename(),data);
			if (!file || !loadPlyFromFile(std::move(file)))
			{
				m_logger->log("Could not show the PLY", ILogger::ELL_ERROR);
				return;
			}
			m_plyLoaded = true;

			if (m_requestedMode==EDisplayMode::Ply)
				setMode(EDisplayMode::Ply);
		}

		bool loadPlyFromFile(smart_refctd_ptr<IFile> file)
		{
			const std::string fileName = file->getFileName().string();
			IAssetLoader::SAssetLoadParams params = {};
			params.logger = m_logger.get();

			if (m_assetMgr->getAsset(file.get(), fileName, params).getContents().empty())
			{
				m_logger->log("Failed to load the downloaded %s", ILogger::ELL_ERROR, PlyFileUrl.c_str());
				return false;
			}

			return loadPly(fileName);
		}

		bool loadPly(const system::path& plyPath)
		{
			m_logger->log("Loading mesh %s", ILogger::ELL_INFO, plyPath.string().c_str());

			IAssetLoader::SAssetLoadParams params = {};
			params.logger = m_logger.get();
			const auto bundle = m_assetMgr->getAsset(plyPath.string(),params);
			if (bundle.getAssetType()!=IAsset::E_TYPE::ET_GEOMETRY)
			{
				m_logger->log("%s did not load as a polygon geometry", ILogger::ELL_ERROR, plyPath.string().c_str());
				return false;
			}
			core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
			for (const auto& item : bundle.getContents())
				if (auto polyGeo = IAsset::castDown<ICPUPolygonGeometry>(item); polyGeo)
					geometries.push_back(polyGeo);
			if (geometries.empty())
			{
				m_logger->log("%s has no polygon geometry", ILogger::ELL_ERROR, plyPath.string().c_str());
				return false;
			}

			auto converter = CAssetConverter::create({.device=m_device.get()});
			CAssetConverter::SInputs inputs = {};
			inputs.logger = m_logger.get();
			core::vector<CAssetConverter::patch_t<ICPUPolygonGeometry>> patches(geometries.size(),CSimpleDebugRenderer::DefaultPolygonGeometryPatch);
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = {&geometries.front().get(),geometries.size()};
			std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = patches;

			auto reservation = converter->reserve(inputs);
			if (!reservation)
			{
				m_logger->log("Failed to reserve GPU objects for the mesh!", ILogger::ELL_ERROR);
				return false;
			}

			// upload on the graphics queue
			{
				auto queue = getGraphicsQueue();
				auto semaphore = m_device->createSemaphore(0u);

				constexpr auto MultiBuffering = 2;
				std::array<smart_refctd_ptr<IGPUCommandBuffer>,MultiBuffering> commandBuffers = {};
				{
					auto pool = m_device->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT|IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
					pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,commandBuffers,smart_refctd_ptr(m_logger));
				}
				commandBuffers.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

				std::array<IQueue::SSubmitInfo::SCommandBufferInfo,MultiBuffering> commandBufferSubmits;
				for (auto i=0; i<MultiBuffering; i++)
					commandBufferSubmits[i].cmdbuf = commandBuffers[i].get();

				SIntendedSubmitInfo transfer = {};
				transfer.queue = queue;
				transfer.scratchCommandBuffers = commandBufferSubmits;
				transfer.scratchSemaphore = {
					.semaphore = semaphore.get(),
					.value = 0u,
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
				};

				CAssetConverter::SConvertParams convertParams = {};
				convertParams.utilities = m_utils.get();
				convertParams.transfer = &transfer;
				auto future = reservation.convert(convertParams);
				if (future.copy()!=IQueue::RESULT::SUCCESS)
				{
					m_logger->log("Failed to upload the mesh!", ILogger::ELL_ERROR);
					return false;
				}
			}

			const auto& converted = reservation.getGPUObjects<ICPUPolygonGeometry>();
			if (!m_renderer->addGeometries({&converted.front().get(),converted.size()}))
			{
				m_logger->log("The renderer rejected the mesh!", ILogger::ELL_ERROR);
				return false;
			}

			const hlsl::float32_t3x4 identity = hlsl::transpose(hlsl::float32_t4x3(
				hlsl::float32_t3(1,0,0),
				hlsl::float32_t3(0,1,0),
				hlsl::float32_t3(0,0,1),
				hlsl::float32_t3(0,0,0)
			));

			for (const auto& geo : m_renderer->getGeometries())
			{
				m_renderer->m_instances.push_back({.world=identity,.packedGeo=&geo});
			}

			//Camera framing
			using aabb_t = hlsl::shapes::AABB<3,double>;
			auto bound = aabb_t::create();

			for (const auto& geom : converted)
			{
				bound = hlsl::shapes::util::union_(geom.value->getAABB<aabb_t>(),bound);
			}

			m_modelCenter = (bound.minVx+bound.maxVx)*0.5;
			m_modelSize = hlsl::length(bound.getExtent());

			if (!std::isfinite(m_modelSize) || !(m_modelSize>0.0))
			{
				m_logger->log("The bounds of %s are not usable, cannot frame it", ILogger::ELL_ERROR, plyPath.string().c_str());
				return false;
			}
			return true;
		}

		// spins the mesh in front of a fixed camera
		void recordMesh(IGPUCommandBuffer* cmdbuf, const float aspectRatio)
		{
			const double distance = m_modelSize;
			const float angle = float(std::chrono::duration<double>(std::chrono::steady_clock::now()-m_startTime).count()*0.5);
			const float s = std::sin(angle);
			const float c = std::cos(angle);
			const hlsl::float32_t3 center = hlsl::float32_t3(m_modelCenter);

			// spin about Y, then tilt about X so flat meshes are never seen edge-on
			constexpr float tilt = 0.4f;
			const float st = std::sin(tilt);
			const float ct = std::cos(tilt);
			auto tiltX = [&](const hlsl::float32_t3 v)->hlsl::float32_t3 { return hlsl::float32_t3(v.x,ct*v.y-st*v.z,st*v.y+ct*v.z); };
			const hlsl::float32_t3 spunX(c,0,-s), spunY(0,1,0), spunZ(s,0,c);

			// rotate around the model's center, then push it in front of the camera
			const hlsl::float32_t3 rotatedCenter = tiltX(hlsl::float32_t3(c*center.x+s*center.z,center.y,-s*center.x+c*center.z));
			const hlsl::float32_t3 translation(-rotatedCenter.x,-rotatedCenter.y,-rotatedCenter.z-float(distance));
			const hlsl::float32_t3x4 world = hlsl::transpose(hlsl::float32_t4x3(tiltX(spunX),tiltX(spunY),tiltX(spunZ),translation));
			for (auto& instance : m_renderer->m_instances)
				instance.world = world;

			const hlsl::float32_t3x4 view = hlsl::transpose(hlsl::float32_t4x3(
				hlsl::float32_t3(1,0,0),
				hlsl::float32_t3(0,1,0),
				hlsl::float32_t3(0,0,1),
				hlsl::float32_t3(0,0,0)
			));
			// reverse Z
			const auto projection = hlsl::math::thin_lens::rhPerspectiveFovMatrix<float>(1.2f,aspectRatio,float(m_modelSize*4.0),float(m_modelSize*0.01));
			m_renderer->render(cmdbuf,CSimpleDebugRenderer::SViewParams(view,projection));
		}

		// only called from `onAppInitialized`
		void requestImage()
		{
			m_logger->log("Loading image from url %s", ILogger::ELL_INFO, ImageFileUrl.c_str());
			m_downloader->downloadFile(ImageFileUrl,[this](bool success, const std::vector<uint8_t>& data)->void
			{
				onImageDownloaded(success,data);
			});
		}

		void onImageDownloaded(const bool success, const std::vector<uint8_t>& data)
		{
			if (!success)
			{
				m_logger->log("Failed to download %s", ILogger::ELL_ERROR, ImageFileUrl.c_str());
				return;
			}

			auto file = CHttpDownloader::createFile(path(ImageFileUrl).filename(),data);
			if (!file)
			{
				m_logger->log("Could not create a file for the downloaded %s", ILogger::ELL_ERROR, ImageFileUrl.c_str());
				return;
			}

			const auto cpuImgView = getImageViewFromFile(std::move(file));
			if (!cpuImgView || !uploadImage(cpuImgView.get()))
			{
				m_logger->log("Failed to load image \"%s\"!", ILogger::ELL_ERROR, ImageFileUrl.c_str());
				return;
			}
			m_imageLoaded = true;
		}

		// converts + uploads the image and points the descriptor sets at it
		bool uploadImage(ICPUImageView* const cpuImgView)
		{
			smart_refctd_ptr<IGPUCommandBuffer> uploadCmdBuf;
			if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &uploadCmdBuf,1 }))
				return logFail("Couldn't create Upload Command Buffer!");
			uploadCmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			smart_refctd_ptr<IGPUImageView> gpuView;
			{
				auto converter = CAssetConverter::create({.device=m_device.get()});
				CAssetConverter::patch_t<ICPUImageView> patch(cpuImgView,IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT);

				// We don't want to generate mip-maps for these images (YET), to ensure that we must override the default callbacks.
				struct SInputs final : CAssetConverter::SInputs
				{
					inline uint8_t getMipLevelCount(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
					{
						return image->getCreationParameters().mipLevels;
					}
					inline uint16_t needToRecomputeMips(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
					{
						return 0b0u;
					}
				} inputs = {};
				ICPUImageView* const cpuImgViewPtr = cpuImgView;
				std::get<CAssetConverter::SInputs::asset_span_t<ICPUImageView>>(inputs.assets) = { &cpuImgViewPtr,1 };
				std::get<CAssetConverter::SInputs::patch_span_t<ICPUImageView>>(inputs.patches) = { &patch,1 };
				inputs.logger = m_logger.get();

				auto reservation = converter->reserve(inputs);
				gpuView = reservation.getGPUObjects<ICPUImageView>().front().value;
				if (!gpuView)
					return logFail("Failed to convert the image to a GPU image view!");
				gpuView->getCreationParameters().image->setObjectDebugName(ImageFileUrl.c_str());

				IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = {uploadCmdBuf.get()};
				m_intendedSubmit.scratchCommandBuffers = {&cmdbufInfo,1};
				CAssetConverter::SConvertParams params = {};
				params.transfer = &m_intendedSubmit;
				params.utilities = m_utils.get();
				auto result = reservation.convert(params);
				if (result.copy()!=IQueue::RESULT::SUCCESS)
					return logFail("Failed to upload the image!");
			}

			for (auto& ds : m_descriptorSets)
			{
				IGPUDescriptorSet::SDescriptorInfo info = {};
				info.desc = gpuView;
				info.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
				const IGPUDescriptorSet::SWriteDescriptorSet writes[] = { {
					.dstSet = ds.get(),
					.binding = 0,
					.arrayElement = 0,
					.count = 1,
					.info = &info
				} };
				m_device->updateDescriptorSets(writes, {});
			}

			const auto& imageParams = cpuImgView->getCreationParameters().image->getCreationParameters();
			const float realLayers = core::max(imageParams.arrayLayers,imageParams.extent.depth);
			m_pc.grid.x = ceil(sqrt(realLayers));
			m_pc.grid.y = ceil(realLayers/float(m_pc.grid.x));

			return true;
		}

		smart_refctd_ptr<ICPUImageView> getImageViewFromFile(smart_refctd_ptr<IFile> file)
		{
			constexpr auto cachingFlags = static_cast<IAssetLoader::E_CACHING_FLAGS>(IAssetLoader::ECF_DONT_CACHE_REFERENCES & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
			const IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags, IAssetLoader::ELPF_NONE, m_logger.get(), m_loadCWD);

			auto bundle = m_assetMgr->getAsset(file.get(), file->getFileName().string(), loadParams);
			auto contents = bundle.getContents();
			if (contents.empty())
			{
				m_logger->log("Failed to load the downloaded %s!", ILogger::ELL_ERROR, file->getFileName().string().c_str());
				return nullptr;
			}

			const auto& asset = contents[0];
			ICPUImageView::SCreationParams viewParams = {};
			switch (asset->getAssetType())
			{
			case IAsset::ET_IMAGE:
				{
					auto image = smart_refctd_ptr_static_cast<ICPUImage>(asset);
					const auto format = image->getCreationParameters().format;

					viewParams =
					{
						.flags = ICPUImageView::E_CREATE_FLAGS::ECF_NONE,
						.image = std::move(image),
						.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D_ARRAY,
						.format = format,
						.subresourceRange = {
							.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
							.baseMipLevel = 0u,
							.levelCount = ICPUImageView::remaining_mip_levels,
							.baseArrayLayer = 0u,
							.layerCount = ICPUImageView::remaining_array_layers
						}
					};
				} break;

			case IAsset::ET_IMAGE_VIEW:
				// loaders may produce 2D views (e.g. KTX), the shader samples a 2D array
				viewParams = smart_refctd_ptr_static_cast<ICPUImageView>(asset)->getCreationParameters();
				viewParams.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D_ARRAY;
				break;

			default:
				m_logger->log("Failed to load ICPUImage or ICPUImageView got some other Asset Type!", ILogger::ELL_ERROR);
				return nullptr;
			}

			return ICPUImageView::create(std::move(viewParams));
		}
};

NBL_MAIN_FUNC(HTTP_AssetLoading)
