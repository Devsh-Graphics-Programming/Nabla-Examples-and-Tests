// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "nbl/examples/examples.hpp"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"


using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"

template<typename SwapchainResources> requires std::is_base_of_v<ISimpleManagedSurface::ISwapchainResources, SwapchainResources>
class CExplicitSurfaceFormatResizeSurface final : public ISimpleManagedSurface
{
public:
	using this_t = CExplicitSurfaceFormatResizeSurface<SwapchainResources>;

	// Factory method so we can fail, requires a `_surface` created from a window and with a callback that inherits from `ICallback` declared just above
	template<typename Surface> requires std::is_base_of_v<CSurface<typename Surface::window_t, typename Surface::immediate_base_t>, Surface>
	static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<Surface>&& _surface)
	{
		if (!_surface)
			return nullptr;

		auto _window = _surface->getWindow();
		ICallback* cb = nullptr;
		if (_window)
			cb = dynamic_cast<ICallback*>(_window->getEventCallback());

		return core::smart_refctd_ptr<this_t>(new this_t(std::move(_surface), cb), core::dont_grab);
	}

	// Factory method so we can fail, requires a `_surface` created from a native surface
	template<typename Surface> requires std::is_base_of_v<CSurfaceNative<typename Surface::window_t, typename Surface::immediate_base_t>, Surface>
	static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<Surface>&& _surface, ICallback* cb)
	{
		if (!_surface)
			return nullptr;

		return core::smart_refctd_ptr<this_t>(new this_t(std::move(_surface), cb), core::dont_grab);
	}

	//
	inline bool init(CThreadSafeQueueAdapter* queue, std::unique_ptr<SwapchainResources>&& scResources, const ISwapchain::SSharedCreationParams& sharedParams = {})
	{
		if (!scResources || !base_init(queue))
			return init_fail();

		m_sharedParams = sharedParams;
		if (!m_sharedParams.deduce(queue->getOriginDevice()->getPhysicalDevice(), getSurface()))
			return init_fail();

		m_swapchainResources = std::move(scResources);
		return true;
	}

	// Can be public because we don't need to worry about mutexes unlike the Smooth Resize class
	inline ISwapchainResources* getSwapchainResources() override { return m_swapchainResources.get(); }

	// need to see if the swapchain is invalidated (e.g. because we're starting from 0-area old Swapchain) and try to recreate the swapchain
	inline SAcquireResult acquireNextImage()
	{
		if (!isWindowOpen())
		{
			becomeIrrecoverable();
			return {};
		}

		if (!m_swapchainResources || (m_swapchainResources->getStatus() != ISwapchainResources::STATUS::USABLE && !recreateSwapchain(m_surfaceFormat)))
			return {};

		return ISimpleManagedSurface::acquireNextImage();
	}

	// its enough to just foward though
	inline bool present(const uint8_t imageIndex, const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores)
	{
		return ISimpleManagedSurface::present(imageIndex, waitSemaphores);
	}

	//
	inline bool recreateSwapchain(const ISurface::SFormat& explicitSurfaceFormat)
	{
		assert(m_swapchainResources);
		// dont assign straight to `m_swapchainResources` because of complex refcounting and cycles
		core::smart_refctd_ptr<ISwapchain> newSwapchain;
		// TODO: This block of code could be rolled up into `ISimpleManagedSurface::ISwapchainResources` eventually
		{
			auto* surface = getSurface();
			auto device = const_cast<ILogicalDevice*>(getAssignedQueue()->getOriginDevice());
			// 0s are invalid values, so they indicate we want them deduced
			m_sharedParams.width = 0;
			m_sharedParams.height = 0;
			// Question: should we re-query the supported queues, formats, present modes, etc. just-in-time??
			auto* swapchain = m_swapchainResources->getSwapchain();
			if (swapchain ? swapchain->deduceRecreationParams(m_sharedParams) : m_sharedParams.deduce(device->getPhysicalDevice(), surface))
			{
				// super special case, we can't re-create the swapchain but its possible to recover later on
				if (m_sharedParams.width == 0 || m_sharedParams.height == 0)
				{
					// we need to keep the old-swapchain around, but can drop the rest
					m_swapchainResources->invalidate();
					return false;
				}
				// now lets try to create a new swapchain
				if (swapchain)
					newSwapchain = swapchain->recreate(m_sharedParams);
				else
				{
					ISwapchain::SCreationParams params = {
						.surface = core::smart_refctd_ptr<ISurface>(surface),
						.surfaceFormat = explicitSurfaceFormat,
						.sharedParams = m_sharedParams
						// we're not going to support concurrent sharing in this simple class
					};
					m_surfaceFormat = explicitSurfaceFormat;
					newSwapchain = CVulkanSwapchain::create(core::smart_refctd_ptr<const ILogicalDevice>(device), std::move(params));
				}
			}
			else // parameter deduction failed
				return false;
		}

		if (newSwapchain)
		{
			m_swapchainResources->invalidate();
			return m_swapchainResources->onCreateSwapchain(getAssignedQueue()->getFamilyIndex(), std::move(newSwapchain));
		}
		else
			becomeIrrecoverable();

		return false;
	}

protected:
	using ISimpleManagedSurface::ISimpleManagedSurface;

	//
	inline void deinit_impl() override final
	{
		becomeIrrecoverable();
	}

	//
	inline void becomeIrrecoverable() override { m_swapchainResources = nullptr; }

	// gets called when OUT_OF_DATE upon an acquire
	inline SAcquireResult handleOutOfDate() override final
	{
		// recreate swapchain and try to acquire again
		if (recreateSwapchain(m_surfaceFormat))
			return ISimpleManagedSurface::acquireNextImage();
		return {};
	}

private:
	// Because the surface can start minimized (extent={0,0}) we might not be able to create the swapchain right away, so store creation parameters until we can create it.
	ISwapchain::SSharedCreationParams m_sharedParams = {};
	// The swapchain might not be possible to create or recreate right away, so this might be
	// either nullptr before the first successful acquire or the old to-be-retired swapchain.
	std::unique_ptr<SwapchainResources> m_swapchainResources = {};

	ISurface::SFormat m_surfaceFormat = {};
};

// In this application we'll cover buffer streaming, Buffer Device Address (BDA) and push constants 
class BDANsightThingApp final : public examples::SimpleWindowedApplication, public examples::BuiltinResourcesApplication
{
		using device_base_t = examples::SimpleWindowedApplication;
		using asset_base_t = examples::BuiltinResourcesApplication;

		constexpr static inline uint32_t WIN_W = 1280;
		constexpr static inline uint32_t WIN_H = 720;
		constexpr static inline uint32_t MaxFramesInFlight = 5;

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		BDANsightThingApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			if (!m_surface)
			{
				{
					auto windowCallback = core::make_smart_refctd_ptr<examples::CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
					ui::IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<ISimpleManagedSurface::ICallback>();
					params.width = WIN_W;
					params.height = WIN_H;
					params.x = 32;
					params.y = 32;
					params.flags = ui::IWindow::ECF_HIDDEN | ui::IWindow::ECF_BORDERLESS | ui::IWindow::ECF_RESIZABLE;
					params.windowCaption = "BDANsightThingApp";
					params.callback = windowCallback;
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}

				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<ui::IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CExplicitSurfaceFormatResizeSurface<ISimpleManagedSurface::ISwapchainResources>::create(std::move(surface));
			}

			if (m_surface)
				return { {m_surface->getSurface()/*,EQF_NONE*/} };

			return {};
		}

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			m_inputSystem = make_smart_refctd_ptr<examples::InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			ISwapchain::SCreationParams swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
			asset::E_FORMAT preferredFormats[] = { asset::EF_R8G8B8A8_UNORM };
			if (!swapchainParams.deduceFormat(m_physicalDevice, preferredFormats))
				return logFail("Could not choose a Surface Format for the Swapchain!");

			swapchainParams.sharedParams.imageUsage = IGPUImage::E_USAGE_FLAGS::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT;

			auto graphicsQueue = getGraphicsQueue();
			if (!m_surface || !m_surface->init(graphicsQueue, std::make_unique<ISimpleManagedSurface::ISwapchainResources>(), swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");

			auto pool = m_device->createCommandPool(graphicsQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

			for (auto i = 0u; i < MaxFramesInFlight; i++)
			{
				if (!pool)
					return logFail("Couldn't create Command Pool!");
				if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
					return logFail("Couldn't create Command Buffer!");
			}

			m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
			m_surface->recreateSwapchain(swapchainParams.surfaceFormat);

			// create image views for swapchain images
			for (uint32_t i = 0; i < ISwapchain::MaxImages; i++)
			{
				IGPUImage* scImg = m_surface->getSwapchainResources()->getImage(i);
				if (scImg == nullptr)
					continue;
				IGPUImageView::SCreationParams viewParams = {
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
					.image = smart_refctd_ptr<IGPUImage>(scImg),
					.viewType = IGPUImageView::ET_2D,
					.format = scImg->getCreationParameters().format
				};
				swapchainImageViews[i] = m_device->createImageView(std::move(viewParams));
			}

			// this time we load a shader directly from a file
			smart_refctd_ptr<IShader> shader;
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = "app_resources"; // virtual root

				auto key = nbl::this_example::builtin::build::get_spirv_key<"shader">(m_device.get());
				auto assetBundle = m_assetMgr->getAsset(key.data(), lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
					return logFail("Could not load shader!");

				shader = IAsset::castDown<IShader>(assets[0]);
				// The down-cast should not fail!
				assert(shader);
			}

			auto dedicatedAllocate = [&](IDeviceMemoryBacked* memBacked, const std::string_view debugName)->bool
				{
					if (!memBacked)
					{
						m_logger->log("Failed to create buffer %s", ILogger::ELL_ERROR, debugName);
						return false;
					}
					memBacked->setObjectDebugName(debugName.data());

					auto mreqs = memBacked->getMemoryReqs();
					mreqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
					using flags_e = IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS;
					core::bitflag<flags_e> flags = flags_e::EMAF_NONE;
					if (memBacked->getObjectType() == IDeviceMemoryBacked::E_OBJECT_TYPE::EOT_BUFFER &&
						static_cast<IGPUBuffer*>(memBacked)->getCreationParams().usage.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT))
						flags |= flags_e::EMAF_DEVICE_ADDRESS_BIT;
					if (!m_device->allocate(mreqs, { memBacked,flags }).isValid())
					{
						m_logger->log("Failed to allocate memory for buffer %s", ILogger::ELL_ERROR, debugName);
						return false;
					}
					return true;
				};

			const uint32_t elementCount = WIN_W * WIN_H;
			const uint32_t hashBufferElemCount = 3200000u;
			{
				IGPUBuffer::SCreationParams params = {};
				params.size = sizeof(SHashAppendData) * elementCount;
				using usage_flags_e = IGPUBuffer::E_USAGE_FLAGS;
				params.usage = usage_flags_e::EUF_STORAGE_BUFFER_BIT | usage_flags_e::EUF_SHADER_DEVICE_ADDRESS_BIT;
				hashAppendBuffer = m_device->createBuffer(std::move(params));
				if (!dedicatedAllocate(hashAppendBuffer.get(), "Hash Append Data"))
					return false;
			}
			{
				IGPUBuffer::SCreationParams params = {};
				params.size = hashBufferElemCount * sizeof(uint32_t);
				using usage_flags_e = IGPUBuffer::E_USAGE_FLAGS;
				params.usage = usage_flags_e::EUF_STORAGE_BUFFER_BIT | usage_flags_e::EUF_SHADER_DEVICE_ADDRESS_BIT;
				cellStorageBuffer = m_device->createBuffer(std::move(params));
				if (!dedicatedAllocate(cellStorageBuffer.get(), "Cell storage"))
					return false;

				params.usage |= usage_flags_e::EUF_TRANSFER_DST_BIT;
				indexBuffer = m_device->createBuffer(std::move(params));
				if (!dedicatedAllocate(indexBuffer.get(), "Index buffer"))
					return false;
			}

			// People love Reflection but I prefer Shader Sources instead!
			const nbl::asset::SPushConstantRange pcRange = {.stageFlags=IShader::E_SHADER_STAGE::ESS_COMPUTE,.offset=0,.size=sizeof(PushConstantData)};
			{
				auto layout = m_device->createPipelineLayout({&pcRange,1});
				IGPUComputePipeline::SCreationParams params = {};
				params.layout = layout.get();
				params.shader.shader = shader.get();
				params.shader.entryPoint = "main";
				if (!m_device->createComputePipelines(nullptr,{&params,1},&m_pipeline))
					return logFail("Failed to create compute pipeline!\n");
			}

			m_winMgr->show(m_window.get());

			return true;
		}

		// Ok this time we'll actually have a work loop (maybe just for the sake of future WASM so we don't timeout a Browser Tab with an unresponsive script)
		bool keepRunning() override { return true; }

		// Finally the first actual work-loop
		void workLoopBody() override
		{
			const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

			const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());

			if (m_realFrameIx >= framesInFlight)
			{
				const ISemaphore::SWaitInfo cbDonePending[] =
				{
					{
						.semaphore = m_semaphore.get(),
						.value = m_realFrameIx + 1 - framesInFlight
					}
				};
				if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

			m_currentImageAcquire = m_surface->acquireNextImage();
			if (!m_currentImageAcquire)
				return;

			auto* const cmdbuf = m_cmdBufs.data()[resourceIx].get();
			{
				cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

				cmdbuf->bindComputePipeline(m_pipeline.get());
				// This is the new fun part, pushing constants
				const PushConstantData pc = {
					.pHashAppend = hashAppendBuffer->getDeviceAddress(),
					.pCellStorage=cellStorageBuffer->getDeviceAddress(),
					.pIndex=indexBuffer->getDeviceAddress(),
					.renderSize = hlsl::uint16_t2(WIN_W, WIN_H)
				};
				cmdbuf->pushConstants(m_pipeline->getLayout(),IShader::E_SHADER_STAGE::ESS_COMPUTE,0u,sizeof(pc),&pc);
				cmdbuf->dispatch((WIN_W - 1) / WorkgroupSize + 1, (WIN_H - 1) / WorkgroupSize + 1,1);
			}


			// barrier transition to PRESENT
			{
				IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[1];
				imageBarriers[0].barrier = {
					   .dep = {
						   .srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
						   .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
						   .dstStageMask = PIPELINE_STAGE_FLAGS::NONE,
						   .dstAccessMask = ACCESS_FLAGS::NONE
						}
				};
				imageBarriers[0].image = m_surface->getSwapchainResources()->getImage(m_currentImageAcquire.imageIndex);
				imageBarriers[0].subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = 1u,
					.baseArrayLayer = 0u,
					.layerCount = 1u
				};
				imageBarriers[0].oldLayout = IImage::LAYOUT::UNDEFINED;
				imageBarriers[0].newLayout = IImage::LAYOUT::PRESENT_SRC;

				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
			}

			cmdbuf->end();

			// submit
			{
				auto* queue = getGraphicsQueue();
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
				{
					{
						.semaphore = m_semaphore.get(),
						.value = ++m_realFrameIx,
						.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
					}
				};
				{
					{
						const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
						{
							{.cmdbuf = cmdbuf }
						};

						const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] =
						{
							{
								.semaphore = m_currentImageAcquire.semaphore,
								.value = m_currentImageAcquire.acquireCount,
								.stageMask = PIPELINE_STAGE_FLAGS::NONE
							}
						};
						const IQueue::SSubmitInfo infos[] =
						{
							{
								.waitSemaphores = acquired,
								.commandBuffers = commandBuffers,
								.signalSemaphores = rendered
							}
						};

						if (queue->submit(infos) == IQueue::RESULT::SUCCESS)
						{
							const nbl::video::ISemaphore::SWaitInfo waitInfos[] =
							{ {
								.semaphore = m_semaphore.get(),
								.value = m_realFrameIx
							} };

							m_device->blockForSemaphores(waitInfos); // this is not solution, quick wa to not throw validation errors
						}
						else
							--m_realFrameIx;
					}
				}

				m_surface->present(m_currentImageAcquire.imageIndex, rendered);
			}
		}

		bool onAppTerminated() override
		{
			return device_base_t::onAppTerminated();
		}

private:
	smart_refctd_ptr<ui::IWindow> m_window;
	smart_refctd_ptr<CExplicitSurfaceFormatResizeSurface<ISimpleManagedSurface::ISwapchainResources>> m_surface;
	smart_refctd_ptr<ISemaphore> m_semaphore;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	std::array<smart_refctd_ptr<IGPUImageView>, ISwapchain::MaxImages> swapchainImageViews;

	smart_refctd_ptr<examples::InputSystem> m_inputSystem;

	smart_refctd_ptr<IGPUComputePipeline> m_pipeline;
	smart_refctd_ptr<IUtilities> m_utils;

	// recreate buffers like we have in ex 40
	smart_refctd_ptr<IGPUBuffer> hashAppendBuffer;
	smart_refctd_ptr<IGPUBuffer> indexBuffer;
	smart_refctd_ptr<IGPUBuffer> cellStorageBuffer;

	constexpr static inline uint64_t MaxIterations = 200;
};


NBL_MAIN_FUNC(BDANsightThingApp)