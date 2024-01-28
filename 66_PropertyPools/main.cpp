// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/video/surface/CSurfaceVulkan.h"

#include "../common/BasicMultiQueueApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

namespace nbl::examples
{

using namespace nbl;
using namespace core;
using namespace system;
using namespace ui;
using namespace asset;
using namespace video;

// Virtual Inheritance because apps might end up doing diamond inheritance
class WindowedApplication : public virtual BasicMultiQueueApplication
{
		using base_t = BasicMultiQueueApplication;

	public:
		using base_t::base_t;

		virtual video::IAPIConnection::SFeatures getAPIFeaturesToEnable() override
		{
			auto retval = base_t::getAPIFeaturesToEnable();
			// We only support one swapchain mode, surface, the other one is Display which we have not implemented yet.
			retval.swapchainMode = video::E_SWAPCHAIN_MODE::ESM_SURFACE;
			return retval;
		}

		// New function, we neeed to know about surfaces to create ahead of time
		virtual core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const = 0;

		virtual core::set<video::IPhysicalDevice*> filterDevices(const core::SRange<video::IPhysicalDevice* const>& physicalDevices) const
		{
			const auto firstFilter = base_t::filterDevices(physicalDevices);

			video::SPhysicalDeviceFilter deviceFilter = {};
			
			const auto surfaces = getSurfaces();
			deviceFilter.requiredSurfaceCompatibilities = { surfaces.data(), surfaces.size() };

			return deviceFilter(physicalDevices);
		}
		
		virtual bool onAppInitialized(smart_refctd_ptr<ISystem>&& system)
		{
			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

		#ifdef _NBL_PLATFORM_WINDOWS_
			m_winMgr = nbl::ui::IWindowManagerWin32::create();
		#else
			#error "Unimplemented!"
		#endif
		}

		core::smart_refctd_ptr<ui::IWindowManager> m_winMgr;
};


// Before we get onto creating a window, we need to discuss how Nabla handles input, clipboards and cursor control
class IWindowClosedCallback : public virtual nbl::ui::IWindow::IEventCallback
{
	public:
		IWindowClosedCallback() : m_gotWindowClosedMsg(false) {}

		// unless you create a separate callback per window, both will "trip" this condition
		bool windowGotClosed() const {return m_gotWindowClosedMsg;}

	private:
		bool onWindowClosed_impl() override
		{
			m_gotWindowClosedMsg = true;
			return true;
		}

		bool m_gotWindowClosedMsg;
};

// We inherit from an application that tries to find Graphics and Compute queues
// because applications with presentable images often want to perform Graphics family operations
// Virtual Inheritance because apps might end up doing diamond inheritance
class SingleNonResizableWindowApplication : public virtual WindowedApplication
{
		using base_t = WindowedApplication;

	protected:
		virtual IWindow::SCreationParams getWindowCreationParams() const
		{
			IWindow::SCreationParams params = {};
			params.callback = make_smart_refctd_ptr<IWindowClosedCallback>();
			params.width = 640;
			params.height = 480;
			params.x = 32;
			params.y = 32;
			params.flags = IWindow::ECF_NONE;
			params.windowCaption = "SingleNonResizableWindowApplication";
			return params;
		}

		core::smart_refctd_ptr<ui::IWindow> m_window;
		core::smart_refctd_ptr<video::ISurfaceVulkan> m_surface;

	public:
		using base_t::base_t;

		virtual bool onAppInitialized(smart_refctd_ptr<nbl::system::ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			m_window = m_winMgr->createWindow(getWindowCreationParams());
			m_surface = video::CSurfaceVulkanWin32::create(core::smart_refctd_ptr(m_api),core::smart_refctd_ptr_static_cast<ui::IWindowWin32>(m_window));
			return true;
		}

		virtual core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const
		{
			return {{m_surface.get()/*,EQF_NONE*/}};
		}

		virtual bool keepRunning() override
		{
			if (!m_window || reinterpret_cast<const IWindowClosedCallback*>(m_window->getEventCallback())->windowGotClosed())
				return false;

			return true;
		}
};
}


using namespace nbl;
using namespace core;
using namespace system;
using namespace ui;
using namespace asset;
using namespace video;


#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"


// In this application we'll cover buffer streaming, Buffer Device Address (BDA) and push constants 
class PropertyPoolsApp final : public examples::SingleNonResizableWindowApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::MonoDeviceApplication;
		using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

		// This is the first example that submits multiple workloads in-flight. 
		// What the shader does is it computes the minimum distance of each point against K other random input points.
		// Having the GPU randomly access parts of the buffer requires it to be DEVICE_LOCAL for performance.
		// Then the CPU downloads the results and finds the median minimum distance via quick-select.
		// This bizzare synthetic workload was specifically chosen for its unfriendliness towards simple buffer usage.
		// The fact we have variable sized workloads and run them in a loop means we either have to dynamically
		// suballocate from a single buffer or have K worst-case sized buffers we round robin for K-workloads in flight.
		// Creating and destroying buffers at runtime is not an option as those are very expensive operations. 
		// Also since CPU needs to heapify the outputs, we need to have the GPU write them into RAM not VRAM.
		smart_refctd_ptr<IGPUComputePipeline> m_pipeline;

		// The Utility class has lots of methods to handle staging without relying on ReBAR or EXT_host_image_copy as well as more complex methods we'll cover later.
		// Until EXT_host_image_copy becomes ubiquitous across all Nabla Core Profile devices, you need to stage image copies from an IGPUBuffer to an IGPUImage.
		// Why use Staging for buffers in the age of ReBAR? While GPU workloads overlap the CPU, individual GPU workloads's execution might not overlap each other
		// but their data might. In this case you want to "precisely" time the data update on the GPU timeline between the end and start of a workload.
		// For very small updates you could use the commandbuffer updateBuffer method, but it has a size limit and the data enqueued takes up space in the commandpool.
		// Sometimes it might be unfeasible to either have multiple copies or update references to those copies without a cascade update.
		// One example is the transformation graph of nodes in a scene, where a copy-on-write of a node would require the update the offset/pointer held by
		// any other node that refers to it. This quickly turns into a cascade that would force you to basically create a full copy of the entire data structure
		// after most updates. Whereas with staging you'd "queue up" the much smaller set of updates to apply between each computation step which uses the graph.
		// Another example are UBO and SSBO bindings, where once you run out of dynamic bindings, you can no longer easily change offsets without introducting extra indirection in shaders.
		// Actually staging can help you re-use a commandbuffer because you don't need to re-record it if you don't need to change the offsets at which you bind!
		// Finally ReBAR is a precious resource, my 8GB RTX 3070 only reports a 214MB Heap backing HOST_VISIBLE and DEVICE_LOCAL device local memory type.
		smart_refctd_ptr<nbl::video::IUtilities> m_utils;

		// We call them downstreaming and upstreaming, simply by how we used them so far.
		// Meaning that upstreaming is uncached and usually ReBAR (DEVICE_LOCAL), for simple memcpy like sequential writes.
		// While the downstreaming is CACHED and not DEVICE_LOCAL for fast random acess by the CPU.
		// However there are cases when you'd want to use a buffer with flags identical to the default downstreaming buffer for uploads,
		// such cases is when a CPU needs to build a data-structure in-place (due to memory constraints) before GPU accesses it,
		// one example are Host Acceleration Structure builds (BVH building requires lots of repeated memory accesses).
		// When choosing the memory properties of a mapped buffer consider which processor (CPU or GPU) needs faster access in event of a cache-miss.
		nbl::video::StreamingTransientDataBufferMT<>* m_upStreamingBuffer;
		StreamingTransientDataBufferMT<>* m_downStreamingBuffer;
		// These are Buffer Device Addresses
		uint64_t m_upStreamingBufferAddress;
		uint64_t m_downStreamingBufferAddress;

		smart_refctd_ptr<CPropertyPoolHandler> m_propertyPoolHandler;
		smart_refctd_ptr<IGPUBuffer> m_scratchBuffer;
		smart_refctd_ptr<IGPUBuffer> m_addressBuffer;
		smart_refctd_ptr<IGPUBuffer> m_transferSrcBuffer;
		smart_refctd_ptr<IGPUBuffer> m_transferDstBuffer;
		std::vector<uint16_t> m_data;

		// You can ask the `nbl::core::GeneralpurposeAddressAllocator` used internally by the Streaming Buffers give out offsets aligned to a certain multiple (not only Power of Two!)
		uint32_t m_alignment;
		
		// The pool cache is just a formalized way of round-robining command pools and resetting + reusing them after their most recent submit signals finished.
		// Its a little more ergonomic to use if you don't have a 1:1 mapping between frames and pools.
		smart_refctd_ptr<nbl::video::ICommandPoolCache> m_poolCache;

		// This example really lets the advantages of a timeline semaphore shine through!
		smart_refctd_ptr<ISemaphore> m_timeline;
		uint64_t m_iteration = 0;
		constexpr static inline uint64_t MaxIterations = 200;

		static constexpr uint64_t TransfersAmount = 1024;
		static constexpr uint64_t MaxValuesPerTransfer = 512;

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		PropertyPoolsApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(std::move(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			m_propertyPoolHandler = core::make_smart_refctd_ptr<CPropertyPoolHandler>(core::smart_refctd_ptr(m_device));

			auto createBuffer = [&](uint64_t size, core::bitflag<asset::IBuffer::E_USAGE_FLAGS> flags, const char* name, bool hostVisible)
			{
				video::IGPUBuffer::SCreationParams creationParams;
				creationParams.size = ((size + 3) / 4) * 4; // Align
				creationParams.usage = flags
					| asset::IBuffer::EUF_STORAGE_BUFFER_BIT
					| asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT 
					| asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;

				auto buffer = m_device->createBuffer(std::move(creationParams));
				nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = buffer->getMemoryReqs();
				if (hostVisible) 
					reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDownStreamingMemoryTypeBits();
				m_device->allocate(reqs, buffer.get(), nbl::video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_DEVICE_ADDRESS_BIT);
				buffer->setObjectDebugName(name);

				return buffer;
			};

			m_scratchBuffer = createBuffer(sizeof(nbl::hlsl::property_pools::TransferRequest) * TransfersAmount, core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT), "m_scratchBuffer", false);
			m_addressBuffer = createBuffer(sizeof(uint32_t) * TransfersAmount * MaxValuesPerTransfer, core::bitflag(asset::IBuffer::EUF_NONE), "m_addressBuffer", false);
			m_transferSrcBuffer = createBuffer(sizeof(uint16_t) * TransfersAmount * MaxValuesPerTransfer, core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT), "m_transferSrcBuffer", false);
			m_transferDstBuffer = createBuffer(sizeof(uint16_t) * TransfersAmount * MaxValuesPerTransfer, core::bitflag(asset::IBuffer::EUF_NONE), "m_transferDstBuffer", true);

			for (uint16_t i = 0; i < uint16_t((uint32_t(1) << 16) - 1); i++)
				m_data.push_back(i);

			// this time we load a shader directly from a file
			smart_refctd_ptr<IGPUShader> shader;
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = ""; // virtual root
				auto assetBundle = m_assetMgr->getAsset("app_resources/shader.comp.hlsl",lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
					return logFail("Could not load shader!");

				// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
				auto source = IAsset::castDown<ICPUShader>(assets[0]);
				// The down-cast should not fail!
				assert(source);

				// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
				shader = m_device->createShader(source.get());
				if (!shader)
					return logFail("Creation of a GPU Shader to from CPU Shader source failed!");
			}

			// The StreamingTransientDataBuffers are actually composed on top of another useful utility called `CAsyncSingleBufferSubAllocator`
			// The difference is that the streaming ones are made on top of ranges of `IGPUBuffer`s backed by mappable memory, whereas the
			// `CAsyncSingleBufferSubAllocator` just allows you suballocate subranges of any `IGPUBuffer` range with deferred/latched frees.
			constexpr uint32_t DownstreamBufferSize = sizeof(output_t)<<24;
			constexpr uint32_t UpstreamBufferSize = sizeof(input_t)<<24;
			m_utils = make_smart_refctd_ptr<IUtilities>(smart_refctd_ptr(m_device),smart_refctd_ptr(m_logger),DownstreamBufferSize,UpstreamBufferSize);
			if (!m_utils)
				return logFail("Failed to create Utilities!");
			m_upStreamingBuffer = m_utils->getDefaultUpStreamingBuffer();
			m_downStreamingBuffer = m_utils->getDefaultDownStreamingBuffer();
			m_upStreamingBufferAddress = m_upStreamingBuffer->getBuffer()->getDeviceAddress();
			m_downStreamingBufferAddress = m_downStreamingBuffer->getBuffer()->getDeviceAddress();

			// People love Reflection but I prefer Shader Sources instead!
			const nbl::asset::SPushConstantRange pcRange = {.stageFlags=IShader::ESS_COMPUTE,.offset=0,.size=sizeof(PushConstantData)};

			// This time we'll have no Descriptor Sets or Layouts because our workload has a widely varying size
			// and using traditional SSBO bindings would force us to update the Descriptor Set every frame.
			// I even started writing this sample with the use of Dynamic SSBOs, however the length of the buffer range is not dynamic
			// only the offset. This means that we'd have to write the "worst case" length into the descriptor set binding.
			// Then this has a knock-on effect that we couldn't allocate closer to the end of the streaming buffer than the "worst case" size.
			{
				auto layout = m_device->createPipelineLayout({&pcRange,1});
				IGPUComputePipeline::SCreationParams params = {};
				params.layout = layout.get();
				params.shader.shader = shader.get();
				if (!m_device->createComputePipelines(nullptr,{&params,1},&m_pipeline))
					return logFail("Failed to create compute pipeline!\n");
			}

			const auto& deviceLimits = m_device->getPhysicalDevice()->getLimits();
			// The ranges of non-coherent mapped memory you flush or invalidate need to be aligned. You'll often see a value of 64 reported by devices
			// which just happens to coincide with a CPU cache line size. So we ask our streaming buffers during allocation to give us properly aligned offsets.
			// Sidenote: For SSBOs, UBOs, BufferViews, Vertex Buffer Bindings, Acceleration Structure BDAs, Shader Binding Tables, Descriptor Buffers, etc.
			// there is also a requirement to bind buffers at offsets which have a certain alignment. Memory binding to Buffers and Images also has those.
			// We'll align to max of coherent atom size even if the memory is coherent,
			// and we also need to take into account BDA shader loads need to be aligned to the type being loaded.
			m_alignment = core::max(deviceLimits.nonCoherentAtomSize,alignof(float));

			// We'll allow subsequent iterations to overlap each other on the GPU, the only limiting factors are
			// the amount of memory in the streaming buffers and the number of commandpools we can use simultaenously.
			constexpr auto MaxConcurrency = 64;

			// Since this time we don't throw the Command Pools away and we'll reset them instead, we don't create the pools with the transient flag
			m_poolCache = ICommandPoolCache::create(core::smart_refctd_ptr(m_device),getComputeQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::NONE,MaxConcurrency);

			// In contrast to fences, we just need one semaphore to rule all dispatches
			m_timeline = m_device->createSemaphore(m_iteration);
			return true;
		}

		// Ok this time we'll actually have a work loop (maybe just for the sake of future WASM so we don't timeout a Browser Tab with an unresponsive script)
		bool keepRunning() override { return m_iteration<MaxIterations; }

		// Finally the first actual work-loop
		void workLoopBody() override
		{
			IQueue* const queue = getComputeQueue();

			// Obtain our command pool once one gets recycled
			uint32_t poolIx;
			do
			{
				poolIx = m_poolCache->acquirePool();
			} while (poolIx==ICommandPoolCache::invalid_index);

			smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
			{
				m_poolCache->getPool(poolIx)->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&cmdbuf,1},core::smart_refctd_ptr(m_logger));
				// lets record, its still a one time submit because we have to re-record with different push constants each time
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				cmdbuf->bindComputePipeline(m_pipeline.get());

				// COMMAND RECORDING
				uint32_t dataSize = (((sizeof(uint16_t) * m_data.size()) + 3) / 4) * 4;
				uint32_t maxUpload = 65536;
				for (uint32_t offset = 0; offset < dataSize; offset += maxUpload)
				{
					cmdbuf->updateBuffer({ offset, maxUpload, core::smart_refctd_ptr<video::IGPUBuffer>(m_transferSrcBuffer) }, &m_data[offset / sizeof(uint16_t)]);
				}
				CPropertyPoolHandler::TransferRequest transferRequest;
				transferRequest.memblock = asset::SBufferRange<video::IGPUBuffer> { 0, sizeof(uint16_t) * m_data.size(), core::smart_refctd_ptr<video::IGPUBuffer>(m_transferSrcBuffer) };
				transferRequest.elementSize = m_data.size();
				transferRequest.elementCount = 1;
				transferRequest.buffer = asset::SBufferBinding<video::IGPUBuffer> { 0, core::smart_refctd_ptr<video::IGPUBuffer>(m_transferDstBuffer) };

				m_propertyPoolHandler->transferProperties(cmdbuf.get(),
					asset::SBufferBinding<video::IGPUBuffer>{0, core::smart_refctd_ptr(m_scratchBuffer)}, 
					asset::SBufferBinding<video::IGPUBuffer>{0, core::smart_refctd_ptr(m_addressBuffer)}, 
					&transferRequest, &transferRequest + 1,
					m_logger.get(), 0, MaxValuesPerTransfer
					);

				cmdbuf->end();
			}


			const auto savedIterNum = m_iteration++;
			{
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo =
				{
					.cmdbuf = cmdbuf.get()
				};
				const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo =
				{
					.semaphore = m_timeline.get(),
					.value = m_iteration,
					.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
				};
				// Generally speaking we don't need to wait on any semaphore because in this example every dispatch gets its own clean piece of memory to use
				// from the point of view of the GPU. Implicit domain operations between Host and Device happen upon a submit and a semaphore/fence signal operation,
				// this ensures we can touch the input and get accurate values from the output memory using the CPU before and after respectively, each submit becoming PENDING.
				// If we actually cared about this submit seeing the memory accesses of a previous dispatch we could add a semaphore wait
				const IQueue::SSubmitInfo submitInfo = {
					.waitSemaphores = {},
					.commandBuffers = {&cmdbufInfo,1},
					.signalSemaphores = {&signalInfo,1}
				};

				queue->startCapture();
				queue->submit({ &submitInfo,1 });
				queue->endCapture();
			}

			{
				// Readback ds
				auto mem = m_transferDstBuffer->getBoundMemory();
				void* ptr = mem.memory->map({ mem.offset, mem.memory->getAllocationSize() });

				auto uint16_t_ptr = reinterpret_cast<uint16_t*>(ptr);

				for (uint32_t i = 0; i < 128; i++)
				{
					uint16_t value = uint16_t_ptr[i];
					std::printf("%i, ", value);
				}
				std::printf("\n");
				bool success = mem.memory->unmap();
				assert(success);
			}
		}

		bool onAppTerminated() override
		{
			// Need to make sure that there are no events outstanding if we want all lambdas to eventually execute before `onAppTerminated`
			// (the destructors of the Command Pool Cache and Streaming buffers will still wait for all lambda events to drain)
			while (m_downStreamingBuffer->cull_frees()) {}

			return device_base_t::onAppTerminated();
		}
};


NBL_MAIN_FUNC(PropertyPoolsApp)