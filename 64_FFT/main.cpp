// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"


using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"


// In this application we'll cover buffer streaming, Buffer Device Address (BDA) and push constants 
class StreamingAndBufferDeviceAddressApp final : public application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

	smart_refctd_ptr<IGPUComputePipeline> m_pipeline;

	smart_refctd_ptr<nbl::video::IUtilities> m_utils;

	nbl::video::StreamingTransientDataBufferMT<>* m_upStreamingBuffer;
	StreamingTransientDataBufferMT<>* m_downStreamingBuffer;
	smart_refctd_ptr<nbl::video::IGPUBuffer> m_deviceLocalBuffer;

	// These are Buffer Device Addresses
	uint64_t m_upStreamingBufferAddress;
	uint64_t m_downStreamingBufferAddress;
	uint64_t m_deviceLocalBufferAddress;

	// You can ask the `nbl::core::GeneralpurposeAddressAllocator` used internally by the Streaming Buffers give out offsets aligned to a certain multiple (not only Power of Two!)
	uint32_t m_alignment;

	// The pool cache is just a formalized way of round-robining command pools and resetting + reusing them after their most recent submit signals finished.
	// Its a little more ergonomic to use if you don't have a 1:1 mapping between frames and pools.
	smart_refctd_ptr<nbl::video::ICommandPoolCache> m_poolCache;

	// This example really lets the advantages of a timeline semaphore shine through!
	smart_refctd_ptr<ISemaphore> m_timeline;
	uint64_t m_iteration = 0;
	constexpr static inline uint64_t MaxIterations = 1;

public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	StreamingAndBufferDeviceAddressApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	// we stuff all our work here because its a "single shot" app
	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		// this time we load a shader directly from a file
		smart_refctd_ptr<IGPUShader> shader;
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto assetBundle = m_assetMgr->getAsset("app_resources/shader.comp.hlsl", lp);
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
		constexpr uint32_t DownstreamBufferSize = sizeof(output_t) << 23;
		constexpr uint32_t UpstreamBufferSize = sizeof(input_t) << 23;

		m_utils = make_smart_refctd_ptr<IUtilities>(smart_refctd_ptr(m_device), smart_refctd_ptr(m_logger), DownstreamBufferSize, UpstreamBufferSize);
		if (!m_utils)
			return logFail("Failed to create Utilities!");
		m_upStreamingBuffer = m_utils->getDefaultUpStreamingBuffer();
		m_downStreamingBuffer = m_utils->getDefaultDownStreamingBuffer();
		m_upStreamingBufferAddress = m_upStreamingBuffer->getBuffer()->getDeviceAddress();
		m_downStreamingBufferAddress = m_downStreamingBuffer->getBuffer()->getDeviceAddress();

		// Create device-local buffer
		
		{
			const uint32_t scalarElementCount = 2 * complexElementCount;
			IGPUBuffer::SCreationParams deviceLocalBufferParams = {};
			
			IQueue* const queue = getComputeQueue();
			uint32_t queueFamilyIndex = queue->getFamilyIndex();
			
			deviceLocalBufferParams.queueFamilyIndexCount = 1;
			deviceLocalBufferParams.queueFamilyIndices = &queueFamilyIndex;
			deviceLocalBufferParams.size = sizeof(input_t) * scalarElementCount;
			deviceLocalBufferParams.usage = nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT | nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;
			
			m_deviceLocalBuffer = m_device->createBuffer(std::move(deviceLocalBufferParams));
			auto mreqs = m_deviceLocalBuffer->getMemoryReqs();
			mreqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto gpubufMem = m_device->allocate(mreqs, m_deviceLocalBuffer.get(), IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_DEVICE_ADDRESS_BIT);

			m_deviceLocalBufferAddress = m_deviceLocalBuffer.get()->getDeviceAddress();
		}
		

		// People love Reflection but I prefer Shader Sources instead!
		const nbl::asset::SPushConstantRange pcRange = { .stageFlags = IShader::ESS_COMPUTE,.offset = 0,.size = sizeof(PushConstantData) };

		{
			auto layout = m_device->createPipelineLayout({ &pcRange,1 });
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = layout.get();
			params.shader.shader = shader.get();
			params.shader.requireFullSubgroups = true;
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_pipeline))
				return logFail("Failed to create compute pipeline!\n");
		}

		const auto& deviceLimits = m_device->getPhysicalDevice()->getLimits();
		// The ranges of non-coherent mapped memory you flush or invalidate need to be aligned. You'll often see a value of 64 reported by devices
		// which just happens to coincide with a CPU cache line size. So we ask our streaming buffers during allocation to give us properly aligned offsets.
		// Sidenote: For SSBOs, UBOs, BufferViews, Vertex Buffer Bindings, Acceleration Structure BDAs, Shader Binding Tables, Descriptor Buffers, etc.
		// there is also a requirement to bind buffers at offsets which have a certain alignment. Memory binding to Buffers and Images also has those.
		// We'll align to max of coherent atom size even if the memory is coherent,
		// and we also need to take into account BDA shader loads need to be aligned to the type being loaded.
		m_alignment = core::max(deviceLimits.nonCoherentAtomSize, alignof(float));

		// We'll allow subsequent iterations to overlap each other on the GPU, the only limiting factors are
		// the amount of memory in the streaming buffers and the number of commandpools we can use simultaenously.
		constexpr auto MaxConcurrency = 64;

		// Since this time we don't throw the Command Pools away and we'll reset them instead, we don't create the pools with the transient flag
		m_poolCache = ICommandPoolCache::create(core::smart_refctd_ptr(m_device), getComputeQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::NONE, MaxConcurrency);

		// In contrast to fences, we just need one semaphore to rule all dispatches
		m_timeline = m_device->createSemaphore(m_iteration);

		return true;
	}

	// Ok this time we'll actually have a work loop (maybe just for the sake of future WASM so we don't timeout a Browser Tab with an unresponsive script)
	bool keepRunning() override { return m_iteration < MaxIterations; }

	// Finally the first actual work-loop
	void workLoopBody() override
	{
		IQueue* const queue = getComputeQueue();

		// Note that I'm using the sample struct with methods that have identical code which compiles as both C++ and HLSL
		auto rng = nbl::hlsl::Xoroshiro64StarStar::construct({ m_iteration ^ 0xdeadbeefu,std::hash<string>()(_NBL_APP_NAME_) });

		const uint32_t scalarElementCount = 2 * complexElementCount;
		const uint32_t inputSize = sizeof(input_t) * scalarElementCount;

		// The allocators can do multiple allocations at once for efficiency
		const uint32_t AllocationCount = 1;

		// It comes with a certain drawback that you need to remember to initialize your "yet unallocated" offsets to the Invalid value
		// this is to allow a set of allocations to fail, and you to re-try after doing something to free up space without repacking args.
		auto inputOffset = m_upStreamingBuffer->invalid_value;

		// We always just wait till an allocation becomes possible (during allocation previous "latched" frees get their latch conditions polled)
		// Freeing of Streaming Buffer Allocations can and should be deferred until an associated polled event signals done (more on that later).
		std::chrono::steady_clock::time_point waitTill(std::chrono::years(45));
		// note that the API takes a time-point not a duration, because there are multiple waits and preemptions possible, so the durations wouldn't add up properly
		m_upStreamingBuffer->multi_allocate(waitTill, AllocationCount, &inputOffset, &inputSize, &m_alignment);

		// Generate our data in-place on the allocated staging buffer
		{	
			auto* const inputPtr = reinterpret_cast<input_t*>(reinterpret_cast<uint8_t*>(m_upStreamingBuffer->getBufferPointer()) + inputOffset);
			std::cout << "Begin array CPU\n";
			for (auto j = 0; j < complexElementCount; j++)
			{
				//Random array
				/*
				float x = rng() / float(nbl::hlsl::numeric_limits<decltype(rng())>::max), y = rng() / float(nbl::hlsl::numeric_limits<decltype(rng())>::max);
				*/
				// FFT( (1,0), (0,0), (0,0),... ) = (1,0), (1,0), (1,0),...
				
				float x = j > 0 ? 0.f : 1.f;
				float y = 0;
				
				// FFT( (c,0), (c,0), (c,0),... ) = (Nc,0), (0,0), (0,0),...
				/*
				float x = 2.f;
				float y = 0.f;
				*/
				inputPtr[2 * j] = x;
				inputPtr[2 * j + 1] = y;
				std::cout << "(" << x << ", " << y << "), ";
			}
			std::cout << "\nEnd array CPU\n";
			// Always remember to flush!
			if (m_upStreamingBuffer->needsManualFlushOrInvalidate())
			{
				const auto bound = m_upStreamingBuffer->getBuffer()->getBoundMemory();
				const ILogicalDevice::MappedMemoryRange range(bound.memory, bound.offset + inputOffset, inputSize);
				m_device->flushMappedMemoryRanges(1, &range);
			}
		}

		// Obtain our command pool once one gets recycled
		uint32_t poolIx;
		do
		{
			poolIx = m_poolCache->acquirePool();
		} while (poolIx == ICommandPoolCache::invalid_index);

		// finally allocate our output range
		const uint32_t outputSize = inputSize;

		auto outputOffset = m_downStreamingBuffer->invalid_value;
		m_downStreamingBuffer->multi_allocate(waitTill, AllocationCount, &outputOffset, &outputSize, &m_alignment);

		smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
		{
			m_poolCache->getPool(poolIx)->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmdbuf,1 }, core::smart_refctd_ptr(m_logger));
			// lets record, its still a one time submit because we have to re-record with different push constants each time
			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->bindComputePipeline(m_pipeline.get());
			// This is the new fun part, pushing constants
			const PushConstantData pc = {
				.inputAddress = m_deviceLocalBufferAddress,
				.outputAddress = m_deviceLocalBufferAddress,
				.dataElementCount = scalarElementCount
			};
			IGPUCommandBuffer::SBufferCopy copyInfo = {};
			copyInfo.srcOffset = 0;
			copyInfo.dstOffset = 0;
			copyInfo.size = m_deviceLocalBuffer->getSize();
			cmdbuf->copyBuffer(m_upStreamingBuffer->getBuffer(), m_deviceLocalBuffer.get(), 1, &copyInfo);
			cmdbuf->pushConstants(m_pipeline->getLayout(), IShader::ESS_COMPUTE, 0u, sizeof(pc), &pc);
			// Good old trick to get rounded up divisions, in case you're not familiar
			cmdbuf->dispatch(1, 1, 1);

			// Pipeline barrier: wait for FFT shader to be done before copying to downstream buffer 
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo pipelineBarrierInfo = {};
			decltype(pipelineBarrierInfo)::buffer_barrier_t barrier = {};
			pipelineBarrierInfo.bufBarriers = {&barrier, 1u};

			barrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			barrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
			barrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
			barrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS;

			cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), pipelineBarrierInfo);
			cmdbuf->copyBuffer(m_deviceLocalBuffer.get(), m_downStreamingBuffer->getBuffer(), 1, &copyInfo);
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

		// We let all latches know what semaphore and counter value has to be passed for the functors to execute
		const ISemaphore::SWaitInfo futureWait = { m_timeline.get(),m_iteration };

		// We can also actually latch our Command Pool reset and its return to the pool of free pools!
		m_poolCache->releasePool(futureWait, poolIx);

		// As promised, we can defer an upstreaming buffer deallocation until a fence is signalled
		// You can also attach an additional optional IReferenceCounted derived object to hold onto until deallocation.
		m_upStreamingBuffer->multi_deallocate(AllocationCount, &inputOffset, &inputSize, futureWait);

		// Now a new and even more advanced usage of the latched events, we make our own refcounted object with a custom destructor and latch that like we did the commandbuffer.
		// Instead of making our own and duplicating logic, we'll use one from IUtilities meant for down-staging memory.
		// Its nice because it will also remember to invalidate our memory mapping if its not coherent.
		auto latchedConsumer = make_smart_refctd_ptr<IUtilities::CDownstreamingDataConsumer>(
			IDeviceMemoryAllocation::MemoryRange(outputOffset, outputSize),
			// Note the use of capture by-value [=] and not by-reference [&] because this lambda will be called asynchronously whenever the event signals
			[=](const size_t dstOffset, const void* bufSrc, const size_t size)->void
			{
				// The unused variable is used for letting the consumer know the subsection of the output we've managed to download
				// But here we're sure we can get the whole thing in one go because we allocated the whole range ourselves.
				assert(dstOffset == 0 && size == outputSize);

				std::cout << "Begin array GPU\n";
				output_t* const data = reinterpret_cast<output_t*>(const_cast<void*>(bufSrc));
				for (auto i = 0u; i < complexElementCount; i++) {
					std::cout << "(" << data[2 * i] << ", " << data[2 * i + 1] << "), ";
				}

				std::cout << "\nEnd array GPU\n";
			},
			// Its also necessary to hold onto the commandbuffer, even though we take care to not reset the parent pool, because if it
			// hits its destructor, our automated reference counting will drop all references to objects used in the recorded commands.
			// It could also be latched in the upstreaming deallocate, because its the same fence.
			std::move(cmdbuf), m_downStreamingBuffer
		);
		// We put a function we want to execute 
		m_downStreamingBuffer->multi_deallocate(AllocationCount, &outputOffset, &outputSize, futureWait, &latchedConsumer.get());
	}

	bool onAppTerminated() override
	{
		// Need to make sure that there are no events outstanding if we want all lambdas to eventually execute before `onAppTerminated`
		// (the destructors of the Command Pool Cache and Streaming buffers will still wait for all lambda events to drain)
		while (m_downStreamingBuffer->cull_frees()) {}
		return device_base_t::onAppTerminated();
	}
};


NBL_MAIN_FUNC(StreamingAndBufferDeviceAddressApp)