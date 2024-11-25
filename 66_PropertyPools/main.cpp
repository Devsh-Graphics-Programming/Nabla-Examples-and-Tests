// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/video/alloc/SubAllocatedDescriptorSet.h"

#include "../common/BasicMultiQueueApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace ui;
using namespace asset;
using namespace video;

#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"

// In this application we'll cover buffer streaming, Buffer Device Address (BDA) and push constants 
class PropertyPoolsApp final : public examples::MonoDeviceApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::MonoDeviceApplication;
		using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

		smart_refctd_ptr<CPropertyPoolHandler> m_propertyPoolHandler;
		smart_refctd_ptr<IGPUBuffer> m_scratchBuffer;
		smart_refctd_ptr<IGPUBuffer> m_addressBuffer;
		smart_refctd_ptr<IGPUBuffer> m_transferSrcBuffer;
		smart_refctd_ptr<IGPUBuffer> m_transferDstBuffer;
		std::vector<uint16_t> m_data;
		
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
			using nbl::video::IGPUDescriptorSetLayout;

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

			m_scratchBuffer = createBuffer(sizeof(nbl::hlsl::property_pools::TransferRequest) * TransfersAmount, core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT), "m_scratchBuffer", true);
			m_addressBuffer = createBuffer(sizeof(uint32_t) * TransfersAmount * MaxValuesPerTransfer, core::bitflag(asset::IBuffer::EUF_NONE), "m_addressBuffer", false);
			m_transferSrcBuffer = createBuffer(sizeof(uint16_t) * TransfersAmount * MaxValuesPerTransfer, core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT), "m_transferSrcBuffer", false);
			m_transferDstBuffer = createBuffer(sizeof(uint16_t) * TransfersAmount * MaxValuesPerTransfer, core::bitflag(asset::IBuffer::EUF_NONE), "m_transferDstBuffer", true);

			for (uint16_t i = 0; i < uint16_t((uint32_t(1) << 16) - 1); i++)
				m_data.push_back(i);

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

				// COMMAND RECORDING
				uint32_t dataSize = (((sizeof(uint16_t) * m_data.size()) + 3) / 4) * 4;
				uint32_t maxUpload = 65536;
				for (uint32_t offset = 0; offset < dataSize; offset += maxUpload)
				{
					cmdbuf->updateBuffer({ offset, maxUpload, core::smart_refctd_ptr<video::IGPUBuffer>(m_transferSrcBuffer) }, &m_data[offset / sizeof(uint16_t)]);
				}
				CPropertyPoolHandler::TransferRequest transferRequest;
				transferRequest.memblock = asset::SBufferRange<video::IGPUBuffer> { 0, sizeof(uint16_t) * m_data.size(), core::smart_refctd_ptr<video::IGPUBuffer>(m_transferSrcBuffer) };
				transferRequest.elementSize = 1;
				transferRequest.elementCount = (m_data.size() * sizeof(uint16_t)) / sizeof(uint32_t);
				transferRequest.buffer = asset::SBufferBinding<video::IGPUBuffer> { 0, core::smart_refctd_ptr<video::IGPUBuffer>(m_transferDstBuffer) };
				transferRequest.srcAddressesOffset = IPropertyPool::invalid;
				transferRequest.dstAddressesOffset = IPropertyPool::invalid;

				m_propertyPoolHandler->transferProperties(cmdbuf.get(),
					asset::SBufferBinding<video::IGPUBuffer>{0, core::smart_refctd_ptr(m_scratchBuffer)}, 
					asset::SBufferBinding<video::IGPUBuffer>{0, core::smart_refctd_ptr(m_addressBuffer)}, 
					&transferRequest, &transferRequest + 1,
					m_logger.get(), 0, m_data.size()
					);

				auto result = cmdbuf->end();
				assert(result);
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
				auto statusCode = queue->submit({ &submitInfo,1 });
				queue->endCapture();
				assert(statusCode == IQueue::RESULT::SUCCESS);
			}

			{
				ISemaphore::SWaitInfo infos[1] = {{.semaphore=m_timeline.get(),.value=m_iteration}};
				m_device->blockForSemaphores(infos);

				// Readback ds
				// (we'll read back the destination buffer and check that copy went through as expected)
				auto mem = m_transferDstBuffer->getBoundMemory(); // Scratch buffer has the transfer requests
				void* ptr = mem.memory->map({ mem.offset, mem.memory->getAllocationSize() });

				for (uint32_t i = 0; i < 1024; /*m_data.size();*/ i++)
				{
					uint16_t expected = reinterpret_cast<uint16_t*>(ptr)[i];
					uint16_t actual = m_data[i];
					std::printf("%i, ", expected);
					assert(expected == actual);
				}
				std::printf("\n");
				bool success = mem.memory->unmap();
				assert(success);
			}
		}
};

NBL_MAIN_FUNC(PropertyPoolsApp)