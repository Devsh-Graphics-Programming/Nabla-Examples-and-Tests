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

#include "nbl/builtin/hlsl/bit.hlsl"

// In this application we'll cover buffer streaming, Buffer Device Address (BDA) and push constants 
class SubAllocatedDescriptorSetApp final : public examples::MonoDeviceApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::MonoDeviceApplication;
		using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

		// The pool cache is just a formalized way of round-robining command pools and resetting + reusing them after their most recent submit signals finished.
		// Its a little more ergonomic to use if you don't have a 1:1 mapping between frames and pools.
		smart_refctd_ptr<nbl::video::ICommandPoolCache> m_poolCache;

		smart_refctd_ptr<nbl::video::SubAllocatedDescriptorSet> m_subAllocDescriptorSet;

		// This example really lets the advantages of a timeline semaphore shine through!
		smart_refctd_ptr<ISemaphore> m_timeline;
		uint64_t m_iteration = 0;
		constexpr static inline uint64_t MaxIterations = 200;

		constexpr static inline uint32_t MaxDescriptorSetAllocationAlignment = 64u*1024u; // if you need larger alignments then you're not right in the head
		constexpr static inline uint32_t MinDescriptorSetAllocationSize = 1u;

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		SubAllocatedDescriptorSetApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
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


			// We'll allow subsequent iterations to overlap each other on the GPU, the only limiting factors are
			// the amount of memory in the streaming buffers and the number of commandpools we can use simultaenously.
			constexpr auto MaxConcurrency = 64;

			// Since this time we don't throw the Command Pools away and we'll reset them instead, we don't create the pools with the transient flag
			m_poolCache = ICommandPoolCache::create(core::smart_refctd_ptr(m_device),getComputeQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::NONE,MaxConcurrency);

			// In contrast to fences, we just need one semaphore to rule all dispatches
			m_timeline = m_device->createSemaphore(m_iteration);

			// Descriptor set sub allocator

			video::IGPUDescriptorSetLayout::SBinding bindings[12];
			{
				for (uint32_t i = 0; i < 12; i++)
				{
					bindings[i].binding = i;
					bindings[i].count = 16000;
					bindings[i].createFlags = core::bitflag(IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT) 
						| IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT 
						| IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_PARTIALLY_BOUND_BIT;
					if (i % 2 == 0) bindings[i].type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
					else if (i % 2 == 1) bindings[i].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
					bindings[i].stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE;
				}
			}

			std::span<video::IGPUDescriptorSetLayout::SBinding> bindingsSpan(bindings);

			auto descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);

			// TODO: I don't think these are needed for sub allocated descriptor sets (alignment isn't needed, and min size is 1)
			auto subAllocatedDescriptorSet = core::make_smart_refctd_ptr<nbl::video::SubAllocatedDescriptorSet>(
				descriptorSetLayout.get(), MaxDescriptorSetAllocationAlignment, MinDescriptorSetAllocationSize
			);

			std::vector<uint32_t> allocation(128, core::PoolAddressAllocator<uint32_t>::invalid_address);
			{
				subAllocatedDescriptorSet->multi_allocate(0, allocation.size(), &allocation[0]);
				for (uint32_t i = 0; i < allocation.size(); i++)
				{
					m_logger->log("allocation[%d]: %d", system::ILogger::ELL_INFO, i, allocation[i]);
					assert(allocation[i] != core::PoolAddressAllocator<uint32_t>::invalid_address);
				}
			}
			{
				std::vector<uint32_t> addr;
				for (uint32_t i = 0; i < allocation.size(); i+=2)
				{
					addr.push_back(allocation[i]);
				}
				subAllocatedDescriptorSet->multi_deallocate(0, addr.size(), &addr[0]);
			}
			m_logger->log("freed half the descriptors", system::ILogger::ELL_INFO);
			std::vector<uint32_t> allocation2(128, core::PoolAddressAllocator<uint32_t>::invalid_address);
			{
				subAllocatedDescriptorSet->multi_allocate(0, allocation2.size(), &allocation2[0]);
				for (uint32_t i = 0; i < allocation2.size(); i++)
				{
					m_logger->log("allocation[%d]: %d", system::ILogger::ELL_INFO, i, allocation2[i]);
					assert(allocation2[i] != core::PoolAddressAllocator<uint32_t>::invalid_address);
				}
			}
			
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
		}
};

NBL_MAIN_FUNC(SubAllocatedDescriptorSetApp)