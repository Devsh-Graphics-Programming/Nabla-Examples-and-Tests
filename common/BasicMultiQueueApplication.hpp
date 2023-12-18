// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_BASIC_MULTI_QUEUE_APPLICATION_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_BASIC_MULTI_QUEUE_APPLICATION_HPP_INCLUDED_

// Build on top of the previous one
#include "../common/MonoDeviceApplication.hpp"

namespace nbl::examples
{

// Virtual Inheritance because apps might end up doing diamond inheritance
class BasicMultiQueueApplication : public virtual MonoDeviceApplication
{
		using base_t = MonoDeviceApplication;

	public:
		using base_t::base_t;

		// So now we need to return "threadsafe" queues because the queues might get aliased and also used on multiple threads
		virtual video::CThreadSafeGPUQueueAdapter* getComputeQueue() const
		{
			return m_device->getThreadSafeQueue(m_computeQueue.famIx,m_computeQueue.qIx);
		}
		virtual video::CThreadSafeGPUQueueAdapter* getGraphicsQueue() const
		{
			if (m_graphicsQueue.famIx!=QueueAllocator::InvalidIndex)
				return m_device->getThreadSafeQueue(m_graphicsQueue.famIx,m_graphicsQueue.qIx);
			assert(isHeadlessCompute());
			return nullptr;
		}

		// virtual to allow aliasing and total flexibility, as with the above
		virtual video::CThreadSafeGPUQueueAdapter* getTransferUpQueue() const
		{
			return m_device->getThreadSafeQueue(m_transferUpQueue.famIx,m_transferUpQueue.qIx);
		}
		virtual video::CThreadSafeGPUQueueAdapter* getTransferDownQueue() const
		{
			return m_device->getThreadSafeQueue(m_transferDownQueue.famIx,m_transferDownQueue.qIx);
		}

	protected:
		// This time we build upon the Mono-System and Mono-Logger application and add the creation of possibly multiple queues and creation of IUtilities
		virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			if (!base_t::onAppInitialized(std::move(system)))
				return false;
			
			using namespace core;
			m_utils = make_smart_refctd_ptr<video::IUtilities>(smart_refctd_ptr(m_device),smart_refctd_ptr(m_logger));
			if (!m_utils)
				return logFail("Failed to create nbl::video::IUtilities!");

			return true;
		}

		// overridable for future graphics queue using examples
		virtual bool isHeadlessCompute() const {return true;}

		using queue_flags_t = video::IPhysicalDevice::E_QUEUE_FLAGS;
		// So because of lovely Intel GPUs that only have one queue, we can't really request anything different
		virtual core::vector<queue_req_t> getQueueRequirements() const override
		{
			queue_req_t singleQueueReq = {.requiredFlags=queue_flags_t::EQF_COMPUTE_BIT|queue_flags_t::EQF_TRANSFER_BIT,.disallowedFlags=queue_flags_t::EQF_NONE,.queueCount=1,.maxImageTransferGranularity={1,1,1}};
			if (!isHeadlessCompute())
				singleQueueReq.requiredFlags |= queue_flags_t::EQF_GRAPHICS_BIT;
			return {singleQueueReq};
		}

		// Their allocation and creation gets more complex
		class QueueAllocator final
		{
			public:
				QueueAllocator() = default;
				QueueAllocator(const queue_family_range_t& familyProperties) : m_remainingQueueCounts(familyProperties.size()), m_familyProperties(familyProperties)
				{
					for (uint8_t i=0u; i<m_familyProperties.size(); i++)
						m_remainingQueueCounts[i] = m_familyProperties[i].queueCount;
				}

				constexpr static inline uint8_t InvalidIndex = 0xff;

				// A little utility to try-allocate N queues from the same (but yet unknown) family
				// the unwantedFlags are listed in order of severity, most unwanted first
				uint8_t allocateFamily(const queue_req_t& originalReq, std::initializer_list<queue_flags_t> unwantedFlags)
				{
					for (size_t flagCount=unwantedFlags.size(); flagCount; flagCount--)
					{
						queue_req_t req = originalReq;
						for (auto it=unwantedFlags.begin(); it!=(unwantedFlags.begin()+flagCount); it++)
							req.disallowedFlags |= *it;
						for (uint8_t i=0u; i<m_familyProperties.size(); i++)
						{
							if (req.familyMatches(m_familyProperties[i]) && m_remainingQueueCounts[i]>=req.queueCount)
							{
								m_remainingQueueCounts[i] -= req.queueCount;
								return i;
							}
						}
						// log compromises?
					}
					return InvalidIndex;
				}

				// to allow try-allocs
				inline void freeQueues(const uint8_t famIx, const uint8_t count)
				{
					assert(famIx<m_remainingQueueCounts.size());
					assert(count<=m_familyProperties[famIx].queueCount);
					m_remainingQueueCounts[famIx] += count;
				}

			private:
				core::vector<uint8_t> m_remainingQueueCounts;
				const queue_family_range_t& m_familyProperties;
		};

		virtual core::vector<video::ILogicalDevice::SQueueCreationParams> getQueueCreationParameters(const queue_family_range_t& familyProperties) override
		{
			QueueAllocator queueAllocator(familyProperties);

			// First thing to make sure we have is a compute queue (so nothing else fails allocation) which should be able to do image transfers of any granularity (transfer only queue families can have problems with that)
			queue_req_t computeQueueRequirement = {.requiredFlags=queue_flags_t::EQF_COMPUTE_BIT,.disallowedFlags=queue_flags_t::EQF_NONE,.queueCount=1,.maxImageTransferGranularity={1,1,1}};
			m_computeQueue.famIx = queueAllocator.allocateFamily(computeQueueRequirement,{queue_flags_t::EQF_GRAPHICS_BIT,queue_flags_t::EQF_TRANSFER_BIT,queue_flags_t::EQF_SPARSE_BINDING_BIT,queue_flags_t::EQF_PROTECTED_BIT});
			// since we requested a device that has a compute capable queue family (unless `getQueueRequirements` got overriden) we're sure we'll get at least one family capable of compute
			assert(m_computeQueue.famIx!=QueueAllocator::InvalidIndex);

			// We'll try to allocate the transfer queues from families that support the least extra bits (most importantly not graphics and not compute)
			{
				constexpr queue_req_t TransferQueueRequirement = {.requiredFlags=queue_flags_t::EQF_TRANSFER_BIT,.disallowedFlags=queue_flags_t::EQF_NONE,.queueCount=1};
				// We'll first try to look for queue family which has a transfer capability but no Graphics or Compute capability, to ensure we're running on some neat DMA engines and not clogging up the main CP
				m_transferUpQueue.famIx = queueAllocator.allocateFamily(TransferQueueRequirement,{queue_flags_t::EQF_GRAPHICS_BIT,queue_flags_t::EQF_COMPUTE_BIT,queue_flags_t::EQF_SPARSE_BINDING_BIT,queue_flags_t::EQF_PROTECTED_BIT});
				// In my opinion the Asynchronicity of the Upload queue is more important, so we assigned that first.
				// We don't need to do anything special to ensure the down transfer queue allocates on the same family as the up transfer queue
				m_transferDownQueue.famIx = queueAllocator.allocateFamily(TransferQueueRequirement,{queue_flags_t::EQF_GRAPHICS_BIT,queue_flags_t::EQF_COMPUTE_BIT,queue_flags_t::EQF_SPARSE_BINDING_BIT,queue_flags_t::EQF_PROTECTED_BIT});
			}
			// If our allocator worked properly, then whatever we've managed to allocate is allocated on a family that supports it and preferably with as few extra caps as it could.
			// Then whatever allocations we've failed could not have been allocated as separate queues and nothing will change that (like backing down on the unwanted bits).

			// This is a sort-of allocator of queue indices for distinct queues
			core::map<uint8_t,uint8_t> familyQueueCounts;

			// Failed to allocate up-transfer queue, then alias it to the compute queue
			if (m_transferUpQueue.famIx==QueueAllocator::InvalidIndex)
			{
				m_logger->log("Not enough queue counts in families, had to alias the Transfer-Up Queue to Compute!",system::ILogger::ELL_PERFORMANCE);
				// but first deallocate the original compute queue
				queueAllocator.freeQueues(m_computeQueue.famIx,1);
				// and allocate again with requirement that compute queue must also support transfer (this might force a change of family)
				computeQueueRequirement.requiredFlags |= queue_flags_t::EQF_TRANSFER_BIT;
				m_computeQueue.famIx = queueAllocator.allocateFamily(computeQueueRequirement,{queue_flags_t::EQF_GRAPHICS_BIT,queue_flags_t::EQF_SPARSE_BINDING_BIT,queue_flags_t::EQF_PROTECTED_BIT});
				// assign queue index within family now
				m_computeQueue.qIx = familyQueueCounts[m_computeQueue.famIx]++;
				// now alias the queue
				m_transferUpQueue = m_computeQueue;
			}
			else // otherwise assign queue indices
			{
				m_computeQueue.qIx = familyQueueCounts[m_computeQueue.famIx]++;
				m_transferUpQueue.qIx = familyQueueCounts[m_transferUpQueue.famIx]++;
			}
			// since we assign first, the compute queue should have the first index within the family
			assert(m_computeQueue.qIx==0);

			// Failed to allocate down-transfer queue, then alias it to the up-transfer
			if (m_transferDownQueue.famIx==QueueAllocator::InvalidIndex)
			{
				m_logger->log("Not enough queue counts in families, had to alias the Transfer-Up Queue to Transfer-Down!",system::ILogger::ELL_PERFORMANCE);
				m_transferDownQueue.famIx = m_transferUpQueue.famIx;
			}
			else
				m_transferDownQueue.qIx = familyQueueCounts[m_transferDownQueue.famIx]++;

			// now after assigning all queues to families and indices, collate the creation parameters
			core::vector<video::ILogicalDevice::SQueueCreationParams> retval(familyQueueCounts.size());
			auto oit = retval.begin();
			for (auto it=familyQueueCounts.begin(); it!=familyQueueCounts.end(); it++,oit++)
			{
				oit->familyIndex = it->first;
				oit->count = it->second;
			}
			return retval;
		}


		core::smart_refctd_ptr<video::IUtilities> m_utils;

	private:
		struct SQueueIndex
		{
			uint8_t famIx=QueueAllocator::InvalidIndex;
			uint8_t qIx=0;
		};
		SQueueIndex m_graphicsQueue={},m_computeQueue={},m_transferUpQueue={},m_transferDownQueue={};
};

}

#endif // _CAMERA_IMPL_