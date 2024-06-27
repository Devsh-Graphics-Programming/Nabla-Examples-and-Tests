// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma once

#include "nbl/video/alloc/IBufferAllocator.h"

#include <type_traits>
#include <map>

using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;


class IndexAllocator : public core::IReferenceCounted 
{
public:
	// address allocator gives offsets
	// reserved allocator allocates memory to keep the address allocator state inside
	using AddressAllocator = core::PoolAddressAllocator<uint32_t>;
	using ReservedAllocator = core::allocator<uint8_t>;
	using size_type = typename AddressAllocator::size_type;
	using value_type = typename AddressAllocator::size_type;
	static constexpr value_type invalid_value = AddressAllocator::invalid_address;

	class DeferredFreeFunctor
	{
	public:
		inline DeferredFreeFunctor(IndexAllocator* composed, size_type count, const value_type* addresses)
			: m_addresses(std::move(core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<value_type>>(count))), 
			  m_parent(composed)
		{
			memcpy(m_addresses->data(), addresses, count * sizeof(value_type));
		}
		inline DeferredFreeFunctor(DeferredFreeFunctor&& other) 
		{
			operator=(std::move(other));
		}

		//
		inline auto getWorstCaseCount() const {return m_addresses->size();}

		inline size_type operator()()
		{
			#ifdef _NBL_DEBUG
			assert(m_parent);
			#endif // _NBL_DEBUG
			m_parent->multi_deallocate(m_addresses->size(), m_addresses->data());
			m_parent->m_totalDeferredFrees -= getWorstCaseCount();

			return m_addresses->size();
		}

		DeferredFreeFunctor(const DeferredFreeFunctor& other) = delete;
		DeferredFreeFunctor& operator=(const DeferredFreeFunctor& other) = delete;
		inline DeferredFreeFunctor& operator=(DeferredFreeFunctor&& other)
		{
			m_parent = other.m_parent;
			m_addresses = other.m_addresses;

			// Nullifying other
			other.m_parent = nullptr;
			other.m_addresses = nullptr;
			return *this;
		}

		// Takes count of allocations we want to free up as reference, true is returned if
		// the amount of allocations freed was >= allocationsToFreeUp
		// False is returned if there are more allocations to free up
		inline bool operator()(size_type& allocationsToFreeUp)
		{
			size_type totalFreed = operator()();

			// This does the same logic as bool operator()(size_type&) on 
			// CAsyncSingleBufferSubAllocator
			bool freedEverything = totalFreed >= allocationsToFreeUp;
		
			if (freedEverything) allocationsToFreeUp = 0u;
			else allocationsToFreeUp -= totalFreed;
			return freedEverything;
		}
	protected:
		core::smart_refctd_dynamic_array<value_type> m_addresses;
		IndexAllocator* m_parent; // TODO: shouldn't be called `composed`, maybe `parent` or something
	};
	using EventHandler = MultiTimelineEventHandlerST<DeferredFreeFunctor>;
protected:
	std::unique_ptr<EventHandler> m_eventHandler = nullptr;
	std::unique_ptr<AddressAllocator> m_addressAllocator = nullptr;
	std::unique_ptr<ReservedAllocator> m_reservedAllocator = nullptr;
	size_t m_reservedSize = 0;
	core::smart_refctd_ptr<video::ILogicalDevice> m_logicalDevice;
	value_type m_totalDeferredFrees = 0;

	#ifdef _NBL_DEBUG
	std::recursive_mutex stAccessVerfier;
	#endif // _NBL_DEBUG

	constexpr static inline uint32_t MaxDescriptorSetAllocationAlignment = 1u; 
	constexpr static inline uint32_t MinDescriptorSetAllocationSize = 1u;

public:

	// constructors
	inline IndexAllocator(core::smart_refctd_ptr<video::ILogicalDevice>&& logicalDevice, uint32_t size)
	{
		m_reservedSize = AddressAllocator::reserved_size(MaxDescriptorSetAllocationAlignment, static_cast<size_type>(size), MinDescriptorSetAllocationSize);
		m_reservedAllocator = std::unique_ptr<ReservedAllocator>(new ReservedAllocator());
		m_addressAllocator = std::unique_ptr<AddressAllocator>(new AddressAllocator(
			m_reservedAllocator->allocate(m_reservedSize, _NBL_SIMD_ALIGNMENT),
			static_cast<size_type>(0), 0u, MaxDescriptorSetAllocationAlignment, static_cast<size_type>(size),
			MinDescriptorSetAllocationSize
		));
		m_eventHandler = std::unique_ptr<EventHandler>(new EventHandler(logicalDevice.get()));
		m_logicalDevice = std::move(logicalDevice);
	}

	inline ~IndexAllocator()
	{
		uint32_t remainingFrees;
		do {
			remainingFrees = cull_frees();
		} while (remainingFrees > 0);

		assert(m_eventHandler->getTimelines().size() == 0);
		auto ptr = reinterpret_cast<const uint8_t*>(core::address_allocator_traits<AddressAllocator>::getReservedSpacePtr(*m_addressAllocator));
		if (ptr)
			m_reservedAllocator->deallocate(const_cast<uint8_t*>(ptr), m_reservedSize);
		m_addressAllocator = nullptr;
	}

	// main methods

#ifdef _NBL_DEBUG
	inline std::unique_lock<std::recursive_mutex> stAccessVerifyDebugGuard()
	{
		std::unique_lock<std::recursive_mutex> tLock(stAccessVerfier,std::try_to_lock_t());
		assert(tLock.owns_lock());
		return tLock;
	}
#else
	inline bool stAccessVerifyDebugGuard() { return false; }
#endif

	//! Warning `outAddresses` needs to be primed with `invalid_value` values, otherwise no allocation happens for elements not equal to `invalid_value`
	inline size_type try_multi_allocate(const size_type count, value_type* outAddresses) noexcept
	{
		auto debugGuard = stAccessVerifyDebugGuard();

		size_type unallocatedSize = 0u;
		for (size_type i=0; i<count; i++)
		{
			if (outAddresses[i]!=AddressAllocator::invalid_address)
				continue;

			outAddresses[i] = m_addressAllocator->alloc_addr(1,1);
			if (outAddresses[i] == AddressAllocator::invalid_address)
			{
				unallocatedSize = count - i;
				break;
			}
		}

		return unallocatedSize;
	}

	template<class Clock=typename std::chrono::steady_clock>
	inline size_type multi_allocate(const std::chrono::time_point<Clock>& maxWaitPoint, const size_type count, value_type* outAddresses) noexcept
	{
		auto debugGuard = stAccessVerifyDebugGuard();

		// try allocate once
		size_type unallocatedSize = try_multi_allocate(count,outAddresses);
		if (!unallocatedSize)
			return 0u;

		// then try to wait at least once and allocate
		do
		{
			m_eventHandler->wait(maxWaitPoint, unallocatedSize);

			// always call with the same parameters, otherwise this turns into a mess with the non invalid_address gaps
			unallocatedSize = try_multi_allocate(count,outAddresses);
			if (!unallocatedSize)
				break;
		} while(Clock::now()<maxWaitPoint);

		return unallocatedSize;
	}

	// default timeout overload
	inline size_type multi_allocate(const size_type count, value_type* outAddresses) noexcept
	{
		// check that the binding is allocatable is done inside anyway
		return multi_allocate(TimelineEventHandlerBase::default_wait(), count, outAddresses);
	}

	// Very explicit low level call you'd need to sync and drop descriptors by yourself
	// Returns: the one-past the last `outNullify` write pointer, this allows you to work out how many descriptors were freed
	inline void multi_deallocate(uint32_t count, const size_type* addr)
	{
		auto debugGuard = stAccessVerifyDebugGuard();

		for (size_type i = 0; i < count; i++)
		{
			if (addr[i] == AddressAllocator::invalid_address)
				continue;

			m_addressAllocator->free_addr(addr[i], 1);
		}
	}

	// 100% will defer
	inline void multi_deallocate(const ISemaphore::SWaitInfo& futureWait, DeferredFreeFunctor&& functor) noexcept
	{
		auto debugGuard = stAccessVerifyDebugGuard();
		m_totalDeferredFrees += functor.getWorstCaseCount();
		m_eventHandler->latch(futureWait,std::move(functor));
	}

	// defers based on the conservative estimation if `futureWait` needs to be waited on, if doesn't will call nullify descriiptors internally immediately
	inline void multi_deallocate(size_type count, const value_type* addr, const ISemaphore::SWaitInfo& futureWait) noexcept
	{
		if (futureWait.semaphore)
			multi_deallocate(futureWait, DeferredFreeFunctor(this, count, addr));
		else
			multi_deallocate(count, addr);
	}

	//! Returns free events still outstanding
	inline uint32_t cull_frees() noexcept
	{
		auto debugGuard = stAccessVerifyDebugGuard();
		uint32_t frees = m_eventHandler->poll().eventsLeft;
		return frees;
	}
};

