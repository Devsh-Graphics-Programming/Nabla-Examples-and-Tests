#pragma once
using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;

using image_id = uint64_t; // Could later be templated or replaced with a stronger type or hash key.
	
struct ImageReference
{
	static constexpr uint32_t InvalidTextureIndex = nbl::hlsl::numeric_limits<uint32_t>::max;
	uint32_t index = InvalidTextureIndex; // index in our array of textures binding
	uint64_t lastUsedSemaphoreValue = 0ull; // last used semaphore value on this image
	uint64_t memoryUsage = 0ull; // TODO: to be considered later

	ImageReference() 
		: index(InvalidTextureIndex)
		, lastUsedSemaphoreValue(0ull)
		, memoryUsage(0ull)
	{}
	
	// In LRU Cache `insert` function, in case of cache miss, we need to construct the refereence with semaphore value
	ImageReference(uint64_t semamphoreVal) 
		: index(InvalidTextureIndex)
		, lastUsedSemaphoreValue(semamphoreVal)
		, memoryUsage(0ull)
	{}

	// In LRU Cache `insert` function, in case of cache hit, we need to assign semaphore value without changing `index`
	inline ImageReference& operator=(uint64_t semamphoreVal) { lastUsedSemaphoreValue = semamphoreVal; return *this;  }
};

// A resource-aware image cache with an LRU eviction policy.
// This cache tracks image usage by ID and provides hooks for eviction logic, such as releasing descriptor slots and deallocating GPU memory.
// Currently, eviction is purely LRU-based. In the future, eviction decisions may incorporate additional factors:
//   - memory usage per image.
//   - lastUsedSemaphoreValue.
// This class does not own GPU resources directly, but helps coordinate their lifetimes in sync with GPU usage via eviction callbacks.
class ImagesUsageCache
{
public:
	ImagesUsageCache(size_t capacity) 
		: lruCache(ImagesLRUCache(capacity))
	{}

	// Attempts to insert a new image into the cache.
	// If the cache is full, invokes the provided `evictCallback` to evict an image.
	// Returns a pointer to the inserted or existing ImageReference.
	template<std::invocable<const ImageReference&> EvictionCallback>
	inline ImageReference* insert(image_id imageID, uint64_t lastUsedSema, EvictionCallback&& evictCallback)
	{
		return lruCache.insert(imageID, lastUsedSema, std::forward<EvictionCallback>(evictCallback));
	}
	
	// Retrieves the image associated with `imageID`, updating its LRU position.
	inline ImageReference* get(image_id imageID)
	{
		return lruCache.get(imageID);
	}
	
	// Retrieves the ImageReference without updating LRU order.
	inline ImageReference* peek(image_id imageID)
	{
		return lruCache.peek(imageID);
	}

	inline size_t size() const { return lruCache.size(); }
	
	// Selects an eviction candidate based on LRU policy.
	// In the future, this could factor in memory pressure or semaphore sync requirements.
	inline image_id select_eviction_candidate() 
	{
		const image_id* lru = lruCache.get_least_recently_used();
		if (lru)
			return *lru;
		else
		{
			// we shouldn't select eviction candidate if lruCache is empty
			_NBL_DEBUG_BREAK_IF(true);
			return 0ull;
		}
	}
	
	// Removes a specific image from the cache (manual eviction).
	inline void erase(image_id imageID)
	{
		lruCache.erase(imageID);
	}

private:
	using ImagesLRUCache = core::ResizableLRUCache<image_id, ImageReference>;
	ImagesLRUCache lruCache; // TODO: for now, work with simple lru cache, later on consider resource usage along with lastUsedSema value
};

/**
 * @class ImagesMemorySubAllocator
 * @brief A memory sub-allocator designed for managing sub-allocations within a pre-allocated GPU memory arena for images.
 * 
 * This class wraps around `nbl::core::GeneralpurposeAddressAllocator` to provide offset-based memory allocation
 * for image resources within a contiguous block of GPU memory.
 *
 * @note This class only manages address offsets. The actual memory must be bound separately.
 */
class ImagesMemorySubAllocator : public core::IReferenceCounted 
{
public:
	using AddressAllocator = nbl::core::GeneralpurposeAddressAllocator<uint64_t>;
	using ReservedAllocator = nbl::core::allocator<uint8_t>;
	static constexpr uint64_t InvalidAddress = AddressAllocator::invalid_address;
	static constexpr uint64_t MaxMemoryAlignment = 4096u; // safe choice based on hardware reports
	static constexpr uint64_t MinAllocSize = 128 * 1024u; // 128KB, the larger this is the better

	ImagesMemorySubAllocator(uint64_t memoryArenaSize)
	{
		m_reservedAllocSize = AddressAllocator::reserved_size(MaxMemoryAlignment, memoryArenaSize, MinAllocSize);
		m_reservedAllocator = std::unique_ptr<ReservedAllocator>(new ReservedAllocator());
		m_reservedAlloc = m_reservedAllocator->allocate(m_reservedAllocSize, _NBL_SIMD_ALIGNMENT);
		m_addressAllocator = std::unique_ptr<AddressAllocator>(new AddressAllocator(
			m_reservedAlloc, 0u, 0u, MaxMemoryAlignment, memoryArenaSize, MinAllocSize
		));
	}

	// return offset, will return InvalidAddress if failed
	uint64_t allocate(uint64_t size, uint64_t alignment)
	{
		return m_addressAllocator->alloc_addr(size, alignment);
	}

	void deallocate(uint64_t addr, uint64_t size)
	{
		m_addressAllocator->free_addr(addr, size);
	}

	~ImagesMemorySubAllocator()
	{
		if (m_reservedAlloc)
			m_reservedAllocator->deallocate(reinterpret_cast<uint8_t*>(m_reservedAlloc), m_reservedAllocSize);
	}
	
private:
	std::unique_ptr<AddressAllocator> m_addressAllocator = nullptr;

	// Memory Allocation Required for the AddressAllocator
	std::unique_ptr<ReservedAllocator> m_reservedAllocator = nullptr;
	void* m_reservedAlloc = nullptr;
	size_t m_reservedAllocSize = 0;

};