#pragma once
using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;

using image_id = uint64_t; // Could later be templated or replaced with a stronger type or hash key.

enum class ImageType : uint8_t
{
    STATIC = 0,                        // Regular non-georeferenced image, fully loaded once
    GEOREFERENCED_STREAMED,            // Streamed image, resolution depends on camera/view
    GEOREFERENCED_FULL_RESOLUTION      // For smaller georeferenced images, entire image is eventually loaded and not streamed or view-dependant
};

struct GeoreferencedImageParams
{
	uint32_t2 imageExtents;
	uint32_t2 viewportExtents;
	asset::E_FORMAT format;
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

// This will be dropped when the descriptor gets dropped from SuballocatedDescriptorSet.
// Destructor will then deallocate from GeneralPurposeAllocator, making the previously allocated range of the image available/free again.
struct ImageCleanup : public core::IReferenceCounted
{
	ImageCleanup()
		: imagesMemorySuballocator(nullptr)
		, addr(ImagesMemorySubAllocator::InvalidAddress)
		, size(0ull)
	{}

	~ImageCleanup() override
	{
		// printf(std::format("Actual Eviction size={}, offset={} \n", size, addr).c_str());
		if (imagesMemorySuballocator && addr != ImagesMemorySubAllocator::InvalidAddress)
			imagesMemorySuballocator->deallocate(addr, size);
	}

	smart_refctd_ptr<ImagesMemorySubAllocator> imagesMemorySuballocator;
	uint64_t addr;
	uint64_t size;

};

struct StaticImageCopy
{
	core::smart_refctd_ptr<ICPUImage> cpuImage;
	core::smart_refctd_ptr<IGPUImageView> gpuImageView;
	uint32_t arrayIndex;
};

// TODO: consider just using the ImagesUsageCache to store this StaticImagesState, i.e. merge this struct with the ImageReference
//		it will be possible after LRUCache improvements and copyability
//		for now this will be a mirror of the LRUCache but in an unordered_map
struct StaticImageState
{
	core::smart_refctd_ptr<ICPUImage> cpuImage = nullptr;
	core::smart_refctd_ptr<IGPUImageView> gpuImageView = nullptr;
	uint64_t allocationOffset = ImagesMemorySubAllocator::InvalidAddress;
	uint64_t allocationSize = 0u;
	uint32_t arrayIndex = ~0u; // in texture array descriptor 
	bool gpuResident = false;
};


struct ImageReference
{
	static constexpr uint32_t InvalidTextureIndex = nbl::hlsl::numeric_limits<uint32_t>::max;
	
	uint32_t arrayIndex = InvalidTextureIndex; // index in our array of textures binding
	ImageType imageType;
	bool gpuResident = false;
	uint64_t lastUsedFrameIndex = 0ull; // last used semaphore value on this image
	uint64_t allocationOffset = ImagesMemorySubAllocator::InvalidAddress;
	uint64_t allocationSize = 0ull;
	core::smart_refctd_ptr<IGPUImageView> gpuImageView = nullptr;

	ImageReference() 
		: arrayIndex(InvalidTextureIndex)
		, lastUsedFrameIndex(0ull)
		, allocationOffset(ImagesMemorySubAllocator::InvalidAddress)
		, allocationSize(0ull)
	{}
	
	// In LRU Cache `insert` function, in case of cache miss, we need to construct the refereence with semaphore value
	ImageReference(uint64_t currentFrameIndex) 
		: arrayIndex(InvalidTextureIndex)
		, lastUsedFrameIndex(currentFrameIndex)
		, allocationOffset(ImagesMemorySubAllocator::InvalidAddress)
		, allocationSize(0ull)
	{}

	// In LRU Cache `insert` function, in case of cache hit, we need to assign semaphore value without changing `index`
	inline ImageReference& operator=(uint64_t currentFrameIndex) { lastUsedFrameIndex = currentFrameIndex; return *this;  }
};

// A resource-aware image cache with an LRU eviction policy.
// This cache tracks image usage by ID and provides hooks for eviction logic, such as releasing descriptor slots and deallocating GPU memory.
// Currently, eviction is purely LRU-based. In the future, eviction decisions may incorporate additional factors:
//   - memory usage per image.
//   - lastUsedFrameIndex.
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
	template<std::invocable<image_id, const ImageReference&> EvictionCallback>
	inline ImageReference* insert(image_id imageID, uint64_t lastUsedSema, EvictionCallback&& evictCallback)
	{
		auto lruEvictionCallback = [&](const ImageReference& evicted)
			{
				const image_id* evictingKey = lruCache.get_least_recently_used();
				assert(evictingKey != nullptr);
				if (evictingKey)
					evictCallback(*evictingKey, evicted);
			};
		return lruCache.insert(imageID, lastUsedSema, lruEvictionCallback);
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
			return ~0ull;
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
