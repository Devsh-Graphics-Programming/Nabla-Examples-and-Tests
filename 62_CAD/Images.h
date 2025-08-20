#pragma once
using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;

using image_id = uint64_t; // Could later be templated or replaced with a stronger type or hash key.

enum class ImageState : uint8_t
{
	INVALID = 0,
	CREATED_AND_MEMORY_BOUND,             // GPU image created, not bound to descriptor set yet
	BOUND_TO_DESCRIPTOR_SET,              // Bound to descriptor set, GPU resident, but may contain uninitialized or partial data
	GPU_RESIDENT_WITH_VALID_STATIC_DATA,  // When data for static images gets issued for upload successfully
};

enum class ImageType : uint8_t
{
	INVALID = 0,
	STATIC,                        // Regular non-georeferenced image, fully loaded once
	GEOREFERENCED_STREAMED,            // Streamed image, resolution depends on camera/view
	GEOREFERENCED_FULL_RESOLUTION      // For smaller georeferenced images, entire image is eventually loaded and not streamed or view-dependant
};

struct GeoreferencedImageParams
{
	OrientedBoundingBox2D worldspaceOBB = {};
	uint32_t2 imageExtents = {};
	uint32_t2 viewportExtents = {};
	asset::E_FORMAT format = {};
	std::filesystem::path storagePath = {};
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

// Measured in tile coordinates in the image that the range spans, and the mip level the tiles correspond to
struct GeoreferencedImageTileRange
{
	uint32_t2 topLeft;
	uint32_t2 bottomRight;
	uint32_t baseMipLevel;
};

// @brief Used to load tiles into VRAM, keep track of loaded tiles, determine how they get sampled etc.
struct GeoreferencedImageStreamingState : public IReferenceCounted
{
	friend class DrawResourcesFiller;

protected:
	static smart_refctd_ptr<GeoreferencedImageStreamingState> create(GeoreferencedImageParams&& _georeferencedImageParams)
	{
		smart_refctd_ptr<GeoreferencedImageStreamingState> retVal(new GeoreferencedImageStreamingState{});
		retVal->georeferencedImageParams = std::move(_georeferencedImageParams);
		//	1. Get the displacement (will be an offset vector in world coords and world units) from the `topLeft` corner of the image to the point
		//	2. Transform this displacement vector into the coordinates in the basis {dirU, dirV} (worldspace vectors that span the sides of the image).
		//	The composition of these matrices there fore transforms any point in worldspace into uv coordinates in imagespace


		// 1. Displacement. The following matrix computes the offset for an input point `p` with homogenous worldspace coordinates. 
		//    By foregoing the homogenous coordinate we can keep only the vector part, that's why it's `2x3` and not `3x3`
		float64_t2 topLeftWorld = retVal->georeferencedImageParams.worldspaceOBB.topLeft;
		float64_t2x3 displacementMatrix(1., 0., - topLeftWorld.x, 0., 1., - topLeftWorld.y);

		// 2. Change of Basis. Since {dirU, dirV} are orthogonal, the matrix to change from world coords to `span{dirU, dirV}` coords has a quite nice expression
		//    Non-uniform scaling doesn't affect this, but this has to change if we allow for shearing (basis vectors stop being orthogonal)
		float64_t2 dirU = retVal->georeferencedImageParams.worldspaceOBB.dirU;
		float64_t2 dirV = float32_t2(dirU.y, -dirU.x) * retVal->georeferencedImageParams.worldspaceOBB.aspectRatio;
		float64_t dirULengthSquared = nbl::hlsl::dot(dirU, dirU);
		float64_t dirVLengthSquared = nbl::hlsl::dot(dirV, dirV);
		float64_t2 firstRow = dirU / dirULengthSquared;
		float64_t2 secondRow = dirV / dirVLengthSquared;
		float64_t2x2 changeOfBasisMatrix(firstRow, secondRow);

		// Put them all together
		retVal->world2UV = nbl::hlsl::mul(changeOfBasisMatrix, displacementMatrix);
		return retVal;
	}

	GeoreferencedImageParams georeferencedImageParams = {};
	std::vector<std::vector<bool>> currentMappedRegionOccupancy = {};

	// These are NOT UV, pixel or tile coords into the mapped image region, rather into the real, huge image
	// Tile coords are always in mip 0 tile size. Translating to other mips levels is trivial
	float64_t2 transformWorldCoordsToUV(const float64_t3 worldCoordsH) const { return nbl::hlsl::mul(world2UV, worldCoordsH); }
	float64_t2 transformWorldCoordsToPixelCoords(const float64_t3 worldCoordsH) const { return float64_t2(georeferencedImageParams.imageExtents) * transformWorldCoordsToUV(worldCoordsH); }
	float64_t2 transformWorldCoordsToTileCoords(const float64_t3 worldCoordsH, const uint32_t TileSize) const { return (1.0 / TileSize) * transformWorldCoordsToPixelCoords(worldCoordsH); }

	// When the current mapped region is inadequate to fit the viewport, we compute a new mapped region
	void remapCurrentRegion(const GeoreferencedImageTileRange& viewportTileRange);

	// Sidelength of the gpu image, in tiles that are `GeoreferencedImageTileSize` pixels wide
	uint32_t maxResidentTiles = {};
	// Size of the image (minus 1), in tiles of `GeoreferencedImageTileSize` sidelength
	uint32_t2 maxImageTileIndices = {};
	// Set topLeft to extreme value so it gets recreated on first iteration
	GeoreferencedImageTileRange currentMappedRegion = { .topLeft = uint32_t2(std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()) };
	// Converts a point (z = 1) in worldspace to UV coordinates in image space (origin shifted to topleft of the image)
	float64_t2x3 world2UV = {};
};

struct CachedImageRecord
{
	static constexpr uint32_t InvalidTextureIndex = nbl::hlsl::numeric_limits<uint32_t>::max;
	
	uint32_t arrayIndex = InvalidTextureIndex; // index in our array of textures binding
	ImageType type = ImageType::INVALID;
	ImageState state = ImageState::INVALID;
	uint64_t lastUsedFrameIndex = 0ull; // last used semaphore value on this image
	uint64_t allocationOffset = ImagesMemorySubAllocator::InvalidAddress;
	uint64_t allocationSize = 0ull;
	core::smart_refctd_ptr<IGPUImageView> gpuImageView = nullptr;
	core::smart_refctd_ptr<ICPUImage> staticCPUImage = nullptr; // cached cpu image for uploading to gpuImageView when needed.
	core::smart_refctd_ptr<GeoreferencedImageStreamingState> georeferencedImageState = nullptr; // Used to track tile residency for georeferenced images
	
	// In LRU Cache `insert` function, in case of cache miss, we need to construct the refereence with semaphore value
	CachedImageRecord(uint64_t currentFrameIndex) 
		: arrayIndex(InvalidTextureIndex)
		, type(ImageType::INVALID)
		, state(ImageState::INVALID)
		, lastUsedFrameIndex(currentFrameIndex)
		, allocationOffset(ImagesMemorySubAllocator::InvalidAddress)
		, allocationSize(0ull)
		, gpuImageView(nullptr)
		, staticCPUImage(nullptr)
	{}
	
	CachedImageRecord() 
		: CachedImageRecord(0ull)
	{}

	// In LRU Cache `insert` function, in case of cache hit, we need to assign semaphore value without changing `index`
	inline CachedImageRecord& operator=(uint64_t currentFrameIndex) { lastUsedFrameIndex = currentFrameIndex; return *this;  }
};

// A resource-aware image cache with an LRU eviction policy.
// This cache tracks image usage by ID and provides hooks for eviction logic (such as releasing descriptor slots and deallocating GPU memory done by user of this class)
// Currently, eviction is purely LRU-based. In the future, eviction decisions may incorporate additional factors:
//   - memory usage per image.
//   - lastUsedFrameIndex.
// This class helps coordinate images' lifetimes in sync with GPU usage via eviction callbacks.
class ImagesCache : public core::ResizableLRUCache<image_id, CachedImageRecord>
{
public:
	using base_t = core::ResizableLRUCache<image_id, CachedImageRecord>;
	
	ImagesCache(size_t capacity) 
		: base_t(capacity)
	{}

	// Attempts to insert a new image into the cache.
	// If the cache is full, invokes the provided `evictCallback` to evict an image.
	// Returns a pointer to the inserted or existing ImageReference.
	template<std::invocable<image_id, const CachedImageRecord&> EvictionCallback>
	inline CachedImageRecord* insert(image_id imageID, uint64_t lastUsedSema, EvictionCallback&& evictCallback)
	{
		return base_t::insert(imageID, lastUsedSema, evictCallback);
	}
	
	// Retrieves the image associated with `imageID`, updating its LRU position.
	inline CachedImageRecord* get(image_id imageID)
	{
		return base_t::get(imageID);
	}
	
	// Retrieves the ImageReference without updating LRU order.
	inline CachedImageRecord* peek(image_id imageID)
	{
		return base_t::peek(imageID);
	}

	inline size_t size() const { return base_t::size(); }
	
	// Selects an eviction candidate based on LRU policy.
	// In the future, this could factor in memory pressure or semaphore sync requirements.
	inline image_id select_eviction_candidate() 
	{
		const image_id* lru = base_t::get_least_recently_used();
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
		base_t::erase(imageID);
	}
};

struct StreamedImageCopy
{
	asset::E_FORMAT srcFormat;
	smart_refctd_ptr<ICPUBuffer> srcBuffer; // Make it 'std::future' later?
	asset::IImage::SBufferCopy region;
};

// TODO: Rename to StaticImageAvailabilityRequest?
struct StaticImageInfo
{
	image_id imageID = ~0ull;
	core::smart_refctd_ptr<ICPUImage> cpuImage = nullptr;
	bool forceUpdate = false; // If true, bypasses the existing GPU-side cache and forces an update of the image data; Useful when replacing the contents of a static image that may already be resident.
	asset::E_FORMAT imageViewFormatOverride = asset::E_FORMAT::EF_COUNT; // if asset::E_FORMAT::EF_COUNT then image view will have the same format as `cpuImage`
};
