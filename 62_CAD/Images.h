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
		//	The composition of these matrices therefore transforms any point in worldspace into uv coordinates in imagespace
		//  To reduce code complexity, instead of computing the product of these matrices, since the first is a pure displacement matrix 
		//  (non-homogenous 2x2 upper left is identity matrix) and the other is a pure rotation matrix (2x2) we can just put them together
		//  by putting the rotation in the upper left 2x2 of the result and the post-rotated displacement in the upper right 2x1.
		//  The result is also 2x3 and not 3x3 because we can drop he homogenous since the displacement yields a vector

		// 2. Change of Basis. Since {dirU, dirV} are orthogonal, the matrix to change from world coords to `span{dirU, dirV}` coords has a quite nice expression
		//    Non-uniform scaling doesn't affect this, but this has to change if we allow for shearing (basis vectors stop being orthogonal)
		const float64_t2 dirU = retVal->georeferencedImageParams.worldspaceOBB.dirU;
		const float64_t2 dirV = float32_t2(dirU.y, -dirU.x) * retVal->georeferencedImageParams.worldspaceOBB.aspectRatio;
		const float64_t dirULengthSquared = nbl::hlsl::dot(dirU, dirU);
		const float64_t dirVLengthSquared = nbl::hlsl::dot(dirV, dirV);
		const float64_t2 firstRow = dirU / dirULengthSquared;
		const float64_t2 secondRow = dirV / dirVLengthSquared;

		const float64_t2 displacement = - retVal->georeferencedImageParams.worldspaceOBB.topLeft;
		// This is the same as multiplying the change of basis matrix by the displacement vector
		const float64_t postRotatedShiftX = nbl::hlsl::dot(firstRow, displacement);
		const float64_t postRotatedShiftY = nbl::hlsl::dot(secondRow, displacement);

		// Put them all together
		retVal->world2UV = float64_t2x3(firstRow.x, firstRow.y, postRotatedShiftX, secondRow.x, secondRow.y, postRotatedShiftY);
		return retVal;
	}

	GeoreferencedImageParams georeferencedImageParams = {};
	std::vector<std::vector<bool>> currentMappedRegionOccupancy = {};

	// These are NOT UV, pixel or tile coords into the mapped image region, rather into the real, huge image
	// Tile coords are always in mip 0 tile size. Translating to other mips levels is trivial
	float64_t2 transformWorldCoordsToUV(const float64_t3 worldCoords) const { return nbl::hlsl::mul(world2UV, worldCoords); }
	float64_t2 transformWorldCoordsToPixelCoords(const float64_t3 worldCoords) const { return float64_t2(georeferencedImageParams.imageExtents) * transformWorldCoordsToUV(worldCoords); }
	float64_t2 transformWorldCoordsToTileCoords(const float64_t3 worldCoords, const uint32_t TileSize) const { return (1.0 / TileSize) * transformWorldCoordsToPixelCoords(worldCoords); }

	// When the current mapped region is inadequate to fit the viewport, we compute a new mapped region
	void remapCurrentRegion(const GeoreferencedImageTileRange& viewportTileRange)
	{
		// Zoomed out
		if (viewportTileRange.baseMipLevel > currentMappedRegion.baseMipLevel)
		{
			// TODO: Here we would move some mip 1 tiles to mip 0 image to save the work of reuploading them, reflect that in the tracked tiles
		}
		// Zoomed in
		else if (viewportTileRange.baseMipLevel < currentMappedRegion.baseMipLevel)
		{
			// TODO: Here we would move some mip 0 tiles to mip 1 image to save the work of reuploading them, reflect that in the tracked tiles
		}
		currentMappedRegion = viewportTileRange;

		currentMappedRegionOccupancy.resize(gpuImageSideLengthTiles);
		for (auto i = 0u; i < gpuImageSideLengthTiles; i++)
		{
			currentMappedRegionOccupancy[i].clear();
			currentMappedRegionOccupancy[i].resize(gpuImageSideLengthTiles, false);
		}
		gpuImageTopLeft = uint32_t2(0, 0);
	}

	// When we can shift the mapped a region a bit and avoid tile uploads by using toroidal shifting
	void shiftAndExpandCurrentRegion(const GeoreferencedImageTileRange& viewportTileRange)
	{
		// `topLeftDiff` starts as the vector (in tiles) from the current mapped region's top left to the top left of the range encompassing the viewport
		int32_t2 topLeftDiff = int32_t2(viewportTileRange.topLeft) - int32_t2(currentMappedRegion.topLeft);
		// Since we only consider expanding the mapped region by moving the top left up and to the left, we clamp the above vector to `(-infty, 0] x (-infty, 0]`
		topLeftDiff = nbl::hlsl::min(topLeftDiff, int32_t2(0, 0));
		int32_t2 nextTopLeft = int32_t2(currentMappedRegion.topLeft) + topLeftDiff;
		// Same logic for bottom right but considering it only moves down and to the right, so clamped to `[0, infty) x [0, infty)`
		int32_t2 bottomRightDiff = int32_t2(viewportTileRange.bottomRight) - int32_t2(currentMappedRegion.bottomRight);
		bottomRightDiff = nbl::hlsl::max(bottomRightDiff, int32_t2(0, 0));
		int32_t2 nextBottomRight = int32_t2(currentMappedRegion.bottomRight) + bottomRightDiff;

		// If the number of tiles resident in this new mapped region along any axis becomes bigger than the max number of tiles the gpu image can hold, 
		// we need to shrink this next mapped region. For this to happen, we have to have expanded in only one direction, the one that has `diff != 0`
		// Therefore, we need to shrink the mapped region along the axis that has `diff = 0`, just enough tiles so that the mapped region's tile size stays within
		// the max number of tiles the gpu image can hold.
		int32_t2 nextMappedRegionDimensions = nextBottomRight - nextTopLeft + 1;
		uint32_t2 currentMappedRegionDimensions = currentMappedRegion.bottomRight - currentMappedRegion.topLeft + 1u;
		uint32_t2 gpuImageBottomRight = (gpuImageTopLeft + currentMappedRegionDimensions - 1u) % gpuImageSideLengthTiles;

		// Shrink along x axis
		if (nextMappedRegionDimensions.x > gpuImageSideLengthTiles)
		{
			int32_t tilesToFit = nextMappedRegionDimensions.x - gpuImageSideLengthTiles;
			if (0 == topLeftDiff.x)
			{
				// Move topLeft to the right to fit tiles on the other side
				nextTopLeft.x += tilesToFit;
				topLeftDiff.x += tilesToFit;
				// Mark all these tiles as non-resident
				for (uint32_t tile = 0; tile < tilesToFit; tile++)
				{
					// Get actual tile index with wraparound
					uint32_t tileIdx = (tile + gpuImageTopLeft.x) % gpuImageSideLengthTiles;
					for (uint32_t i = 0u; i < gpuImageSideLengthTiles; i++)
						currentMappedRegionOccupancy[tileIdx][i] = false;
				}
			}
			else
			{
				// Move bottomRight to the left to fit tiles on the other side
				nextBottomRight.x -= tilesToFit;
				// Mark all these tiles as non-resident
				for (uint32_t tile = 0; tile < tilesToFit; tile++)
				{
					// Get actual tile index with wraparound
					uint32_t tileIdx = (gpuImageBottomRight.x + (gpuImageSideLengthTiles - tile)) % gpuImageSideLengthTiles;
					for (uint32_t i = 0u; i < gpuImageSideLengthTiles; i++)
						currentMappedRegionOccupancy[tileIdx][i] = false;
				}
			}
		}
		// Shrink along y axis
		if (nextMappedRegionDimensions.y > gpuImageSideLengthTiles)
		{
			int32_t tilesToFit = nextMappedRegionDimensions.y - gpuImageSideLengthTiles;
			if (0 == topLeftDiff.y)
			{
				// Move topLeft down to fit tiles on the other side
				nextTopLeft.y += tilesToFit;
				topLeftDiff.y += tilesToFit;
				// Mark all these tiles as non-resident
				for (uint32_t tile = 0; tile < tilesToFit; tile++)
				{
					// Get actual tile index with wraparound
					uint32_t tileIdx = (tile + gpuImageTopLeft.y) % gpuImageSideLengthTiles;
					for (uint32_t i = 0u; i < gpuImageSideLengthTiles; i++)
						currentMappedRegionOccupancy[i][tileIdx] = false;
				}
			}
			else
			{
				// Move bottomRight up to fit tiles on the other side
				nextBottomRight.y -= tilesToFit;
				// Mark all these tiles as non-resident
				for (uint32_t tile = 0; tile < tilesToFit; tile++)
				{
					// Get actual tile index with wraparound
					uint32_t tileIdx = (gpuImageBottomRight.y + (gpuImageSideLengthTiles - tile)) % gpuImageSideLengthTiles;
					for (uint32_t i = 0u; i < gpuImageSideLengthTiles; i++)
						currentMappedRegionOccupancy[i][tileIdx] = false;
				}
			}
		}

		// Set new values for mapped region
		currentMappedRegion.topLeft = nextTopLeft;
		currentMappedRegion.bottomRight = nextBottomRight;

		// Toroidal shift for the gpu image top left
		gpuImageTopLeft = (gpuImageTopLeft + uint32_t2(topLeftDiff + int32_t(gpuImageSideLengthTiles))) % gpuImageSideLengthTiles;
	}

	// Sidelength of the gpu image, in tiles that are `GeoreferencedImageTileSize` pixels wide
	uint32_t gpuImageSideLengthTiles = {};
	// Size of the image (minus 1), in tiles of `GeoreferencedImageTileSize` sidelength
	uint32_t2 fullImageLastTileIndices = {};
	// Set mip level to extreme value so it gets recreated on first iteration
	GeoreferencedImageTileRange currentMappedRegion = { .baseMipLevel = std::numeric_limits<uint32_t>::max() };
	// Indicates on which tile of the gpu image the current mapped region's `topLeft` resides
	uint32_t2 gpuImageTopLeft = {};
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
