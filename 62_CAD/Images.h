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

// Measures a range of mip `baseMipLevel` tiles in the georeferenced image, starting at `topLeftTile` that is `nTiles` long
struct GeoreferencedImageTileRange
{
	uint32_t2 topLeftTile;
	uint32_t2 bottomRightTile;
	uint32_t baseMipLevel;
};

// @brief Used to load tiles into VRAM, keep track of loaded tiles, determine how they get sampled etc.
struct GeoreferencedImageStreamingState : public IReferenceCounted
{
	friend class DrawResourcesFiller;

protected:
	static smart_refctd_ptr<GeoreferencedImageStreamingState> create(GeoreferencedImageParams&& _georeferencedImageParams, uint32_t TileSize)
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

		// Also set the maxMipLevel
		uint32_t2 maxMipLevels = nbl::hlsl::findMSB(nbl::hlsl::roundUpToPoT(retVal->georeferencedImageParams.imageExtents / TileSize));
		retVal->maxMipLevel = nbl::hlsl::min(maxMipLevels.x, maxMipLevels.y);

		// Set max number of mip 0 tiles
		retVal->fullImageTileLength = (retVal->georeferencedImageParams.imageExtents - 1u) / TileSize + 1u;

		return retVal;
	}

	GeoreferencedImageParams georeferencedImageParams = {};
	std::vector<std::vector<bool>> currentMappedRegionOccupancy = {};

	// These are NOT UV, pixel or tile coords into the mapped image region, rather into the real, huge image
	// Tile coords are always in mip 0 tile size. Translating to other mips levels is trivial
	float64_t2 transformWorldCoordsToUV(const float64_t3 worldCoords) const { return nbl::hlsl::mul(world2UV, worldCoords); }
	float64_t2 transformWorldCoordsToPixelCoords(const float64_t3 worldCoords) const { return float64_t2(georeferencedImageParams.imageExtents) * transformWorldCoordsToUV(worldCoords); }
	float64_t2 transformWorldCoordsToTileCoords(const float64_t3 worldCoords, const uint32_t TileSize) const { return (1.0 / TileSize) * transformWorldCoordsToPixelCoords(worldCoords); }

	void ensureMappedRegionCoversViewport(const GeoreferencedImageTileRange& viewportTileRange)
	{
		// A base mip level of x in the current mapped region means we can handle the viewport having mip level y, with x <= y < x + 1.0
		// without needing to remap the region. When the user starts zooming in or out and the mip level of the viewport falls outside this range, we have to remap
		// the mapped region.
		const bool mipBoundaryCrossed = viewportTileRange.baseMipLevel != currentMappedRegion.baseMipLevel;

		// If we moved a huge amount in any direction, no tiles will remain resident, so we simply reset state
		// This only need be evaluated if the mip boundary was not already crossed
		const bool relativeShiftTooBig = !mipBoundaryCrossed &&
											nbl::hlsl::any
											(
												nbl::hlsl::abs(int32_t2(viewportTileRange.topLeftTile) - int32_t2(currentMappedRegion.topLeftTile)) >= int32_t2(gpuImageSideLengthTiles, gpuImageSideLengthTiles)
											)
											|| nbl::hlsl::any
											(
												nbl::hlsl::abs(int32_t2(viewportTileRange.bottomRightTile) - int32_t2(currentMappedRegion.bottomRightTile)) >= int32_t2(gpuImageSideLengthTiles, gpuImageSideLengthTiles)
											);

		if (mipBoundaryCrossed || relativeShiftTooBig)
			remapCurrentRegion(viewportTileRange);
		else
			slideCurrentRegion(viewportTileRange);
	}

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
		// We can expand the currentMappedRegion to make it as big as possible, at no extra cost since we only upload tiles on demand
		// Since we use toroidal updating it's kinda the same which way we expand the region. We first tryo to expand it downwards to the right
		const uint32_t2 currentTileExtents = currentMappedRegion.bottomRightTile - currentMappedRegion.topLeftTile + uint32_t2(1, 1);
		// Extend extent up to `gpuImageSideLengthTiles` by moving the `bottomRightTile` an appropriate amount downwards to the right
		currentMappedRegion.bottomRightTile += uint32_t2(gpuImageSideLengthTiles, gpuImageSideLengthTiles) - currentTileExtents;
		// This extension can cause the mapped region to fall out of bounds on border cases, therefore we clamp it and extend it in the other direction
		// by the amount of tiles we removed during clamping
		const uint32_t2 excessTiles = uint32_t2(nbl::hlsl::max(int32_t2(0, 0), int32_t2(currentMappedRegion.bottomRightTile + 1u) - int32_t2(fullImageTileLength)));
		currentMappedRegion.bottomRightTile -= excessTiles;
		// Now, on some pathological cases (such as an image that is not long along one dimension but very long along the other) shifting of the topLeftTile
		// could fall out of bounds. So we shift if possible, otherwise set it to 0
		currentMappedRegion.topLeftTile = uint32_t2(nbl::hlsl::max(int32_t2(0, 0), int32_t2(currentMappedRegion.topLeftTile) - int32_t2(excessTiles)));

		currentMappedRegionOccupancy.resize(gpuImageSideLengthTiles);
		for (auto i = 0u; i < gpuImageSideLengthTiles; i++)
		{
			currentMappedRegionOccupancy[i].clear();
			currentMappedRegionOccupancy[i].resize(gpuImageSideLengthTiles, false);
		}
		gpuImageTopLeft = uint32_t2(0, 0);
	}

	// Checks whether the viewport falls entirely withing the current mapped region and slides the latter otherwise, just enough until it covers the viewport
	void slideCurrentRegion(const GeoreferencedImageTileRange& viewportTileRange)
	{
		// `topLeftShift` represents how many tiles up and to the left we have to move the mapped region to fit the viewport. 
		// First we compute a vector from the current mapped region's topleft to the viewport's topleft. If this vector is positive along a dimension it means
		// the viewport's topleft is to the right or below the current mapped region's topleft, so we don't have to shift the mapped region to the left/up in that case
		const int32_t2 topLeftShift = nbl::hlsl::min(int32_t2(0, 0), int32_t2(viewportTileRange.topLeftTile) - int32_t2(currentMappedRegion.topLeftTile));
		// `bottomRightShift` represents the same as above but in the other direction.
		const int32_t2 bottomRightShift = nbl::hlsl::max(int32_t2(0, 0), int32_t2(viewportTileRange.bottomRightTile) - int32_t2(currentMappedRegion.bottomRightTile));

		// Mark dropped tiles as dirty/non-resident
		// The following is not necessarily equal to `gpuImageSideLengthTiles` since there can be pathological cases, as explained in the remapping method
		const uint32_t2 mappedRegionDimensions = currentMappedRegion.bottomRightTile - currentMappedRegion.topLeftTile + 1u;
		const uint32_t2 gpuImageBottomRight = (gpuImageTopLeft + mappedRegionDimensions - 1u) % gpuImageSideLengthTiles;

		if (topLeftShift.x < 0)
		{
			// Shift left
			const uint32_t tilesToFit = -topLeftShift.x;
			for (uint32_t tile = 0; tile < tilesToFit; tile++)
			{
				// Get actual tile index with wraparound
				uint32_t tileIdx = (gpuImageBottomRight.x + (gpuImageSideLengthTiles - tile)) % gpuImageSideLengthTiles;
				for (uint32_t i = 0u; i < gpuImageSideLengthTiles; i++)
					currentMappedRegionOccupancy[tileIdx][i] = false;
			}
		}
		else if (bottomRightShift.x > 0)
		{
			//Shift right
			const uint32_t tilesToFit = bottomRightShift.x;
			for (uint32_t tile = 0; tile < tilesToFit; tile++)
			{
				// Get actual tile index with wraparound
				uint32_t tileIdx = (tile + gpuImageTopLeft.x) % gpuImageSideLengthTiles;
				for (uint32_t i = 0u; i < gpuImageSideLengthTiles; i++)
					currentMappedRegionOccupancy[tileIdx][i] = false;
			}
		}

		if (topLeftShift.y < 0)
		{
			// Shift up
			const uint32_t tilesToFit = -topLeftShift.y;
			for (uint32_t tile = 0; tile < tilesToFit; tile++)
			{
				// Get actual tile index with wraparound
				uint32_t tileIdx = (gpuImageBottomRight.y + (gpuImageSideLengthTiles - tile)) % gpuImageSideLengthTiles;
				for (uint32_t i = 0u; i < gpuImageSideLengthTiles; i++)
					currentMappedRegionOccupancy[i][tileIdx] = false;
			}
		}
		else if (bottomRightShift.y > 0)
		{
			//Shift down
			const uint32_t tilesToFit = bottomRightShift.y;
			for (uint32_t tile = 0; tile < tilesToFit; tile++)
			{
				// Get actual tile index with wraparound
				uint32_t tileIdx = (tile + gpuImageTopLeft.y) % gpuImageSideLengthTiles;
				for (uint32_t i = 0u; i < gpuImageSideLengthTiles; i++)
					currentMappedRegionOccupancy[i][tileIdx] = false;
			}
		}

		// Shift the mapped region accordingly
		// A nice consequence of the mapped region being always maximally - sized is that
		// along any dimension, only a shift in one direction is necessary, so we can simply add up the shifts
		currentMappedRegion.topLeftTile = uint32_t2(int32_t2(currentMappedRegion.topLeftTile) + topLeftShift + bottomRightShift); 
		currentMappedRegion.bottomRightTile = uint32_t2(int32_t2(currentMappedRegion.bottomRightTile) + topLeftShift + bottomRightShift);

		// Toroidal shift for the gpu image top left
		gpuImageTopLeft = (gpuImageTopLeft + uint32_t2(topLeftShift + bottomRightShift + int32_t(gpuImageSideLengthTiles))) % gpuImageSideLengthTiles;
	}

	// This can become a rectangle if we implement the by-rectangle upload instead of tile-by-tile to reduce loader calls
	struct ImageTileToGPUTileCorrespondence
	{
		uint32_t2 imageTileIndex;
		uint32_t2 gpuImageTileIndex;
	};

	// Given a tile range covering the viewport, returns which tiles (at the mip level of the current mapped region) need to be made resident to draw it,
	// returning a vector of `ImageTileToGPUTileCorrespondence`, each indicating that tile `imageTileIndex` in the full image needs to be uploaded to tile
	// `gpuImageTileIndex` in the gpu image
	core::vector<ImageTileToGPUTileCorrespondence> tilesToLoad(const GeoreferencedImageTileRange& viewportTileRange)
	{
		core::vector<ImageTileToGPUTileCorrespondence> retVal;
		for (uint32_t tileX = viewportTileRange.topLeftTile.x; tileX <= viewportTileRange.bottomRightTile.x; tileX++)
			for (uint32_t tileY = viewportTileRange.topLeftTile.y; tileY <= viewportTileRange.bottomRightTile.y; tileY++)
			{
				uint32_t2 imageTileIndex = uint32_t2(tileX, tileY);
				uint32_t2 gpuImageTileIndex = ((imageTileIndex - currentMappedRegion.topLeftTile) + gpuImageTopLeft) % gpuImageSideLengthTiles;
				if (!currentMappedRegionOccupancy[gpuImageTileIndex.x][gpuImageTileIndex.y])
					retVal.push_back({ imageTileIndex , gpuImageTileIndex });
			}
		return retVal;
	}

	// Sidelength of the gpu image, in tiles that are `TileSize` pixels wide
	uint32_t gpuImageSideLengthTiles = {};
	// We establish a max mipLevel for the image, which is the mip level at which any of width, height fit in a single Tile
	uint32_t maxMipLevel = {};
	// Size of the image in tiles of `TileSize` sidelength
	uint32_t2 fullImageTileLength = {};
	// Indicates on which tile of the gpu image the current mapped region's `topLeft` resides
	uint32_t2 gpuImageTopLeft = {};
	// Converts a point (z = 1) in worldspace to UV coordinates in image space (origin shifted to topleft of the image)
	float64_t2x3 world2UV = {};
	// If the image dimensions are not exactly divisible by `TileSize`, then the last tile along a dimension only holds a proportion of `lastTileFraction` pixels along that dimension  
	float64_t lastTileFraction = {};
	// Set mip level to extreme value so it gets recreated on first iteration
	GeoreferencedImageTileRange currentMappedRegion = { .baseMipLevel = std::numeric_limits<uint32_t>::max() };
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
