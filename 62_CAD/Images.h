/* DrawResourcesFiller: This class provides important functionality to manage resources needed for a draw.
/******************************************************************************/
#pragma once

#include "shaders/globals.hlsl"
#include <future>

using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;

using image_id = uint64_t; // Could later be templated or replaced with a stronger type or hash key.

// These are mip 0 pixels per tile, also size of each physical tile into the gpu resident image
constexpr static uint32_t GeoreferencedImageTileSize = 128u;
// Mip 1 tiles are naturally half the size
constexpr static uint32_t GeoreferencedImageTileSizeMip1 = GeoreferencedImageTileSize / 2;
// How many tiles of extra padding we give to the gpu image holding the tiles for a georeferenced image
constexpr static uint32_t GeoreferencedImagePaddingTiles = 2;

enum class ImageState : uint8_t
{
	INVALID = 0,
	CREATED_AND_MEMORY_BOUND,             // GPU image created, not bound to descriptor set yet
	GPU_RESIDENT_WITH_VALID_STATIC_DATA,  // When data for static images gets issued for upload successfully, may not be bound to it's descriptor binding array index yet
	BOUND_TO_DESCRIPTOR_SET,              // Bound to descriptor set, GPU resident
};

enum class ImageType : uint8_t
{
	INVALID = 0,
	STATIC,                        // Regular non-georeferenced image, fully loaded once
	GEOREFERENCED_STREAMED,        // Streamed image, resolution depends on camera/view // TODO[DEVSH]: Probably best to rename this to STREAMED image
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
	
	uint64_t getFreeSize() const
	{
		return m_addressAllocator->get_free_size();
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
	ImageCleanup();

	~ImageCleanup() override;

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
public:
	
	GeoreferencedImageStreamingState()
	{ }

	//! Creates a new streaming state for a georeferenced image
	/*
	  Initializes CPU-side state for image streaming.  
	  Sets up world-to-UV transform, computes mip hierarchy parameters,  
	  and stores metadata about the image.

	  @param worldspaceOBB       Oriented bounding box of the image in world space
	  @param fullResImageExtents Full resolution image size in pixels (width, height)
	  @param format              Pixel format of the image
	  @param storagePath         Filesystem path for image tiles
	*/
	bool init(const OrientedBoundingBox2D& worldSpaceOBB, const uint32_t2 fullResImageExtents, const asset::E_FORMAT format, const std::filesystem::path& storagePath);

	/**
	 * @brief Update the mapped region to cover the current viewport.
	 *
	 * Computes the required tile range from the viewport and updates
	 * `currentMappedRegion` by remapping or sliding as needed.
	 *
	 * @param currentViewportExtents  Viewport size in pixels.
	 * @param ndcToWorldMat      NDC to world space mattix.
	 *
	 * @see tilesToLoad
	 */
	void updateStreamingStateForViewport(const uint32_t2 viewportExtent, const float64_t3x3& ndcToWorldMat);

	// @brief Info to match a gpu tile to the tile in the real image it should hold image data for
	struct ImageTileToGPUTileCorrespondence
	{
		uint32_t2 imageTileIndex;
		uint32_t2 gpuImageTileIndex;
	};

	/*
	 * @brief Get the tiles required for rendering the current viewport.
	 * Uses the region set by `updateStreamingStateForViewport()` to return
	 * which image tiles need loading and their target GPU tile indices.
	 */
	core::vector<ImageTileToGPUTileCorrespondence> tilesToLoad() const;

	// @brief Returns the index of the last tile when covering the image with `mipLevel` tiles
	inline uint32_t2 getLastTileIndex(uint32_t mipLevel) const
	{
		return (fullImageTileLength - 1u) >> mipLevel;
	}

	// @brief Returns whether the last tile in the image (along each dimension) is visible from the current viewport
	inline bool2 isLastTileVisible(const uint32_t2 viewportBottomRightTile) const
	{
		const uint32_t2 lastTileIndex = getLastTileIndex(currentMappedRegionTileRange.baseMipLevel);
		return bool2(lastTileIndex.x == viewportBottomRightTile.x, lastTileIndex.y == viewportBottomRightTile.y);
	}

	/**
	* @brief Compute viewport positioning and UV addressing for a georeferenced image.
	*
	* Returns a `GeoreferencedImageInfo` filled with:
	*   - `topLeft`, `dirU`, `aspectRatio` (world-space OBB)
	*   - `minUV`, `maxUV` (UV addressing for the viewport)
	*
	* Leaves `textureID` unmodified.
	*
	* @note Make sure to call `updateStreamingStateForViewport()` first so that
	*       the OBB and UVs reflect the latest viewport.
	*
	* @param imageStreamingState The streaming state of the georeferenced image.
	* @return GeoreferencedImageInfo containing viewport positioning and UV info.
	*/
	GeoreferencedImageInfo computeGeoreferencedImageAddressingAndPositioningInfo();

	bool isOutOfDate() const { return outOfDate; }

private:
	// These are NOT UV, pixel or tile coords into the mapped image region, rather into the real, huge image
	// Tile coords are always in mip 0 tile size. Translating to other mips levels is trivial

	// @brief Transform worldspace coordinates into UV coordinates into the image
	float64_t2 transformWorldCoordsToUV(const float64_t3 worldCoords) const { return nbl::hlsl::mul(worldToUV, worldCoords); }
	// @brief Transform worldspace coordinates into texel coordinates into the image
	float64_t2 transformWorldCoordsToTexelCoords(const float64_t3 worldCoords) const { return float64_t2(fullResImageExtents) * transformWorldCoordsToUV(worldCoords); }
	// @brief Transform worldspace coordinates into tile coordinates into the image, where the image is broken up into tiles of size `GeoreferencedImageTileSize` 
	float64_t2 transformWorldCoordsToTileCoords(const float64_t3 worldCoords) const { return (1.0 / GeoreferencedImageTileSize) * transformWorldCoordsToTexelCoords(worldCoords); }

	/**
	* @brief Compute the tile range and mip level needed to cover the viewport.
	*
	* Calculates which portion of the source image is visible through the given
	* viewport and chooses the optimal mip level based on zoom (viewport size
	* relative to the image). The returned range is always a subset of
	* `currentMappedRegion` and covers only the visible tiles.
	*
	* @param currentViewportExtents Size of the viewport in pixels.
	* @param ndcToWorldMat     Transform from NDC to world space, used to project
	*                       the viewport onto the image.
	*
	* @return A tile range (`GeoreferencedImageTileRange`) representing the
	*         visible region at the chosen mip level.
	*/
	GeoreferencedImageTileRange computeViewportTileRange(const uint32_t2 viewportExtent, const float64_t3x3& ndcToWorldMat);

	/*
	* @brief The GPU image backs a mapped region which is a rectangular sub-region of the original image. Note that a region being mapped does NOT imply it's currently resident in GPU memory.
	*        To display the iomage on the screen, before even checking that the tiles needed to render the portion of the image currently visible are resident in GPU memory, we first must ensure that
	*        said region is included (as a sub-rectangle) in the mapped region.
	*
	* @param viewportTileRange Range of tiles + mip level indicating what sub-rectangle (and at which mip level) of the image is going to be visible from the viewport
	*/
	void ensureMappedRegionCoversViewport(const GeoreferencedImageTileRange& viewportTileRange);

	/*
	* @brief Sets the mapped region into the image so it at least covers the sub-rectangle currently visible from the viewport. Also marks all gpu tiles dirty since none can be recycled
	*
	* @param viewportTileRange Range of tiles + mip level indicating a sub-rectangle of the image (visible from viewport) that the mapped region needs to cover
	*/
	void remapCurrentRegion(const GeoreferencedImageTileRange& viewportTileRange);

	/**
	* @brief Resets the streaming state's GPU tile occupancy map.
	* - Clears all previously marked resident tiles.
	* - After this call, every entry in `currentMappedRegionOccupancy` is `false`,
	*   meaning the GPU image is considered completely dirty (no tiles mapped).
	*/
	void ResetTileOccupancyState();

	/*
	* @brief Slides the mapped region along the image, marking the tiles dropped as dirty but preserving the residency for tiles that are inside both the previous and new mapped regions.
	*		 Note that the checks for whether this is valid to do happen outside of this function.
	*
	* @param viewportTileRange Range of tiles + mip level indicating a sub-rectangle of the image (visible from viewport) that the mapped region needs to cover
	*/
	void slideCurrentRegion(const GeoreferencedImageTileRange& viewportTileRange);

protected:
	friend class DrawResourcesFiller;

	// Oriented bounding box of the original image in world space (position + orientation)
	OrientedBoundingBox2D worldspaceOBB = {};
	// Full resolution original image size in pixels (width, height)
	uint32_t2 fullResImageExtents = {};
	// Pixel format of the image as provided by storage/loader (may differ from GPU format)
	asset::E_FORMAT sourceImageFormat = {};
	// Filesystem path where image tiles are stored
	std::filesystem::path storagePath = {};
	// GPU Image Params for the image to be created with
	IGPUImage::SCreationParams gpuImageCreationParams = {};
	// 2D bool set for tile validity of the currentMappedRegionTileRange
	std::vector<std::vector<bool>> currentMappedRegionOccupancy = {};
	// Sidelength of the gpu image, in mip 0 tiles that are `TileSize` (creation parameter) texels wide
	uint32_t gpuImageSideLengthTiles = {};
	// We establish a max mipLevel for the image, which is the mip level at which any of width, height fit in a single tile
	uint32_t maxMipLevel = {};
	// Number of mip 0 tiles needed to cover the whole image, counting the last tile that might be fractional if the image size is not perfectly divisible by TileSize
	uint32_t2 fullImageTileLength = {};
	// Indicates on which tile of the gpu image the current mapped region's `topLeft` resides
	uint32_t2 gpuImageTopLeft = {};
	// Converts a point (z = 1) in worldspace to UV coordinates in image space (origin shifted to topleft of the image)
	float64_t2x3 worldToUV = {};
	// The GPU-mapped region covering a subrectangle of the source image
	GeoreferencedImageTileRange currentMappedRegionTileRange = { .baseMipLevel = std::numeric_limits<uint32_t>::max() };
	// Tile range covering only the tiles currently visible in the viewport
	GeoreferencedImageTileRange currentViewportTileRange = { .baseMipLevel = std::numeric_limits<uint32_t>::max() };
	// Extents used for sampling the last tile (handles partial tiles / NPOT images); gets updated with `updateStreamingStateForViewport`
	uint32_t2 lastTileSamplingExtent; 
	// Extents used when writing/updating the last tile in GPU memory (handles partial tiles / NPOT images); gets updated with `updateStreamingStateForViewport`
	uint32_t2 lastTileTargetExtent;
	// We set this to true when image is evicted from cache, hinting at other places holding a smart_refctd_ptr to this objet that the GeoreferencedImageStreamingState isn't valid anymore and needs recreation/update
	bool outOfDate = false;
};

struct CachedImageRecord
{
	static constexpr uint32_t InvalidTextureIndex = nbl::hlsl::numeric_limits<uint32_t>::max;
	
	uint32_t arrayIndex = InvalidTextureIndex; // index in our array of textures binding
	bool arrayIndexAllocatedUsingImageDescriptorIndexAllocator; // whether the index of this cache entry was allocated using suballocated descriptor set which ensures correct synchronized access to a set index. (if not extra synchro is needed)
	ImageType type = ImageType::INVALID;
	ImageState state = ImageState::INVALID;
	nbl::asset::IImage::LAYOUT currentLayout = nbl::asset::IImage::LAYOUT::UNDEFINED;
	uint64_t lastUsedFrameIndex = 0ull; // last used semaphore value on this image
	uint64_t allocationOffset = ImagesMemorySubAllocator::InvalidAddress;
	uint64_t allocationSize = 0ull;
	core::smart_refctd_ptr<IGPUImageView> gpuImageView = nullptr;
	core::smart_refctd_ptr<ICPUImage> staticCPUImage = nullptr; // cached cpu image for uploading to gpuImageView when needed.
	core::smart_refctd_ptr<GeoreferencedImageStreamingState> georeferencedImageState = nullptr; // Used to track tile residency for georeferenced images
	
	// In LRU Cache `insert` function, in case of cache miss, we need to construct the refereence with semaphore value
	CachedImageRecord(uint64_t currentFrameIndex) 
		: arrayIndex(InvalidTextureIndex)
		, arrayIndexAllocatedUsingImageDescriptorIndexAllocator(false)
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

	std::string toString(uint64_t imageID = std::numeric_limits<uint64_t>::max()) const;

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
	template<std::invocable<image_id, CachedImageRecord&> EvictionCallback>
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
	
	inline void logState(nbl::system::logger_opt_smart_ptr logger)
	{
		logger.log("=== Image Cache Status ===", nbl::system::ILogger::ELL_INFO);
		for (const auto& [imageID, record] : *this)
		{
			logger.log(("\n" + record.toString(imageID)).c_str(), nbl::system::ILogger::ELL_INFO);
		}
		logger.log("=== End of Image Cache ===", nbl::system::ILogger::ELL_INFO);
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
	std::future<smart_refctd_ptr<ICPUBuffer>> srcBufferFuture;
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

/// @brief Abstract class with two overridable methods to load a region of an image, either by requesting a region at a target extent (like the loaders in n4ce do) or to request a specific region from a mip level
//         (like precomputed mips solution would use).
struct IImageRegionLoader : IReferenceCounted
{
	/**
	* @brief Load a region from an image - used to load from images with precomputed mips
	*
	* @param imagePath Path to file holding the image data
	* @param offset Offset into the image (at requested mipLevel!) at which the region begins
	* @param extent Extent of the region to load (at requested mipLevel!)
	* @param mipLevel From which mip level image to retrieve the data from
	* @param downsample True if this request is supposed to go into GPU mip level 1, false otherwise
	*
	* @return ICPUBuffer with the requested image data
	*/
	core::smart_refctd_ptr<ICPUBuffer> load(std::filesystem::path imagePath, uint32_t2 offset, uint32_t2 extent, uint32_t mipLevel, bool downsample)
	{
		assert(hasPrecomputedMips(imagePath));
		return load_impl(imagePath, offset, extent, mipLevel, downsample);
	}

	/**
	* @brief Load a region from an image - used to load from images using the n4ce loaders. Loads a region given by `offset, extent` as an image of size `targetExtent`
	*        where `targetExtent <= extent` so the loader is in charge of downsampling.
	*
	* @param imagePath Path to file holding the image data
	* @param offset Offset into the image at which the region begins
	* @param extent Extent of the region to load
	* @param targetExtent Extent of the resulting image. Should NEVER be bigger than `extent`
	*
	* @return ICPUBuffer with the requested image data
	*/
	core::smart_refctd_ptr<ICPUBuffer> load(std::filesystem::path imagePath, uint32_t2 offset, uint32_t2 extent, uint32_t2 targetExtent)
	{
		assert(!hasPrecomputedMips(imagePath));
		return load_impl(imagePath, offset, extent, targetExtent);
	}

	// @brief Get the extents (in texels) of an image.
	virtual uint32_t2 getExtents(std::filesystem::path imagePath) = 0;

	/**
	* @brief Get the texel format for an image.
	*/
	virtual asset::E_FORMAT getFormat(std::filesystem::path imagePath) = 0;

	// @brief Returns whether the image should be loaded with the precomputed mip method or the n4ce loader method.
	virtual bool hasPrecomputedMips(std::filesystem::path imagePath) const = 0;
private:

	// @brief Override to support loading with precomputed mips
	virtual core::smart_refctd_ptr<ICPUBuffer> load_impl(std::filesystem::path imagePath, uint32_t2 offset, uint32_t2 extent, uint32_t mipLevel, bool downsample) { return nullptr; }

	// @brief Override to support loading with n4ce-style loaders
	virtual core::smart_refctd_ptr<ICPUBuffer> load_impl(std::filesystem::path imagePath, uint32_t2 offset, uint32_t2 extent, uint32_t2 targetExtent) { return nullptr; }
};