#pragma once
#include "Polyline.h"
#include "CTriangleMesh.h"
#include "Hatch.h"
#include "IndexAllocator.h"
#include "Images.h"
#include <nbl/video/utilities/SIntendedSubmitInfo.h>
#include <nbl/core/containers/LRUCache.h>  
#include <nbl/ext/TextRendering/TextRendering.h>
// #include <nbl/video/alloc/SubAllocatedDescriptorSet.h>
using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::ext::TextRendering;

static_assert(sizeof(DrawObject) == 16u);
static_assert(sizeof(MainObject) == 20u);
static_assert(sizeof(LineStyle) == 88u);

// ! DrawResourcesFiller
// ! This class provides important functionality to manage resources needed for a draw.
// ! Drawing new objects (polylines, hatches, etc.) should go through this function.
// ! Contains all the scene resources (buffers and images)
// ! In the case of overflow (i.e. not enough remaining v-ram) will auto-submit/render everything recorded so far,
//   and additionally makes sure relavant data needed for those draw calls are present in memory
struct DrawResourcesFiller
{
public:
	
	// We pack multiple data types in a single buffer, we need to makes sure each offset starts aligned to avoid mis-aligned accesses
	static constexpr size_t ResourcesMaxNaturalAlignment = 8u;

	/// @brief general parent struct for 1.ReservedCompute and 2.CPUGenerated Resources
	struct ResourceBase
	{
		static constexpr size_t InvalidBufferOffset = ~0u;
		size_t bufferOffset = InvalidBufferOffset; // set when copy to gpu buffer is issued
		virtual size_t getCount() const = 0;
		virtual size_t getStorageSize() const = 0;
		virtual size_t getAlignedStorageSize() const { return core::alignUp(getStorageSize(), ResourcesMaxNaturalAlignment); }
	};

	/// @brief ResourceBase reserved for compute shader stages input/output
	template <typename T>
	struct ReservedComputeResource : ResourceBase
	{
		size_t count = 0ull;
		size_t getCount() const override { return count; }
		size_t getStorageSize() const override  { return count * sizeof(T); }
	};

	/// @brief ResourceBase which is filled by CPU, packed and sent to GPU
	template <typename T>
	struct CPUGeneratedResource : ResourceBase
	{
		core::vector<T> vector;
		size_t getCount() const { return vector.size(); }
		size_t getStorageSize() const { return vector.size() * sizeof(T); }
		
		/// @return pointer to start of the data to be filled, up to additionalCount
		T* increaseCountAndGetPtr(size_t additionalCount) 
		{
			size_t offset = vector.size();
			vector.resize(offset + additionalCount);
			return &vector[offset];
		}

		/// @brief increases size of general-purpose resources that hold bytes
		/// @param alignment: Alignment of the pointer returned to be filled, should be PoT and <= ResourcesMaxNaturalAlignment, only use this if storing raw bytes in vector
		/// @return pointer to start of the data to be filled, up to additional size
		size_t increaseSizeAndGetOffset(size_t additionalSize, size_t alignment) 
		{
			assert(core::isPoT(alignment) && alignment <= ResourcesMaxNaturalAlignment);
			size_t offset = core::alignUp(vector.size(), alignment);
			vector.resize(offset + additionalSize);
			return offset;
		}
		
		uint32_t addAndGetOffset(const T& val)
		{
			vector.push_back(val);
			return vector.size() - 1u;
		}

		T* data() { return vector.data(); }
	};

	/// @brief struct to hold all resources
	// TODO: rename to staged resources buffers or something like that
	struct ResourcesCollection
	{
		// auto-submission level 0 resources (settings that mainObj references)
		CPUGeneratedResource<LineStyle> lineStyles;
		CPUGeneratedResource<DTMSettings> dtmSettings;
		CPUGeneratedResource<float64_t3x3> customProjections;
		CPUGeneratedResource<WorldClipRect> customClipRects;
	
		// auto-submission level 1 buffers (mainObj that drawObjs references, if all drawObjs+idxBuffer+geometryInfo doesn't fit into mem this will be broken down into many)
		CPUGeneratedResource<MainObject> mainObjects;

		// auto-submission level 2 buffers
		CPUGeneratedResource<DrawObject> drawObjects;
		CPUGeneratedResource<uint32_t> indexBuffer; // TODO: this is going to change to ReservedComputeResource where index buffer gets filled by compute shaders
		CPUGeneratedResource<uint8_t> geometryInfo; // general purpose byte buffer for custom data for geometries (eg. line points, bezier definitions, aabbs)

		// Get Total memory consumption, If all ResourcesCollection get packed together with ResourcesMaxNaturalAlignment
		// used to decide the remaining memory and when to overflow
		size_t calculateTotalConsumption() const
		{
			return
				lineStyles.getAlignedStorageSize() +
				dtmSettings.getAlignedStorageSize() +
				customProjections.getAlignedStorageSize() +
				customClipRects.getAlignedStorageSize() +
				mainObjects.getAlignedStorageSize() +
				drawObjects.getAlignedStorageSize() +
				indexBuffer.getAlignedStorageSize() +
				geometryInfo.getAlignedStorageSize();
		}
	};
	
	DrawResourcesFiller();

	DrawResourcesFiller(smart_refctd_ptr<IUtilities>&& utils, IQueue* copyQueue);

	typedef std::function<void(SIntendedSubmitInfo&)> SubmitFunc;
	void setSubmitDrawsFunction(const SubmitFunc& func);
	
	// DrawResourcesFiller needs to access these in order to allocate GPUImages and write the to their correct descriptor set binding
	void setTexturesDescriptorSetAndBinding(core::smart_refctd_ptr<video::IGPUDescriptorSet>&& descriptorSet, uint32_t binding);

	/// @brief Get minimum required size for resources buffer (containing objects and geometry info and their settings)
	static constexpr size_t getMinimumRequiredResourcesBufferSize()
	{
		// for auto-submission to work correctly, memory needs to serve at least 2 linestyle, 1 dtm settings, 1 clip proj, 1 main obj, 1 draw obj and 512 bytes of additional mem for geometries and index buffer
		// this is the ABSOLUTE MINIMUM (if this value is used rendering will probably be as slow as CPU drawing :D)
		return core::alignUp(sizeof(LineStyle) + sizeof(LineStyle) * DTMSettings::MaxContourSettings + sizeof(DTMSettings) + sizeof(WorldClipRect) + sizeof(float64_t3x3) + sizeof(MainObject) + sizeof(DrawObject) + 512ull, ResourcesMaxNaturalAlignment);
	}

	void allocateResourcesBuffer(ILogicalDevice* logicalDevice, size_t size);

	void allocateMSDFTextures(ILogicalDevice* logicalDevice, uint32_t maxMSDFs, uint32_t2 msdfsExtent);

	// functions that user should set to get MSDF texture if it's not available in cache.
	// it's up to user to return cached or generate on the fly.
	typedef std::function<core::smart_refctd_ptr<ICPUImage>(nbl::ext::TextRendering::FontFace* /*face*/, uint32_t /*glyphIdx*/)> GetGlyphMSDFTextureFunc;
	typedef std::function<core::smart_refctd_ptr<ICPUImage>(HatchFillPattern/*pattern*/)> GetHatchFillPatternMSDFTextureFunc;
	void setGlyphMSDFTextureFunction(const GetGlyphMSDFTextureFunc& func);
	void setHatchFillMSDFTextureFunction(const GetHatchFillPatternMSDFTextureFunc& func);

	// Must be called at the end of each frame.
	// right before submitting the main draw that uses the currently queued geometry, images, or other objects/resources.
	// Registers the semaphore/value that will signal completion of this frame�s draw,
	// This allows future frames to safely deallocate or evict resources used in the current frame by waiting on this signal before reuse or destruction.
	// `drawSubmitWaitValue` should reference the wait value of the draw submission finishing this frame using the `intendedNextSubmit`; 
	void markFrameUsageComplete(uint64_t drawSubmitWaitValue);

	// TODO[Przemek]: try to draft up a `CTriangleMesh` Class in it's own header (like CPolyline), simplest form is basically two cpu buffers (1 array of uint index buffer, 1 array of float64_t3 vertexBuffer)
	// TODO[Przemek]: Then have a `drawMesh` function here similar to drawXXX's below, this will fit both vertex and index buffer in the `geometryBuffer`.
	// take a `SIntendedSubmitInfo` like others, but don't use it as I don't want you to handle anything regarding autoSubmit
	// somehow retrieve or calculate the geometry buffer offsets of your vertex and index buffer to be used outside for binding purposes

	//! this function fills buffers required for drawing a polyline and submits a draw through provided callback when there is not enough memory.
	void drawPolyline(const CPolylineBase& polyline, const LineStyleInfo& lineStyleInfo, SIntendedSubmitInfo& intendedNextSubmit);

	//! Draws a fixed-geometry polyline using a custom transformation.
	//! TODO: Change `polyline` input to an ID referencing a possibly cached instance in our buffers, allowing reuse and avoiding redundant uploads.
	void drawFixedGeometryPolyline(const CPolylineBase& polyline, const LineStyleInfo& lineStyleInfo, const float64_t3x3& transformation, TransformationType transformationType, SIntendedSubmitInfo& intendedNextSubmit);
	
	/// Use this in a begin/endMainObject scope when you want to draw different polylines that should essentially be a single main object (no self-blending between components of a single main object)
	/// WARNING: make sure this function  is called within begin/endMainObject scope
	void drawPolyline(const CPolylineBase& polyline, SIntendedSubmitInfo& intendedNextSubmit);
	
	void drawTriangleMesh(
		const CTriangleMesh& mesh,
		const DTMSettingsInfo& dtmSettingsInfo,
		SIntendedSubmitInfo& intendedNextSubmit);

	// ! Convinience function for Hatch with MSDF Pattern and a solid background
	void drawHatch(
		const Hatch& hatch,
		const float32_t4& foregroundColor, 
		const float32_t4& backgroundColor,
		const HatchFillPattern fillPattern,
		SIntendedSubmitInfo& intendedNextSubmit);
	
	// ! Hatch with MSDF Pattern
	void drawHatch(
		const Hatch& hatch,
		const float32_t4& color,
		const HatchFillPattern fillPattern,
		SIntendedSubmitInfo& intendedNextSubmit);

	// ! Solid Fill Hacth
	void drawHatch(
		const Hatch& hatch,
		const float32_t4& color,
		SIntendedSubmitInfo& intendedNextSubmit);
	
	/// Used by SingleLineText, Issue drawing a font glyph
	/// WARNING: make sure this function  is called within begin/endMainObject scope
	void drawFontGlyph(
		nbl::ext::TextRendering::FontFace* fontFace,
		uint32_t glyphIdx,
		float64_t2 topLeft,
		float32_t2 dirU,
		float32_t  aspectRatio,
		float32_t2 minUV,
		SIntendedSubmitInfo& intendedNextSubmit);

	void drawGridDTM(const float64_t2& topLeft,
		float64_t width,
		float64_t height,
		float gridCellWidth,
		uint64_t textureID,
		const DTMSettingsInfo& dtmSettingsInfo,
		SIntendedSubmitInfo& intendedNextSubmit);
	
	struct StaticImageInfo
	{
		image_id imageID;
		core::smart_refctd_ptr<ICPUImage> cpuImage;
	};

	/**
	 * @brief Adds a static 2D image to the draw resource set for rendering.
	 *
	 * This function ensures that a given image is available as a GPU-resident texture for future draw submissions.
	 * It uses an LRU cache to manage descriptor set slots and evicts old images if necessary to make room for new ones.
	 *
	 * If the image is already cached and its slot is valid, it returns true;
	 * Otherwise, it performs the following:
	 *   - Allocates a new descriptor set slot.
	 *   - Promotes the image format to be GPU-compatible.
	 *   - Creates a GPU image and GPU image view.
	 *   - Queues the image for uploading via staging in the next submit.
	 *   - If memory is constrained, attempts to evict other images to free up space.
	 *
	 * @param staticImage              Unique identifier for the image resource plus the CPU-side image resource to (possibly) upload.
	 * @param intendedNextSubmit   Struct representing the upcoming submission, including a semaphore for safe scheduling.
	 *
	 * @note This function ensures that the descriptor slot is not reused while the GPU may still be reading from it.
	 *       If an eviction is required and the evicted image is scheduled to be used in the next submit, it triggers
	 *       a flush of pending draws to preserve correctness.
	 *
	 * @note The function uses the `imagesCache` LRU cache to track usage and validity of texture slots.
	 *       If an insertion leads to an eviction, a callback ensures proper deallocation and synchronization.
	 * @return true if the image was successfully cached and is ready for use; false if allocation failed most likely due to the image being larger than the memory arena allocated for all images.
	 */
	bool ensureStaticImageAvailability(const StaticImageInfo& staticImage, SIntendedSubmitInfo& intendedNextSubmit);
	
	/**
	 * @brief Ensures that multiple static 2D images are resident and ready for rendering.
	 *
	 * Attempts to make all provided static images GPU-resident by calling `ensureStaticImageAvailability`
	 * for each. Afterward, it verifies that none of the newly ensured images have been evicted,
	 * which could happen due to limited VRAM or memory fragmentation.
	 *
	 * This function is expected to succeed if:
	 * - The number of images does not exceed `ImagesBindingArraySize`.
	 * - Each image individually fits into the image memory arena.
	 * - There is enough VRAM to hold all images simultaneously.
	 *
	 * @param staticImages A span of StaticImageInfo structures describing the images to be ensured.
	 * @param intendedNextSubmit Struct representing the upcoming submission, including a semaphore for safe scheduling.
	 *
	 * @return true If all images were successfully made resident and none were evicted during the process.
	 * @return false If:
	 *   - The number of images exceeds the descriptor binding array size.
	 *   - Any individual image could not be made resident (e.g., larger than the allocator can support).
	 *   - Some images were evicted due to VRAM pressure or allocator fragmentation, in which case Clearing the image cache and retrying MIGHT be a success (TODO: handle internally)
	 */
	bool ensureMultipleStaticImagesAvailability(std::span<StaticImageInfo> staticImages, SIntendedSubmitInfo& intendedNextSubmit);

	/**
	 * @brief Ensures a GPU-resident georeferenced image exists in the cache, allocating resources if necessary.
	 * 
	 * If the specified image ID is not already present in the cache, or if the cached version is incompatible
	 * with the requested parameters (e.g. extent, format, or type), this function allocates GPU memory,
	 * creates the image and its view, to be bound to a descriptor binding in the future.
	 * 
	 * If the image already exists and matches the requested parameters, its usage metadata is updated.
	 * In either case, the cache is updated to reflect usage in the current frame.
	 * 
	 * This function also handles automatic eviction of old images via an LRU policy when space is limited.
	 * 
	 * @param imageID                Unique identifier of the image to add or reuse.
	 * @param params                 Georeferenced Image Params
	 * @param intendedNextSubmit     Submit info object used to track resources pending GPU submission.
	 * 
	 * @return true if the image was successfully cached and is ready for use; false if allocation failed.
	 * [TODO]: should be internal protected member function.
	 */
	bool ensureGeoreferencedImageAvailability_AllocateIfNeeded(image_id imageID, const GeoreferencedImageParams& params, SIntendedSubmitInfo& intendedNextSubmit);

	// [TODO]: should be internal protected member function.
	bool queueGeoreferencedImageCopy_Internal(image_id imageID, const StreamedImageCopy& imageCopy);

	// This function must be called immediately after `addStaticImage` for the same imageID.
	void addImageObject(image_id imageID, const OrientedBoundingBox2D& obb, SIntendedSubmitInfo& intendedNextSubmit);
	
	// This function must be called immediately after `addStaticImage` for the same imageID.
	void addGeoreferencedImage(image_id imageID, const GeoreferencedImageParams& params, SIntendedSubmitInfo& intendedNextSubmit);

	/// @brief call this function before submitting to ensure all buffer and textures resourcesCollection requested via drawing calls are copied to GPU
	/// records copy command into intendedNextSubmit's active command buffer and might possibly submits if fails allocation on staging upload memory.
	bool pushAllUploads(SIntendedSubmitInfo& intendedNextSubmit);

	/// @brief  resets staging buffers and images
	void reset()
	{
		resetDrawObjects();
		resetMainObjects();
		resetCustomProjections();
		resetCustomClipRects();
		resetLineStyles();
		resetDTMSettings();

		drawObjectsFlushedToDrawCalls = 0ull;
		drawCalls.clear();
	}

	/// @brief collection of all the resources that will eventually be reserved or copied to in the resourcesGPUBuffer, will be accessed via individual BDA pointers in shaders
	const ResourcesCollection& getResourcesCollection() const;

	/// @brief buffer containing all non-texture type resources
	nbl::core::smart_refctd_ptr<IGPUBuffer> getResourcesGPUBuffer() const { return resourcesGPUBuffer; }

	/// @return how far resourcesGPUBuffer was copied to by `finalizeAllCopiesToGPU` in `resourcesCollection` 
	const size_t getCopiedResourcesSize() { return copiedResourcesSize; }

	// Setting Active Resources:
	void setActiveLineStyle(const LineStyleInfo& lineStyle);
	void setActiveDTMSettings(const DTMSettingsInfo& dtmSettingsInfo);

	void beginMainObject(MainObjectType type, TransformationType transformationType = TransformationType::TT_NORMAL);
	void endMainObject();

	void pushCustomProjection(const float64_t3x3& projection);
	void popCustomProjection();
	
	void pushCustomClipRect(const WorldClipRect& clipRect);
	void popCustomClipRect();

	const std::deque<float64_t3x3>& getCustomProjectionStack() const { return activeProjections; }
	const std::deque<WorldClipRect>& getCustomClipRectsStack() const { return activeClipRects; }

	smart_refctd_ptr<IGPUImageView> getMSDFsTextureArray() { return msdfTextureArray; }

	uint32_t2 getMSDFResolution() {
		auto extents = msdfTextureArray->getCreationParameters().image->getCreationParameters().extent;
		return uint32_t2(extents.width, extents.height);
	}
	uint32_t getMSDFMips() {
		return msdfTextureArray->getCreationParameters().image->getCreationParameters().mipLevels;
	}

	/// For advanced use only, (passed to shaders for them to know if we overflow-submitted in the middle if a main obj
	uint32_t getActiveMainObjectIndex() const;

	struct MSDFImageState
	{
		core::smart_refctd_ptr<ICPUImage> image;
		bool uploadedToGPU : 1u;

		bool isValid() const { return image.get() != nullptr; }
		void evict()
		{
			image = nullptr;
			uploadedToGPU = false;
		}
	};

	// NOTE: Most probably Going to get removed soon with a single draw call in GPU-driven rendering
	struct DrawCallData
	{
		union
		{
			struct Dtm
			{
				uint64_t indexBufferOffset;
				uint64_t indexCount;
				uint64_t triangleMeshVerticesBaseAddress;
				uint32_t triangleMeshMainObjectIndex;
			} dtm;
			struct DrawObj
			{
				uint64_t drawObjectStart = 0ull;
				uint64_t drawObjectCount = 0ull;
			} drawObj;
		};
		bool isDTMRendering;
	};

	const std::vector<DrawCallData>& getDrawCalls() const;

	/// @brief Stores all CPU-side resources that were staged and prepared for a single GPU submission.
	///
	/// *** This cache includes anything used or referenced from DrawResourcesFiller in the Draw Submit:
	/// - Buffer data (geometry, indices, etc.)
	/// - MSDF CPU images
	/// - Draw call metadata
	/// - Active MainObject Index --> this is another state of the submit that we need to store 
	///
	/// The data is fully preprocessed and ready to be pushed to the GPU with no further transformation.
	/// This enables efficient replays without traversing or re-generating scene content.
	struct ReplayCache
	{
		std::vector<DrawCallData> drawCallsData;
		ResourcesCollection resourcesCollection;
		std::vector<MSDFImageState> msdfImagesState;
		std::unique_ptr<ImagesCache> imagesCache;
		uint32_t activeMainObjectIndex = InvalidMainObjectIdx;
		// TODO: non msdf general CPU Images
		// TODO: Get total memory consumption for logging?
	};

	/// @brief Creates a snapshot of all currently staged CPU-side resourcesCollection for future replay or deferred submission.
	/// 
	/// @warning This cache corresponds to a **single intended GPU submit**. 
	/// If your frame submission overflows into multiple submits due to staging memory limits or batching,
	/// you are responsible for creating **multiple ReplayCache instances**, one per submit.
	///
	/// @return A heap-allocated ReplayCache containing a copy of all staged CPU-side resourcesCollection and draw call data.
	std::unique_ptr<ReplayCache> createReplayCache();

	/// @brief Redirects all subsequent resource upload and getters to use an external ReplayCache.
	///
	/// After calling this function, staging, resource getters, and upload mechanisms will pull data from the given ReplayCache
	/// instead of the internal accumulation cache.
	///
	/// User is responsible for management of cache and making sure it's alive in the ReplayCache scope
	void setReplayCache(ReplayCache* cache);
	
	/// @brief Reverts internal logic to use the default internal staging and resource accumulation cache.
	/// Must be called once per corresponding `pushReplayCacheUse()`.
	void unsetReplayCache();

protected:

	SubmitFunc submitDraws;

	/// @brief Records GPU copy commands for all staged buffer resourcesCollection into the active command buffer.
	bool pushBufferUploads(SIntendedSubmitInfo& intendedNextSubmit, ResourcesCollection& resourcesCollection);
	
	/// @brief Records GPU copy commands for all staged msdf images into the active command buffer.
	bool pushMSDFImagesUploads(SIntendedSubmitInfo& intendedNextSubmit, std::vector<MSDFImageState>& msdfImagesState);

	/// @brief binds cached images into their correct descriptor set slot if not already resident.
	bool bindImagesToArrayIndices(ImagesCache& imagesCache);

	/// @brief Records GPU copy commands for all staged images into the active command buffer.
	bool pushStaticImagesUploads(SIntendedSubmitInfo& intendedNextSubmit, ImagesCache& imagesCache);
	
	/// @brief copies the queued up streamed copies.
	bool pushStreamedImagesUploads(SIntendedSubmitInfo& intendedNextSubmit);

	const size_t calculateRemainingResourcesSize() const;

	/// @brief Internal Function to call whenever we overflow when we can't fill all of mainObject's drawObjects
	/// @param intendedNextSubmit 
	/// @param mainObjectIndex: function updates mainObjectIndex after submitting, clearing everything and acquiring  mainObjectIndex again.
	void submitCurrentDrawObjectsAndReset(SIntendedSubmitInfo& intendedNextSubmit, uint32_t& mainObjectIndex);

	// Gets resource index to the active linestyle data from the top of stack 
	// If it's been invalidated then it will request to add to resources again ( auto-submission happens If there is not enough memory to add again)
	uint32_t acquireActiveLineStyleIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit);
	
	// Gets resource index to the active linestyle data from the top of stack 
	// If it's been invalidated then it will request to add to resources again ( auto-submission happens If there is not enough memory to add again)
	uint32_t acquireActiveDTMSettingsIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit);

	// Gets resource index to the active projection data from the top of stack 
	// If it's been invalidated then it will request to add to resources again ( auto-submission happens If there is not enough memory to add again)
	uint32_t acquireActiveCustomProjectionIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit);
	
	// Gets resource index to the active clip data from the top of stack 
	// If it's been invalidated then it will request to add to resources again ( auto-submission happens If there is not enough memory to add again)
	uint32_t acquireActiveCustomClipRectIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit);
	
	// Gets resource index to the active main object data
	// If it's been invalidated then it will request to add to resources again ( auto-submission happens If there is not enough memory to add again)
	uint32_t acquireActiveMainObjectIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit);

	/// Attempts to add lineStyle to resources. If it fails to do, due to resource limitations, auto-submits and tries again. 
	uint32_t addLineStyle_SubmitIfNeeded(const LineStyleInfo& lineStyle, SIntendedSubmitInfo& intendedNextSubmit);
	
	/// Attempts to add dtmSettings to resources. If it fails to do, due to resource limitations, auto-submits and tries again. 
	uint32_t addDTMSettings_SubmitIfNeeded(const DTMSettingsInfo& dtmSettings, SIntendedSubmitInfo& intendedNextSubmit);
	
	/// Attempts to add custom projection to gpu resources. If it fails to do, due to resource limitations, auto-submits and tries again. 
	uint32_t addCustomProjection_SubmitIfNeeded(const float64_t3x3& projection, SIntendedSubmitInfo& intendedNextSubmit);
	
	/// Attempts to add custom clip to gpu resources. If it fails to do, due to resource limitations, auto-submits and tries again. 
	uint32_t addCustomClipRect_SubmitIfNeeded(const WorldClipRect& clipRect, SIntendedSubmitInfo& intendedNextSubmit);
	
	/// returns index to added LineStyleInfo, returns Invalid index if it exceeds resource limitations
	uint32_t addLineStyle_Internal(const LineStyleInfo& lineStyleInfo);
	
	/// returns index to added DTMSettingsInfo, returns Invalid index if it exceeds resource limitations
	uint32_t addDTMSettings_Internal(const DTMSettingsInfo& dtmSettings, SIntendedSubmitInfo& intendedNextSubmit);
	
	/// Attempts to upload as many draw objects as possible within the given polyline section considering resource limitations
	void addPolylineObjects_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx);
	
	/// Attempts to upload as many draw objects as possible within the given polyline connectors considering resource limitations
	void addPolylineConnectors_Internal(const CPolylineBase& polyline, uint32_t& currentPolylineConnectorObj, uint32_t mainObjIdx);
	
	/// Attempts to upload as many draw objects as possible within the given polyline section considering resource limitations
	void addLines_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx);
	
	/// Attempts to upload as many draw objects as possible within the given polyline section considering resource limitations
	void addQuadBeziers_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx);
	
	/// Attempts to upload as many draw objects as possible within the given hatch considering resource limitations
	void addHatch_Internal(const Hatch& hatch, uint32_t& currentObjectInSection, uint32_t mainObjIndex);
	
	/// Attempts to upload a single GlyphInfo considering resource limitations
	bool addFontGlyph_Internal(const GlyphInfo& glyphInfo, uint32_t mainObjIdx);
	
	/// Attempts to upload a single GridDTMInfo considering resource limitations
	bool addGridDTM_Internal(const GridDTMInfo& gridDTMInfo, uint32_t mainObjIdx);
	/// Attempts to upload a single image object considering resource limitations (not accounting for the resource image added using ensureStaticImageAvailability function)
	bool addImageObject_Internal(const ImageObjectInfo& imageObjectInfo, uint32_t mainObjIdx);;
	
	/// Attempts to upload a georeferenced image info considering resource limitations (not accounting for the resource image added using ensureStaticImageAvailability function)
	bool addGeoreferencedImageInfo_Internal(const GeoreferencedImageInfo& georeferencedImageInfo, uint32_t mainObjIdx);;
	
	uint32_t getImageIndexFromID(image_id imageID, const SIntendedSubmitInfo& intendedNextSubmit);

	/**
	 * @brief Evicts a GPU image and deallocates its associated descriptor and memory, flushing draws if needed.
	 *
	 * This function is called when an image must be removed from GPU memory (typically due to VRAM pressure).
	 * If the evicted image is scheduled to be used in the next draw submission, a flush is performed to avoid
	 * use-after-free issues. Otherwise, it proceeds with deallocation immediately.
	 *
	 * It prepares a cleanup object that ensures the memory range used by the image will be returned to the suballocator
	 * only after the GPU has finished using it, guarded by a semaphore wait.
	 *
	 * @param imageID The unique ID of the image being evicted.
	 * @param evicted A reference to the evicted image, containing metadata such as allocation offset, size, usage frame, etc.
	 * @param intendedNextSubmit Reference to the intended submit information. Used for synchronizing draw submission and safe deallocation.
	 *
	 * @warning Deallocation may use a conservative semaphore wait value if exact usage information is unavailable. [future todo: fix] 
	 */
	void evictImage_SubmitIfNeeded(image_id imageID, const CachedImageRecord& evicted, SIntendedSubmitInfo& intendedNextSubmit);
	
	struct ImageAllocateResults
	{
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> gpuImageView = nullptr;
		uint64_t allocationOffset = ImagesMemorySubAllocator::InvalidAddress;
		uint64_t allocationSize = 0ull;
		bool isValid() const { return (gpuImageView && (allocationOffset != ImagesMemorySubAllocator::InvalidAddress)); }
	};

	/**
	 * @brief Attempts to create and allocate a GPU image and its view, with fallback eviction on failure.
	 *
	 * This function tries to create a GPU image using the specified creation parameters, allocate memory
	 * from the shared image memory arena, bind it to device-local memory, and create an associated image view.
	 * If memory allocation fails (e.g. due to VRAM exhaustion), the function will evict textures from the internal
	 * LRU cache and retry the operation until successful, or until only the currently-inserted image remains.
	 *
	 * This is primarily used by the draw resource filler to manage GPU image memory for streamed or cached images.
	 *
	 * @param imageParams Creation parameters for the image. Should match `nbl::asset::IImage::SCreationParams`.
	 * @param intendedNextSubmit Reference to the current intended submit info. Used for synchronizing evictions.
	 * @param imageDebugName Debug name assigned to the image and its view for easier profiling/debugging.
	 *
	 * @return ImageAllocateResults A struct containing:
	 * - `allocationOffset`: Offset into the memory arena (or InvalidAddress on failure).
	 * - `allocationSize`: Size of the allocated memory region.
	 * - `gpuImageView`: The created GPU image view (nullptr if creation failed).
	 */
	ImageAllocateResults tryCreateAndAllocateImage_SubmitIfNeeded(const nbl::asset::IImage::SCreationParams& imageParams, nbl::video::SIntendedSubmitInfo& intendedNextSubmit, std::string debugName = "UnnamedNablaImage");

	/**
	 * @brief Determines creation parameters for a georeferenced image based on heuristics.
	 *
	 * This function decides whether a georeferenced image should be treated as a fully resident GPU texture
	 * or as a streamable image based on the relationship between its total resolution and the viewport size.
	 * It then fills out the appropriate Nabla image creation parameters.
	 *
	 * @param[out] outImageParams Structure to be filled with image creation parameters (format, size, etc.).
	 * @param[out] outImageType Indicates whether the image should be fully resident or streamed.
	 * @param[in] georeferencedImageParams Parameters describing the full image extents, viewport extents, and format.
	*/
	void determineGeoreferencedImageCreationParams(nbl::asset::IImage::SCreationParams& outImageParams, ImageType& outImageType, const GeoreferencedImageParams& georeferencedImageParams);

	void resetMainObjects()
	{
		resourcesCollection.mainObjects.vector.clear();
		activeMainObjectIndex = InvalidMainObjectIdx;
	}

	// these resources are data related to chunks of a whole mainObject
	void resetDrawObjects()
	{
		resourcesCollection.drawObjects.vector.clear();
		resourcesCollection.indexBuffer.vector.clear();
		resourcesCollection.geometryInfo.vector.clear();
	}

	void resetCustomProjections()
	{
		resourcesCollection.customProjections.vector.clear();
		
		// Invalidate all the clip projection addresses because activeProjections buffer got reset
		for (auto& addr : activeProjectionIndices)
			addr = InvalidCustomProjectionIndex;
	}

	void resetCustomClipRects()
	{
		resourcesCollection.customClipRects.vector.clear();
		
		// Invalidate all the clip projection addresses because activeProjections buffer got reset
		for (auto& addr : activeClipRectIndices)
			addr = InvalidCustomClipRectIndex;
	}

	void resetLineStyles()
	{
		resourcesCollection.lineStyles.vector.clear();
		activeLineStyleIndex = InvalidStyleIdx;
	}

	void resetDTMSettings()
	{
		resourcesCollection.dtmSettings.vector.clear();
		activeDTMSettingsIndex = InvalidDTMSettingsIdx;
	}
	
	// MSDF Hashing and Caching Internal Functions 
	enum class MSDFType : uint8_t
	{
		HATCH_FILL_PATTERN,
		FONT_GLYPH,
	};

	struct MSDFInputInfo
	{
		// It's a font glyph
		MSDFInputInfo(core::blake3_hash_t fontFaceHash, uint32_t glyphIdx)
			: type(MSDFType::FONT_GLYPH)
			, faceHash(fontFaceHash)
			, glyphIndex(glyphIdx)
		{
			computeBlake3Hash();
		}

		// It's a hatch fill pattern
		MSDFInputInfo(HatchFillPattern fillPattern)
			: type(MSDFType::HATCH_FILL_PATTERN)
			, faceHash({})
			, fillPattern(fillPattern)
		{
			computeBlake3Hash();
		}
		
		bool operator==(const MSDFInputInfo& rhs) const
		{ return hash == rhs.hash && glyphIndex == rhs.glyphIndex && type == rhs.type;
		}

		MSDFType type;
		uint8_t pad[3u]; // 3 bytes pad
		union
		{
			uint32_t glyphIndex;
			HatchFillPattern fillPattern;
		};
		static_assert(sizeof(uint32_t) == sizeof(HatchFillPattern));
		
		core::blake3_hash_t faceHash = {};
		core::blake3_hash_t hash = {}; // actual hash, we will check in == operator
		size_t lookupHash = 0ull; // for containers expecting size_t hash


	private:
		
		void computeBlake3Hash()
		{
			core::blake3_hasher hasher;
			hasher.update(&type, sizeof(MSDFType));
			hasher.update(&glyphIndex, sizeof(uint32_t));
			hasher.update(&faceHash, sizeof(core::blake3_hash_t));
			hash = static_cast<core::blake3_hash_t>(hasher);
			lookupHash = std::hash<core::blake3_hash_t>{}(hash); // hashing the hash :D
		}

	};

	struct MSDFInputInfoHash { std::size_t operator()(const MSDFInputInfo& info) const { return info.lookupHash; } };
	
	struct MSDFReference
	{
		uint32_t alloc_idx;
		uint64_t lastUsedFrameIndex;

		MSDFReference(uint32_t alloc_idx, uint64_t semaphoreVal) : alloc_idx(alloc_idx), lastUsedFrameIndex(semaphoreVal) {}
		MSDFReference(uint64_t currentFrameIndex) : MSDFReference(InvalidTextureIndex, currentFrameIndex) {}
		MSDFReference() : MSDFReference(InvalidTextureIndex, ~0ull) {}

		// In LRU Cache `insert` function, in case of cache hit, we need to assign semaphore value to MSDFReference without changing `alloc_idx`
		inline MSDFReference& operator=(uint64_t currentFrameIndex) { lastUsedFrameIndex = currentFrameIndex; return *this;  }
	};
	
	uint32_t getMSDFIndexFromInputInfo(const MSDFInputInfo& msdfInfo, const SIntendedSubmitInfo& intendedNextSubmit);
	
	uint32_t addMSDFTexture(const MSDFInputInfo& msdfInput, core::smart_refctd_ptr<ICPUImage>&& cpuImage, SIntendedSubmitInfo& intendedNextSubmit);
	
	// Flushes Current Draw Call and adds to drawCalls
	void flushDrawObjects();

	// FrameIndex used as a criteria for resource/image eviction in case of limitations
	uint32_t currentFrameIndex = 0u;

	// Replay Cache override
	ReplayCache* currentReplayCache = nullptr;

	// DrawCalls Data
	uint64_t drawObjectsFlushedToDrawCalls = 0ull;
	std::vector<DrawCallData> drawCalls; // either dtms or objects

	// ResourcesCollection and packed into GPUBuffer
	ResourcesCollection resourcesCollection;
	nbl::core::smart_refctd_ptr<IGPUBuffer> resourcesGPUBuffer;
	size_t copiedResourcesSize;

	// GPUImages Memory Arena + AddressAllocator
	IDeviceMemoryAllocator::SAllocation imagesMemoryArena;
	smart_refctd_ptr<ImagesMemorySubAllocator> imagesMemorySubAllocator;

	// Members
	smart_refctd_ptr<IUtilities> m_utilities;
	IQueue* m_copyQueue;

	// Active Resources we need to keep track of and push to resources buffer if needed.
	LineStyleInfo activeLineStyle;
	uint32_t activeLineStyleIndex = InvalidStyleIdx;

	DTMSettingsInfo activeDTMSettings;
	uint32_t activeDTMSettingsIndex = InvalidDTMSettingsIdx;

	MainObjectType activeMainObjectType;
	TransformationType activeMainObjectTransformationType;

	uint32_t activeMainObjectIndex = InvalidMainObjectIdx;

	// The ClipRects & Projections are stack, because user can push/pop ClipRects & Projections in any order
	std::deque<float64_t3x3> activeProjections; // stack of projections stored so we can resubmit them if geometry buffer got reset.
	std::deque<uint32_t> activeProjectionIndices; // stack of projection gpu addresses in geometry buffer. to keep track of them in push/pops
	
	std::deque<WorldClipRect> activeClipRects; // stack of clips stored so we can resubmit them if geometry buffer got reset.
	std::deque<uint32_t> activeClipRectIndices; // stack of clips gpu addresses in geometry buffer. to keep track of them in push/pops

	GetGlyphMSDFTextureFunc getGlyphMSDF;
	GetHatchFillPatternMSDFTextureFunc getHatchFillPatternMSDF;

	using MSDFsLRUCache = core::ResizableLRUCache<MSDFInputInfo, MSDFReference, MSDFInputInfoHash>;
	smart_refctd_ptr<IGPUImageView>		msdfTextureArray; // view to the resource holding all the msdfs in it's layers
	smart_refctd_ptr<IndexAllocator>	msdfTextureArrayIndexAllocator;
	std::unique_ptr<MSDFsLRUCache>		msdfLRUCache; // LRU Cache to evict Least Recently Used in case of overflow

	std::vector<MSDFImageState>			msdfImagesState = {}; // cached cpu imaged + their status, size equals to LRUCache size
	static constexpr asset::E_FORMAT	MSDFTextureFormat = asset::E_FORMAT::EF_R8G8B8A8_SNORM;
	bool m_hasInitializedMSDFTextureArrays = false;
	
	// Images:
	std::unique_ptr<ImagesCache> imagesCache;
	smart_refctd_ptr<SubAllocatedDescriptorSet> suballocatedDescriptorSet;
	uint32_t imagesArrayBinding = 0u;

	std::unordered_map<image_id, std::vector<StreamedImageCopy>> streamedImageCopies;
};

