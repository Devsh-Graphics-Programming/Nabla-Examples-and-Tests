#pragma once
#include "Polyline.h"
#include "CTriangleMesh.h"
#include "Hatch.h"
#include "IndexAllocator.h"
#include <nbl/video/utilities/SIntendedSubmitInfo.h>
#include <nbl/core/containers/LRUCache.h>  
#include <nbl/ext/TextRendering/TextRendering.h>

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
	
	void _test_addImageObject(
		float64_t2 topLeftPos,
		float32_t2 size,
		float32_t rotation,
		SIntendedSubmitInfo& intendedNextSubmit);

	/// @brief call this function before submitting to ensure all resources are copied
	/// records copy command into intendedNextSubmit's active command buffer and might possibly submits if fails allocation on staging upload memory.
	bool finalizeAllCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);

	/// @brief  resets resources buffers
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
	const ResourcesCollection& getResourcesCollection() const { return resourcesCollection; }

	/// @brief buffer containing all non-texture type resources
	nbl::core::smart_refctd_ptr<IGPUBuffer> getResourcesGPUBuffer() const { return resourcesGPUBuffer; }

	/// @return how far resourcesGPUBuffer was copied to by `finalizeAllCopiesToGPU` in `resourcesCollection` 
	const size_t getCopiedResourcesSize() { return copiedResourcesSize; }

	// Setting Active Resources:
	void setActiveLineStyle(const LineStyleInfo& lineStyle);
	void setActiveDTMSettings(const DTMSettingsInfo& dtmSettingsInfo);

	void beginMainObject(MainObjectType type, TransformationType transformationType = TransformationType::NORMAL);
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
	uint32_t getActiveMainObjectIndex() const { return activeMainObjectIndex; }

	// TODO: Remove these later, these are for multiple draw calls instead of a single one.
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

	uint64_t drawObjectsFlushedToDrawCalls = 0ull;

	void flushDrawObjects()
	{
		if (resourcesCollection.drawObjects.getCount() > drawObjectsFlushedToDrawCalls)
		{
			DrawCallData drawCall = {};
			drawCall.isDTMRendering = false;
			drawCall.drawObj.drawObjectStart = drawObjectsFlushedToDrawCalls;
			drawCall.drawObj.drawObjectCount = resourcesCollection.drawObjects.getCount() - drawObjectsFlushedToDrawCalls;
			drawCalls.push_back(drawCall);
			drawObjectsFlushedToDrawCalls = resourcesCollection.drawObjects.getCount();
		}
	}

	std::vector<DrawCallData> drawCalls; // either dtms or objects


protected:
	
	struct MSDFTextureCopy
	{
		core::smart_refctd_ptr<ICPUImage> image;
		uint32_t index;
	};

	SubmitFunc submitDraws;
	
	bool finalizeBufferCopies(SIntendedSubmitInfo& intendedNextSubmit);

	bool finalizeTextureCopies(SIntendedSubmitInfo& intendedNextSubmit);

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
		uint64_t lastUsedSemaphoreValue;

		MSDFReference(uint32_t alloc_idx, uint64_t semaphoreVal) : alloc_idx(alloc_idx), lastUsedSemaphoreValue(semaphoreVal) {}
		MSDFReference(uint64_t semaphoreVal) : MSDFReference(InvalidTextureIdx, semaphoreVal) {}
		MSDFReference() : MSDFReference(InvalidTextureIdx, ~0ull) {}

		// In LRU Cache `insert` function, in case of cache hit, we need to assign semaphore value to MSDFReference without changing `alloc_idx`
		inline MSDFReference& operator=(uint64_t semamphoreVal) { lastUsedSemaphoreValue = semamphoreVal; return *this;  }
	};
	
	uint32_t getMSDFIndexFromInputInfo(const MSDFInputInfo& msdfInfo, SIntendedSubmitInfo& intendedNextSubmit)
	{
		uint32_t textureIdx = InvalidTextureIdx;
		MSDFReference* tRef = msdfLRUCache->get(msdfInfo);
		if (tRef)
		{
			textureIdx = tRef->alloc_idx;
			tRef->lastUsedSemaphoreValue = intendedNextSubmit.getFutureScratchSemaphore().value; // update this because the texture will get used on the next submit
		}
		return textureIdx;
	}
	
	// ! mainObjIdx: make sure to pass your mainObjIdx to it if you want it to stay synced/updated if some overflow submit occured which would potentially erase what your mainObject points at.
	// If you haven't created a mainObject yet, then pass InvalidMainObjectIdx
	uint32_t addMSDFTexture(const MSDFInputInfo& msdfInput, core::smart_refctd_ptr<ICPUImage>&& cpuImage, SIntendedSubmitInfo& intendedNextSubmit);
	
	// ResourcesCollection and packed into GPUBuffer
	ResourcesCollection resourcesCollection;
	nbl::core::smart_refctd_ptr<IGPUBuffer> resourcesGPUBuffer;
	size_t copiedResourcesSize;

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

	// MSDF
	GetGlyphMSDFTextureFunc getGlyphMSDF;
	GetHatchFillPatternMSDFTextureFunc getHatchFillPatternMSDF;

	using MSDFsLRUCache = core::LRUCache<MSDFInputInfo, MSDFReference, MSDFInputInfoHash>;
	smart_refctd_ptr<IGPUImageView>		msdfTextureArray; // view to the resource holding all the msdfs in it's layers
	smart_refctd_ptr<IndexAllocator>	msdfTextureArrayIndexAllocator;
	std::set<uint32_t>					msdfTextureArrayIndicesUsed = {}; // indices in the msdf texture array allocator that have been used in the current frame // TODO: make this a dynamic bitset
	std::vector<MSDFTextureCopy>		msdfTextureCopies = {}; // queued up texture copies
	std::unique_ptr<MSDFsLRUCache>		msdfLRUCache; // LRU Cache to evict Least Recently Used in case of overflow
	static constexpr asset::E_FORMAT	MSDFTextureFormat = asset::E_FORMAT::EF_R8G8B8A8_SNORM;

	bool m_hasInitializedMSDFTextureArrays = false;
};

