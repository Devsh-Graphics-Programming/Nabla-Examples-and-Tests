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
static_assert(sizeof(MainObject) == 12u);
static_assert(sizeof(LineStyle) == 88u);
static_assert(sizeof(ClipProjectionData) == 88u);

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
		virtual size_t getAlignedStorageSize() const { core::alignUp(getStorageSize(), ResourcesMaxNaturalAlignment); }
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
	};

	/// @brief struct to hold all resources
	struct ResourcesCollection
	{
		// auto-submission level 0 resources (settings that mainObj references)
		// Not enough VRAM available to serve adding one of the level 0 resources: they clear themselves and everything from higher levels after doing submission
		CPUGeneratedResource<LineStyle> lineStyles;
		CPUGeneratedResource<DTMSettings> dtmSettings;
		CPUGeneratedResource<ClipProjectionData> clipProjections;
	
		// auto-submission level 1 buffers (mainObj that drawObjs references, if all drawObjs+idxBuffer+geometryInfo doesn't fit into mem this will be broken down into many)
		CPUGeneratedResource<MainObject> mainObjects;

		// auto-submission level 2 buffers
		CPUGeneratedResource<DrawObject> drawObjects;
		CPUGeneratedResource<uint32_t> indexBuffer; // this is going to change to ReservedComputeResource where index buffer gets filled by compute shaders
		CPUGeneratedResource<uint8_t> geometryInfo; // general purpose byte buffer for custom geometries, etc

		// Get Total memory consumption, If all ResourcesCollection get packed together with ResourcesMaxNaturalAlignment
		// used to decide when to overflow
		size_t calculateTotalConsumption() const
		{
			return
				lineStyles.getAlignedStorageSize() +
				dtmSettings.getAlignedStorageSize() +
				clipProjections.getAlignedStorageSize() +
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
	consteval size_t getMinimumRequiredResourcesBufferSize() const
	{
		// for auto-submission to work correctly, memory needs to serve at least 2 linestyle, 1 dtm settings, 1 clip proj, 1 main obj, 1 draw obj and 512 bytes of additional mem for geometries and index buffer
		// this is the ABSOLUTE MINIMUM (if this value is used rendering will probably be as slow as CPU drawing :D)
		return core::alignUp(sizeof(LineStyle) * 2u + sizeof(DTMSettings) + sizeof(ClipProjectionData) + sizeof(MainObject) + sizeof(DrawObject) + 512ull, ResourcesMaxNaturalAlignment);
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

	void drawPolyline(const CPolylineBase& polyline, uint32_t polylineMainObjIdx, SIntendedSubmitInfo& intendedNextSubmit);
	
	void drawTriangleMesh(const CTriangleMesh& mesh, CTriangleMesh::DrawData& drawData, const DTMSettingsInfo& dtmSettings, SIntendedSubmitInfo& intendedNextSubmit);

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

	// ! Draw Font Glyph, will auto submit if there is no space
	void drawFontGlyph(
		nbl::ext::TextRendering::FontFace* fontFace,
		uint32_t glyphIdx,
		float64_t2 topLeft,
		float32_t2 dirU,
		float32_t  aspectRatio,
		float32_t2 minUV,
		uint32_t mainObjIdx,
		SIntendedSubmitInfo& intendedNextSubmit);
	
	void _test_addImageObject(
		float64_t2 topLeftPos,
		float32_t2 size,
		float32_t rotation,
		SIntendedSubmitInfo& intendedNextSubmit);

	bool finalizeAllCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);

	void reset()
	{
		resetDrawObjects();
		resetMainObjects();
		resetLineStyles();
		resetDTMSettings();
	}

	/// @brief collection of all the resources that will eventually be reserved or copied to in the resourcesGPUBuffer, will be accessed via individual BDA pointers in shaders
	const ResourcesCollection& getResourcesCollection() const { return &resourcesCollection; }

	/// @brief buffer containing all non-texture type resources
	nbl::core::smart_refctd_ptr<IGPUBuffer> getResourcesGPUBuffer() const { return resourcesGPUBuffer; }

	/// @return how far resourcesGPUBuffer was copied to by `finalizeAllCopiesToGPU` in `resourcesCollection` 
	const size_t getCopiedResourcesSize() { return copiedResourcesSize; }

	uint32_t addLineStyle_SubmitIfNeeded(const LineStyleInfo& lineStyle, SIntendedSubmitInfo& intendedNextSubmit);

	uint32_t addDTMSettings_SubmitIfNeeded(const DTMSettingsInfo& dtmSettings, SIntendedSubmitInfo& intendedNextSubmit);
	
	// TODO[Przemek]: Read after reading the fragment shader comments and having a basic understanding of the relationship between "mainObject" and our programmable blending resolve:
	// Use `addMainObject_SubmitIfNeeded` to push your single mainObject you'll be using for the enitre triangle mesh (this will ensure overlaps between triangles of the same mesh is resolved correctly)
	// Delete comment when you understand this

	// [ADVANCED] Do not use this function unless you know what you're doing (It may cause auto submit)
	// Never call this function multiple times in a row before indexing it in a drawable, because future auto-submits may invalidate mainObjects, so do them one by one, for example:
	// Valid: addMainObject1 --> addXXX(mainObj1) ---> addMainObject2 ---> addXXX(mainObj2) ....
	// Invalid: addMainObject1 ---> addMainObject2 ---> addXXX(mainObj1) ---> addXXX(mainObj2) ....
	uint32_t addMainObject_SubmitIfNeeded(uint32_t styleIdx, uint32_t dtmSettingsIdx, SIntendedSubmitInfo& intendedNextSubmit);

	// we need to store the clip projection stack to make sure the front is always available in memory
	void pushClipProjectionData(const ClipProjectionData& clipProjectionData);
	void popClipProjectionData();
	const std::deque<ClipProjectionData>& getClipProjectionStack() const { return clipProjections; }

	smart_refctd_ptr<IGPUImageView> getMSDFsTextureArray() { return msdfTextureArray; }

	uint32_t2 getMSDFResolution() {
		auto extents = msdfTextureArray->getCreationParameters().image->getCreationParameters().extent;
		return uint32_t2(extents.width, extents.height);
	}
	uint32_t getMSDFMips() {
		return msdfTextureArray->getCreationParameters().image->getCreationParameters().mipLevels;
	}

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

	// Internal Function to call whenever we overflow when we can't fill all of mainObject's drawObjects
	void submitCurrentDrawObjectsAndReset(SIntendedSubmitInfo& intendedNextSubmit, uint32_t mainObjectIndex);

	/// @return index to added main object.
	///		It will return `InvalidMainObjectIndex` if it there isn't enough remaining resources memory OR the index would exceed MaxIndexableMainObjects
	uint32_t addMainObject_Internal(const MainObject& mainObject);

	uint32_t addLineStyle_Internal(const LineStyleInfo& lineStyleInfo);

	uint32_t addDTMSettings_Internal(const DTMSettingsInfo& dtmSettings, SIntendedSubmitInfo& intendedNextSubmit);

	// Gets the current clip projection data (the top of stack) gpu addreess inside the geometryBuffer
	// If it's been invalidated then it will request to upload again with a possible auto-submit on low geometry buffer memory.
	uint32_t acquireCurrentClipProjectionAddress(SIntendedSubmitInfo& intendedNextSubmit);
	
	uint32_t addClipProjectionData_SubmitIfNeeded(const ClipProjectionData& clipProjectionData, SIntendedSubmitInfo& intendedNextSubmit);

	static constexpr uint32_t getCageCountPerPolylineObject(ObjectType type)
	{
		if (type == ObjectType::LINE)
			return 1u;
		else if (type == ObjectType::QUAD_BEZIER)
			return 3u;
		return 0u;
	};

	void addPolylineObjects_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx);

	void addPolylineConnectors_Internal(const CPolylineBase& polyline, uint32_t& currentPolylineConnectorObj, uint32_t mainObjIdx);

	void addLines_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx);

	void addQuadBeziers_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx);

	void addHatch_Internal(const Hatch& hatch, uint32_t& currentObjectInSection, uint32_t mainObjIndex);
	
	bool addFontGlyph_Internal(const GlyphInfo& glyphInfo, uint32_t mainObjIdx);
	
	void resetMainObjects()
	{
		resourcesCollection.mainObjects.vector.clear();
	}

	// these resources are data related to chunks of a whole mainObject
	void resetDrawObjects()
	{
		resourcesCollection.drawObjects.vector.clear();
		resourcesCollection.indexBuffer.vector.clear();
		resourcesCollection.geometryInfo.vector.clear();
	}

	void resetCustomClipProjections()
	{
		resourcesCollection.clipProjections.vector.clear();
	}

	void resetLineStyles()
	{
		resourcesCollection.lineStyles.vector.clear();
	}

	void resetDTMSettings()
	{
		resourcesCollection.dtmSettings.vector.clear();
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
	uint32_t addMSDFTexture(const MSDFInputInfo& msdfInput, core::smart_refctd_ptr<ICPUImage>&& cpuImage, uint32_t mainObjIdx, SIntendedSubmitInfo& intendedNextSubmit);
	
	// ResourcesCollection and packed into GPUBuffer
	ResourcesCollection resourcesCollection;
	nbl::core::smart_refctd_ptr<IGPUBuffer> resourcesGPUBuffer;
	size_t copiedResourcesSize;

	// Members
	smart_refctd_ptr<IUtilities> m_utilities;
	IQueue* m_copyQueue;

	std::deque<ClipProjectionData> clipProjections; // stack of clip projectios stored so we can resubmit them if geometry buffer got reset.
	std::deque<uint64_t> clipProjectionAddresses; // stack of clip projection gpu addresses in geometry buffer. to keep track of them in push/pops

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

