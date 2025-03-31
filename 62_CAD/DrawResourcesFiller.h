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
	
	/// @brief general parent struct for 1.ComputeReserved and 2.CPUFilled DrawBuffers
	struct DrawBuffer
	{
		static constexpr size_t Alignment = 8u;
		static constexpr size_t InvalidBufferOffset = ~0u;
		size_t bufferOffset = InvalidBufferOffset; // set when copy to gpu buffer is issued
		virtual size_t getCount() const = 0;
		virtual size_t getStorageSize() const = 0;
		virtual size_t getAlignedStorageSize() const { core::alignUp(getStorageSize(), Alignment); }
	};

	/// @brief DrawBuffer reserved for compute shader stages input/output
	template <typename T>
	struct ComputeReservedDrawBuffer : DrawBuffer
	{
		size_t count = 0ull;
		size_t getCount() const override { return count; }
		size_t getStorageSize() const override  { return count * sizeof(T); }
	};

	/// @brief DrawBuffer which is filled by CPU, packed and sent to GPU
	template <typename T>
	struct CPUFilledDrawBuffer : DrawBuffer
	{
		core::vector<T> vector;
		size_t getCount() const { return vector.size(); }
		size_t getStorageSize() const { return vector.size() * sizeof(T); }
	};

	/// @brief struct to hold all draw buffers
	struct DrawBuffers
	{
		// auto-submission level 0 buffers (settings that mainObj references)
		CPUFilledDrawBuffer<LineStyle> lineStyles;
		CPUFilledDrawBuffer<DTMSettings> dtmSettings;
		CPUFilledDrawBuffer<ClipProjectionData> clipProjections;
	
		// auto-submission level 1 buffers (mainObj that drawObjs references, if all drawObjs+idxBuffer+geometryInfo doesn't fit into mem this will be broken down into many)
		CPUFilledDrawBuffer<MainObject> mainObjects;

		// auto-submission level 2 buffers
		CPUFilledDrawBuffer<DrawObject> drawObjects;
		CPUFilledDrawBuffer<uint32_t> indexBuffer;
		CPUFilledDrawBuffer<uint8_t> geometryInfo; // general purpose byte buffer for custom geometries, etc

		// Get Total memory consumption, If all DrawBuffers get packed together with DrawBuffer::Alignment
		// Useful to know when to know when to overflow
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

	void allocateDrawResourcesBuffer(ILogicalDevice* logicalDevice, size_t size);

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
		resetGeometryCounters();
		resetMainObjectCounters();
		resetLineStyleCounters();
		resetDTMSettingsCounters();
	}

	DrawBuffers drawBuffers; // will be compacted and copied into gpu draw resources
	nbl::core::smart_refctd_ptr<IGPUBuffer> drawResourcesGPUBuffer;

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

	// Internal Function to call whenever we overflow while filling our buffers with geometry (potential limiters: indexBuffer, drawObjectsBuffer or geometryBuffer)
	// ! mainObjIdx: is the mainObject the "overflowed" drawObjects belong to.
	//		mainObjIdx is required to ensure that valid data, especially the `clipProjectionData`, remains linked to the main object.
	//		This is important because, while other data may change during overflow handling, the main object must persist to maintain consistency throughout rendering all parts of it. (for example all lines and beziers of a single polyline)
	//		[ADVANCED] If you have not created your mainObject yet, pass `InvalidMainObjectIdx` (See drawHatch)
	void submitCurrentDrawObjectsAndReset(SIntendedSubmitInfo& intendedNextSubmit, uint32_t mainObjectIndex);

	uint32_t addMainObject_Internal(const MainObject& mainObject);

	uint32_t addLineStyle_Internal(const LineStyleInfo& lineStyleInfo);

	uint32_t addDTMSettings_Internal(const DTMSettingsInfo& dtmSettings, SIntendedSubmitInfo& intendedNextSubmit);

	// Gets the current clip projection data (the top of stack) gpu addreess inside the geometryBuffer
	// If it's been invalidated then it will request to upload again with a possible auto-submit on low geometry buffer memory.
	uint64_t acquireCurrentClipProjectionAddress(SIntendedSubmitInfo& intendedNextSubmit);
	
	uint64_t addClipProjectionData_SubmitIfNeeded(const ClipProjectionData& clipProjectionData, SIntendedSubmitInfo& intendedNextSubmit);

	uint64_t addClipProjectionData_Internal(const ClipProjectionData& clipProjectionData);

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
	
	void resetMainObjectCounters()
	{
		inMemMainObjectCount = 0u;
		currentMainObjectCount = 0u;
	}

	// WARN: If you plan to use this, make sure you either reset the mainObjectCounters as well
	//			Or if you want to keep your  mainObject around, make sure you're using the `submitCurrentObjectsAndReset` function instead of calling this directly
	//			So that it makes your mainObject point to the correct clipProjectionData (which exists in the geometry buffer)
	void resetGeometryCounters()
	{
		inMemDrawObjectCount = 0u;
		currentDrawObjectCount = 0u;

		inMemGeometryBufferSize = 0u;
		currentGeometryBufferSize = 0u;

		// Invalidate all the clip projection addresses because geometry buffer got reset
		for (auto& clipProjAddr : clipProjectionAddresses)
			clipProjAddr = InvalidClipProjectionAddress;
	}

	void resetLineStyleCounters()
	{
		currentLineStylesCount = 0u;
		inMemLineStylesCount = 0u;
	}

	void resetDTMSettingsCounters()
	{
		currentDTMSettingsCount = 0u;
		inMemDTMSettingsCount = 0u;
	}

	MainObject* getMainObject(uint32_t idx)
	{
		MainObject* mainObjsArray = reinterpret_cast<MainObject*>(cpuDrawBuffers.mainObjectsBuffer->getPointer());
		return &mainObjsArray[idx];
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
	
	// Members
	smart_refctd_ptr<IUtilities> m_utilities;
	IQueue* m_copyQueue;

	uint64_t drawResourcesBDA = 0u; // Actual BDA offset 0 of the gpu buffer

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

