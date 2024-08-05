#pragma once
#include "Polyline.h"
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
static_assert(sizeof(MainObject) == 16u);
static_assert(sizeof(Globals) == 128u);
static_assert(sizeof(LineStyle) == 96u);
static_assert(sizeof(ClipProjectionData) == 88u);

template <typename BufferType>
struct DrawBuffers
{
	smart_refctd_ptr<BufferType> indexBuffer; // only is valid for IGPUBuffer because it's filled at allocation time and never touched again
	smart_refctd_ptr<BufferType> mainObjectsBuffer;
	smart_refctd_ptr<BufferType> drawObjectsBuffer;
	smart_refctd_ptr<BufferType> geometryBuffer;
	smart_refctd_ptr<BufferType> lineStylesBuffer;
};

// ! DrawResourcesFiller
// ! This class provides important functionality to manage resources needed for a draw.
// ! Drawing new objects (polylines, hatches, etc.) should go through this function.
// ! Contains all the scene resources (buffers and images)
// ! In the case of overflow (i.e. not enough remaining v-ram) will auto-submit/render everything recorded so far,
//   and additionally makes sure relavant data needed for those draw calls are present in memory
struct DrawResourcesFiller
{
public:

	typedef uint32_t index_buffer_type;

	DrawResourcesFiller();

	DrawResourcesFiller(smart_refctd_ptr<IUtilities>&& utils, IQueue* copyQueue);

	typedef std::function<void(SIntendedSubmitInfo&)> SubmitFunc;
	void setSubmitDrawsFunction(const SubmitFunc& func);

	void allocateIndexBuffer(ILogicalDevice* logicalDevice, uint32_t indices);

	void allocateMainObjectsBuffer(ILogicalDevice* logicalDevice, uint32_t mainObjects);

	void allocateDrawObjectsBuffer(ILogicalDevice* logicalDevice, uint32_t drawObjects);

	void allocateGeometryBuffer(ILogicalDevice* logicalDevice, size_t size);

	void allocateStylesBuffer(ILogicalDevice* logicalDevice, uint32_t lineStylesCount);
	
	void allocateMSDFTextures(ILogicalDevice* logicalDevice, uint32_t maxMSDFs, uint32_t2 msdfsExtent);

	// functions that user should set to get MSDF texture if it's not available in cache.
	// it's up to user to return cached or generate on the fly.
	typedef std::function<core::smart_refctd_ptr<ICPUBuffer>(nbl::ext::TextRendering::FontFace* /*face*/, uint32_t /*glyphIdx*/)> GetGlyphMSDFTextureFunc;
	typedef std::function<core::smart_refctd_ptr<ICPUBuffer>(HatchFillPattern/*pattern*/)> GetHatchFillPatternMSDFTextureFunc;
	void setGlyphMSDFTextureFunction(const GetGlyphMSDFTextureFunc& func);
	void setHatchFillMSDFTextureFunction(const GetHatchFillPatternMSDFTextureFunc& func);

	//! this function fills buffers required for drawing a polyline and submits a draw through provided callback when there is not enough memory.
	void drawPolyline(const CPolylineBase& polyline, const LineStyleInfo& lineStyleInfo, SIntendedSubmitInfo& intendedNextSubmit);

	void drawPolyline(const CPolylineBase& polyline, uint32_t polylineMainObjIdx, SIntendedSubmitInfo& intendedNextSubmit);
	
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
		SIntendedSubmitInfo& intendedNextSubmit)
	{
		auto addImageObject_Internal = [&](const ImageObjectInfo& imageObjectInfo, uint32_t mainObjIdx) -> bool
			{
				const auto maxGeometryBufferImageObjects = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(ImageObjectInfo);
				uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
				uploadableObjects = min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
				uploadableObjects = min(uploadableObjects, maxGeometryBufferImageObjects);

				if (uploadableObjects >= 1u)
				{
					void* dstGeom = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
					memcpy(dstGeom, &imageObjectInfo, sizeof(ImageObjectInfo));
					uint64_t geomBufferAddr = geometryBufferAddress + currentGeometryBufferSize;
					currentGeometryBufferSize += sizeof(ImageObjectInfo);

					DrawObject drawObj = {};
					drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::IMAGE) | (0 << 16)); // TODO: use custom pack/unpack function
					drawObj.mainObjIndex = mainObjIdx;
					drawObj.geometryAddress = geomBufferAddr;
					void* dstDrawObj = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
					memcpy(dstDrawObj, &drawObj, sizeof(DrawObject));
					currentDrawObjectCount += 1u;

					return true;
				}
				else
					return false;
			};
		
		uint32_t mainObjIdx = addMainObject_SubmitIfNeeded(InvalidStyleIdx, intendedNextSubmit);

		ImageObjectInfo info = {};
		info.topLeft = topLeftPos;
		info.dirU = float32_t2(size.x * cos(rotation), size.x * sin(rotation)); // 
		info.aspectRatio = size.y / size.x;
		info.textureID = 0u;
		if (!addImageObject_Internal(info, mainObjIdx))
		{
			// single image object couldn't fit into memory to push to gpu, so we submit rendering current objects and reset geometry buffer and draw objects
			submitCurrentObjectsAndReset(intendedNextSubmit, mainObjIdx);
			bool success = addImageObject_Internal(info, mainObjIdx);
			assert(success); // this should always be true, otherwise it's either bug in code or not enough memory allocated to hold a single image object 
		}
	}

	void finalizeAllCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);

	inline uint32_t getLineStyleCount() const { return currentLineStylesCount; }

	inline uint32_t getDrawObjectCount() const { return currentDrawObjectCount; } 

	inline uint32_t getMainObjectCount() const { return currentMainObjectCount; }

	inline size_t getCurrentMainObjectsBufferSize() const
	{
		return sizeof(MainObject) * currentMainObjectCount;
	}

	inline size_t getCurrentDrawObjectsBufferSize() const
	{
		return sizeof(DrawObject) * currentDrawObjectCount;
	}

	inline size_t getCurrentGeometryBufferSize() const
	{
		return currentGeometryBufferSize;
	}

	inline size_t getCurrentLineStylesBufferSize() const
	{
		return sizeof(LineStyle) * currentLineStylesCount;
	}

	void reset()
	{
		resetGeometryCounters();
		resetMainObjectCounters();
		resetLineStyleCounters();
	}

	DrawBuffers<ICPUBuffer> cpuDrawBuffers;
	DrawBuffers<IGPUBuffer> gpuDrawBuffers;

	uint32_t addLineStyle_SubmitIfNeeded(const LineStyleInfo& lineStyle, SIntendedSubmitInfo& intendedNextSubmit);
	
	uint32_t addMainObject_SubmitIfNeeded(uint32_t styleIdx, SIntendedSubmitInfo& intendedNextSubmit);

	// we need to store the clip projection stack to make sure the front is always available in memory
	void pushClipProjectionData(const ClipProjectionData& clipProjectionData);
	void popClipProjectionData();

	smart_refctd_ptr<IGPUImageView> getMSDFsTextureArray() { return msdfTextureArray; }

	uint32_t2 getMSDFResolution() {
		auto extents = msdfTextureArray->getCreationParameters().image->getCreationParameters().extent;
		return uint32_t2(extents.width, extents.height);
	}

protected:
	
	struct TextureCopy
	{
		core::smart_refctd_ptr<ICPUBuffer> srcBuffer;
		uint64_t bufferOffset;
		uint32_t3 imageExtent;
		uint32_t index;
	};

	SubmitFunc submitDraws;
	
	static constexpr uint32_t InvalidStyleIdx = ~0u;

	void finalizeMainObjectCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);

	void finalizeGeometryCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);

	void finalizeLineStyleCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);
	
	void finalizeCustomClipProjectionCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);
	
	void finalizeTextureCopies(SIntendedSubmitInfo& intendedNextSubmit);

	// A hatch and a polyline are considered a "Main Object" which consists of smaller geometries such as beziers, lines, connectors, hatchBoxes
	// If the whole polyline can't fit into memory for draw, then we submit the render of smaller geometries midway and continue
	void submitCurrentObjectsAndReset(SIntendedSubmitInfo& intendedNextSubmit, uint32_t mainObjectIndex);

	uint32_t addMainObject_Internal(const MainObject& mainObject);

	uint32_t addLineStyle_Internal(const LineStyleInfo& lineStyleInfo);

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

	MainObject* getMainObject(uint32_t idx)
	{
		MainObject* mainObjsArray = reinterpret_cast<MainObject*>(cpuDrawBuffers.mainObjectsBuffer->getPointer());
		return &mainObjsArray[idx];
	}

	// MSDF Hashing and Caching Internal Functions 
	static constexpr uint64_t InvalidMSDFHash = std::numeric_limits<uint64_t>::max();
	using msdf_hash = std::size_t;
	enum class MSDFType : uint8_t
	{
		HATCH_FILL_PATTERN,
		FONT_GLYPH,
	};

	static msdf_hash hashFillPattern(HatchFillPattern fillPattern);

	static msdf_hash hashFontGlyph(size_t fontHash, uint32_t glyphIndex);

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
	
	uint32_t getMSDFTextureIndex(msdf_hash hash);
	
	uint32_t getTextureIndexFromHash(const msdf_hash msdfTexture, SIntendedSubmitInfo& intendedNextSubmit)
	{
		uint32_t textureIdx = InvalidTextureIdx;
		if (msdfTexture != InvalidMSDFHash)
		{
			MSDFReference* tRef = textureLRUCache->get(msdfTexture);
			if (tRef)
			{
				textureIdx = tRef->alloc_idx;
				tRef->lastUsedSemaphoreValue = intendedNextSubmit.getFutureScratchSemaphore().value; // update this because the texture will get used on the next submit
			}
		}
		return textureIdx;
	}

	uint32_t addMSDFTexture(std::function<core::smart_refctd_ptr<ICPUBuffer>()> createResourceIfEmpty, msdf_hash hash, SIntendedSubmitInfo& intendedNextSubmit);

	uint32_t addMSDFTexture(core::smart_refctd_ptr<ICPUBuffer> textureBuffer, msdf_hash hash, SIntendedSubmitInfo& intendedNextSubmit);
	
	// Members
	smart_refctd_ptr<IUtilities> m_utilities;
	IQueue* m_copyQueue;

	uint32_t maxIndexCount;

	uint32_t inMemMainObjectCount = 0u;
	uint32_t currentMainObjectCount = 0u;
	uint32_t maxMainObjects = 0u;

	uint32_t inMemDrawObjectCount = 0u;
	uint32_t currentDrawObjectCount = 0u;
	uint32_t maxDrawObjects = 0u;

	uint64_t inMemGeometryBufferSize = 0u;
	uint64_t currentGeometryBufferSize = 0u;
	uint64_t maxGeometryBufferSize = 0u;

	uint32_t inMemLineStylesCount = 0u;
	uint32_t currentLineStylesCount = 0u;
	uint32_t maxLineStyles = 0u;

	uint64_t geometryBufferAddress = 0u; // Actual BDA offset 0 of the gpu buffer

	std::deque<ClipProjectionData> clipProjections; // stack of clip projectios stored so we can resubmit them if geometry buffer got reset.
	std::deque<uint64_t> clipProjectionAddresses; // stack of clip projection gpu addresses in geometry buffer. to keep track of them in push/pops

	// MSDF
	GetGlyphMSDFTextureFunc getGlyphMSDF;
	GetHatchFillPatternMSDFTextureFunc getHatchFillPatternMSDF;

	using MSDFsLRUCache = core::LRUCache<msdf_hash, MSDFReference>;
	smart_refctd_ptr<IGPUImageView>		msdfTextureArray; // view to the resource holding all the msdfs in it's layers
	smart_refctd_ptr<IndexAllocator>	msdfTextureArrayIndexAllocator;
	std::set<uint32_t>					msdfTextureArrayIndicesUsed = {}; // indices in the msdf texture array allocator that have been used in the current frame // TODO: make this a dynamic bitset
	std::vector<TextureCopy>			textureCopies = {}; // queued up texture copies, @Lucas change to deque if possible
	std::unique_ptr<MSDFsLRUCache>		textureLRUCache; // LRU Cache to evict Least Recently Used in case of overflow
	static constexpr asset::E_FORMAT	MSDFTextureFormat = asset::E_FORMAT::EF_R8G8B8_SNORM;

	bool m_hasInitializedMSDFTextureArrays = false;
};

