#include "Polyline.h"
#include "Hatch.h"
#include <nbl/video/utilities/SIntendedSubmitInfo.h>

using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;

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

	// function is called when buffer is filled and we should submit draws and clear the buffers and continue filling
	void setSubmitDrawsFunction(SubmitFunc func);

	void allocateIndexBuffer(ILogicalDevice* logicalDevice, uint32_t indices);

	void allocateMainObjectsBuffer(ILogicalDevice* logicalDevice, uint32_t mainObjects);

	void allocateDrawObjectsBuffer(ILogicalDevice* logicalDevice, uint32_t drawObjects);

	void allocateGeometryBuffer(ILogicalDevice* logicalDevice, size_t size);

	void allocateStylesBuffer(ILogicalDevice* logicalDevice, uint32_t lineStylesCount);
	
	void allocateMSDFTextures(ILogicalDevice* logicalDevice, uint32_t maxMSDFs);
	
	using texture_hash = uint64_t;
	static constexpr uint64_t InvalidTextureHash = std::numeric_limits<uint64_t>::max();
	
	// ! return index to be used later in hatch fill style or text glyph object
	void addMSDFTexture(ICPUBuffer const* srcBuffer, const asset::IImage::SBufferCopy& region, texture_hash hash, SIntendedSubmitInfo& intendedNextSubmit);

	//! this function fills buffers required for drawing a polyline and submits a draw through provided callback when there is not enough memory.
	void drawPolyline(const CPolylineBase& polyline, const LineStyleInfo& lineStyleInfo, SIntendedSubmitInfo& intendedNextSubmit);

	void drawPolyline(const CPolylineBase& polyline, uint32_t polylineMainObjIdx, SIntendedSubmitInfo& intendedNextSubmit);
	
	// ! Convinience function for Hatch with MSDF Pattern and a solid background
	void drawHatch(
		const Hatch& hatch,
		const float32_t4& foregroundColor, 
		const float32_t4& backgroundColor,
		const texture_hash msdfTexture,
		SIntendedSubmitInfo& intendedNextSubmit);
	
	// ! Hatch with MSDF Pattern
	void drawHatch(
		const Hatch& hatch,
		const float32_t4& color,
		const texture_hash msdfTexture,
		SIntendedSubmitInfo& intendedNextSubmit);

	// ! Solid Fill Hacth
	void drawHatch(
		const Hatch& hatch,
		const float32_t4& color,
		SIntendedSubmitInfo& intendedNextSubmit);
	
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

protected:
	
	struct TextureCopy
	{
		ICPUBuffer const* srcBuffer;
		const asset::IImage::SBufferCopy& region;
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

	std::stack<ClipProjectionData> clipProjections; // stack of clip projectios stored so we can resubmit them if geometry buffer got reset.
	std::deque<uint64_t> clipProjectionAddresses; // stack of clip projection gpu addresses in geometry buffer. to keep track of them in push/pops
	
	struct TextureReference
	{
		uint32_t alloc_idx;
		uint64_t lastUsedSemaphoreValue;

		TextureReference(uint32_t alloc_idx, uint64_t semaphoreVal) : alloc_idx(alloc_idx), lastUsedSemaphoreValue(semaphoreVal) {}
		TextureReference(uint64_t semaphoreVal) : TextureReference(InvalidTextureIdx, semaphoreVal) {}
		TextureReference() : TextureReference(InvalidTextureIdx, ~0ull) {}

		// In LRU Cache `insert` function, in case of cache hit, we need to assign semaphore value to TextureReference without changing `alloc_idx`
		inline TextureReference& operator=(uint64_t semamphoreVal) { lastUsedSemaphoreValue = semamphoreVal; return *this;  }
	};

	using TextureLRUCache = core::LRUCache<texture_hash, TextureReference>;

	smart_refctd_ptr<IGPUImageView>		msdfTextureArray; // view to the resource holding all the msdfs in it's layers
	std::vector<TextureCopy>			textureCopies; // queued up texture copies, @Lucas change to deque if possible
	TextureLRUCache						textureLRUCache; // LRU Cache to evict Least Recently Used in case of overflow
};
