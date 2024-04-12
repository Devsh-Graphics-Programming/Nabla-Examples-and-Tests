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

// ! this is just a buffers filler with autosubmission features used for convenience to how you feed our CAD renderer
struct DrawBuffersFiller
{
public:

	typedef uint32_t index_buffer_type;

	DrawBuffersFiller() {}

	DrawBuffersFiller(smart_refctd_ptr<IUtilities>&& utils, IQueue* copyQueue);

	typedef std::function<void(SIntendedSubmitInfo&)> SubmitFunc;

	// function is called when buffer is filled and we should submit draws and clear the buffers and continue filling
	void setSubmitDrawsFunction(SubmitFunc func);

	void allocateIndexBuffer(ILogicalDevice* logicalDevice, uint32_t indices);

	void allocateMainObjectsBuffer(ILogicalDevice* logicalDevice, uint32_t mainObjects);

	void allocateDrawObjectsBuffer(ILogicalDevice* logicalDevice, uint32_t drawObjects);

	void allocateGeometryBuffer(ILogicalDevice* logicalDevice, size_t size);

	void allocateStylesBuffer(ILogicalDevice* logicalDevice, uint32_t lineStylesCount);

	//! this function fills buffers required for drawing a polyline and submits a draw through provided callback when there is not enough memory.
	void drawPolyline(const CPolylineBase& polyline, const LineStyleInfo& lineStyleInfo, SIntendedSubmitInfo& intendedNextSubmit);

	void drawPolyline(const CPolylineBase& polyline, uint32_t polylineMainObjIdx, SIntendedSubmitInfo& intendedNextSubmit);
	
	// !drawSolid
	void drawHatch(
		const Hatch& hatch,
		const float32_t4& foregroundColor, 
		const float32_t4& backgroundColor,
		/* something that gets you the texture id */
		SIntendedSubmitInfo& intendedNextSubmit);

	void drawHatch(
		const Hatch& hatch,
		const float32_t4& color,
		/* something that gets you the texture id */
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

protected:

	SubmitFunc submitDraws;
	static constexpr uint32_t InvalidStyleIdx = ~0u;

	void finalizeMainObjectCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);

	void finalizeGeometryCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);

	void finalizeLineStyleCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);
	
	void finalizeCustomClipProjectionCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);
	
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
};
