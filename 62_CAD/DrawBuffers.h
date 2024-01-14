#include "Polyline.h"
#include "Hatch.h"

template <typename BufferType>
struct DrawBuffers
{
	nbl::core::smart_refctd_ptr<BufferType> indexBuffer;
	nbl::core::smart_refctd_ptr<BufferType> mainObjectsBuffer;
	nbl::core::smart_refctd_ptr<BufferType> drawObjectsBuffer;
	nbl::core::smart_refctd_ptr<BufferType> geometryBuffer;
	nbl::core::smart_refctd_ptr<BufferType> lineStylesBuffer;
	nbl::core::smart_refctd_ptr<BufferType> customClipProjectionBuffer;
};

// ! this is just a buffers filler with autosubmission features used for convenience to how you feed our CAD renderer
struct DrawBuffersFiller
{
public:

	typedef uint32_t index_buffer_type;

	DrawBuffersFiller() {}

	DrawBuffersFiller(nbl::core::smart_refctd_ptr<nbl::video::IUtilities>&& utils);

	typedef std::function<nbl::video::IGPUQueue::SSubmitInfo(nbl::video::IGPUQueue*, nbl::video::IGPUFence*, nbl::video::IGPUQueue::SSubmitInfo)> SubmitFunc;

	// function is called when buffer is filled and we should submit draws and clear the buffers and continue filling
	void setSubmitDrawsFunction(SubmitFunc func);

	void allocateIndexBuffer(nbl::video::ILogicalDevice* logicalDevice, uint32_t indices);

	void allocateMainObjectsBuffer(nbl::video::ILogicalDevice* logicalDevice, uint32_t mainObjects);

	void allocateDrawObjectsBuffer(nbl::video::ILogicalDevice* logicalDevice, uint32_t drawObjects);

	void allocateGeometryBuffer(nbl::video::ILogicalDevice* logicalDevice, size_t size);

	void allocateStylesBuffer(nbl::video::ILogicalDevice* logicalDevice, uint32_t stylesCount);

	void allocateCustomClipProjectionBuffer(nbl::video::ILogicalDevice* logicalDevice, uint32_t ClipProjectionDataCount);

	// TODO
	//uint32_t getIndexCount() const { return currentIndexCount; }

	//! this function fills buffers required for drawing a polyline and submits a draw through provided callback when there is not enough memory.
	nbl::video::IGPUQueue::SSubmitInfo drawPolyline(
		const CPolylineBase& polyline,
		const CPULineStyle& cpuLineStyle,
		const uint32_t clipProjectionIdx,
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit);

	nbl::video::IGPUQueue::SSubmitInfo drawPolyline(
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit,
		const CPolylineBase& polyline,
		const uint32_t polylineMainObjIdx);

	// If we had infinite mem, we would first upload all curves into geometry buffer then upload the "CurveBoxes" with correct gpu addresses to those
	// But we don't have that so we have to follow a similar auto submission as the "drawPolyline" function with some mutations:
	// We have to find the MAX number of "CurveBoxes" we could draw, and since both the "Curves" and "CurveBoxes" reside in geometry buffer,
	// it has to be taken into account when calculating "how many curve boxes we could draw and when we need to submit/clear"
	// So same as drawPolylines, we would first try to fill the geometry buffer and index buffer that corresponds to "backfaces or even provoking vertices"
	// then change index buffer to draw front faces of the curveBoxes that already reside in geometry buffer memory
	// then if anything was left (the ones that weren't in memory for front face of the curveBoxes) we copy their geom to mem again and use frontface/oddProvoking vertex
	nbl::video::IGPUQueue::SSubmitInfo drawHatch(
		const Hatch& hatch,
		// If more parameters from cpu line style are used here later, make a new HatchStyle & use that
		const float32_t4 color,
		const uint32_t clipProjectionIdx,
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit);

	nbl::video::IGPUQueue::SSubmitInfo finalizeAllCopiesToGPU(
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit);

	inline uint32_t getIndexCount() const { return currentIndexCount; }

	inline uint32_t getLineStyleCount() const { return currentLineStylesCount; }

	inline uint32_t getDrawObjectCount() const { return currentDrawObjectCount; }

	inline uint32_t getMainObjectCount() const { return currentMainObjectCount; }

	inline size_t getCurrentIndexBufferSize() const
	{
		return sizeof(index_buffer_type) * currentIndexCount;
	}

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

	inline size_t getCurrentCustomClipProjectionBufferSize() const
	{
		return sizeof(ClipProjectionData) * currentClipProjectionDataCount;
	}

	void reset()
	{
		resetAllCounters();
	}

	DrawBuffers<nbl::asset::ICPUBuffer> cpuDrawBuffers;
	DrawBuffers<nbl::video::IGPUBuffer> gpuDrawBuffers;

	nbl::video::IGPUQueue::SSubmitInfo addLineStyle_SubmitIfNeeded(
		const CPULineStyle& lineStyle,
		uint32_t& outLineStyleIdx,
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit);

	nbl::video::IGPUQueue::SSubmitInfo addMainObject_SubmitIfNeeded(
		const MainObject& mainObject,
		uint32_t& outMainObjectIdx,
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit);

	nbl::video::IGPUQueue::SSubmitInfo addClipProjectionData_SubmitIfNeeded(
		const ClipProjectionData& clipProjectionData,
		uint32_t& outClipProjectionIdx,
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit);

protected:

	SubmitFunc submitDraws;
	static constexpr uint32_t InvalidLineStyleIdx = ~0u;

	nbl::video::IGPUQueue::SSubmitInfo finalizeIndexCopiesToGPU(
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit);

	nbl::video::IGPUQueue::SSubmitInfo finalizeMainObjectCopiesToGPU(
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit);

	nbl::video::IGPUQueue::SSubmitInfo finalizeGeometryCopiesToGPU(
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit);

	nbl::video::IGPUQueue::SSubmitInfo finalizeLineStyleCopiesToGPU(
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit);

	nbl::video::IGPUQueue::SSubmitInfo finalizeCustomClipProjectionCopiesToGPU(
		nbl::video::IGPUQueue* submissionQueue,
		nbl::video::IGPUFence* submissionFence,
		nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit);

	uint32_t addMainObject_Internal(const MainObject& mainObject);

	uint32_t addLineStyle_Internal(const CPULineStyle& cpuLineStyle);

	uint32_t addClipProjectionData_Internal(const ClipProjectionData& clipProjectionData);

	static constexpr uint32_t getCageCountPerPolylineObject(ObjectType type)
	{
		if (type == ObjectType::LINE)
			return 1u;
		else if (type == ObjectType::QUAD_BEZIER)
			return 3u;
		return 0u;
	};

	void addPolylineObjects_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx);

	// TODO[Prezmek]: another function named addPolylineConnectors_Internal and you pass a nbl::core::Range<PolylineConnectorInfo>, uint32_t currentPolylineConnectorObj, uint32_t mainObjIdx
	// And implement it similar to addLines/QuadBeziers_Internal which is check how much memory is left and how many PolylineConnectors you can fit into the current geometry and drawobj memory left and return to the drawPolylinefunction
	void addPolylineConnectors_Internal(const CPolylineBase& polyline, uint32_t& currentPolylineConnectorObj, uint32_t mainObjIdx);

	// TODO[Przemek]: this function will change a little as you'll be copying LinePointInfos instead of double2's
	// Make sure to test with small memory to trigger submitInBetween function when you run out of memory to see if your changes here didn't mess things up, ask Lucas for help if you're not sure on how to do this
	void addLines_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx);

	void addQuadBeziers_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx);

	void addHatch_Internal(const Hatch& hatch, uint32_t& currentObjectInSection, uint32_t mainObjIndex);

	//@param oddProvokingVertex is used for our polyline-wide transparency algorithm where we draw the object twice, once to resolve the alpha and another time to draw them
	void addCagedObjectIndices_Internal(uint32_t startObject, uint32_t objectCount);

	void resetAllCounters()
	{
		resetMainObjectCounters();
		resetGeometryCounters();
		resetIndexCounters();
		resetStyleCounters();
		resetCustomClipProjectionCounters();
	}

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
	}

	void resetIndexCounters()
	{
		inMemIndexCount = 0u;
		currentIndexCount = 0u;
	}

	void resetStyleCounters()
	{
		currentLineStylesCount = 0u;
		inMemLineStylesCount = 0u;
	}

	void resetCustomClipProjectionCounters()
	{
		currentClipProjectionDataCount = 0u;
		inMemClipProjectionDataCount = 0u;
	}

	nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;

	uint32_t inMemIndexCount = 0u;
	uint32_t currentIndexCount = 0u;
	uint32_t maxIndices = 0u;

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

	uint32_t inMemClipProjectionDataCount = 0u;
	uint32_t currentClipProjectionDataCount = 0u;
	uint32_t maxClipProjectionData = 0u;

	uint64_t geometryBufferAddress = 0u; // Actual BDA offset 0 of the gpu buffer
};
