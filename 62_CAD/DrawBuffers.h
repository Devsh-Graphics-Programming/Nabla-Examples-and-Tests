#include "Polyline.h"
#include "Hatch.h"
#include <nbl/video/utilities/SIntendedSubmitInfo.h>

using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;

template <typename BufferType>
struct DrawBuffers
{
	smart_refctd_ptr<BufferType> indexBuffer; // only is valid for IGPUBuffer because it's filled at allocation time and never touched again
	smart_refctd_ptr<BufferType> mainObjectsBuffer;
	smart_refctd_ptr<BufferType> drawObjectsBuffer;
	smart_refctd_ptr<BufferType> geometryBuffer;
	smart_refctd_ptr<BufferType> lineStylesBuffer;
	smart_refctd_ptr<BufferType> customClipProjectionBuffer;
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

	void allocateStylesBuffer(ILogicalDevice* logicalDevice, uint32_t stylesCount);

	void allocateCustomClipProjectionBuffer(ILogicalDevice* logicalDevice, uint32_t ClipProjectionDataCount);

	//! this function fills buffers required for drawing a polyline and submits a draw through provided callback when there is not enough memory.
	void drawPolyline(
		const CPolylineBase& polyline,
		const CPULineStyle& cpuLineStyle,
		const uint32_t clipProjectionIdx,
		SIntendedSubmitInfo& intendedNextSubmit);

	void drawPolyline(
		const CPolylineBase& polyline,
		const uint32_t polylineMainObjIdx,
		SIntendedSubmitInfo& intendedNextSubmit);

	void drawHatch(
		const Hatch& hatch,
		// If more parameters from cpu line style are used here later, make a new HatchStyle & use that
		const float32_t4 color,
		const uint32_t clipProjectionIdx,
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

	inline size_t getCurrentCustomClipProjectionBufferSize() const
	{
		return sizeof(ClipProjectionData) * currentClipProjectionDataCount;
	}

	void reset()
	{
		resetAllCounters();
	}

	DrawBuffers<ICPUBuffer> cpuDrawBuffers;
	DrawBuffers<IGPUBuffer> gpuDrawBuffers;

	void addLineStyle_SubmitIfNeeded(
		const CPULineStyle& lineStyle,
		uint32_t& outLineStyleIdx,
		SIntendedSubmitInfo& intendedNextSubmit);

	void addMainObject_SubmitIfNeeded(
		const MainObject& mainObject,
		uint32_t& outMainObjectIdx,
		SIntendedSubmitInfo& intendedNextSubmit);

	void addClipProjectionData_SubmitIfNeeded(
		const ClipProjectionData& clipProjectionData,
		uint32_t& outClipProjectionIdx,
		SIntendedSubmitInfo& intendedNextSubmit);

protected:

	SubmitFunc submitDraws;
	static constexpr uint32_t InvalidLineStyleIdx = ~0u;

	void finalizeMainObjectCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);

	void finalizeGeometryCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);

	void finalizeLineStyleCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);

	void finalizeCustomClipProjectionCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit);

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

	// TODO[Prezmek]: another function named addPolylineConnectors_Internal and you pass a Range<PolylineConnectorInfo>, uint32_t currentPolylineConnectorObj, uint32_t mainObjIdx
	// And implement it similar to addLines/QuadBeziers_Internal which is check how much memory is left and how many PolylineConnectors you can fit into the current geometry and drawobj memory left and return to the drawPolylinefunction
	void addPolylineConnectors_Internal(const CPolylineBase& polyline, uint32_t& currentPolylineConnectorObj, uint32_t mainObjIdx);

	// TODO[Przemek]: this function will change a little as you'll be copying LinePointInfos instead of double2's
	// Make sure to test with small memory to trigger submitInBetween function when you run out of memory to see if your changes here didn't mess things up, ask Lucas for help if you're not sure on how to do this
	void addLines_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx);

	void addQuadBeziers_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx);

	void addHatch_Internal(const Hatch& hatch, uint32_t& currentObjectInSection, uint32_t mainObjIndex);

	void resetAllCounters()
	{
		resetMainObjectCounters();
		resetGeometryCounters();
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

	uint32_t inMemClipProjectionDataCount = 0u;
	uint32_t currentClipProjectionDataCount = 0u;
	uint32_t maxClipProjectionData = 0u;

	uint64_t geometryBufferAddress = 0u; // Actual BDA offset 0 of the gpu buffer
};
