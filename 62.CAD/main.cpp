#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/core/SRange.h"
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include "curves.h"

static constexpr bool DebugMode = false;
static constexpr bool DebugRotatingViewProj = false;
static constexpr bool FragmentShaderPixelInterlock = true;

enum class ExampleMode
{
	CASE_0, // Simple Line, Camera Zoom In/Out
	CASE_1,	// Overdraw Fragment Shader Stress Test
	CASE_2, // hatches
	CASE_3, // CURVES AND LINES
	CASE_4, // STIPPLE PATTERN
};

constexpr ExampleMode mode = ExampleMode::CASE_2;

typedef uint32_t uint;

#include "common.hlsl"

using namespace nbl::hlsl;

bool operator==(const LineStyle& lhs, const LineStyle& rhs)
{
	const bool areParametersEqual =
		lhs.color == rhs.color &&
		lhs.screenSpaceLineWidth == rhs.screenSpaceLineWidth &&
		lhs.worldSpaceLineWidth == rhs.worldSpaceLineWidth &&
		lhs.stipplePatternSize == rhs.stipplePatternSize &&
		lhs.reciprocalStipplePatternLen == rhs.reciprocalStipplePatternLen &&
		lhs.phaseShift == rhs.phaseShift;

	if (!areParametersEqual)
		return false;
	return true;

	if (lhs.stipplePatternSize == -1)
		return true;

	assert(lhs.stipplePatternSize <= LineStyle::STIPPLE_PATTERN_MAX_SZ && lhs.stipplePatternSize >= 0);

	const bool isStipplePatternArrayEqual = std::memcmp(lhs.stipplePattern, rhs.stipplePattern, sizeof(decltype(lhs.stipplePatternSize)) * lhs.stipplePatternSize);

	return areParametersEqual && isStipplePatternArrayEqual;
}

// holds values for `LineStyle` struct and caculates stipple pattern re values, cant think of better name
struct CPULineStyle
{
	static constexpr int32_t InvalidStipplePatternSize = -1;
	static const uint32_t STIPPLE_PATTERN_MAX_SZ = 15u;

	float32_t4 color;
	float screenSpaceLineWidth;
	float worldSpaceLineWidth;
	// gpu stipple pattern data form
	int32_t stipplePatternSize;
	float reciprocalStipplePatternLen;
	float stipplePattern[STIPPLE_PATTERN_MAX_SZ];
	float phaseShift;

	void setStipplePatternData(const nbl::core::SRange<float>& stipplePatternCPURepresentation)
	//void prepareGPUStipplePatternData(const nbl::core::vector<float>& stipplePatternCPURepresentation)
	{
		assert(stipplePatternCPURepresentation.size() <= STIPPLE_PATTERN_MAX_SZ);

		if (stipplePatternCPURepresentation.size() == 0)
		{
			stipplePatternSize = 0;
			return;
		}

		nbl::core::vector<float> stipplePatternTransformed;

		// just to make sure we have a consistent definition of what's positive and what's negative
		auto isValuePositive = [](float x)
			{
				return (x >= 0);
			};

		// merge redundant values
		for (auto it = stipplePatternCPURepresentation.begin(); it != stipplePatternCPURepresentation.end();)
		{
			float redundantConsecutiveValuesSum = 0.0f;
			const bool firstValueSign = isValuePositive(*it);
			do
			{
				redundantConsecutiveValuesSum += *it;
				it++;
			} while (it != stipplePatternCPURepresentation.end() && (firstValueSign == isValuePositive(*it)));

			stipplePatternTransformed.push_back(redundantConsecutiveValuesSum);
		}

		if (stipplePatternTransformed.size() == 1)
		{
			stipplePatternSize = stipplePatternTransformed[0] < 0.0f ? InvalidStipplePatternSize : 0;
			return;
		}

		// merge first and last value if their sign matches
		phaseShift = 0.0f;
		const bool firstComponentPositive = isValuePositive(stipplePatternTransformed[0]);
		const bool lastComponentPositive = isValuePositive(stipplePatternTransformed[stipplePatternTransformed.size() - 1]);
		if (firstComponentPositive == lastComponentPositive)
		{
			phaseShift = std::abs(stipplePatternTransformed[stipplePatternTransformed.size() - 1]);
			stipplePatternTransformed[0] += stipplePatternTransformed[stipplePatternTransformed.size() - 1];
			stipplePatternTransformed.pop_back();
		}

		if (stipplePatternTransformed.size() == 1)
		{
			stipplePatternSize = (firstComponentPositive) ? 0 : InvalidStipplePatternSize;
			return;
		}

		// rotate values if first value is negative value
		if (!firstComponentPositive)
		{
			std::rotate(stipplePatternTransformed.rbegin(), stipplePatternTransformed.rbegin() + 1, stipplePatternTransformed.rend());
			phaseShift += std::abs(stipplePatternTransformed[0]);
		}

		// calculate normalized prefix sum
		const uint32_t PREFIX_SUM_MAX_SZ = LineStyle::STIPPLE_PATTERN_MAX_SZ - 1u;
		const uint32_t PREFIX_SUM_SZ = stipplePatternTransformed.size() - 1;

		float prefixSum[PREFIX_SUM_MAX_SZ];
		prefixSum[0] = stipplePatternTransformed[0];

		for (uint32_t i = 1u; i < PREFIX_SUM_SZ; i++)
			prefixSum[i] = abs(stipplePatternTransformed[i]) + prefixSum[i - 1];

		reciprocalStipplePatternLen = 1.0f / (prefixSum[PREFIX_SUM_SZ - 1] + abs(stipplePatternTransformed[PREFIX_SUM_SZ]));

		for (int i = 0; i < PREFIX_SUM_SZ; i++)
			prefixSum[i] *= reciprocalStipplePatternLen;

		stipplePatternSize = PREFIX_SUM_SZ;
		std::memcpy(stipplePattern, prefixSum, sizeof(prefixSum));

		phaseShift = phaseShift * reciprocalStipplePatternLen;
	}

	LineStyle getAsGPUData() const
	{
		LineStyle ret;
		std::memcpy(ret.stipplePattern, stipplePattern, STIPPLE_PATTERN_MAX_SZ*sizeof(float));
		ret.color = color;
		ret.screenSpaceLineWidth = screenSpaceLineWidth;
		ret.worldSpaceLineWidth = worldSpaceLineWidth;
		ret.stipplePatternSize = stipplePatternSize;
		ret.reciprocalStipplePatternLen = reciprocalStipplePatternLen;
		ret.phaseShift = phaseShift;

		return ret;
	}

	inline bool isVisible() const { return stipplePatternSize != InvalidStipplePatternSize; }
};

static_assert(sizeof(DrawObject) == 16u);
static_assert(sizeof(MainObject) == 8u);
static_assert(sizeof(Globals) == 112u);
static_assert(sizeof(LineStyle) == 96u);
static_assert(sizeof(ClipProjectionData) == 88u);

using namespace nbl;
using namespace ui;

class Camera2D : public core::IReferenceCounted
{
public:
	Camera2D()
	{}

	void setOrigin(const float64_t2& origin)
	{
		m_origin = origin;
	}

	void setAspectRatio(const double& aspectRatio)
	{
		m_aspectRatio = aspectRatio;
	}

	void setSize(const double size)
	{
		m_bounds = float64_t2{ size * m_aspectRatio, size };
	}

	float64_t2 getBounds() const
	{
		return m_bounds;
	}

	float64_t3x3 constructViewProjection()
	{
		auto ret = float64_t3x3();

		ret[0][0] = 2.0 / m_bounds.x;
		ret[1][1] = -2.0 / m_bounds.y;
		ret[2][2] = 1.0;
		
		ret[0][2] = (-2.0 * m_origin.x) / m_bounds.x;
		ret[1][2] = (2.0 * m_origin.y) / m_bounds.y;

		return ret;
	}

	void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
	{
		for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
		{
			auto ev = *eventIt;

			if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL)
			{
				m_bounds = m_bounds + float64_t2{ (double)ev.scrollEvent.verticalScroll * -0.1 * m_aspectRatio, (double)ev.scrollEvent.verticalScroll * -0.1};
				m_bounds = float64_t2{ core::max(m_aspectRatio, m_bounds.x), core::max(1.0, m_bounds.y) };
			}
		}
	}

	void keyboardProcess(const IKeyboardEventChannel::range_t& events)
	{
		for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
		{
			auto ev = *eventIt;

			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_W)
			{
				m_origin.y += 2;
			}
			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_A)
			{
				m_origin.x -= 2;
			}
			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_S)
			{
				m_origin.y -= 2;
			}
			if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_D)
			{
				m_origin.x += 2;
			}
		}
	}
private:

	double m_aspectRatio = 0.0;
	float64_t2 m_bounds = {};
	float64_t2 m_origin = {};
};

// It is not optimized because how you feed a Polyline to our cad renderer is your choice. this is just for convenience
// This is a Nabla Polyline used to feed to our CAD renderer. You can convert your Polyline to this class. or just use it directly.
class CPolyline
{
public:

	// each section consists of multiple connected lines or multiple connected ellipses
	struct SectionInfo
	{
		ObjectType	type;
		uint32_t	index; // can't make this a void* cause of vector resize
		uint32_t	count;
	};

	size_t getSectionsCount() const { return m_sections.size(); }

	const SectionInfo& getSectionInfoAt(const uint32_t idx) const
	{
		return m_sections[idx];
	}

	const QuadraticBezierInfo& getQuadBezierInfoAt(const uint32_t idx) const
	{
		return m_quadBeziers[idx];
	}

	const float64_t2& getLinePointAt(const uint32_t idx) const
	{
		return m_linePoints[idx];
	}

	void clearEverything()
	{
		m_sections.clear();
		m_linePoints.clear();
		m_quadBeziers.clear();
	}

	// Reserves memory with worst case
	void reserveMemory(uint32_t noOfLines, uint32_t noOfBeziers)
	{
		m_sections.reserve(noOfLines + noOfBeziers);
		m_linePoints.reserve(noOfLines * 2u);
		m_quadBeziers.reserve(noOfBeziers);
	}

	void addLinePoints(const core::SRange<float64_t2>& linePoints, bool addToPreviousLineSectionIfAvailable = false)
	{
		if (linePoints.size() <= 1u)
			return;

		const bool previousSectionIsLine = m_sections.size() > 0u && m_sections[m_sections.size() - 1u].type == ObjectType::LINE;
		const bool alwaysAddNewSection = !addToPreviousLineSectionIfAvailable;
		bool addNewSection = alwaysAddNewSection || !previousSectionIsLine;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::LINE;
			newSection.index = static_cast<uint32_t>(m_linePoints.size());
			newSection.count = static_cast<uint32_t>(linePoints.size() - 1u);
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += static_cast<uint32_t>(linePoints.size());
		}
		m_linePoints.insert(m_linePoints.end(), linePoints.begin(), linePoints.end());
	}

	void addEllipticalArcs(const core::SRange<curves::EllipticalArcInfo>& ellipses)
	{
		// TODO[Erfan] Approximate with quadratic beziers
	}

	// TODO[Przemek]: this input should be nbl::hlsl::QuadraticBezier instead cause `QuadraticBezierInfo` includes precomputed data I don't want user to see
	void addQuadBeziers(const core::SRange<QuadraticBezierInfo>& quadBeziers)
	{
		bool addNewSection = m_sections.size() == 0u || m_sections[m_sections.size() - 1u].type != ObjectType::QUAD_BEZIER;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::QUAD_BEZIER;
			newSection.index = static_cast<uint32_t>(m_quadBeziers.size());
			newSection.count = static_cast<uint32_t>(quadBeziers.size());
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += static_cast<uint32_t>(quadBeziers.size());
		}
		m_quadBeziers.insert(m_quadBeziers.end(), quadBeziers.begin(), quadBeziers.end());
	}

	// TODO[Przemek]: Add a function here named preprocessPolylineWithStyle -> give it the line style
	/*
	*  this preprocess should:
	*	1. if style has road info try to generate miters:
	*		if tangents are not in the same direction with some error add a PolylineConnector object
		2. go over the list of sections (line and beziers in order) compute the phase shift by computing their arclen and divisions with style length and
			fill the phaseShift part of the QuadraticBezierInfo and LinePointInfo, 
			you initially set them to 0 in addLinePoints/addQuadBeziers

		NOTE that PolylineConnectors are special object types, user does not add them and they should not be part of m_sections vector
	*/ 

protected:
	// TODO[Przemek]: a vector of polyline connetor objects
	std::vector<SectionInfo> m_sections;
	// TODO[Przemek]: instead of float64_t2 for linePoints, store LinePointInfo
	std::vector<float64_t2> m_linePoints;
	std::vector<QuadraticBezierInfo> m_quadBeziers;
};

#include "Hatch.cpp"

template <typename BufferType>
struct DrawBuffers
{
	core::smart_refctd_ptr<BufferType> indexBuffer;
	core::smart_refctd_ptr<BufferType> mainObjectsBuffer;
	core::smart_refctd_ptr<BufferType> drawObjectsBuffer;
	core::smart_refctd_ptr<BufferType> geometryBuffer;
	core::smart_refctd_ptr<BufferType> lineStylesBuffer;
	core::smart_refctd_ptr<BufferType> customClipProjectionBuffer;
};

// ! this is just a buffers filler with autosubmission features used for convenience to how you feed our CAD renderer
struct DrawBuffersFiller
{
public:

	typedef uint32_t index_buffer_type;

	DrawBuffersFiller() {}

	DrawBuffersFiller(core::smart_refctd_ptr<nbl::video::IUtilities>&& utils)
	{
		utilities = utils;
	}

	typedef std::function<video::IGPUQueue::SSubmitInfo(video::IGPUQueue*, video::IGPUFence*, video::IGPUQueue::SSubmitInfo)> SubmitFunc;

	// function is called when buffer is filled and we should submit draws and clear the buffers and continue filling
	void setSubmitDrawsFunction(SubmitFunc func)
	{
		submitDraws = func;
	}

	void allocateIndexBuffer(video::ILogicalDevice* logicalDevice, uint32_t indices)
	{
		maxIndices = indices;
		const size_t indexBufferSize = maxIndices * sizeof(uint32_t);

		video::IGPUBuffer::SCreationParams indexBufferCreationParams = {};
		indexBufferCreationParams.size = indexBufferSize;
		indexBufferCreationParams.usage = video::IGPUBuffer::EUF_INDEX_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.indexBuffer = logicalDevice->createBuffer(std::move(indexBufferCreationParams));
		gpuDrawBuffers.indexBuffer->setObjectDebugName("indexBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.indexBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto indexBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.indexBuffer.get());

		cpuDrawBuffers.indexBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(indexBufferSize);
	}

	void allocateMainObjectsBuffer(video::ILogicalDevice* logicalDevice, uint32_t mainObjects)
	{
		maxMainObjects = mainObjects;
		size_t mainObjectsBufferSize = mainObjects * sizeof(MainObject);

		video::IGPUBuffer::SCreationParams mainObjectsCreationParams = {};
		mainObjectsCreationParams.size = mainObjectsBufferSize;
		mainObjectsCreationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.mainObjectsBuffer = logicalDevice->createBuffer(std::move(mainObjectsCreationParams));
		gpuDrawBuffers.mainObjectsBuffer->setObjectDebugName("mainObjectsBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.mainObjectsBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto mainObjectsBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.mainObjectsBuffer.get());

		cpuDrawBuffers.mainObjectsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(mainObjectsBufferSize);
	}

	void allocateDrawObjectsBuffer(video::ILogicalDevice* logicalDevice, uint32_t drawObjects)
	{
		maxDrawObjects = drawObjects;
		size_t drawObjectsBufferSize = drawObjects * sizeof(DrawObject);
		 
		video::IGPUBuffer::SCreationParams drawObjectsCreationParams = {};
		drawObjectsCreationParams.size = drawObjectsBufferSize;
		drawObjectsCreationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.drawObjectsBuffer = logicalDevice->createBuffer(std::move(drawObjectsCreationParams));
		gpuDrawBuffers.drawObjectsBuffer->setObjectDebugName("drawObjectsBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.drawObjectsBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto drawObjectsBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.drawObjectsBuffer.get());

		cpuDrawBuffers.drawObjectsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(drawObjectsBufferSize);
	}

	void allocateGeometryBuffer(video::ILogicalDevice* logicalDevice, size_t size)
	{
		maxGeometryBufferSize = size;

		video::IGPUBuffer::SCreationParams geometryCreationParams = {};
		geometryCreationParams.size = size;
		geometryCreationParams.usage = core::bitflag(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.geometryBuffer = logicalDevice->createBuffer(std::move(geometryCreationParams));
		gpuDrawBuffers.geometryBuffer->setObjectDebugName("geometryBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.geometryBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto geometryBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.geometryBuffer.get(), video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
		geometryBufferAddress = logicalDevice->getBufferDeviceAddress(gpuDrawBuffers.geometryBuffer.get());

		cpuDrawBuffers.geometryBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(size);
	}

	void allocateStylesBuffer(video::ILogicalDevice* logicalDevice, uint32_t stylesCount)
	{
		maxLineStyles = stylesCount;
		size_t lineStylesBufferSize = stylesCount * sizeof(LineStyle);

		video::IGPUBuffer::SCreationParams lineStylesCreationParams = {};
		lineStylesCreationParams.size = lineStylesBufferSize;
		lineStylesCreationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.lineStylesBuffer = logicalDevice->createBuffer(std::move(lineStylesCreationParams));
		gpuDrawBuffers.lineStylesBuffer->setObjectDebugName("lineStylesBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.lineStylesBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto stylesBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.lineStylesBuffer.get());

		cpuDrawBuffers.lineStylesBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(lineStylesBufferSize);
	}

	void allocateCustomClipProjectionBuffer(video::ILogicalDevice* logicalDevice, uint32_t ClipProjectionDataCount)
	{
		maxClipProjectionData = ClipProjectionDataCount;
		size_t customClipProjectionBufferSize = maxClipProjectionData * sizeof(ClipProjectionData);

		video::IGPUBuffer::SCreationParams customClipProjectionCreationParams = {};
		customClipProjectionCreationParams.size = customClipProjectionBufferSize;
		customClipProjectionCreationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.customClipProjectionBuffer = logicalDevice->createBuffer(std::move(customClipProjectionCreationParams));
		gpuDrawBuffers.customClipProjectionBuffer->setObjectDebugName("customClipProjectionBuffer");

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.customClipProjectionBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto customClipProjectionBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.customClipProjectionBuffer.get());

		cpuDrawBuffers.customClipProjectionBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(customClipProjectionBufferSize);
	}
	
	// TODO
	//uint32_t getIndexCount() const { return currentIndexCount; }
	
	//! this function fills buffers required for drawing a polyline and submits a draw through provided callback when there is not enough memory.
	video::IGPUQueue::SSubmitInfo drawPolyline(
		const CPolyline& polyline,
		const CPULineStyle& cpuLineStyle,
		const uint32_t clipProjectionIdx,
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		if (!cpuLineStyle.isVisible())
			return intendedNextSubmit;

		uint32_t styleIdx;
		intendedNextSubmit = addLineStyle_SubmitIfNeeded(cpuLineStyle, styleIdx, submissionQueue, submissionFence, intendedNextSubmit);
		
		MainObject mainObj = {};
		mainObj.styleIdx = styleIdx;
		mainObj.clipProjectionIdx = clipProjectionIdx;
		uint32_t mainObjIdx;
		intendedNextSubmit = addMainObject_SubmitIfNeeded(mainObj, mainObjIdx, submissionQueue, submissionFence, intendedNextSubmit);

		return drawPolyline(submissionQueue, submissionFence, intendedNextSubmit, polyline, mainObjIdx);
	}

	video::IGPUQueue::SSubmitInfo drawPolyline(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit,
		const CPolyline& polyline,
		const uint32_t polylineMainObjIdx)
	{
		if (polylineMainObjIdx == InvalidMainObjectIdx)
		{
			// TODO: assert or log error here
			return intendedNextSubmit;
		}

		const auto sectionsCount = polyline.getSectionsCount();

		uint32_t currentSectionIdx = 0u;
		uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject used in vertex shader. You can think of it as a Cage.

		while (currentSectionIdx < sectionsCount)
		{
			bool shouldSubmit = false;
			const auto& currentSection = polyline.getSectionInfoAt(currentSectionIdx);
			addPolylineObjects_Internal(polyline, currentSection, currentObjectInSection, polylineMainObjIdx);

			if (currentObjectInSection >= currentSection.count)
			{
				currentSectionIdx++;
				currentObjectInSection = 0u;
			}
			else
				shouldSubmit = true;

			if (shouldSubmit)
			{
				intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
				intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
				resetIndexCounters();
				resetGeometryCounters();
				// We don't reset counters for linestyles, mainObjects and customClipProjection because we will be reusing them
				shouldSubmit = false;
			}
		}

		// TODO[Prezmek]: A similar and simpler while loop as above where you try to addPolylineConnectors_Internal, If you couldn't do the whole section completely then -> finalizeAllCopies, submit and reset stuff as above.

		return intendedNextSubmit;
	}

	// If we had infinite mem, we would first upload all curves into geometry buffer then upload the "CurveBoxes" with correct gpu addresses to those
	// But we don't have that so we have to follow a similar auto submission as the "drawPolyline" function with some mutations:
	// We have to find the MAX number of "CurveBoxes" we could draw, and since both the "Curves" and "CurveBoxes" reside in geometry buffer,
	// it has to be taken into account when calculating "how many curve boxes we could draw and when we need to submit/clear"
	// So same as drawPolylines, we would first try to fill the geometry buffer and index buffer that corresponds to "backfaces or even provoking vertices"
	// then change index buffer to draw front faces of the curveBoxes that already reside in geometry buffer memory
	// then if anything was left (the ones that weren't in memory for front face of the curveBoxes) we copy their geom to mem again and use frontface/oddProvoking vertex
	video::IGPUQueue::SSubmitInfo drawHatch(
		const Hatch& hatch,
		// If more parameters from cpu line style are used here later, make a new HatchStyle & use that
		const float32_t4 color, 
		const uint32_t clipProjectionIdx,
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		CPULineStyle lineStyle;
		lineStyle.color = color;

		uint32_t styleIdx;
		intendedNextSubmit = addLineStyle_SubmitIfNeeded(lineStyle, styleIdx, submissionQueue, submissionFence, intendedNextSubmit);
		
		MainObject mainObj = {};
		mainObj.styleIdx = styleIdx;
		mainObj.clipProjectionIdx = clipProjectionIdx;
		uint32_t mainObjIdx;
		intendedNextSubmit = addMainObject_SubmitIfNeeded(mainObj, mainObjIdx, submissionQueue, submissionFence, intendedNextSubmit);

		const auto sectionsCount = 1;

		uint32_t currentSectionIdx = 0u;
		uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject used in vertex shader. You can think of it as a Cage.

		while (currentSectionIdx < sectionsCount)
		{
			bool shouldSubmit = false;
			addHatch_Internal(hatch, currentObjectInSection, mainObjIdx);

			const auto sectionObjectCount = hatch.getHatchBoxCount();
			if (currentObjectInSection >= sectionObjectCount)
			{
				currentSectionIdx++;
				currentObjectInSection = 0u;
			}
			else
				shouldSubmit = true;

			if (shouldSubmit)
			{
				intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
				intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
				resetIndexCounters();
				resetGeometryCounters();
				// We don't reset counters for linestyles and mainObjects because we will be reusing them
				shouldSubmit = false;
			}
		}

		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo finalizeAllCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		intendedNextSubmit = finalizeIndexCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		intendedNextSubmit = finalizeMainObjectCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		intendedNextSubmit = finalizeGeometryCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		intendedNextSubmit = finalizeLineStyleCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		intendedNextSubmit = finalizeCustomClipProjectionCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);

		return intendedNextSubmit;
	}

	inline uint32_t getIndexCount() const { return currentIndexCount; }

	inline uint32_t getLineStyleCount() const { return currentLineStylesCount; }

	inline uint32_t getDrawObjectCount() const { return currentDrawObjectCount; }

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

	DrawBuffers<asset::ICPUBuffer> cpuDrawBuffers;
	DrawBuffers<video::IGPUBuffer> gpuDrawBuffers;

	video::IGPUQueue::SSubmitInfo addLineStyle_SubmitIfNeeded(
		const CPULineStyle& lineStyle,
		uint32_t& outLineStyleIdx,
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		outLineStyleIdx = addLineStyle_Internal(lineStyle);
		if (outLineStyleIdx == InvalidLineStyleIdx)
		{
			intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
			intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
			resetAllCounters();
			outLineStyleIdx = addLineStyle_Internal(lineStyle);
			assert(outLineStyleIdx != InvalidLineStyleIdx);
		}
		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo addMainObject_SubmitIfNeeded(
		const MainObject& mainObject,
		uint32_t& outMainObjectIdx,
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		outMainObjectIdx = addMainObject_Internal(mainObject);
		if (outMainObjectIdx == InvalidMainObjectIdx)
		{
			intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
			intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
			resetAllCounters();
			outMainObjectIdx = addMainObject_Internal(mainObject);
			assert(outMainObjectIdx != InvalidMainObjectIdx);
		}
		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo addClipProjectionData_SubmitIfNeeded(
		const ClipProjectionData& clipProjectionData,
		uint32_t& outClipProjectionIdx,
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		outClipProjectionIdx = addClipProjectionData_Internal(clipProjectionData);
		if (outClipProjectionIdx == InvalidClipProjectionIdx)
		{
			intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
			intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
			resetAllCounters();
			outClipProjectionIdx = addClipProjectionData_Internal(clipProjectionData);
			assert(outClipProjectionIdx != InvalidClipProjectionIdx);
		}
		return intendedNextSubmit;
	}

protected:

	SubmitFunc submitDraws;
	static constexpr uint32_t InvalidLineStyleIdx = ~0u;

	video::IGPUQueue::SSubmitInfo finalizeIndexCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Copy Indices
		uint32_t remainingIndexCount = currentIndexCount - inMemIndexCount;
		asset::SBufferRange<video::IGPUBuffer> indicesRange = { sizeof(index_buffer_type) * inMemIndexCount, sizeof(index_buffer_type) * remainingIndexCount, gpuDrawBuffers.indexBuffer };
		const index_buffer_type* srcIndexData = reinterpret_cast<index_buffer_type*>(cpuDrawBuffers.indexBuffer->getPointer()) + inMemIndexCount;
		if (indicesRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(indicesRange, srcIndexData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemIndexCount = currentIndexCount;
		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo finalizeMainObjectCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Copy MainObjects
		uint32_t remainingMainObjects = currentMainObjectCount - inMemMainObjectCount;
		asset::SBufferRange<video::IGPUBuffer> mainObjectsRange = { sizeof(MainObject) * inMemMainObjectCount, sizeof(MainObject) * remainingMainObjects, gpuDrawBuffers.mainObjectsBuffer };
		const MainObject* srcMainObjData = reinterpret_cast<MainObject*>(cpuDrawBuffers.mainObjectsBuffer->getPointer()) + inMemMainObjectCount;
		if (mainObjectsRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(mainObjectsRange, srcMainObjData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemMainObjectCount = currentMainObjectCount;
		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo finalizeGeometryCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Copy DrawObjects
		uint32_t remainingDrawObjects = currentDrawObjectCount - inMemDrawObjectCount;
		asset::SBufferRange<video::IGPUBuffer> drawObjectsRange = { sizeof(DrawObject) * inMemDrawObjectCount, sizeof(DrawObject) * remainingDrawObjects, gpuDrawBuffers.drawObjectsBuffer };
		const DrawObject* srcDrawObjData = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + inMemDrawObjectCount;
		if (drawObjectsRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(drawObjectsRange, srcDrawObjData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemDrawObjectCount = currentDrawObjectCount;

		// Copy GeometryBuffer
		uint64_t remainingGeometrySize = currentGeometryBufferSize - inMemGeometryBufferSize;
		asset::SBufferRange<video::IGPUBuffer> geomRange = { inMemGeometryBufferSize, remainingGeometrySize, gpuDrawBuffers.geometryBuffer };
		const uint8_t* srcGeomData = reinterpret_cast<uint8_t*>(cpuDrawBuffers.geometryBuffer->getPointer()) + inMemGeometryBufferSize;
		if (geomRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(geomRange, srcGeomData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemGeometryBufferSize = currentGeometryBufferSize;

		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo finalizeLineStyleCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Copy LineStyles
		uint32_t remainingLineStyles = currentLineStylesCount - inMemLineStylesCount;
		asset::SBufferRange<video::IGPUBuffer> stylesRange = { sizeof(LineStyle) * inMemLineStylesCount, sizeof(LineStyle) * remainingLineStyles, gpuDrawBuffers.lineStylesBuffer };
		const LineStyle* srcLineStylesData = reinterpret_cast<LineStyle*>(cpuDrawBuffers.lineStylesBuffer->getPointer()) + inMemLineStylesCount;
		if (stylesRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(stylesRange, srcLineStylesData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemLineStylesCount = currentLineStylesCount;
		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo finalizeCustomClipProjectionCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Copy LineStyles
		uint32_t remainingClipProjectionData = currentClipProjectionDataCount - inMemClipProjectionDataCount;
		asset::SBufferRange<video::IGPUBuffer> clipProjectionRange = { sizeof(ClipProjectionData) * inMemClipProjectionDataCount, sizeof(ClipProjectionData) * remainingClipProjectionData, gpuDrawBuffers.customClipProjectionBuffer };
		const ClipProjectionData* srcClipProjectionData = reinterpret_cast<ClipProjectionData*>(cpuDrawBuffers.customClipProjectionBuffer->getPointer()) + inMemClipProjectionDataCount;
		if (clipProjectionRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(clipProjectionRange, srcClipProjectionData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemClipProjectionDataCount = currentClipProjectionDataCount;
		return intendedNextSubmit;
	}

	uint32_t addMainObject_Internal(const MainObject& mainObject)
	{
		MainObject* mainObjsArray = reinterpret_cast<MainObject*>(cpuDrawBuffers.mainObjectsBuffer->getPointer());
		if (currentMainObjectCount >= maxMainObjects)
			return InvalidMainObjectIdx;

		void* dst = mainObjsArray + currentMainObjectCount;
		memcpy(dst, &mainObject, sizeof(MainObject));
		uint32_t ret = (currentMainObjectCount % MaxIndexableMainObjects); // just to wrap around if it ever exceeded (we pack this id into 24 bits)
		currentMainObjectCount++;
		return ret;
	}

	uint32_t addLineStyle_Internal(const CPULineStyle& cpuLineStyle)
	{
		LineStyle gpuLineStyle = cpuLineStyle.getAsGPUData();
		LineStyle* stylesArray = reinterpret_cast<LineStyle*>(cpuDrawBuffers.lineStylesBuffer->getPointer());
		for (uint32_t i = 0u; i < currentLineStylesCount; ++i)
		{
			const LineStyle& itr = stylesArray[i];

			if(itr == gpuLineStyle)
				return i;
		}

		if (currentLineStylesCount >= maxLineStyles)
			return InvalidLineStyleIdx;

		void* dst = stylesArray + currentLineStylesCount;
		memcpy(dst, &gpuLineStyle, sizeof(LineStyle));
		return currentLineStylesCount++;
	}

	uint32_t addClipProjectionData_Internal(const ClipProjectionData& clipProjectionData)
	{
		ClipProjectionData* clipProjectionArray = reinterpret_cast<ClipProjectionData*>(cpuDrawBuffers.customClipProjectionBuffer->getPointer());
		if (currentClipProjectionDataCount >= maxClipProjectionData)
			return InvalidClipProjectionIdx;

		void* dst = clipProjectionArray + currentClipProjectionDataCount;
		memcpy(dst, &clipProjectionData, sizeof(ClipProjectionData));
		return currentClipProjectionDataCount++;
	}

	static constexpr uint32_t getCageCountPerPolylineObject(ObjectType type)
	{
		if (type == ObjectType::LINE)
			return 1u;
		else if (type == ObjectType::QUAD_BEZIER)
			return 3u;
		return 0u;
	};

	void addPolylineObjects_Internal(const CPolyline& polyline, const CPolyline::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
	{
		if (section.type == ObjectType::LINE)
			addLines_Internal(polyline, section, currentObjectInSection, mainObjIdx);
		else if (section.type == ObjectType::QUAD_BEZIER)
			addQuadBeziers_Internal(polyline, section, currentObjectInSection, mainObjIdx);
		else
			assert(false); // we don't handle other object types
	}

	// TODO[Prezmek]: another function named addPolylineConnectors_Internal and you pass a core::Range<PolylineConnectorInfo>, uint32_t currentPolylineConnectorObj, uint32_t mainObjIdx
	// And implement it similar to addLines/QuadBeziers_Internal which is check how much memory is left and how many PolylineConnectors you can fit into the current geometry and drawobj memory left and return to the drawPolylinefunction

	// TODO[Prezmek]: this function will change a little as you'll be copying LinePointInfos instead of double2's
	// Make sure to test with small memory to trigger submitInBetween function when you run out of memory to see if your changes here didn't mess things up, ask Lucas for help if you're not sure on how to do this
	void addLines_Internal(const CPolyline& polyline, const CPolyline::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
	{
		assert(section.count >= 1u);
		assert(section.type == ObjectType::LINE);

		const auto maxGeometryBufferPoints = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(float64_t2);
		const auto maxGeometryBufferLines = (maxGeometryBufferPoints <= 1u) ? 0u : maxGeometryBufferPoints - 1u;

		uint32_t uploadableObjects = (maxIndices - currentIndexCount) / 6u;
		uploadableObjects = core::min(uploadableObjects, maxGeometryBufferLines);
		uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

		const auto lineCount = section.count;
		const auto remainingObjects = lineCount - currentObjectInSection;
		uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

		// Add Indices
		addCagedObjectIndices_Internal(currentDrawObjectCount, objectsToUpload);

		// Add DrawObjs
		DrawObject drawObj = {};
		drawObj.mainObjIndex = mainObjIdx;
		drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::LINE) | 0 << 16);
		drawObj.geometryAddress = geometryBufferAddress + currentGeometryBufferSize;
		for (uint32_t i = 0u; i < objectsToUpload; ++i)
		{
			void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
			memcpy(dst, &drawObj, sizeof(DrawObject));
			currentDrawObjectCount += 1u;
			drawObj.geometryAddress += sizeof(float64_t2);
		}

		// Add Geometry
		if (objectsToUpload > 0u)
		{
			const auto pointsByteSize = sizeof(float64_t2) * (objectsToUpload + 1u);
			void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
			auto& linePoint = polyline.getLinePointAt(section.index + currentObjectInSection);
			memcpy(dst, &linePoint, pointsByteSize);
			currentGeometryBufferSize += pointsByteSize;
		}

		currentObjectInSection += objectsToUpload;
	}

	void addQuadBeziers_Internal(const CPolyline& polyline, const CPolyline::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
	{
		constexpr uint32_t CagesPerQuadBezier = getCageCountPerPolylineObject(ObjectType::QUAD_BEZIER);
		constexpr uint32_t IndicesPerQuadBezier = 6u * CagesPerQuadBezier;
		assert(section.type == ObjectType::QUAD_BEZIER);

		const auto maxGeometryBufferBeziers = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(QuadraticBezierInfo);

		uint32_t uploadableObjects = (maxIndices - currentIndexCount) / IndicesPerQuadBezier;
		uploadableObjects = core::min(uploadableObjects, maxGeometryBufferBeziers);
		uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

		const auto beziersCount = section.count;
		const auto remainingObjects = beziersCount - currentObjectInSection;
		uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

		// Add Indices
		addCagedObjectIndices_Internal(currentDrawObjectCount, objectsToUpload * CagesPerQuadBezier);

		// Add DrawObjs
		DrawObject drawObj = {};
		drawObj.mainObjIndex = mainObjIdx;
		drawObj.geometryAddress = geometryBufferAddress + currentGeometryBufferSize;
		for (uint32_t i = 0u; i < objectsToUpload; ++i)
		{
			for (uint16_t subObject = 0; subObject < CagesPerQuadBezier; subObject++)
			{
				drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::QUAD_BEZIER) | (subObject << 16));
				void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
				memcpy(dst, &drawObj, sizeof(DrawObject));
				currentDrawObjectCount += 1u;
			}
			drawObj.geometryAddress += sizeof(QuadraticBezierInfo);
		}

		// Add Geometry
		if (objectsToUpload > 0u)
		{
			const auto beziersByteSize = sizeof(QuadraticBezierInfo) * (objectsToUpload);
			void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
			auto& quadBezier = polyline.getQuadBezierInfoAt(section.index + currentObjectInSection);
			memcpy(dst, &quadBezier, beziersByteSize);
			currentGeometryBufferSize += beziersByteSize;
		}

		currentObjectInSection += objectsToUpload;
	}

	void addHatch_Internal(const Hatch& hatch, uint32_t& currentObjectInSection, uint32_t mainObjIndex)
	{
		constexpr uint32_t IndicesPerHatchBox = 6u;

		const auto maxGeometryBufferHatchBoxes = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(Hatch::CurveHatchBox);

		uint32_t uploadableObjects = (maxIndices - currentIndexCount) / IndicesPerHatchBox;
		uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
		uploadableObjects = core::min(uploadableObjects, maxGeometryBufferHatchBoxes);

		uint32_t remainingObjects = hatch.getHatchBoxCount() - currentObjectInSection;
		uploadableObjects = core::min(uploadableObjects, remainingObjects);

		for (uint32_t i = 0; i < uploadableObjects; i++)
		{
			const Hatch::CurveHatchBox& hatchBox = hatch.getHatchBox(i + currentObjectInSection);

			uint64_t hatchBoxAddress;
			{
				CurveBox curveBox;
				curveBox.aabbMin = hatchBox.aabbMin;
				curveBox.aabbMax = hatchBox.aabbMax;
				memcpy(&curveBox.curveMin[0], &hatchBox.curveMin[0], sizeof(float64_t2) * 3);
				memcpy(&curveBox.curveMax[0], &hatchBox.curveMax[0], sizeof(float64_t2) * 3);

				void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
				memcpy(dst, &curveBox, sizeof(CurveBox));
				hatchBoxAddress = geometryBufferAddress + currentGeometryBufferSize;
				currentGeometryBufferSize += sizeof(CurveBox);
			}

			DrawObject drawObj = {};
			drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::CURVE_BOX) | (0 << 16));
			drawObj.mainObjIndex = mainObjIndex;
			drawObj.geometryAddress = hatchBoxAddress;
			void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount + i;
			memcpy(dst, &drawObj, sizeof(DrawObject));
		}

		// Add Indices
		addCagedObjectIndices_Internal(currentDrawObjectCount, uploadableObjects);
		currentDrawObjectCount += uploadableObjects;
		currentObjectInSection += uploadableObjects;
	}

	//@param oddProvokingVertex is used for our polyline-wide transparency algorithm where we draw the object twice, once to resolve the alpha and another time to draw them
	void addCagedObjectIndices_Internal(uint32_t startObject, uint32_t objectCount)
	{
		constexpr bool oddProvokingVertex = true; // was useful before, might probably deprecate it later for simplicity or it might be useful for some tricks later on
		index_buffer_type* indices = reinterpret_cast<index_buffer_type*>(cpuDrawBuffers.indexBuffer->getPointer()) + currentIndexCount;
		for (uint32_t i = 0u; i < objectCount; ++i)
		{
			index_buffer_type objIndex = startObject + i;
			if (oddProvokingVertex)
			{
				indices[i * 6] = objIndex * 4u + 1u;
				indices[i * 6 + 1u] = objIndex * 4u + 0u;
			}
			else
			{
				indices[i * 6] = objIndex * 4u + 0u;
				indices[i * 6 + 1u] = objIndex * 4u + 1u;
			}
			indices[i * 6 + 2u] = objIndex * 4u + 2u;

			if (oddProvokingVertex)
			{
				indices[i * 6 + 3u] = objIndex * 4u + 1u;
				indices[i * 6 + 4u] = objIndex * 4u + 2u;
			}
			else
			{
				indices[i * 6 + 3u] = objIndex * 4u + 2u;
				indices[i * 6 + 4u] = objIndex * 4u + 1u;
			}
			indices[i * 6 + 5u] = objIndex * 4u + 3u;
		}
		currentIndexCount += objectCount * 6u;
	}

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

	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;

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

class CADApp : public ApplicationBase
{
	constexpr static uint32_t FRAMES_IN_FLIGHT = 3u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	constexpr static uint32_t REQUESTED_WIN_W = 1600u;
	constexpr static uint32_t REQUESTED_WIN_H = 900u;

	//constexpr static uint32_t REQUESTED_WIN_W = 3840u;
	//constexpr static uint32_t REQUESTED_WIN_H = 2160u;

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	core::smart_refctd_ptr<video::IQueryPool> pipelineStatsPool;
	core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpassInitial; // this renderpass will clear the attachment and transition it to COLOR_ATTACHMENT_OPTIMAL
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpassInBetween; // this renderpass will load the attachment and transition it to COLOR_ATTACHMENT_OPTIMAL
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpassFinal; // this renderpass will load the attachment and transition it to PRESENT
	nbl::core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>> framebuffersDynArraySmartPtr;
	std::array<std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxFramesInFlight>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;
	core::smart_refctd_ptr<video::IGPUImage> m_swapchainImages[CommonAPI::InitOutput::MaxSwapChainImageCount];

	int32_t m_resourceIx = -1;
	uint32_t m_SwapchainImageIx = ~0u;

	core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_uploadCmdBuf[FRAMES_IN_FLIGHT] = { nullptr };

	nbl::video::ISwapchain::SCreationParams m_swapchainCreationParams;

	// Related to Drawing Stuff
	Camera2D m_Camera;

	core::smart_refctd_ptr<video::IGPUImageView> pseudoStencilImageView[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUBuffer> globalsBuffer[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUDescriptorSet> descriptorSets[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> graphicsPipeline;
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> debugGraphicsPipeline;
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> descriptorSetLayout;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> graphicsPipelineLayout;

	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> resolveAlphaGraphicsPipeline;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> resolveAlphaPipeLayout;

	DrawBuffersFiller drawBuffers[FRAMES_IN_FLIGHT];

	// For stress test CASE_1
	CPolyline bigPolyline;
	CPolyline bigPolyline2;

	bool fragmentShaderInterlockEnabled = true;

	// TODO: Needs better info about regular scenes and main limiters to improve the allocations in this function
	void initDrawObjects(uint32_t maxObjects)
	{
		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			drawBuffers[i] = DrawBuffersFiller(core::smart_refctd_ptr(utilities));

			uint32_t maxIndices = maxObjects * 6u * 2u;
			drawBuffers[i].allocateIndexBuffer(logicalDevice.get(), maxIndices);
			drawBuffers[i].allocateMainObjectsBuffer(logicalDevice.get(), maxObjects);
			drawBuffers[i].allocateDrawObjectsBuffer(logicalDevice.get(), maxObjects * 5u);
			drawBuffers[i].allocateStylesBuffer(logicalDevice.get(), 16u);
			drawBuffers[i].allocateCustomClipProjectionBuffer(logicalDevice.get(), 128u);

			// * 3 because I just assume there is on average 3x beziers per actual object (cause we approximate other curves/arcs with beziers now)
			size_t geometryBufferSize = maxObjects * sizeof(QuadraticBezierInfo) * 3;
			drawBuffers[i].allocateGeometryBuffer(logicalDevice.get(), geometryBufferSize);
		}

		for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i)
		{
			video::IGPUBuffer::SCreationParams globalsCreationParams = {};
			globalsCreationParams.size = sizeof(Globals);
			globalsCreationParams.usage = video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
			globalsBuffer[i] = logicalDevice->createBuffer(std::move(globalsCreationParams));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = globalsBuffer[i]->getMemoryReqs();
			memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto globalsBufferMem = logicalDevice->allocate(memReq, globalsBuffer[i].get());
		}

		// pseudoStencil

		asset::E_FORMAT pseudoStencilFormat = asset::EF_R32_UINT;

		video::IPhysicalDevice::SImageFormatPromotionRequest promotionRequest = {};
		promotionRequest.originalFormat = asset::EF_R32_UINT;
		promotionRequest.usages = {};
		promotionRequest.usages.storageImageAtomic = true;
		pseudoStencilFormat = physicalDevice->promoteImageFormat(promotionRequest, video::IGPUImage::ET_OPTIMAL);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			video::IGPUImage::SCreationParams imgInfo;
			imgInfo.format = pseudoStencilFormat;
			imgInfo.type = video::IGPUImage::ET_2D;
			imgInfo.extent.width = window->getWidth();
			imgInfo.extent.height = window->getHeight();
			imgInfo.extent.depth = 1u;
			imgInfo.mipLevels = 1u;
			imgInfo.arrayLayers = 1u;
			imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
			imgInfo.flags = asset::IImage::E_CREATE_FLAGS::ECF_NONE;
			imgInfo.usage = asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT;
			imgInfo.initialLayout = video::IGPUImage::EL_UNDEFINED;
			imgInfo.tiling = video::IGPUImage::ET_OPTIMAL;

			auto image = logicalDevice->createImage(std::move(imgInfo));
			auto imageMemReqs = image->getMemoryReqs();
			imageMemReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			logicalDevice->allocate(imageMemReqs, image.get());

			image->setObjectDebugName("pseudoStencil Image");

			video::IGPUImageView::SCreationParams imgViewInfo;
			imgViewInfo.image = std::move(image);
			imgViewInfo.format = pseudoStencilFormat;
			imgViewInfo.viewType = video::IGPUImageView::ET_2D;
			imgViewInfo.flags = video::IGPUImageView::E_CREATE_FLAGS::ECF_NONE;
			imgViewInfo.subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			imgViewInfo.subresourceRange.baseArrayLayer = 0u;
			imgViewInfo.subresourceRange.baseMipLevel = 0u;
			imgViewInfo.subresourceRange.layerCount = 1u;
			imgViewInfo.subresourceRange.levelCount = 1u;

			pseudoStencilImageView[i] = logicalDevice->createImageView(std::move(imgViewInfo));
		}
	}

public:
	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) override
	{
		system = std::move(system);
	}
	video::IAPIConnection* getAPIConnection() override
	{
		return apiConnection.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return logicalDevice.get();
	}
	video::IGPURenderpass* getRenderpass() override
	{
		return renderpassFinal.get();
	}
	void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
	{
		surface = std::move(s);
	}
	void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
	{
		for (int i = 0; i < f.size(); i++)
		{
			auto& fboDynArray = *(framebuffersDynArraySmartPtr.get());
			fboDynArray[i] = core::smart_refctd_ptr(f[i]);
		}
	}
	void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
	{
		swapchain = std::move(s);
	}
	uint32_t getSwapchainImageCount() override
	{
		return swapchain->getImageCount();
	}
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_UNKNOWN;
	}

	nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> createRenderpass(
		nbl::asset::E_FORMAT colorAttachmentFormat,
		nbl::asset::E_FORMAT baseDepthFormat,
		nbl::video::IGPURenderpass::E_LOAD_OP loadOp,
		nbl::asset::IImage::E_LAYOUT initialLayout,
		nbl::asset::IImage::E_LAYOUT finalLayout)
	{
		using namespace nbl;

		bool useDepth = baseDepthFormat != nbl::asset::EF_UNKNOWN;
		nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN;
		if (useDepth)
		{
			depthFormat = logicalDevice->getPhysicalDevice()->promoteImageFormat(
				{ baseDepthFormat, nbl::video::IPhysicalDevice::SFormatImageUsages::SUsage(nbl::asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT) },
				nbl::video::IGPUImage::ET_OPTIMAL
			);
			assert(depthFormat != nbl::asset::EF_UNKNOWN);
		}

		nbl::video::IGPURenderpass::SCreationParams::SAttachmentDescription attachments[2];
		attachments[0].initialLayout = initialLayout;
		attachments[0].finalLayout = finalLayout;
		attachments[0].format = colorAttachmentFormat;
		attachments[0].samples = asset::IImage::ESCF_1_BIT;
		attachments[0].loadOp = loadOp;
		attachments[0].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

		attachments[1].initialLayout = asset::IImage::EL_UNDEFINED;
		attachments[1].finalLayout = asset::IImage::EL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].format = depthFormat;
		attachments[1].samples = asset::IImage::ESCF_1_BIT;
		attachments[1].loadOp = loadOp;
		attachments[1].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
		colorAttRef.attachment = 0u;
		colorAttRef.layout = asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef depthStencilAttRef;
		depthStencilAttRef.attachment = 1u;
		depthStencilAttRef.layout = asset::IImage::EL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription sp;
		sp.pipelineBindPoint = asset::EPBP_GRAPHICS;
		sp.colorAttachmentCount = 1u;
		sp.colorAttachments = &colorAttRef;
		if (useDepth) {
			sp.depthStencilAttachment = &depthStencilAttRef;
		}
		else {
			sp.depthStencilAttachment = nullptr;
		}
		sp.flags = nbl::video::IGPURenderpass::ESDF_NONE;
		sp.inputAttachmentCount = 0u;
		sp.inputAttachments = nullptr;
		sp.preserveAttachmentCount = 0u;
		sp.preserveAttachments = nullptr;
		sp.resolveAttachments = nullptr;

		nbl::video::IGPURenderpass::SCreationParams rp_params;
		rp_params.attachmentCount = (useDepth) ? 2u : 1u;
		rp_params.attachments = attachments;
		rp_params.dependencies = nullptr;
		rp_params.dependencyCount = 0u;
		rp_params.subpasses = &sp;
		rp_params.subpassCount = 1u;

		return logicalDevice->createRenderpass(rp_params);
	}

	void getAndLogQueryPoolResults()
	{
#ifdef BEZIER_CAGE_ADAPTIVE_T_FIND // results for bezier show an optimal number of 0.14 for T
		{
			uint32_t samples_passed[1] = {};
			auto queryResultFlags = core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS>(video::IQueryPool::EQRF_WAIT_BIT);
			logicalDevice->getQueryPoolResults(pipelineStatsPool.get(), 0u, 1u, sizeof(samples_passed), samples_passed, sizeof(uint32_t), queryResultFlags);
			logger->log("[WAIT] SamplesPassed[0] = %d", system::ILogger::ELL_INFO, samples_passed[0]);
			std::cout << MinT << ", " << PrevSamples << std::endl;
			if (PrevSamples > samples_passed[0]) {
				PrevSamples = samples_passed[0];
				MinT = (sin(T) + 1.01f) / 4.03f;
			}
		}
#endif
	}

	APP_CONSTRUCTOR(CADApp);

	void onAppInitialized_impl() override
	{
		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
		std::array<asset::E_FORMAT, 1> acceptableSurfaceFormats = { asset::EF_B8G8R8A8_UNORM };

		CommonAPI::InitParams initParams;
		initParams.windowCb = core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback>(this);
		initParams.window = core::smart_refctd_ptr(window);
		initParams.apiType = video::EAT_VULKAN;
		initParams.appName = { "62.CAD" };
		initParams.framesInFlight = FRAMES_IN_FLIGHT;
		initParams.windowWidth = REQUESTED_WIN_W;
		initParams.windowHeight = REQUESTED_WIN_H;
		initParams.swapchainImageCount = 3u;
		initParams.swapchainImageUsage = swapchainImageUsage;
		initParams.depthFormat = getDepthFormat();
		initParams.acceptableSurfaceFormats = acceptableSurfaceFormats.data();
		initParams.acceptableSurfaceFormatCount = acceptableSurfaceFormats.size();
		initParams.physicalDeviceFilter.requiredFeatures.bufferDeviceAddress = true;
		initParams.physicalDeviceFilter.requiredFeatures.shaderFloat64 = true;
		initParams.physicalDeviceFilter.requiredFeatures.fillModeNonSolid = DebugMode;
		initParams.physicalDeviceFilter.requiredFeatures.fragmentShaderPixelInterlock = FragmentShaderPixelInterlock;
		initParams.physicalDeviceFilter.requiredFeatures.pipelineStatisticsQuery = true;
		initParams.physicalDeviceFilter.requiredFeatures.shaderClipDistance = true;
		initParams.physicalDeviceFilter.requiredFeatures.scalarBlockLayout = true;
		auto initOutput = CommonAPI::InitWithDefaultExt(std::move(initParams));

		system = std::move(initOutput.system);
		window = std::move(initParams.window);
		windowCb = std::move(initParams.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		physicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		utilities = std::move(initOutput.utilities);
		queues = std::move(initOutput.queues);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);
		windowManager = std::move(initOutput.windowManager);
		// renderpass = std::move(initOutput.renderToSwapchainRenderpass);
		m_swapchainCreationParams = std::move(initOutput.swapchainCreationParams);

		fragmentShaderInterlockEnabled = logicalDevice->getEnabledFeatures().fragmentShaderPixelInterlock;

		{
			video::IQueryPool::SCreationParams queryPoolCreationParams = {};
			queryPoolCreationParams.queryType = video::IQueryPool::EQT_PIPELINE_STATISTICS;
			queryPoolCreationParams.queryCount = 1u;
			queryPoolCreationParams.pipelineStatisticsFlags = video::IQueryPool::EPSF_FRAGMENT_SHADER_INVOCATIONS_BIT;
			pipelineStatsPool = logicalDevice->createQueryPool(std::move(queryPoolCreationParams));
		}

		renderpassInitial = createRenderpass(m_swapchainCreationParams.surfaceFormat.format, getDepthFormat(), nbl::video::IGPURenderpass::ELO_CLEAR, asset::IImage::EL_UNDEFINED, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL);
		renderpassInBetween = createRenderpass(m_swapchainCreationParams.surfaceFormat.format, getDepthFormat(), nbl::video::IGPURenderpass::ELO_LOAD, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL);
		renderpassFinal = createRenderpass(m_swapchainCreationParams.surfaceFormat.format, getDepthFormat(), nbl::video::IGPURenderpass::ELO_LOAD, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL, asset::IImage::EL_PRESENT_SRC);

		commandPools = std::move(initOutput.commandPools);
		const auto& graphicsCommandPools = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS];
		const auto& transferCommandPools = commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP];

		CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, window->getWidth(), window->getHeight(), swapchain);

		framebuffersDynArraySmartPtr = CommonAPI::createFBOWithSwapchainImages(
			swapchain->getImageCount(), window->getWidth(), window->getHeight(),
			logicalDevice, swapchain, renderpassFinal,
			getDepthFormat()
		);

		const uint32_t swapchainImageCount = swapchain->getImageCount();
		for (uint32_t i = 0; i < swapchainImageCount; ++i)
		{
			auto& fboDynArray = *(framebuffersDynArraySmartPtr.get());
			m_swapchainImages[i] = fboDynArray[i]->getCreationParameters().attachments[0u]->getCreationParameters().image;
		}

		video::IGPUObjectFromAssetConverter CPU2GPU;

		core::smart_refctd_ptr<video::IGPUSpecializedShader> shaders[4u] = {};
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			core::smart_refctd_ptr<asset::ICPUSpecializedShader> cpuShaders[4u] = {};
			constexpr auto vertexShaderPath = "../vertex_shader.hlsl";
			constexpr auto fragmentShaderPath = "../fragment_shader.hlsl";
			constexpr auto debugfragmentShaderPath = "../fragment_shader_debug.hlsl";
			constexpr auto resolveAlphasShaderPath = "../resolve_alphas.hlsl";
			cpuShaders[0u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(vertexShaderPath, params).getContents().begin());
			cpuShaders[1u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(fragmentShaderPath, params).getContents().begin());
			cpuShaders[2u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(debugfragmentShaderPath, params).getContents().begin());
			cpuShaders[3u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(resolveAlphasShaderPath, params).getContents().begin());
			cpuShaders[0u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			cpuShaders[1u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			cpuShaders[2u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			cpuShaders[3u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			auto gpuShaders = CPU2GPU.getGPUObjectsFromAssets(cpuShaders, cpuShaders + 4u, cpu2gpuParams);
			shaders[0u] = gpuShaders->begin()[0u];
			shaders[1u] = gpuShaders->begin()[1u];
			shaders[2u] = gpuShaders->begin()[2u];
			shaders[3u] = gpuShaders->begin()[3u];
		}

		initDrawObjects(20480u);

		// Create DescriptorSetLayout, PipelineLayout and update DescriptorSets
		{
			constexpr uint32_t BindingCount = 6u;
			video::IGPUDescriptorSetLayout::SBinding bindings[BindingCount] = {};
			bindings[0].binding = 0u;
			bindings[0].type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER;
			bindings[0].count = 1u;
			bindings[0].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

			bindings[1].binding = 1u;
			bindings[1].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bindings[1].count = 1u;
			bindings[1].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

			bindings[2].binding = 2u;
			bindings[2].type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
			bindings[2].count = 1u;
			bindings[2].stageFlags = asset::IShader::ESS_FRAGMENT;

			bindings[3].binding = 3u;
			bindings[3].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bindings[3].count = 1u;
			bindings[3].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

			bindings[4].binding = 4u;
			bindings[4].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bindings[4].count = 1u;
			bindings[4].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

			bindings[5].binding = 5u;
			bindings[5].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bindings[5].count = 1u;
			bindings[5].stageFlags = asset::IShader::ESS_VERTEX;

			descriptorSetLayout = logicalDevice->createDescriptorSetLayout(bindings, bindings + BindingCount);

			nbl::core::smart_refctd_ptr<nbl::video::IDescriptorPool> descriptorPool = nullptr;
			{
				nbl::video::IDescriptorPool::SCreateInfo createInfo = {};
				createInfo.flags = nbl::video::IDescriptorPool::ECF_NONE;
				createInfo.maxSets = 128u;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] = FRAMES_IN_FLIGHT;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] = 4 * FRAMES_IN_FLIGHT;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)] = FRAMES_IN_FLIGHT;

				descriptorPool = logicalDevice->createDescriptorPool(std::move(createInfo));
			}

			for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
			{
				descriptorSets[i] = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(descriptorSetLayout));
				constexpr uint32_t DescriptorCount = 6u;
				video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[DescriptorCount] = {};
				descriptorInfos[0u].info.buffer.offset = 0u;
				descriptorInfos[0u].info.buffer.size = globalsBuffer[i]->getCreationParams().size;
				descriptorInfos[0u].desc = globalsBuffer[i];

				descriptorInfos[1u].info.buffer.offset = 0u;
				descriptorInfos[1u].info.buffer.size = drawBuffers[i].gpuDrawBuffers.drawObjectsBuffer->getCreationParams().size;
				descriptorInfos[1u].desc = drawBuffers[i].gpuDrawBuffers.drawObjectsBuffer;

				descriptorInfos[2u].info.image.imageLayout = asset::IImage::E_LAYOUT::EL_GENERAL;
				descriptorInfos[2u].info.image.sampler = nullptr;
				descriptorInfos[2u].desc = pseudoStencilImageView[i];

				descriptorInfos[3u].info.buffer.offset = 0u;
				descriptorInfos[3u].info.buffer.size = drawBuffers[i].gpuDrawBuffers.lineStylesBuffer->getCreationParams().size;
				descriptorInfos[3u].desc = drawBuffers[i].gpuDrawBuffers.lineStylesBuffer;

				descriptorInfos[4u].info.buffer.offset = 0u;
				descriptorInfos[4u].info.buffer.size = drawBuffers[i].gpuDrawBuffers.mainObjectsBuffer->getCreationParams().size;
				descriptorInfos[4u].desc = drawBuffers[i].gpuDrawBuffers.mainObjectsBuffer;

				descriptorInfos[5u].info.buffer.offset = 0u;
				descriptorInfos[5u].info.buffer.size = drawBuffers[i].gpuDrawBuffers.customClipProjectionBuffer->getCreationParams().size;
				descriptorInfos[5u].desc = drawBuffers[i].gpuDrawBuffers.customClipProjectionBuffer;

				video::IGPUDescriptorSet::SWriteDescriptorSet descriptorUpdates[6u] = {};
				descriptorUpdates[0u].dstSet = descriptorSets[i].get();
				descriptorUpdates[0u].binding = 0u;
				descriptorUpdates[0u].arrayElement = 0u;
				descriptorUpdates[0u].count = 1u;
				descriptorUpdates[0u].descriptorType = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER;
				descriptorUpdates[0u].info = &descriptorInfos[0u];

				descriptorUpdates[1u].dstSet = descriptorSets[i].get();
				descriptorUpdates[1u].binding = 1u;
				descriptorUpdates[1u].arrayElement = 0u;
				descriptorUpdates[1u].count = 1u;
				descriptorUpdates[1u].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				descriptorUpdates[1u].info = &descriptorInfos[1u];

				descriptorUpdates[2u].dstSet = descriptorSets[i].get();
				descriptorUpdates[2u].binding = 2u;
				descriptorUpdates[2u].arrayElement = 0u;
				descriptorUpdates[2u].count = 1u;
				descriptorUpdates[2u].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
				descriptorUpdates[2u].info = &descriptorInfos[2u];

				descriptorUpdates[3u].dstSet = descriptorSets[i].get();
				descriptorUpdates[3u].binding = 3u;
				descriptorUpdates[3u].arrayElement = 0u;
				descriptorUpdates[3u].count = 1u;
				descriptorUpdates[3u].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				descriptorUpdates[3u].info = &descriptorInfos[3u];

				descriptorUpdates[4u].dstSet = descriptorSets[i].get();
				descriptorUpdates[4u].binding = 4u;
				descriptorUpdates[4u].arrayElement = 0u;
				descriptorUpdates[4u].count = 1u;
				descriptorUpdates[4u].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				descriptorUpdates[4u].info = &descriptorInfos[4u];

				descriptorUpdates[5u].dstSet = descriptorSets[i].get();
				descriptorUpdates[5u].binding = 5u;
				descriptorUpdates[5u].arrayElement = 0u;
				descriptorUpdates[5u].count = 1u;
				descriptorUpdates[5u].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				descriptorUpdates[5u].info = &descriptorInfos[5u];

				logicalDevice->updateDescriptorSets(DescriptorCount, descriptorUpdates, 0u, nullptr);
			}

			graphicsPipelineLayout = logicalDevice->createPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);
		}

		// Shared Blend Params between pipelines
		asset::SBlendParams blendParams = {};
		blendParams.blendParams[0u].blendEnable = true;
		blendParams.blendParams[0u].srcColorFactor = asset::EBF_SRC_ALPHA;
		blendParams.blendParams[0u].dstColorFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
		blendParams.blendParams[0u].colorBlendOp = asset::EBO_ADD;
		blendParams.blendParams[0u].srcAlphaFactor = asset::EBF_ONE;
		blendParams.blendParams[0u].dstAlphaFactor = asset::EBF_ZERO;
		blendParams.blendParams[0u].alphaBlendOp = asset::EBO_ADD;
		blendParams.blendParams[0u].colorWriteMask = (1u << 4u) - 1u;

		// Create Alpha Resovle Pipeline
		{
			auto fsTriangleProtoPipe = nbl::ext::FullScreenTriangle::createProtoPipeline(cpu2gpuParams, 0u);
			std::get<asset::SBlendParams>(fsTriangleProtoPipe) = blendParams;

			auto constants = std::get<asset::SPushConstantRange>(fsTriangleProtoPipe);
			resolveAlphaPipeLayout = logicalDevice->createPipelineLayout(&constants, &constants+1, core::smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);
			auto fsTriangleRenderPassIndependantPipe = nbl::ext::FullScreenTriangle::createRenderpassIndependentPipeline(logicalDevice.get(), fsTriangleProtoPipe, core::smart_refctd_ptr(shaders[3u]), core::smart_refctd_ptr(resolveAlphaPipeLayout));

			video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineCreateInfo = {};
			graphicsPipelineCreateInfo.renderpassIndependent = fsTriangleRenderPassIndependantPipe;
			graphicsPipelineCreateInfo.renderpass = renderpassFinal;
			resolveAlphaGraphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(graphicsPipelineCreateInfo));
		}

		// Create Main Graphics Pipelines 
		{
			video::IGPURenderpassIndependentPipeline::SCreationParams renderpassIndependantPipeInfo = {};
			renderpassIndependantPipeInfo.layout = graphicsPipelineLayout;
			renderpassIndependantPipeInfo.shaders[0u] = shaders[0u];
			renderpassIndependantPipeInfo.shaders[1u] = shaders[1u];
			// renderpassIndependantPipeInfo.vertexInput; no gpu vertex buffers
			renderpassIndependantPipeInfo.blend = blendParams;

			renderpassIndependantPipeInfo.primitiveAssembly.primitiveType = asset::E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST;
			renderpassIndependantPipeInfo.rasterization.depthTestEnable = false;
			renderpassIndependantPipeInfo.rasterization.depthWriteEnable = false;
			renderpassIndependantPipeInfo.rasterization.stencilTestEnable = false;
			renderpassIndependantPipeInfo.rasterization.polygonMode = asset::EPM_FILL;
			renderpassIndependantPipeInfo.rasterization.faceCullingMode = asset::EFCM_NONE;

			core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> renderpassIndependant;
			bool succ = logicalDevice->createRenderpassIndependentPipelines(
				nullptr,
				core::SRange<const video::IGPURenderpassIndependentPipeline::SCreationParams>(&renderpassIndependantPipeInfo, &renderpassIndependantPipeInfo + 1u),
				&renderpassIndependant);
			assert(succ);

			video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineCreateInfo = {};
			graphicsPipelineCreateInfo.renderpassIndependent = renderpassIndependant;
			graphicsPipelineCreateInfo.renderpass = renderpassFinal;
			graphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(graphicsPipelineCreateInfo));

			if constexpr (DebugMode)
			{
				core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> renderpassIndependantDebug;
				renderpassIndependantPipeInfo.shaders[1u] = shaders[2u];
				renderpassIndependantPipeInfo.rasterization.polygonMode = asset::EPM_LINE;
				succ = logicalDevice->createRenderpassIndependentPipelines(
					nullptr,
					core::SRange<const video::IGPURenderpassIndependentPipeline::SCreationParams>(&renderpassIndependantPipeInfo, &renderpassIndependantPipeInfo + 1u),
					&renderpassIndependantDebug);
				assert(succ);

				video::IGPUGraphicsPipeline::SCreationParams debugGraphicsPipelineCreateInfo = {};
				debugGraphicsPipelineCreateInfo.renderpassIndependent = renderpassIndependantDebug;
				debugGraphicsPipelineCreateInfo.renderpass = renderpassFinal;
				debugGraphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(debugGraphicsPipelineCreateInfo));
			}
		}

		for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
		{
			logicalDevice->createCommandBuffers(
				graphicsCommandPools[i].get(),
				video::IGPUCommandBuffer::EL_PRIMARY,
				1,
				m_cmdbuf + i);

			logicalDevice->createCommandBuffers(
				transferCommandPools[i].get(),
				video::IGPUCommandBuffer::EL_PRIMARY,
				1,
				m_uploadCmdBuf + i);
		}

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			m_frameComplete[i] = logicalDevice->createFence(video::IGPUFence::ECF_SIGNALED_BIT);
			m_imageAcquire[i] = logicalDevice->createSemaphore();
			m_renderFinished[i] = logicalDevice->createSemaphore();
		}

		m_Camera.setOrigin({ 0.0, 0.0 });
		m_Camera.setAspectRatio((double)window->getWidth() / window->getHeight());
		m_Camera.setSize(200.0);

		m_timeElapsed = 0.0;


		if constexpr (mode == ExampleMode::CASE_1)
		{
			{
				std::vector<float64_t2> linePoints;
				for (uint32_t i = 0u; i < 20u; ++i)
				{
					for (uint32_t i = 0u; i < 256u; ++i)
					{
						double y = -112.0 + i * 1.1;
						linePoints.push_back({ -200.0, y });
						linePoints.push_back({ +200.0, y });
					}
					for (uint32_t i = 0u; i < 256u; ++i)
					{
						double x = -200.0 + i * 1.5;
						linePoints.push_back({ x, -100.0 });
						linePoints.push_back({ x, +100.0 });
					}
				}
				bigPolyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
			}
			{
				std::vector<float64_t2> linePoints;
				for (uint32_t i = 0u; i < 20u; ++i)
				{
					for (uint32_t i = 0u; i < 256u; ++i)
					{
						double y = -112.0 + i * 1.1;
						double x = -200.0 + i * 1.5;
						linePoints.push_back({ -200.0 + x, y });
						linePoints.push_back({ +200.0 + x, y });
					}
					for (uint32_t i = 0u; i < 256u; ++i)
					{
						double y = -112.0 + i * 1.1;
						double x = -200.0 + i * 1.5;
						linePoints.push_back({ x, -100.0 + y });
						linePoints.push_back({ x, +100.0 + y });
					}
				}
				bigPolyline2.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
			}
		}

	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	double getScreenToWorldRatio(const float64_t3x3& viewProjectionMatrix, uint32_t2 windowSize)
	{
		double idx_0_0 = viewProjectionMatrix[0u][0u] * (windowSize.x / 2.0);
		double idx_1_1 = viewProjectionMatrix[1u][1u] * (windowSize.y / 2.0);
		double det_2x2_mat = idx_0_0 * idx_1_1;
		return static_cast<float>(core::sqrt(core::abs(det_2x2_mat)));
	}

	void beginFrameRender()
	{
		auto& cb = m_cmdbuf[m_resourceIx];
		auto& commandPool = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS][m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];
		logicalDevice->blockForFences(1u, &fence.get());
		logicalDevice->resetFences(1u, &fence.get());

		m_SwapchainImageIx = 0u;
		auto acquireResult = swapchain->acquireNextImage(m_imageAcquire[m_resourceIx].get(), nullptr, &m_SwapchainImageIx);
		assert(acquireResult == video::ISwapchain::E_ACQUIRE_IMAGE_RESULT::EAIR_SUCCESS);

		core::smart_refctd_ptr<video::IGPUImage> swapchainImg = m_swapchainImages[m_SwapchainImageIx];

		// safe to proceed
		cb->reset(video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT); // TODO: Begin doesn't release the resources in the command pool, meaning the old swapchains never get dropped
		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT); // TODO: Reset Frame's CommandPool
		cb->beginDebugMarker("Frame");

		

		float64_t3x3 projectionToNDC;
		// TODO: figure out why the matrix multiplication overload isn't getting detected here
		// 
		//if constexpr (DebugRotatingViewProj)
		//{
		//	double theta = (m_timeElapsed * 0.00008) * (2.0 * nbl::core::PI<double>());
		//
		//	auto rotation = float64_t3x3(
		//		cos(theta), -sin(theta), 0.0,
		//		sin(theta), cos(theta), 1.0,
		//		0.0, 0.0, 1.0
		//	);
		//
		//	auto vp = m_Camera.constructViewProjection();
		//	projectionToNDC = nbl::hlsl::mul(rotation, vp);
		//}
		//else
		//{
		//	projectionToNDC = m_Camera.constructViewProjection();
		//}
		projectionToNDC = m_Camera.constructViewProjection();
		
		{
			double u0 = 0;
			double u1 = 0;
			double u2 = -30647296;
			double u3 = -2.8396454064e10;
			double u4 = 1.7068323831e10;
			std::array<double, 4> solution = hatchutils::solveQuarticRoots(u0, u1, u2, u3, u4, 0.0, 1.0);
			printf("quartic roots %fx^4 + %fx^3 %fx^2 + %fx + %f = 0\n%f %f %f %f\nresposta wolfram: -927.157273 0.600682946\n", u0, u1, u2, u3, u4, solution[0], solution[1], solution[2], solution[3]);
		}

		{
			double u0 = -2261956;
			double u1 = 11632344;
			double u2 = -879456248;
			double u3 = -4.0629051748e10;
			double u4 = 2.5159935175e10;
			std::array<double, 4> solution = hatchutils::solveQuarticRoots(u0, u1, u2, u3, u4, 0.0, 1.0);
			printf("quartic roots %fx^4 + %fx^3 %fx^2 + %fx + %f = 0\n%f %f %f %f\nresposta wolfram: -20.3710249, 0.611230330\n", u0, u1, u2, u3, u4, solution[0], solution[1], solution[2], solution[3]);
		}

		Globals globalData = {};
		globalData.antiAliasingFactor = 1.0; // +abs(cos(m_timeElapsed * 0.0008)) * 20.0f;
		globalData.resolution = uint32_t2{ window->getWidth(), window->getHeight() };
		globalData.defaultClipProjection.projectionToNDC = projectionToNDC;
		globalData.defaultClipProjection.minClipNDC = float32_t2(-1.0, -1.0);
		globalData.defaultClipProjection.maxClipNDC = float32_t2(+1.0, +1.0);
		auto screenToWorld = getScreenToWorldRatio(globalData.defaultClipProjection.projectionToNDC, globalData.resolution);
		globalData.screenToWorldRatio = (float) screenToWorld;
		globalData.worldToScreenRatio = (float) (1.0f/screenToWorld);
		bool updateSuccess = cb->updateBuffer(globalsBuffer[m_resourceIx].get(), 0ull, sizeof(Globals), &globalData);
		assert(updateSuccess);

		// Clear pseudoStencil
		{
			auto pseudoStencilImage = pseudoStencilImageView[m_resourceIx]->getCreationParameters().image;

			nbl::video::IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
			imageBarriers[0].barrier.srcAccessMask = nbl::asset::EAF_NONE;
			imageBarriers[0].barrier.dstAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
			imageBarriers[0].oldLayout = nbl::asset::IImage::EL_UNDEFINED;
			imageBarriers[0].newLayout = nbl::asset::IImage::EL_GENERAL;
			imageBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].image = pseudoStencilImage;
			imageBarriers[0].subresourceRange.aspectMask = nbl::asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(nbl::asset::EPSF_TOP_OF_PIPE_BIT, nbl::asset::EPSF_TRANSFER_BIT, nbl::asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);

			uint32_t pseudoStencilInvalidValue = core::bitfieldInsert<uint32_t>(0u, InvalidMainObjectIdx, AlphaBits, MainObjectIdxBits);
			asset::SClearColorValue clear = {};
			clear.uint32[0] = pseudoStencilInvalidValue;

			asset::IImage::SSubresourceRange subresourceRange = {};
			subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			subresourceRange.baseArrayLayer = 0u;
			subresourceRange.baseMipLevel = 0u;
			subresourceRange.layerCount = 1u;
			subresourceRange.levelCount = 1u;

			cb->clearColorImage(pseudoStencilImage.get(), asset::IImage::EL_GENERAL, &clear, 1u, &subresourceRange);
		}

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { window->getWidth(), window->getHeight() };
			asset::SClearValue clear[2] = {};
			clear[0].color.float32[0] = 0.8f;
			clear[0].color.float32[1] = 0.8f;
			clear[0].color.float32[2] = 0.8f;
			clear[0].color.float32[3] = 0.f;
			clear[1].depthStencil.depth = 1.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = framebuffersDynArraySmartPtr->begin()[m_SwapchainImageIx];
			beginInfo.renderpass = renderpassInitial;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		// you could do this later but only use renderpassInitial on first draw
		cb->beginRenderPass(&beginInfo, asset::ESC_INLINE);
		cb->endRenderPass();
	}

	void pipelineBarriersBeforeDraw(video::IGPUCommandBuffer* const cb)
	{
		auto& currentDrawBuffers = drawBuffers[m_resourceIx];
		{
			auto pseudoStencilImage = pseudoStencilImageView[m_resourceIx]->getCreationParameters().image;
			nbl::video::IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
			imageBarriers[0].barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
			imageBarriers[0].barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT | nbl::asset::EAF_SHADER_WRITE_BIT; // SYNC_FRAGMENT_SHADER_SHADER_SAMPLED_READ | SYNC_FRAGMENT_SHADER_SHADER_STORAGE_READ | SYNC_FRAGMENT_SHADER_UNIFORM_READ
			imageBarriers[0].oldLayout = nbl::asset::IImage::EL_GENERAL;
			imageBarriers[0].newLayout = nbl::asset::IImage::EL_GENERAL;
			imageBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].image = pseudoStencilImage;
			imageBarriers[0].subresourceRange.aspectMask = nbl::asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(nbl::asset::EPSF_TRANSFER_BIT, nbl::asset::EPSF_FRAGMENT_SHADER_BIT, nbl::asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);

		}
		{
			nbl::video::IGPUCommandBuffer::SBufferMemoryBarrier bufferBarriers[1u] = {};
			bufferBarriers[0u].barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
			bufferBarriers[0u].barrier.dstAccessMask = nbl::asset::EAF_INDEX_READ_BIT;
			bufferBarriers[0u].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[0u].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[0u].buffer = currentDrawBuffers.gpuDrawBuffers.indexBuffer;
			bufferBarriers[0u].offset = 0u;
			bufferBarriers[0u].size = currentDrawBuffers.getCurrentIndexBufferSize();
			cb->pipelineBarrier(nbl::asset::EPSF_TRANSFER_BIT, nbl::asset::EPSF_VERTEX_INPUT_BIT, nbl::asset::EDF_NONE, 0u, nullptr, 1u, bufferBarriers, 0u, nullptr);
		}
		{
			constexpr uint32_t MaxBufferBarriersCount = 5u;
			uint32_t bufferBarriersCount = 0u;
			nbl::video::IGPUCommandBuffer::SBufferMemoryBarrier bufferBarriers[MaxBufferBarriersCount] = {};

			if (globalsBuffer[m_resourceIx]->getSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
				bufferBarrier.barrier.dstAccessMask = nbl::asset::EAF_UNIFORM_READ_BIT;
				bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.buffer = globalsBuffer[m_resourceIx];
				bufferBarrier.offset = 0u;
				bufferBarrier.size = globalsBuffer[m_resourceIx]->getSize();
			}
			if (currentDrawBuffers.getCurrentDrawObjectsBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
				bufferBarrier.barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT;
				bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.buffer = currentDrawBuffers.gpuDrawBuffers.drawObjectsBuffer;
				bufferBarrier.offset = 0u;
				bufferBarrier.size = currentDrawBuffers.getCurrentDrawObjectsBufferSize();
			}
			if (currentDrawBuffers.getCurrentGeometryBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
				bufferBarrier.barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT;
				bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.buffer = currentDrawBuffers.gpuDrawBuffers.geometryBuffer;
				bufferBarrier.offset = 0u;
				bufferBarrier.size = currentDrawBuffers.getCurrentGeometryBufferSize();
			}
			if (currentDrawBuffers.getCurrentLineStylesBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
				bufferBarrier.barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT;
				bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.buffer = currentDrawBuffers.gpuDrawBuffers.lineStylesBuffer;
				bufferBarrier.offset = 0u;
				bufferBarrier.size = currentDrawBuffers.getCurrentLineStylesBufferSize();
			}
			if (currentDrawBuffers.getCurrentCustomClipProjectionBufferSize() > 0u)
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
				bufferBarrier.barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT;
				bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bufferBarrier.buffer = currentDrawBuffers.gpuDrawBuffers.customClipProjectionBuffer;
				bufferBarrier.offset = 0u;
				bufferBarrier.size = currentDrawBuffers.getCurrentCustomClipProjectionBufferSize();
			}
			cb->pipelineBarrier(nbl::asset::EPSF_TRANSFER_BIT, nbl::asset::EPSF_VERTEX_SHADER_BIT | nbl::asset::EPSF_FRAGMENT_SHADER_BIT, nbl::asset::EDF_NONE, 0u, nullptr, bufferBarriersCount, bufferBarriers, 0u, nullptr);
		}
	}

	uint32_t m_hatchDebugStep = 0u;

	void endFrameRender()
	{
		auto& cb = m_cmdbuf[m_resourceIx];

		uint32_t windowWidth = swapchain->getCreationParameters().width;
		uint32_t windowHeight = swapchain->getCreationParameters().height;

		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = windowWidth;
		vp.height = windowHeight;
		cb->setViewport(0u, 1u, &vp);

		VkRect2D scissor;
		scissor.extent = { windowWidth, windowHeight };
		scissor.offset = { 0, 0 };
		cb->setScissor(0u, 1u, &scissor);

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { window->getWidth(), window->getHeight() };

			beginInfo.clearValueCount = 0u;
			beginInfo.framebuffer = framebuffersDynArraySmartPtr->begin()[m_SwapchainImageIx];
			beginInfo.renderpass = renderpassFinal;
			beginInfo.renderArea = area;
			beginInfo.clearValues = nullptr;
		}

		pipelineBarriersBeforeDraw(cb.get());

		cb->resetQueryPool(pipelineStatsPool.get(), 0u, 1u);
		cb->beginQuery(pipelineStatsPool.get(), 0);

		cb->beginRenderPass(&beginInfo, asset::ESC_INLINE);

		const uint32_t currentIndexCount = drawBuffers[m_resourceIx].getIndexCount();
		cb->bindDescriptorSets(asset::EPBP_GRAPHICS, graphicsPipelineLayout.get(), 0u, 1u, &descriptorSets[m_resourceIx].get());
		cb->bindIndexBuffer(drawBuffers[m_resourceIx].gpuDrawBuffers.indexBuffer.get(), 0u, asset::EIT_32BIT);
		cb->bindGraphicsPipeline(graphicsPipeline.get());
		cb->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);

		if (fragmentShaderInterlockEnabled)
		{
			cb->bindDescriptorSets(asset::EPBP_GRAPHICS, resolveAlphaPipeLayout.get(), 0u, 1u, &descriptorSets[m_resourceIx].get());
			cb->bindGraphicsPipeline(resolveAlphaGraphicsPipeline.get());
			nbl::ext::FullScreenTriangle::recordDrawCalls(resolveAlphaGraphicsPipeline, 0u, swapchain->getPreTransform(), cb.get());
		}

		if constexpr (DebugMode)
		{
			cb->bindDescriptorSets(asset::EPBP_GRAPHICS, graphicsPipelineLayout.get(), 0u, 1u, &descriptorSets[m_resourceIx].get());
			cb->bindGraphicsPipeline(debugGraphicsPipeline.get());
			cb->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);
		}
		cb->endQuery(pipelineStatsPool.get(), 0);
		cb->endRenderPass();

		cb->endDebugMarker();
		cb->end();

	}

	video::IGPUQueue::SSubmitInfo addObjects(video::IGPUQueue* submissionQueue, video::IGPUFence* submissionFence, video::IGPUQueue::SSubmitInfo& intendedNextSubmit)
	{
		// we record upload of our objects and if we failed to allocate we submit everything
		if (!intendedNextSubmit.isValid() || intendedNextSubmit.commandBufferCount <= 0u)
		{
			// log("intendedNextSubmit is invalid.", nbl::system::ILogger::ELL_ERROR);
			assert(false);
			return intendedNextSubmit;
		}

		// Use the last command buffer in intendedNextSubmit, it should be in recording state
		auto& cmdbuf = intendedNextSubmit.commandBuffers[intendedNextSubmit.commandBufferCount - 1];

		assert(cmdbuf->getState() == video::IGPUCommandBuffer::ES_RECORDING && cmdbuf->isResettable());
		assert(cmdbuf->getRecordingFlags().hasFlags(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT));

		auto* cmdpool = cmdbuf->getPool();
		assert(cmdpool->getQueueFamilyIndex() == submissionQueue->getFamilyIndex());

		auto& currentDrawBuffers = drawBuffers[m_resourceIx];
		currentDrawBuffers.setSubmitDrawsFunction(
			[&](video::IGPUQueue* submissionQueue, video::IGPUFence* submissionFence, video::IGPUQueue::SSubmitInfo intendedNextSubmit)
			{
				return submitInBetweenDraws(m_resourceIx, submissionQueue, submissionFence, intendedNextSubmit);
			}
		);
		currentDrawBuffers.reset();

		if constexpr (mode == ExampleMode::CASE_0)
		{
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 5.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

			CPolyline polyline;
			{
				std::vector<float64_t2> linePoints;
				linePoints.push_back({ -50.0, -50.0 });
				linePoints.push_back({ 50.0, 50.0 });
				polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
			}

			intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_1)
		{
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 0.8f;
			style.color = float32_t4(0.619f, 0.325f, 0.709f, 0.2f);

			CPULineStyle style2 = {};
			style2.screenSpaceLineWidth = 0.0f;
			style2.worldSpaceLineWidth = 0.8f;
			style2.color = float32_t4(0.119f, 0.825f, 0.709f, 0.5f);

			intendedNextSubmit = currentDrawBuffers.drawPolyline(bigPolyline, style, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
			intendedNextSubmit = currentDrawBuffers.drawPolyline(bigPolyline2, style2, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_2)
		{
			auto debug = [&](CPolyline polyline, CPULineStyle lineStyle)
			{
				intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline, lineStyle, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
			};
			
			int32_t hatchDebugStep = m_hatchDebugStep;

			{
				std::vector <CPolyline> polylines;
				auto line = [&](float64_t2 begin, float64_t2 end) {
					std::vector<float64_t2> points = {
						begin, end
					};
					CPolyline polyline;
					polyline.addLinePoints(core::SRange<float64_t2>(points.data(), points.data() + points.size()));
					polylines.push_back(polyline);
				};
				{
					CPolyline polyline;
					std::vector<QuadraticBezierInfo> beziers;

					beziers.push_back({ float64_t2(-100, -100), float64_t2(-20, 40), float64_t2(0, -40), });
					line(float64_t2(-100, -100), float64_t2(0.0, -40));
					beziers.push_back({ float64_t2(-10, -50), float64_t2(10, -10), float64_t2(30, -50), });
					line(float64_t2(-10, -50), float64_t2(30, -50));
					beziers.push_back({ float64_t2(-50, 20), float64_t2(0, -80), float64_t2(50, 20), });
					line(float64_t2(-50, 20), float64_t2(50, 20));

					line(float64_t2(-30, -100), float64_t2(-30, -50));
					line(float64_t2(100, -100), float64_t2(100, -50));

					polyline.addQuadBeziers(nbl::core::SRange<QuadraticBezierInfo>(beziers.data(), beziers.data() + beziers.size()));

					polylines.push_back(polyline);
				}

				for (auto polyline = polylines.begin(); polyline != polylines.end(); polyline++)
				{
					CPULineStyle style2 = {};
					style2.screenSpaceLineWidth = 1.0f;
					style2.worldSpaceLineWidth = 0.0f;
					style2.color = float32_t4(0.0, 0.0, 0.0, 1.0);
					debug(*polyline, style2);
				}

				Hatch hatch(core::SRange<CPolyline>(polylines.data(), polylines.data() + polylines.size()), SelectedMajorAxis, hatchDebugStep, debug);
				intendedNextSubmit = currentDrawBuffers.drawHatch(hatch, float32_t4(0.6, 0.6, 1.0, 1.0f), UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
			}

			//if (hatchDebugStep > 0)
			//{
			//	std::vector <CPolyline> polylines;
			//	{
			//		std::vector<float64_t2> points = {
			//			float64_t2(119.196, -152.568),
			//			float64_t2(121.566, -87.564),
			//			float64_t2(237.850, -85.817),
			//			float64_t2(236.852, -152.194),
			//			float64_t2(206.159, -150.447),
			//			float64_t2(205.785, -125.618),
			//			float64_t2(205.785, -125.618),
			//			float64_t2(196.180, -122.051),
			//			float64_t2(186.820, -124.870),
			//			float64_t2(185.733, -136.350),
			//			float64_t2(185.822, -149.075),
			//			float64_t2(172.488, -155.349),
			//			float64_t2(159.621, -150.447),
			//			float64_t2(159.638, -137.831),
			//			float64_t2(159.246, -125.618),
			//			float64_t2(149.309, -121.398),
			//			float64_t2(139.907, -123.872),
			//			float64_t2(140.281, -149.075),
			//			float64_t2(140.281, -149.075),
			//			float64_t2(119.196, -152.568)
			//		};
			//		CPolyline polyline;
			//		polyline.addLinePoints(core::SRange<float64_t2>(points.data(), points.data() + points.size()));
			//		polylines.push_back(polyline);
			//	}
			//	{
			//		std::vector<float64_t2> points = {
			//			float64_t2(110.846, -97.918),
			//			float64_t2(113.217, -32.914),
			//			float64_t2(229.501, -31.167),
			//			float64_t2(228.503, -97.544),
			//			float64_t2(197.810, -95.797),
			//			float64_t2(197.435, -70.968),
			//			float64_t2(197.435, -70.968),
			//			float64_t2(187.831, -67.401),
			//			float64_t2(178.471, -70.220),
			//			float64_t2(177.384, -81.700),
			//			float64_t2(177.473, -94.425),
			//			float64_t2(164.138, -100.699),
			//			float64_t2(151.271, -95.797),
			//			float64_t2(151.289, -83.181),
			//			float64_t2(150.897, -70.968),
			//			float64_t2(140.960, -66.748),
			//			float64_t2(131.558, -69.222),
			//			float64_t2(131.932, -94.425),
			//			float64_t2(131.932, -94.425),
			//			float64_t2(110.846, -97.918)
			//		};
			//		CPolyline polyline;
			//		polyline.addLinePoints(core::SRange<float64_t2>(points.data(), points.data() + points.size()));
			//		polylines.push_back(polyline);
			//	}
			//	{
			//		std::vector<float64_t2> points = {
			//			float64_t2(50.504, -128.469),
			//			float64_t2(52.874, -63.465),
			//			float64_t2(169.158, -61.718),
			//			float64_t2(168.160, -128.095),
			//			float64_t2(137.467, -126.348),
			//			float64_t2(137.093, -101.519),
			//			float64_t2(137.093, -101.519),
			//			float64_t2(127.488, -97.952),
			//			float64_t2(118.128, -100.771),
			//			float64_t2(117.041, -112.251),
			//			float64_t2(117.130, -124.976),
			//			float64_t2(103.796, -131.250),
			//			float64_t2(90.929, -126.348),
			//			float64_t2(90.946, -113.732),
			//			float64_t2(90.554, -101.519),
			//			float64_t2(80.617, -97.298),
			//			float64_t2(71.215, -99.772),
			//			float64_t2(71.589, -124.976),
			//			float64_t2(71.589, -124.976),
			//			float64_t2(50.504, -128.469)
			//		};
			//		CPolyline polyline;
			//		polyline.addLinePoints(core::SRange<float64_t2>(points.data(), points.data() + points.size()));
			//		polylines.push_back(polyline);
			//	}
			//	{
			//		std::vector<float64_t2> points = {
			//			float64_t2(98.133, -111.581),
			//			float64_t2(100.503, -46.577),
			//			float64_t2(216.787, -44.830),
			//			float64_t2(215.789, -111.206),
			//			float64_t2(185.096, -109.460),
			//			float64_t2(184.722, -84.631),
			//			float64_t2(184.722, -84.631),
			//			float64_t2(175.117, -81.064),
			//			float64_t2(165.757, -83.882),
			//			float64_t2(164.670, -95.363),
			//			float64_t2(164.759, -108.087),
			//			float64_t2(151.425, -114.361),
			//			float64_t2(138.558, -109.460),
			//			float64_t2(138.575, -96.843),
			//			float64_t2(138.183, -84.631),
			//			float64_t2(128.246, -80.410),
			//			float64_t2(118.844, -82.884),
			//			float64_t2(119.218, -108.087),
			//			float64_t2(119.218, -108.087),
			//			float64_t2(98.133, -111.581)
			//		};
			//		CPolyline polyline;
			//		polyline.addLinePoints(core::SRange<float64_t2>(points.data(), points.data() + points.size()));
			//		polylines.push_back(polyline);
			//	}
			//	Hatch hatch(core::SRange<CPolyline>(polylines.data(), polylines.data() + polylines.size()), SelectedMajorAxis, hatchDebugStep, debug);
			//	intendedNextSubmit = currentDrawBuffers.drawHatch(hatch, float32_t4(0.0, 0.0, 1.0, 1.0f), UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
			//}
			//
			//if (hatchDebugStep > 0)
			//{
			//	std::vector<float64_t2> points;
			//	double sqrt3 = sqrt(3.0);
			//	points.push_back(float64_t2(0, 1));
			//	points.push_back(float64_t2(sqrt3 / 2, 0.5));
			//	points.push_back(float64_t2(sqrt3 / 2, -0.5));
			//	points.push_back(float64_t2(0, -1));
			//	points.push_back(float64_t2(-sqrt3 / 2, -0.5));
			//	points.push_back(float64_t2(-sqrt3 / 2, 0.5));
			//	points.push_back(float64_t2(0, 1));
			//
			//	std::vector<QuadraticBezierInfo> beziers;
			//	beziers.push_back({
			//		float64_t2(-0.5, -0.25),
			//		float64_t2(-sqrt3 / 2, 0.0),
			//		float64_t2(-0.5, 0.25) });
			//	beziers.push_back({
			//		float64_t2(0.5, -0.25),
			//		float64_t2(sqrt3 / 2, 0.0),
			//		float64_t2(0.5, 0.25) });
			//
			//	for (uint32_t i = 0; i < points.size(); i++)
			//		points[i] = float64_t2(-200.0, 0.0) + float64_t2(10.0 + abs(cos(m_timeElapsed * 0.00008)) * 150.0f, 100.0) * points[i];
			//	for (uint32_t i = 0; i < beziers.size(); i++)
			//		for (uint32_t j = 0; j < 3; j++)
			//			beziers[i].p[j] = float64_t2(-200.0, 0.0) + float64_t2(10.0 + abs(cos(m_timeElapsed * 0.00008)) * 150.0f, 100.0) * beziers[i].p[j];
			//
			//	CPolyline polyline;
			//	polyline.addLinePoints(core::SRange<float64_t2>(points.data(), points.data() + points.size()));
			//	polyline.addQuadBeziers(core::SRange<QuadraticBezierInfo>(beziers.data(), beziers.data() + beziers.size()));
			//
			//	core::SRange<CPolyline> polylines = core::SRange<CPolyline>(&polyline, &polyline + 1);
			//	Hatch hatch(polylines, SelectedMajorAxis, hatchDebugStep, debug);
			//	intendedNextSubmit = currentDrawBuffers.drawHatch(hatch, float32_t4(1.0f, 0.325f, 0.103f, 1.0f), UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
			//}
			//
			//if (hatchDebugStep > 0)
			//{
			//	CPolyline polyline;
			//	std::vector<QuadraticBezierInfo> beziers;
			//	beziers.push_back({
			//		100.0 * float64_t2(-0.4, 0.13),
			//		100.0 * float64_t2(7.7, 3.57),
			//		100.0 * float64_t2(8.8, 7.27) });
			//	beziers.push_back({
			//		100.0 * float64_t2(6.6, 0.13),
			//		100.0 * float64_t2(-1.97, 3.2),
			//		100.0 * float64_t2(3.7, 7.27) });
			//	polyline.addQuadBeziers(core::SRange<QuadraticBezierInfo>(beziers.data(), beziers.data() + beziers.size()));
			//
			//	core::SRange<CPolyline> polylines = core::SRange<CPolyline>(&polyline, &polyline + 1);
			//	Hatch hatch(polylines, SelectedMajorAxis, hatchDebugStep, debug);
			//	intendedNextSubmit = currentDrawBuffers.drawHatch(hatch, float32_t4(0.619f, 0.325f, 0.709f, 0.9f), UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
			//}
		}
		else if (mode == ExampleMode::CASE_3)
		{
			CPULineStyle style = {};
			style.screenSpaceLineWidth = 4.0f;
			style.worldSpaceLineWidth = 0.0f;
			style.color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

			CPULineStyle style2 = {};
			style2.screenSpaceLineWidth = 5.0f;
			style2.worldSpaceLineWidth = 0.0f;
			style2.color = float32_t4(0.2f, 0.6f, 0.2f, 0.5f);


			CPolyline polyline;
			CPolyline polyline2;
			
			{

				float Left = -100;
				float Right = 100;
				float Base = -25;
				srand(95);
				std::vector<QuadraticBezierInfo> quadBeziers;
				for (int i = 0; i < 1; i++) {
					QuadraticBezierInfo quadratic1;
					quadratic1.p[0] = float64_t2((rand() % 200 - 100), (rand() % 200 - 100));
					quadratic1.p[1] = float64_t2(0 + (rand() % 200 - 100), (rand() % 200 - 100));
					quadratic1.p[2] = float64_t2((rand() % 200 - 100), (rand() % 200 - 100));
					quadBeziers.push_back(quadratic1);
				}
				
				//{
				//	QuadraticBezierInfo quadratic1;
				//	quadratic1.p[0] = float64_t2(50,0);
				//	quadratic1.p[1] = float64_t2(50,100);
				//	quadratic1.p[2] = float64_t2(100,100);
				//	quadBeziers.push_back(quadratic1);
				//}
				//{
				//	QuadraticBezierInfo quadratic1;
				//	quadratic1.p[0] = float64_t2(100, 100);
				//	quadratic1.p[1] = float64_t2(200, -200);
				//	quadratic1.p[2] = float64_t2(300, 300);
				//	quadBeziers.push_back(quadratic1);
				//}
				polyline.addQuadBeziers(core::SRange<QuadraticBezierInfo>(quadBeziers.data(), quadBeziers.data() + quadBeziers.size()));
			}
			{
				std::vector<QuadraticBezierInfo> quadBeziers;
				{
					QuadraticBezierInfo quadratic1;
					quadratic1.p[0] = float64_t2(0.0, 0.0);
					quadratic1.p[1] = float64_t2(20.0, 50.0);
					quadratic1.p[2] = float64_t2(80.0, 0.0);
					//quadBeziers.push_back(quadratic1);
				}
				{
					QuadraticBezierInfo quadratic1;
					quadratic1.p[0] = float64_t2(80.0, 0.0);
					quadratic1.p[1] = float64_t2(220.0, 50.0);
					quadratic1.p[2] = float64_t2(180.0, 200.0);
					//quadBeziers.push_back(quadratic1);
				}
				{
					QuadraticBezierInfo quadratic1;
					quadratic1.p[0] = float64_t2(180.0, 200.0);
					quadratic1.p[1] = float64_t2(-20.0, 100.0);
					quadratic1.p[2] = float64_t2(30.0, -50.0);
					//quadBeziers.push_back(quadratic1);
				}

				// curves::ExplicitEllipse myCurve = curves::ExplicitEllipse(20.0, 50.0);
				// curves::ExplicitMixedCircle myCurve = curves::ExplicitMixedCircle::fromFourPoints(float64_t2(-25, 10.0), float64_t2(-20, 0.0), float64_t2(20.0, 0.0), float64_t2(0.0, -20.0));
				// curves::Parabola myCurve = curves::Parabola::fromThreePoints(float64_t2(-6.0, 4.0), float64_t2(0.0, 0.0), float64_t2(5.0, 0.0));
				// curves::MixedParabola myCurve = curves::MixedParabola::fromFourPoints(float64_t2(-60.0, 90.0), float64_t2(0.0, 0.0), float64_t2(50.0, 0.0), float64_t2(60.0,-20.0));
				//curves::CubicCurve myCurve = curves::CubicCurve(float64_t4(-10.0, 15.0, 5.0, 0.0), float64_t4(-8.0, 10.0, -5.0, 0.0));
				curves::EllipticalArcInfo myCurve;
				myCurve.majorAxis = {50.0, 50.0};
				myCurve.center = { 50.0, 50.0 };
				myCurve.angleBounds = { 
					nbl::core::PI<double>() * 1.25,
					nbl::core::PI<double>() * 1.25 + abs(cos(m_timeElapsed*0.001)) * nbl::core::PI<double>() * 2.0 };
				myCurve.eccentricity = 0.5;

				// curves::CircularArc arc1 = curves::CircularArc(float64_t2(-6, 50));
				// curves::CircularArc arc2 = curves::CircularArc(float64_t2(-6, -1));
				// curves::MixedParametricCurves myCurve = curves::MixedParametricCurves(&arc1, &arc2);

				curves::Subdivision::AddBezierFunc addToBezier = [&](QuadraticBezierInfo&& info) -> void
					{
						quadBeziers.push_back(info);
					};

				static int ix = 0;
				ix++;
				const int pp = (ix / 30) % 10;
				double error = pow(10.0, -1.0 * double(pp + 1));

				// TODO[Przemek]: this is how you use the adaptive subdivision algorithm, which construct beziers that estimate the original shape. you can use the tests commented above, all vars name "myCurve"
				curves::Subdivision::adaptive(myCurve, 1e-5, addToBezier, 10u);

				polyline2.addQuadBeziers(core::SRange<QuadraticBezierInfo>(quadBeziers.data(), quadBeziers.data() + quadBeziers.size()));

				// VISUALIZE INFLECTION POINT
				std::vector<float64_t2> linePoints;
				// auto inflectionPointT = myCurve.computeInflectionPoint(1e-5);
				// auto inflectionPoint = myCurve.computePosition(inflectionPointT);
				linePoints.push_back({ 0.0, 0.0 });
				linePoints.push_back({ 100.0, 100.0 });
				polyline2.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
			}

			//intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline, style, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
			intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline2, style2, UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);

			ClipProjectionData customClipProject = {};
			customClipProject.projectionToNDC = m_Camera.constructViewProjection();
			customClipProject.projectionToNDC[0][0] *= 1.003f;
			customClipProject.projectionToNDC[1][1] *= 1.003f;
			customClipProject.maxClipNDC = float32_t2(0.5, 0.5);
			customClipProject.minClipNDC = float32_t2(-0.5, -0.5);
			uint32_t clipProjIdx = InvalidClipProjectionIdx;
			// intendedNextSubmit = currentDrawBuffers.addClipProjectionData_SubmitIfNeeded(customClipProject, clipProjIdx, submissionQueue, submissionFence, intendedNextSubmit);
			// intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline, style2, clipProjIdx, submissionQueue, submissionFence, intendedNextSubmit);
		}
		else if (mode == ExampleMode::CASE_4)
		{
			constexpr uint32_t CURVE_CNT = 16u;
			constexpr uint32_t SPECIAL_CASE_CNT = 6u;

			CPULineStyle cpuLineStyle;
			cpuLineStyle.screenSpaceLineWidth = 7.0f;
			cpuLineStyle.worldSpaceLineWidth = 0.0f;
			cpuLineStyle.color = float32_t4(0.0f, 0.3f, 0.0f, 0.5f);

			std::vector<CPULineStyle> cpuLineStyles(CURVE_CNT, cpuLineStyle);
			std::vector<CPolyline> polylines(CURVE_CNT);

			{
				std::vector<QuadraticBezierInfo> quadratics(CURVE_CNT);

				// setting controll points
				{
					float64_t2 P0(-90, 68);
					float64_t2 P1(-41, 118);
					float64_t2 P2(88, 19);

					const float64_t2 translationVector(0, -5);

					uint32_t curveIdx = 0;
					while(curveIdx < CURVE_CNT - SPECIAL_CASE_CNT)
					{
						quadratics[curveIdx].p[0] = P0;
						quadratics[curveIdx].p[1] = P1;
						quadratics[curveIdx].p[2] = P2;

						P0 += translationVector;
						P1 += translationVector;
						P2 += translationVector;

						curveIdx++;
					}

					// special case 0 (line, evenly spaced points)
					const double prevLineLowestY = quadratics[curveIdx - 1].p[2].y;
					double lineY = prevLineLowestY - 10.0;

					quadratics[curveIdx].p[0] = float64_t2(-100, lineY);
					quadratics[curveIdx].p[1] = float64_t2(0, lineY);
					quadratics[curveIdx].p[2] = float64_t2(100, lineY);
					cpuLineStyles[curveIdx].color = float64_t4(0.7f, 0.3f, 0.1f, 0.5f);

					// special case 1 (line, not evenly spaced points)
					lineY -= 10.0;
					curveIdx++;

					quadratics[curveIdx].p[0] = float64_t2(-100, lineY);
					quadratics[curveIdx].p[1] = float64_t2(20, lineY);
					quadratics[curveIdx].p[2] = float64_t2(100, lineY);

					// special case 2 (folded line)
					lineY -= 10.0;
					curveIdx++;

					/*quadratics[curveIdx].p[0] = float64_t2(-100, lineY);
					quadratics[curveIdx].p[1] = float64_t2(200, lineY);
					quadratics[curveIdx].p[2] = float64_t2(100, lineY);*/

					quadratics[curveIdx].p[0] = float64_t2(-100, lineY);
					quadratics[curveIdx].p[1] = float64_t2(100, lineY);
					quadratics[curveIdx].p[2] = float64_t2(50, lineY);

					// oblique line
					curveIdx++;
					quadratics[curveIdx].p[0] = float64_t2(-100, 100);
					quadratics[curveIdx].p[1] = float64_t2(50.0, -50.0);
					quadratics[curveIdx].p[2] = float64_t2(100, -100);

					// special case 3 (A.x == 0)
					curveIdx++;
					quadratics[curveIdx].p[0] = float64_t2(0.0, 0.0);
					quadratics[curveIdx].p[1] = float64_t2(3.0, 4.14);
					quadratics[curveIdx].p[2] = float64_t2(6.0, 4.0);
					cpuLineStyles[curveIdx].color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);

						// make sure A.x == 0
					float64_t2 A = quadratics[curveIdx].p[0] - 2.0 * quadratics[curveIdx].p[1] + quadratics[curveIdx].p[2];
					assert(A.x == 0);

					// special case 4 (symetric parabola)
					curveIdx++;
					quadratics[curveIdx].p[0] = float64_t2(-150.0, 1.0);
					quadratics[curveIdx].p[1] = float64_t2(2000.0, 0.0);
					quadratics[curveIdx].p[2] = float64_t2(-150.0, -1.0);
					cpuLineStyles[curveIdx].color = float32_t4(0.7f, 0.3f, 0.1f, 0.5f);
				}

				std::array<core::vector<float>, CURVE_CNT> stipplePatterns;

				// TODO: fix uninvited circles at beggining and end of curves, solve with clipping (precalc tMin, tMax)

					// test case 0: test curve
				stipplePatterns[0] = { 0.0f, -5.0f, 2.0f, -5.0f };
					// test case 1: lots of redundant values, should look exactly like stipplePattern[0]
				stipplePatterns[1] = { 1.0f, 2.0f, 2.0f, -4.0f, -1.0f, 1.0f, -3.0f, -1.5f, -0.3f, -0.2f }; 
					// test case 2:stipplePattern[0] but shifted curve but shifted to left by 2.5f
				stipplePatterns[2] = { 2.5f, -5.0f, 1.0f, -5.0f, 2.5f };
					// test case 3: starts and ends with negative value, stipplePattern[2] reversed (I'm suspisious about that, need more testing)
				stipplePatterns[3] = { -2.5f, 5.0f, -1.0f, 5.0f, -2.5f };
					// test case 4: starts with "don't draw section"
				stipplePatterns[4] = { -5.0f, 5.0f };
					// test case 5: invisible curve (shouldn't be send to GPU)
				stipplePatterns[5] = { -1.0f };
					// test case 6: invisible curve (shouldn't be send to GPU)
				stipplePatterns[6] = { -1.0f, -5.0f, -10.0f };
					// test case 7: continous curuve
				stipplePatterns[7] = { 25.0f, 25.0f };
					// test case 8: empty pattern (draw line with no pattern)
				stipplePatterns[8] = {};
					// test case 9: max pattern size
				stipplePatterns[9] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -2.0f };
					// test case 10: A = 0 (line), evenly distributed controll points (doesn't work)
				stipplePatterns[10] = { 5.0f, -5.0f, 1.0f, -5.0f };
					// test case 11: A = 0 (line), not evenly distributed controll points
				stipplePatterns[11] = { 5.0f, -5.0f, 1.0f, -5.0f };
					// test case 12: A = 0 (line), folds itself
				stipplePatterns[12] = { 5.0f, -5.0f, 1.0f, -5.0f };
					// test case 13: oblique line 
				stipplePatterns[13] = { 5.0f, -5.0f, 1.0f, -5.0f };
					// test case 14: curve with A.x = 0
				stipplePatterns[14] = { 0.0f, -0.5f, 0.2f, -0.5f };
					// test case 15: long parabola
				stipplePatterns[15] = { 5.0f, -5.0f, 1.0f, -5.0f };

				std::vector<uint32_t> activIdx = { 10 };
				for (uint32_t i = 0u; i < CURVE_CNT; i++)
				{
					cpuLineStyles[i].setStipplePatternData(nbl::core::SRange<float>(stipplePatterns[i].begin()._Ptr, stipplePatterns[i].end()._Ptr));
					polylines[i].addQuadBeziers(core::SRange<QuadraticBezierInfo>(&quadratics[i], &quadratics[i] + 1u));

					float64_t2 linePoints[2u] = {};
					linePoints[0] = { -200.0, 50.0 - 5.0 * i };
					linePoints[1] = { -100.0, 50.0 - 6.0 * i };
					polylines[i].addLinePoints(core::SRange<float64_t2>(linePoints, linePoints + 2));

					activIdx.push_back(i);
					if (std::find(activIdx.begin(), activIdx.end(), i) == activIdx.end())
						cpuLineStyles[i].stipplePatternSize = -1;
				}
			}

			for (uint32_t i = 0u; i < CURVE_CNT; i++)
				intendedNextSubmit = currentDrawBuffers.drawPolyline(polylines[i], cpuLineStyles[i], UseDefaultClipProjectionIdx, submissionQueue, submissionFence, intendedNextSubmit);
		}

		intendedNextSubmit = currentDrawBuffers.finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo submitInBetweenDraws(uint32_t resourceIdx, video::IGPUQueue* submissionQueue, video::IGPUFence* submissionFence, video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Use the last command buffer in intendedNextSubmit, it should be in recording state
		auto& cmdbuf = intendedNextSubmit.commandBuffers[intendedNextSubmit.commandBufferCount - 1];

		auto& currentDrawBuffers = drawBuffers[resourceIdx];

		uint32_t windowWidth = swapchain->getCreationParameters().width;
		uint32_t windowHeight = swapchain->getCreationParameters().height;

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { windowWidth, windowHeight };

			beginInfo.clearValueCount = 0u;
			beginInfo.framebuffer = framebuffersDynArraySmartPtr->begin()[m_SwapchainImageIx];
			beginInfo.renderpass = renderpassInBetween;
			beginInfo.renderArea = area;
			beginInfo.clearValues = nullptr;
		}

		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = windowWidth;
		vp.height = windowHeight;
		cmdbuf->setViewport(0u, 1u, &vp);

		VkRect2D scissor;
		scissor.extent = { windowWidth, windowHeight };
		scissor.offset = { 0, 0 };
		cmdbuf->setScissor(0u, 1u, &scissor);

		pipelineBarriersBeforeDraw(cmdbuf);

		cmdbuf->beginRenderPass(&beginInfo, asset::ESC_INLINE);

		const uint32_t currentIndexCount = drawBuffers[resourceIdx].getIndexCount();
		cmdbuf->bindDescriptorSets(asset::EPBP_GRAPHICS, graphicsPipelineLayout.get(), 0u, 1u, &descriptorSets[resourceIdx].get());
		cmdbuf->bindIndexBuffer(drawBuffers[resourceIdx].gpuDrawBuffers.indexBuffer.get(), 0u, asset::EIT_32BIT);
		cmdbuf->bindGraphicsPipeline(graphicsPipeline.get());
		cmdbuf->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);

		if constexpr (DebugMode)
		{
			cmdbuf->bindGraphicsPipeline(debugGraphicsPipeline.get());
			cmdbuf->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);
		}
		
		cmdbuf->endRenderPass();

		cmdbuf->end();

		video::IGPUQueue::SSubmitInfo submit = intendedNextSubmit;
		submit.signalSemaphoreCount = 0u;
		submit.pSignalSemaphores = nullptr;
		assert(submit.isValid());
		submissionQueue->submit(1u, &submit, submissionFence);
		intendedNextSubmit.commandBufferCount = 1u;
		intendedNextSubmit.commandBuffers = &cmdbuf;
		intendedNextSubmit.waitSemaphoreCount = 0u;
		intendedNextSubmit.pWaitSemaphores = nullptr;
		intendedNextSubmit.pWaitDstStageMask = nullptr;
		// we can reset the fence and commandbuffer because we fully wait for the GPU to finish here
		logicalDevice->blockForFences(1u, &submissionFence);
		logicalDevice->resetFences(1u, &submissionFence);
		cmdbuf->reset(video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
		cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

		// reset things
		// currentDrawBuffers.clear();

		return intendedNextSubmit;
	}

	double dt = 0;
	double m_timeElapsed = 0.0;
	std::chrono::steady_clock::time_point lastTime;

	void workLoopBody() override
	{
		m_resourceIx++;
		if (m_resourceIx >= FRAMES_IN_FLIGHT)
			m_resourceIx = 0;

		auto now = std::chrono::high_resolution_clock::now();
		dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
		lastTime = now;
		m_timeElapsed += dt;

		if constexpr (mode == ExampleMode::CASE_0)
		{
			m_Camera.setSize(20.0 + abs(cos(m_timeElapsed * 0.001)) * 600);
		}

		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
			{
				m_Camera.mouseProcess(events);
			}
		, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
			{
				m_Camera.keyboardProcess(events);

				for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
				{
					auto ev = *eventIt;

					if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_E)
					{
						m_hatchDebugStep++;
					}
					if (ev.action == nbl::ui::SKeyboardEvent::E_KEY_ACTION::ECA_PRESSED && ev.keyCode == nbl::ui::E_KEY_CODE::EKC_Q)
					{
						m_hatchDebugStep--;
					}
				}
			}
		, logger.get());

		auto& cb = m_cmdbuf[m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];

		auto& graphicsQueue = queues[CommonAPI::InitOutput::EQT_GRAPHICS];

		nbl::video::IGPUQueue::SSubmitInfo submit;
		submit.commandBufferCount = 1u;
		submit.commandBuffers = &cb.get();
		submit.signalSemaphoreCount = 1u;
		submit.pSignalSemaphores = &m_renderFinished[m_resourceIx].get();
		nbl::video::IGPUSemaphore* waitSemaphores[1u] = { m_imageAcquire[m_resourceIx].get() };
		asset::E_PIPELINE_STAGE_FLAGS waitStages[1u] = { nbl::asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT };
		submit.waitSemaphoreCount = 1u;
		submit.pWaitSemaphores = waitSemaphores;
		submit.pWaitDstStageMask = waitStages;

		beginFrameRender();

		submit = addObjects(graphicsQueue, fence.get(), submit);

		endFrameRender();

		graphicsQueue->submit(1u, &submit, fence.get());

		CommonAPI::Present(
			logicalDevice.get(),
			swapchain.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			m_renderFinished[m_resourceIx].get(),
			m_SwapchainImageIx);

		getAndLogQueryPoolResults();
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}
};

//NBL_COMMON_API_MAIN(CADApp)
int main(int argc, char** argv) {
	CommonAPI::main<CADApp>(argc, argv);
}