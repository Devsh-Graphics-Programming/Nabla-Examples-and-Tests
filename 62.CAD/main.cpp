#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"


static constexpr bool DebugMode = false;
static constexpr bool FragmentShaderPixelInterlock = true;

enum class ExampleMode
{
	CASE_0, // Zooming In/Out
	CASE_1, // Rotating Line
	CASE_2, // Straight Line Moving up and down
	CASE_3, // Ellipses
};

constexpr ExampleMode mode = ExampleMode::CASE_3;


struct double4x4
{
	double _r0[4u];
	double _r1[4u];
	double _r2[4u];
	double _r3[4u];
};

struct float4
{
	float4() {}

	float4(const float x, const float y, const float z, const float w)
	{
		val[0u] = x;
		val[1u] = y;
		val[2u] = z;
		val[3u] = w;
	}

	float val[4u];

	inline bool operator ==(const float4& other) const
	{
		return val[0u] == other.val[0u];
	}
};

typedef nbl::core::vector2d<double> double2;
typedef nbl::core::vector2d<uint32_t> uint2;

#include "common.hlsl"

static_assert(sizeof(DrawObject) == 16u);
static_assert(sizeof(PackedEllipseInfo) == 48u);
static_assert(sizeof(Globals) == 152u);
static_assert(sizeof(LineStyle) == 32u);

using namespace nbl;
using namespace ui;

// TODO: Use a math lib?
double dot(const double2& a, const double2& b)
{
	return a.X * b.X + a.Y * b.Y;
}
double2 normalize(const double2& x)
{
	double len = dot(x,x);
#ifdef __NBL_FAST_MATH
	return x * core::inversesqrt<double>(len);
#else
	return x / core::sqrt<double>(len);
#endif
}

class Camera2D : public core::IReferenceCounted
{
public:
	Camera2D()
	{}

	void setOrigin(const double2& origin)
	{
		m_origin = origin;
	}

	void setAspectRatio(const double& aspectRatio)
	{
		m_aspectRatio = aspectRatio;
	}

	void setSize(const double size)
	{
		m_bounds = double2{ size * m_aspectRatio, size };
	}

	double2 getBounds() const
	{
		return m_bounds;
	}

	double4x4 constructViewProjection()
	{
		double4x4 ret = {};

		ret._r0[0] = 2.0 / m_bounds.X;
		ret._r1[1] = -2.0 / m_bounds.Y;
		ret._r2[2] = 1.0;

		ret._r2[0] = (-2.0 * m_origin.X) / m_bounds.X;
		ret._r2[1] = (2.0 * m_origin.Y) / m_bounds.Y;

		return ret;
	}

	void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
	{
		for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
		{
			auto ev = *eventIt;

			if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL)
			{
				m_bounds = m_bounds + double2{ (double)ev.scrollEvent.verticalScroll * -0.1 * m_aspectRatio, (double)ev.scrollEvent.verticalScroll * -0.1};
				m_bounds = double2 {core::max(m_aspectRatio, m_bounds.X), core::max(1.0, m_bounds.Y)};
			}
		}
	}

private:

	double m_aspectRatio = 0.0;
	double2 m_bounds = {};
	double2 m_origin = {};
};

constexpr double maxEllipticalArcAngle = nbl::core::PI<double>() / 2.0f;
static_assert(maxEllipticalArcAngle <= nbl::core::PI<double>()); // !important cause our shaders will be messed up with arcs > 180 degrees

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

	struct EllipticalArcInfo
	{
		double2 majorAxis;
		double2 center;
		double2 angleBounds; // [0, 2Pi)
		double eccentricity; // (0, 1]

		bool isValid() const
		{
			if (eccentricity > 1.0 || eccentricity < 0.0)
				return false;
			if (angleBounds.Y < angleBounds.X)
				return false;
			if ((angleBounds.Y - angleBounds.X) > 2 * core::PI<double>())
				return false;
			return true;
		}
	};

	size_t getSectionsCount() const { return m_sections.size(); }

	const SectionInfo& getSectionInfoAt(const uint32_t idx) const
	{
		return m_sections[idx];
	}

	const PackedEllipseInfo& getEllipseInfoAt(const uint32_t idx) const
	{
		return m_ellipses[idx];
	}

	const QuadraticBezierInfo& getQuadBezierInfoAt(const uint32_t idx) const
	{
		return m_quadBeziers[idx];
	}

	const CubicBezierInfo& getCubicBezierInfoAt(const uint32_t idx) const
	{
		return m_cubicBeziers[idx];
	}

	const double2& getLinePointAt(const uint32_t idx) const
	{
		return m_linePoints[idx];
	}

	void clearEverything()
	{
		m_sections.clear();
		m_linePoints.clear();
		m_ellipses.clear();
	}

	// Reserves memory with worst case
	void reserveMemory(uint32_t noOfLines, uint32_t noOfEllipses)
	{
		m_sections.reserve(noOfLines + noOfEllipses); // worst case
		m_linePoints.reserve(noOfLines * 2u); // worst case
		const uint32_t maxEllipseParts = std::ceil((2.0 * nbl::core::PI<double>()) / maxEllipticalArcAngle);
		m_ellipses.reserve(noOfEllipses * maxEllipseParts); // "* maxEllipseParts" because we will chop up the arcs
	}

	void addLinePoints(std::vector<double2>&& linePoints)
	{
		if (linePoints.size() <= 1u)
			return;

		bool addNewSection = m_sections.size() == 0u || m_sections[m_sections.size() - 1u].type != ObjectType::LINE;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::LINE;
			newSection.index = m_linePoints.size();
			newSection.count = linePoints.size() - 1u;
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += linePoints.size();
		}
		m_linePoints.insert(m_linePoints.end(), linePoints.begin(), linePoints.end());
	}

	void addEllipticalArcs(std::vector<EllipticalArcInfo>&& ellipses)
	{
		std::vector<PackedEllipseInfo> packedEllipses;
		// We will chop up the ellipses
		const uint32_t maxEllipseParts = std::ceil((2.0 * nbl::core::PI<double>()) / maxEllipticalArcAngle);
		for (uint32_t i = 0u; i < ellipses.size(); ++i)
		{
			const auto& mainEllipse = ellipses[i];
			assert(mainEllipse.isValid());
			const double endAngle = mainEllipse.angleBounds.Y;
			double startAngle = mainEllipse.angleBounds.X;
			for (uint32_t e = 0u; e < maxEllipseParts && startAngle < endAngle; e++)
			{
				double maxNextAngle = startAngle + maxEllipticalArcAngle;
				double nextEndAngle = core::min(maxNextAngle, mainEllipse.angleBounds.Y);

				constexpr double twoPi = core::PI<double>() * 2.0;
				PackedEllipseInfo packedInfo = {};
				packedInfo.center = mainEllipse.center;
				packedInfo.majorAxis = mainEllipse.majorAxis;
				packedInfo.angleBoundsPacked.X = static_cast<uint32_t>((startAngle / twoPi) * UINT32_MAX);
				packedInfo.angleBoundsPacked.Y = static_cast<uint32_t>((nextEndAngle / twoPi) * UINT32_MAX);
				packedInfo.eccentricityPacked = static_cast<uint32_t>(mainEllipse.eccentricity * UINT32_MAX);
				packedEllipses.push_back(packedInfo);

				startAngle = nextEndAngle;
			}
		}
		addEllipticalArcs_Internal(std::move(packedEllipses));
	}

	void addQuadBeziers(std::vector<QuadraticBezierInfo>&& quadBeziers)
	{
		bool addNewSection = m_sections.size() == 0u || m_sections[m_sections.size() - 1u].type != ObjectType::QUAD_BEZIER;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::QUAD_BEZIER;
			newSection.index = m_quadBeziers.size();
			newSection.count = quadBeziers.size();
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += quadBeziers.size();
		}
		m_quadBeziers.insert(m_quadBeziers.end(), quadBeziers.begin(), quadBeziers.end());
	}

	void addCubicBeziers(std::vector<CubicBezierInfo>&& cubicBeziers)
	{
		bool addNewSection = m_sections.size() == 0u || m_sections[m_sections.size() - 1u].type != ObjectType::CUBIC_BEZIER;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::CUBIC_BEZIER;
			newSection.index = m_cubicBeziers.size();
			newSection.count = cubicBeziers.size();
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += cubicBeziers.size();
		}
		m_cubicBeziers.insert(m_cubicBeziers.end(), cubicBeziers.begin(), cubicBeziers.end());
	}

protected:

	void addEllipticalArcs_Internal(std::vector<PackedEllipseInfo>&& ellipses)
	{
		bool addNewSection = m_sections.size() == 0u || m_sections[m_sections.size() - 1u].type != ObjectType::ELLIPSE;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::ELLIPSE;
			newSection.index = m_ellipses.size();
			newSection.count = ellipses.size();
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += ellipses.size();
		}
		m_ellipses.insert(m_ellipses.end(), ellipses.begin(), ellipses.end());
	}

	std::vector<SectionInfo> m_sections;
	std::vector<double2> m_linePoints;
	std::vector<PackedEllipseInfo> m_ellipses;
	std::vector<QuadraticBezierInfo> m_quadBeziers;
	std::vector<CubicBezierInfo> m_cubicBeziers;
};

template <typename BufferType>
struct DrawBuffers
{
	core::smart_refctd_ptr<BufferType> indexBuffer;
	core::smart_refctd_ptr<BufferType> drawObjectsBuffer;
	core::smart_refctd_ptr<BufferType> geometryBuffer;
	core::smart_refctd_ptr<BufferType> lineStylesBuffer;
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

	void allocateIndexBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, uint32_t indices)
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

	void allocateDrawObjectsBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, uint32_t drawObjects)
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

	void allocateGeometryBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, size_t size)
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

	void allocateStylesBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, uint32_t stylesCount)
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

	uint32_t getIndexCount() const { return currentIndexCount; }

	//! this function fills buffers required for drawing a polyline and submits a draw through provided callback when there is not enough memory.
	video::IGPUQueue::SSubmitInfo drawPolyline(
		const CPolyline& polyline,
		const LineStyle& lineStyle,
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		uint32_t styleIdx;
		intendedNextSubmit = addLineStyle_SubmitIfNeeded(lineStyle, styleIdx, submissionQueue, submissionFence, intendedNextSubmit);

		const auto sectionsCount = polyline.getSectionsCount();

		// We keep track of the last section and object, to know which geometries are still available in memory
		uint32_t startDrawObjectCount = currentDrawObjectCount;
		uint32_t previousSectionIdx = 0u;
		uint32_t previousObjectInSection = 0u;

		// Fill all back faces (and draw when overflow)
		// backface is our slang for even provoking vertex
		{
			uint32_t currentSectionIdx = 0u;
			uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject used in vertex shader. You can think of it as a Cage.

			while (currentSectionIdx < sectionsCount)
			{
				bool shouldSubmit = false;
				const auto& currentSection = polyline.getSectionInfoAt(currentSectionIdx);
				addObjects_Internal(polyline, currentSection, currentObjectInSection, styleIdx, false);
				
				if (currentObjectInSection >= currentSection.count)
				{
					currentSectionIdx++;
					currentObjectInSection = 0u;
				}
				else
					shouldSubmit = true;

				if (shouldSubmit)
				{
					intendedNextSubmit = finalizeLineStyleCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
					intendedNextSubmit = finalizeIndexCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
					intendedNextSubmit = finalizeGeometryCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
					intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
					resetIndexCounters();
					resetGeometryCounters();

					startDrawObjectCount = 0u;
					previousSectionIdx = currentSectionIdx;
					previousObjectInSection = currentObjectInSection;

					shouldSubmit = false;
				}
			}
		}

		// Fill all front faces (and draw when overflow)
		// frontface is our slang for odd provoking vertex
		{
			// all front faces only using index buffer for those object that are already in cpu memory (ready for upload)
			{
				uint32_t currentSectionIdx = previousSectionIdx;
				uint32_t currentObjectInSection = previousObjectInSection;

				while (currentSectionIdx < sectionsCount)
				{
					bool shouldSubmit = false;
					const auto& currentSection = polyline.getSectionInfoAt(currentSectionIdx);

					// we only care about indices because the geometry and drawData is already in memory
					const uint32_t uploadableObjects = (maxIndices - currentIndexCount) / 6u;
					const auto objectsRemaining = currentSection.count - currentObjectInSection;
					const auto objectsToUpload = core::min(uploadableObjects, objectsRemaining);

					addObjectIndices_Internal(true, startDrawObjectCount, objectsToUpload);

					currentObjectInSection += objectsToUpload;

					if (currentObjectInSection >= currentSection.count)
					{
						currentSectionIdx++;
						currentObjectInSection = 0u;
					}
					else
						shouldSubmit = true;

					startDrawObjectCount += objectsToUpload;

					if (shouldSubmit)
					{
						intendedNextSubmit = finalizeGeometryCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
						intendedNextSubmit = finalizeLineStyleCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
						intendedNextSubmit = finalizeIndexCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
						intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
						resetIndexCounters();
						// we don't reset the geometry counters cause we overflowed on index memory not geometry memory

						shouldSubmit = false;
					}
				}
			}
			// remaining front faces where their geometry is non-existent in memory due to previous submit clears
			{
				const uint32_t lastSectionIdx = previousSectionIdx;
				const uint32_t lastObjectInSection = previousObjectInSection;

				uint32_t currentSectionIdx = 0u;
				uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject used in vertex shader. You can think of it as a Cage.

				while (currentSectionIdx < lastSectionIdx || (currentSectionIdx == lastSectionIdx && lastObjectInSection > 0u))
				{
					bool shouldSubmit = false;
					auto currentSection = polyline.getSectionInfoAt(currentSectionIdx);
					if (currentSectionIdx == lastSectionIdx)
						currentSection.count = lastObjectInSection;

					addObjects_Internal(polyline, currentSection, currentObjectInSection, styleIdx, true);

					if (currentObjectInSection >= currentSection.count)
					{
						currentSectionIdx++;
						currentObjectInSection = 0u;
					}
					else
						shouldSubmit = true;

					if (shouldSubmit)
					{
						intendedNextSubmit = finalizeLineStyleCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
						intendedNextSubmit = finalizeIndexCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
						intendedNextSubmit = finalizeGeometryCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
						intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
						resetIndexCounters();
						resetGeometryCounters();
						shouldSubmit = false;
					}
				}
			}
		}

		return intendedNextSubmit;
	}

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

	video::IGPUQueue::SSubmitInfo finalizeGeometryCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		// Copy DrawBuffers
		uint32_t remainingDrawObjects = currentDrawObjectCount - inMemDrawObjectCount;
		asset::SBufferRange<video::IGPUBuffer> drawObjectsRange = { sizeof(DrawObject) * inMemDrawObjectCount, sizeof(DrawObject) * remainingDrawObjects, gpuDrawBuffers.drawObjectsBuffer };
		const DrawObject* srcDrawObjData = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + inMemDrawObjectCount;
		if (drawObjectsRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(drawObjectsRange, srcDrawObjData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemDrawObjectCount = currentDrawObjectCount;

		// Copy GeometryBuffer
		uint32_t remainingGeometrySize = currentGeometryBufferSize - inMemGeometryBufferSize;
		asset::SBufferRange<video::IGPUBuffer> geomRange = { inMemGeometryBufferSize, remainingGeometrySize, gpuDrawBuffers.geometryBuffer };
		const uint8_t* srcGeomData = reinterpret_cast<uint8_t*>(cpuDrawBuffers.geometryBuffer->getPointer()) + inMemGeometryBufferSize;
		if (geomRange.size > 0u)
			intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(geomRange, srcGeomData, submissionQueue, submissionFence, intendedNextSubmit);
		inMemGeometryBufferSize = currentGeometryBufferSize;

		return intendedNextSubmit;
	}

	video::IGPUQueue::SSubmitInfo finalizeAllCopiesToGPU(
		video::IGPUQueue* submissionQueue,
		video::IGPUFence* submissionFence,
		video::IGPUQueue::SSubmitInfo intendedNextSubmit)
	{
		intendedNextSubmit = finalizeIndexCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		intendedNextSubmit = finalizeGeometryCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		intendedNextSubmit = finalizeLineStyleCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);

		return intendedNextSubmit;
	}

	size_t getCurrentIndexBufferSize() const
	{
		return sizeof(index_buffer_type) * currentIndexCount;
	}

	size_t getCurrentLineStylesBufferSize() const
	{
		return sizeof(LineStyle) * currentLineStylesCount;
	}

	size_t getCurrentDrawObjectsBufferSize() const
	{
		return sizeof(DrawObject) * currentDrawObjectCount;
	}

	size_t getCurrentGeometryBufferSize() const
	{
		return currentGeometryBufferSize;
	}


	void reset()
	{
		resetAllCounters();
	}

	DrawBuffers<asset::ICPUBuffer> cpuDrawBuffers;
	DrawBuffers<video::IGPUBuffer> gpuDrawBuffers;

protected:

	SubmitFunc submitDraws;

	static constexpr uint32_t InvalidLineStyleIdx = ~0u;

	video::IGPUQueue::SSubmitInfo addLineStyle_SubmitIfNeeded(
		const LineStyle& lineStyle,
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

	uint32_t addLineStyle_Internal(const LineStyle& lineStyle)
	{
		LineStyle* stylesArray = reinterpret_cast<LineStyle*>(cpuDrawBuffers.lineStylesBuffer->getPointer());
		for (uint32_t i = 0u; i < currentLineStylesCount; ++i)
		{
			const LineStyle& itr = stylesArray[i];
			if (lineStyle.screenSpaceLineWidth == itr.screenSpaceLineWidth)
				if (lineStyle.worldSpaceLineWidth == itr.worldSpaceLineWidth)
					if (lineStyle.color == itr.color)
						return i;
		}

		if (currentLineStylesCount >= maxLineStyles)
			return InvalidLineStyleIdx;

		void* dst = stylesArray + currentLineStylesCount;
		memcpy(dst, &lineStyle, sizeof(LineStyle));
		return currentLineStylesCount++;
	}

	//@param oddProvokingVertex is used for our polyline-wide transparency algorithm where we draw the object twice, once to resolve the alpha and another time to draw them
	void addObjects_Internal(const CPolyline& polyline, const CPolyline::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t styleIdx, bool oddProvokingVertex)
	{
		if (section.type == ObjectType::LINE)
			addLines_Internal(polyline, section, currentObjectInSection, styleIdx, oddProvokingVertex);
		else if (section.type == ObjectType::ELLIPSE)
			addEllipses_Internal(polyline, section, currentObjectInSection, styleIdx, oddProvokingVertex);
		else if (section.type == ObjectType::QUAD_BEZIER)
			addQuadBeziers_Internal(polyline, section, currentObjectInSection, styleIdx, oddProvokingVertex);
		else if (section.type == ObjectType::CUBIC_BEZIER)
			addCubicBeziers_Internal(polyline, section, currentObjectInSection, styleIdx, oddProvokingVertex);
		else
			assert(false); // we don't handle other object types
	}

	//@param oddProvokingVertex is used for our polyline-wide transparency algorithm where we draw the object twice, once to resolve the alpha and another time to draw them
	void addLines_Internal(const CPolyline& polyline, const CPolyline::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t styleIdx, bool oddProvokingVertex)
	{
		assert(section.count >= 1u);
		assert(section.type == ObjectType::LINE);

		const auto maxGeometryBufferPoints = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(double2);
		const auto maxGeometryBufferLines = (maxGeometryBufferPoints <= 1u) ? 0u : maxGeometryBufferPoints - 1u;

		uint32_t uploadableObjects = (maxIndices - currentIndexCount) / 6u;
		uploadableObjects = core::min(uploadableObjects, maxGeometryBufferLines);
		uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

		const auto lineCount = section.count;
		const auto remainingObjects = lineCount - currentObjectInSection;
		uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

		// Add Indices
		addObjectIndices_Internal(oddProvokingVertex, currentDrawObjectCount, objectsToUpload);

		// Add DrawObjs
		DrawObject drawObj = {};
		drawObj.type = ObjectType::LINE;
		drawObj.address = geometryBufferAddress + currentGeometryBufferSize;
		drawObj.styleIdx = styleIdx;
		for (uint32_t i = 0u; i < objectsToUpload; ++i)
		{
			void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
			memcpy(dst, &drawObj, sizeof(DrawObject));
			currentDrawObjectCount += 1u;
			drawObj.address += sizeof(double2);
		}

		// Add Geometry
		if (objectsToUpload > 0u)
		{
			const auto pointsByteSize = sizeof(double2) * (objectsToUpload + 1u);
			void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
			auto& linePoint = polyline.getLinePointAt(section.index + currentObjectInSection);
			memcpy(dst, &linePoint, pointsByteSize);
			currentGeometryBufferSize += pointsByteSize;
		}

		currentObjectInSection += objectsToUpload;
	}

	//@param oddProvokingVertex is used for our polyline-wide transparency algorithm where we draw the object twice, once to resolve the alpha and another time to draw them
	void addEllipses_Internal(const CPolyline& polyline, const CPolyline::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t styleIdx, bool oddProvokingVertex)
	{
		assert(section.type == ObjectType::ELLIPSE);

		const auto maxGeometryBufferEllipses = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(PackedEllipseInfo);

		uint32_t uploadableObjects = (maxIndices - currentIndexCount) / 6u;
		uploadableObjects = core::min(uploadableObjects, maxGeometryBufferEllipses);
		uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

		const auto ellipseCount = section.count;
		const auto remainingObjects = ellipseCount - currentObjectInSection;
		uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

		// Add Indices
		addObjectIndices_Internal(oddProvokingVertex, currentDrawObjectCount, objectsToUpload);

		// Add DrawObjs
		DrawObject drawObj = {};
		drawObj.type = ObjectType::ELLIPSE;
		drawObj.address = geometryBufferAddress + currentGeometryBufferSize;
		drawObj.styleIdx = styleIdx;
		for (uint32_t i = 0u; i < objectsToUpload; ++i)
		{
			void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
			memcpy(dst, &drawObj, sizeof(DrawObject));
			currentDrawObjectCount += 1u;
			drawObj.address += sizeof(PackedEllipseInfo);
		}

		// Add Geometry
		if (objectsToUpload > 0u)
		{
			const auto ellipsesByteSize = sizeof(PackedEllipseInfo) * (objectsToUpload);
			void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
			auto& ellipse = polyline.getEllipseInfoAt(section.index + currentObjectInSection);
			memcpy(dst, &ellipse, ellipsesByteSize);
			currentGeometryBufferSize += ellipsesByteSize;
		}

		currentObjectInSection += objectsToUpload;
	}

	//@param oddProvokingVertex is used for our polyline-wide transparency algorithm where we draw the object twice, once to resolve the alpha and another time to draw them
	void addQuadBeziers_Internal(const CPolyline& polyline, const CPolyline::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t styleIdx, bool oddProvokingVertex)
	{
		assert(section.type == ObjectType::QUAD_BEZIER);

		const auto maxGeometryBufferEllipses = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(QuadraticBezierInfo);

		uint32_t uploadableObjects = (maxIndices - currentIndexCount) / 6u;
		uploadableObjects = core::min(uploadableObjects, maxGeometryBufferEllipses);
		uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

		const auto beziersCount = section.count;
		const auto remainingObjects = beziersCount - currentObjectInSection;
		uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

		// Add Indices
		addObjectIndices_Internal(oddProvokingVertex, currentDrawObjectCount, objectsToUpload);

		// Add DrawObjs
		DrawObject drawObj = {};
		drawObj.type = ObjectType::QUAD_BEZIER;
		drawObj.address = geometryBufferAddress + currentGeometryBufferSize;
		drawObj.styleIdx = styleIdx;
		for (uint32_t i = 0u; i < objectsToUpload; ++i)
		{
			void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
			memcpy(dst, &drawObj, sizeof(DrawObject));
			currentDrawObjectCount += 1u;
			drawObj.address += sizeof(QuadraticBezierInfo);
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

	//@param oddProvokingVertex is used for our polyline-wide transparency algorithm where we draw the object twice, once to resolve the alpha and another time to draw them
	void addCubicBeziers_Internal(const CPolyline& polyline, const CPolyline::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t styleIdx, bool oddProvokingVertex)
	{
		assert(section.type == ObjectType::CUBIC_BEZIER);

		const auto maxGeometryBufferEllipses = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(CubicBezierInfo);

		uint32_t uploadableObjects = (maxIndices - currentIndexCount) / 6u;
		uploadableObjects = core::min(uploadableObjects, maxGeometryBufferEllipses);
		uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

		const auto beziersCount = section.count;
		const auto remainingObjects = beziersCount - currentObjectInSection;
		uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

		// Add Indices
		addObjectIndices_Internal(oddProvokingVertex, currentDrawObjectCount, objectsToUpload);

		// Add DrawObjs
		DrawObject drawObj = {};
		drawObj.type = ObjectType::CUBIC_BEZIER;
		drawObj.address = geometryBufferAddress + currentGeometryBufferSize;
		drawObj.styleIdx = styleIdx;
		for (uint32_t i = 0u; i < objectsToUpload; ++i)
		{
			void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
			memcpy(dst, &drawObj, sizeof(DrawObject));
			currentDrawObjectCount += 1u;
			drawObj.address += sizeof(CubicBezierInfo);
		}

		// Add Geometry
		if (objectsToUpload > 0u)
		{
			const auto beziersByteSize = sizeof(CubicBezierInfo) * (objectsToUpload);
			void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
			auto& cubicBezier = polyline.getCubicBezierInfoAt(section.index + currentObjectInSection);
			memcpy(dst, &cubicBezier, beziersByteSize);
			currentGeometryBufferSize += beziersByteSize;
		}

		currentObjectInSection += objectsToUpload;
	}

	//@param oddProvokingVertex is used for our polyline-wide transparency algorithm where we draw the object twice, once to resolve the alpha and another time to draw them
	void addObjectIndices_Internal(bool oddProvokingVertex, uint32_t startObject, uint32_t objectCount)
	{
		index_buffer_type* indices = reinterpret_cast<index_buffer_type*>(cpuDrawBuffers.indexBuffer->getPointer()) + currentIndexCount;
		for (uint32_t i = 0u; i < objectCount; ++i)
		{
			index_buffer_type objIndex = startObject + i;
			if(oddProvokingVertex)
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
		resetGeometryCounters();
		resetIndexCounters();
		resetStyleCounters();
	}

	void resetGeometryCounters()
	{
		inMemDrawObjectCount = 0u;
		inMemGeometryBufferSize = 0u;
		currentDrawObjectCount = 0u;
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

	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> device;

	uint32_t inMemIndexCount = 0u;
	uint32_t currentIndexCount = 0u;
	uint32_t maxIndices = 0u;

	uint32_t inMemDrawObjectCount = 0u;
	uint32_t currentDrawObjectCount = 0u;
	uint32_t maxDrawObjects = 0u;

	uint32_t inMemLineStylesCount = 0u;
	uint32_t currentLineStylesCount = 0u;
	uint32_t maxLineStyles = 0u;

	uint64_t geometryBufferAddress = 0u;

	uint64_t inMemGeometryBufferSize = 0u;
	uint64_t currentGeometryBufferSize = 0u;
	uint64_t maxGeometryBufferSize = 0u;
};

class CADApp : public ApplicationBase
{
	constexpr static uint32_t FRAMES_IN_FLIGHT = 3u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	constexpr static uint32_t WIN_W = 1280u;
	constexpr static uint32_t WIN_H = 720u;

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

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
	core::smart_refctd_ptr<video::IGPUPipelineLayout> graphicsPipelineLayout;

	DrawBuffersFiller drawBuffers[FRAMES_IN_FLIGHT];

	constexpr size_t getMaxMemoryNeeded(uint32_t numberOfLines, uint32_t numberOfEllipses)
	{
		size_t mem = sizeof(Globals);
		uint32_t allObjectsCount = numberOfLines + numberOfEllipses;
		mem += allObjectsCount * 6u * sizeof(uint32_t); // Index Buffer 6 indices per object cage
		mem += allObjectsCount * sizeof(DrawObject); // One DrawObject struct per object
		mem += numberOfLines * 4u * sizeof(double2); // 4 points per line max (generated before/after for calculations)
		mem += numberOfEllipses * sizeof(PackedEllipseInfo);
		return mem;
	}

	void initDrawObjects(uint32_t maxObjects = 128u)
	{
		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			drawBuffers[i] = DrawBuffersFiller(core::smart_refctd_ptr(utilities));

			size_t maxIndices = maxObjects * 6u * 2u;
			drawBuffers[i].allocateIndexBuffer(logicalDevice, maxIndices);
			drawBuffers[i].allocateDrawObjectsBuffer(logicalDevice, maxObjects);
			drawBuffers[i].allocateStylesBuffer(logicalDevice, 16u);

			size_t geometryBufferSize = maxObjects * sizeof(PackedEllipseInfo);
			drawBuffers[i].allocateGeometryBuffer(logicalDevice, geometryBufferSize);
		}

		for(uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i)
		{
			video::IGPUBuffer::SCreationParams globalsCreationParams = {};
			globalsCreationParams.size = sizeof(Globals);
			globalsCreationParams.usage = video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
			globalsBuffer[i] = logicalDevice->createBuffer(std::move(globalsCreationParams));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = globalsBuffer[i]->getMemoryReqs();
			memReq.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
			auto globalsBufferMem = logicalDevice->allocate(memReq, globalsBuffer[i].get());
		}

		// pseudoStencil

		asset::E_FORMAT pseudoStencilFormat = asset::EF_R32_UINT;

		video::IPhysicalDevice::SImageFormatPromotionRequest promotionRequest = {};
		promotionRequest.originalFormat = asset::EF_R8_UINT;
		promotionRequest.usages = {};
		promotionRequest.usages.storageImageAtomic = true;
		pseudoStencilFormat = physicalDevice->promoteImageFormat(promotionRequest, video::IGPUImage::ET_OPTIMAL);
		
		for(uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			video::IGPUImage::SCreationParams imgInfo;
			imgInfo.format = pseudoStencilFormat;
			imgInfo.type = video::IGPUImage::ET_2D;
			imgInfo.extent.width = WIN_W;
			imgInfo.extent.height = WIN_H;
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
		initParams.windowWidth = WIN_W;
		initParams.windowHeight = WIN_H;
		initParams.swapchainImageCount = 3u;
		initParams.swapchainImageUsage = swapchainImageUsage;
		initParams.depthFormat = getDepthFormat();
		initParams.acceptableSurfaceFormats = acceptableSurfaceFormats.data();
		initParams.acceptableSurfaceFormatCount = acceptableSurfaceFormats.size();
		initParams.physicalDeviceFilter.requiredFeatures.bufferDeviceAddress = true;
		initParams.physicalDeviceFilter.requiredFeatures.shaderFloat64 = true;
		initParams.physicalDeviceFilter.requiredFeatures.fillModeNonSolid = DebugMode;
		initParams.physicalDeviceFilter.requiredFeatures.fragmentShaderPixelInterlock = FragmentShaderPixelInterlock;
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

		renderpassInitial = createRenderpass(m_swapchainCreationParams.surfaceFormat.format, getDepthFormat(), nbl::video::IGPURenderpass::ELO_CLEAR, asset::IImage::EL_UNDEFINED, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL);
		renderpassInBetween = createRenderpass(m_swapchainCreationParams.surfaceFormat.format, getDepthFormat(), nbl::video::IGPURenderpass::ELO_LOAD, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL);
		renderpassFinal = createRenderpass(m_swapchainCreationParams.surfaceFormat.format, getDepthFormat(), nbl::video::IGPURenderpass::ELO_LOAD, asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL, asset::IImage::EL_PRESENT_SRC);
		
		commandPools = std::move(initOutput.commandPools);
		const auto& graphicsCommandPools = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS];
		const auto& transferCommandPools = commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP];

		CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, WIN_W, WIN_H, swapchain);

		framebuffersDynArraySmartPtr = CommonAPI::createFBOWithSwapchainImages(
			swapchain->getImageCount(), WIN_W, WIN_H,
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

		// Used to load SPIR-V directly, if HLSL Compiler doesn't work
		auto loadSPIRVShader = [&](const std::string& filePath, asset::IShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<asset::ICPUShader>
		{
			system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> file_future;
			system->createFile(file_future, filePath, core::bitflag(system::IFile::ECF_READ)|system::IFile::ECF_MAPPABLE);
			if (auto shader_file=file_future.acquire())
			{
				auto shaderSizeInBytes = (*shader_file)->getSize();
				auto shaderData = (*shader_file)->getMappedPointer();
				// copy the data in so it survives past the destruction of the mapped file
				auto vertexShaderSPIRVBuffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<>>(shaderSizeInBytes,shaderData);
				return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(vertexShaderSPIRVBuffer), stage, asset::IShader::E_CONTENT_TYPE::ECT_SPIRV, std::string(filePath));
			}
			return nullptr;
		};

		core::smart_refctd_ptr<video::IGPUSpecializedShader> shaders[3u] = {};
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			core::smart_refctd_ptr<asset::ICPUSpecializedShader> cpuShaders[3u] = {};
			constexpr auto vertexShaderPath = "../vertex_shader.hlsl";
			constexpr auto fragmentShaderPath = "../fragment_shader.hlsl";
			constexpr auto debugfragmentShaderPath = "../fragment_shader_debug.hlsl";
			cpuShaders[0u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(vertexShaderPath, params).getContents().begin());
			cpuShaders[1u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(fragmentShaderPath, params).getContents().begin());
			cpuShaders[2u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(debugfragmentShaderPath, params).getContents().begin());
			cpuShaders[0u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			cpuShaders[1u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			cpuShaders[2u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			auto gpuShaders = CPU2GPU.getGPUObjectsFromAssets(cpuShaders, cpuShaders + 3u, cpu2gpuParams);
			shaders[0u] = gpuShaders->begin()[0u];
			shaders[1u] = gpuShaders->begin()[1u];
			shaders[2u] = gpuShaders->begin()[2u];
		}

		initDrawObjects();

		video::IGPUDescriptorSetLayout::SBinding bindings[4u] = {};
		bindings[0u].binding = 0u;
		bindings[0u].type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER;
		bindings[0u].count = 1u;
		bindings[0u].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

		bindings[1u].binding = 1u;
		bindings[1u].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
		bindings[1u].count = 1u;
		bindings[1u].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

		bindings[2u].binding = 2u;
		bindings[2u].type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
		bindings[2u].count = 1u;
		bindings[2u].stageFlags = asset::IShader::ESS_FRAGMENT;

		bindings[3u].binding = 3u;
		bindings[3u].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
		bindings[3u].count = 1u;
		bindings[3u].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

		auto descriptorSetLayout = logicalDevice->createDescriptorSetLayout(bindings, bindings+4u);

		nbl::core::smart_refctd_ptr<nbl::video::IDescriptorPool> descriptorPool = nullptr;
		{
			nbl::video::IDescriptorPool::SCreateInfo createInfo = {};
			createInfo.flags = nbl::video::IDescriptorPool::ECF_NONE;
			createInfo.maxSets = 128u;
			createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] = FRAMES_IN_FLIGHT;
			createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] = 2*FRAMES_IN_FLIGHT;
			createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)] = FRAMES_IN_FLIGHT;

			descriptorPool = logicalDevice->createDescriptorPool(std::move(createInfo));
		}

		for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
		{
			descriptorSets[i] = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(descriptorSetLayout));
			video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[4u] = {};
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

			video::IGPUDescriptorSet::SWriteDescriptorSet descriptorUpdates[4u] = {};
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

			logicalDevice->updateDescriptorSets(4u, descriptorUpdates, 0u, nullptr);
		}

		graphicsPipelineLayout = logicalDevice->createPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);

		video::IGPURenderpassIndependentPipeline::SCreationParams renderpassIndependantPipeInfo = {};
		renderpassIndependantPipeInfo.layout = graphicsPipelineLayout;
		renderpassIndependantPipeInfo.shaders[0u] = shaders[0u];
		renderpassIndependantPipeInfo.shaders[1u] = shaders[1u];
		// renderpassIndependantPipeInfo.vertexInput; no gpu vertex buffers
		renderpassIndependantPipeInfo.blend.blendParams[0u].blendEnable = true;
		renderpassIndependantPipeInfo.blend.blendParams[0u].srcColorFactor = asset::EBF_SRC_ALPHA;
		renderpassIndependantPipeInfo.blend.blendParams[0u].dstColorFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
		renderpassIndependantPipeInfo.blend.blendParams[0u].colorBlendOp = asset::EBO_ADD;
		renderpassIndependantPipeInfo.blend.blendParams[0u].srcAlphaFactor = asset::EBF_ONE;
		renderpassIndependantPipeInfo.blend.blendParams[0u].dstAlphaFactor = asset::EBF_ZERO;
		renderpassIndependantPipeInfo.blend.blendParams[0u].alphaBlendOp = asset::EBO_ADD;
		renderpassIndependantPipeInfo.blend.blendParams[0u].colorWriteMask = (1u << 4u) - 1u;

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
		m_Camera.setAspectRatio((double)WIN_W / WIN_H);
		m_Camera.setSize(200.0);

		m_timeElapsed = 0.0;
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	double getScreenToWorldRatio(const double4x4& viewProjectionMatrix, uint2 windowSize)
	{
		double idx_0_0 = viewProjectionMatrix._r0[0u] * (windowSize.X / 2.0);
		double idx_1_1 = viewProjectionMatrix._r1[1u] * (windowSize.Y / 2.0);
		double det_2x2_mat = idx_0_0 * idx_1_1;
		return core::sqrt(core::abs(det_2x2_mat));
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

		Globals globalData = {};
		globalData.antiAliasingFactor = 1.0f;// + abs(cos(m_timeElapsed * 0.0008))*20.0f;
		globalData.resolution = uint2{ WIN_W, WIN_H };
		globalData.viewProjection = m_Camera.constructViewProjection();
		globalData.screenToWorldRatio = getScreenToWorldRatio(globalData.viewProjection, globalData.resolution);

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

			asset::SClearColorValue clear = {};
			clear.uint32[0] = 0u;

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
			area.extent = { WIN_W, WIN_H };
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
			nbl::video::IGPUCommandBuffer::SBufferMemoryBarrier bufferBarriers[4u] = {};
			bufferBarriers[0].barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
			bufferBarriers[0].barrier.dstAccessMask = nbl::asset::EAF_UNIFORM_READ_BIT;
			bufferBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[0].buffer = globalsBuffer[m_resourceIx];
			bufferBarriers[0].offset = 0u;
			bufferBarriers[0].size = globalsBuffer[m_resourceIx]->getSize();

			bufferBarriers[1].barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
			bufferBarriers[1].barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT;
			bufferBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[1].buffer = currentDrawBuffers.gpuDrawBuffers.drawObjectsBuffer;
			bufferBarriers[1].offset = 0u;
			bufferBarriers[1].size = currentDrawBuffers.getCurrentDrawObjectsBufferSize();

			bufferBarriers[2].barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
			bufferBarriers[2].barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT;
			bufferBarriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[2].buffer = currentDrawBuffers.gpuDrawBuffers.geometryBuffer;
			bufferBarriers[2].offset = 0u;
			bufferBarriers[2].size = currentDrawBuffers.getCurrentGeometryBufferSize();

			bufferBarriers[3].barrier.srcAccessMask = nbl::asset::EAF_MEMORY_WRITE_BIT;
			bufferBarriers[3].barrier.dstAccessMask = nbl::asset::EAF_SHADER_READ_BIT;
			bufferBarriers[3].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[3].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarriers[3].buffer = currentDrawBuffers.gpuDrawBuffers.lineStylesBuffer;
			bufferBarriers[3].offset = 0u;
			bufferBarriers[3].size = currentDrawBuffers.getCurrentLineStylesBufferSize();
			cb->pipelineBarrier(nbl::asset::EPSF_TRANSFER_BIT, nbl::asset::EPSF_VERTEX_SHADER_BIT | nbl::asset::EPSF_FRAGMENT_SHADER_BIT, nbl::asset::EDF_NONE, 0u, nullptr, 4u, bufferBarriers, 0u, nullptr);
		}
	}

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
			area.extent = { WIN_W, WIN_H };

			beginInfo.clearValueCount = 0u;
			beginInfo.framebuffer = framebuffersDynArraySmartPtr->begin()[m_SwapchainImageIx];
			beginInfo.renderpass = renderpassFinal;
			beginInfo.renderArea = area;
			beginInfo.clearValues = nullptr;
		}

		pipelineBarriersBeforeDraw(cb.get());

		cb->beginRenderPass(&beginInfo, asset::ESC_INLINE);

		const uint32_t currentIndexCount = drawBuffers[m_resourceIx].getIndexCount();
		cb->bindDescriptorSets(asset::EPBP_GRAPHICS, graphicsPipelineLayout.get(), 0u, 1u, &descriptorSets[m_resourceIx].get());
		cb->bindIndexBuffer(drawBuffers[m_resourceIx].gpuDrawBuffers.indexBuffer.get(), 0u, asset::EIT_32BIT);
		cb->bindGraphicsPipeline(graphicsPipeline.get());
		cb->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);

		if constexpr (DebugMode)
		{
			cb->bindGraphicsPipeline(debugGraphicsPipeline.get());
			cb->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);
		}

		cb->endRenderPass();

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
			//std::vector<double2> linePoints;
			//linePoints.push_back({ -50.0, 0.0 });
			//linePoints.push_back({ 0.0, 0.0 });
			//linePoints.push_back({ 80.0, 10.0 });
			//linePoints.push_back({ 40.0, 40.0 });
			//linePoints.push_back({ 0.0, 40.0 });
			//linePoints.push_back({ 30.0, 80.0 });
			//linePoints.push_back({ -30.0, 50.0 });
			//linePoints.push_back({ -30.0, 110.0 });
			//linePoints.push_back({ +30.0, -112.0 });
			//currentDrawBuffers.addLines(std::move(linePoints));
		}
		else if (mode == ExampleMode::CASE_1)
		{
			//std::vector<double2> linePoints;
			//linePoints.push_back({ 0.0, 0.0 });
			//linePoints.push_back({ 30.0, 30.0 });
			//currentDrawBuffers.addLines(std::move(linePoints));
		}
		else if (mode == ExampleMode::CASE_2)
		{
			//std::vector<double2> linePoints;
			//linePoints.push_back({ -70.0, cos(m_timeElapsed * 0.00003) * 10 });
			//linePoints.push_back({ 70.0, cos(m_timeElapsed * 0.00003) * 10 });
			//currentDrawBuffers.addLines(std::move(linePoints));
		}
		else if (mode == ExampleMode::CASE_3)
		{
			//constexpr double twoPi = core::PI<double>() * 2.0;
			//PackedEllipseInfo ellipse = {};
			//const double a = m_timeElapsed * 0.001;
			//// ellipse.majorAxis = { 40.0 * cos(a), 40.0 * sin(a) };
			//ellipse.majorAxis = double2{ 30.0, 0.0 };
			//ellipse.center = double2{ 0, 0 };
			//ellipse.eccentricityPacked = (0.6 * UINT32_MAX);

			//ellipse.angleBoundsPacked = uint2{
			//	static_cast<uint32_t>(((0.0) / twoPi) * UINT32_MAX),
			//	static_cast<uint32_t>(((core::PI<double>() * 0.5) / twoPi) * UINT32_MAX)
			//};
			////currentDrawBuffers.addEllipse(ellipse);

			//ellipse.angleBoundsPacked = uint2{
			//	static_cast<uint32_t>(((core::PI<double>() * 0.5) / twoPi) * UINT32_MAX),
			//	static_cast<uint32_t>(((core::PI<double>()) / twoPi) * UINT32_MAX)
			//};
			////currentDrawBuffers.addEllipse(ellipse);
			//ellipse.angleBoundsPacked = uint2{
			//	static_cast<uint32_t>(((core::PI<double>()) / twoPi) * UINT32_MAX),
			//	static_cast<uint32_t>(((core::PI<double>() * 1.5) / twoPi) * UINT32_MAX)
			//};
			////currentDrawBuffers.addEllipse(ellipse);
			//ellipse.majorAxis = double2{ 50.0, 0.0 };
			//double start = 0.0 * core::PI<double>();
			//double end = abs(sin(m_timeElapsed * 0.0005)) * core::PI<double>();
			//ellipse.angleBoundsPacked = uint2{
			//	static_cast<uint32_t>(((start) / twoPi) * UINT32_MAX),
			//	static_cast<uint32_t>(((end) / twoPi) * UINT32_MAX)
			//};
			//currentDrawBuffers.addEllipse(ellipse, firstStyleIdx);
			//ellipse.majorAxis = double2{ 30.0 * sin(m_timeElapsed * 0.0005), 30.0 * cos(m_timeElapsed * 0.0005) };
			//ellipse.center = double2{ 50, 50 };
			//ellipse.angleBoundsPacked = uint2{
			//	static_cast<uint32_t>(((core::PI<double>() * 0) / twoPi) * UINT32_MAX),
			//	static_cast<uint32_t>(((core::PI<double>() * 2.0 / 3.0) / twoPi) * UINT32_MAX)
			//};
			//currentDrawBuffers.addEllipse(ellipse, firstStyleIdx);

			// intendedNextSubmit = submitInBetweenDraws(m_resourceIx, submissionQueue, submissionFence, intendedNextSubmit);

			LineStyle style = {};
			style.screenSpaceLineWidth = 0.0f;
			style.worldSpaceLineWidth = 5.0f;
			style.color = float4(0.7f, 0.3f, 0.1f, 0.5f);

			LineStyle style2 = {};
			style2.screenSpaceLineWidth = 5.0f;
			style2.worldSpaceLineWidth = 0.0f;
			style2.color = float4(0.2f, 0.6f, 0.2f, 0.5f);


			CPolyline polyline;
			{
				std::vector<double2> linePoints;
				linePoints.push_back({ -50.0, 0.0 });
				linePoints.push_back({ sin(m_timeElapsed * 0.005) * 20, cos(m_timeElapsed * 0.005) * 20 });
				linePoints.push_back({ -sin(m_timeElapsed * 0.005) * 20, -cos(m_timeElapsed * 0.005) * 20 });
				linePoints.push_back({ 50.0, 0.0 });
				polyline.addLinePoints(std::move(linePoints));
			}
			{
				std::vector<CPolyline::EllipticalArcInfo> ellipticalArcs;
				CPolyline::EllipticalArcInfo ellipse;
				ellipse.center = double2(80.0, 0.0);
				ellipse.majorAxis = double2(30.0, 0.0);
				ellipse.eccentricity = 1.0;
				ellipse.angleBounds.X = 0.0;
				ellipse.angleBounds.Y = nbl::core::PI<double>();
				ellipticalArcs.push_back(ellipse);
				polyline.addEllipticalArcs(std::move(ellipticalArcs));
			}
			{
				std::vector<double2> linePoints;
				linePoints.push_back({ 110.0, 0.0 });
				linePoints.push_back({ 110.0, 100.0 });
				linePoints.push_back({ 150.0, 100.0 });
				polyline.addLinePoints(std::move(linePoints));
			}

			intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline, style, submissionQueue, submissionFence, intendedNextSubmit);
			intendedNextSubmit = currentDrawBuffers.drawPolyline(polyline, style2, submissionQueue, submissionFence, intendedNextSubmit);
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
			m_Camera.setSize(20.0 + abs(cos(m_timeElapsed * 0.0001)) * 7000);
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
				// TODO:
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