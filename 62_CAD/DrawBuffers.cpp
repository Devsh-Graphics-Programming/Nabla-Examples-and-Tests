#include "DrawBuffers.h"

DrawBuffersFiller::DrawBuffersFiller(nbl::core::smart_refctd_ptr<nbl::video::IUtilities>&& utils)
{
	utilities = utils;
}

// function is called when buffer is filled and we should submit draws and clear the buffers and continue filling

void DrawBuffersFiller::setSubmitDrawsFunction(SubmitFunc func)
{
	submitDraws = func;
}

void DrawBuffersFiller::allocateIndexBuffer(nbl::video::ILogicalDevice* logicalDevice, uint32_t indices)
{
	maxIndices = indices;
	const size_t indexBufferSize = maxIndices * sizeof(uint32_t);

	nbl::video::IGPUBuffer::SCreationParams indexBufferCreationParams = {};
	indexBufferCreationParams.size = indexBufferSize;
	indexBufferCreationParams.usage = nbl::video::IGPUBuffer::EUF_INDEX_BUFFER_BIT | nbl::video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuDrawBuffers.indexBuffer = logicalDevice->createBuffer(std::move(indexBufferCreationParams));
	gpuDrawBuffers.indexBuffer->setObjectDebugName("indexBuffer");

	nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.indexBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto indexBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.indexBuffer.get());

	cpuDrawBuffers.indexBuffer = nbl::core::make_smart_refctd_ptr<nbl::asset::ICPUBuffer>(indexBufferSize);
}

void DrawBuffersFiller::allocateMainObjectsBuffer(nbl::video::ILogicalDevice* logicalDevice, uint32_t mainObjects)
{
	maxMainObjects = mainObjects;
	size_t mainObjectsBufferSize = mainObjects * sizeof(MainObject);

	nbl::video::IGPUBuffer::SCreationParams mainObjectsCreationParams = {};
	mainObjectsCreationParams.size = mainObjectsBufferSize;
	mainObjectsCreationParams.usage = nbl::video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | nbl::video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuDrawBuffers.mainObjectsBuffer = logicalDevice->createBuffer(std::move(mainObjectsCreationParams));
	gpuDrawBuffers.mainObjectsBuffer->setObjectDebugName("mainObjectsBuffer");

	nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.mainObjectsBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto mainObjectsBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.mainObjectsBuffer.get());

	cpuDrawBuffers.mainObjectsBuffer = nbl::core::make_smart_refctd_ptr<nbl::asset::ICPUBuffer>(mainObjectsBufferSize);
}

void DrawBuffersFiller::allocateDrawObjectsBuffer(nbl::video::ILogicalDevice* logicalDevice, uint32_t drawObjects)
{
	maxDrawObjects = drawObjects;
	size_t drawObjectsBufferSize = drawObjects * sizeof(DrawObject);

	nbl::video::IGPUBuffer::SCreationParams drawObjectsCreationParams = {};
	drawObjectsCreationParams.size = drawObjectsBufferSize;
	drawObjectsCreationParams.usage = nbl::video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | nbl::video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuDrawBuffers.drawObjectsBuffer = logicalDevice->createBuffer(std::move(drawObjectsCreationParams));
	gpuDrawBuffers.drawObjectsBuffer->setObjectDebugName("drawObjectsBuffer");

	nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.drawObjectsBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto drawObjectsBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.drawObjectsBuffer.get());

	cpuDrawBuffers.drawObjectsBuffer = nbl::core::make_smart_refctd_ptr<nbl::asset::ICPUBuffer>(drawObjectsBufferSize);
}

void DrawBuffersFiller::allocateGeometryBuffer(nbl::video::ILogicalDevice* logicalDevice, size_t size)
{
	maxGeometryBufferSize = size;

	nbl::video::IGPUBuffer::SCreationParams geometryCreationParams = {};
	geometryCreationParams.size = size;
	geometryCreationParams.usage = nbl::core::bitflag(nbl::video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | nbl::video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | nbl::video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuDrawBuffers.geometryBuffer = logicalDevice->createBuffer(std::move(geometryCreationParams));
	gpuDrawBuffers.geometryBuffer->setObjectDebugName("geometryBuffer");

	nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.geometryBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto geometryBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.geometryBuffer.get(), nbl::video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
	geometryBufferAddress = logicalDevice->getBufferDeviceAddress(gpuDrawBuffers.geometryBuffer.get());

	cpuDrawBuffers.geometryBuffer = nbl::core::make_smart_refctd_ptr<nbl::asset::ICPUBuffer>(size);
}

void DrawBuffersFiller::allocateStylesBuffer(nbl::video::ILogicalDevice* logicalDevice, uint32_t stylesCount)
{
	maxLineStyles = stylesCount;
	size_t lineStylesBufferSize = stylesCount * sizeof(LineStyle);

	nbl::video::IGPUBuffer::SCreationParams lineStylesCreationParams = {};
	lineStylesCreationParams.size = lineStylesBufferSize;
	lineStylesCreationParams.usage = nbl::video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | nbl::video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuDrawBuffers.lineStylesBuffer = logicalDevice->createBuffer(std::move(lineStylesCreationParams));
	gpuDrawBuffers.lineStylesBuffer->setObjectDebugName("lineStylesBuffer");

	nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.lineStylesBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto stylesBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.lineStylesBuffer.get());

	cpuDrawBuffers.lineStylesBuffer = nbl::core::make_smart_refctd_ptr<nbl::asset::ICPUBuffer>(lineStylesBufferSize);
}

void DrawBuffersFiller::allocateCustomClipProjectionBuffer(nbl::video::ILogicalDevice* logicalDevice, uint32_t ClipProjectionDataCount)
{
	maxClipProjectionData = ClipProjectionDataCount;
	size_t customClipProjectionBufferSize = maxClipProjectionData * sizeof(ClipProjectionData);

	nbl::video::IGPUBuffer::SCreationParams customClipProjectionCreationParams = {};
	customClipProjectionCreationParams.size = customClipProjectionBufferSize;
	customClipProjectionCreationParams.usage = nbl::video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | nbl::video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuDrawBuffers.customClipProjectionBuffer = logicalDevice->createBuffer(std::move(customClipProjectionCreationParams));
	gpuDrawBuffers.customClipProjectionBuffer->setObjectDebugName("customClipProjectionBuffer");

	nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.customClipProjectionBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto customClipProjectionBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.customClipProjectionBuffer.get());

	cpuDrawBuffers.customClipProjectionBuffer = nbl::core::make_smart_refctd_ptr<nbl::asset::ICPUBuffer>(customClipProjectionBufferSize);
}

//! this function fills buffers required for drawing a polyline and submits a draw through provided callback when there is not enough memory.

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::drawPolyline(const CPolylineBase& polyline, const CPULineStyle& cpuLineStyle, const uint32_t clipProjectionIdx, nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit)
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

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::drawPolyline(nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit, const CPolylineBase& polyline, const uint32_t polylineMainObjIdx)
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

	if (!polyline.getConnectors().empty())
	{
		uint32_t currentConnectorPolylineObject = 0u;
		while (true)
		{
			addPolylineConnectors_Internal(polyline, currentConnectorPolylineObject, polylineMainObjIdx);

			if (currentConnectorPolylineObject >= polyline.getConnectors().size())
			{
				break;
			}
			else
			{
				intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
				intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
				resetIndexCounters();
				resetGeometryCounters();
				// We don't reset counters for linestyles, mainObjects and customClipProjection because we will be reusing them
			}
		}
	}

	return intendedNextSubmit;
}

// If we had infinite mem, we would first upload all curves into geometry buffer then upload the "CurveBoxes" with correct gpu addresses to those
// But we don't have that so we have to follow a similar auto submission as the "drawPolyline" function with some mutations:
// We have to find the MAX number of "CurveBoxes" we could draw, and since both the "Curves" and "CurveBoxes" reside in geometry buffer,
// it has to be taken into account when calculating "how many curve boxes we could draw and when we need to submit/clear"
// So same as drawPolylines, we would first try to fill the geometry buffer and index buffer that corresponds to "backfaces or even provoking vertices"
// then change index buffer to draw front faces of the curveBoxes that already reside in geometry buffer memory
// then if anything was left (the ones that weren't in memory for front face of the curveBoxes) we copy their geom to mem again and use frontface/oddProvoking vertex

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::drawHatch(const Hatch& hatch, const float32_t4 color, const uint32_t clipProjectionIdx, nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit)
{
	CPULineStyle lineStyle;
	lineStyle.color = color;
	lineStyle.stipplePatternSize = 0u;

	uint32_t styleIdx;
	intendedNextSubmit = addLineStyle_SubmitIfNeeded(lineStyle, styleIdx, submissionQueue, submissionFence, intendedNextSubmit);

	MainObject mainObj = {};
	mainObj.styleIdx = styleIdx;
	mainObj.clipProjectionIdx = clipProjectionIdx;
	uint32_t mainObjIdx;
	intendedNextSubmit = addMainObject_SubmitIfNeeded(mainObj, mainObjIdx, submissionQueue, submissionFence, intendedNextSubmit);

	const auto sectionsCount = 1;

	uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject used in vertex shader. You can think of it as a Cage.

	while (true)
	{
		bool shouldSubmit = false;
		addHatch_Internal(hatch, currentObjectInSection, mainObjIdx);

		const auto sectionObjectCount = hatch.getHatchBoxCount();
		if (currentObjectInSection >= sectionObjectCount)
			break;

		intendedNextSubmit = finalizeAllCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
		intendedNextSubmit = submitDraws(submissionQueue, submissionFence, intendedNextSubmit);
		resetIndexCounters();
		resetGeometryCounters();
	}

	return intendedNextSubmit;
}

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::finalizeAllCopiesToGPU(nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit)
{
	intendedNextSubmit = finalizeIndexCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
	intendedNextSubmit = finalizeMainObjectCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
	intendedNextSubmit = finalizeGeometryCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
	intendedNextSubmit = finalizeLineStyleCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);
	intendedNextSubmit = finalizeCustomClipProjectionCopiesToGPU(submissionQueue, submissionFence, intendedNextSubmit);

	return intendedNextSubmit;
}

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::addLineStyle_SubmitIfNeeded(const CPULineStyle& lineStyle, uint32_t& outLineStyleIdx, nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit)
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

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::addMainObject_SubmitIfNeeded(const MainObject& mainObject, uint32_t& outMainObjectIdx, nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit)
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

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::addClipProjectionData_SubmitIfNeeded(const ClipProjectionData& clipProjectionData, uint32_t& outClipProjectionIdx, nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit)
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

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::finalizeIndexCopiesToGPU(nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit)
{
	// Copy Indices
	uint32_t remainingIndexCount = currentIndexCount - inMemIndexCount;
	nbl::asset::SBufferRange<nbl::video::IGPUBuffer> indicesRange = { sizeof(index_buffer_type) * inMemIndexCount, sizeof(index_buffer_type) * remainingIndexCount, gpuDrawBuffers.indexBuffer };
	const index_buffer_type* srcIndexData = reinterpret_cast<index_buffer_type*>(cpuDrawBuffers.indexBuffer->getPointer()) + inMemIndexCount;
	if (indicesRange.size > 0u)
		intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(indicesRange, srcIndexData, submissionQueue, submissionFence, intendedNextSubmit);
	inMemIndexCount = currentIndexCount;
	return intendedNextSubmit;
}

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::finalizeMainObjectCopiesToGPU(nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit)
{
	// Copy MainObjects
	uint32_t remainingMainObjects = currentMainObjectCount - inMemMainObjectCount;
	nbl::asset::SBufferRange<nbl::video::IGPUBuffer> mainObjectsRange = { sizeof(MainObject) * inMemMainObjectCount, sizeof(MainObject) * remainingMainObjects, gpuDrawBuffers.mainObjectsBuffer };
	const MainObject* srcMainObjData = reinterpret_cast<MainObject*>(cpuDrawBuffers.mainObjectsBuffer->getPointer()) + inMemMainObjectCount;
	if (mainObjectsRange.size > 0u)
		intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(mainObjectsRange, srcMainObjData, submissionQueue, submissionFence, intendedNextSubmit);
	inMemMainObjectCount = currentMainObjectCount;
	return intendedNextSubmit;
}

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::finalizeGeometryCopiesToGPU(nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit)
{
	// Copy DrawObjects
	uint32_t remainingDrawObjects = currentDrawObjectCount - inMemDrawObjectCount;
	nbl::asset::SBufferRange<nbl::video::IGPUBuffer> drawObjectsRange = { sizeof(DrawObject) * inMemDrawObjectCount, sizeof(DrawObject) * remainingDrawObjects, gpuDrawBuffers.drawObjectsBuffer };
	const DrawObject* srcDrawObjData = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + inMemDrawObjectCount;
	if (drawObjectsRange.size > 0u)
		intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(drawObjectsRange, srcDrawObjData, submissionQueue, submissionFence, intendedNextSubmit);
	inMemDrawObjectCount = currentDrawObjectCount;

	// Copy GeometryBuffer
	uint64_t remainingGeometrySize = currentGeometryBufferSize - inMemGeometryBufferSize;
	nbl::asset::SBufferRange<nbl::video::IGPUBuffer> geomRange = { inMemGeometryBufferSize, remainingGeometrySize, gpuDrawBuffers.geometryBuffer };
	const uint8_t* srcGeomData = reinterpret_cast<uint8_t*>(cpuDrawBuffers.geometryBuffer->getPointer()) + inMemGeometryBufferSize;
	if (geomRange.size > 0u)
		intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(geomRange, srcGeomData, submissionQueue, submissionFence, intendedNextSubmit);
	inMemGeometryBufferSize = currentGeometryBufferSize;

	return intendedNextSubmit;
}

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::finalizeLineStyleCopiesToGPU(nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit)
{
	// Copy LineStyles
	uint32_t remainingLineStyles = currentLineStylesCount - inMemLineStylesCount;
	nbl::asset::SBufferRange<nbl::video::IGPUBuffer> stylesRange = { sizeof(LineStyle) * inMemLineStylesCount, sizeof(LineStyle) * remainingLineStyles, gpuDrawBuffers.lineStylesBuffer };
	const LineStyle* srcLineStylesData = reinterpret_cast<LineStyle*>(cpuDrawBuffers.lineStylesBuffer->getPointer()) + inMemLineStylesCount;
	if (stylesRange.size > 0u)
		intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(stylesRange, srcLineStylesData, submissionQueue, submissionFence, intendedNextSubmit);
	inMemLineStylesCount = currentLineStylesCount;
	return intendedNextSubmit;
}

nbl::video::IGPUQueue::SSubmitInfo DrawBuffersFiller::finalizeCustomClipProjectionCopiesToGPU(nbl::video::IGPUQueue* submissionQueue, nbl::video::IGPUFence* submissionFence, nbl::video::IGPUQueue::SSubmitInfo intendedNextSubmit)
{
	// Copy LineStyles
	uint32_t remainingClipProjectionData = currentClipProjectionDataCount - inMemClipProjectionDataCount;
	nbl::asset::SBufferRange<nbl::video::IGPUBuffer> clipProjectionRange = { sizeof(ClipProjectionData) * inMemClipProjectionDataCount, sizeof(ClipProjectionData) * remainingClipProjectionData, gpuDrawBuffers.customClipProjectionBuffer };
	const ClipProjectionData* srcClipProjectionData = reinterpret_cast<ClipProjectionData*>(cpuDrawBuffers.customClipProjectionBuffer->getPointer()) + inMemClipProjectionDataCount;
	if (clipProjectionRange.size > 0u)
		intendedNextSubmit = utilities->updateBufferRangeViaStagingBuffer(clipProjectionRange, srcClipProjectionData, submissionQueue, submissionFence, intendedNextSubmit);
	inMemClipProjectionDataCount = currentClipProjectionDataCount;
	return intendedNextSubmit;
}

uint32_t DrawBuffersFiller::addMainObject_Internal(const MainObject& mainObject)
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

uint32_t DrawBuffersFiller::addLineStyle_Internal(const CPULineStyle& cpuLineStyle)
{
	LineStyle gpuLineStyle = cpuLineStyle.getAsGPUData();
	_NBL_DEBUG_BREAK_IF(gpuLineStyle.stipplePatternSize > LineStyle::STIPPLE_PATTERN_MAX_SZ); // Oops, even after style normalization the style is too long to be in gpu mem :(
	LineStyle* stylesArray = reinterpret_cast<LineStyle*>(cpuDrawBuffers.lineStylesBuffer->getPointer());
	for (uint32_t i = 0u; i < currentLineStylesCount; ++i)
	{
		const LineStyle& itr = stylesArray[i];

		if (itr == gpuLineStyle)
			return i;
	}

	if (currentLineStylesCount >= maxLineStyles)
		return InvalidLineStyleIdx;

	void* dst = stylesArray + currentLineStylesCount;
	memcpy(dst, &gpuLineStyle, sizeof(LineStyle));
	return currentLineStylesCount++;
}

uint32_t DrawBuffersFiller::addClipProjectionData_Internal(const ClipProjectionData& clipProjectionData)
{
	ClipProjectionData* clipProjectionArray = reinterpret_cast<ClipProjectionData*>(cpuDrawBuffers.customClipProjectionBuffer->getPointer());
	if (currentClipProjectionDataCount >= maxClipProjectionData)
		return InvalidClipProjectionIdx;

	void* dst = clipProjectionArray + currentClipProjectionDataCount;
	memcpy(dst, &clipProjectionData, sizeof(ClipProjectionData));
	return currentClipProjectionDataCount++;
}

void DrawBuffersFiller::addPolylineObjects_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
{
	if (section.type == ObjectType::LINE)
		addLines_Internal(polyline, section, currentObjectInSection, mainObjIdx);
	else if (section.type == ObjectType::QUAD_BEZIER)
		addQuadBeziers_Internal(polyline, section, currentObjectInSection, mainObjIdx);
	else
		assert(false); // we don't handle other object types
}

// TODO[Prezmek]: another function named addPolylineConnectors_Internal and you pass a nbl::core::Range<PolylineConnectorInfo>, uint32_t currentPolylineConnectorObj, uint32_t mainObjIdx
// And implement it similar to addLines/QuadBeziers_Internal which is check how much memory is left and how many PolylineConnectors you can fit into the current geometry and drawobj memory left and return to the drawPolylinefunction
void DrawBuffersFiller::addPolylineConnectors_Internal(const CPolylineBase& polyline, uint32_t& currentPolylineConnectorObj, uint32_t mainObjIdx)
{
	const auto maxGeometryBufferConnectors = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(PolylineConnector);

	constexpr uint32_t INDEX_COUNT_PER_CAGE = 6u;
	uint32_t uploadableObjects = (maxIndices - currentIndexCount) / INDEX_COUNT_PER_CAGE;
	uploadableObjects = nbl::core::min(uploadableObjects, maxGeometryBufferConnectors);
	uploadableObjects = nbl::core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

	const auto connectorCount = polyline.getConnectors().size();
	const auto remainingObjects = connectorCount - currentPolylineConnectorObj;

	const uint32_t objectsToUpload = nbl::core::min(uploadableObjects, remainingObjects);

	// Add Indices
	addCagedObjectIndices_Internal(currentDrawObjectCount, objectsToUpload);

	// Add DrawObjs
	DrawObject drawObj = {};
	drawObj.mainObjIndex = mainObjIdx;
	drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::POLYLINE_CONNECTOR) | 0 << 16);
	drawObj.geometryAddress = geometryBufferAddress + currentGeometryBufferSize;
	for (uint32_t i = 0u; i < objectsToUpload; ++i)
	{
		void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
		memcpy(dst, &drawObj, sizeof(DrawObject));
		currentDrawObjectCount += 1u;
		drawObj.geometryAddress += sizeof(PolylineConnector);
	}

	// Add Geometry
	if (objectsToUpload > 0u)
	{
		const auto connectorsByteSize = sizeof(PolylineConnector) * objectsToUpload;
		void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
		auto& connector = polyline.getConnectors()[currentPolylineConnectorObj];
		memcpy(dst, &connector, connectorsByteSize);
		currentGeometryBufferSize += connectorsByteSize;
	}

	currentPolylineConnectorObj += objectsToUpload;
}

// TODO[Przemek]: this function will change a little as you'll be copying LinePointInfos instead of double2's
// Make sure to test with small memory to trigger submitInBetween function when you run out of memory to see if your changes here didn't mess things up, ask Lucas for help if you're not sure on how to do this
void DrawBuffersFiller::addLines_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
{
	assert(section.count >= 1u);
	assert(section.type == ObjectType::LINE);

	const auto maxGeometryBufferPoints = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(LinePointInfo);
	const auto maxGeometryBufferLines = (maxGeometryBufferPoints <= 1u) ? 0u : maxGeometryBufferPoints - 1u;

	uint32_t uploadableObjects = (maxIndices - currentIndexCount) / 6u;
	uploadableObjects = nbl::core::min(uploadableObjects, maxGeometryBufferLines);
	uploadableObjects = nbl::core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

	const auto lineCount = section.count;
	const auto remainingObjects = lineCount - currentObjectInSection;
	uint32_t objectsToUpload = nbl::core::min(uploadableObjects, remainingObjects);

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
		drawObj.geometryAddress += sizeof(LinePointInfo);
	}

	// Add Geometry
	if (objectsToUpload > 0u)
	{
		const auto pointsByteSize = sizeof(LinePointInfo) * (objectsToUpload + 1u);
		void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
		auto& linePoint = polyline.getLinePointAt(section.index + currentObjectInSection);
		memcpy(dst, &linePoint, pointsByteSize);
		currentGeometryBufferSize += pointsByteSize;
	}

	currentObjectInSection += objectsToUpload;
}

void DrawBuffersFiller::addQuadBeziers_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
{
	constexpr uint32_t CagesPerQuadBezier = getCageCountPerPolylineObject(ObjectType::QUAD_BEZIER);
	constexpr uint32_t IndicesPerQuadBezier = 6u * CagesPerQuadBezier;
	assert(section.type == ObjectType::QUAD_BEZIER);

	const auto maxGeometryBufferBeziers = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(QuadraticBezierInfo);

	uint32_t uploadableObjects = (maxIndices - currentIndexCount) / IndicesPerQuadBezier;
	uploadableObjects = nbl::core::min(uploadableObjects, maxGeometryBufferBeziers);
	uploadableObjects = nbl::core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

	const auto beziersCount = section.count;
	const auto remainingObjects = beziersCount - currentObjectInSection;
	uint32_t objectsToUpload = nbl::core::min(uploadableObjects, remainingObjects);

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

void DrawBuffersFiller::addHatch_Internal(const Hatch& hatch, uint32_t& currentObjectInSection, uint32_t mainObjIndex)
{
	constexpr uint32_t IndicesPerHatchBox = 6u;

	const auto maxGeometryBufferHatchBoxes = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(Hatch::CurveHatchBox);

	uint32_t uploadableObjects = (maxIndices - currentIndexCount) / IndicesPerHatchBox;
	uploadableObjects = nbl::core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
	uploadableObjects = nbl::core::min(uploadableObjects, maxGeometryBufferHatchBoxes);

	uint32_t remainingObjects = hatch.getHatchBoxCount() - currentObjectInSection;
	uploadableObjects = nbl::core::min(uploadableObjects, remainingObjects);

	for (uint32_t i = 0; i < uploadableObjects; i++)
	{
		const Hatch::CurveHatchBox& hatchBox = hatch.getHatchBox(i + currentObjectInSection);

		uint64_t hatchBoxAddress;
		{			
			static_assert(sizeof(CurveBox) == sizeof(Hatch::CurveHatchBox));
			void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
			memcpy(dst, &hatchBox, sizeof(CurveBox));
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
void DrawBuffersFiller::addCagedObjectIndices_Internal(uint32_t startObject, uint32_t objectCount)
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
