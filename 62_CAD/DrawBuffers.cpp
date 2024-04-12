#include "DrawBuffers.h"

DrawBuffersFiller::DrawBuffersFiller(smart_refctd_ptr<IUtilities>&& utils, IQueue* copyQueue)
{
	m_utilities = utils;
	m_copyQueue = copyQueue;
}

// function is called when buffer is filled and we should submit draws and clear the buffers and continue filling

void DrawBuffersFiller::setSubmitDrawsFunction(SubmitFunc func)
{
	submitDraws = func;
}

void DrawBuffersFiller::allocateIndexBuffer(ILogicalDevice* logicalDevice, uint32_t maxIndices)
{
	maxIndexCount = maxIndices;
	const size_t indexBufferSize = maxIndices * sizeof(index_buffer_type);
	auto indexBuffer = make_smart_refctd_ptr<ICPUBuffer>(indexBufferSize);

	index_buffer_type* indices = reinterpret_cast<index_buffer_type*>(indexBuffer->getPointer());
	for (uint32_t i = 0u; i < maxIndices / 6u; ++i)
	{
		index_buffer_type objIndex = i;
		indices[i * 6] = objIndex * 4u + 1u;
		indices[i * 6 + 1u] = objIndex * 4u + 0u;
		indices[i * 6 + 2u] = objIndex * 4u + 2u;

		indices[i * 6 + 3u] = objIndex * 4u + 1u;
		indices[i * 6 + 4u] = objIndex * 4u + 2u;
		indices[i * 6 + 5u] = objIndex * 4u + 3u;
	}
	
	IGPUBuffer::SCreationParams indexBufferCreationParams = {};
	indexBufferCreationParams.size = indexBufferSize;
	indexBufferCreationParams.usage = IGPUBuffer::EUF_INDEX_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;

	SIntendedSubmitInfo::SFrontHalf intendedNextSubmit; 
	intendedNextSubmit.queue = m_copyQueue;
	gpuDrawBuffers.indexBuffer = m_utilities->createFilledDeviceLocalBufferOnDedMem(intendedNextSubmit, std::move(indexBufferCreationParams), indices);
	gpuDrawBuffers.indexBuffer->setObjectDebugName("indexBuffer");
}

void DrawBuffersFiller::allocateMainObjectsBuffer(ILogicalDevice* logicalDevice, uint32_t mainObjects)
{
	maxMainObjects = mainObjects;
	size_t mainObjectsBufferSize = mainObjects * sizeof(MainObject);

	IGPUBuffer::SCreationParams mainObjectsCreationParams = {};
	mainObjectsCreationParams.size = mainObjectsBufferSize;
	mainObjectsCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuDrawBuffers.mainObjectsBuffer = logicalDevice->createBuffer(std::move(mainObjectsCreationParams));
	gpuDrawBuffers.mainObjectsBuffer->setObjectDebugName("mainObjectsBuffer");

	IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.mainObjectsBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto mainObjectsBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.mainObjectsBuffer.get());

	cpuDrawBuffers.mainObjectsBuffer = make_smart_refctd_ptr<ICPUBuffer>(mainObjectsBufferSize);
}

void DrawBuffersFiller::allocateDrawObjectsBuffer(ILogicalDevice* logicalDevice, uint32_t drawObjects)
{
	maxDrawObjects = drawObjects;
	size_t drawObjectsBufferSize = drawObjects * sizeof(DrawObject);

	IGPUBuffer::SCreationParams drawObjectsCreationParams = {};
	drawObjectsCreationParams.size = drawObjectsBufferSize;
	drawObjectsCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuDrawBuffers.drawObjectsBuffer = logicalDevice->createBuffer(std::move(drawObjectsCreationParams));
	gpuDrawBuffers.drawObjectsBuffer->setObjectDebugName("drawObjectsBuffer");

	IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.drawObjectsBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto drawObjectsBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.drawObjectsBuffer.get());

	cpuDrawBuffers.drawObjectsBuffer = make_smart_refctd_ptr<ICPUBuffer>(drawObjectsBufferSize);
}

void DrawBuffersFiller::allocateGeometryBuffer(ILogicalDevice* logicalDevice, size_t size)
{
	maxGeometryBufferSize = size;

	IGPUBuffer::SCreationParams geometryCreationParams = {};
	geometryCreationParams.size = size;
	geometryCreationParams.usage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuDrawBuffers.geometryBuffer = logicalDevice->createBuffer(std::move(geometryCreationParams));
	gpuDrawBuffers.geometryBuffer->setObjectDebugName("geometryBuffer");

	IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.geometryBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto geometryBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.geometryBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
	geometryBufferAddress = gpuDrawBuffers.geometryBuffer->getDeviceAddress();

	cpuDrawBuffers.geometryBuffer = make_smart_refctd_ptr<ICPUBuffer>(size);
}

void DrawBuffersFiller::allocateStylesBuffer(ILogicalDevice* logicalDevice, uint32_t lineStylesCount)
{
	{
		maxLineStyles = lineStylesCount;
		size_t lineStylesBufferSize = lineStylesCount * sizeof(LineStyle);

		IGPUBuffer::SCreationParams lineStylesCreationParams = {};
		lineStylesCreationParams.size = lineStylesBufferSize;
		lineStylesCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuDrawBuffers.lineStylesBuffer = logicalDevice->createBuffer(std::move(lineStylesCreationParams));
		gpuDrawBuffers.lineStylesBuffer->setObjectDebugName("lineStylesBuffer");

		IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.lineStylesBuffer->getMemoryReqs();
		memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto stylesBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.lineStylesBuffer.get());

		cpuDrawBuffers.lineStylesBuffer = make_smart_refctd_ptr<ICPUBuffer>(lineStylesBufferSize);
	}
}

//! this function fills buffers required for drawing a polyline and submits a draw through provided callback when there is not enough memory.

void DrawBuffersFiller::drawPolyline(const CPolylineBase& polyline, const LineStyleInfo& lineStyleInfo, SIntendedSubmitInfo& intendedNextSubmit)
{
	if (!lineStyleInfo.isVisible())
		return;

	uint32_t styleIdx = addLineStyle_SubmitIfNeeded(lineStyleInfo, intendedNextSubmit);

	uint32_t mainObjIdx = addMainObject_SubmitIfNeeded(styleIdx, intendedNextSubmit);

	drawPolyline(polyline, mainObjIdx, intendedNextSubmit);
}

void DrawBuffersFiller::drawPolyline(const CPolylineBase& polyline, uint32_t polylineMainObjIdx, SIntendedSubmitInfo& intendedNextSubmit)
{
	if (polylineMainObjIdx == InvalidMainObjectIdx)
	{
		// TODO: assert or log error here
		assert(false);
		return;
	}
	
	const auto sectionsCount = polyline.getSectionsCount();

	uint32_t currentSectionIdx = 0u;
	uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject used in vertex shader. You can think of it as a Cage.

	while (currentSectionIdx < sectionsCount)
	{
		const auto& currentSection = polyline.getSectionInfoAt(currentSectionIdx);
		addPolylineObjects_Internal(polyline, currentSection, currentObjectInSection, polylineMainObjIdx);

		if (currentObjectInSection >= currentSection.count)
		{
			currentSectionIdx++;
			currentObjectInSection = 0u;
		}
		else
			submitCurrentObjectsAndReset(intendedNextSubmit, polylineMainObjIdx);
	}

	if (!polyline.getConnectors().empty())
	{
		uint32_t currentConnectorPolylineObject = 0u;
		while (currentConnectorPolylineObject < polyline.getConnectors().size())
		{
			addPolylineConnectors_Internal(polyline, currentConnectorPolylineObject, polylineMainObjIdx);

			if (currentConnectorPolylineObject < polyline.getConnectors().size())
				submitCurrentObjectsAndReset(intendedNextSubmit, polylineMainObjIdx);
		}
	}
}

void DrawBuffersFiller::drawHatch(
		const Hatch& hatch,
		const float32_t4& foregroundColor, 
		const float32_t4& backgroundColor,
		/* something that gets you the texture id */
		SIntendedSubmitInfo& intendedNextSubmit)
{
	drawHatch(hatch, backgroundColor /* , invalid texture id to get solid fill */, intendedNextSubmit);
	// drawHatch(hatch, foregroundColor, /* valid texture id to get foreground pattern msdf fill */, intendedNextSubmit);
}

void DrawBuffersFiller::drawHatch(
		const Hatch& hatch,
		const float32_t4& color,
		/* something that gets you the texture id */
		SIntendedSubmitInfo& intendedNextSubmit)
{
	uint32_t textureIdx = InvalidTextureIdx; // TODO
	// TODO: figure out a good way to alias every style into a single structured buffer 
	// because mainObj fragInterlock color lookup is going to be much easier not caring about where to get the color info from.
	// as of now resolve uses the mainObject.styleIdx to index into lineStyles structured buffer
	LineStyleInfo lineStyle = {};
	lineStyle.color = color;
	lineStyle.screenSpaceLineWidth = nbl::hlsl::bit_cast<float, uint32_t>(textureIdx);

	const uint32_t styleIdx = addLineStyle_SubmitIfNeeded(lineStyle, intendedNextSubmit);

	uint32_t mainObjIdx = addMainObject_SubmitIfNeeded(styleIdx, intendedNextSubmit);
	MainObject mainObjCached = *getMainObject(mainObjIdx);

	const auto sectionsCount = 1;

	uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject used in vertex shader. You can think of it as a Cage.

	while (currentObjectInSection < hatch.getHatchBoxCount())
	{
		addHatch_Internal(hatch, currentObjectInSection, mainObjIdx);
		if (currentObjectInSection < hatch.getHatchBoxCount())
			submitCurrentObjectsAndReset(intendedNextSubmit, mainObjIdx);
	}
}

void DrawBuffersFiller::finalizeAllCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit)
{
	finalizeMainObjectCopiesToGPU(intendedNextSubmit);
	finalizeGeometryCopiesToGPU(intendedNextSubmit);
	finalizeLineStyleCopiesToGPU(intendedNextSubmit);
}

uint32_t DrawBuffersFiller::addLineStyle_SubmitIfNeeded(const LineStyleInfo& lineStyle, SIntendedSubmitInfo& intendedNextSubmit)
{
	uint32_t outLineStyleIdx = addLineStyle_Internal(lineStyle);
	if (outLineStyleIdx == InvalidStyleIdx)
	{
		finalizeAllCopiesToGPU(intendedNextSubmit);
		submitDraws(intendedNextSubmit);
		resetGeometryCounters();
		resetMainObjectCounters();
		resetLineStyleCounters();
		outLineStyleIdx = addLineStyle_Internal(lineStyle);
		assert(outLineStyleIdx != InvalidStyleIdx);
	}
	return outLineStyleIdx;
}

uint32_t DrawBuffersFiller::addMainObject_SubmitIfNeeded(uint32_t styleIdx, SIntendedSubmitInfo& intendedNextSubmit)
{
	MainObject mainObject = {};
	mainObject.styleIdx = styleIdx;
	mainObject.clipProjectionAddress = getCurrentClipProjectionAddress(intendedNextSubmit);
	uint32_t outMainObjectIdx = addMainObject_Internal(mainObject);
	if (outMainObjectIdx == InvalidMainObjectIdx)
	{
		finalizeAllCopiesToGPU(intendedNextSubmit);
		submitDraws(intendedNextSubmit);

		// geometries needs to be reset because they reference draw objects and draw objects reference main objects that are now unavailable and reset
		resetGeometryCounters();
		// mainObjects needs to be reset because we submitted every previous main object
		resetMainObjectCounters();
		// getCurrentClipProjectionAddress again here because clip projection exists in the geometry buffer, and reseting geometry counters will invalidate the current clip proj and requires repush
		mainObject.clipProjectionAddress = getCurrentClipProjectionAddress(intendedNextSubmit);
		outMainObjectIdx = addMainObject_Internal(mainObject);
		assert(outMainObjectIdx != InvalidMainObjectIdx);
	}
	
	return outMainObjectIdx;
}

void DrawBuffersFiller::pushClipProjectionData(const ClipProjectionData& clipProjectionData)
{
	clipProjectionStack.push(clipProjectionData);
}

void DrawBuffersFiller::popClipProjectionData()
{
	clipProjectionStack.pop();
}

void DrawBuffersFiller::finalizeMainObjectCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit)
{
	// Copy MainObjects
	uint32_t remainingMainObjects = currentMainObjectCount - inMemMainObjectCount;
	SBufferRange<IGPUBuffer> mainObjectsRange = { sizeof(MainObject) * inMemMainObjectCount, sizeof(MainObject) * remainingMainObjects, gpuDrawBuffers.mainObjectsBuffer };
	const MainObject* srcMainObjData = reinterpret_cast<MainObject*>(cpuDrawBuffers.mainObjectsBuffer->getPointer()) + inMemMainObjectCount;
	if (mainObjectsRange.size > 0u)
		m_utilities->updateBufferRangeViaStagingBuffer(intendedNextSubmit, mainObjectsRange, srcMainObjData);
	inMemMainObjectCount = currentMainObjectCount;
}

void DrawBuffersFiller::finalizeGeometryCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit)
{
	// Copy DrawObjects
	uint32_t remainingDrawObjects = currentDrawObjectCount - inMemDrawObjectCount;
	SBufferRange<IGPUBuffer> drawObjectsRange = { sizeof(DrawObject) * inMemDrawObjectCount, sizeof(DrawObject) * remainingDrawObjects, gpuDrawBuffers.drawObjectsBuffer };
	const DrawObject* srcDrawObjData = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + inMemDrawObjectCount;
	if (drawObjectsRange.size > 0u)
		m_utilities->updateBufferRangeViaStagingBuffer(intendedNextSubmit, drawObjectsRange, srcDrawObjData);
	inMemDrawObjectCount = currentDrawObjectCount;

	// Copy GeometryBuffer
	uint64_t remainingGeometrySize = currentGeometryBufferSize - inMemGeometryBufferSize;
	SBufferRange<IGPUBuffer> geomRange = { inMemGeometryBufferSize, remainingGeometrySize, gpuDrawBuffers.geometryBuffer };
	const uint8_t* srcGeomData = reinterpret_cast<uint8_t*>(cpuDrawBuffers.geometryBuffer->getPointer()) + inMemGeometryBufferSize;
	if (geomRange.size > 0u)
		m_utilities->updateBufferRangeViaStagingBuffer(intendedNextSubmit, geomRange, srcGeomData);
	inMemGeometryBufferSize = currentGeometryBufferSize;
}

void DrawBuffersFiller::finalizeLineStyleCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit)
{
	// Copy LineStyles
	uint32_t remainingLineStyles = currentLineStylesCount - inMemLineStylesCount;
	SBufferRange<IGPUBuffer> stylesRange = { sizeof(LineStyle) * inMemLineStylesCount, sizeof(LineStyle) * remainingLineStyles, gpuDrawBuffers.lineStylesBuffer };
	const LineStyle* srcLineStylesData = reinterpret_cast<LineStyle*>(cpuDrawBuffers.lineStylesBuffer->getPointer()) + inMemLineStylesCount;
	if (stylesRange.size > 0u)
		m_utilities->updateBufferRangeViaStagingBuffer(intendedNextSubmit, stylesRange, srcLineStylesData);
	inMemLineStylesCount = currentLineStylesCount;
}

void DrawBuffersFiller::submitCurrentObjectsAndReset(SIntendedSubmitInfo& intendedNextSubmit, uint32_t mainObjectIndex)
{
	finalizeAllCopiesToGPU(intendedNextSubmit);
	submitDraws(intendedNextSubmit);

	// We reset Geometry Counters (drawObj+geometryInfos) because we're done rendering previous geometry
	// We don't reset counters for styles because we will be reusing them
	resetGeometryCounters();

	uint32_t newClipProjectionAddress = getCurrentClipProjectionAddress(intendedNextSubmit);
	// If there clip projection stack is non-empty, then it means we need to re-push the clipProjectionData (because it exists in geometry data)
	if (newClipProjectionAddress != InvalidClipProjectionAddress)
	{
		// then modify the mainObject data
		getMainObject(mainObjectIndex)->clipProjectionAddress = newClipProjectionAddress;
		// we need to modify inMemMainObjectCount to this mainObjIndex so it re-uploads the current mainObject (because we modified it)
		inMemMainObjectCount = min(inMemMainObjectCount, mainObjectIndex);
	}
}

uint32_t DrawBuffersFiller::addMainObject_Internal(const MainObject& mainObject)
{
	MainObject* mainObjsArray = reinterpret_cast<MainObject*>(cpuDrawBuffers.mainObjectsBuffer->getPointer());
	if (currentMainObjectCount >= MaxIndexableMainObjects)
		return InvalidMainObjectIdx;

	void* dst = mainObjsArray + currentMainObjectCount;
	memcpy(dst, &mainObject, sizeof(MainObject));
	uint32_t ret = currentMainObjectCount;
	currentMainObjectCount++;
	return ret;
}

uint32_t DrawBuffersFiller::addLineStyle_Internal(const LineStyleInfo& lineStyleInfo)
{
	LineStyle gpuLineStyle = lineStyleInfo.getAsGPUData();
	_NBL_DEBUG_BREAK_IF(gpuLineStyle.stipplePatternSize > LineStyle::StipplePatternMaxSize); // Oops, even after style normalization the style is too long to be in gpu mem :(
	LineStyle* stylesArray = reinterpret_cast<LineStyle*>(cpuDrawBuffers.lineStylesBuffer->getPointer());
	for (uint32_t i = 0u; i < currentLineStylesCount; ++i)
	{
		const LineStyle& itr = stylesArray[i];

		if (itr == gpuLineStyle)
			return i;
	}

	if (currentLineStylesCount >= maxLineStyles)
		return InvalidStyleIdx;

	void* dst = stylesArray + currentLineStylesCount;
	memcpy(dst, &gpuLineStyle, sizeof(LineStyle));
	return currentLineStylesCount++;
}

inline uint64_t DrawBuffersFiller::getCurrentClipProjectionAddress(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (clipProjectionStackTopAddressInGPU == InvalidClipProjectionAddress && !clipProjectionStack.empty())
		clipProjectionStackTopAddressInGPU = addClipProjectionData_SubmitIfNeeded(clipProjectionStack.top(), intendedNextSubmit);
	return clipProjectionStackTopAddressInGPU;
}

uint64_t DrawBuffersFiller::addClipProjectionData_SubmitIfNeeded(const ClipProjectionData& clipProjectionData, SIntendedSubmitInfo& intendedNextSubmit)
{
	uint64_t outClipProjectionAddress = addClipProjectionData_Internal(clipProjectionData);
	if (outClipProjectionAddress == InvalidClipProjectionAddress)
	{
		finalizeAllCopiesToGPU(intendedNextSubmit);
		submitDraws(intendedNextSubmit);

		resetGeometryCounters();
		resetMainObjectCounters();

		outClipProjectionAddress = addClipProjectionData_Internal(clipProjectionData);
		assert(outClipProjectionAddress != InvalidClipProjectionAddress);
	}
	return outClipProjectionAddress;
}

uint64_t DrawBuffersFiller::addClipProjectionData_Internal(const ClipProjectionData& clipProjectionData)
{
	const uint64_t maxGeometryBufferClipProjData = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(ClipProjectionData);
	if (maxGeometryBufferClipProjData <= 0)
		return InvalidClipProjectionAddress;
	
	void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
	memcpy(dst, &clipProjectionData, sizeof(ClipProjectionData));

	const uint64_t ret = currentGeometryBufferSize + geometryBufferAddress;
	currentGeometryBufferSize += sizeof(ClipProjectionData);
	return ret;
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

// TODO[Prezmek]: another function named addPolylineConnectors_Internal and you pass a Range<PolylineConnectorInfo>, uint32_t currentPolylineConnectorObj, uint32_t mainObjIdx
// And implement it similar to addLines/QuadBeziers_Internal which is check how much memory is left and how many PolylineConnectors you can fit into the current geometry and drawobj memory left and return to the drawPolylinefunction
void DrawBuffersFiller::addPolylineConnectors_Internal(const CPolylineBase& polyline, uint32_t& currentPolylineConnectorObj, uint32_t mainObjIdx)
{
	const auto maxGeometryBufferConnectors = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(PolylineConnector);

	uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
	uploadableObjects = min(uploadableObjects, maxGeometryBufferConnectors);
	uploadableObjects = min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

	const auto connectorCount = polyline.getConnectors().size();
	const auto remainingObjects = connectorCount - currentPolylineConnectorObj;

	const uint32_t objectsToUpload = min(uploadableObjects, remainingObjects);

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

	uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
	uploadableObjects = min(uploadableObjects, maxGeometryBufferLines);
	uploadableObjects = min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

	const auto lineCount = section.count;
	const auto remainingObjects = lineCount - currentObjectInSection;
	uint32_t objectsToUpload = min(uploadableObjects, remainingObjects);

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
	assert(section.type == ObjectType::QUAD_BEZIER);

	const auto maxGeometryBufferBeziers = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(QuadraticBezierInfo);
	
	uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
	uploadableObjects = min(uploadableObjects, maxGeometryBufferBeziers);
	uploadableObjects = min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
	uploadableObjects /= CagesPerQuadBezier;

	const auto beziersCount = section.count;
	const auto remainingObjects = beziersCount - currentObjectInSection;
	uint32_t objectsToUpload = min(uploadableObjects, remainingObjects);

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
	const auto maxGeometryBufferHatchBoxes = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(Hatch::CurveHatchBox);
	
	uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
	uploadableObjects = min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
	uploadableObjects = min(uploadableObjects, maxGeometryBufferHatchBoxes);

	uint32_t remainingObjects = hatch.getHatchBoxCount() - currentObjectInSection;
	uploadableObjects = min(uploadableObjects, remainingObjects);

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
	currentDrawObjectCount += uploadableObjects;
	currentObjectInSection += uploadableObjects;
}