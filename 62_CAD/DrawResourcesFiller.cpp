#include "DrawResourcesFiller.h"

DrawResourcesFiller::DrawResourcesFiller()
{}

DrawResourcesFiller::DrawResourcesFiller(smart_refctd_ptr<IUtilities>&& utils, IQueue* copyQueue, core::smart_refctd_ptr<system::ILogger>&& logger) :
	m_utilities(std::move(utils)),
	m_copyQueue(copyQueue),
	m_logger(std::move(logger))
{
	imagesCache = std::unique_ptr<ImagesCache>(new ImagesCache(ImagesBindingArraySize));
}

// function is called when buffer is filled and we should submit draws and clear the buffers and continue filling

void DrawResourcesFiller::setSubmitDrawsFunction(const SubmitFunc& func)
{
	submitDraws = func;
}

// DrawResourcesFiller needs to access these in order to allocate GPUImages and write the to their correct descriptor set binding
void DrawResourcesFiller::setTexturesDescriptorSetAndBinding(core::smart_refctd_ptr<video::IGPUDescriptorSet>&& descriptorSet, uint32_t binding)
{
	imagesArrayBinding = binding;
	suballocatedDescriptorSet = core::make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(descriptorSet));
}

bool DrawResourcesFiller::allocateDrawResources(ILogicalDevice* logicalDevice, size_t requiredImageMemorySize, size_t requiredBufferMemorySize)
{
	// single memory allocation sectioned into images+buffers (images start at offset=0)
	const size_t adjustedImagesMemorySize = core::alignUp(requiredImageMemorySize, GPUStructsMaxNaturalAlignment);
	const size_t adjustedBuffersMemorySize = core::max(requiredBufferMemorySize, getMinimumRequiredResourcesBufferSize());
	const size_t totalResourcesSize = adjustedImagesMemorySize + adjustedBuffersMemorySize;

	IGPUBuffer::SCreationParams resourcesBufferCreationParams = {};
	resourcesBufferCreationParams.size = adjustedBuffersMemorySize;
	resourcesBufferCreationParams.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INDEX_BUFFER_BIT;
	resourcesGPUBuffer = logicalDevice->createBuffer(std::move(resourcesBufferCreationParams));
	resourcesGPUBuffer->setObjectDebugName("drawResourcesBuffer");

	IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = resourcesGPUBuffer->getMemoryReqs();
	
	nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements gpuBufferMemoryReqs = resourcesGPUBuffer->getMemoryReqs();
	const bool memoryRequirementsMatch =
		(logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits() & gpuBufferMemoryReqs.memoryTypeBits) != 0 && // should have device local memory compatible
		(gpuBufferMemoryReqs.requiresDedicatedAllocation == false); // should not require dedicated allocation

	if (!memoryRequirementsMatch)
	{
		m_logger.log("Shouldn't happen: Buffer Memory Requires Dedicated Allocation or can't biind to device local memory.", nbl::system::ILogger::ELL_ERROR);
		return false;
	}
	
	const auto& memoryProperties = logicalDevice->getPhysicalDevice()->getMemoryProperties();

	uint32_t memoryTypeIdx = ~0u;

	video::IDeviceMemoryAllocator::SAllocation allocation = {};
	for (uint32_t i = 0u; i < memoryProperties.memoryTypeCount; ++i)
	{
		if (memoryProperties.memoryTypes[i].propertyFlags.hasFlags(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT))
		{
			memoryTypeIdx = i;

			IDeviceMemoryAllocator::SAllocateInfo allocationInfo =
			{
				.size = totalResourcesSize,
				.flags = IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_DEVICE_ADDRESS_BIT, // for the buffers
				.memoryTypeIndex = memoryTypeIdx,
				.dedication = nullptr,
			};

			allocation = logicalDevice->allocate(allocationInfo);
			
			if (allocation.isValid())
				break;
		}
	}

	if (memoryTypeIdx == ~0u)
	{
		m_logger.log("allocateResourcesBuffer: no device local memory type found!", nbl::system::ILogger::ELL_ERROR);
		return false;
	}

	if (!allocation.isValid())
		return false;

	imagesMemoryArena = {
		.memory = allocation.memory,
		.offset = allocation.offset,
	};

	buffersMemoryArena = {
		.memory = allocation.memory,
		.offset = core::alignUp(allocation.offset + adjustedImagesMemorySize, GPUStructsMaxNaturalAlignment), // first natural alignment after images section of the memory allocation
	};

	imagesMemorySubAllocator = core::make_smart_refctd_ptr<ImagesMemorySubAllocator>(adjustedImagesMemorySize);

	video::ILogicalDevice::SBindBufferMemoryInfo bindBufferMemory = {
		.buffer = resourcesGPUBuffer.get(),
		.binding = {
			.memory = buffersMemoryArena.memory.get(),
			.offset  = buffersMemoryArena.offset,
		}
	};

	if (!logicalDevice->bindBufferMemory(1, &bindBufferMemory))
	{
		m_logger.log("DrawResourcesFiller::allocateDrawResources, bindBufferMemory failed.", nbl::system::ILogger::ELL_ERROR);
		return false;
	}

	return true;
}

bool DrawResourcesFiller::allocateDrawResourcesWithinAvailableVRAM(ILogicalDevice* logicalDevice, size_t maxImageMemorySize, size_t maxBufferMemorySize, uint32_t reductionPercent, uint32_t maxTries)
{
	const size_t minimumAcceptableSize = core::max(MinimumDrawResourcesMemorySize, getMinimumRequiredResourcesBufferSize());

	size_t currentBufferSize = maxBufferMemorySize;
	size_t currentImageSize = maxImageMemorySize;
	const size_t totalInitialSize = currentBufferSize + currentImageSize;

	// If initial size is less than minimum acceptable then increase the buffer and image size to sum up to minimumAcceptableSize with image:buffer ratios preserved
	if (totalInitialSize < minimumAcceptableSize)
	{
		// Preserve ratio: R = buffer / (buffer + image)
		// scaleFactor = minimumAcceptableSize / totalInitialSize;
		const double scaleFactor = static_cast<double>(minimumAcceptableSize) / totalInitialSize;
		currentBufferSize = static_cast<size_t>(currentBufferSize * scaleFactor);
		currentImageSize = minimumAcceptableSize - currentBufferSize; // ensures exact sum
	}

	uint32_t numTries = 0u;
	while ((currentBufferSize + currentImageSize) >= minimumAcceptableSize && numTries < maxTries)
	{
		if (allocateDrawResources(logicalDevice, currentBufferSize, currentImageSize))
			return true;

		currentBufferSize = (currentBufferSize * (100 - reductionPercent)) / 100;
		currentImageSize = (currentImageSize * (100 - reductionPercent)) / 100;
		numTries++;
		m_logger.log("Allocation of memory for images(%zu) and buffers(%zu) failed; Reducing allocation size by %u%% and retrying...", system::ILogger::ELL_WARNING, currentImageSize, currentBufferSize, reductionPercent);
	}

	m_logger.log("All attempts to allocate memory for images(%zu) and buffers(%zu) failed.", system::ILogger::ELL_ERROR, currentImageSize, currentBufferSize);
	return false;
}

bool DrawResourcesFiller::allocateMSDFTextures(ILogicalDevice* logicalDevice, uint32_t maxMSDFs, uint32_t2 msdfsExtent)
{
	// TODO: Make this function failable and report insufficient memory
	asset::E_FORMAT msdfFormat = MSDFTextureFormat;
	asset::VkExtent3D MSDFsExtent = { msdfsExtent.x, msdfsExtent.y, 1u }; 
	if (maxMSDFs > logicalDevice->getPhysicalDevice()->getLimits().maxImageArrayLayers)
	{
		m_logger.log("requested maxMSDFs is greater than maxImageArrayLayers. lowering the limit...", nbl::system::ILogger::ELL_WARNING);
		maxMSDFs = logicalDevice->getPhysicalDevice()->getLimits().maxImageArrayLayers;
	}
	
	IPhysicalDevice::SImageFormatPromotionRequest promotionRequest = {};
	promotionRequest.originalFormat = msdfFormat;
	promotionRequest.usages = {};
	promotionRequest.usages.sampledImage = true;
	msdfFormat = logicalDevice->getPhysicalDevice()->promoteImageFormat(promotionRequest, IGPUImage::TILING::OPTIMAL);

	{
		IGPUImage::SCreationParams imgInfo;
		imgInfo.format = msdfFormat;
		imgInfo.type = IGPUImage::ET_2D;
		imgInfo.extent = MSDFsExtent;
		imgInfo.mipLevels = MSDFMips; 
		imgInfo.arrayLayers = maxMSDFs;
		imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
		imgInfo.flags = asset::IImage::E_CREATE_FLAGS::ECF_NONE;
		imgInfo.usage = asset::IImage::EUF_SAMPLED_BIT | asset::IImage::EUF_TRANSFER_DST_BIT;
		imgInfo.tiling = IGPUImage::TILING::OPTIMAL;

		auto image = logicalDevice->createImage(std::move(imgInfo));
		auto imageMemReqs = image->getMemoryReqs();
		imageMemReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		const auto allocation = logicalDevice->allocate(imageMemReqs, image.get());

		if (!allocation.isValid())
			return false;

		image->setObjectDebugName("MSDFs Texture Array");

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.image = std::move(image);
		imgViewInfo.format = msdfFormat;
		imgViewInfo.viewType = IGPUImageView::ET_2D_ARRAY;
		imgViewInfo.flags = IGPUImageView::E_CREATE_FLAGS::ECF_NONE;
		imgViewInfo.subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		imgViewInfo.subresourceRange.baseArrayLayer = 0u;
		imgViewInfo.subresourceRange.baseMipLevel = 0u;
		imgViewInfo.subresourceRange.layerCount = maxMSDFs;
		imgViewInfo.subresourceRange.levelCount = MSDFMips;

		msdfTextureArray = logicalDevice->createImageView(std::move(imgViewInfo));
	}

	if (!msdfTextureArray)
		return false;

	msdfLRUCache = std::unique_ptr<MSDFsLRUCache>(new MSDFsLRUCache(maxMSDFs));
	msdfTextureArrayIndexAllocator = core::make_smart_refctd_ptr<IndexAllocator>(core::smart_refctd_ptr<ILogicalDevice>(logicalDevice), maxMSDFs);
	msdfImagesState.resize(maxMSDFs);
	return true;
}

void DrawResourcesFiller::drawPolyline(const CPolylineBase& polyline, const LineStyleInfo& lineStyleInfo, SIntendedSubmitInfo& intendedNextSubmit)
{
	if (!lineStyleInfo.isVisible())
		return;

	setActiveLineStyle(lineStyleInfo);
	
	beginMainObject(MainObjectType::POLYLINE, TransformationType::TT_NORMAL);
	drawPolyline(polyline, intendedNextSubmit);
	endMainObject();
}

void DrawResourcesFiller::drawFixedGeometryPolyline(const CPolylineBase& polyline, const LineStyleInfo& lineStyleInfo, const float64_t3x3& transformation, TransformationType transformationType, SIntendedSubmitInfo& intendedNextSubmit)
{
	if (!lineStyleInfo.isVisible())
		return;

	setActiveLineStyle(lineStyleInfo);
	
	pushCustomProjection(getFixedGeometryFinalTransformationMatrix(transformation, transformationType));
	beginMainObject(MainObjectType::POLYLINE, transformationType);
	drawPolyline(polyline, intendedNextSubmit);
	endMainObject();
	popCustomProjection();
}

void DrawResourcesFiller::drawPolyline(const CPolylineBase& polyline, SIntendedSubmitInfo& intendedNextSubmit)
{
	uint32_t mainObjectIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
	if (mainObjectIdx == InvalidMainObjectIdx)
	{
		m_logger.log("drawPolyline: acquireActiveMainObjectIndex returned invalid index", nbl::system::ILogger::ELL_ERROR);
		assert(false);
		return;
	}
	
	const auto sectionsCount = polyline.getSectionsCount();

	uint32_t currentSectionIdx = 0u;
	uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject used in vertex shader. You can think of it as a Cage.

	while (currentSectionIdx < sectionsCount)
	{
		const auto& currentSection = polyline.getSectionInfoAt(currentSectionIdx);
		addPolylineObjects_Internal(polyline, currentSection, currentObjectInSection, mainObjectIdx);

		if (currentObjectInSection >= currentSection.count)
		{
			currentSectionIdx++;
			currentObjectInSection = 0u;
		}
		else
			submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjectIdx);
	}

	if (!polyline.getConnectors().empty())
	{
		uint32_t currentConnectorPolylineObject = 0u;
		while (currentConnectorPolylineObject < polyline.getConnectors().size())
		{
			addPolylineConnectors_Internal(polyline, currentConnectorPolylineObject, mainObjectIdx);

			if (currentConnectorPolylineObject < polyline.getConnectors().size())
				submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjectIdx);
		}
	}
}

void DrawResourcesFiller::drawTriangleMesh(
	const CTriangleMesh& mesh,
	const DTMSettingsInfo& dtmSettingsInfo,
	SIntendedSubmitInfo& intendedNextSubmit)
{
	flushDrawObjects(); // flushes draw call construction of any possible draw objects before dtm, because currently we're sepaerating dtm draw calls from drawObj draw calls

	setActiveDTMSettings(dtmSettingsInfo);
	beginMainObject(MainObjectType::DTM);

	uint32_t mainObjectIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
	if (mainObjectIdx == InvalidMainObjectIdx)
	{
		m_logger.log("drawTriangleMesh: acquireActiveMainObjectIndex returned invalid index", nbl::system::ILogger::ELL_ERROR);
		assert(false);
		return;
	}

	DrawCallData drawCallData = {}; 
	drawCallData.isDTMRendering = true;

	ICPUBuffer::SCreationParams geometryBuffParams;

	// concatenate the index and vertex buffer into the geometry buffer
	const auto& indexBuffer = mesh.getIndices();
	const auto& vertexBuffer = mesh.getVertices();
	assert(indexBuffer.size() == vertexBuffer.size()); // We don't have any vertex re-use due to other limitations at the moemnt.
	

	const uint32_t numTriangles = indexBuffer.size() / 3u;
	uint32_t trianglesUploaded = 0;
	while (trianglesUploaded < numTriangles)
	{
		const size_t remainingResourcesSize = calculateRemainingResourcesSize();
		const uint32_t maxUploadableVertices = remainingResourcesSize / (sizeof(CTriangleMesh::vertex_t) + sizeof(CTriangleMesh::index_t));
		const uint32_t maxUploadableTriangles = maxUploadableVertices / 3u;
		const uint32_t remainingTrianglesToUpload = numTriangles - trianglesUploaded;
		const uint32_t trianglesToUpload = core::min(remainingTrianglesToUpload, maxUploadableTriangles);
		const size_t vtxBuffByteSize = trianglesToUpload * 3u * sizeof(CTriangleMesh::vertex_t);
		const size_t indexBuffByteSize = trianglesToUpload * 3u * sizeof(CTriangleMesh::index_t);
		const size_t trianglesToUploadByteSize = vtxBuffByteSize + indexBuffByteSize;

		// Copy VertexBuffer
		size_t geometryBufferOffset = resourcesCollection.geometryInfo.increaseSizeAndGetOffset(trianglesToUploadByteSize, alignof(CTriangleMesh::vertex_t));
		void* dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
		// the actual bda address will be determined only after all copies are finalized, later we will do += `baseBDAAddress + geometryInfo.bufferOffset`
		// the - is a small hack because index buffer grows but vertex buffer needs to start from 0, remove that once we either get rid of the index buffer or implement an algorithm that can have vertex reuse
		drawCallData.dtm.triangleMeshVerticesBaseAddress = geometryBufferOffset - (sizeof(CTriangleMesh::vertex_t) * trianglesUploaded * 3); 
		memcpy(dst, &vertexBuffer[trianglesUploaded * 3u], vtxBuffByteSize);
		geometryBufferOffset += vtxBuffByteSize; 

		// Copy IndexBuffer
		dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
		drawCallData.dtm.indexBufferOffset = geometryBufferOffset;
		memcpy(dst, &indexBuffer[trianglesUploaded * 3u], indexBuffByteSize);
		geometryBufferOffset += indexBuffByteSize;
		
		trianglesUploaded += trianglesToUpload;
		
		drawCallData.dtm.triangleMeshMainObjectIndex = mainObjectIdx;
		drawCallData.dtm.indexCount = trianglesToUpload * 3u;
		drawCalls.push_back(drawCallData);

		//if (trianglesUploaded == 0u)
		//{
		//	m_logger.log("drawTriangleMesh: not enough vram allocation for a single triangle!", nbl::system::ILogger::ELL_ERROR);
		//	assert(false);
		//	break;
		//}

		// Requires Auto-Submit If All Triangles of the Mesh couldn't fit into Memory
		if (trianglesUploaded < numTriangles)
			submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjectIdx);
	}

	endMainObject();
}

// TODO[Erfan]: Makes more sense if parameters are: solidColor + fillPattern + patternColor
void DrawResourcesFiller::drawHatch(
		const Hatch& hatch,
		const float32_t4& foregroundColor, 
		const float32_t4& backgroundColor,
		const HatchFillPattern fillPattern,
		SIntendedSubmitInfo& intendedNextSubmit)
{
	// TODO[Optimization Idea]: don't draw hatch twice, we now have color storage buffer and we can treat rendering hatches like a procedural texture (requires 2 colors so no more abusing of linestyle for hatches)

	// if backgroundColor is visible
	drawHatch(hatch, backgroundColor, intendedNextSubmit);
	// if foregroundColor is visible
	drawHatch(hatch, foregroundColor, fillPattern, intendedNextSubmit);
}

void DrawResourcesFiller::drawHatch(
		const Hatch& hatch,
		const float32_t4& color,
		const HatchFillPattern fillPattern,
		SIntendedSubmitInfo& intendedNextSubmit)
{
	drawHatch_impl(hatch, color, fillPattern, intendedNextSubmit);
}

void DrawResourcesFiller::drawHatch(const Hatch& hatch, const float32_t4& color, SIntendedSubmitInfo& intendedNextSubmit)
{
	drawHatch(hatch, color, HatchFillPattern::SOLID_FILL, intendedNextSubmit);
}

void DrawResourcesFiller::drawFixedGeometryHatch(
		const Hatch& hatch,
		const float32_t4& foregroundColor,
		const float32_t4& backgroundColor,
		const HatchFillPattern fillPattern,
		const float64_t3x3& transformation,
		TransformationType transformationType, 
		SIntendedSubmitInfo& intendedNextSubmit)
{
	// TODO[Optimization Idea]: don't draw hatch twice, we now have color storage buffer and we can treat rendering hatches like a procedural texture (requires 2 colors so no more abusing of linestyle for hatches)

	// if backgroundColor is visible
	drawFixedGeometryHatch(hatch, backgroundColor, transformation, transformationType, intendedNextSubmit);
	// if foregroundColor is visible
	drawFixedGeometryHatch(hatch, foregroundColor, fillPattern, transformation, transformationType, intendedNextSubmit);
}

void DrawResourcesFiller::drawFixedGeometryHatch(
	const Hatch& hatch,
	const float32_t4& color,
	const HatchFillPattern fillPattern,
	const float64_t3x3& transformation,
	TransformationType transformationType,
	SIntendedSubmitInfo& intendedNextSubmit)
{
	pushCustomProjection(getFixedGeometryFinalTransformationMatrix(transformation, transformationType));
	drawHatch_impl(hatch, color, fillPattern, intendedNextSubmit, transformationType);
	popCustomProjection();
}

void DrawResourcesFiller::drawFixedGeometryHatch(
	const Hatch& hatch,
	const float32_t4& color,
	const float64_t3x3& transformation,
	TransformationType transformationType,
	SIntendedSubmitInfo& intendedNextSubmit)
{
	drawFixedGeometryHatch(hatch, color, HatchFillPattern::SOLID_FILL, transformation, transformationType, intendedNextSubmit);
}

void DrawResourcesFiller::drawHatch_impl(
	const Hatch& hatch,
	const float32_t4& color,
	const HatchFillPattern fillPattern,
	SIntendedSubmitInfo& intendedNextSubmit,
	TransformationType transformationType)
{
	if (color.a == 0.0f) // not visible
		return;

	uint32_t textureIdx = InvalidTextureIndex;
	if (fillPattern != HatchFillPattern::SOLID_FILL)
	{
		MSDFInputInfo msdfInfo = MSDFInputInfo(fillPattern);
		textureIdx = getMSDFIndexFromInputInfo(msdfInfo, intendedNextSubmit);
		if (textureIdx == InvalidTextureIndex)
			textureIdx = addMSDFTexture(msdfInfo, getHatchFillPatternMSDF(fillPattern), intendedNextSubmit);
		_NBL_DEBUG_BREAK_IF(textureIdx == InvalidTextureIndex); // probably getHatchFillPatternMSDF returned nullptr
	}

	LineStyleInfo lineStyle = {};
	lineStyle.color = color;
	lineStyle.screenSpaceLineWidth = nbl::hlsl::bit_cast<float, uint32_t>(textureIdx);

	setActiveLineStyle(lineStyle);
	beginMainObject(MainObjectType::HATCH, transformationType);

	uint32_t mainObjectIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
	uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject. You can think of it as a Cage.
	while (currentObjectInSection < hatch.getHatchBoxCount())
	{
		addHatch_Internal(hatch, currentObjectInSection, mainObjectIdx);
		if (currentObjectInSection < hatch.getHatchBoxCount())
			submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjectIdx);
	}

	endMainObject();
}

void DrawResourcesFiller::drawFontGlyph(
		nbl::ext::TextRendering::FontFace* fontFace,
		uint32_t glyphIdx,
		float64_t2 topLeft,
		float32_t2 dirU,
		float32_t  aspectRatio,
		float32_t2 minUV,
		SIntendedSubmitInfo& intendedNextSubmit)
{
	uint32_t textureIdx = InvalidTextureIndex;
	const MSDFInputInfo msdfInput = MSDFInputInfo(fontFace->getHash(), glyphIdx);
	textureIdx = getMSDFIndexFromInputInfo(msdfInput, intendedNextSubmit);
	if (textureIdx == InvalidTextureIndex)
		textureIdx = addMSDFTexture(msdfInput, getGlyphMSDF(fontFace, glyphIdx), intendedNextSubmit);

	uint32_t mainObjIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
	if (mainObjIdx == InvalidMainObjectIdx)
	{
		m_logger.log("drawFontGlyph: acquireActiveMainObjectIndex returned invalid index", nbl::system::ILogger::ELL_ERROR);
		assert(false);
		return;
	}
	
	if (textureIdx != InvalidTextureIndex)
	{
		GlyphInfo glyphInfo = GlyphInfo(topLeft, dirU, aspectRatio, textureIdx, minUV);
		if (!addFontGlyph_Internal(glyphInfo, mainObjIdx))
		{
			// single font glyph couldn't fit into memory to push to gpu, so we submit rendering current objects and reset geometry buffer and draw objects
			submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjIdx);
			const bool success = addFontGlyph_Internal(glyphInfo, mainObjIdx);
			if (!success)
			{
				m_logger.log("addFontGlyph_Internal failed, even after overflow-submission, this is irrecoverable.", nbl::system::ILogger::ELL_ERROR);
				assert(false);
			}
		}
	}
	else
	{
		m_logger.log("drawFontGlyph: textureIdx is invalid.", nbl::system::ILogger::ELL_ERROR);
		_NBL_DEBUG_BREAK_IF(true);
	}
}

bool DrawResourcesFiller::ensureStaticImageAvailability(const StaticImageInfo& staticImage, SIntendedSubmitInfo& intendedNextSubmit)
{
	// Try inserting or updating the image usage in the cache.
	// If the image is already present, updates its semaphore value.
	auto evictCallback = [&](image_id imageID, const CachedImageRecord& evicted) { evictImage_SubmitIfNeeded(imageID, evicted, intendedNextSubmit); };
	CachedImageRecord* cachedImageRecord = imagesCache->insert(staticImage.imageID, intendedNextSubmit.getFutureScratchSemaphore().value, evictCallback);
	cachedImageRecord->lastUsedFrameIndex = currentFrameIndex; // in case there was an eviction + auto-submit, we need to update AGAIN

	if (cachedImageRecord->arrayIndex != InvalidTextureIndex && staticImage.forceUpdate)
	{
		// found in cache, and we want to force new data into the image
		if (cachedImageRecord->staticCPUImage)
		{
			const auto cachedImageParams = cachedImageRecord->staticCPUImage->getCreationParameters();
			const auto newImageParams = staticImage.cpuImage->getCreationParameters();
			const bool needsRecreation = newImageParams != cachedImageParams;
			if (needsRecreation)
			{
				// call the eviction callback so the currently cached imageID gets eventually deallocated from memory arena along with it's allocated array slot from the suballocated descriptor set
				evictCallback(staticImage.imageID, *cachedImageRecord);
					
				// Instead of erasing and inserting the imageID into the cache, we just reset it, so the next block of code goes into array index allocation + creating our new image
				// imagesCache->erase(imageID);
				// cachedImageRecord = imagesCache->insert(imageID, intendedNextSubmit.getFutureScratchSemaphore().value, evictCallback);
				*cachedImageRecord = CachedImageRecord(currentFrameIndex);
			}
			else
			{
				// Doesn't need image recreation, we'll use the same array index in descriptor set + the same bound memory.
				// reset it's state + update the cpu image used for copying.
				cachedImageRecord->state = ImageState::CREATED_AND_MEMORY_BOUND; 
				cachedImageRecord->staticCPUImage = staticImage.cpuImage;
			}
		}
		else
		{
			m_logger.log("found static image has empty cpu image, shouldn't happen", nbl::system::ILogger::ELL_ERROR);
		}
	}

	// if cachedImageRecord->index was not InvalidTextureIndex then it means we had a cache hit and updated the value of our sema
	// in which case we don't queue anything for upload, and return the idx
	if (cachedImageRecord->arrayIndex == InvalidTextureIndex)
	{
		// This is a new image (cache miss). Allocate a descriptor index for it.
		cachedImageRecord->arrayIndex = video::SubAllocatedDescriptorSet::AddressAllocator::invalid_address;
		// Blocking allocation attempt; if the descriptor pool is exhausted, this may stall.
		suballocatedDescriptorSet->multi_allocate(std::chrono::time_point<std::chrono::steady_clock>::max(), imagesArrayBinding, 1u, &cachedImageRecord->arrayIndex); // if the prev submit causes DEVICE_LOST then we'll get a deadlock here since we're using max timepoint

		if (cachedImageRecord->arrayIndex != video::SubAllocatedDescriptorSet::AddressAllocator::invalid_address)
		{
			auto* device = m_utilities->getLogicalDevice();
			auto* physDev = m_utilities->getLogicalDevice()->getPhysicalDevice();

			IGPUImage::SCreationParams imageParams = {};
			imageParams = staticImage.cpuImage->getCreationParameters();
			imageParams.usage |= IGPUImage::EUF_TRANSFER_DST_BIT|IGPUImage::EUF_SAMPLED_BIT;
			// promote format because RGB8 and friends don't actually exist in HW
			{
				const IPhysicalDevice::SImageFormatPromotionRequest request = {
					.originalFormat = imageParams.format,
					.usages = IPhysicalDevice::SFormatImageUsages::SUsage(imageParams.usage)
				};
				imageParams.format = physDev->promoteImageFormat(request,imageParams.tiling);
			}

			// Attempt to create a GPU image and image view for this texture.
			ImageAllocateResults allocResults = tryCreateAndAllocateImage_SubmitIfNeeded(imageParams, staticImage.imageViewFormatOverride, intendedNextSubmit, std::to_string(staticImage.imageID));

			if (allocResults.isValid())
			{
				cachedImageRecord->type = ImageType::STATIC;
				cachedImageRecord->state = ImageState::CREATED_AND_MEMORY_BOUND;
				cachedImageRecord->lastUsedFrameIndex = currentFrameIndex; // there was an eviction + auto-submit, we need to update AGAIN
				cachedImageRecord->allocationOffset = allocResults.allocationOffset;
				cachedImageRecord->allocationSize = allocResults.allocationSize;
				cachedImageRecord->gpuImageView = allocResults.gpuImageView;
				cachedImageRecord->staticCPUImage = staticImage.cpuImage;
				cachedImageRecord->georeferencedImageState = nullptr;
			}
			else
			{
				// All attempts to try create the GPU image and its corresponding view have failed.
				// Most likely cause: insufficient GPU memory or unsupported image parameters.
				m_logger.log("ensureStaticImageAvailability failed, likely due to low VRAM.", nbl::system::ILogger::ELL_ERROR);
				_NBL_DEBUG_BREAK_IF(true);

				if (cachedImageRecord->allocationOffset != ImagesMemorySubAllocator::InvalidAddress)
				{
					// We previously successfully create and allocated memory for the Image
					// but failed to bind and create image view
					// It's crucial to deallocate the offset+size form our images memory suballocator
					imagesMemorySubAllocator->deallocate(cachedImageRecord->allocationOffset, cachedImageRecord->allocationSize);
				}

				if (cachedImageRecord->arrayIndex != InvalidTextureIndex)
				{
					// We previously allocated a descriptor index, but failed to create a usable GPU image.
					// It's crucial to deallocate this index to avoid leaks and preserve descriptor pool space.
					// No semaphore wait needed here, as the GPU never got to use this slot.
					suballocatedDescriptorSet->multi_deallocate(imagesArrayBinding, 1u, &cachedImageRecord->arrayIndex, {});
					cachedImageRecord->arrayIndex = InvalidTextureIndex;
				}

				// erase the entry we failed to allocate an image for, no need for `evictImage_SubmitIfNeeded`, because it didn't get to be used in any submit to defer it's memory and index deallocation
				imagesCache->erase(staticImage.imageID);
			}
		}
		else
		{
			m_logger.log("ensureStaticImageAvailability failed index allocation. shouldn't have happened.", nbl::system::ILogger::ELL_ERROR);
			cachedImageRecord->arrayIndex = InvalidTextureIndex;
		}
	}
	
	
	// cached or just inserted, we update the lastUsedFrameIndex
	cachedImageRecord->lastUsedFrameIndex = currentFrameIndex;

	assert(cachedImageRecord->arrayIndex != InvalidTextureIndex); // shouldn't happen, because we're using LRU cache, so worst case eviction will happen + multi-deallocate and next next multi_allocate should definitely succeed
	return cachedImageRecord->arrayIndex != InvalidTextureIndex;
}

bool DrawResourcesFiller::ensureMultipleStaticImagesAvailability(std::span<StaticImageInfo> staticImages, SIntendedSubmitInfo& intendedNextSubmit)
{
	if (staticImages.size() > ImagesBindingArraySize)
		return false;

	for (auto& staticImage : staticImages)
	{
		if (!ensureStaticImageAvailability(staticImage, intendedNextSubmit))
			return false; // failed ensuring a single staticImage is available, shouldn't happen unless the image is larger than the memory arena allocated for images.
	}
	for (auto& staticImage : staticImages)
	{
		if (imagesCache->peek(staticImage.imageID) == nullptr)
			return false; // this means one of the images evicted another, most likely due to VRAM limitations not all images can be resident all at once.
	}
	return true;
}

bool DrawResourcesFiller::ensureGeoreferencedImageAvailability_AllocateIfNeeded(image_id imageID, GeoreferencedImageParams&& params, SIntendedSubmitInfo& intendedNextSubmit)
{
	auto* device = m_utilities->getLogicalDevice();
	auto* physDev = m_utilities->getLogicalDevice()->getPhysicalDevice();

	// Try inserting or updating the image usage in the cache.
	// If the image is already present, updates its semaphore value.
	auto evictCallback = [&](image_id imageID, const CachedImageRecord& evicted) { evictImage_SubmitIfNeeded(imageID, evicted, intendedNextSubmit); };
	CachedImageRecord* cachedImageRecord = imagesCache->insert(imageID, intendedNextSubmit.getFutureScratchSemaphore().value, evictCallback);

	// TODO: Function call that gets you image creaation params based on georeferencedImageParams (extents and mips and whatever), it will also get you the GEOREFERENCED TYPE
	IGPUImage::SCreationParams imageCreationParams = {};
	ImageType imageType = determineGeoreferencedImageCreationParams(imageCreationParams, params);

	// imageParams = cpuImage->getCreationParameters();
	imageCreationParams.usage |= IGPUImage::EUF_TRANSFER_DST_BIT|IGPUImage::EUF_SAMPLED_BIT;
	// promote format because RGB8 and friends don't actually exist in HW
	{
		const IPhysicalDevice::SImageFormatPromotionRequest request = {
			.originalFormat = imageCreationParams.format,
			.usages = IPhysicalDevice::SFormatImageUsages::SUsage(imageCreationParams.usage)
		};
		imageCreationParams.format = physDev->promoteImageFormat(request,imageCreationParams.tiling);
	}
	
	// if cachedImageRecord->index was not InvalidTextureIndex then it means we had a cache hit and updated the value of our sema
	// But we need to check if the cached image needs resizing/recreation.
	if (cachedImageRecord->arrayIndex != InvalidTextureIndex)
	{
		// found in cache, but does it require resize? recreation?
		if (cachedImageRecord->gpuImageView)
		{
			auto imgViewParams = cachedImageRecord->gpuImageView->getCreationParameters();
			if (imgViewParams.image)
			{
				const auto cachedParams = static_cast<asset::IImage::SCreationParams>(imgViewParams.image->getCreationParameters());
				const auto cachedImageType = cachedImageRecord->type;
				// image type and creation params (most importantly extent and format) should match, otherwise we evict, recreate and re-pus
				const auto currentParams = static_cast<asset::IImage::SCreationParams>(imageCreationParams);
				const bool needsRecreation = cachedImageType != imageType || cachedParams != currentParams;
				if (needsRecreation)
				{
					// call the eviction callback so the currently cached imageID gets eventually deallocated from memory arena.
					evictCallback(imageID, *cachedImageRecord);
					
					// instead of erasing and inserting the imageID into the cache, we just reset it, so the next block of code goes into array index allocation + creating our new image
					*cachedImageRecord = CachedImageRecord(currentFrameIndex);
					// imagesCache->erase(imageID);
					// cachedImageRecord = imagesCache->insert(imageID, intendedNextSubmit.getFutureScratchSemaphore().value, evictCallback);
				}
			}
			else
			{
				m_logger.log("Cached georeferenced image has invalid gpu image.", nbl::system::ILogger::ELL_ERROR);
			}
		}
		else
		{
			m_logger.log("Cached georeferenced image has invalid gpu image view.", nbl::system::ILogger::ELL_ERROR);
		}
	}

	// in which case we don't queue anything for upload, and return the idx
	if (cachedImageRecord->arrayIndex == InvalidTextureIndex)
	{
		// This is a new image (cache miss). Allocate a descriptor index for it.
		cachedImageRecord->arrayIndex = video::SubAllocatedDescriptorSet::AddressAllocator::invalid_address;
		// Blocking allocation attempt; if the descriptor pool is exhausted, this may stall.
		suballocatedDescriptorSet->multi_allocate(std::chrono::time_point<std::chrono::steady_clock>::max(), imagesArrayBinding, 1u, &cachedImageRecord->arrayIndex); // if the prev submit causes DEVICE_LOST then we'll get a deadlock here since we're using max timepoint

		if (cachedImageRecord->arrayIndex != video::SubAllocatedDescriptorSet::AddressAllocator::invalid_address)
		{
			// Attempt to create a GPU image and image view for this texture.
			ImageAllocateResults allocResults = tryCreateAndAllocateImage_SubmitIfNeeded(imageCreationParams, asset::E_FORMAT::EF_COUNT, intendedNextSubmit, std::to_string(imageID));

			if (allocResults.isValid())
			{
				cachedImageRecord->type = imageType;
				cachedImageRecord->state = ImageState::CREATED_AND_MEMORY_BOUND;
				cachedImageRecord->lastUsedFrameIndex = currentFrameIndex; // there was an eviction + auto-submit, we need to update AGAIN
				cachedImageRecord->allocationOffset = allocResults.allocationOffset;
				cachedImageRecord->allocationSize = allocResults.allocationSize;
				cachedImageRecord->gpuImageView = allocResults.gpuImageView;
				cachedImageRecord->staticCPUImage = nullptr;
				cachedImageRecord->georeferencedImageState = GeoreferencedImageStreamingState::create(std::move(params), GeoreferencedImageTileSize);

				// This is because gpu image is square
				cachedImageRecord->georeferencedImageState->gpuImageSideLengthTiles = cachedImageRecord->gpuImageView->getCreationParameters().image->getCreationParameters().extent.width / GeoreferencedImageTileSize;
			}
			else
			{
				// All attempts to try create the GPU image and its corresponding view have failed.
				// Most likely cause: insufficient GPU memory or unsupported image parameters.
				
				m_logger.log("ensureGeoreferencedImageAvailability_AllocateIfNeeded failed, likely due to low VRAM.", nbl::system::ILogger::ELL_ERROR);
				_NBL_DEBUG_BREAK_IF(true);

				if (cachedImageRecord->allocationOffset != ImagesMemorySubAllocator::InvalidAddress)
				{
					// We previously successfully create and allocated memory for the Image
					// but failed to bind and create image view
					// It's crucial to deallocate the offset+size form our images memory suballocator
					imagesMemorySubAllocator->deallocate(cachedImageRecord->allocationOffset, cachedImageRecord->allocationSize);
				}

				if (cachedImageRecord->arrayIndex != InvalidTextureIndex)
				{
					// We previously allocated a descriptor index, but failed to create a usable GPU image.
					// It's crucial to deallocate this index to avoid leaks and preserve descriptor pool space.
					// No semaphore wait needed here, as the GPU never got to use this slot.
					suballocatedDescriptorSet->multi_deallocate(imagesArrayBinding, 1u, &cachedImageRecord->arrayIndex, {});
					cachedImageRecord->arrayIndex = InvalidTextureIndex;
				}
				
				// erase the entry we failed to fill, no need for `evictImage_SubmitIfNeeded`, because it didn't get to be used in any submit to defer it's memory and index deallocation
				imagesCache->erase(imageID);
			}
		}
		else
		{
			m_logger.log("ensureGeoreferencedImageAvailability_AllocateIfNeeded failed index allocation. shouldn't have happened.", nbl::system::ILogger::ELL_ERROR);
			cachedImageRecord->arrayIndex = InvalidTextureIndex;
		}
	}

	
	// cached or just inserted, we update the lastUsedFrameIndex
	cachedImageRecord->lastUsedFrameIndex = currentFrameIndex;

	assert(cachedImageRecord->arrayIndex != InvalidTextureIndex); // shouldn't happen, because we're using LRU cache, so worst case eviction will happen + multi-deallocate and next next multi_allocate should definitely succeed
	return (cachedImageRecord->arrayIndex != InvalidTextureIndex);
}

bool DrawResourcesFiller::queueGeoreferencedImageCopy_Internal(image_id imageID, const StreamedImageCopy& imageCopy)
{
	auto& vec = streamedImageCopies[imageID];
	vec.emplace_back(imageCopy);
	return true;
}

// TODO[Przemek]: similar to other drawXXX and drawXXX_internal functions that create mainobjects, drawObjects and push additional info in geometry buffer, input to function would be a GridDTMInfo
// We don't have an allocator or memory management for texture updates yet, see how `_test_addImageObject` is being temporarily used (Descriptor updates and pipeline barriers) to upload an image into gpu and update a descriptor slot (it will become more sophisticated but doesn't block you)
void DrawResourcesFiller::drawGridDTM(
	const float64_t2& topLeft,
	float64_t2 worldSpaceExtents,
	float gridCellWidth,
	uint64_t textureID,
	const DTMSettingsInfo& dtmSettingsInfo,
	SIntendedSubmitInfo& intendedNextSubmit)
{
	if (dtmSettingsInfo.mode == 0u)
		return;

	GridDTMInfo gridDTMInfo;
	gridDTMInfo.topLeft = topLeft;
	gridDTMInfo.worldSpaceExtents = worldSpaceExtents;
	gridDTMInfo.gridCellWidth = gridCellWidth;
	if (textureID != InvalidTextureIndex)
		gridDTMInfo.textureID = getImageIndexFromID(textureID, intendedNextSubmit); // for this to be valid and safe, this function needs to be called immediately after `addStaticImage` function to make sure image is in memory
	else
		gridDTMInfo.textureID = InvalidTextureIndex;

	// determine the thickes line
	float thickestLineThickness = 0.0f;
	if (dtmSettingsInfo.mode & E_DTM_MODE::OUTLINE)
	{
		thickestLineThickness = dtmSettingsInfo.outlineStyleInfo.worldSpaceLineWidth + dtmSettingsInfo.outlineStyleInfo.screenSpaceLineWidth;
	}
	else if (dtmSettingsInfo.mode & E_DTM_MODE::CONTOUR)
	{
		for (int i = 0; i < dtmSettingsInfo.contourSettingsCount; ++i)
		{
			const auto& contourLineStyle = dtmSettingsInfo.contourSettings[i].lineStyleInfo;
			const float contourLineThickness = contourLineStyle.worldSpaceLineWidth + contourLineStyle.screenSpaceLineWidth;
			thickestLineThickness = std::max(thickestLineThickness, contourLineThickness);
		}
	}
	gridDTMInfo.thicknessOfTheThickestLine = thickestLineThickness;

	setActiveDTMSettings(dtmSettingsInfo);
	beginMainObject(MainObjectType::GRID_DTM);

	uint32_t mainObjectIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
	if (mainObjectIdx == InvalidMainObjectIdx)
	{
		m_logger.log("drawGridDTM: acquireActiveMainObjectIndex returned invalid index", nbl::system::ILogger::ELL_ERROR);
		assert(false);
		return;
	}

	if (!addGridDTM_Internal(gridDTMInfo, mainObjectIdx))
	{
		// single grid DTM couldn't fit into memory to push to gpu, so we submit rendering current objects and reset geometry buffer and draw objects
		submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjectIdx);
		const bool success = addGridDTM_Internal(gridDTMInfo, mainObjectIdx);
		if (!success)
		{
			m_logger.log("addGridDTM_Internal failed, even after overflow-submission, this is irrecoverable.", nbl::system::ILogger::ELL_ERROR);
			assert(false);
		}
	}

	endMainObject();
}

void DrawResourcesFiller::addImageObject(image_id imageID, const OrientedBoundingBox2D& obb, SIntendedSubmitInfo& intendedNextSubmit)
{
	beginMainObject(MainObjectType::STATIC_IMAGE);

	uint32_t mainObjIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
	if (mainObjIdx == InvalidMainObjectIdx)
	{
		m_logger.log("addImageObject: acquireActiveMainObjectIndex returned invalid index", nbl::system::ILogger::ELL_ERROR);
		assert(false);
		return;
	}

	ImageObjectInfo info = {};
	info.topLeft = obb.topLeft;
	info.dirU = obb.dirU;
	info.aspectRatio = obb.aspectRatio;
	info.textureID = getImageIndexFromID(imageID, intendedNextSubmit); // for this to be valid and safe, this function needs to be called immediately after `addStaticImage` function to make sure image is in memory
	if (!addImageObject_Internal(info, mainObjIdx))
	{
		// single image object couldn't fit into memory to push to gpu, so we submit rendering current objects and reset geometry buffer and draw objects
		submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjIdx);
		const bool success = addImageObject_Internal(info, mainObjIdx);
		if (!success)
		{
			m_logger.log("addImageObject_Internal failed, even after overflow-submission, this is irrecoverable.", nbl::system::ILogger::ELL_ERROR);
			assert(false);
		}
	}

	endMainObject();
}

void DrawResourcesFiller::addGeoreferencedImage(image_id imageID, const float64_t3x3& NDCToWorld, SIntendedSubmitInfo& intendedNextSubmit)
{
	beginMainObject(MainObjectType::STREAMED_IMAGE);

	uint32_t mainObjIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
	if (mainObjIdx == InvalidMainObjectIdx)
	{
		m_logger.log("addGeoreferencedImage: acquireActiveMainObjectIndex returned invalid index", nbl::system::ILogger::ELL_ERROR);
		assert(false);
		return;
	}

	// Query imageType
	auto cachedImageRecord = imagesCache->peek(imageID);
	if (!cachedImageRecord)
	{
		m_logger.log("addGeoreferencedImage was not called immediately after enforceGeoreferencedImageAvailability!", nbl::system::ILogger::ELL_ERROR);
		assert(false);
		return;
	}

	// Generate upload data
	auto uploadData = generateTileUploadData(cachedImageRecord->type, NDCToWorld, cachedImageRecord->georeferencedImageState.get());

	// Queue image uploads
	for (const auto& imageCopy : uploadData.tiles)
		queueGeoreferencedImageCopy_Internal(imageID, imageCopy);

	GeoreferencedImageInfo info = {};
	info.topLeft = uploadData.viewportEncompassingOBB.topLeft;
	info.dirU = uploadData.viewportEncompassingOBB.dirU;
	info.aspectRatio = uploadData.viewportEncompassingOBB.aspectRatio;
	info.textureID = getImageIndexFromID(imageID, intendedNextSubmit); // for this to be valid and safe, this function needs to be called immediately after `addStaticImage` function to make sure image is in memory
	info.minUV = uploadData.minUV;
	info.maxUV = uploadData.maxUV;
	if (!addGeoreferencedImageInfo_Internal(info, mainObjIdx))
	{
		// single image object couldn't fit into memory to push to gpu, so we submit rendering current objects and reset geometry buffer and draw objects
		submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjIdx);
		const bool success = addGeoreferencedImageInfo_Internal(info, mainObjIdx);
		if (!success)
		{
			m_logger.log("addGeoreferencedImageInfo_Internal failed, even after overflow-submission, this is irrecoverable.", nbl::system::ILogger::ELL_ERROR);
			assert(false);
		}
	}

	endMainObject();
}

bool DrawResourcesFiller::pushAllUploads(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (!intendedNextSubmit.valid())
	{
		// It is a caching submit without command buffer, just for the purpose of accumulation of staging resources
		// In that case we don't push any uploads (i.e. we don't record any imageRecord commmand in active command buffer, because there is no active command buffer)
		return false;
	}

	bool success = true;
	if (currentReplayCache)
	{
		// This means we're in a replay cache scope, use the replay cache to push to GPU instead of internal accumulation
		success &= pushBufferUploads(intendedNextSubmit, currentReplayCache->resourcesCollection);
		success &= pushMSDFImagesUploads(intendedNextSubmit, currentReplayCache->msdfImagesState);

		// Push Static Images Uploads from replay cache, all the work below is necessary to detect whether our image to replay is already in the cache in the exact form OR we need to create new image + bind memory and set array index
		auto* device = m_utilities->getLogicalDevice();
		bool replayCacheFullyCovered = true;
		for (auto& [imageID, toReplayRecord] : *currentReplayCache->imagesCache)
		{
			if (toReplayRecord.type != ImageType::STATIC) // non-static images (Georeferenced) won't be replayed like this
				continue;

			auto* cachedRecord = imagesCache->peek(imageID);
			bool alreadyResident = false;

			// compare with existing state, and check whether image id is already resident.
			if (cachedRecord != nullptr)
			{
				const bool allocationMatches =
					cachedRecord->allocationOffset == toReplayRecord.allocationOffset &&
					cachedRecord->allocationSize == toReplayRecord.allocationSize;

				const bool arrayIndexMatches = cachedRecord->arrayIndex == toReplayRecord.arrayIndex;

				alreadyResident = allocationMatches && arrayIndexMatches && cachedRecord->state == ImageState::GPU_RESIDENT_WITH_VALID_STATIC_DATA;
			}

			// if already resident, just update the state to the cached state (to make sure it doesn't get issued for upload again) and move on.
			if (alreadyResident)
			{
				toReplayRecord.state = cachedRecord->state; // update the toReplayImageRecords's state, to completely match the currently resident state
				continue;
			}

			replayCacheFullyCovered = false;

			bool successCreateNewImage = false;

			// Not already resident, we need to recreate the image and bind the image memory to correct location again, and update the descriptor set and push the uploads
			auto existingGPUImageViewParams = toReplayRecord.gpuImageView->getCreationParameters();
			IGPUImage::SCreationParams imageParams = {};
			imageParams = existingGPUImageViewParams.image->getCreationParameters();

			auto newGPUImage = device->createImage(std::move(imageParams));
			if (newGPUImage)
			{
				nbl::video::ILogicalDevice::SBindImageMemoryInfo bindImageMemoryInfo =
				{
					.image = newGPUImage.get(),
					.binding = {.memory = imagesMemoryArena.memory.get(), .offset = imagesMemoryArena.offset + toReplayRecord.allocationOffset }
				};

				const bool boundToMemorySuccessfully = device->bindImageMemory({ &bindImageMemoryInfo, 1u });
				if (boundToMemorySuccessfully)
				{
					newGPUImage->setObjectDebugName((std::to_string(imageID) + " Static Image 2D").c_str());
					IGPUImageView::SCreationParams viewParams = existingGPUImageViewParams;
					viewParams.image = newGPUImage;

					auto newGPUImageView = device->createImageView(std::move(viewParams));
					if (newGPUImageView)
					{
						successCreateNewImage = true;
						toReplayRecord.gpuImageView = newGPUImageView;
						toReplayRecord.state = ImageState::CREATED_AND_MEMORY_BOUND;
						newGPUImageView->setObjectDebugName((std::to_string(imageID) + " Static Image View 2D").c_str());
					}

				}
			}

			if (!successCreateNewImage)
			{
				m_logger.log("Couldn't create new gpu image in pushAllUploads: cache and replay mode.", nbl::system::ILogger::ELL_ERROR);
				_NBL_DEBUG_BREAK_IF(true);
				success = false;
			}
		}
		
		// Our actual `imageCache` (which represents GPU state) didn't cover the replayCache fully, so new images had to be created, bound to memory. and they need to be written into their respective descriptor array indices again.
		// imagesCache = std::make_unique<ImagesCache>(*currentReplayCache->imagesCache);
		imagesCache->clear();
		for (auto it = currentReplayCache->imagesCache->rbegin(); it != currentReplayCache->imagesCache->rend(); it++)
			imagesCache->base_t::insert(it->first, it->second);

		if (!replayCacheFullyCovered)
		{
			// We need to block for previous submit in order to safely update the descriptor set array index next.
			// 
			// [FUTURE_CONSIDERATION]: To avoid stalling the CPU when replaying caches that overflow GPU memory,
			// we could recreate the image and image view, binding them to entirely new memory locations.
			// This would require an indirection mechanism in the shader to remap references from cached geometry or objects to the new image array indices.
			// Note: This isn't a problem if the replayed scene fits in memory and doesn't require overflow submissions due to image memory exhaustion.
			nbl::video::ISemaphore::SWaitInfo waitInfo = { .semaphore = intendedNextSubmit.scratchSemaphore.semaphore, .value = intendedNextSubmit.scratchSemaphore.value };
			device->blockForSemaphores({ &waitInfo, 1u });
		}

		success &= bindImagesToArrayIndices(*imagesCache);
		success &= pushStaticImagesUploads(intendedNextSubmit, *imagesCache);
		// Streamed uploads in cache&replay?!
	}
	else
	{
		flushDrawObjects();
		success &= pushBufferUploads(intendedNextSubmit, resourcesCollection);
		success &= pushMSDFImagesUploads(intendedNextSubmit, msdfImagesState);
		success &= bindImagesToArrayIndices(*imagesCache);
		success &= pushStaticImagesUploads(intendedNextSubmit, *imagesCache);
		success &= pushStreamedImagesUploads(intendedNextSubmit);
	}
	return success;
}

const DrawResourcesFiller::ResourcesCollection& DrawResourcesFiller::getResourcesCollection() const
{
	if (currentReplayCache)
		return currentReplayCache->resourcesCollection;
	else
		return resourcesCollection;
}

void DrawResourcesFiller::setActiveLineStyle(const LineStyleInfo& lineStyle)
{
	activeLineStyle = lineStyle;
	activeLineStyleIndex = InvalidStyleIdx;
}

void DrawResourcesFiller::setActiveDTMSettings(const DTMSettingsInfo& dtmSettingsInfo)
{
	activeDTMSettings = dtmSettingsInfo;
	activeDTMSettingsIndex = InvalidDTMSettingsIdx;
}

void DrawResourcesFiller::beginMainObject(MainObjectType type, TransformationType transformationType)
{
	activeMainObjectType = type;
	activeMainObjectTransformationType = transformationType;
	activeMainObjectIndex = InvalidMainObjectIdx;
}

void DrawResourcesFiller::endMainObject()
{
	activeMainObjectType = MainObjectType::NONE;
	activeMainObjectTransformationType = TransformationType::TT_NORMAL;
	activeMainObjectIndex = InvalidMainObjectIdx;
}

void DrawResourcesFiller::pushCustomProjection(const float64_t3x3& projection)
{
	activeProjections.push_back(projection);
	activeProjectionIndices.push_back(InvalidCustomProjectionIndex);
}

void DrawResourcesFiller::popCustomProjection()
{
	if (activeProjections.empty())
		return;

	activeProjections.pop_back();
	activeProjectionIndices.pop_back();
}

void DrawResourcesFiller::pushCustomClipRect(const WorldClipRect& clipRect)
{
	activeClipRects.push_back(clipRect);
	activeClipRectIndices.push_back(InvalidCustomClipRectIndex);
}

void DrawResourcesFiller::popCustomClipRect()
{	if (activeClipRects.empty())
		return;

	activeClipRects.pop_back();
	activeClipRectIndices.pop_back();
}

/// For advanced use only, (passed to shaders for them to know if we overflow-submitted in the middle if a main obj
uint32_t DrawResourcesFiller::getActiveMainObjectIndex() const
{
	if (currentReplayCache)
		return currentReplayCache->activeMainObjectIndex;
	else
		return activeMainObjectIndex;
}

const std::vector<DrawResourcesFiller::DrawCallData>& DrawResourcesFiller::getDrawCalls() const
{
	if (currentReplayCache)
		return currentReplayCache->drawCallsData;
	else
		return drawCalls;
}

std::unique_ptr<DrawResourcesFiller::ReplayCache> DrawResourcesFiller::createReplayCache()
{
	flushDrawObjects();
	std::unique_ptr<ReplayCache> ret = std::unique_ptr<ReplayCache>(new ReplayCache);
	ret->resourcesCollection = resourcesCollection;
	ret->msdfImagesState = msdfImagesState;
	for (auto& stagedMSDF : ret->msdfImagesState)
		stagedMSDF.uploadedToGPU = false; // to trigger upload for all msdf functions again.
	ret->drawCallsData = drawCalls;
	ret->activeMainObjectIndex = activeMainObjectIndex;
	ret->imagesCache = std::unique_ptr<ImagesCache>(new ImagesCache(*imagesCache));
	return ret;
}

void DrawResourcesFiller::setReplayCache(ReplayCache* cache)
{
	currentReplayCache = cache;
}

void DrawResourcesFiller::unsetReplayCache()
{
	currentReplayCache = nullptr;
}

bool DrawResourcesFiller::pushBufferUploads(SIntendedSubmitInfo& intendedNextSubmit, ResourcesCollection& resources)
{
	copiedResourcesSize = 0ull;

	if (resourcesCollection.calculateTotalConsumption() > resourcesGPUBuffer->getSize())
	{
		m_logger.log("some bug has caused the resourcesCollection to consume more memory than available in resourcesGPUBuffer without overflow submit", nbl::system::ILogger::ELL_ERROR);
		assert(false);
		return false;
	}

	auto copyCPUFilledDrawBuffer = [&](auto& drawBuffer) -> bool
		{
			// drawBuffer must be of type CPUGeneratedResource<T>
			SBufferRange<IGPUBuffer> copyRange = { copiedResourcesSize, drawBuffer.getStorageSize(), resourcesGPUBuffer};

			if (copyRange.offset + copyRange.size > resourcesGPUBuffer->getSize())
			{
				m_logger.log("`copyRange.offset + copyRange.size > resourcesGPUBuffer->getSize()` is true in `copyCPUFilledDrawBuffer`, this shouldn't happen with correct auto-submission mechanism.", nbl::system::ILogger::ELL_ERROR);
				assert(false);
				return false;
			}

			drawBuffer.bufferOffset = copyRange.offset;
			if (copyRange.size > 0ull)
			{
				if (!m_utilities->updateBufferRangeViaStagingBuffer(intendedNextSubmit, copyRange, drawBuffer.vector.data()))
					return false;
				copiedResourcesSize += drawBuffer.getAlignedStorageSize();
			}
			return true;
		};
	
	auto addComputeReservedFilledDrawBuffer = [&](auto& drawBuffer) -> bool
		{
			// drawBuffer must be of type ReservedComputeResource<T>
			SBufferRange<IGPUBuffer> copyRange = { copiedResourcesSize, drawBuffer.getStorageSize(), resourcesGPUBuffer};

			if (copyRange.offset + copyRange.size > resourcesGPUBuffer->getSize())
			{
				m_logger.log("`copyRange.offset + copyRange.size > resourcesGPUBuffer->getSize()` is true in `addComputeReservedFilledDrawBuffer`, this shouldn't happen with correct auto-submission mechanism.", nbl::system::ILogger::ELL_ERROR);
				assert(false);
				return false;
			}

			drawBuffer.bufferOffset = copyRange.offset;
			copiedResourcesSize += drawBuffer.getAlignedStorageSize();
		};

	copyCPUFilledDrawBuffer(resources.lineStyles);
	copyCPUFilledDrawBuffer(resources.dtmSettings);
	copyCPUFilledDrawBuffer(resources.customProjections);
	copyCPUFilledDrawBuffer(resources.customClipRects);
	copyCPUFilledDrawBuffer(resources.mainObjects);
	copyCPUFilledDrawBuffer(resources.drawObjects);
	copyCPUFilledDrawBuffer(resources.indexBuffer);
	copyCPUFilledDrawBuffer(resources.geometryInfo);
	
	return true;
}

bool DrawResourcesFiller::pushMSDFImagesUploads(SIntendedSubmitInfo& intendedNextSubmit, std::vector<MSDFImageState>& stagedMSDFCPUImages)
{
	auto* cmdBuffInfo = intendedNextSubmit.getCommandBufferForRecording();
	
	if (cmdBuffInfo)
	{
		IGPUCommandBuffer* commandBuffer = cmdBuffInfo->cmdbuf;

		auto msdfImage = msdfTextureArray->getCreationParameters().image;

		// preparing msdfs for imageRecord
		using image_barrier_t = IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t;
		image_barrier_t beforeTransferImageBarrier[] =
		{
			{
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
						.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
						.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
						.dstAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS,
					}
					// .ownershipOp. No queueFam ownership transfer
				},
				.image = msdfImage.get(),
				.subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = msdfImage->getCreationParameters().mipLevels,
					.baseArrayLayer = 0u,
					.layerCount = msdfTextureArray->getCreationParameters().image->getCreationParameters().arrayLayers,
				},
				.oldLayout = m_hasInitializedMSDFTextureArrays ? IImage::LAYOUT::READ_ONLY_OPTIMAL : IImage::LAYOUT::UNDEFINED,
				.newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
			}
		};
		commandBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = beforeTransferImageBarrier });

		// Do the copies and advance the iterator.
		// this is the pattern we use for iterating when entries will get erased if processed successfully, but may get skipped for later.
		for (uint32_t i = 0u; i < stagedMSDFCPUImages.size(); ++i)
		{
			auto& stagedMSDF = stagedMSDFCPUImages[i];
			if (stagedMSDF.image && i < msdfImage->getCreationParameters().arrayLayers)
			{
				for (uint32_t mip = 0; mip < stagedMSDF.image->getCreationParameters().mipLevels; mip++)
				{
					auto mipImageRegion = stagedMSDF.image->getRegion(mip, core::vectorSIMDu32(0u, 0u));
					if (mipImageRegion)
					{
						asset::IImage::SBufferCopy region = {};
						region.imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
						region.imageSubresource.mipLevel = mipImageRegion->imageSubresource.mipLevel;
						region.imageSubresource.baseArrayLayer = i;
						region.imageSubresource.layerCount = 1u;
						region.bufferOffset = 0u;
						region.bufferRowLength = mipImageRegion->getExtent().width;
						region.bufferImageHeight = 0u;
						region.imageExtent = mipImageRegion->imageExtent;
						region.imageOffset = { 0u, 0u, 0u };

						auto buffer = reinterpret_cast<uint8_t*>(stagedMSDF.image->getBuffer()->getPointer());
						auto bufferOffset = mipImageRegion->bufferOffset;

						stagedMSDF.uploadedToGPU = m_utilities->updateImageViaStagingBuffer(
							intendedNextSubmit,
							buffer + bufferOffset,
							nbl::ext::TextRendering::TextRenderer::MSDFTextureFormat,
							msdfImage.get(),
							IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
							{ &region, &region + 1 });
					}
					else
					{
						assert(false);
						stagedMSDF.uploadedToGPU = false;
					}
				}
			}
			else
			{
				stagedMSDF.uploadedToGPU = false;
			}
		}

		commandBuffer = intendedNextSubmit.getCommandBufferForRecording()->cmdbuf; // overflow-submit in utilities calls might've cause current recording command buffer to change

		// preparing msdfs for use
		image_barrier_t afterTransferImageBarrier[] =
		{
			{
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
						.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS,
						.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT, // we READ/SAMPLE on FRAG_SHADER
						.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS,
					}
					// .ownershipOp. No queueFam ownership transfer
				},
				.image = msdfImage.get(),
				.subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = msdfImage->getCreationParameters().mipLevels,
					.baseArrayLayer = 0u,
					.layerCount = msdfTextureArray->getCreationParameters().image->getCreationParameters().arrayLayers,
				},
				.oldLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				.newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
			}
		};
		commandBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = afterTransferImageBarrier });

		if (!m_hasInitializedMSDFTextureArrays)
			m_hasInitializedMSDFTextureArrays = true;

		return true;
	}
	else
	{
		m_logger.log("`copyRange.offset + copyRange.size > resourcesGPUBuffer->getSize()` is true in `addComputeReservedFilledDrawBuffer`, this shouldn't happen with correct auto-submission mechanism.", nbl::system::ILogger::ELL_ERROR);
		return false;
	}
}

bool DrawResourcesFiller::bindImagesToArrayIndices(ImagesCache& imagesCache)
{
	bool success = true;
	
	auto* device = m_utilities->getLogicalDevice();
	auto* descriptorSet = suballocatedDescriptorSet->getDescriptorSet();

	// DescriptorSet Updates
	std::vector<video::IGPUDescriptorSet::SDescriptorInfo> descriptorInfos;
	std::vector<IGPUDescriptorSet::SWriteDescriptorSet> descriptorWrites;
	descriptorInfos.resize(imagesCache.size());
	descriptorWrites.resize(imagesCache.size());

	uint32_t descriptorWriteCount = 0u;
	for (auto& [id, record] : imagesCache)
	{
		if (record.state >= ImageState::BOUND_TO_DESCRIPTOR_SET || !record.gpuImageView)
			continue;

		// Bind gpu image view to descriptor set
		video::IGPUDescriptorSet::SDescriptorInfo descriptorInfo = {};
		descriptorInfo.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
		descriptorInfo.desc = record.gpuImageView;
		descriptorInfos[descriptorWriteCount] = descriptorInfo;

		// consider batching contiguous writes, if descriptor set updating was a hotspot
		IGPUDescriptorSet::SWriteDescriptorSet descriptorWrite = {};
		descriptorWrite.dstSet = descriptorSet;
		descriptorWrite.binding = imagesArrayBinding;
		descriptorWrite.arrayElement = record.arrayIndex;
		descriptorWrite.count = 1u;
		descriptorWrite.info = &descriptorInfos[descriptorWriteCount];
		descriptorWrites[descriptorWriteCount] = descriptorWrite;

		record.state = ImageState::BOUND_TO_DESCRIPTOR_SET;
		descriptorWriteCount++;
	}

	if (descriptorWriteCount > 0u)
		success &= device->updateDescriptorSets(descriptorWriteCount, descriptorWrites.data(), 0u, nullptr);
	return success;
}

bool DrawResourcesFiller::pushStaticImagesUploads(SIntendedSubmitInfo& intendedNextSubmit, ImagesCache& imagesCache)
{
	bool success = true;

	// Push Static Images Uploads, only those who are not gpu resident
	// TODO: remove this vector and check state in each for loop below?
	std::vector<CachedImageRecord*> nonResidentImageRecords;
	for (auto& [id, record] : imagesCache)
	{
		if (record.staticCPUImage && (record.type == ImageType::STATIC || record.type == ImageType::GEOREFERENCED_FULL_RESOLUTION) && record.state < ImageState::GPU_RESIDENT_WITH_VALID_STATIC_DATA)
			nonResidentImageRecords.push_back(&record);
	}

	if (nonResidentImageRecords.size() > 0ull)
	{
		auto* device = m_utilities->getLogicalDevice();
		auto* cmdBuffInfo = intendedNextSubmit.getCommandBufferForRecording();
	
		if (cmdBuffInfo)
		{
			IGPUCommandBuffer* commandBuffer = cmdBuffInfo->cmdbuf;

			std::vector<IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t> beforeCopyImageBarriers;
			beforeCopyImageBarriers.resize(nonResidentImageRecords.size());

			// Pipeline Barriers before imageRecord
			for (uint32_t i = 0u; i < nonResidentImageRecords.size(); ++i)
			{
				auto& imageRecord = *nonResidentImageRecords[i];
				const auto& gpuImg = imageRecord.gpuImageView->getCreationParameters().image;
				beforeCopyImageBarriers[i] =
				{
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::NONE, // previous top of pipe -> top_of_pipe in first scope = none
							.srcAccessMask = ACCESS_FLAGS::NONE,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
							.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
						}
						// .ownershipOp. No queueFam ownership transfer
					},
					.image = gpuImg.get(),
					.subresourceRange = {
						.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
						.baseMipLevel = 0u,
						.levelCount = ICPUImageView::remaining_mip_levels,
						.baseArrayLayer = 0u,
						.layerCount = ICPUImageView::remaining_array_layers
					},
					.oldLayout = IImage::LAYOUT::UNDEFINED,
					.newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				};
			}
			success &= commandBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = beforeCopyImageBarriers });

			for (uint32_t i = 0u; i < nonResidentImageRecords.size(); ++i)
			{
				auto& imageRecord = *nonResidentImageRecords[i];
				auto& gpuImg = imageRecord.gpuImageView->getCreationParameters().image;
				success &= m_utilities->updateImageViaStagingBuffer(
					intendedNextSubmit,
					imageRecord.staticCPUImage->getBuffer()->getPointer(), imageRecord.staticCPUImage->getCreationParameters().format,
					gpuImg.get(), IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
					imageRecord.staticCPUImage->getRegions());

				if (success)
					imageRecord.state = ImageState::GPU_RESIDENT_WITH_VALID_STATIC_DATA;
				else
				{
					m_logger.log("Failed `updateImageViaStagingBuffer` in pushStaticImagesUploads.", nbl::system::ILogger::ELL_ERROR);
				}
			}

			commandBuffer = intendedNextSubmit.getCommandBufferForRecording()->cmdbuf; // overflow-submit in utilities calls might've cause current recording command buffer to change

			std::vector<IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t> afterCopyImageBarriers;
			afterCopyImageBarriers.resize(nonResidentImageRecords.size());

			// Pipeline Barriers before imageRecord
			for (uint32_t i = 0u; i < nonResidentImageRecords.size(); ++i)
			{
				auto& imageRecord = *nonResidentImageRecords[i];
				const auto& gpuImg = imageRecord.gpuImageView->getCreationParameters().image;
				afterCopyImageBarriers[i] =
				{
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT, // previous top of pipe -> top_of_pipe in first scope = none
							.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
							.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS,
						}
						// .ownershipOp. No queueFam ownership transfer
					},
					.image = gpuImg.get(),
					.subresourceRange = {
						.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
						.baseMipLevel = 0u,
						.levelCount = ICPUImageView::remaining_mip_levels,
						.baseArrayLayer = 0u,
						.layerCount = ICPUImageView::remaining_array_layers
					},
					.oldLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
					.newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
				};
			}
			success &= commandBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = afterCopyImageBarriers });
		}
		else
		{
			_NBL_DEBUG_BREAK_IF(true);
			success = false;
		}
	}

	if (!success)
	{
		m_logger.log("Failure in `pushStaticImagesUploads`.", nbl::system::ILogger::ELL_ERROR);
		_NBL_DEBUG_BREAK_IF(true);
	}
	return success;
}

bool DrawResourcesFiller::pushStreamedImagesUploads(SIntendedSubmitInfo& intendedNextSubmit)
{
	bool success = true;

	if (streamedImageCopies.size() > 0ull)
	{
		auto* device = m_utilities->getLogicalDevice();
		auto* cmdBuffInfo = intendedNextSubmit.getCommandBufferForRecording();
	
		if (cmdBuffInfo)
		{
			IGPUCommandBuffer* commandBuffer = cmdBuffInfo->cmdbuf;

			std::vector<IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t> beforeCopyImageBarriers;
			beforeCopyImageBarriers.reserve(streamedImageCopies.size());

			// Pipeline Barriers before imageCopy
			for (auto& [imageID, imageCopies] : streamedImageCopies)
			{
				auto* imageRecord = imagesCache->peek(imageID);
				if (imageRecord == nullptr)
					continue;

				const auto& gpuImg = imageRecord->gpuImageView->getCreationParameters().image;

				beforeCopyImageBarriers.push_back(
					{
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::NONE, // previous top of pipe -> top_of_pipe in first scope = none
								.srcAccessMask = ACCESS_FLAGS::NONE,
								.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
								.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
							}
							// .ownershipOp. No queueFam ownership transfer
						},
						.image = gpuImg.get(),
						.subresourceRange = {
							.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
							.baseMipLevel = 0u,
							.levelCount = ICPUImageView::remaining_mip_levels,
							.baseArrayLayer = 0u,
							.layerCount = ICPUImageView::remaining_array_layers
						},
						.oldLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
						.newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
					});
			}
			success &= commandBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = beforeCopyImageBarriers });
			
			for (auto& [imageID, imageCopies] : streamedImageCopies)
			{
				auto* imageRecord = imagesCache->peek(imageID);
				if (imageRecord == nullptr)
					continue;

				const auto& gpuImg = imageRecord->gpuImageView->getCreationParameters().image;

				for (auto& imageCopy : imageCopies)
				{
					success &= m_utilities->updateImageViaStagingBuffer(
						intendedNextSubmit,
						imageCopy.srcBuffer->getPointer(), imageCopy.srcFormat,
						gpuImg.get(), IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
						{ &imageCopy.region, 1u });
				}
			}

			commandBuffer = intendedNextSubmit.getCommandBufferForRecording()->cmdbuf; // overflow-submit in utilities calls might've cause current recording command buffer to change

			std::vector<IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t> afterCopyImageBarriers;
			afterCopyImageBarriers.reserve(streamedImageCopies.size());

			// Pipeline Barriers after imageCopy
			for (auto& [imageID, imageCopies] : streamedImageCopies)
			{
				auto* imageRecord = imagesCache->peek(imageID);
				if (imageRecord == nullptr)
					continue;

				const auto& gpuImg = imageRecord->gpuImageView->getCreationParameters().image;

				afterCopyImageBarriers.push_back (
					{
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT, // previous top of pipe -> top_of_pipe in first scope = none
								.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
								.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
								.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS,
							}
							// .ownershipOp. No queueFam ownership transfer
						},
						.image = gpuImg.get(),
						.subresourceRange = {
							.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
							.baseMipLevel = 0u,
							.levelCount = ICPUImageView::remaining_mip_levels,
							.baseArrayLayer = 0u,
							.layerCount = ICPUImageView::remaining_array_layers
						},
						.oldLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
						.newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
					});
			}
			success &= commandBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = afterCopyImageBarriers });

			streamedImageCopies.clear();
		}
		else
		{
			_NBL_DEBUG_BREAK_IF(true);
			success = false;
		}
	}

	if (!success)
	{
		m_logger.log("Failure in `pushStreamedImagesUploads`.", nbl::system::ILogger::ELL_ERROR);
		_NBL_DEBUG_BREAK_IF(true);
	}
	return success;
}

const size_t DrawResourcesFiller::calculateRemainingResourcesSize() const
{
	assert(resourcesGPUBuffer->getSize() >= resourcesCollection.calculateTotalConsumption());
	return resourcesGPUBuffer->getSize() - resourcesCollection.calculateTotalConsumption();
}

void DrawResourcesFiller::submitCurrentDrawObjectsAndReset(SIntendedSubmitInfo& intendedNextSubmit, uint32_t& mainObjectIndex)
{
	submitDraws(intendedNextSubmit);
	reset(); // resets everything, things referenced through mainObj and other shit will be pushed again through acquireXXX_SubmitIfNeeded
	mainObjectIndex = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit); // it will be 0 because it's first mainObjectIndex after reset and invalidation
}

uint32_t DrawResourcesFiller::addLineStyle_Internal(const LineStyleInfo& lineStyleInfo)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	const bool enoughMem = remainingResourcesSize >= sizeof(LineStyle); // enough remaining memory for 1 more linestyle?
	if (!enoughMem)
		return InvalidStyleIdx;
	// TODO: Maybe constraint by a max size? and return InvalidIdx if it would exceed

	LineStyle gpuLineStyle = lineStyleInfo.getAsGPUData();
	_NBL_DEBUG_BREAK_IF(gpuLineStyle.stipplePatternSize > LineStyle::StipplePatternMaxSize); // Oops, even after style normalization the style is too long to be in gpu mem :(
	for (uint32_t i = 0u; i < resourcesCollection.lineStyles.vector.size(); ++i)
	{
		const LineStyle& itr = resourcesCollection.lineStyles.vector[i];
		if (itr == gpuLineStyle)
			return i;
	}

	return resourcesCollection.lineStyles.addAndGetOffset(gpuLineStyle); // this will implicitly increase total resource consumption and reduce remaining size --> no need for mem size trackers
}

uint32_t DrawResourcesFiller::addDTMSettings_Internal(const DTMSettingsInfo& dtmSettingsInfo, SIntendedSubmitInfo& intendedNextSubmit)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	const size_t noOfLineStylesRequired = ((dtmSettingsInfo.mode & E_DTM_MODE::OUTLINE) ? 1u : 0u) + dtmSettingsInfo.contourSettingsCount;
	const size_t maxMemRequired = sizeof(DTMSettings) + noOfLineStylesRequired * sizeof(LineStyle);
	const bool enoughMem = remainingResourcesSize >= maxMemRequired; // enough remaining memory for 1 more dtm settings with 2 referenced line styles?

	if (!enoughMem)
		return InvalidDTMSettingsIdx;
	// TODO: Maybe constraint by a max size? and return InvalidIdx if it would exceed

	DTMSettings dtmSettings;

	////dtmSettingsInfo.mode = E_DTM_MODE::HEIGHT_SHADING | E_DTM_MODE::CONTOUR | E_DTM_MODE::OUTLINE;

	dtmSettings.mode = dtmSettingsInfo.mode;
	if (dtmSettings.mode & E_DTM_MODE::HEIGHT_SHADING)
	{
		switch (dtmSettingsInfo.heightShadingInfo.heightShadingMode)
		{
		case E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS:
			dtmSettings.heightShadingSettings.intervalLength = std::numeric_limits<float>::infinity();
			break;
		case E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS:
			dtmSettings.heightShadingSettings.intervalLength = dtmSettingsInfo.heightShadingInfo.intervalLength;
			break;
		case E_HEIGHT_SHADING_MODE::CONTINOUS_INTERVALS:
			dtmSettings.heightShadingSettings.intervalLength = 0.0f;
			break;
		}
		dtmSettings.heightShadingSettings.intervalIndexToHeightMultiplier = dtmSettingsInfo.heightShadingInfo.intervalIndexToHeightMultiplier;
		dtmSettings.heightShadingSettings.isCenteredShading = static_cast<int>(dtmSettingsInfo.heightShadingInfo.isCenteredShading);
		dtmSettingsInfo.heightShadingInfo.fillShaderDTMSettingsHeightColorMap(dtmSettings);
	}
	if (dtmSettings.mode & E_DTM_MODE::CONTOUR)
	{
		dtmSettings.contourSettingsCount = dtmSettingsInfo.contourSettingsCount;
		for (uint32_t i = 0u; i < dtmSettings.contourSettingsCount; ++i)
		{
			dtmSettings.contourSettings[i].contourLinesStartHeight = dtmSettingsInfo.contourSettings[i].startHeight;
			dtmSettings.contourSettings[i].contourLinesEndHeight = dtmSettingsInfo.contourSettings[i].endHeight;
			dtmSettings.contourSettings[i].contourLinesHeightInterval = dtmSettingsInfo.contourSettings[i].heightInterval;
			dtmSettings.contourSettings[i].contourLineStyleIdx = addLineStyle_Internal(dtmSettingsInfo.contourSettings[i].lineStyleInfo);
		}
	}
	if (dtmSettings.mode & E_DTM_MODE::OUTLINE)
	{
		dtmSettings.outlineLineStyleIdx = addLineStyle_Internal(dtmSettingsInfo.outlineStyleInfo);
	}

	for (uint32_t i = 0u; i < resourcesCollection.dtmSettings.vector.size(); ++i)
	{
		const DTMSettings& itr = resourcesCollection.dtmSettings.vector[i];
		if (itr == dtmSettings)
			return i;
	}

	return resourcesCollection.dtmSettings.addAndGetOffset(dtmSettings); // this will implicitly increase total resource consumption and reduce remaining size --> no need for mem size trackers
}

float64_t3x3 DrawResourcesFiller::getFixedGeometryFinalTransformationMatrix(const float64_t3x3& transformation, TransformationType transformationType) const
{
	if (!activeProjections.empty())
	{
		float64_t3x3 newTransformation = nbl::hlsl::mul(activeProjections.back(), transformation);

		if (transformationType == TransformationType::TT_NORMAL)
		{
			return newTransformation;
		}
		else if (transformationType == TransformationType::TT_FIXED_SCREENSPACE_SIZE)
		{
			// Extract normalized rotation columns
			float64_t2 column0 = nbl::hlsl::normalize(float64_t2(newTransformation[0][0], newTransformation[1][0]));
			float64_t2 column1 = nbl::hlsl::normalize(float64_t2(newTransformation[0][1], newTransformation[1][1]));

			// Extract fixed screen-space scale from the original transformation
			float64_t2 fixedScale = float64_t2(
				nbl::hlsl::length(float64_t2(transformation[0][0], transformation[1][0])),
				nbl::hlsl::length(float64_t2(transformation[0][1], transformation[1][1])));

			// Apply fixed scale to normalized directions
			column0 *= fixedScale.x;
			column1 *= fixedScale.y;

			// Compose final matrix with adjusted columns
			newTransformation[0][0] = column0[0];
			newTransformation[1][0] = column0[1];
			newTransformation[0][1] = column1[0];
			newTransformation[1][1] = column1[1];

			return newTransformation;
		}
		else
		{
			// Fallback if transformationType is unrecognized, shouldn't happen
			return newTransformation;
		}
	}
	else
	{
		// Within no active projection scope, return transformation directly
		return transformation;
	}
}

uint32_t DrawResourcesFiller::acquireActiveLineStyleIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (activeLineStyleIndex == InvalidStyleIdx)
		activeLineStyleIndex = addLineStyle_SubmitIfNeeded(activeLineStyle, intendedNextSubmit);
	
	return activeLineStyleIndex;
}

uint32_t DrawResourcesFiller::acquireActiveDTMSettingsIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (activeDTMSettingsIndex == InvalidDTMSettingsIdx)
		activeDTMSettingsIndex = addDTMSettings_SubmitIfNeeded(activeDTMSettings, intendedNextSubmit);
	
	return activeDTMSettingsIndex;
}

uint32_t DrawResourcesFiller::acquireActiveCustomProjectionIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (activeProjectionIndices.empty())
		return InvalidCustomProjectionIndex;

	if (activeProjectionIndices.back() == InvalidCustomProjectionIndex)
		activeProjectionIndices.back() = addCustomProjection_SubmitIfNeeded(activeProjections.back(), intendedNextSubmit);
	
	return activeProjectionIndices.back();
}

uint32_t DrawResourcesFiller::acquireActiveCustomClipRectIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (activeClipRectIndices.empty())
		return InvalidCustomClipRectIndex;

	if (activeClipRectIndices.back() == InvalidCustomClipRectIndex)
		activeClipRectIndices.back() = addCustomClipRect_SubmitIfNeeded(activeClipRects.back(), intendedNextSubmit);
	
	return activeClipRectIndices.back();
}

uint32_t DrawResourcesFiller::acquireActiveMainObjectIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (activeMainObjectIndex != InvalidMainObjectIdx)
		return activeMainObjectIndex;
	if (activeMainObjectType == MainObjectType::NONE)
	{
		assert(false); // You're probably trying to acquire mainObjectIndex outside of startMainObject, endMainObject scope
		return InvalidMainObjectIdx;
	}

	const bool needsLineStyle =
		(activeMainObjectType == MainObjectType::POLYLINE) ||
		(activeMainObjectType == MainObjectType::HATCH) ||
		(activeMainObjectType == MainObjectType::TEXT);
	const bool needsDTMSettings = (activeMainObjectType == MainObjectType::DTM || activeMainObjectType == MainObjectType::GRID_DTM);
	const bool needsCustomProjection = (!activeProjectionIndices.empty());
	const bool needsCustomClipRect = (!activeClipRectIndices.empty());

	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	// making sure MainObject and everything it references fits into remaining resources mem
	size_t memRequired = sizeof(MainObject);
	if (needsLineStyle) memRequired += sizeof(LineStyle);
	if (needsDTMSettings) memRequired += sizeof(DTMSettings);
	if (needsCustomProjection) memRequired += sizeof(float64_t3x3);
	if (needsCustomClipRect) memRequired += sizeof(WorldClipRect);

	const bool enoughMem = remainingResourcesSize >= memRequired; // enough remaining memory for 1 more dtm settings with 2 referenced line styles?
	const bool needToOverflowSubmit = (!enoughMem) || (resourcesCollection.mainObjects.vector.size() >= MaxIndexableMainObjects);
	
	if (needToOverflowSubmit)
	{
		// failed to fit into remaining resources mem or exceeded max indexable mainobj
		submitDraws(intendedNextSubmit);
		reset(); // resets everything! be careful!
	}
	
	MainObject mainObject = {};
	// These 3 calls below shouldn't need to Submit because we made sure there is enough memory for all of them.
	// if something here triggers a auto-submit it's a possible bug with calculating `memRequired` above, TODO: assert that somehow?
	mainObject.styleIdx = (needsLineStyle) ? acquireActiveLineStyleIndex_SubmitIfNeeded(intendedNextSubmit) : InvalidStyleIdx;
	mainObject.dtmSettingsIdx = (needsDTMSettings) ? acquireActiveDTMSettingsIndex_SubmitIfNeeded(intendedNextSubmit) : InvalidDTMSettingsIdx;
	mainObject.customProjectionIndex = (needsCustomProjection) ? acquireActiveCustomProjectionIndex_SubmitIfNeeded(intendedNextSubmit) : InvalidCustomProjectionIndex;
	mainObject.customClipRectIndex = (needsCustomClipRect) ? acquireActiveCustomClipRectIndex_SubmitIfNeeded(intendedNextSubmit) : InvalidCustomClipRectIndex;
	mainObject.transformationType = (uint32_t)activeMainObjectTransformationType;
	activeMainObjectIndex = resourcesCollection.mainObjects.addAndGetOffset(mainObject);
	return activeMainObjectIndex;
}

uint32_t DrawResourcesFiller::addLineStyle_SubmitIfNeeded(const LineStyleInfo& lineStyle, SIntendedSubmitInfo& intendedNextSubmit)
{
	uint32_t outLineStyleIdx = addLineStyle_Internal(lineStyle);
	if (outLineStyleIdx == InvalidStyleIdx)
	{
		// There wasn't enough resource memory remaining to fit a single LineStyle
		submitDraws(intendedNextSubmit);
		reset(); // resets everything! be careful!

		outLineStyleIdx = addLineStyle_Internal(lineStyle);
		assert(outLineStyleIdx != InvalidStyleIdx);
	}

	return outLineStyleIdx;
}

uint32_t DrawResourcesFiller::addDTMSettings_SubmitIfNeeded(const DTMSettingsInfo& dtmSettings, SIntendedSubmitInfo& intendedNextSubmit)
{
	// before calling `addDTMSettings_Internal` we have made sute we have enough mem for 
	uint32_t outDTMSettingIdx = addDTMSettings_Internal(dtmSettings, intendedNextSubmit);
	if (outDTMSettingIdx == InvalidDTMSettingsIdx)
	{
		// There wasn't enough resource memory remaining to fit dtmsettings struct + 2 linestyles structs.
		submitDraws(intendedNextSubmit);
		reset(); // resets everything! be careful!

		outDTMSettingIdx = addDTMSettings_Internal(dtmSettings, intendedNextSubmit);
		assert(outDTMSettingIdx != InvalidDTMSettingsIdx);
	}
	return outDTMSettingIdx;
}

uint32_t DrawResourcesFiller::addCustomProjection_SubmitIfNeeded(const float64_t3x3& projection, SIntendedSubmitInfo& intendedNextSubmit)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	const size_t memRequired = sizeof(float64_t3x3);
	const bool enoughMem = remainingResourcesSize >= memRequired; // enough remaining memory for 1 more dtm settings with 2 referenced line styles?

	if (!enoughMem)
	{
		submitDraws(intendedNextSubmit);
		reset(); // resets everything! be careful!
	}
	
	resourcesCollection.customProjections.vector.push_back(projection); // this will implicitly increase total resource consumption and reduce remaining size --> no need for mem size trackers
	return resourcesCollection.customProjections.vector.size() - 1u;
}

uint32_t DrawResourcesFiller::addCustomClipRect_SubmitIfNeeded(const WorldClipRect& clipRect, SIntendedSubmitInfo& intendedNextSubmit)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	const size_t memRequired = sizeof(WorldClipRect);
	const bool enoughMem = remainingResourcesSize >= memRequired; // enough remaining memory for 1 more dtm settings with 2 referenced line styles?

	if (!enoughMem)
	{
		submitDraws(intendedNextSubmit);
		reset(); // resets everything! be careful!
	}
	
	resourcesCollection.customClipRects.vector.push_back(clipRect); // this will implicitly increase total resource consumption and reduce remaining size --> no need for mem size trackers
	return resourcesCollection.customClipRects.vector.size() - 1u;
}

void DrawResourcesFiller::addPolylineObjects_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
{
	if (section.type == ObjectType::LINE)
		addLines_Internal(polyline, section, currentObjectInSection, mainObjIdx);
	else if (section.type == ObjectType::QUAD_BEZIER)
		addQuadBeziers_Internal(polyline, section, currentObjectInSection, mainObjIdx);
	else
		assert(false); // we don't handle other object types
}

void DrawResourcesFiller::addPolylineConnectors_Internal(const CPolylineBase& polyline, uint32_t& currentPolylineConnectorObj, uint32_t mainObjIdx)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();

	const uint32_t uploadableObjects = (remainingResourcesSize) / (sizeof(PolylineConnector) + sizeof(DrawObject) + sizeof(uint32_t) * 6u);
	// TODO[ERFAN]: later take into account: our maximum indexable vertex 
	
	const uint32_t connectorCount = static_cast<uint32_t>(polyline.getConnectors().size());
	const uint32_t remainingObjects = connectorCount - currentPolylineConnectorObj;
	const uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

	if (objectsToUpload <= 0u)
		return;

	// Add Geometry
	const auto connectorsByteSize = sizeof(PolylineConnector) * objectsToUpload;
	size_t geometryBufferOffset = resourcesCollection.geometryInfo.increaseSizeAndGetOffset(connectorsByteSize, alignof(PolylineConnector));
	void* dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
	const PolylineConnector& connector = polyline.getConnectors()[currentPolylineConnectorObj];
	memcpy(dst, &connector, connectorsByteSize);

	// Push Indices, remove later when compute fills this
	uint32_t* indexBufferToBeFilled = resourcesCollection.indexBuffer.increaseCountAndGetPtr(6u * objectsToUpload);
	const uint32_t startObj = resourcesCollection.drawObjects.getCount();
	for (uint32_t i = 0u; i < objectsToUpload; ++i)
	{
		indexBufferToBeFilled[i*6]		= (startObj+i)*4u + 1u;
		indexBufferToBeFilled[i*6 + 1u]	= (startObj+i)*4u + 0u;
		indexBufferToBeFilled[i*6 + 2u]	= (startObj+i)*4u + 2u;
		indexBufferToBeFilled[i*6 + 3u]	= (startObj+i)*4u + 1u;
		indexBufferToBeFilled[i*6 + 4u]	= (startObj+i)*4u + 2u;
		indexBufferToBeFilled[i*6 + 5u]	= (startObj+i)*4u + 3u;
	}

	// Add DrawObjs
	DrawObject* drawObjectsToBeFilled = resourcesCollection.drawObjects.increaseCountAndGetPtr(objectsToUpload);
	DrawObject drawObj = {};
	drawObj.mainObjIndex = mainObjIdx;
	drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::POLYLINE_CONNECTOR) | 0 << 16);
	drawObj.geometryAddress = geometryBufferOffset;
	for (uint32_t i = 0u; i < objectsToUpload; ++i)
	{
		drawObjectsToBeFilled[i] = drawObj;
		drawObj.geometryAddress += sizeof(PolylineConnector);
	} 

	currentPolylineConnectorObj += objectsToUpload;
}

void DrawResourcesFiller::addLines_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
{
	assert(section.count >= 1u);
	assert(section.type == ObjectType::LINE);


	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	if (remainingResourcesSize < sizeof(LinePointInfo))
		return;

	// how many lines fit into mem? --> memConsumption = sizeof(LinePointInfo) + sizeof(LinePointInfo)*lineCount + sizeof(DrawObject)*lineCount + sizeof(uint32_t) * 6u * lineCount
	const uint32_t uploadableObjects = (remainingResourcesSize - sizeof(LinePointInfo)) / (sizeof(LinePointInfo) + sizeof(DrawObject) + sizeof(uint32_t) * 6u);
	// TODO[ERFAN]: later take into account: our maximum indexable vertex 

	const uint32_t lineCount = section.count;
	const uint32_t remainingObjects = lineCount - currentObjectInSection;
	const uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

	if (objectsToUpload <= 0u)
		return;

	// Add Geometry
	const auto pointsByteSize = sizeof(LinePointInfo) * (objectsToUpload + 1u);
	size_t geometryBufferOffset = resourcesCollection.geometryInfo.increaseSizeAndGetOffset(pointsByteSize, alignof(LinePointInfo));
	void* dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
	const LinePointInfo& linePoint = polyline.getLinePointAt(section.index + currentObjectInSection);
	memcpy(dst, &linePoint, pointsByteSize);

	// Push Indices, remove later when compute fills this
	uint32_t* indexBufferToBeFilled = resourcesCollection.indexBuffer.increaseCountAndGetPtr(6u * objectsToUpload);
	const uint32_t startObj = resourcesCollection.drawObjects.getCount();
	for (uint32_t i = 0u; i < objectsToUpload; ++i)
	{
		indexBufferToBeFilled[i*6]		= (startObj+i)*4u + 1u;
		indexBufferToBeFilled[i*6 + 1u]	= (startObj+i)*4u + 0u;
		indexBufferToBeFilled[i*6 + 2u]	= (startObj+i)*4u + 2u;
		indexBufferToBeFilled[i*6 + 3u]	= (startObj+i)*4u + 1u;
		indexBufferToBeFilled[i*6 + 4u]	= (startObj+i)*4u + 2u;
		indexBufferToBeFilled[i*6 + 5u]	= (startObj+i)*4u + 3u;
	}

	// Add DrawObjs
	DrawObject* drawObjectsToBeFilled = resourcesCollection.drawObjects.increaseCountAndGetPtr(objectsToUpload);
	DrawObject drawObj = {};
	drawObj.mainObjIndex = mainObjIdx;
	drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::LINE) | 0 << 16);
	drawObj.geometryAddress = geometryBufferOffset;
	for (uint32_t i = 0u; i < objectsToUpload; ++i)
	{
		drawObjectsToBeFilled[i] = drawObj;
		drawObj.geometryAddress += sizeof(LinePointInfo);
	} 

	currentObjectInSection += objectsToUpload;
}

void DrawResourcesFiller::addQuadBeziers_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
{
	constexpr uint32_t CagesPerQuadBezier = 3u; // TODO: Break into 3 beziers in compute shader.

	assert(section.type == ObjectType::QUAD_BEZIER);

	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	// how many quad bezier objects fit into mem?
	// memConsumption = quadBezCount * (sizeof(QuadraticBezierInfo) + 3*(sizeof(DrawObject)+6u*sizeof(uint32_t))
	const uint32_t uploadableObjects = (remainingResourcesSize) / (sizeof(QuadraticBezierInfo) + (sizeof(DrawObject) + 6u * sizeof(uint32_t)) * CagesPerQuadBezier);
	// TODO[ERFAN]: later take into account: our maximum indexable vertex 
	
	const uint32_t beziersCount = section.count;
	const uint32_t remainingObjects = beziersCount - currentObjectInSection;
	const uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);
	const uint32_t cagesCount = objectsToUpload * CagesPerQuadBezier;

	if (objectsToUpload <= 0u)
		return;
	
	// Add Geometry
	const auto beziersByteSize = sizeof(QuadraticBezierInfo) * (objectsToUpload);
	size_t geometryBufferOffset = resourcesCollection.geometryInfo.increaseSizeAndGetOffset(beziersByteSize, alignof(QuadraticBezierInfo));
	void* dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
	const QuadraticBezierInfo& quadBezier = polyline.getQuadBezierInfoAt(section.index + currentObjectInSection);
	memcpy(dst, &quadBezier, beziersByteSize);



	// Push Indices, remove later when compute fills this
	uint32_t* indexBufferToBeFilled = resourcesCollection.indexBuffer.increaseCountAndGetPtr(6u*cagesCount);
	const uint32_t startObj = resourcesCollection.drawObjects.getCount();
	for (uint32_t i = 0u; i < cagesCount; ++i)
	{
		indexBufferToBeFilled[i*6]		= (startObj+i)*4u + 1u;
		indexBufferToBeFilled[i*6 + 1u]	= (startObj+i)*4u + 0u;
		indexBufferToBeFilled[i*6 + 2u]	= (startObj+i)*4u + 2u;
		indexBufferToBeFilled[i*6 + 3u]	= (startObj+i)*4u + 1u;
		indexBufferToBeFilled[i*6 + 4u]	= (startObj+i)*4u + 2u;
		indexBufferToBeFilled[i*6 + 5u]	= (startObj+i)*4u + 3u;
	}
	
	// Add DrawObjs
	DrawObject* drawObjectsToBeFilled = resourcesCollection.drawObjects.increaseCountAndGetPtr(cagesCount);
	DrawObject drawObj = {};
	drawObj.mainObjIndex = mainObjIdx;
	drawObj.geometryAddress = geometryBufferOffset;
	for (uint32_t i = 0u; i < objectsToUpload; ++i)
	{
		for (uint16_t subObject = 0; subObject < CagesPerQuadBezier; subObject++)
		{
			drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::QUAD_BEZIER) | (subObject << 16));
			drawObjectsToBeFilled[i * CagesPerQuadBezier + subObject] = drawObj;
		}
		drawObj.geometryAddress += sizeof(QuadraticBezierInfo);
	}


	currentObjectInSection += objectsToUpload;
}

void DrawResourcesFiller::addHatch_Internal(const Hatch& hatch, uint32_t& currentObjectInSection, uint32_t mainObjIndex)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();

	const uint32_t uploadableObjects = (remainingResourcesSize) / (sizeof(Hatch::CurveHatchBox) + sizeof(DrawObject) + sizeof(uint32_t) * 6u);
	// TODO[ERFAN]: later take into account: our maximum indexable vertex 
	
	uint32_t remainingObjects = hatch.getHatchBoxCount() - currentObjectInSection;
	const uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

	if (objectsToUpload <= 0u)
		return;

	// Add Geometry
	static_assert(sizeof(CurveBox) == sizeof(Hatch::CurveHatchBox));
	const auto curveBoxesByteSize = sizeof(Hatch::CurveHatchBox) * objectsToUpload;
	size_t geometryBufferOffset = resourcesCollection.geometryInfo.increaseSizeAndGetOffset(curveBoxesByteSize, alignof(Hatch::CurveHatchBox));
	void* dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
	const Hatch::CurveHatchBox& hatchBox = hatch.getHatchBox(currentObjectInSection); // WARNING: This is assuming hatch boxes are contigous in memory, TODO: maybe make that more obvious through Hatch interface
	memcpy(dst, &hatchBox, curveBoxesByteSize);
	
	// Push Indices, remove later when compute fills this
	uint32_t* indexBufferToBeFilled = resourcesCollection.indexBuffer.increaseCountAndGetPtr(6u * objectsToUpload);
	const uint32_t startObj = resourcesCollection.drawObjects.getCount();
	for (uint32_t i = 0u; i < objectsToUpload; ++i)
	{
		indexBufferToBeFilled[i*6]		= (startObj+i)*4u + 1u;
		indexBufferToBeFilled[i*6 + 1u]	= (startObj+i)*4u + 0u;
		indexBufferToBeFilled[i*6 + 2u]	= (startObj+i)*4u + 2u;
		indexBufferToBeFilled[i*6 + 3u]	= (startObj+i)*4u + 1u;
		indexBufferToBeFilled[i*6 + 4u]	= (startObj+i)*4u + 2u;
		indexBufferToBeFilled[i*6 + 5u]	= (startObj+i)*4u + 3u;
	}
	
	// Add DrawObjs
	DrawObject* drawObjectsToBeFilled = resourcesCollection.drawObjects.increaseCountAndGetPtr(objectsToUpload);
	DrawObject drawObj = {};
	drawObj.mainObjIndex = mainObjIndex;
	drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::CURVE_BOX) | (0 << 16));
	drawObj.geometryAddress = geometryBufferOffset;
	for (uint32_t i = 0u; i < objectsToUpload; ++i)
	{
		drawObjectsToBeFilled[i] = drawObj;
		drawObj.geometryAddress += sizeof(Hatch::CurveHatchBox);
	}

	// Add Indices
	currentObjectInSection += uploadableObjects;
}

bool DrawResourcesFiller::addFontGlyph_Internal(const GlyphInfo& glyphInfo, uint32_t mainObjIdx)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();

	const uint32_t uploadableObjects = (remainingResourcesSize) / (sizeof(GlyphInfo) + sizeof(DrawObject) + sizeof(uint32_t) * 6u);
	// TODO[ERFAN]: later take into account: our maximum indexable vertex 
	
	if (uploadableObjects <= 0u)
		return false;

	// Add Geometry
	size_t geometryBufferOffset = resourcesCollection.geometryInfo.increaseSizeAndGetOffset(sizeof(GlyphInfo), alignof(GlyphInfo));
	void* dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
	memcpy(dst, &glyphInfo, sizeof(GlyphInfo));

	// Push Indices, remove later when compute fills this
	uint32_t* indexBufferToBeFilled = resourcesCollection.indexBuffer.increaseCountAndGetPtr(6u * 1u);
	const uint32_t startObj = resourcesCollection.drawObjects.getCount();
	uint32_t i = 0u;
	indexBufferToBeFilled[i*6]		= (startObj+i)*4u + 1u;
	indexBufferToBeFilled[i*6 + 1u]	= (startObj+i)*4u + 0u;
	indexBufferToBeFilled[i*6 + 2u]	= (startObj+i)*4u + 2u;
	indexBufferToBeFilled[i*6 + 3u]	= (startObj+i)*4u + 1u;
	indexBufferToBeFilled[i*6 + 4u]	= (startObj+i)*4u + 2u;
	indexBufferToBeFilled[i*6 + 5u]	= (startObj+i)*4u + 3u;

	// Add DrawObjs
	DrawObject* drawObjectsToBeFilled = resourcesCollection.drawObjects.increaseCountAndGetPtr(1u);
	DrawObject drawObj = {};
	drawObj.mainObjIndex = mainObjIdx;
	drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::FONT_GLYPH) | (0 << 16));
	drawObj.geometryAddress = geometryBufferOffset;
	drawObjectsToBeFilled[0u] = drawObj;

	return true;
}

bool DrawResourcesFiller::addGridDTM_Internal(const GridDTMInfo& gridDTMInfo, uint32_t mainObjIdx)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();

	const uint32_t uploadableObjects = (remainingResourcesSize) / (sizeof(GridDTMInfo) + sizeof(DrawObject) + sizeof(uint32_t) * 6u);
	// TODO[ERFAN]: later take into account: our maximum indexable vertex 

	if (uploadableObjects <= 0u)
		return false;

	// Add Geometry
	size_t geometryBufferOffset = resourcesCollection.geometryInfo.increaseSizeAndGetOffset(sizeof(GridDTMInfo), alignof(GridDTMInfo));
	void* dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
	memcpy(dst, &gridDTMInfo, sizeof(GridDTMInfo));

	// Push Indices, remove later when compute fills this
	uint32_t* indexBufferToBeFilled = resourcesCollection.indexBuffer.increaseCountAndGetPtr(6u);
	const uint32_t startObj = resourcesCollection.drawObjects.getCount();
	uint32_t i = 0u;
	indexBufferToBeFilled[i * 6] = (startObj + i) * 4u + 1u;
	indexBufferToBeFilled[i * 6 + 1u] = (startObj + i) * 4u + 0u;
	indexBufferToBeFilled[i * 6 + 2u] = (startObj + i) * 4u + 2u;
	indexBufferToBeFilled[i * 6 + 3u] = (startObj + i) * 4u + 1u;
	indexBufferToBeFilled[i * 6 + 4u] = (startObj + i) * 4u + 2u;
	indexBufferToBeFilled[i * 6 + 5u] = (startObj + i) * 4u + 3u;

	// Add DrawObjs
	DrawObject* drawObjectsToBeFilled = resourcesCollection.drawObjects.increaseCountAndGetPtr(1u);
	DrawObject drawObj = {};
	drawObj.mainObjIndex = mainObjIdx;
	drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::GRID_DTM) | (0 << 16));
	drawObj.geometryAddress = geometryBufferOffset;
	drawObjectsToBeFilled[0u] = drawObj;

	return true;
}

bool DrawResourcesFiller::addImageObject_Internal(const ImageObjectInfo& imageObjectInfo, uint32_t mainObjIdx)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();

	const uint32_t uploadableObjects = (remainingResourcesSize) / (sizeof(ImageObjectInfo) + sizeof(DrawObject) + sizeof(uint32_t) * 6u);
	// TODO[ERFAN]: later take into account: our maximum indexable vertex 

	if (uploadableObjects <= 0u)
		return false;

	// Add Geometry
	size_t geometryBufferOffset = resourcesCollection.geometryInfo.increaseSizeAndGetOffset(sizeof(ImageObjectInfo), alignof(ImageObjectInfo));
	void* dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
	memcpy(dst, &imageObjectInfo, sizeof(ImageObjectInfo));

	// Push Indices, remove later when compute fills this
	uint32_t* indexBufferToBeFilled = resourcesCollection.indexBuffer.increaseCountAndGetPtr(6u * 1u);
	const uint32_t startObj = resourcesCollection.drawObjects.getCount();
	uint32_t i = 0u;
	indexBufferToBeFilled[i * 6] = (startObj + i) * 4u + 1u;
	indexBufferToBeFilled[i * 6 + 1u] = (startObj + i) * 4u + 0u;
	indexBufferToBeFilled[i * 6 + 2u] = (startObj + i) * 4u + 2u;
	indexBufferToBeFilled[i * 6 + 3u] = (startObj + i) * 4u + 1u;
	indexBufferToBeFilled[i * 6 + 4u] = (startObj + i) * 4u + 2u;
	indexBufferToBeFilled[i * 6 + 5u] = (startObj + i) * 4u + 3u;

	// Add DrawObjs
	DrawObject* drawObjectsToBeFilled = resourcesCollection.drawObjects.increaseCountAndGetPtr(1u);
	DrawObject drawObj = {};
	drawObj.mainObjIndex = mainObjIdx;
	drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::STATIC_IMAGE) | (0 << 16)); // TODO: use custom pack/unpack function
	drawObj.geometryAddress = geometryBufferOffset;
	drawObjectsToBeFilled[0u] = drawObj;

	return true;
}

bool DrawResourcesFiller::addGeoreferencedImageInfo_Internal(const GeoreferencedImageInfo& georeferencedImageInfo, uint32_t mainObjIdx)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();

	const uint32_t uploadableObjects = (remainingResourcesSize) / (sizeof(GeoreferencedImageInfo) + sizeof(DrawObject) + sizeof(uint32_t) * 6u);
	// TODO[ERFAN]: later take into account: our maximum indexable vertex 

	if (uploadableObjects <= 0u)
		return false;

	// Add Geometry
	size_t geometryBufferOffset = resourcesCollection.geometryInfo.increaseSizeAndGetOffset(sizeof(GeoreferencedImageInfo), alignof(GeoreferencedImageInfo));
	void* dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
	memcpy(dst, &georeferencedImageInfo, sizeof(GeoreferencedImageInfo));

	// Push Indices, remove later when compute fills this
	uint32_t* indexBufferToBeFilled = resourcesCollection.indexBuffer.increaseCountAndGetPtr(6u * 1u);
	const uint32_t startObj = resourcesCollection.drawObjects.getCount();
	uint32_t i = 0u;
	indexBufferToBeFilled[i * 6] = (startObj + i) * 4u + 1u;
	indexBufferToBeFilled[i * 6 + 1u] = (startObj + i) * 4u + 0u;
	indexBufferToBeFilled[i * 6 + 2u] = (startObj + i) * 4u + 2u;
	indexBufferToBeFilled[i * 6 + 3u] = (startObj + i) * 4u + 1u;
	indexBufferToBeFilled[i * 6 + 4u] = (startObj + i) * 4u + 2u;
	indexBufferToBeFilled[i * 6 + 5u] = (startObj + i) * 4u + 3u;

	// Add DrawObjs
	DrawObject* drawObjectsToBeFilled = resourcesCollection.drawObjects.increaseCountAndGetPtr(1u);
	DrawObject drawObj = {};
	drawObj.mainObjIndex = mainObjIdx;
	drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::STREAMED_IMAGE) | (0 << 16)); // TODO: use custom pack/unpack function
	drawObj.geometryAddress = geometryBufferOffset;
	drawObjectsToBeFilled[0u] = drawObj;

	return true;
}

uint32_t DrawResourcesFiller::getImageIndexFromID(image_id imageID, const SIntendedSubmitInfo& intendedNextSubmit)
{
	uint32_t textureIdx = InvalidTextureIndex;
	CachedImageRecord* imageRef = imagesCache->get(imageID);
	if (imageRef)
	{
		textureIdx = imageRef->arrayIndex;
		imageRef->lastUsedFrameIndex = currentFrameIndex; // update this because the texture will get used on the next frane
	}
	return textureIdx;
}

void DrawResourcesFiller::evictImage_SubmitIfNeeded(image_id imageID, const CachedImageRecord& evicted, SIntendedSubmitInfo& intendedNextSubmit)
{
	if (evicted.arrayIndex == InvalidTextureIndex)
	{
		m_logger.log("evictImage_SubmitIfNeeded: `evicted.arrayIndex == InvalidTextureIndex` is true, shouldn't happen under normal circumstances.", nbl::system::ILogger::ELL_WARNING);
		_NBL_DEBUG_BREAK_IF(true);
		return;
	}
	// Later used to release the image's memory range.
	core::smart_refctd_ptr<ImageCleanup> cleanupObject = core::make_smart_refctd_ptr<ImageCleanup>();
	cleanupObject->imagesMemorySuballocator = imagesMemorySubAllocator;
	cleanupObject->addr = evicted.allocationOffset;
	cleanupObject->size = evicted.allocationSize;

	const bool imageUsedForNextIntendedSubmit = (evicted.lastUsedFrameIndex == currentFrameIndex);

	// NOTE: `deallocationWaitInfo` is crucial for both paths, we need to make sure we'll write to a descriptor arrayIndex when it's 100% done with previous usages.
	if (imageUsedForNextIntendedSubmit)
	{
		// The evicted image is scheduled for use in the upcoming submit.
		// To avoid rendering artifacts, we must flush the current draw queue now.
		// After submission, we reset state so that data referencing the evicted slot can be re-uploaded.
		submitDraws(intendedNextSubmit);
		reset(); // resets everything, things referenced through mainObj and other shit will be pushed again through acquireXXX_SubmitIfNeeded

		// Prepare wait info to defer index deallocation until the GPU has finished using the resource.
		// we wait on the signal semaphore for the submit we just did above.
		ISemaphore::SWaitInfo deallocationWaitInfo = { .semaphore = intendedNextSubmit.scratchSemaphore.semaphore, .value = intendedNextSubmit.scratchSemaphore.value };
		suballocatedDescriptorSet->multi_deallocate(imagesArrayBinding, 1u, &evicted.arrayIndex, deallocationWaitInfo, &cleanupObject.get());
	}
	else
	{
		// The image is not used in the current frame, so we can deallocate without submitting any draws.
		// Still wait on the semaphore to ensure past GPU usage is complete.
		// TODO: We don't know which semaphore value the frame with `evicted.lastUsedFrameIndex` index was submitted with, so we wait for the worst case value conservatively, which is the immediate prev submit.
		ISemaphore::SWaitInfo deallocationWaitInfo = { .semaphore = intendedNextSubmit.scratchSemaphore.semaphore, .value = intendedNextSubmit.scratchSemaphore.value };
		suballocatedDescriptorSet->multi_deallocate(imagesArrayBinding, 1u, &evicted.arrayIndex, deallocationWaitInfo, &cleanupObject.get());
	}
}

DrawResourcesFiller::ImageAllocateResults DrawResourcesFiller::tryCreateAndAllocateImage_SubmitIfNeeded(
	const nbl::asset::IImage::SCreationParams& imageParams,
	const asset::E_FORMAT imageViewFormatOverride,
	nbl::video::SIntendedSubmitInfo& intendedNextSubmit,
	std::string imageDebugName)
{
	ImageAllocateResults ret = {};

	auto* device = m_utilities->getLogicalDevice();
	auto* physDev = m_utilities->getLogicalDevice()->getPhysicalDevice();

	bool alreadyBlockedForDeferredFrees = false;

	// Attempt to create a GPU image and corresponding image view for this texture.
	// If creation or memory allocation fails (likely due to VRAM exhaustion),
	// we'll evict another texture from the LRU cache and retry until successful, or until only the currently-cachedImageRecord image remains.
	while (imagesCache->size() > 0u)
	{
		// Try creating the image and allocating memory for it:
		nbl::video::IGPUImage::SCreationParams params = {};
		params = imageParams;
		
		if (imageViewFormatOverride != asset::E_FORMAT::EF_COUNT && imageViewFormatOverride != imageParams.format)
		{
			params.viewFormats.set(static_cast<size_t>(imageViewFormatOverride), true);
			params.flags |= asset::IImage::E_CREATE_FLAGS::ECF_MUTABLE_FORMAT_BIT;
		}
		auto gpuImage = device->createImage(std::move(params));

		if (gpuImage)
		{
			nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements gpuImageMemoryRequirements = gpuImage->getMemoryReqs();
			uint32_t actualAlignment = 1u << gpuImageMemoryRequirements.alignmentLog2;
			const bool imageMemoryRequirementsMatch =
				(physDev->getDeviceLocalMemoryTypeBits() & gpuImageMemoryRequirements.memoryTypeBits) != 0 && // should have device local memory compatible
				(gpuImageMemoryRequirements.requiresDedicatedAllocation == false) && // should not require dedicated allocation
				((ImagesMemorySubAllocator::MaxMemoryAlignment % actualAlignment) == 0u); // should be consistent with our suballocator's max alignment

			if (imageMemoryRequirementsMatch)
			{
				ret.allocationOffset = imagesMemorySubAllocator->allocate(gpuImageMemoryRequirements.size, 1u << gpuImageMemoryRequirements.alignmentLog2);
				const bool allocationFromImagesMemoryArenaSuccessfull = ret.allocationOffset != ImagesMemorySubAllocator::InvalidAddress;
				if (allocationFromImagesMemoryArenaSuccessfull)
				{
					ret.allocationSize = gpuImageMemoryRequirements.size;
					nbl::video::ILogicalDevice::SBindImageMemoryInfo bindImageMemoryInfo =
					{
						.image = gpuImage.get(),
						.binding = { .memory = imagesMemoryArena.memory.get(), .offset = imagesMemoryArena.offset + ret.allocationOffset }
					};
					const bool boundToMemorySuccessfully = device->bindImageMemory({ &bindImageMemoryInfo, 1u });
					if (boundToMemorySuccessfully)
					{
						gpuImage->setObjectDebugName(imageDebugName.c_str());
						IGPUImageView::SCreationParams viewParams = {
							.image = gpuImage,
							.viewType = IGPUImageView::ET_2D,
							.format = (imageViewFormatOverride == asset::E_FORMAT::EF_COUNT) ? gpuImage->getCreationParameters().format : imageViewFormatOverride
						};

						const uint32_t channelCount = nbl::asset::getFormatChannelCount(viewParams.format);
						if (channelCount == 1u)
						{
							// for rendering grayscale:
							viewParams.components.r = nbl::asset::IImageViewBase::SComponentMapping::E_SWIZZLE::ES_R;
							viewParams.components.g = nbl::asset::IImageViewBase::SComponentMapping::E_SWIZZLE::ES_R;
							viewParams.components.b = nbl::asset::IImageViewBase::SComponentMapping::E_SWIZZLE::ES_R;
							viewParams.components.a = nbl::asset::IImageViewBase::SComponentMapping::E_SWIZZLE::ES_ONE;
						}

						ret.gpuImageView = device->createImageView(std::move(viewParams));
						if (ret.gpuImageView)
						{
							// SUCCESS!
							ret.gpuImageView->setObjectDebugName((imageDebugName + " View").c_str());
						}
						else
						{
							// irrecoverable error if simple image creation fails.
							m_logger.log("tryCreateAndAllocateImage_SubmitIfNeeded: gpuImageView creation failed, that's rare and irrecoverable when adding a new image.", nbl::system::ILogger::ELL_ERROR);
							_NBL_DEBUG_BREAK_IF(true);
						}

						// succcessful with everything, just break and get out of this retry loop
						break;
					}
					else
					{
						// irrecoverable error if simple bindImageMemory fails.
						m_logger.log("tryCreateAndAllocateImage_SubmitIfNeeded: bindImageMemory failed, that's irrecoverable when adding a new image.", nbl::system::ILogger::ELL_ERROR);
						_NBL_DEBUG_BREAK_IF(true);
						break;
					}
				}
				else
				{
					// printf(std::format("Allocation Failed, Trying again, ImageID={} Size={} \n", imageID, gpuImageMemoryRequirements.size).c_str());
					// recoverable error when allocation fails, we don't log anything, next code will try evicting other images and retry
				}
			}
			else
			{
				m_logger.log("tryCreateAndAllocateImage_SubmitIfNeeded: memory requirements of the gpu image doesn't match our preallocated device memory, that's irrecoverable when adding a new image.", nbl::system::ILogger::ELL_ERROR);
				_NBL_DEBUG_BREAK_IF(true);
				break;
			}
		}
		else
		{
			m_logger.log("tryCreateAndAllocateImage_SubmitIfNeeded: gpuImage creation failed, that's irrecoverable when adding a new image.", nbl::system::ILogger::ELL_ERROR);
			_NBL_DEBUG_BREAK_IF(true);
			break;
		}

		// Getting here means we failed creating or allocating the image, evict and retry.


		// If imageCache size is 1 it means there is nothing else to evict, but there may still be already evicts/frees queued up.
		// `cull_frees` will make sure all pending deallocations will be blocked for.
		if (imagesCache->size() == 1u && alreadyBlockedForDeferredFrees)
		{
			// We give up, it's really nothing we can do, no image to evict (alreadyBlockedForDeferredFrees==1) and no more memory to free up (alreadyBlockedForDeferredFrees).
			// We probably have evicted almost every other texture except the one we just allocated an index for. 
			// This is most likely due to current image memory requirement being greater than the whole memory allocated for all images
			m_logger.log("tryCreateAndAllocateImage_SubmitIfNeeded: failed allocating an image, there is nothing more from mcache to evict, the current memory requirement is simply greater than the whole memory allocated for all images.", nbl::system::ILogger::ELL_ERROR);
			_NBL_DEBUG_BREAK_IF(true);
			break;
		}

		if (imagesCache->size() > 1u)
		{
			const image_id evictionCandidate = imagesCache->select_eviction_candidate();
			CachedImageRecord* imageRef = imagesCache->peek(evictionCandidate);
			if (imageRef)
				evictImage_SubmitIfNeeded(evictionCandidate, *imageRef, intendedNextSubmit);
			imagesCache->erase(evictionCandidate);
		}

		while (suballocatedDescriptorSet->cull_frees()) {}; // to make sure deallocation requests in eviction callback are blocked for.
		alreadyBlockedForDeferredFrees = true;

		// we don't hold any references to the GPUImageView or GPUImage so descriptor binding will be the last reference
		// hopefully by here the suballocated descriptor set freed some VRAM by dropping the image last ref and it's dedicated allocation.
	}

	return ret;
}

ImageType DrawResourcesFiller::determineGeoreferencedImageCreationParams(nbl::asset::IImage::SCreationParams& outImageParams, const GeoreferencedImageParams& params)
{
	// Decide whether the image can reside fully into memory rather than get streamed.
	// TODO: Improve logic, currently just a simple check to see if the full-screen image has more pixels that viewport or not
	// TODO: add criterial that the size of the full-res image shouldn't  consume more than 30% of the total memory arena for images (if we allowed larger than viewport extents)
	const bool betterToResideFullyInMem = params.imageExtents.x * params.imageExtents.y <= params.viewportExtents.x * params.viewportExtents.y;

	ImageType imageType;

	if (betterToResideFullyInMem)
		imageType = ImageType::GEOREFERENCED_FULL_RESOLUTION;
	else
		imageType = ImageType::GEOREFERENCED_STREAMED;

	outImageParams.type = asset::IImage::ET_2D;
	outImageParams.samples = asset::IImage::ESCF_1_BIT;
	outImageParams.format = params.format;

	if (imageType == ImageType::GEOREFERENCED_FULL_RESOLUTION)
	{
		outImageParams.extent = { params.imageExtents.x, params.imageExtents.y, 1u };
	}
	else
	{
		// Enough to cover twice the viewport at mip 0 (so that when zooming out to mip 1 the whole viewport still gets covered with mip 0 tiles) 
		// and in any rotation (taking the longest side suffices). Can be increased to avoid frequent tile eviction when moving the camera at mip close to 1
		const uint32_t diagonal = static_cast<uint32_t>(nbl::hlsl::ceil(
															nbl::hlsl::sqrt(static_cast<float32_t>(params.viewportExtents.x * params.viewportExtents.x 
																								   + params.viewportExtents.y * params.viewportExtents.y))
															)
														);
		const uint32_t gpuImageSidelength = 2 * core::roundUp(diagonal, GeoreferencedImageTileSize) + GeoreferencedImagePaddingTiles * GeoreferencedImageTileSize;
		outImageParams.extent = { gpuImageSidelength, gpuImageSidelength, 1u };
	}

	outImageParams.mipLevels = 2u;
	outImageParams.arrayLayers = 1u;

	return imageType;
}

void DrawResourcesFiller::setGlyphMSDFTextureFunction(const GetGlyphMSDFTextureFunc& func)
{
	getGlyphMSDF = func;
}

void DrawResourcesFiller::setHatchFillMSDFTextureFunction(const GetHatchFillPatternMSDFTextureFunc& func)
{
	getHatchFillPatternMSDF = func;
}

void DrawResourcesFiller::markFrameUsageComplete(uint64_t drawSubmitWaitValue)
{
	currentFrameIndex++;
	// TODO[LATER]: take into account that currentFrameIndex was submitted with drawSubmitWaitValue; Use that value when deallocating the resources marked with this frame index
	//				Currently, for evictions the worst case value will be waited for, as there is no way yet to know which semaphoroe value will signal the completion of the (to be evicted) resource's usage
}

uint32_t DrawResourcesFiller::getMSDFIndexFromInputInfo(const MSDFInputInfo& msdfInfo, const SIntendedSubmitInfo& intendedNextSubmit)
{
	uint32_t textureIdx = InvalidTextureIndex;
	MSDFReference* tRef = msdfLRUCache->get(msdfInfo);
	if (tRef)
	{
		textureIdx = tRef->alloc_idx;
		tRef->lastUsedFrameIndex = currentFrameIndex; // update this because the texture will get used on the next frame
	}
	return textureIdx;
}

uint32_t DrawResourcesFiller::addMSDFTexture(const MSDFInputInfo& msdfInput, core::smart_refctd_ptr<ICPUImage>&& cpuImage, SIntendedSubmitInfo& intendedNextSubmit)
{
	if (!cpuImage)
	{
		m_logger.log("addMSDFTexture: cpuImage is nullptr.", nbl::system::ILogger::ELL_ERROR);
		return InvalidTextureIndex;
	}

	const auto cpuImageSize = cpuImage->getMipSize(0);
	const bool sizeMatch = cpuImageSize.x == getMSDFResolution().x && cpuImageSize.y == getMSDFResolution().y && cpuImageSize.z == 1u;
	if (!sizeMatch)
	{
		m_logger.log("addMSDFTexture: cpuImage size doesn't match with msdf array image.", nbl::system::ILogger::ELL_ERROR);
		return InvalidTextureIndex;
	}

	/*
	 * The `msdfTextureArrayIndexAllocator` manages indices (slots) into a texture array for MSDF images.
	 * When all slots are occupied, the least recently used entry is evicted via `msdfLRUCache`.
	 * This callback is invoked on eviction, and must:
	 *   - Ensure safe deallocation of the slot.
	 *   - Submit any pending draw calls if the evicted MSDF was scheduled to be used in the upcoming submission.
	 */
	auto evictionCallback = [&](const MSDFReference& evicted)
	{
		// `deallocationWaitInfo` is used to prepare wait info to defer index deallocation until the GPU has finished using the resource.
		// NOTE: `deallocationWaitInfo` is currently *not* required for correctness because:
		//   - Both the image upload (msdfImagesState) and usage occur within the same timeline (`intendedNextSubmit`).
		//   - timeline semaphores guarantee proper ordering: the next submit's msdfImagesState will wait on the prior usage.
		//   - Therefore, we can safely overwrite or reallocate the slot without waiting for explicit GPU completion.
		//
		// However, this `deallocationWaitInfo` *will* become essential if we start interacting with MSDF images
		// outside the `intendedNextSubmit` timeline for example, issuing uploads via a transfer queue or using a separate command buffer and timeline.

		const bool imageUsedForNextIntendedSubmit = (evicted.lastUsedFrameIndex == currentFrameIndex);

		if (imageUsedForNextIntendedSubmit)
		{
			// The evicted image is scheduled for use in the upcoming submit.
			// To avoid rendering artifacts, we must flush the current draw queue now.
			// After submission, we reset state so that data referencing the evicted slot can be re-uploaded.
			submitDraws(intendedNextSubmit);
			reset(); // resets everything, things referenced through mainObj and other shit will be pushed again through acquireXXX_SubmitIfNeeded

			// Prepare wait info to defer index deallocation until the GPU has finished using the resource.
			// we wait on the signal semaphore for the submit we just did above.
			ISemaphore::SWaitInfo deallocationWaitInfo = { .semaphore = intendedNextSubmit.scratchSemaphore.semaphore, .value = intendedNextSubmit.scratchSemaphore.value };
			msdfTextureArrayIndexAllocator->multi_deallocate(1u, &evicted.alloc_idx, deallocationWaitInfo);
		} 
		else
		{
			// The image is not used in the current frame, so we can deallocate without submitting any draws.
			// Still wait on the semaphore to ensure past GPU usage is complete.
			// TODO: We don't know which semaphore value the frame with `evicted.lastUsedFrameIndex` index was submitted with, so we wait for the worst case value which is the immediate prev submit (scratchSemaphore.value).
			ISemaphore::SWaitInfo deallocationWaitInfo = { .semaphore = intendedNextSubmit.scratchSemaphore.semaphore, .value = intendedNextSubmit.scratchSemaphore.value };
			msdfTextureArrayIndexAllocator->multi_deallocate(1u, &evicted.alloc_idx, deallocationWaitInfo);
		}
		
		// Clear CPU-side metadata associated with the evicted slot.
		msdfImagesState[evicted.alloc_idx].evict();
	};
	
	// We pass nextSemaValue instead of constructing a new MSDFReference and passing it into `insert` that's because we might get a cache hit and only update the value of the nextSema
	MSDFReference* inserted = msdfLRUCache->insert(msdfInput, currentFrameIndex, evictionCallback);
	
	inserted->lastUsedFrameIndex = currentFrameIndex; // in case there was an eviction + auto-submit, we need to update AGAIN

	// if cachedImageRecord->alloc_idx was not InvalidTextureIndex then it means we had a cache hit and updated the value of our sema, in which case we don't queue anything for upload, and return the idx
	if (inserted->alloc_idx == InvalidTextureIndex)
	{
		// New insertion == cache miss happened and insertion was successfull
		inserted->alloc_idx = IndexAllocator::AddressAllocator::invalid_address;
		msdfTextureArrayIndexAllocator->multi_allocate(std::chrono::time_point<std::chrono::steady_clock>::max(), 1u, &inserted->alloc_idx); // if the prev submit causes DEVICE_LOST then we'll get a deadlock here since we're using max timepoint

		if (inserted->alloc_idx != IndexAllocator::AddressAllocator::invalid_address)
		{
			// We stage msdfImagesState, pushMSDFImagesUploads will push it into GPU
			msdfImagesState[inserted->alloc_idx].image = std::move(cpuImage);
			msdfImagesState[inserted->alloc_idx].uploadedToGPU = false;
		}
		else
		{
			m_logger.log("addMSDFTexture: index allocation failed.", nbl::system::ILogger::ELL_ERROR);
			inserted->alloc_idx = InvalidTextureIndex;
		}
	}
	
	assert(inserted->alloc_idx != InvalidTextureIndex); // shouldn't happen, because we're using LRU cache, so worst case eviction will happen + multi-deallocate and next next multi_allocate should definitely succeed

	return inserted->alloc_idx;
}

void DrawResourcesFiller::flushDrawObjects()
{
	if (resourcesCollection.drawObjects.getCount() > drawObjectsFlushedToDrawCalls)
	{
		DrawCallData drawCall = {};
		drawCall.isDTMRendering = false;
		drawCall.drawObj.drawObjectStart = drawObjectsFlushedToDrawCalls;
		drawCall.drawObj.drawObjectCount = resourcesCollection.drawObjects.getCount() - drawObjectsFlushedToDrawCalls;
		drawCalls.push_back(drawCall);
		drawObjectsFlushedToDrawCalls = resourcesCollection.drawObjects.getCount();
	}
}

DrawResourcesFiller::TileUploadData DrawResourcesFiller::generateTileUploadData(const ImageType imageType, const float64_t3x3& NDCToWorld, GeoreferencedImageStreamingState* imageStreamingState)
{
	// I think eventually it's better to just transform georeferenced images that aren't big enough into static images and forget about them
	if (imageType == ImageType::GEOREFERENCED_FULL_RESOLUTION) //Pass imageID as parameter, down from the addGeoRef call
		return TileUploadData{ {}, imageStreamingState->georeferencedImageParams.worldspaceOBB };

	// Compute the mip level and tile range we would need to encompass the viewport
	// `viewportTileRange` is always should be a subset of `currentMappedRegion`, covering only the tiles visible in the viewport
	// This also computes the optimal mip level for these tiles (basically a measure of how zoomed in or out the viewport is from the image)
	GeoreferencedImageTileRange viewportTileRange = computeViewportTileRange(NDCToWorld, imageStreamingState);

	// Slide or remap the current mapped region to ensure the viewport falls inside it
	imageStreamingState->ensureMappedRegionCoversViewport(viewportTileRange);

	// DEBUG - Sampled mip level
	{
		// Get world coordinates for each corner of the mapped region
		const float32_t2 oneTileDirU = imageStreamingState->georeferencedImageParams.worldspaceOBB.dirU / float32_t(imageStreamingState->fullImageTileLength.x) * float32_t(1u << imageStreamingState->currentMappedRegion.baseMipLevel);
		const float32_t2 fullImageDirV = float32_t2(imageStreamingState->georeferencedImageParams.worldspaceOBB.dirU.y, -imageStreamingState->georeferencedImageParams.worldspaceOBB.dirU.x);
		const float32_t2 oneTileDirV = fullImageDirV / float32_t(imageStreamingState->fullImageTileLength.y) * float32_t(1u << imageStreamingState->currentMappedRegion.baseMipLevel);
		float64_t2 topLeftMappedRegionWorld = imageStreamingState->georeferencedImageParams.worldspaceOBB.topLeft;
		topLeftMappedRegionWorld += oneTileDirU * float32_t(imageStreamingState->currentMappedRegion.topLeftTile.x) + oneTileDirV * float32_t(imageStreamingState->currentMappedRegion.topLeftTile.y);
		const uint32_t2 mappedRegionTileLength = imageStreamingState->currentMappedRegion.bottomRightTile - imageStreamingState->currentMappedRegion.topLeftTile + uint32_t2(1, 1);
		float64_t2 bottomRightMappedRegionWorld = topLeftMappedRegionWorld;
		bottomRightMappedRegionWorld += oneTileDirU * float32_t(mappedRegionTileLength.x) + oneTileDirV * float32_t(mappedRegionTileLength.y);

		// With the above, get an affine transform that maps points in worldspace to their pixel coordinates in the mapped region tile space. This can be done by mapping
		// `topLeftMappedRegionWorld -> (0,0)` and `bottomRightMappedRegionWorld -> mappedRegionPixelLength - 1`
		const uint32_t2 mappedRegionPixelLength = GeoreferencedImageTileSize * mappedRegionTileLength;
		
		// 1. Displacement
		// Multiplying a (homogenous) point p by this matrix yields the displacement vector `p - topLeftMappedRegionWorld`
		float64_t2x3 displacementMatrix(1., 0., -topLeftMappedRegionWorld.x, 0., 1., -topLeftMappedRegionWorld.y);

		// 2. Change of Basis. We again abuse the fact that the basis vectors are orthogonal
		float64_t2 dirU = oneTileDirU * float32_t(mappedRegionTileLength.x);
		float64_t2 dirV = oneTileDirV * float32_t(mappedRegionTileLength.y);
		float64_t dirULengthSquared = nbl::hlsl::dot(dirU, dirU);
		float64_t dirVLengthSquared = nbl::hlsl::dot(dirV, dirV);
		float64_t2 firstRow = dirU / dirULengthSquared;
		float64_t2 secondRow = dirV / dirVLengthSquared;
		float64_t2x2 changeOfBasisMatrix(firstRow, secondRow);

		// 3. Rescaling. The above matrix yields uv coordinates in the rectangle spanned by the mapped region. To get pixel coordinates, we simply multiply each coordinate by
		// how many pixels they span in the gpu image
		float64_t2x2 scalingMatrix(mappedRegionTileLength.x * GeoreferencedImageTileSize, 0.0, 0.0, mappedRegionTileLength.y * GeoreferencedImageTileSize);

		// Put them all together
		float64_t2x3 toPixelCoordsMatrix = nbl::hlsl::mul(scalingMatrix, nbl::hlsl::mul(changeOfBasisMatrix, displacementMatrix));

		// These are vulkan standard, might be different in n4ce!
		constexpr static float64_t3 topLeftViewportNDC = float64_t3(-1.0, -1.0, 1.0);
		constexpr static float64_t3 topRightViewportNDC = float64_t3(1.0, -1.0, 1.0);
		constexpr static float64_t3 bottomLeftViewportNDC = float64_t3(-1.0, 1.0, 1.0);
		constexpr static float64_t3 bottomRightViewportNDC = float64_t3(1.0, 1.0, 1.0);

		// Map viewport points to world
		const float64_t3 topLeftViewportWorld = nbl::hlsl::mul(NDCToWorld, topLeftViewportNDC);
		const float64_t3 topRightViewportWorld = nbl::hlsl::mul(NDCToWorld, topRightViewportNDC);
		const float64_t3 bottomLeftViewportWorld = nbl::hlsl::mul(NDCToWorld, bottomLeftViewportNDC);

		// Get pixel coordinates vectors for each side
		const float64_t2 viewportWidthPixelLengthVector = nbl::hlsl::mul(toPixelCoordsMatrix, topRightViewportWorld - topLeftViewportWorld);
		const float64_t2 viewportHeightPixelLengthVector = nbl::hlsl::mul(toPixelCoordsMatrix, bottomLeftViewportWorld - topLeftViewportWorld);

		// Get pixel length for each of these vectors
		const auto viewportWidthPixelLength = nbl::hlsl::length(viewportWidthPixelLengthVector);
		const auto viewportHeightPixelLength = nbl::hlsl::length(viewportHeightPixelLengthVector);
		
		// Mip is decided based on max of these
		float64_t pixelRatio = nbl::hlsl::max(viewportWidthPixelLength / imageStreamingState->georeferencedImageParams.viewportExtents.x, viewportHeightPixelLength / imageStreamingState->georeferencedImageParams.viewportExtents.y);
		pixelRatio = pixelRatio < 1.0 ? 1.0 : pixelRatio;

		std::cout << "Sampled mip level: " << nbl::hlsl::log2(pixelRatio) << std::endl;
	}
		
	// We need to make every tile that covers the viewport resident. We reserve the amount of tiles needed for upload.
	core::vector<StreamedImageCopy> tiles;
	auto tilesToLoad = imageStreamingState->tilesToLoad(viewportTileRange);
	tiles.reserve(tilesToLoad.size());

	for (auto [imageTileIndex, gpuImageTileIndex] : tilesToLoad)
	{
		uint32_t2 gpuMip0Texels(GeoreferencedImageTileSize, GeoreferencedImageTileSize);
		core::smart_refctd_ptr<ICPUBuffer> gpuMip0Tile = nullptr;
		core::smart_refctd_ptr<ICPUBuffer> gpuMip1Tile = nullptr;

		{
			uint32_t2 georeferencedImageMip0SampledTexels = uint32_t2(GeoreferencedImageTileSize, GeoreferencedImageTileSize) << viewportTileRange.baseMipLevel;
			const uint32_t2 georeferencedImageMip0SamplingOffset = (imageTileIndex * GeoreferencedImageTileSize) << viewportTileRange.baseMipLevel;
			const uint32_t2 lastTileIndex = imageStreamingState->getLastTileIndex(viewportTileRange.baseMipLevel);

			// If on the last tile, we might not load a full `GeoreferencedImageTileSize x GeoreferencedImageTileSize` tile, so we figure out how many pixels to load in this case to have
			// minimal artifacts and no stretching
			if (imageTileIndex.x == lastTileIndex.x)
			{
				georeferencedImageMip0SampledTexels.x = imageStreamingState->georeferencedImageParams.imageExtents.x - georeferencedImageMip0SamplingOffset.x;
				uint32_t gpuMip1Texels = georeferencedImageMip0SampledTexels.x >> (viewportTileRange.baseMipLevel + 1);
				gpuMip0Texels.x = 2 * gpuMip1Texels;
				imageStreamingState->lastImageTileFractionalSpan.x = float32_t(gpuMip0Texels.x) / GeoreferencedImageTileSize;
			}
			if (imageTileIndex.y == lastTileIndex.y)
			{
				georeferencedImageMip0SampledTexels.y = imageStreamingState->georeferencedImageParams.imageExtents.y - georeferencedImageMip0SamplingOffset.y;
				uint32_t gpuMip1Texels = georeferencedImageMip0SampledTexels.y >> (viewportTileRange.baseMipLevel + 1);
				gpuMip0Texels.y = 2 * gpuMip1Texels;
				imageStreamingState->lastImageTileFractionalSpan.y = float32_t(gpuMip0Texels.y) / GeoreferencedImageTileSize;
			}

			// If the last tile is too small just ignore it - given the way we set up stuff it's valid to check if these floats are exactly equal to 0, 
			// they're always a fraction of the form `x / GeoreferencedImageTileSize` with `0 <= x <= GeoreferencedImageTileSize` and `GeoreferencedImageTileSize` is PoT
			// If this looks bad we can do fractional pixelage by moving the uv an even tinier amount but at high zoom levels it should be imperceptible
			if ((imageStreamingState->lastImageTileFractionalSpan.x == 0.f) || (imageStreamingState->lastImageTileFractionalSpan.y == 0.f))
				continue;
			if (!georeferencedImageLoader->hasPrecomputedMips(imageStreamingState->georeferencedImageParams.storagePath))
			{
				gpuMip0Tile = georeferencedImageLoader->load(imageStreamingState->georeferencedImageParams.storagePath, (imageTileIndex * GeoreferencedImageTileSize) << viewportTileRange.baseMipLevel, georeferencedImageMip0SampledTexels, gpuMip0Texels);
				gpuMip1Tile = georeferencedImageLoader->load(imageStreamingState->georeferencedImageParams.storagePath, (imageTileIndex * GeoreferencedImageTileSize) << viewportTileRange.baseMipLevel, georeferencedImageMip0SampledTexels, gpuMip0Texels / 2u);
			}
			else
			{
				gpuMip0Tile = georeferencedImageLoader->load(imageStreamingState->georeferencedImageParams.storagePath, imageTileIndex * GeoreferencedImageTileSize, gpuMip0Texels, imageStreamingState->currentMappedRegion.baseMipLevel, false);
				gpuMip1Tile = georeferencedImageLoader->load(imageStreamingState->georeferencedImageParams.storagePath, imageTileIndex * GeoreferencedImageTileSizeMip1, gpuMip0Texels / 2u, imageStreamingState->currentMappedRegion.baseMipLevel, true);
			}
		}
		
		asset::IImage::SBufferCopy bufCopy;
		bufCopy.bufferOffset = 0;
		bufCopy.bufferRowLength = gpuMip0Texels.x;
		bufCopy.bufferImageHeight = 0;
		bufCopy.imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
		bufCopy.imageSubresource.mipLevel = 0u;
		bufCopy.imageSubresource.baseArrayLayer = 0u;
		bufCopy.imageSubresource.layerCount = 1u;
		uint32_t2 gpuImageOffset = gpuImageTileIndex * GeoreferencedImageTileSize;
		bufCopy.imageOffset = { gpuImageOffset.x, gpuImageOffset.y, 0u };
		bufCopy.imageExtent.width = gpuMip0Texels.x;
		bufCopy.imageExtent.height = gpuMip0Texels.y;
		bufCopy.imageExtent.depth = 1;

		tiles.emplace_back(imageStreamingState->georeferencedImageParams.format, std::move(gpuMip0Tile), std::move(bufCopy));

		// Upload the smaller tile to mip 1
		bufCopy = {};

		bufCopy.bufferOffset = 0;
		bufCopy.bufferRowLength = gpuMip0Texels.x / 2;
		bufCopy.bufferImageHeight = 0;
		bufCopy.imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
		bufCopy.imageSubresource.mipLevel = 1u;
		bufCopy.imageSubresource.baseArrayLayer = 0u;
		bufCopy.imageSubresource.layerCount = 1u;
		gpuImageOffset /= 2; // Half tile size!
		bufCopy.imageOffset = { gpuImageOffset.x, gpuImageOffset.y, 0u };
		bufCopy.imageExtent.width = gpuMip0Texels.x / 2;
		bufCopy.imageExtent.height = gpuMip0Texels.y / 2;
		bufCopy.imageExtent.depth = 1;

		tiles.emplace_back(imageStreamingState->georeferencedImageParams.format, std::move(gpuMip1Tile), std::move(bufCopy));

		// Mark tile as resident
		imageStreamingState->currentMappedRegionOccupancy[gpuImageTileIndex.x][gpuImageTileIndex.y] = true;
	}

	// Figure out an obb that covers only the currently loaded tiles
	OrientedBoundingBox2D viewportEncompassingOBB = imageStreamingState->georeferencedImageParams.worldspaceOBB;
	// The original image `dirU` corresponds to `maxImageTileIndices.x + 1` mip 0 tiles (provided it's exactly that length in tiles)
	// Dividing dirU by `maxImageTileIndices + (1,1)` we therefore get a vector that spans exactly one mip 0 tile (in the u direction) in worldspace. 
	// Multiplying that by `2^mipLevel` we get a vector that spans exactly one mip `mipLevel` tile (in the u direction)
	const float32_t2 oneTileDirU = imageStreamingState->georeferencedImageParams.worldspaceOBB.dirU / float32_t(imageStreamingState->fullImageTileLength.x) * float32_t(1u << imageStreamingState->currentMappedRegion.baseMipLevel);
	const float32_t2 fullImageDirV = float32_t2(imageStreamingState->georeferencedImageParams.worldspaceOBB.dirU.y, -imageStreamingState->georeferencedImageParams.worldspaceOBB.dirU.x) * imageStreamingState->georeferencedImageParams.worldspaceOBB.aspectRatio;
	const float32_t2 oneTileDirV = fullImageDirV / float32_t(imageStreamingState->fullImageTileLength.y) * float32_t(1u << imageStreamingState->currentMappedRegion.baseMipLevel);
	viewportEncompassingOBB.topLeft += oneTileDirU * float32_t(viewportTileRange.topLeftTile.x);
	viewportEncompassingOBB.topLeft += oneTileDirV * float32_t(viewportTileRange.topLeftTile.y);

	const uint32_t2 viewportTileLength = viewportTileRange.bottomRightTile - viewportTileRange.topLeftTile + uint32_t2(1, 1);
	// If the last tile is visible, we use the fractional span for the last tile. Otherwise it's just a normal tile
	const bool2 isLastTileVisible = imageStreamingState->isLastTileVisible(viewportTileRange.bottomRightTile);
	const float32_t2 lastGPUImageTileFractionalSpan = { isLastTileVisible.x ? imageStreamingState->lastImageTileFractionalSpan.x : 1.f, isLastTileVisible.y ? imageStreamingState->lastImageTileFractionalSpan.y : 1.f };

	viewportEncompassingOBB.dirU = oneTileDirU * (float32_t(viewportTileLength.x - 1u) + lastGPUImageTileFractionalSpan.x);
	viewportEncompassingOBB.aspectRatio = (float32_t(viewportTileLength.y - 1u) + lastGPUImageTileFractionalSpan.y) / (float32_t(viewportTileLength.x - 1u) + lastGPUImageTileFractionalSpan.x);

	// UV logic currently ONLY works when the image not only fits an integer amount of tiles, but also when it's a PoT amount of them
	// (this means every mip level also gets an integer amount of tiles).
	// When porting to n4ce, for the image to fit an integer amount of tiles (instead of rewriting the logic) we can just pad the right/bottom sides with alpha=0 pixels
	// The UV logic will have to change to consider what happens to the last loaded tile (or, alternatively, we can also fill the empty tiles with alpha=0 pixels)

	// Compute minUV, maxUV
	const float32_t2 uvPerTile = float32_t2(1.f, 1.f) / float32_t2(imageStreamingState->gpuImageSideLengthTiles, imageStreamingState->gpuImageSideLengthTiles);
	const float32_t2 minUV = uvPerTile * float32_t2(((viewportTileRange.topLeftTile - imageStreamingState->currentMappedRegion.topLeftTile) + imageStreamingState->gpuImageTopLeft) % imageStreamingState->gpuImageSideLengthTiles);
	float32_t2 maxUV = minUV + uvPerTile * float32_t2(viewportTileLength - 1u);
	// uvPerTile is the uv per GeoreferencedImageTileSize pixels. Since the last tile might not be fully resident with pixels, we don't add the uv for it above and add the proper uv it should be sampled at here
	maxUV += uvPerTile * lastGPUImageTileFractionalSpan;
	return TileUploadData{ std::move(tiles), viewportEncompassingOBB, minUV, maxUV };
}

GeoreferencedImageTileRange DrawResourcesFiller::computeViewportTileRange(const float64_t3x3& NDCToWorld, const GeoreferencedImageStreamingState* imageStreamingState)
{
	// These are vulkan standard, might be different in n4ce!
	constexpr static float64_t3 topLeftViewportNDC = float64_t3(-1.0, -1.0, 1.0);
	constexpr static float64_t3 topRightViewportNDC = float64_t3(1.0, -1.0, 1.0);
	constexpr static float64_t3 bottomLeftViewportNDC = float64_t3(-1.0, 1.0, 1.0);
	constexpr static float64_t3 bottomRightViewportNDC = float64_t3(1.0, 1.0, 1.0);

	// First get world coordinates for each of the viewport's corners
	const float64_t3 topLeftViewportWorld = nbl::hlsl::mul(NDCToWorld, topLeftViewportNDC);
	const float64_t3 topRightViewportWorld = nbl::hlsl::mul(NDCToWorld, topRightViewportNDC);
	const float64_t3 bottomLeftViewportWorld = nbl::hlsl::mul(NDCToWorld, bottomLeftViewportNDC);
	const float64_t3 bottomRightViewportWorld = nbl::hlsl::mul(NDCToWorld, bottomRightViewportNDC);

	// Then we get mip 0 tiles coordinates for each of them, into the image
	const float64_t2 topLeftTileLattice = imageStreamingState->transformWorldCoordsToTileCoords(topLeftViewportWorld, GeoreferencedImageTileSize);
	const float64_t2 topRightTileLattice = imageStreamingState->transformWorldCoordsToTileCoords(topRightViewportWorld, GeoreferencedImageTileSize);
	const float64_t2 bottomLeftTileLattice = imageStreamingState->transformWorldCoordsToTileCoords(bottomLeftViewportWorld, GeoreferencedImageTileSize);
	const float64_t2 bottomRightTileLattice = imageStreamingState->transformWorldCoordsToTileCoords(bottomRightViewportWorld, GeoreferencedImageTileSize);

	// Get the min and max of each lattice coordinate to get a bounding rectangle
	const float64_t2 minTop = nbl::hlsl::min(topLeftTileLattice, topRightTileLattice);
	const float64_t2 minBottom = nbl::hlsl::min(bottomLeftTileLattice, bottomRightTileLattice);
	const float64_t2 minAll = nbl::hlsl::min(minTop, minBottom);

	const float64_t2 maxTop = nbl::hlsl::max(topLeftTileLattice, topRightTileLattice);
	const float64_t2 maxBottom = nbl::hlsl::max(bottomLeftTileLattice, bottomRightTileLattice);
	// Edge case padding - there seems to be some numerical error going on when really close to tile boundaries
	const float64_t2 maxAll = nbl::hlsl::max(maxTop, maxBottom) + float64_t2(0.5, 0.5);

	// Floor them to get an integer coordinate (index) for the tiles they fall in
	int32_t2 minAllFloored = nbl::hlsl::floor(minAll);
	int32_t2 maxAllFloored = nbl::hlsl::floor(maxAll);
	
	// We're undoing a previous division. Could be avoided but won't restructure the code atp.
	// Here we compute how many image pixels each side of the viewport spans 
	const float64_t2 viewportSideUImageTexelsVector = float64_t(GeoreferencedImageTileSize) * (topRightTileLattice - topLeftTileLattice);
	const float64_t2 viewportSideVImageTexelsVector = float64_t(GeoreferencedImageTileSize) * (bottomLeftTileLattice - topLeftTileLattice);

	// WARNING: This assumes pixels in the image are the same size along each axis. If the image is nonuniformly scaled or sheared, I *think* it should not matter
	// (since the pixel span takes that transformation into account), BUT we have to check if we plan on allowing those
	// Compute the side vectors of the viewport in image pixel(texel) space.
	// These vectors represent how many image pixels each side of the viewport spans.
	// They correspond to the local axes of the mapped OBB (not the mapped region one, the viewport one) in texel coordinates.
	const float64_t viewportSideUImageTexels = nbl::hlsl::length(viewportSideUImageTexelsVector);
	const float64_t viewportSideVImageTexels = nbl::hlsl::length(viewportSideVImageTexelsVector);

	// Mip is decided based on max of these
	float64_t pixelRatio = nbl::hlsl::max(viewportSideUImageTexels / imageStreamingState->georeferencedImageParams.viewportExtents.x, 
										  viewportSideVImageTexels / imageStreamingState->georeferencedImageParams.viewportExtents.y);
	pixelRatio = pixelRatio < 1.0 ? 1.0 : pixelRatio;
	
	// DEBUG - Clamped at 0 for magnification
	{
		std::cout << "Real mip level:    " << nbl::hlsl::log2(pixelRatio) << std::endl;
	}
	
	GeoreferencedImageTileRange retVal = {};
	// Clamp mip level so we don't consider tiles that are too small along one dimension
	// If on a pathological case this gets too expensive because the GPU starts sampling a lot, we can consider changing this, but I doubt that will happen
	retVal.baseMipLevel = nbl::hlsl::min(nbl::hlsl::findMSB(uint32_t(nbl::hlsl::floor(pixelRatio))), int32_t(imageStreamingState->maxMipLevel));
	
	// Current tiles are measured in mip 0. We want the result to measure mip `retVal.baseMipLevel` tiles. Each next mip level divides by 2.
	minAllFloored >>= retVal.baseMipLevel;
	maxAllFloored >>= retVal.baseMipLevel;


	// Clamp them to reasonable tile indices
	int32_t2 lastTileIndex = imageStreamingState->getLastTileIndex(retVal.baseMipLevel);
	retVal.topLeftTile = nbl::hlsl::clamp(minAllFloored, int32_t2(0, 0), lastTileIndex);
	retVal.bottomRightTile = nbl::hlsl::clamp(maxAllFloored, int32_t2(0, 0), lastTileIndex);

	return retVal;
}