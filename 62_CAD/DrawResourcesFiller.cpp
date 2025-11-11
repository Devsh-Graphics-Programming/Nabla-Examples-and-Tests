#include "DrawResourcesFiller.h"

using namespace nbl;

DrawResourcesFiller::DrawResourcesFiller()
{}

DrawResourcesFiller::DrawResourcesFiller(smart_refctd_ptr<video::ILogicalDevice>&& device, smart_refctd_ptr<IUtilities>&& bufferUploadUtils, smart_refctd_ptr<IUtilities>&& imageUploadUtils, IQueue* copyQueue, core::smart_refctd_ptr<system::ILogger>&& logger) :
	m_device(std::move(device)),
	m_bufferUploadUtils(std::move(bufferUploadUtils)),
	m_imageUploadUtils(std::move(imageUploadUtils)),
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
	imagesDescriptorIndexAllocator = core::make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(descriptorSet));
}

bool DrawResourcesFiller::allocateDrawResources(ILogicalDevice* logicalDevice, size_t requiredImageMemorySize, size_t requiredBufferMemorySize, std::span<uint32_t> memoryTypeIndexTryOrder)
{
	// requiredImageMemorySize = core::alignUp(50'399'744 * 2, 1024);
	// single memory allocation sectioned into images+buffers (images start at offset=0)
	const size_t adjustedImagesMemorySize = core::alignUp(requiredImageMemorySize, GPUStructsMaxNaturalAlignment);
	const size_t adjustedBuffersMemorySize = core::max(requiredBufferMemorySize, getMinimumRequiredResourcesBufferSize());
	const size_t totalResourcesSize = adjustedImagesMemorySize + adjustedBuffersMemorySize;

	IGPUBuffer::SCreationParams resourcesBufferCreationParams = {};
	resourcesBufferCreationParams.size = adjustedBuffersMemorySize;
	resourcesBufferCreationParams.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INDEX_BUFFER_BIT;
	resourcesGPUBuffer = logicalDevice->createBuffer(std::move(resourcesBufferCreationParams));

	if (!resourcesGPUBuffer)
	{
		m_logger.log("Failed to create resourcesGPUBuffer.", nbl::system::ILogger::ELL_ERROR);
		return false;
	}

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

	video::IDeviceMemoryAllocator::SAllocation allocation = {};
	for (const auto& memoryTypeIdx : memoryTypeIndexTryOrder)
	{
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

	if (!allocation.isValid())
	{
		m_logger.log("Failed Allocation for draw resources!", nbl::system::ILogger::ELL_ERROR);
		return false;
	}

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

bool DrawResourcesFiller::allocateDrawResourcesWithinAvailableVRAM(ILogicalDevice* logicalDevice, size_t maxImageMemorySize, size_t maxBufferMemorySize, std::span<uint32_t> memoryTypeIndexTryOrder, uint32_t reductionPercent, uint32_t maxTries)
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
		if (allocateDrawResources(logicalDevice, currentImageSize, currentBufferSize, memoryTypeIndexTryOrder))
		{
			m_logger.log("Successfully allocated memory for images (%zu) and buffers (%zu).", system::ILogger::ELL_INFO, currentImageSize, currentBufferSize);
			return true;
		}

		m_logger.log("Allocation of memory for images(%zu) and buffers(%zu) failed; Reducing allocation size by %u%% and retrying...", system::ILogger::ELL_WARNING, currentImageSize, currentBufferSize, reductionPercent);
		currentBufferSize = (currentBufferSize * (100 - reductionPercent)) / 100;
		currentImageSize = (currentImageSize * (100 - reductionPercent)) / 100;
		numTries++;
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
	// imagesCache->logState(m_logger);
	
	// Check if image already exists and requires force update. We do this before insertion and updating `lastUsedFrameIndex` to get correct overflow-submit behaviour
	// otherwise we'd always overflow submit, even if not needed and image was not queued/intended to use in the next submit.
	CachedImageRecord* cachedImageRecord = imagesCache->get(staticImage.imageID);
	
	if (cachedImageRecord && cachedImageRecord->arrayIndex != InvalidTextureIndex && staticImage.forceUpdate)
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
				evictImage_SubmitIfNeeded(staticImage.imageID, *cachedImageRecord, intendedNextSubmit);
					
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

	// Try inserting or updating the image usage in the cache.
	// If the image is already present, updates its semaphore value.
	auto evictCallback = [&](image_id imageID, const CachedImageRecord& evicted) { evictImage_SubmitIfNeeded(imageID, evicted, intendedNextSubmit); };
	cachedImageRecord = imagesCache->insert(staticImage.imageID, currentFrameIndex, evictCallback);
	cachedImageRecord->lastUsedFrameIndex = currentFrameIndex; // in case there was an eviction + auto-submit, we need to update AGAIN


	// if cachedImageRecord->index was not InvalidTextureIndex then it means we had a cache hit and updated the value of our sema
	// in which case we don't queue anything for upload, and return the idx
	if (cachedImageRecord->arrayIndex == InvalidTextureIndex)
	{
		// This is a new image (cache miss). Allocate a descriptor index for it.
		cachedImageRecord->arrayIndex = video::SubAllocatedDescriptorSet::AddressAllocator::invalid_address;
		// Blocking allocation attempt; if the descriptor pool is exhausted, this may stall.
		imagesDescriptorIndexAllocator->multi_allocate(std::chrono::time_point<std::chrono::steady_clock>::max(), imagesArrayBinding, 1u, &cachedImageRecord->arrayIndex); // if the prev submit causes DEVICE_LOST then we'll get a deadlock here since we're using max timepoint
		cachedImageRecord->arrayIndexAllocatedUsingImageDescriptorIndexAllocator = true;

		if (cachedImageRecord->arrayIndex != video::SubAllocatedDescriptorSet::AddressAllocator::invalid_address)
		{
			auto* physDev = m_device->getPhysicalDevice();

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
				cachedImageRecord->currentLayout  = nbl::asset::IImage::LAYOUT::UNDEFINED;
				cachedImageRecord->lastUsedFrameIndex = currentFrameIndex; // there was an eviction + auto-submit, we need to update AGAIN
				cachedImageRecord->allocationOffset = allocResults.allocationOffset;
				cachedImageRecord->allocationSize = allocResults.allocationSize;
				cachedImageRecord->gpuImageView = allocResults.gpuImageView;
				cachedImageRecord->staticCPUImage = staticImage.cpuImage;
				cachedImageRecord->georeferencedImageState = nullptr;
				evictConflictingImagesInCache_SubmitIfNeeded(staticImage.imageID, *cachedImageRecord, intendedNextSubmit);
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
					imagesDescriptorIndexAllocator->multi_deallocate(imagesArrayBinding, 1u, &cachedImageRecord->arrayIndex, {});
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

uint32_t2 DrawResourcesFiller::computeStreamingImageExtentsForViewportCoverage(const uint32_t2 viewportExtents)
{
	const uint32_t diagonal = static_cast<uint32_t>(nbl::hlsl::ceil(
		nbl::hlsl::sqrt(static_cast<float32_t>(
			viewportExtents.x * viewportExtents.x + viewportExtents.y * viewportExtents.y))
	));

	const uint32_t gpuImageSidelength =
		2 * core::roundUp(diagonal, GeoreferencedImageTileSize) +
		GeoreferencedImagePaddingTiles * GeoreferencedImageTileSize;

	return { gpuImageSidelength, gpuImageSidelength };
}

nbl::core::smart_refctd_ptr<GeoreferencedImageStreamingState> DrawResourcesFiller::ensureGeoreferencedImageEntry(image_id imageID, const OrientedBoundingBox2D& worldSpaceOBB, const uint32_t2 currentViewportExtents, const float64_t3x3& ndcToWorldMat, const std::filesystem::path& storagePath)
{
	nbl::core::smart_refctd_ptr<GeoreferencedImageStreamingState> ret = nullptr;

	auto* physDev = m_device->getPhysicalDevice();

	if (!imageLoader)
	{
		m_logger.log("imageLoader is null/empty. make sure to register your loader!", nbl::system::ILogger::ELL_ERROR);
		return nullptr;
	}

	uint32_t2 fullResImageExtents = imageLoader->getExtents(storagePath);
	asset::E_FORMAT format = imageLoader->getFormat(storagePath);

	uint32_t2 gpuImageExtents = computeStreamingImageExtentsForViewportCoverage(currentViewportExtents);

	IGPUImage::SCreationParams gpuImageCreationParams = {};
	gpuImageCreationParams.type = asset::IImage::ET_2D;
	gpuImageCreationParams.samples = asset::IImage::ESCF_1_BIT;
	gpuImageCreationParams.format = format;
	gpuImageCreationParams.extent = { .width = gpuImageExtents.x, .height = gpuImageExtents.y, .depth = 1u };
	gpuImageCreationParams.mipLevels = 2u;
	gpuImageCreationParams.arrayLayers = 1u;

	gpuImageCreationParams.usage |= IGPUImage::EUF_TRANSFER_DST_BIT|IGPUImage::EUF_SAMPLED_BIT;
	// promote format because RGB8 and friends don't actually exist in HW
	{
		const IPhysicalDevice::SImageFormatPromotionRequest request = {
			.originalFormat = gpuImageCreationParams.format,
			.usages = IPhysicalDevice::SFormatImageUsages::SUsage(gpuImageCreationParams.usage)
		};
		gpuImageCreationParams.format = physDev->promoteImageFormat(request,gpuImageCreationParams.tiling);
	}
	
	CachedImageRecord* cachedImageRecord = imagesCache->get(imageID);
	if (!cachedImageRecord)
	{
		ret = nbl::core::make_smart_refctd_ptr<GeoreferencedImageStreamingState>();
		const bool initSuccess = ret->init(worldSpaceOBB, fullResImageExtents, format, storagePath);
		if (!initSuccess)
			m_logger.log("Failed to init GeoreferencedImageStreamingState!", nbl::system::ILogger::ELL_ERROR);
	}
	else
	{
		// StreamingState already in cache, we return it;
		if (!cachedImageRecord->georeferencedImageState)
			m_logger.log("image had entry in the cache but cachedImageRecord->georeferencedImageState was nullptr, this shouldn't happen!", nbl::system::ILogger::ELL_ERROR);
		ret = cachedImageRecord->georeferencedImageState;
	}
	
	// Update GeoreferencedImageState with new viewport width/height and requirements

	// width only because gpu image is square
	const uint32_t newGPUImageSideLengthTiles = gpuImageCreationParams.extent.width / GeoreferencedImageTileSize;
	
	// This will reset the residency state after a resize. it makes sense because when gpu image is resized, it's recreated and no previous tile is resident anymore
	// We don't copy tiles between prev/next resized image, we're more focused on optimizing pan/zoom with a fixed window size.
	if (ret->gpuImageSideLengthTiles != newGPUImageSideLengthTiles)
	{
		ret->gpuImageSideLengthTiles = newGPUImageSideLengthTiles;
		ret->ResetTileOccupancyState();
		ret->currentMappedRegionTileRange = { .baseMipLevel = std::numeric_limits<uint32_t>::max() };
	}
	
	ret->gpuImageCreationParams = std::move(gpuImageCreationParams);
	// Update with current viewport
	ret->updateStreamingStateForViewport(currentViewportExtents, ndcToWorldMat);

	return ret;
}

bool DrawResourcesFiller::launchGeoreferencedImageTileLoads(image_id imageID, GeoreferencedImageStreamingState* imageStreamingState, const WorldClipRect clipRect)
{
	if (!imageStreamingState)
	{
		m_logger.log("imageStreamingState is null/empty, make sure `ensureGeoreferencedImageEntry` was called beforehand!", nbl::system::ILogger::ELL_ERROR);
		assert(false);
		return false;
	}

	auto& thisImageQueuedCopies = streamedImageCopies[imageID];

	const auto& viewportTileRange = imageStreamingState->currentViewportTileRange;
	const uint32_t2 lastTileIndex = imageStreamingState->getLastTileIndex(viewportTileRange.baseMipLevel);

	// We need to make every tile that covers the viewport resident. We reserve the amount of tiles needed for upload.
	auto tilesToLoad = imageStreamingState->tilesToLoad();

	
	// m_logger.log(std::format("Tiles to Load = {}.", tilesToLoad.size()).c_str(), nbl::system::ILogger::ELL_INFO);


	const uint32_t2 imageExtents = imageStreamingState->fullResImageExtents;
	const std::filesystem::path imageStoragePath = imageStreamingState->storagePath;

	// Figure out worldspace coordinates for each of the tile's corners - these are used if there's a clip rect
	const float64_t2 imageTopLeft = imageStreamingState->worldspaceOBB.topLeft;
	const float64_t2 dirU = float64_t2(imageStreamingState->worldspaceOBB.dirU);
	const float64_t2 dirV = float64_t2(dirU.y, -dirU.x) * float64_t(imageStreamingState->worldspaceOBB.aspectRatio);
	const uint32_t tileMipLevel = imageStreamingState->currentViewportTileRange.baseMipLevel;

	uint32_t ignored = 0;
	for (auto [imageTileIndex, gpuImageTileIndex] : tilesToLoad)
	{
		// clip against current rect, if valid
		if (clipRect.minClip.x != std::numeric_limits<float64_t>::signaling_NaN())
		{
			float64_t2 topLeftWorld = imageTopLeft + dirU * (float64_t(GeoreferencedImageTileSize * imageTileIndex.x << tileMipLevel) / float64_t(imageExtents.x)) + dirV * (float64_t(GeoreferencedImageTileSize * imageTileIndex.y << tileMipLevel) / float64_t(imageExtents.y));
			float64_t2 topRightWorld = imageTopLeft + dirU * (float64_t(GeoreferencedImageTileSize * (imageTileIndex.x + 1) << tileMipLevel) / float64_t(imageExtents.x)) + dirV * (float64_t(GeoreferencedImageTileSize * imageTileIndex.y << tileMipLevel) / float64_t(imageExtents.y));
			float64_t2 bottomLeftWorld = imageTopLeft + dirU * (float64_t(GeoreferencedImageTileSize * imageTileIndex.x << tileMipLevel) / float64_t(imageExtents.x)) + dirV * (float64_t(GeoreferencedImageTileSize * (imageTileIndex.y + 1) << tileMipLevel) / float64_t(imageExtents.y));
			float64_t2 bottomRightWorld = imageTopLeft + dirU * (float64_t(GeoreferencedImageTileSize * (imageTileIndex.x + 1) << tileMipLevel) / float64_t(imageExtents.x)) + dirV * (float64_t(GeoreferencedImageTileSize * (imageTileIndex.y + 1) << tileMipLevel) / float64_t(imageExtents.y));

			float64_t minX = std::min({ topLeftWorld.x, topRightWorld.x, bottomLeftWorld.x, bottomRightWorld.x });
			float64_t minY = std::min({ topLeftWorld.y, topRightWorld.y, bottomLeftWorld.y, bottomRightWorld.y });
			float64_t maxX = std::max({ topLeftWorld.x, topRightWorld.x, bottomLeftWorld.x, bottomRightWorld.x });
			float64_t maxY = std::max({ topLeftWorld.y, topRightWorld.y, bottomLeftWorld.y, bottomRightWorld.y });
			
			// Check if the tile intersects clip rect at all. Note that y clips are inverted
			if (maxX < clipRect.minClip.x || minX > clipRect.maxClip.x || maxY < clipRect.maxClip.y || minY > clipRect.minClip.y)
				continue;
		}

		uint32_t2 targetExtentMip0(GeoreferencedImageTileSize, GeoreferencedImageTileSize);
		std::future<core::smart_refctd_ptr<ICPUBuffer>> gpuMip0Tile;
		std::future<core::smart_refctd_ptr<ICPUBuffer>> gpuMip1Tile;

		{
			uint32_t2 samplingExtentMip0 = uint32_t2(GeoreferencedImageTileSize, GeoreferencedImageTileSize) << viewportTileRange.baseMipLevel;
			const uint32_t2 samplingOffsetMip0 = (imageTileIndex * GeoreferencedImageTileSize) << viewportTileRange.baseMipLevel;

			// If on the last tile, we might not load a full `GeoreferencedImageTileSize x GeoreferencedImageTileSize` tile, so we figure out how many pixels to load in this case to have
			// minimal artifacts and no stretching
			if (imageTileIndex.x == lastTileIndex.x)
			{
				samplingExtentMip0.x = imageStreamingState->lastTileSamplingExtent.x;
				targetExtentMip0.x = imageStreamingState->lastTileTargetExtent.x;
				// If the last tile is too small just ignore it
				if (targetExtentMip0.x == 0u)
					continue;
			}
			if (imageTileIndex.y == lastTileIndex.y)
			{
				samplingExtentMip0.y = imageStreamingState->lastTileSamplingExtent.y;
				targetExtentMip0.y = imageStreamingState->lastTileTargetExtent.y;
				// If the last tile is too small just ignore it
				if (targetExtentMip0.y == 0u)
					continue;
			}

			if (!imageLoader->hasPrecomputedMips(imageStoragePath))
			{
				gpuMip0Tile = std::async(std::launch::async, [=, this]() {
					return imageLoader->load(imageStoragePath, samplingOffsetMip0, samplingExtentMip0, targetExtentMip0);
				});
				gpuMip1Tile = std::async(std::launch::async, [=, this]() {
					return imageLoader->load(imageStoragePath, samplingOffsetMip0, samplingExtentMip0, targetExtentMip0 / 2u);
				});
			}
			else
			{
				gpuMip0Tile = std::async(std::launch::async, [=, this]() {
					return imageLoader->load(imageStoragePath, imageTileIndex * GeoreferencedImageTileSize, targetExtentMip0, imageStreamingState->currentMappedRegionTileRange.baseMipLevel, false);
				});
				gpuMip1Tile = std::async(std::launch::async, [=, this]() {
					return imageLoader->load(imageStoragePath, imageTileIndex * GeoreferencedImageTileSizeMip1, targetExtentMip0 / 2u, imageStreamingState->currentMappedRegionTileRange.baseMipLevel, true);
				});
			}
		}

		asset::IImage::SBufferCopy bufCopy;
		bufCopy.bufferOffset = 0;
		bufCopy.bufferRowLength = targetExtentMip0.x;
		bufCopy.bufferImageHeight = 0;
		bufCopy.imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
		bufCopy.imageSubresource.mipLevel = 0u;
		bufCopy.imageSubresource.baseArrayLayer = 0u;
		bufCopy.imageSubresource.layerCount = 1u;
		uint32_t2 gpuImageOffset = gpuImageTileIndex * GeoreferencedImageTileSize;
		bufCopy.imageOffset = { gpuImageOffset.x, gpuImageOffset.y, 0u };
		bufCopy.imageExtent.width = targetExtentMip0.x;
		bufCopy.imageExtent.height = targetExtentMip0.y;
		bufCopy.imageExtent.depth = 1;

		thisImageQueuedCopies.emplace_back(imageStreamingState->sourceImageFormat, std::move(gpuMip0Tile), std::move(bufCopy));

		// Upload the smaller tile to mip 1
		bufCopy = {};

		bufCopy.bufferOffset = 0;
		bufCopy.bufferRowLength = targetExtentMip0.x / 2;
		bufCopy.bufferImageHeight = 0;
		bufCopy.imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
		bufCopy.imageSubresource.mipLevel = 1u;
		bufCopy.imageSubresource.baseArrayLayer = 0u;
		bufCopy.imageSubresource.layerCount = 1u;
		gpuImageOffset /= 2; // Half tile size!
		bufCopy.imageOffset = { gpuImageOffset.x, gpuImageOffset.y, 0u };
		bufCopy.imageExtent.width = targetExtentMip0.x / 2;
		bufCopy.imageExtent.height = targetExtentMip0.y / 2;
		bufCopy.imageExtent.depth = 1;

		thisImageQueuedCopies.emplace_back(imageStreamingState->sourceImageFormat, std::move(gpuMip1Tile), std::move(bufCopy));

		// Mark tile as resident
		imageStreamingState->currentMappedRegionOccupancy[gpuImageTileIndex.x][gpuImageTileIndex.y] = true;
	}

	return true;
}

bool DrawResourcesFiller::cancelGeoreferencedImageTileLoads(image_id imageID)
{
	auto it = streamedImageCopies.find(imageID);
	if (it != streamedImageCopies.end())
		it->second.clear(); // clear the vector of copies for this image

	return true;
}

void DrawResourcesFiller::drawGeoreferencedImage(image_id imageID, nbl::core::smart_refctd_ptr<GeoreferencedImageStreamingState>&& imageStreamingState, SIntendedSubmitInfo& intendedNextSubmit)
{
	// OutputDebugStringA(std::format("Image Cache Size = {}  ", imagesCache->size()).c_str());

	const bool resourcesEnsured = ensureGeoreferencedImageResources_AllocateIfNeeded(imageID, std::move(imageStreamingState), intendedNextSubmit);
	if (resourcesEnsured)
	{
		// Georefernced Image Data in the cache was already pre-transformed from local to main worldspace coordinates for tile calculation purposes
		// Because of this reason, the pre-transformed obb in the cache doesn't need to be transformed by custom projection again anymore.
		// we push the identity transform to prevent any more tranformation on the obb which is already in worldspace units.
		float64_t3x3 identity = float64_t3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
		pushCustomProjection(identity);

		beginMainObject(MainObjectType::STREAMED_IMAGE);

		uint32_t mainObjIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
		if (mainObjIdx != InvalidMainObjectIdx)
		{
			// Query imageType
			auto cachedImageRecord = imagesCache->peek(imageID);
			if (cachedImageRecord)
			{
				GeoreferencedImageInfo info = cachedImageRecord->georeferencedImageState->computeGeoreferencedImageAddressingAndPositioningInfo();
				info.textureID = getImageIndexFromID(imageID, intendedNextSubmit); // for this to be valid and safe, this function needs to be called immediately after `addStaticImage` function to make sure image is in memory
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
			}
			else
			{
				m_logger.log("drawGeoreferencedImage was not called immediately after enforceGeoreferencedImageAvailability!", nbl::system::ILogger::ELL_ERROR);
				assert(false);
			}
		}
		else
		{
			m_logger.log("drawGeoreferencedImage: acquireActiveMainObjectIndex returned invalid index", nbl::system::ILogger::ELL_ERROR);
			assert(false);
		}

		endMainObject();

		popCustomProjection();
	}
	else
	{
		m_logger.log("Failed to ensure resources (memory and descriptorIndex) for georeferencedImage", nbl::system::ILogger::ELL_ERROR);
	}
}

bool DrawResourcesFiller::finalizeGeoreferencedImageTileLoads(SIntendedSubmitInfo& intendedNextSubmit)
{
	bool success = true;

	if (streamedImageCopies.size() > 0ull)
	{
		auto* cmdBuffInfo = intendedNextSubmit.getCommandBufferForRecording();

		if (cmdBuffInfo)
		{
			std::vector<decltype(streamedImageCopies)::iterator> validCopies;
			validCopies.reserve(streamedImageCopies.size());

			// Step 1: collect valid image iters
			for (auto it = streamedImageCopies.begin(); it != streamedImageCopies.end(); ++it)
			{
				const auto& imageID = it->first;
				auto* imageRecord = imagesCache->peek(imageID);

				if (it->second.size() > 0u)
				{
					if (imageRecord && imageRecord->gpuImageView && imageRecord->georeferencedImageState)
						validCopies.push_back(it);
					else
						m_logger.log(std::format("Can't upload to imageId {} yet. (no gpu record yet).", imageID).c_str(), nbl::system::ILogger::ELL_INFO);
				}
			}
			
			// m_logger.log(std::format("{} Valid Copies, Frame Idx = {}.", validCopies.size(), currentFrameIndex).c_str(), nbl::system::ILogger::ELL_INFO);

			if (validCopies.size() > 0u)
			{
				IGPUCommandBuffer* commandBuffer = cmdBuffInfo->cmdbuf;
				std::vector<IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t> beforeCopyImageBarriers;
				beforeCopyImageBarriers.reserve(streamedImageCopies.size());

				// Pipeline Barriers before imageCopy
				for (auto it : validCopies)
				{
					auto& [imageID, imageCopies] = *it;
					// OutputDebugStringA(std::format("Copying {} copies for Id = {} \n", imageCopies.size(), imageID).c_str());

					auto* imageRecord = imagesCache->peek(imageID);
					if (imageRecord == nullptr)
					{
						m_logger.log(std::format("`pushStreamedImagesUploads` failed, no image record found for image id {}.", imageID).c_str(), nbl::system::ILogger::ELL_ERROR);
						continue;
					}

					const auto& gpuImg = imageRecord->gpuImageView->getCreationParameters().image;

					IImage::LAYOUT newLayout = IImage::LAYOUT::GENERAL;

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
							.oldLayout = imageRecord->currentLayout,
							.newLayout = newLayout,
						});
					imageRecord->currentLayout = newLayout;
				}
				success &= commandBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = beforeCopyImageBarriers });

				for (auto it : validCopies)
				{
					auto& [imageID, imageCopies] = *it;
					auto* imageRecord = imagesCache->peek(imageID);
					if (imageRecord == nullptr)
						continue;

					const auto& gpuImg = imageRecord->gpuImageView->getCreationParameters().image;

					for (auto& imageCopy : imageCopies)
					{
						auto srcBuffer = imageCopy.srcBufferFuture.get();
						if (srcBuffer)
						{
							const bool copySuccess = m_imageUploadUtils->updateImageViaStagingBuffer(
								intendedNextSubmit,
								srcBuffer->getPointer(), imageCopy.srcFormat,
								gpuImg.get(), IImage::LAYOUT::GENERAL,
								{ &imageCopy.region, 1u });
							success &= copySuccess;
							if (!copySuccess)
							{
								m_logger.log(std::format("updateImageViaStagingBuffer failed. region offset = ({}, {}), region size = ({}, {}), gpu image size = ({}, {})",
									imageCopy.region.imageOffset.x,imageCopy.region.imageOffset.y,
									imageCopy.region.imageExtent.width, imageCopy.region.imageExtent.height,
									gpuImg->getCreationParameters().extent.width, gpuImg->getCreationParameters().extent.height).c_str(), nbl::system::ILogger::ELL_ERROR);
							}
						}
						else
							m_logger.log(std::format("srcBuffer was invalid for image id {}.", imageID).c_str(), nbl::system::ILogger::ELL_ERROR);
					}
				}

				commandBuffer = intendedNextSubmit.getCommandBufferForRecording()->cmdbuf; // overflow-submit in utilities calls might've cause current recording command buffer to change

				std::vector<IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t> afterCopyImageBarriers;
				afterCopyImageBarriers.reserve(streamedImageCopies.size());

				// Pipeline Barriers after imageCopy
				for (auto it : validCopies)
				{
					auto& [imageID, imageCopies] = *it;
					auto* imageRecord = imagesCache->peek(imageID);
					if (imageRecord == nullptr)
					{
						m_logger.log(std::format("`pushStreamedImagesUploads` failed, no image record found for image id {}.", imageID).c_str(), nbl::system::ILogger::ELL_ERROR);
						continue;
					}

					const auto& gpuImg = imageRecord->gpuImageView->getCreationParameters().image;

					IImage::LAYOUT newLayout = IImage::LAYOUT::GENERAL;

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
							.oldLayout = imageRecord->currentLayout,
							.newLayout = newLayout,
						});
					imageRecord->currentLayout = newLayout;
				}
				success &= commandBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = afterCopyImageBarriers });
				// Remove the processed valid ones, keep invalids for later retries
				for (auto it : validCopies)
					streamedImageCopies.erase(it);
			}
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
		// In rare cases, we need to wait for the previous frame's submit to ensure all GPU usage of the any images has completed.
		nbl::video::ISemaphore::SWaitInfo previousSubmitWaitInfo = { .semaphore = intendedNextSubmit.scratchSemaphore.semaphore, .value = intendedNextSubmit.scratchSemaphore.value };

		// This means we're in a replay cache scope, use the replay cache to push to GPU instead of internal accumulation
		success &= pushBufferUploads(intendedNextSubmit, currentReplayCache->resourcesCollection);
		success &= pushMSDFImagesUploads(intendedNextSubmit, currentReplayCache->msdfImagesState);

		bool evictedAnotherImage = false;

		// Push Static Images Uploads from replay cache, all the work below is necessary to detect whether our image to replay is already in the cache in the exact form OR we need to create new image + bind memory and set array index
		for (auto& [toReplayImageID, toReplayRecord] : *currentReplayCache->imagesCache)
		{
			if (toReplayRecord.type != ImageType::STATIC) // non-static images (Georeferenced) won't be replayed like this
				continue;

			auto* cachedRecord = imagesCache->peek(toReplayImageID);
			bool alreadyResident = false;

			// compare with existing state, and check whether image id is already resident.
			if (cachedRecord != nullptr)
			{
				const bool allocationMatches =
					cachedRecord->allocationOffset == toReplayRecord.allocationOffset &&
					cachedRecord->allocationSize == toReplayRecord.allocationSize;

				const bool arrayIndexMatches = cachedRecord->arrayIndex == toReplayRecord.arrayIndex;

				alreadyResident = allocationMatches && arrayIndexMatches && cachedRecord->state != ImageState::INVALID;
			}

			// if already resident, ignore, no need to insert into cache anymore
			if (alreadyResident)
			{
				cachedRecord->lastUsedFrameIndex = currentFrameIndex;
			}
			else
			{
				// make sure to evict any cache entry that conflicts with the new entry (either in memory allocation or descriptor index)
				if (evictConflictingImagesInCache_SubmitIfNeeded(toReplayImageID, toReplayRecord, intendedNextSubmit))
					evictedAnotherImage = true;

				// creating and inserting new entry
				bool successCreateNewImage = false;
				{
					// Not already resident, we need to recreate the image and bind the image memory to correct location again, and update the descriptor set and push the uploads
					auto existingGPUImageViewParams = toReplayRecord.gpuImageView->getCreationParameters();
					IGPUImage::SCreationParams imageParams = {};
					imageParams = existingGPUImageViewParams.image->getCreationParameters();

					auto newGPUImage = m_device->createImage(std::move(imageParams));
					if (newGPUImage)
					{
						nbl::video::ILogicalDevice::SBindImageMemoryInfo bindImageMemoryInfo =
						{
							.image = newGPUImage.get(),
							.binding = {.memory = imagesMemoryArena.memory.get(), .offset = imagesMemoryArena.offset + toReplayRecord.allocationOffset }
						};

						const bool boundToMemorySuccessfully = m_device->bindImageMemory({ &bindImageMemoryInfo, 1u });
						if (boundToMemorySuccessfully)
						{
							newGPUImage->setObjectDebugName((std::to_string(toReplayImageID) + " Static Image 2D").c_str());
							IGPUImageView::SCreationParams viewParams = existingGPUImageViewParams;
							viewParams.image = newGPUImage;

							auto newGPUImageView = m_device->createImageView(std::move(viewParams));
							if (newGPUImageView)
							{
								successCreateNewImage = true;
								toReplayRecord.arrayIndexAllocatedUsingImageDescriptorIndexAllocator = false; // array index wasn't allocated useing desc set suballocator. it's being replayed
								toReplayRecord.gpuImageView = newGPUImageView;
								toReplayRecord.state = ImageState::CREATED_AND_MEMORY_BOUND;
								toReplayRecord.currentLayout = nbl::asset::IImage::LAYOUT::UNDEFINED;
								toReplayRecord.lastUsedFrameIndex = currentFrameIndex;
								newGPUImageView->setObjectDebugName((std::to_string(toReplayImageID) + " Static Image View 2D").c_str());
							}

						}
					}
				}

				if (successCreateNewImage)
				{
					// inserting the new entry into the cache (With new image and memory binding)
					imagesCache->base_t::insert(toReplayImageID, toReplayRecord);
				}
				else
				{
					m_logger.log("Couldn't create new gpu image in pushAllUploads: cache and replay mode.", nbl::system::ILogger::ELL_ERROR);
					_NBL_DEBUG_BREAK_IF(true);
					success = false;
				}
				
			}
		}

		success &= pushStaticImagesUploads(intendedNextSubmit, *imagesCache);

		if (evictedAnotherImage)
		{
			// We're about to update the descriptor set binding using the replay's array indices.
			// Normally, descriptor-set allocation and updates are synchronized to ensure the GPU
			// isn't still using the same descriptor indices we're about to overwrite.
			//
			// However, in this case we bypassed the descriptor-set allocator (imagesDescriptorIndexAllocator) and are writing directly into the set.
			// This means proper synchronization is not guaranteed.
			//
			// Since evicting another image can happen due to array index conflicts,
			// we must ensure that any prior GPU work using those descriptor indices has finished before we update them.
			// Therefore, wait for the previous frame (and any usage of these indices) to complete before proceeding to bind/write our images to their descriptor
			m_device->blockForSemaphores({ &previousSubmitWaitInfo, 1u });
		}

		success &= updateDescriptorSetImageBindings(*imagesCache);
	}
	else
	{
		flushDrawObjects();
		success &= pushBufferUploads(intendedNextSubmit, resourcesCollection);
		success &= pushMSDFImagesUploads(intendedNextSubmit, msdfImagesState);
		success &= pushStaticImagesUploads(intendedNextSubmit, *imagesCache);
		success &= updateDescriptorSetImageBindings(*imagesCache);
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
	ret->imagesCache = std::unique_ptr<ImagesCache>(new ImagesCache(ImagesBindingArraySize));

	// m_logger.log(std::format("== createReplayCache, currentFrameIndex = {} ==", currentFrameIndex).c_str(), nbl::system::ILogger::ELL_INFO);
	// imagesCache->logState(m_logger);
	
	for (auto& [imageID, record] : *imagesCache)
	{
		// Only return images in the cache used within the last frame
		if (record.lastUsedFrameIndex == currentFrameIndex)
			ret->imagesCache->base_t::insert(imageID, record);
	}

	return ret;
}

void DrawResourcesFiller::setReplayCache(ReplayCache* cache)
{
	currentReplayCache = cache;
	// currentReplayCache->imagesCache->logState(m_logger);
}

void DrawResourcesFiller::unsetReplayCache()
{
	currentReplayCache = nullptr;
}

uint64_t DrawResourcesFiller::getImagesMemoryConsumption() const
{
	uint64_t ret = 0ull;
	for (auto& [imageID, record] : *imagesCache)
		ret += record.allocationSize;
	return ret;
}

DrawResourcesFiller::UsageData DrawResourcesFiller::getCurrentUsageData()
{
	UsageData ret = {};
	const auto& resources = getResourcesCollection();
	ret.lineStyleCount = resources.lineStyles.getCount();
	ret.dtmSettingsCount = resources.dtmSettings.getCount();
	ret.customProjectionsCount = resources.customProjections.getCount();
	ret.mainObjectCount = resources.mainObjects.getCount();
	ret.drawObjectCount = resources.drawObjects.getCount();
	ret.geometryBufferSize = resources.geometryInfo.getStorageSize();
	ret.bufferMemoryConsumption = resources.calculateTotalConsumption();
	ret.imageMemoryConsumption = getImagesMemoryConsumption();
	return ret;
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
				if (!m_bufferUploadUtils->updateBufferRangeViaStagingBuffer(intendedNextSubmit, copyRange, drawBuffer.vector.data()))
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

						stagedMSDF.uploadedToGPU = m_bufferUploadUtils->updateImageViaStagingBuffer(
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

bool DrawResourcesFiller::updateDescriptorSetImageBindings(ImagesCache& imagesCache)
{
	bool success = true;
	
	auto* descriptorSet = imagesDescriptorIndexAllocator->getDescriptorSet();

	// DescriptorSet Updates
	std::vector<video::IGPUDescriptorSet::SDescriptorInfo> descriptorInfos;
	std::vector<IGPUDescriptorSet::SWriteDescriptorSet> descriptorWrites;
	descriptorInfos.resize(imagesCache.size());
	descriptorWrites.resize(imagesCache.size());

	// Potential GPU waits before writing to descriptor bindings that were previously deallocated manually (bypassing the imagesDescriptorIndexAllocator). 
	// The allocator normally guarantees safe reuse of array indices by synchronizing allocations and deallocations internally.
	// but since these bindings were queued for deferred deallocation, we must ensure their previous GPU usage has completed before writing new data into those slots.
	std::vector<nbl::video::ISemaphore::SWaitInfo> waitInfos;
	waitInfos.reserve(deferredDescriptorIndexDeallocations.size());

	uint32_t descriptorWriteCount = 0u;
	for (auto& [id, record] : imagesCache)
	{
		if (record.state >= ImageState::BOUND_TO_DESCRIPTOR_SET || !record.gpuImageView)
			continue;
		
		// Check if this writing to this array index has a deferred deallocation pending
		if (auto it = deferredDescriptorIndexDeallocations.find(record.arrayIndex); it != deferredDescriptorIndexDeallocations.end())
		{
			// TODO: Assert we're not waiting for a value which hasn't been submitted yet.
			waitInfos.push_back(it->second);
			// erase -> it's a one-time wait:
			deferredDescriptorIndexDeallocations.erase(it);
		}

		// Bind gpu image view to descriptor set
		video::IGPUDescriptorSet::SDescriptorInfo descriptorInfo = {};
		descriptorInfo.info.image.imageLayout = (record.type == ImageType::STATIC) ? IImage::LAYOUT::READ_ONLY_OPTIMAL : IImage::LAYOUT::GENERAL; // WARN: don't use `record.currentLayout`, it's the layout "At the time" the image is going to be accessed
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

	if (!waitInfos.empty())
		m_device->blockForSemaphores(waitInfos, /*waitAll=*/true);

	if (descriptorWriteCount > 0u)
		success &= m_device->updateDescriptorSets(descriptorWriteCount, descriptorWrites.data(), 0u, nullptr);

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
		if (record.staticCPUImage && record.type == ImageType::STATIC && record.state < ImageState::GPU_RESIDENT_WITH_VALID_STATIC_DATA)
			nonResidentImageRecords.push_back(&record);
	}

	if (nonResidentImageRecords.size() > 0ull)
	{
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
					.oldLayout = imageRecord.currentLayout,
					.newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				};
				imageRecord.currentLayout = beforeCopyImageBarriers[i].newLayout;
			}
			success &= commandBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = beforeCopyImageBarriers });

			for (uint32_t i = 0u; i < nonResidentImageRecords.size(); ++i)
			{
				auto& imageRecord = *nonResidentImageRecords[i];
				auto& gpuImg = imageRecord.gpuImageView->getCreationParameters().image;
				success &= m_imageUploadUtils->updateImageViaStagingBuffer(
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
					.oldLayout = imageRecord.currentLayout,
					.newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
				};
				imageRecord.currentLayout = afterCopyImageBarriers[i].newLayout;
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

bool DrawResourcesFiller::evictConflictingImagesInCache_SubmitIfNeeded(image_id toInsertImageID, const CachedImageRecord& toInsertRecord, nbl::video::SIntendedSubmitInfo& intendedNextSubmit)
{
	bool evictedSomething = false;
	for (auto& [cachedImageID, cachedRecord] : *imagesCache)
	{
		bool cachedImageConflictsWithImageToReplay = false;

		// Case 1: Same imageID, but params differ (offset/size/arrayIndex mismatch) conflict
		if (cachedImageID == toInsertImageID)
		{
			const bool allocationMatches =
				cachedRecord.allocationOffset == toInsertRecord.allocationOffset &&
				cachedRecord.allocationSize == toInsertRecord.allocationSize;
			const bool arrayIndexMatches = cachedRecord.arrayIndex == toInsertRecord.arrayIndex;
			const bool exactSameImage = allocationMatches && arrayIndexMatches;
			if (!exactSameImage)
				cachedImageConflictsWithImageToReplay = true;
		}
		else
		{
			// Different Image ID:
			// Conflicted if: 1. same array index or 2. conflict in allocation/mem
			const bool sameArrayIndex = cachedRecord.arrayIndex == toInsertRecord.arrayIndex;
			const bool conflictingMemory =
				(cachedRecord.allocationOffset < toInsertRecord.allocationOffset + toInsertRecord.allocationSize) &&
				(toInsertRecord.allocationOffset < cachedRecord.allocationOffset + cachedRecord.allocationSize);

			if (sameArrayIndex || conflictingMemory)
				cachedImageConflictsWithImageToReplay = true;
		}

		if (cachedImageConflictsWithImageToReplay)
		{
			evictImage_SubmitIfNeeded(cachedImageID, cachedRecord, intendedNextSubmit);
			imagesCache->erase(cachedImageID);
			evictedSomething = true;
		}
	}
	return evictedSomething;
}

bool DrawResourcesFiller::ensureGeoreferencedImageResources_AllocateIfNeeded(image_id imageID, nbl::core::smart_refctd_ptr<GeoreferencedImageStreamingState>&& imageStreamingState, SIntendedSubmitInfo& intendedNextSubmit)
{
	auto* physDev = m_device->getPhysicalDevice();

	// Check if image already exists and requires resize. We do this before insertion and updating `lastUsedFrameIndex` to get correct overflow-submit behaviour
	// otherwise we'd always overflow submit, even if not needed and image was not queued/intended to use in the next submit.
	CachedImageRecord* cachedImageRecord = imagesCache->get(imageID);
	
	// if cachedImageRecord->index was not InvalidTextureIndex then it means we had a cache hit and updated the value of our sema
	// But we need to check if the cached image needs resizing/recreation.
	if (cachedImageRecord && cachedImageRecord->arrayIndex != InvalidTextureIndex)
	{
		// found in cache, but does it require resize? recreation?
		if (cachedImageRecord->gpuImageView)
		{
			auto imgViewParams = cachedImageRecord->gpuImageView->getCreationParameters();
			if (imgViewParams.image)
			{
				const auto cachedParams = static_cast<asset::IImage::SCreationParams>(imgViewParams.image->getCreationParameters());
				// image type and creation params (most importantly extent and format) should match, otherwise we evict, recreate and re-pus
				const auto toCreateParams = static_cast<asset::IImage::SCreationParams>(imageStreamingState->gpuImageCreationParams);
				const bool needsRecreation = cachedParams != toCreateParams;
				if (needsRecreation)
				{
					// call the eviction callback so the currently cached imageID gets eventually deallocated from memory arena.
					// note: it doesn't remove the entry from lru cache.
					evictImage_SubmitIfNeeded(imageID, *cachedImageRecord, intendedNextSubmit);
					
					// instead of erasing and inserting the imageID into the cache, we just reset it, so the next block of code goes into array index allocation + creating our new image
					CachedImageRecord newRecord = CachedImageRecord(currentFrameIndex); //reset everything except image streaming state
					*cachedImageRecord = std::move(newRecord);
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


	// Try inserting or updating the image usage in the cache.
	// If the image is already present, updates its semaphore value.
	auto evictCallback = [&](image_id imageID, const CachedImageRecord& evicted) { evictImage_SubmitIfNeeded(imageID, evicted, intendedNextSubmit); };
	cachedImageRecord = imagesCache->insert(imageID, currentFrameIndex, evictCallback);
	cachedImageRecord->lastUsedFrameIndex = currentFrameIndex; // in case there was an eviction + auto-submit, we need to update AGAIN
	
	// Setting the image streaming state returned in `ensureGeoreferencedImageEntry` which was either creating anew or gotten from this very own cache
	cachedImageRecord->georeferencedImageState = std::move(imageStreamingState);
	cachedImageRecord->georeferencedImageState->outOfDate = false;

	if (cachedImageRecord == nullptr)
	{
		m_logger.log("Couldn't insert image in cache; make sure you called `ensureGeoreferencedImageEntry` before anything else.", nbl::system::ILogger::ELL_ERROR);
		return false;
	}

	// in which case we don't queue anything for upload, and return the idx
	if (cachedImageRecord->arrayIndex == InvalidTextureIndex)
	{
		// This is a new image (cache miss). Allocate a descriptor index for it.
		cachedImageRecord->arrayIndex = video::SubAllocatedDescriptorSet::AddressAllocator::invalid_address;
		// Blocking allocation attempt; if the descriptor pool is exhausted, this may stall.
		imagesDescriptorIndexAllocator->multi_allocate(std::chrono::time_point<std::chrono::steady_clock>::max(), imagesArrayBinding, 1u, &cachedImageRecord->arrayIndex); // if the prev submit causes DEVICE_LOST then we'll get a deadlock here since we're using max timepoint
		cachedImageRecord->arrayIndexAllocatedUsingImageDescriptorIndexAllocator = true;

		if (cachedImageRecord->arrayIndex != video::SubAllocatedDescriptorSet::AddressAllocator::invalid_address)
		{
			const auto& imageCreationParams = cachedImageRecord->georeferencedImageState->gpuImageCreationParams;

			std::string debugName = cachedImageRecord->georeferencedImageState->storagePath.string();

			// Attempt to create a GPU image and image view for this texture.
			ImageAllocateResults allocResults = tryCreateAndAllocateImage_SubmitIfNeeded(imageCreationParams, asset::E_FORMAT::EF_COUNT, intendedNextSubmit, debugName);
			if (allocResults.isValid())
			{
				cachedImageRecord->type = ImageType::GEOREFERENCED_STREAMED;
				cachedImageRecord->state = ImageState::CREATED_AND_MEMORY_BOUND;
				cachedImageRecord->currentLayout  = nbl::asset::IImage::LAYOUT::UNDEFINED;
				cachedImageRecord->lastUsedFrameIndex = currentFrameIndex; // there was an eviction + auto-submit, we need to update AGAIN
				cachedImageRecord->allocationOffset = allocResults.allocationOffset;
				cachedImageRecord->allocationSize = allocResults.allocationSize;
				cachedImageRecord->gpuImageView = allocResults.gpuImageView;
				cachedImageRecord->staticCPUImage = nullptr;
				evictConflictingImagesInCache_SubmitIfNeeded(imageID, *cachedImageRecord, intendedNextSubmit);
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
					imagesDescriptorIndexAllocator->multi_deallocate(imagesArrayBinding, 1u, &cachedImageRecord->arrayIndex, {});
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

#if 0
	m_logger.log(("Evicting Image: \n" + evicted.toString(imageID)).c_str(), nbl::system::ILogger::ELL_INFO);
#endif
	
	const bool imageUsedForNextIntendedSubmit = (evicted.lastUsedFrameIndex == currentFrameIndex);

	if (evicted.arrayIndexAllocatedUsingImageDescriptorIndexAllocator)
	{
		// Image being evicted was allocated using image descriptor set allocator
		// Later used to release the image's memory range.
		core::smart_refctd_ptr<ImageCleanup> cleanupObject = core::make_smart_refctd_ptr<ImageCleanup>();
		cleanupObject->imagesMemorySuballocator = imagesMemorySubAllocator;
		cleanupObject->addr = evicted.allocationOffset;
		cleanupObject->size = evicted.allocationSize;

		if (evicted.type == ImageType::GEOREFERENCED_STREAMED)
		{
			// Important to mark this as out of date. 
			// because any other place still holding on to the state (which is possible) need to know the image associated with the state has been evicted and the state is no longer valid and needs to "ensure"d again.
			evicted.georeferencedImageState->outOfDate = true;
			// cancelGeoreferencedImageTileLoads(imageID); // clear any of the pending loads/futures requested for the image
		}

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
			imagesDescriptorIndexAllocator->multi_deallocate(imagesArrayBinding, 1u, &evicted.arrayIndex, deallocationWaitInfo, &cleanupObject.get());
		}
		else
		{
			// The image is not used in the current frame, so we can deallocate without submitting any draws.
			// Still wait on the semaphore to ensure past GPU usage is complete.
			// TODO: We don't know which semaphore value the frame with `evicted.lastUsedFrameIndex` index was submitted with, so we wait for the worst case value conservatively, which is the immediate prev submit.
			ISemaphore::SWaitInfo deallocationWaitInfo = { .semaphore = intendedNextSubmit.scratchSemaphore.semaphore, .value = intendedNextSubmit.scratchSemaphore.value };
			imagesDescriptorIndexAllocator->multi_deallocate(imagesArrayBinding, 1u, &evicted.arrayIndex, deallocationWaitInfo, &cleanupObject.get());
		}
	}
	else
	{
		// Less often case: index wasn't allocated using imageDescriptorSetAllocator, like replayed images which skip the allocator to write to the set directly.
		// we won't cleanup + multi_dealloc in this case, instead we queue the deallocations and wait for them before any next image writes into the same index.
		if (!imageUsedForNextIntendedSubmit)
			deferredDescriptorIndexDeallocations[evicted.arrayIndex] = ISemaphore::SWaitInfo{ .semaphore = intendedNextSubmit.scratchSemaphore.semaphore, .value = intendedNextSubmit.scratchSemaphore.value };
		else
		{
			m_logger.log(std::format("Image which is being evicted and had skipped descriptor set allocator requires overflow submit; This shouldn't happen. Image Info = {}", evicted.toString(imageID)).c_str(), nbl::system::ILogger::ELL_ERROR);
			imagesCache->logState(m_logger);
		}
		
	}
}

DrawResourcesFiller::ImageAllocateResults DrawResourcesFiller::tryCreateAndAllocateImage_SubmitIfNeeded(
	const nbl::asset::IImage::SCreationParams& imageParams,
	const asset::E_FORMAT imageViewFormatOverride,
	nbl::video::SIntendedSubmitInfo& intendedNextSubmit,
	std::string imageDebugName)
{
	ImageAllocateResults ret = {};

	auto* physDev = m_device->getPhysicalDevice();

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
		auto gpuImage = m_device->createImage(std::move(params));

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
				// OutputDebugStringA(std::format("ALlocating {} !!!!\n", gpuImageMemoryRequirements.size).c_str());
				// m_logger.log(std::format(" [BEFORE] Allocator Free Size={} \n",imagesMemorySubAllocator->getFreeSize()).c_str(), nbl::system::ILogger::ELL_INFO);
				ret.allocationOffset = imagesMemorySubAllocator->allocate(gpuImageMemoryRequirements.size, 1u << gpuImageMemoryRequirements.alignmentLog2);
				// m_logger.log(std::format(" [AFTER] Alloc Size = {}, Alloc Offset = {}, Alignment = {} \n",gpuImageMemoryRequirements.size, ret.allocationOffset, 1u << gpuImageMemoryRequirements.alignmentLog2).c_str(), nbl::system::ILogger::ELL_INFO);
				// m_logger.log(std::format(" [AFTER] Allocator Free Size={} \n",imagesMemorySubAllocator->getFreeSize()).c_str(), nbl::system::ILogger::ELL_INFO);
				const bool allocationFromImagesMemoryArenaSuccessfull = ret.allocationOffset != ImagesMemorySubAllocator::InvalidAddress;
				if (allocationFromImagesMemoryArenaSuccessfull)
				{
					ret.allocationSize = gpuImageMemoryRequirements.size;
					nbl::video::ILogicalDevice::SBindImageMemoryInfo bindImageMemoryInfo =
					{
						.image = gpuImage.get(),
						.binding = { .memory = imagesMemoryArena.memory.get(), .offset = imagesMemoryArena.offset + ret.allocationOffset }
					};
					const bool boundToMemorySuccessfully = m_device->bindImageMemory({ &bindImageMemoryInfo, 1u });
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

						ret.gpuImageView = m_device->createImageView(std::move(viewParams));
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
					m_logger.log(std::format("Retrying Allocation after failure with Allocation Size={}, Allocator Free Size={} \n", gpuImageMemoryRequirements.size, imagesMemorySubAllocator->getFreeSize()).c_str(), nbl::system::ILogger::ELL_INFO);
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

		while (imagesDescriptorIndexAllocator->cull_frees()) {}; // to make sure deallocation requests in eviction callback are blocked for.
		alreadyBlockedForDeferredFrees = true;

		// we don't hold any references to the GPUImageView or GPUImage so descriptor binding will be the last reference
		// hopefully by here the suballocated descriptor set freed some VRAM by dropping the image last ref and it's dedicated allocation.
	}

	return ret;
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
	// m_logger.log(std::format("Finished Frame Idx = {}", currentFrameIndex).c_str(), nbl::system::ILogger::ELL_INFO);
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