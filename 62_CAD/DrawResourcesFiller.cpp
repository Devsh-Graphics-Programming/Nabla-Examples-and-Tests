#include "DrawResourcesFiller.h"

DrawResourcesFiller::DrawResourcesFiller()
{}

DrawResourcesFiller::DrawResourcesFiller(smart_refctd_ptr<IUtilities>&& utils, IQueue* copyQueue) :
	m_utilities(utils),
	m_copyQueue(copyQueue)
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

void DrawResourcesFiller::allocateResourcesBuffer(ILogicalDevice* logicalDevice, size_t size)
{
	// TODO: Make this function failable and report insufficient memory if less that getMinimumRequiredResourcesBufferSize, TODO: Have retry mechanism to allocate less mem
	// TODO: Allocate buffer memory and image memory with 1 allocation, so that failure and retries are more straightforward.
	size = core::alignUp(size, ResourcesMaxNaturalAlignment);
	size = core::max(size, getMinimumRequiredResourcesBufferSize());
	// size = 368u; STRESS TEST
	IGPUBuffer::SCreationParams geometryCreationParams = {};
	geometryCreationParams.size = size;
	geometryCreationParams.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INDEX_BUFFER_BIT;
	resourcesGPUBuffer = logicalDevice->createBuffer(std::move(geometryCreationParams));
	resourcesGPUBuffer->setObjectDebugName("drawResourcesBuffer");

	IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = resourcesGPUBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto mem = logicalDevice->allocate(memReq, resourcesGPUBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

	// Allocate for Images  
	{
		const auto& memoryProperties = logicalDevice->getPhysicalDevice()->getMemoryProperties();
		uint32_t memoryTypeIdx = ~0u;
		for (uint32_t i = 0u; i < memoryProperties.memoryTypeCount; ++i)
		{
			if (memoryProperties.memoryTypes[i].propertyFlags.hasFlags(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT))
			{
				memoryTypeIdx = i;
				break;
			}
		}

		if (memoryTypeIdx == ~0u)
		{
			// TODO: Log, no device local memory found?! weird
			assert(false);
		}

		IDeviceMemoryAllocator::SAllocateInfo allocationInfo =
		{
			// TODO: Get from user side.
			.size = 65 * 1024 * 1024, // 70 MB
			.flags = IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE,
			.memoryTypeIndex = memoryTypeIdx,
			.dedication = nullptr,
		};
		imagesMemoryArena = logicalDevice->allocate(allocationInfo);

		if (imagesMemoryArena.isValid())
		{
			imagesMemorySubAllocator = core::make_smart_refctd_ptr<ImagesMemorySubAllocator>(static_cast<uint64_t>(allocationInfo.size));
		}
		else
		{
			// LOG: Allocation failure to allocate memory arena for images 
			assert(false);
		}
	}

}

void DrawResourcesFiller::allocateMSDFTextures(ILogicalDevice* logicalDevice, uint32_t maxMSDFs, uint32_t2 msdfsExtent)
{
	// TODO: Make this function failable and report insufficient memory
	asset::E_FORMAT msdfFormat = MSDFTextureFormat;
	asset::VkExtent3D MSDFsExtent = { msdfsExtent.x, msdfsExtent.y, 1u }; 
	assert(maxMSDFs <= logicalDevice->getPhysicalDevice()->getLimits().maxImageArrayLayers);

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
		logicalDevice->allocate(imageMemReqs, image.get());

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

	msdfLRUCache = std::unique_ptr<MSDFsLRUCache>(new MSDFsLRUCache(maxMSDFs));
	msdfTextureArrayIndexAllocator = core::make_smart_refctd_ptr<IndexAllocator>(core::smart_refctd_ptr<ILogicalDevice>(logicalDevice), maxMSDFs);
	msdfImagesState.resize(maxMSDFs);
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
	
	if (!activeProjections.empty())
	{
		// if there is already an active custom projection, it should be considered into the transformation of the fixed geometry polyline
		float64_t3x3 newTransformation = nbl::hlsl::mul(activeProjections.back(), transformation);
		pushCustomProjection(newTransformation);
	}
	else
	{
		// will be multiplied by the default projection matrix from the left (in shader), no need to consider it here
		pushCustomProjection(transformation);
	}

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

	DrawCallData drawCallData = {}; 
	drawCallData.isDTMRendering = true;

	uint32_t mainObjectIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
	drawCallData.dtm.triangleMeshMainObjectIndex = mainObjectIdx;

	ICPUBuffer::SCreationParams geometryBuffParams;
	
	// concatenate the index and vertex buffer into the geometry buffer
	const size_t indexBuffByteSize = mesh.getIndexBuffByteSize();
	const size_t vtxBuffByteSize = mesh.getVertexBuffByteSize();
	const size_t dataToAddByteSize = vtxBuffByteSize + indexBuffByteSize;

	const size_t remainingResourcesSize = calculateRemainingResourcesSize();

	// TODO: assert of geometry buffer size, do i need to check if size of objects to be added <= remainingResourcesSize?
	// TODO: auto submit instead of assert
	assert(dataToAddByteSize <= remainingResourcesSize);

	{
		// NOTE[ERFAN]: these push contants will be removed, everything will be accessed by dtmSettings, including where the vertex buffer data resides

		// Copy VertexBuffer
		size_t geometryBufferOffset = resourcesCollection.geometryInfo.increaseSizeAndGetOffset(dataToAddByteSize, alignof(CTriangleMesh::vertex_t));
		void* dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
		// the actual bda address will be determined only after all copies are finalized, later we will do += `baseBDAAddress + geometryInfo.bufferOffset`
		drawCallData.dtm.triangleMeshVerticesBaseAddress = geometryBufferOffset;
		memcpy(dst, mesh.getVertices().data(), vtxBuffByteSize);
		geometryBufferOffset += vtxBuffByteSize; 

		// Copy IndexBuffer
		dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
		drawCallData.dtm.indexBufferOffset = geometryBufferOffset;
		memcpy(dst, mesh.getIndices().data(), indexBuffByteSize);
		geometryBufferOffset += indexBuffByteSize;
	}

	drawCallData.dtm.indexCount = mesh.getIndexCount();
	drawCalls.push_back(drawCallData);
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
	beginMainObject(MainObjectType::HATCH);
	
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

void DrawResourcesFiller::drawHatch(const Hatch& hatch, const float32_t4& color, SIntendedSubmitInfo& intendedNextSubmit)
{
	drawHatch(hatch, color, HatchFillPattern::SOLID_FILL, intendedNextSubmit);
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
	assert(mainObjIdx != InvalidMainObjectIdx);

	if (textureIdx != InvalidTextureIndex)
	{
		GlyphInfo glyphInfo = GlyphInfo(topLeft, dirU, aspectRatio, textureIdx, minUV);
		if (!addFontGlyph_Internal(glyphInfo, mainObjIdx))
		{
			// single font glyph couldn't fit into memory to push to gpu, so we submit rendering current objects and reset geometry buffer and draw objects
			submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjIdx);
			bool success = addFontGlyph_Internal(glyphInfo, mainObjIdx);
			assert(success); // this should always be true, otherwise it's either bug in code or not enough memory allocated to hold a single GlyphInfo
		}
	}
	else
	{
		// TODO: Log, probably getGlyphMSDF(face,glyphIdx) returned nullptr ICPUImage ptr
		_NBL_DEBUG_BREAK_IF(true);
	}
}

bool DrawResourcesFiller::ensureStaticImageAvailability(const StaticImageInfo& staticImage, SIntendedSubmitInfo& intendedNextSubmit)
{
	const auto& imageID = staticImage.imageID;
	const auto& cpuImage = staticImage.cpuImage;
	
	// Try inserting or updating the image usage in the cache.
	// If the image is already present, updates its semaphore value.
	auto evictCallback = [&](image_id imageID, const CachedImageRecord& evicted) { evictImage_SubmitIfNeeded(imageID, evicted, intendedNextSubmit); };
	CachedImageRecord* cachedImageRecord = imagesCache->insert(imageID, intendedNextSubmit.getFutureScratchSemaphore().value, evictCallback);
	cachedImageRecord->lastUsedFrameIndex = currentFrameIndex; // in case there was an eviction + auto-submit, we need to update AGAIN

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
			imageParams = cpuImage->getCreationParameters();
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
			ImageAllocateResults allocResults = tryCreateAndAllocateImage_SubmitIfNeeded(imageParams, intendedNextSubmit, std::to_string(imageID));

			if (allocResults.isValid())
			{
				cachedImageRecord->type = ImageType::STATIC;
				cachedImageRecord->state = ImageState::CREATED_AND_MEMORY_BOUND;
				cachedImageRecord->lastUsedFrameIndex = currentFrameIndex; // there was an eviction + auto-submit, we need to update AGAIN
				cachedImageRecord->allocationOffset = allocResults.allocationOffset;
				cachedImageRecord->allocationSize = allocResults.allocationSize;
				cachedImageRecord->gpuImageView = allocResults.gpuImageView;
				cachedImageRecord->staticCPUImage = cpuImage;
			}
			else
			{
				// All attempts to try create the GPU image and its corresponding view have failed.
				// Most likely cause: insufficient GPU memory or unsupported image parameters.
				// TODO: Log a warning or error here � `addStaticImage2D` failed, likely due to low VRAM.
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
			// TODO: log here, index allocation failed.
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

bool DrawResourcesFiller::ensureGeoreferencedImageAvailability_AllocateIfNeeded(image_id imageID, const GeoreferencedImageParams& params, SIntendedSubmitInfo& intendedNextSubmit)
{
	auto* device = m_utilities->getLogicalDevice();
	auto* physDev = m_utilities->getLogicalDevice()->getPhysicalDevice();

	// Try inserting or updating the image usage in the cache.
	// If the image is already present, updates its semaphore value.
	auto evictCallback = [&](image_id imageID, const CachedImageRecord& evicted) { evictImage_SubmitIfNeeded(imageID, evicted, intendedNextSubmit); };
	CachedImageRecord* cachedImageRecord = imagesCache->insert(imageID, intendedNextSubmit.getFutureScratchSemaphore().value, evictCallback);

	// TODO: Function call that gets you image creaation params based on georeferencedImageParams (extents and mips and whatever), it will also get you the GEOREFERENED TYPE
	IGPUImage::SCreationParams imageCreationParams = {};
	ImageType georeferenceImageType;
	determineGeoreferencedImageCreationParams(imageCreationParams, georeferenceImageType, params);

	assert(georeferenceImageType != ImageType::STATIC);

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
				const bool needsRecreation = cachedImageType != georeferenceImageType || cachedParams != currentParams;
				if (needsRecreation)
				{
					// call the eviction callbacl so the currently cached imageID gets eventually deallocated from memory arena.
					evictCallback(imageID, *cachedImageRecord);
					
					// instead of erasing and inserting the imageID into the cache, we just reset it, so the next block of code goes into array index allocation + creating our new image
					*cachedImageRecord = CachedImageRecord(currentFrameIndex);
					// imagesCache->erase(imageID);
					// cachedImageRecord = imagesCache->insert(imageID, intendedNextSubmit.getFutureScratchSemaphore().value, evictCallback);
				}
			}
			else
			{
				// TODO[LOG]
			}
		}
		else
		{
			// TODO[LOG]
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
			ImageAllocateResults allocResults = tryCreateAndAllocateImage_SubmitIfNeeded(imageCreationParams, intendedNextSubmit, std::to_string(imageID));

			if (allocResults.isValid())
			{
				cachedImageRecord->type = georeferenceImageType;
				cachedImageRecord->state = ImageState::CREATED_AND_MEMORY_BOUND;
				cachedImageRecord->lastUsedFrameIndex = currentFrameIndex; // there was an eviction + auto-submit, we need to update AGAIN
				cachedImageRecord->allocationOffset = allocResults.allocationOffset;
				cachedImageRecord->allocationSize = allocResults.allocationSize;
				cachedImageRecord->gpuImageView = allocResults.gpuImageView;
				cachedImageRecord->staticCPUImage = nullptr;
			}
			else
			{
				// All attempts to try create the GPU image and its corresponding view have failed.
				// Most likely cause: insufficient GPU memory or unsupported image parameters.
				// TODO: Log a warning or error here � `addStaticImage2D` failed, likely due to low VRAM.
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
			// TODO: log here, index allocation failed.
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
	float64_t height,
	float64_t width,
	float gridCellWidth,
	const DTMSettingsInfo& dtmSettingsInfo,
	SIntendedSubmitInfo& intendedNextSubmit)
{
	GridDTMInfo gridDTMInfo;
	gridDTMInfo.topLeft = topLeft;
	gridDTMInfo.height = height;
	gridDTMInfo.width = width;
	gridDTMInfo.gridCellWidth = gridCellWidth;

	if (dtmSettingsInfo.mode & E_DTM_MODE::OUTLINE)
	{
		const bool isOutlineStippled = dtmSettingsInfo.outlineStyleInfo.stipplePatternSize > 0;
		gridDTMInfo.outlineStipplePatternLengthReciprocal = isOutlineStippled ? dtmSettingsInfo.outlineStyleInfo.reciprocalStipplePatternLen : 0.0f;
	}

	setActiveDTMSettings(dtmSettingsInfo);
	beginMainObject(MainObjectType::GRID_DTM);

	uint32_t mainObjectIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
	assert(mainObjectIdx != InvalidMainObjectIdx);

	addGridDTM_Internal(gridDTMInfo, mainObjectIdx);

	endMainObject();
}

void DrawResourcesFiller::addImageObject(image_id imageID, const OrientedBoundingBox2D& obb, SIntendedSubmitInfo& intendedNextSubmit)
{
	beginMainObject(MainObjectType::STATIC_IMAGE);

	uint32_t mainObjIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);

	ImageObjectInfo info = {};
	info.topLeft = obb.topLeft;
	info.dirU = obb.dirU;
	info.aspectRatio = obb.aspectRatio;
	info.textureID = getImageIndexFromID(imageID, intendedNextSubmit); // for this to be valid and safe, this function needs to be called immediately after `addStaticImage` function to make sure image is in memory
	if (!addImageObject_Internal(info, mainObjIdx))
	{
		// single image object couldn't fit into memory to push to gpu, so we submit rendering current objects and reset geometry buffer and draw objects
		submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjIdx);
		bool success = addImageObject_Internal(info, mainObjIdx);
		assert(success); // this should always be true, otherwise it's either bug in code or not enough memory allocated to hold a single image object 
	}

	endMainObject();
}

void DrawResourcesFiller::addGeoreferencedImage(image_id imageID, const GeoreferencedImageParams& params, SIntendedSubmitInfo& intendedNextSubmit)
{
	beginMainObject(MainObjectType::STREAMED_IMAGE);

	uint32_t mainObjIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);

	GeoreferencedImageInfo info = {};
	info.topLeft = params.worldspaceOBB.topLeft;
	info.dirU = params.worldspaceOBB.dirU;
	info.aspectRatio = params.worldspaceOBB.aspectRatio;
	info.textureID = getImageIndexFromID(imageID, intendedNextSubmit); // for this to be valid and safe, this function needs to be called immediately after `addStaticImage` function to make sure image is in memory
	if (!addGeoreferencedImageInfo_Internal(info, mainObjIdx))
	{
		// single image object couldn't fit into memory to push to gpu, so we submit rendering current objects and reset geometry buffer and draw objects
		submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjIdx);
		bool success = addGeoreferencedImageInfo_Internal(info, mainObjIdx);
		assert(success); // this should always be true, otherwise it's either bug in code or not enough memory allocated to hold a single GeoreferencedImageInfo 
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
				// TODO: Log
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

	assert(resourcesCollection.calculateTotalConsumption() <= resourcesGPUBuffer->getSize());

	auto copyCPUFilledDrawBuffer = [&](auto& drawBuffer) -> bool
		{
			// drawBuffer must be of type CPUGeneratedResource<T>
			SBufferRange<IGPUBuffer> copyRange = { copiedResourcesSize, drawBuffer.getStorageSize(), resourcesGPUBuffer};

			if (copyRange.offset + copyRange.size > resourcesGPUBuffer->getSize())
			{
				// TODO: LOG ERROR, this shouldn't happen with correct auto-submission mechanism
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
				// TODO: LOG ERROR, this shouldn't happen with correct auto-submission mechanism
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
		// TODO: Log no valid command buffer to record into
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
		if (record.staticCPUImage && record.type == ImageType::STATIC && record.state < ImageState::GPU_RESIDENT_WITH_VALID_STATIC_DATA)
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
					// TODO: LOG
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
		// TODO: Log
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
						.oldLayout = IImage::LAYOUT::UNDEFINED,
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

			// Pipeline Barriers before imageCopy
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
		// TODO: Log
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
		_NBL_DEBUG_BREAK_IF(!dtmSettingsInfo.heightShadingInfo.fillShaderDTMSettingsHeightColorMap(dtmSettings));
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
	//drawObj.geometryAddress = 0;
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
		_NBL_DEBUG_BREAK_IF(true); // shouldn't happen under normal circumstances, TODO: LOG warning
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

DrawResourcesFiller::ImageAllocateResults DrawResourcesFiller::tryCreateAndAllocateImage_SubmitIfNeeded(const nbl::asset::IImage::SCreationParams& imageParams, nbl::video::SIntendedSubmitInfo& intendedNextSubmit, std::string imageDebugName)
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
							.format = gpuImage->getCreationParameters().format
						};
						ret.gpuImageView = device->createImageView(std::move(viewParams));
						if (ret.gpuImageView)
						{
							// SUCCESS!
							ret.gpuImageView->setObjectDebugName((imageDebugName + " View").c_str());
						}
						else
						{
							// irrecoverable error if simple image creation fails.
							// TODO[LOG]: that's rare, image view creation failed.
							_NBL_DEBUG_BREAK_IF(true);
						}

						// succcessful with everything, just break and get out of this retry loop
						break;
					}
					else
					{
						// irrecoverable error if simple bindImageMemory fails.
						// TODO: LOG
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
				// irrecoverable error if memory requirements of the image don't match our preallocated devicememory
				// TODO: LOG
				_NBL_DEBUG_BREAK_IF(true);
				break;
			}
		}
		else
		{
			// irrecoverable error if simple image creation fails.
			// TODO: LOG
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
			_NBL_DEBUG_BREAK_IF(true);
			// TODO[LOG]
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

void DrawResourcesFiller::determineGeoreferencedImageCreationParams(nbl::asset::IImage::SCreationParams& outImageParams, ImageType& outImageType, const GeoreferencedImageParams& georeferencedImageParams)
{
	// Decide whether the image can reside fully into memory rather than get streamed.
	// TODO: Improve logic, currently just a simple check to see if the full-screen image has more pixels that viewport or not
	// TODO: add criterial that the size of the full-res image shouldn't  consume more than 30% of the total memory arena for images (if we allowed larger than viewport extents)
	const bool betterToResideFullyInMem = georeferencedImageParams.imageExtents.x * georeferencedImageParams.imageExtents.y <= georeferencedImageParams.viewportExtents.x * georeferencedImageParams.viewportExtents.y;

	if (betterToResideFullyInMem)
		outImageType = ImageType::GEOREFERENCED_FULL_RESOLUTION;
	else
		outImageType = ImageType::GEOREFERENCED_STREAMED;

	outImageParams.type = asset::IImage::ET_2D;
	outImageParams.samples = asset::IImage::ESCF_1_BIT;
	outImageParams.format = georeferencedImageParams.format;

	if (outImageType == ImageType::GEOREFERENCED_FULL_RESOLUTION)
	{
		outImageParams.extent = { georeferencedImageParams.imageExtents.x, georeferencedImageParams.imageExtents.y, 1u };
	}
	else
	{
		// TODO: Better Logic, area around the view, etc...
		outImageParams.extent = { georeferencedImageParams.viewportExtents.x, georeferencedImageParams.viewportExtents.y, 1u };
	}


	outImageParams.mipLevels = 1u; // TODO: Later do mipmapping
	outImageParams.arrayLayers = 1u;
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
		return InvalidTextureIndex; // TODO: Log

	const auto cpuImageSize = cpuImage->getMipSize(0);
	const bool sizeMatch = cpuImageSize.x == getMSDFResolution().x && cpuImageSize.y == getMSDFResolution().y && cpuImageSize.z == 1u;
	if (!sizeMatch)
		return InvalidTextureIndex; // TODO: Log

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
			// TODO: log here, assert will be called in a few lines
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