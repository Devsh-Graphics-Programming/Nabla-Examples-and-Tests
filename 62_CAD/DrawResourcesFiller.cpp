#include "DrawResourcesFiller.h"

DrawResourcesFiller::DrawResourcesFiller()
{}

DrawResourcesFiller::DrawResourcesFiller(smart_refctd_ptr<IUtilities>&& utils, IQueue* copyQueue) :
	m_utilities(utils),
	m_copyQueue(copyQueue)
{}

// function is called when buffer is filled and we should submit draws and clear the buffers and continue filling

void DrawResourcesFiller::setSubmitDrawsFunction(const SubmitFunc& func)
{
	submitDraws = func;
}

void DrawResourcesFiller::allocateResourcesBuffer(ILogicalDevice* logicalDevice, size_t size)
{
	size = core::alignUp(size, ResourcesMaxNaturalAlignment);
	size = core::max(size, getMinimumRequiredResourcesBufferSize());
	IGPUBuffer::SCreationParams geometryCreationParams = {};
	geometryCreationParams.size = size;
	geometryCreationParams.usage = bitflag(IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INDEX_BUFFER_BIT;
	resourcesGPUBuffer = logicalDevice->createBuffer(std::move(geometryCreationParams));
	resourcesGPUBuffer->setObjectDebugName("drawResourcesBuffer");

	IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = resourcesGPUBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto mem = logicalDevice->allocate(memReq, resourcesGPUBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
}

void DrawResourcesFiller::allocateMSDFTextures(ILogicalDevice* logicalDevice, uint32_t maxMSDFs, uint32_t2 msdfsExtent)
{
	msdfLRUCache = std::unique_ptr<MSDFsLRUCache>(new MSDFsLRUCache(maxMSDFs));
	msdfTextureArrayIndexAllocator = core::make_smart_refctd_ptr<IndexAllocator>(core::smart_refctd_ptr<ILogicalDevice>(logicalDevice), maxMSDFs);

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
}

void DrawResourcesFiller::drawPolyline(const CPolylineBase& polyline, const LineStyleInfo& lineStyleInfo, SIntendedSubmitInfo& intendedNextSubmit)
{
	if (!lineStyleInfo.isVisible())
		return;

	uint32_t styleIdx = addLineStyle_SubmitIfNeeded(lineStyleInfo, intendedNextSubmit);

	uint32_t mainObjIdx = addMainObject_SubmitIfNeeded(styleIdx, InvalidDTMSettingsIdx, intendedNextSubmit);

	drawPolyline(polyline, mainObjIdx, intendedNextSubmit);
}

void DrawResourcesFiller::drawPolyline(const CPolylineBase& polyline, uint32_t polylineMainObjIdx, SIntendedSubmitInfo& intendedNextSubmit)
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
			submitCurrentDrawObjectsAndReset(intendedNextSubmit, polylineMainObjIdx);
	}

	return; // TODO: Remove

	if (!polyline.getConnectors().empty())
	{
		uint32_t currentConnectorPolylineObject = 0u;
		while (currentConnectorPolylineObject < polyline.getConnectors().size())
		{
			addPolylineConnectors_Internal(polyline, currentConnectorPolylineObject, polylineMainObjIdx);

			if (currentConnectorPolylineObject < polyline.getConnectors().size())
				submitCurrentDrawObjectsAndReset(intendedNextSubmit, polylineMainObjIdx);
		}
	}
}

void DrawResourcesFiller::drawTriangleMesh(const CTriangleMesh& mesh, CTriangleMesh::DrawData& drawData, const DTMSettingsInfo& dtmSettingsInfo, SIntendedSubmitInfo& intendedNextSubmit)
{
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
		drawData.pushConstants.triangleMeshVerticesBaseAddress = geometryBufferOffset;
		memcpy(dst, mesh.getVertices().data(), vtxBuffByteSize);
		geometryBufferOffset += vtxBuffByteSize; 

		// Copy IndexBuffer
		dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
		drawData.indexBufferOffset = geometryBufferOffset;
		memcpy(dst, mesh.getIndices().data(), indexBuffByteSize);
		geometryBufferOffset += indexBuffByteSize;
	}

	drawData.indexCount = mesh.getIndexCount();

	// call addMainObject_SubmitIfNeeded, use its index in push constants

	uint32_t dtmSettingsIndex = addDTMSettings_SubmitIfNeeded(dtmSettingsInfo, intendedNextSubmit);

	drawData.pushConstants.triangleMeshMainObjectIndex = addMainObject_SubmitIfNeeded(InvalidStyleIdx, dtmSettingsIndex, intendedNextSubmit);
}

// TODO[Erfan]: Makes more sense if parameters are: solidColor + fillPattern + patternColor
void DrawResourcesFiller::drawHatch(
		const Hatch& hatch,
		const float32_t4& foregroundColor, 
		const float32_t4& backgroundColor,
		const HatchFillPattern fillPattern,
		SIntendedSubmitInfo& intendedNextSubmit)
{
	// TODO[Optimization Idea]: don't draw hatch twice if both colors are visible: instead do the msdf inside the alpha resolve by detecting mainObj being a hatch
	// https://discord.com/channels/593902898015109131/856835291712716820/1228337893366300743
	// TODO: Come back to this idea when doing color resolve for ecws (they don't have mainObj/style Index, instead they have uv into a texture	

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
	return; // TODO: Remove
	if (color.a == 0.0f) // not visible
		return;

	uint32_t textureIdx = InvalidTextureIdx;
	if (fillPattern != HatchFillPattern::SOLID_FILL)
	{
		MSDFInputInfo msdfInfo = MSDFInputInfo(fillPattern);
		textureIdx = getMSDFIndexFromInputInfo(msdfInfo, intendedNextSubmit);
		if (textureIdx == InvalidTextureIdx)
			textureIdx = addMSDFTexture(msdfInfo, getHatchFillPatternMSDF(fillPattern), InvalidMainObjectIdx, intendedNextSubmit);
		_NBL_DEBUG_BREAK_IF(textureIdx == InvalidTextureIdx); // probably getHatchFillPatternMSDF returned nullptr
	}

	LineStyleInfo lineStyle = {};
	lineStyle.color = color;
	lineStyle.screenSpaceLineWidth = nbl::hlsl::bit_cast<float, uint32_t>(textureIdx);
	const uint32_t styleIdx = addLineStyle_SubmitIfNeeded(lineStyle, intendedNextSubmit);

	uint32_t mainObjIdx = addMainObject_SubmitIfNeeded(styleIdx, InvalidDTMSettingsIdx, intendedNextSubmit);
	uint32_t currentObjectInSection = 0u; // Object here refers to DrawObject used in vertex shader. You can think of it as a Cage.
	while (currentObjectInSection < hatch.getHatchBoxCount())
	{
		addHatch_Internal(hatch, currentObjectInSection, mainObjIdx);
		if (currentObjectInSection < hatch.getHatchBoxCount())
			submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjIdx);
	}
}

void DrawResourcesFiller::drawHatch(const Hatch& hatch, const float32_t4& color, SIntendedSubmitInfo& intendedNextSubmit)
{
	drawHatch(hatch, color, HatchFillPattern::SOLID_FILL, intendedNextSubmit);
}

// TODO: FIX
void DrawResourcesFiller::drawFontGlyph(
		nbl::ext::TextRendering::FontFace* fontFace,
		uint32_t glyphIdx,
		float64_t2 topLeft,
		float32_t2 dirU,
		float32_t  aspectRatio,
		float32_t2 minUV,
		uint32_t mainObjIdx,
		SIntendedSubmitInfo& intendedNextSubmit)
{
#if 0
	uint32_t textureIdx = InvalidTextureIdx;
	const MSDFInputInfo msdfInput = MSDFInputInfo(fontFace->getHash(), glyphIdx);
	textureIdx = getMSDFIndexFromInputInfo(msdfInput, intendedNextSubmit);
	if (textureIdx == InvalidTextureIdx)
		textureIdx = addMSDFTexture(msdfInput, getGlyphMSDF(fontFace, glyphIdx), mainObjIdx, intendedNextSubmit);

	if (textureIdx != InvalidTextureIdx)
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
#endif
}

// TODO: FIX
void DrawResourcesFiller::_test_addImageObject(float64_t2 topLeftPos, float32_t2 size, float32_t rotation, SIntendedSubmitInfo& intendedNextSubmit)
{
#if 0
	auto addImageObject_Internal = [&](const ImageObjectInfo& imageObjectInfo, uint32_t mainObjIdx) -> bool
		{
			const uint32_t maxGeometryBufferImageObjects = static_cast<uint32_t>((maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(ImageObjectInfo));
			uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
			uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
			uploadableObjects = core::min(uploadableObjects, maxGeometryBufferImageObjects);

			if (uploadableObjects >= 1u)
			{
				void* dstGeom = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
				memcpy(dstGeom, &imageObjectInfo, sizeof(ImageObjectInfo));
				uint64_t geomBufferAddr = drawResourcesBDA + currentGeometryBufferSize;
				currentGeometryBufferSize += sizeof(ImageObjectInfo);

				DrawObject drawObj = {};
				drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::IMAGE) | (0 << 16)); // TODO: use custom pack/unpack function
				drawObj.mainObjIndex = mainObjIdx;
				drawObj.geometryAddress = geomBufferAddr;
				void* dstDrawObj = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
				memcpy(dstDrawObj, &drawObj, sizeof(DrawObject));
				currentDrawObjectCount += 1u;

				return true;
			}
			else
				return false;
		};

	uint32_t mainObjIdx = addMainObject_SubmitIfNeeded(InvalidStyleIdx, InvalidDTMSettingsIdx, intendedNextSubmit);

	ImageObjectInfo info = {};
	info.topLeft = topLeftPos;
	info.dirU = float32_t2(size.x * cos(rotation), size.x * sin(rotation)); // 
	info.aspectRatio = size.y / size.x;
	info.textureID = 0u;
	if (!addImageObject_Internal(info, mainObjIdx))
	{
		// single image object couldn't fit into memory to push to gpu, so we submit rendering current objects and reset geometry buffer and draw objects
		submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjIdx);
		bool success = addImageObject_Internal(info, mainObjIdx);
		assert(success); // this should always be true, otherwise it's either bug in code or not enough memory allocated to hold a single image object 
	}
#endif
}

bool DrawResourcesFiller::finalizeAllCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit)
{
	bool success = true;
	success &= finalizeBufferCopies(intendedNextSubmit);
	success &= finalizeTextureCopies(intendedNextSubmit);
	return success;
}

uint32_t DrawResourcesFiller::addLineStyle_SubmitIfNeeded(const LineStyleInfo& lineStyle, SIntendedSubmitInfo& intendedNextSubmit)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	const bool enoughMem = remainingResourcesSize >= sizeof(LineStyle); // enough remaining memory for 1 more linestyle?
	
	uint32_t outLineStyleIdx = addLineStyle_Internal(lineStyle);
	if (outLineStyleIdx == InvalidStyleIdx)
	{
		// There wasn't enough resource memory remaining to fit a single LineStyle
		finalizeAllCopiesToGPU(intendedNextSubmit);
		submitDraws(intendedNextSubmit);
		
		// resets itself
		resetLineStyles();
		// resets higher level resources
		resetMainObjects();
		resetDrawObjects();

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
		finalizeAllCopiesToGPU(intendedNextSubmit);
		submitDraws(intendedNextSubmit);
		
		// resets itself
		resetDTMSettings();
		resetLineStyles(); // additionally resets linestyles as well, just to be safe
		// resets higher level resources
		resetMainObjects();
		resetDrawObjects();

		outDTMSettingIdx = addDTMSettings_Internal(dtmSettings, intendedNextSubmit);
		assert(outDTMSettingIdx != InvalidDTMSettingsIdx);
	}
	return outDTMSettingIdx;
}

uint32_t DrawResourcesFiller::addMainObject_SubmitIfNeeded(uint32_t styleIdx, uint32_t dtmSettingsIdx, SIntendedSubmitInfo& intendedNextSubmit)
{
	MainObject mainObject = {};
	mainObject.styleIdx = styleIdx;
	mainObject.dtmSettingsIdx = dtmSettingsIdx;
	mainObject.clipProjectionIndex = acquireCurrentClipProjectionIndex(intendedNextSubmit);
	uint32_t outMainObjectIdx = addMainObject_Internal(mainObject);
	if (outMainObjectIdx == InvalidMainObjectIdx)
	{
		// failed to fit into remaining resources mem or exceeded max indexable mainobj
		finalizeAllCopiesToGPU(intendedNextSubmit);
		submitDraws(intendedNextSubmit);
		
		// resets itself
		resetMainObjects();
		// resets higher level resources
		resetDrawObjects();
		// we shouldn't reset lower level resources like linestyles and clip projections here because it was possibly requested to push to mem before addMainObjects

		// try to add again
		outMainObjectIdx = addMainObject_Internal(mainObject);
		assert(outMainObjectIdx != InvalidMainObjectIdx);
	}
	
	return outMainObjectIdx;
}

void DrawResourcesFiller::pushClipProjectionData(const ClipProjectionData& clipProjectionData)
{
	clipProjections.push_back(clipProjectionData);
	clipProjectionIndices.push_back(InvalidClipProjectionIndex);
}

void DrawResourcesFiller::popClipProjectionData()
{
	if (clipProjections.empty())
		return;

	clipProjections.pop_back();
	clipProjectionIndices.pop_back();
}

bool DrawResourcesFiller::finalizeBufferCopies(SIntendedSubmitInfo& intendedNextSubmit)
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

			if (copyRange.size > 0ull)
			{
				drawBuffer.bufferOffset = copyRange.offset;
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

	copyCPUFilledDrawBuffer(resourcesCollection.lineStyles);
	copyCPUFilledDrawBuffer(resourcesCollection.dtmSettings);
	copyCPUFilledDrawBuffer(resourcesCollection.clipProjections);
	copyCPUFilledDrawBuffer(resourcesCollection.mainObjects);
	copyCPUFilledDrawBuffer(resourcesCollection.drawObjects);
	copyCPUFilledDrawBuffer(resourcesCollection.indexBuffer);
	copyCPUFilledDrawBuffer(resourcesCollection.geometryInfo);
	
	return true;
}

bool DrawResourcesFiller::finalizeTextureCopies(SIntendedSubmitInfo& intendedNextSubmit)
{
	msdfTextureArrayIndicesUsed.clear(); // clear msdf textures used in the frame, because the frame finished and called this function.

	if (!msdfTextureCopies.size() && m_hasInitializedMSDFTextureArrays) // even if the textureCopies are empty, we want to continue if not initialized yet so that the layout of all layers become READ_ONLY_OPTIMAL
		return true; // yay successfully copied nothing

	auto* cmdBuffInfo = intendedNextSubmit.getCommandBufferForRecording();
	
	if (cmdBuffInfo)
	{
		IGPUCommandBuffer* cmdBuff = cmdBuffInfo->cmdbuf;

		auto msdfImage = msdfTextureArray->getCreationParameters().image;

		// preparing msdfs for copy
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
		cmdBuff->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = beforeTransferImageBarrier });

		// Do the copies and advance the iterator.
		// this is the pattern we use for iterating when entries will get erased if processed successfully, but may get skipped for later.
		auto oit = msdfTextureCopies.begin();
		for (auto iit = msdfTextureCopies.begin(); iit != msdfTextureCopies.end(); iit++)
		{
			bool copySuccess = true;
			if (iit->image && iit->index < msdfImage->getCreationParameters().arrayLayers)
			{
				for (uint32_t mip = 0; mip < iit->image->getCreationParameters().mipLevels; mip++)
				{
					auto mipImageRegion = iit->image->getRegion(mip, core::vectorSIMDu32(0u, 0u));
					if (mipImageRegion)
					{
						asset::IImage::SBufferCopy region = {};
						region.imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
						region.imageSubresource.mipLevel = mipImageRegion->imageSubresource.mipLevel;
						region.imageSubresource.baseArrayLayer = iit->index;
						region.imageSubresource.layerCount = 1u;
						region.bufferOffset = 0u;
						region.bufferRowLength = mipImageRegion->getExtent().width;
						region.bufferImageHeight = 0u;
						region.imageExtent = mipImageRegion->imageExtent;
						region.imageOffset = { 0u, 0u, 0u };

						auto buffer = reinterpret_cast<uint8_t*>(iit->image->getBuffer()->getPointer());
						auto bufferOffset = mipImageRegion->bufferOffset;

						if (!m_utilities->updateImageViaStagingBuffer(
							intendedNextSubmit,
							buffer + bufferOffset,
							nbl::ext::TextRendering::TextRenderer::MSDFTextureFormat,
							msdfImage.get(),
							IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
							{ &region, &region + 1 }))
						{
							// TODO: Log which mip failed
							copySuccess = false;
						}
					}
					else
					{
						// TODO: Log
						copySuccess = false;
					}
				}
			}
			else
			{
				assert(false);
				copySuccess = false;
			}

			if (!copySuccess)
			{
				// we move the failed copy to the oit and advance it
				if (oit != iit)
					*oit = *iit;
				oit++;
			}
		}
		// trim
		const auto newSize = std::distance(msdfTextureCopies.begin(), oit);
		_NBL_DEBUG_BREAK_IF(newSize != 0u); // we had failed copies
		msdfTextureCopies.resize(newSize);

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
		cmdBuff->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = afterTransferImageBarrier });
		
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

const size_t DrawResourcesFiller::calculateRemainingResourcesSize() const
{
	assert(resourcesGPUBuffer->getSize() >= resourcesCollection.calculateTotalConsumption());
	return resourcesGPUBuffer->getSize() - resourcesCollection.calculateTotalConsumption();
}

void DrawResourcesFiller::submitCurrentDrawObjectsAndReset(SIntendedSubmitInfo& intendedNextSubmit, uint32_t mainObjectIndex)
{
	finalizeAllCopiesToGPU(intendedNextSubmit);
	submitDraws(intendedNextSubmit);

	// We reset Geometry Counters (drawObj+geometryInfos) because we're done rendering previous geometry
	// We don't reset counters for styles because we will be reusing them
	resetDrawObjects();
}

uint32_t DrawResourcesFiller::addMainObject_Internal(const MainObject& mainObject)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	const size_t memRequired = sizeof(MainObject);
	const bool enoughMem = remainingResourcesSize >= memRequired; // enough remaining memory for 1 more dtm settings with 2 referenced line styles?
	if (!enoughMem)
		return InvalidMainObjectIdx;
	if (resourcesCollection.mainObjects.vector.size() >= MaxIndexableMainObjects)
		return InvalidMainObjectIdx;
	resourcesCollection.mainObjects.vector.push_back(mainObject); // this will implicitly increase total resource consumption and reduce remaining size --> no need for mem size trackers
	return resourcesCollection.mainObjects.vector.size() - 1u;
}

uint32_t DrawResourcesFiller::addLineStyle_Internal(const LineStyleInfo& lineStyleInfo)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	const bool enoughMem = remainingResourcesSize >= sizeof(LineStyle); // enough remaining memory for 1 more linestyle?
	if (!enoughMem)
		return InvalidStyleIdx;
	// TODO: Additionally constraint by a max size? and return InvalidIdx if it would exceed


	LineStyle gpuLineStyle = lineStyleInfo.getAsGPUData();
	_NBL_DEBUG_BREAK_IF(gpuLineStyle.stipplePatternSize > LineStyle::StipplePatternMaxSize); // Oops, even after style normalization the style is too long to be in gpu mem :(
	for (uint32_t i = 0u; i < resourcesCollection.lineStyles.vector.size(); ++i)
	{
		const LineStyle& itr = resourcesCollection.lineStyles.vector[i];
		if (itr == gpuLineStyle)
			return i;
	}

	resourcesCollection.lineStyles.vector.push_back(gpuLineStyle); // this will implicitly increase total resource consumption and reduce remaining size --> no need for mem size trackers
	return resourcesCollection.lineStyles.vector.size() - 1u;
}

uint32_t DrawResourcesFiller::addDTMSettings_Internal(const DTMSettingsInfo& dtmSettingsInfo, SIntendedSubmitInfo& intendedNextSubmit)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	const size_t maxMemRequired = sizeof(DTMSettings) + 2 * sizeof(LineStyle);
	const bool enoughMem = remainingResourcesSize >= maxMemRequired; // enough remaining memory for 1 more dtm settings with 2 referenced line styles?

	if (!enoughMem)
		return InvalidDTMSettingsIdx;
	// TODO: Additionally constraint by a max size? and return InvalidIdx if it would exceed

	DTMSettings dtmSettings;
	dtmSettings.contourLinesStartHeight = dtmSettingsInfo.contourLinesStartHeight;
	dtmSettings.contourLinesEndHeight = dtmSettingsInfo.contourLinesEndHeight;
	dtmSettings.contourLinesHeightInterval = dtmSettingsInfo.contourLinesHeightInterval;

	dtmSettings.outlineLineStyleIdx = addLineStyle_Internal(dtmSettingsInfo.outlineLineStyleInfo);
	dtmSettings.contourLineStyleIdx = addLineStyle_Internal(dtmSettingsInfo.contourLineStyleInfo);

	switch (dtmSettingsInfo.heightShadingMode)
	{
	case DTMSettingsInfo::E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS:
		dtmSettings.intervalWidth = std::numeric_limits<float>::infinity();
		break;
	case DTMSettingsInfo::E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS:
		dtmSettings.intervalWidth = dtmSettingsInfo.intervalWidth;
		break;
	case DTMSettingsInfo::E_HEIGHT_SHADING_MODE::CONTINOUS_INTERVALS:
		dtmSettings.intervalWidth = 0.0f;
		break;
	}
	_NBL_DEBUG_BREAK_IF(!dtmSettingsInfo.fillShaderDTMSettingsHeightColorMap(dtmSettings));

	for (uint32_t i = 0u; i < resourcesCollection.dtmSettings.vector.size(); ++i)
	{
		const DTMSettings& itr = resourcesCollection.dtmSettings.vector[i];
		if (itr == dtmSettings)
			return i;
	}
	
	resourcesCollection.dtmSettings.vector.push_back(dtmSettings); // this will implicitly increase total resource consumption and reduce remaining size --> no need for mem size trackers
	return resourcesCollection.dtmSettings.vector.size() - 1u;
}

uint32_t DrawResourcesFiller::acquireCurrentClipProjectionIndex(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (clipProjectionIndices.empty())
		return InvalidClipProjectionIndex;

	if (clipProjectionIndices.back() == InvalidClipProjectionIndex)
		clipProjectionIndices.back() = addClipProjectionData_SubmitIfNeeded(clipProjections.back(), intendedNextSubmit);
	
	return clipProjectionIndices.back();
}

uint32_t DrawResourcesFiller::addClipProjectionData_SubmitIfNeeded(const ClipProjectionData& clipProjectionData, SIntendedSubmitInfo& intendedNextSubmit)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	const size_t memRequired = sizeof(ClipProjectionData);
	const bool enoughMem = remainingResourcesSize >= memRequired; // enough remaining memory for 1 more dtm settings with 2 referenced line styles?

	if (!enoughMem)
	{
		finalizeAllCopiesToGPU(intendedNextSubmit);
		submitDraws(intendedNextSubmit);
		
		// resets itself
		resetCustomClipProjections();
		// resets higher level resources
		resetMainObjects();
		resetDrawObjects();
	}
	
	resourcesCollection.clipProjections.vector.push_back(clipProjectionData); // this will implicitly increase total resource consumption and reduce remaining size --> no need for mem size trackers
	return resourcesCollection.clipProjections.vector.size() - 1u;
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

// TODO: FIX
void DrawResourcesFiller::addPolylineConnectors_Internal(const CPolylineBase& polyline, uint32_t& currentPolylineConnectorObj, uint32_t mainObjIdx)
{
#if 0
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();

	const uint32_t uploadableObjects = (remainingResourcesSize) / (sizeof(PolylineConnector) + sizeof(DrawObject) + sizeof(uint32_t) * 6u);
	// TODO[ERFAN]: later take into account, our limit of max index buffer and vettex buffer size or constrainst other than mem
	
	const uint32_t connectorCount = static_cast<uint32_t>(polyline.getConnectors().size());
	const uint32_t remainingObjects = connectorCount - currentPolylineConnectorObj;
	const uint32_t objectsToUpload = core::min(uploadableObjects, remainingObjects);

	if (objectsToUpload <= 0u)
		return;





	// TODO: 





	// Add DrawObjs
	DrawObject drawObj = {};
	drawObj.mainObjIndex = mainObjIdx;
	drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::POLYLINE_CONNECTOR) | 0 << 16);
	drawObj.geometryAddress = drawResourcesBDA + currentGeometryBufferSize;
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
#endif
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
	// TODO[ERFAN]: later take into account, our limit of max index buffer and vettex buffer size or constrainst other than mem

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
	for (uint32_t i = 0u; i < objectsToUpload; ++i)
	{
		indexBufferToBeFilled[i*6]		= i*4u + 1u;
		indexBufferToBeFilled[i*6 + 1u]	= i*4u + 0u;
		indexBufferToBeFilled[i*6 + 2u]	= i*4u + 2u;
		indexBufferToBeFilled[i*6 + 3u]	= i*4u + 1u;
		indexBufferToBeFilled[i*6 + 4u]	= i*4u + 2u;
		indexBufferToBeFilled[i*6 + 5u]	= i*4u + 3u;
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
	// TODO[ERFAN]: later take into account, our limit of max index buffer and vettex buffer size or constrainst other than mem
	
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
	for (uint32_t i = 0u; i < cagesCount; ++i)
	{
		indexBufferToBeFilled[i*6]		= i*4u + 1u;
		indexBufferToBeFilled[i*6 + 1u]	= i*4u + 0u;
		indexBufferToBeFilled[i*6 + 2u]	= i*4u + 2u;
		indexBufferToBeFilled[i*6 + 3u]	= i*4u + 1u;
		indexBufferToBeFilled[i*6 + 4u]	= i*4u + 2u;
		indexBufferToBeFilled[i*6 + 5u]	= i*4u + 3u;
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

// TODO: FIX
void DrawResourcesFiller::addHatch_Internal(const Hatch& hatch, uint32_t& currentObjectInSection, uint32_t mainObjIndex)
{
#if 0
	const uint32_t maxGeometryBufferHatchBoxes = static_cast<uint32_t>((maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(Hatch::CurveHatchBox));
	
	uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
	uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
	uploadableObjects = core::min(uploadableObjects, maxGeometryBufferHatchBoxes);

	uint32_t remainingObjects = hatch.getHatchBoxCount() - currentObjectInSection;
	uploadableObjects = core::min(uploadableObjects, remainingObjects);

	for (uint32_t i = 0; i < uploadableObjects; i++)
	{
		const Hatch::CurveHatchBox& hatchBox = hatch.getHatchBox(i + currentObjectInSection);

		uint64_t hatchBoxAddress;
		{			
			static_assert(sizeof(CurveBox) == sizeof(Hatch::CurveHatchBox));
			void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
			memcpy(dst, &hatchBox, sizeof(CurveBox));
			hatchBoxAddress = drawResourcesBDA + currentGeometryBufferSize;
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
#endif
}

// TODO: FIX
bool DrawResourcesFiller::addFontGlyph_Internal(const GlyphInfo& glyphInfo, uint32_t mainObjIdx)
{
#if 0
	const uint32_t maxGeometryBufferFontGlyphs = static_cast<uint32_t>((maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(GlyphInfo));
	
	uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
	uploadableObjects = core::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
	uploadableObjects = core::min(uploadableObjects, maxGeometryBufferFontGlyphs);

	if (uploadableObjects >= 1u)
	{
		void* geomDst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
		memcpy(geomDst, &glyphInfo, sizeof(GlyphInfo));
		uint64_t fontGlyphAddr = drawResourcesBDA + currentGeometryBufferSize;
		currentGeometryBufferSize += sizeof(GlyphInfo);

		DrawObject drawObj = {};
		drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::FONT_GLYPH) | (0 << 16));
		drawObj.mainObjIndex = mainObjIdx;
		drawObj.geometryAddress = fontGlyphAddr;
		void* drawObjDst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
		memcpy(drawObjDst, &drawObj, sizeof(DrawObject));
		currentDrawObjectCount += 1u;

		return true;
	}
	else
	{
		return false;
	}
#endif
}

void DrawResourcesFiller::setGlyphMSDFTextureFunction(const GetGlyphMSDFTextureFunc& func)
{
	getGlyphMSDF = func;
}

void DrawResourcesFiller::setHatchFillMSDFTextureFunction(const GetHatchFillPatternMSDFTextureFunc& func)
{
	getHatchFillPatternMSDF = func;
}

uint32_t DrawResourcesFiller::addMSDFTexture(const MSDFInputInfo& msdfInput, core::smart_refctd_ptr<ICPUImage>&& cpuImage, uint32_t mainObjIdx, SIntendedSubmitInfo& intendedNextSubmit)
{
	if (!cpuImage)
		return InvalidTextureIdx; // TODO: Log

	const auto cpuImageSize = cpuImage->getMipSize(0);
	const bool sizeMatch = cpuImageSize.x == getMSDFResolution().x && cpuImageSize.y == getMSDFResolution().y && cpuImageSize.z == 1u;
	if (!sizeMatch)
		return InvalidTextureIdx; // TODO: Log

	// TextureReferences hold the semaValue related to the "scratch semaphore" in IntendedSubmitInfo
	// Every single submit increases this value by 1
	// The reason for hiolding on to the lastUsedSema is deferred dealloc, which we call in the case of eviction, making sure we get rid of the entry inside the allocator only when the texture is done being used
	const auto nextSemaSignal = intendedNextSubmit.getFutureScratchSemaphore();

	auto evictionCallback = [&](const MSDFReference& evicted)
	{
		if (msdfTextureArrayIndicesUsed.contains(evicted.alloc_idx)) 
		{
			// Dealloc once submission is finished
			msdfTextureArrayIndexAllocator->multi_deallocate(1u, &evicted.alloc_idx, nextSemaSignal);

			// If we reset main objects will cause an auto submission bug, where adding an msdf texture while constructing glyphs will have wrong main object references (See how SingleLineTexts add Glyphs with a single mainObject)
			// for the same reason we don't reset line styles
			submitCurrentDrawObjectsAndReset(intendedNextSubmit, mainObjIdx);
		} 
		else
		{
			// We didn't use it this frame, so it's safe to dealloc now, withou needing to "overflow" submit
			msdfTextureArrayIndexAllocator->multi_deallocate(1u, &evicted.alloc_idx);
		}
	};
	
	// We pass nextSemaValue instead of constructing a new MSDFReference and passing it into `insert` that's because we might get a cache hit and only update the value of the nextSema
	MSDFReference* inserted = msdfLRUCache->insert(msdfInput, nextSemaSignal.value, evictionCallback);
	
	// if inserted->alloc_idx was not InvalidTextureIdx then it means we had a cache hit and updated the value of our sema, in which case we don't queue anything for upload, and return the idx
	if (inserted->alloc_idx == InvalidTextureIdx)
	{
		// New insertion == cache miss happened and insertion was successfull
		inserted->alloc_idx = IndexAllocator::AddressAllocator::invalid_address;
		msdfTextureArrayIndexAllocator->multi_allocate(std::chrono::time_point<std::chrono::steady_clock>::max(), 1u, &inserted->alloc_idx); // if the prev submit causes DEVICE_LOST then we'll get a deadlock here since we're using max timepoint

		if (inserted->alloc_idx != IndexAllocator::AddressAllocator::invalid_address)
		{
			// We queue copy and finalize all on `finalizeTextureCopies` function called before draw calls to make sure it's in mem
			msdfTextureCopies.push_back({ .image = std::move(cpuImage), .index = inserted->alloc_idx });
		}
		else
		{
			// TODO: log here, assert will be called in a few lines
			inserted->alloc_idx = InvalidTextureIdx;
		}
	}
	
	assert(inserted->alloc_idx != InvalidTextureIdx); // shouldn't happen, because we're using LRU cache, so worst case eviction will happen + multi-deallocate and next next multi_allocate should definitely succeed
	if (inserted->alloc_idx != InvalidTextureIdx)
		msdfTextureArrayIndicesUsed.emplace(inserted->alloc_idx);

	return inserted->alloc_idx;
}