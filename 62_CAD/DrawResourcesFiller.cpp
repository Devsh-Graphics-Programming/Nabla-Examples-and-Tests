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

void DrawResourcesFiller::allocateIndexBuffer(ILogicalDevice* logicalDevice, uint32_t maxIndices)
{
	maxIndexCount = maxIndices;
	const size_t indexBufferSize = maxIndices * sizeof(index_buffer_type);
	auto indexBuffer = ICPUBuffer::create({ indexBufferSize });

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

	m_utilities->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{.queue=m_copyQueue}, std::move(indexBufferCreationParams), indices).move_into(gpuDrawBuffers.indexBuffer);
	gpuDrawBuffers.indexBuffer->setObjectDebugName("indexBuffer");
}

void DrawResourcesFiller::allocateMainObjectsBuffer(ILogicalDevice* logicalDevice, uint32_t mainObjects)
{
	maxMainObjects = mainObjects;
	size_t mainObjectsBufferSize = maxMainObjects * sizeof(MainObject);

	IGPUBuffer::SCreationParams mainObjectsCreationParams = {};
	mainObjectsCreationParams.size = mainObjectsBufferSize;
	mainObjectsCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuDrawBuffers.mainObjectsBuffer = logicalDevice->createBuffer(std::move(mainObjectsCreationParams));
	gpuDrawBuffers.mainObjectsBuffer->setObjectDebugName("mainObjectsBuffer");

	IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.mainObjectsBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto mainObjectsBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.mainObjectsBuffer.get());

	cpuDrawBuffers.mainObjectsBuffer = ICPUBuffer::create({ mainObjectsBufferSize });
}

void DrawResourcesFiller::allocateDrawObjectsBuffer(ILogicalDevice* logicalDevice, uint32_t drawObjects)
{
	maxDrawObjects = drawObjects;
	size_t drawObjectsBufferSize = maxDrawObjects * sizeof(DrawObject);

	IGPUBuffer::SCreationParams drawObjectsCreationParams = {};
	drawObjectsCreationParams.size = drawObjectsBufferSize;
	drawObjectsCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuDrawBuffers.drawObjectsBuffer = logicalDevice->createBuffer(std::move(drawObjectsCreationParams));
	gpuDrawBuffers.drawObjectsBuffer->setObjectDebugName("drawObjectsBuffer");

	IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.drawObjectsBuffer->getMemoryReqs();
	memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	auto drawObjectsBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.drawObjectsBuffer.get());

	cpuDrawBuffers.drawObjectsBuffer = ICPUBuffer::create({ drawObjectsBufferSize });
}

void DrawResourcesFiller::allocateGeometryBuffer(ILogicalDevice* logicalDevice, size_t size)
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

	cpuDrawBuffers.geometryBuffer = ICPUBuffer::create({ size });
}

void DrawResourcesFiller::allocateStylesBuffer(ILogicalDevice* logicalDevice, uint32_t lineStylesCount)
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

		cpuDrawBuffers.lineStylesBuffer = ICPUBuffer::create({ lineStylesBufferSize });
	}
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

	uint32_t mainObjIdx = addMainObject_SubmitIfNeeded(styleIdx, intendedNextSubmit);

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

	uint32_t mainObjIdx = addMainObject_SubmitIfNeeded(styleIdx, intendedNextSubmit);
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
}

bool DrawResourcesFiller::finalizeAllCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit)
{
	bool success = true;
	success &= finalizeMainObjectCopiesToGPU(intendedNextSubmit);
	success &= finalizeGeometryCopiesToGPU(intendedNextSubmit);
	success &= finalizeLineStyleCopiesToGPU(intendedNextSubmit);
	success &= finalizeTextureCopies(intendedNextSubmit);
	return success;
}

uint32_t DrawResourcesFiller::addLineStyle_SubmitIfNeeded(const LineStyleInfo& lineStyle, SIntendedSubmitInfo& intendedNextSubmit)
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

uint32_t DrawResourcesFiller::addMainObject_SubmitIfNeeded(uint32_t styleIdx, SIntendedSubmitInfo& intendedNextSubmit)
{
	MainObject mainObject = {};
	mainObject.styleIdx = styleIdx;
	mainObject.clipProjectionAddress = acquireCurrentClipProjectionAddress(intendedNextSubmit);
	uint32_t outMainObjectIdx = addMainObject_Internal(mainObject);
	if (outMainObjectIdx == InvalidMainObjectIdx)
	{
		finalizeAllCopiesToGPU(intendedNextSubmit);
		submitDraws(intendedNextSubmit);

		// geometries needs to be reset because they reference draw objects and draw objects reference main objects that are now unavailable and reset
		resetGeometryCounters();
		// mainObjects needs to be reset because we submitted every previous main object
		resetMainObjectCounters();
		// we shouldn't reset linestyles and clip projections here because it was possibly requested to push to mem before addMainObjects
		// but clip projections are reset due to geometry/bda buffer being reset so we need to push again
		
		// acquireCurrentClipProjectionAddress again here because clip projection should exist in the geometry buffer, and reseting geometry counters will invalidate the current clip proj and requires repush
		mainObject.clipProjectionAddress = acquireCurrentClipProjectionAddress(intendedNextSubmit);
		outMainObjectIdx = addMainObject_Internal(mainObject);
		assert(outMainObjectIdx != InvalidMainObjectIdx);
	}
	
	return outMainObjectIdx;
}

void DrawResourcesFiller::pushClipProjectionData(const ClipProjectionData& clipProjectionData)
{
	clipProjections.push_back(clipProjectionData);
	clipProjectionAddresses.push_back(InvalidClipProjectionAddress);
}

void DrawResourcesFiller::popClipProjectionData()
{
	if (clipProjections.empty())
		return;

	clipProjections.pop_back();
	clipProjectionAddresses.pop_back();
}

bool DrawResourcesFiller::finalizeMainObjectCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit)
{
	bool success = true;
	// Copy MainObjects
	uint32_t remainingMainObjects = currentMainObjectCount - inMemMainObjectCount;
	SBufferRange<IGPUBuffer> mainObjectsRange = { sizeof(MainObject) * inMemMainObjectCount, sizeof(MainObject) * remainingMainObjects, gpuDrawBuffers.mainObjectsBuffer };
	if (mainObjectsRange.size > 0u)
	{
		const MainObject* srcMainObjData = reinterpret_cast<MainObject*>(cpuDrawBuffers.mainObjectsBuffer->getPointer()) + inMemMainObjectCount;
		if (m_utilities->updateBufferRangeViaStagingBuffer(intendedNextSubmit, mainObjectsRange, srcMainObjData))
			inMemMainObjectCount = currentMainObjectCount;
		else
		{
			// TODO: Log
			success = false;
		}
	}
	return success;
}

bool DrawResourcesFiller::finalizeGeometryCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit)
{
	bool success = true;
	// Copy DrawObjects
	uint32_t remainingDrawObjects = currentDrawObjectCount - inMemDrawObjectCount;
	SBufferRange<IGPUBuffer> drawObjectsRange = { sizeof(DrawObject) * inMemDrawObjectCount, sizeof(DrawObject) * remainingDrawObjects, gpuDrawBuffers.drawObjectsBuffer };
	if (drawObjectsRange.size > 0u)
	{
		const DrawObject* srcDrawObjData = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + inMemDrawObjectCount;
		if (m_utilities->updateBufferRangeViaStagingBuffer(intendedNextSubmit, drawObjectsRange, srcDrawObjData))
			inMemDrawObjectCount = currentDrawObjectCount;
		else
		{
			// TODO: Log
			success = false;
		}
	}

	// Copy GeometryBuffer
	uint64_t remainingGeometrySize = currentGeometryBufferSize - inMemGeometryBufferSize;
	SBufferRange<IGPUBuffer> geomRange = { inMemGeometryBufferSize, remainingGeometrySize, gpuDrawBuffers.geometryBuffer };
	if (geomRange.size > 0u)
	{
		const uint8_t* srcGeomData = reinterpret_cast<uint8_t*>(cpuDrawBuffers.geometryBuffer->getPointer()) + inMemGeometryBufferSize;
		if (m_utilities->updateBufferRangeViaStagingBuffer(intendedNextSubmit, geomRange, srcGeomData))
			inMemGeometryBufferSize = currentGeometryBufferSize;
		else
		{
			// TODO: Log
			success = false;
		}
	}
	return success;
}

bool DrawResourcesFiller::finalizeLineStyleCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit)
{
	bool success = true;
	// Copy LineStyles
	uint32_t remainingLineStyles = currentLineStylesCount - inMemLineStylesCount;
	SBufferRange<IGPUBuffer> stylesRange = { sizeof(LineStyle) * inMemLineStylesCount, sizeof(LineStyle) * remainingLineStyles, gpuDrawBuffers.lineStylesBuffer };
	if (stylesRange.size > 0u)
	{
		const LineStyle* srcLineStylesData = reinterpret_cast<LineStyle*>(cpuDrawBuffers.lineStylesBuffer->getPointer()) + inMemLineStylesCount;
		if (m_utilities->updateBufferRangeViaStagingBuffer(intendedNextSubmit, stylesRange, srcLineStylesData))
			inMemLineStylesCount = currentLineStylesCount;
		else
		{
			// TODO: Log
			success = false;
		}
	}
	return success;
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

void DrawResourcesFiller::submitCurrentDrawObjectsAndReset(SIntendedSubmitInfo& intendedNextSubmit, uint32_t mainObjectIndex)
{
	finalizeAllCopiesToGPU(intendedNextSubmit);
	submitDraws(intendedNextSubmit);

	// We reset Geometry Counters (drawObj+geometryInfos) because we're done rendering previous geometry
	// We don't reset counters for styles because we will be reusing them
	resetGeometryCounters();
	
#if 1
	if (mainObjectIndex < maxMainObjects)
	{
		// Check if user is following proper usage, mainObjectIndex should be the last mainObj added before an autosubmit, because this is the only mainObj we want to maintain.
		// See comments on`addMainObject_SubmitIfNeeded` function
		// TODO: consider forcing this by not expose mainObjectIndex to user and keep track of a "currentMainObj" (?)
		_NBL_DEBUG_BREAK_IF(mainObjectIndex != (currentMainObjectCount - 1u)); 

		// If the clip projection stack is non-empty, then it means we need to re-push the clipProjectionData (because it existed in geometry data and it was erased)
		uint64_t newClipProjectionAddress = acquireCurrentClipProjectionAddress(intendedNextSubmit);
		// only re-upload mainObjData if it's clipProjectionAddress was changed
		if (newClipProjectionAddress != getMainObject(mainObjectIndex)->clipProjectionAddress)
		{
			// then modify the mainObject data
			getMainObject(mainObjectIndex)->clipProjectionAddress = newClipProjectionAddress;
			// we need to rewind back inMemMainObjectCount to this mainObjIndex so it re-uploads the current mainObject (because we modified it)
			inMemMainObjectCount = std::min(inMemMainObjectCount, mainObjectIndex);
		}
	}

	// TODO: Consider resetting MainObjects here as well and addMainObject for the new data again, but account for the fact that mainObjectIndex now changed (either change through uint32_t& or keeping track of "currentMainObj" in drawResourcesFiller
#else
	resetMainObjectCounters();

	// If there is a mainObject data we need to maintain and keep it's clipProjectionAddr valid
	if (mainObjectIndex < maxMainObjects)
	{
		MainObject mainObjToMaintain = *getMainObject(mainObjectIndex);

		// If the clip projection stack is non-empty, then it means we need to re-push the clipProjectionData (because it exists in geometry data and it was reset)
		// `acquireCurrentClipProjectionAddress` shouldn't/won't trigger auto-submit because geometry buffer counters were reset and our geometry buffer is supposed to be larger than a single clipProjectionData
		mainObjToMaintain->clipProjectionAddress = acquireCurrentClipProjectionAddress(intendedNextSubmit);
		
		// We're calling `addMainObject_Internal` instead of safer `addMainObject_SubmitIfNeeded` because we've reset our mainObject and we're sure this won't need an autoSubmit.
		addMainObject_Internal(mainObjToMaintain);
	}
#endif
}

uint32_t DrawResourcesFiller::addMainObject_Internal(const MainObject& mainObject)
{
	MainObject* mainObjsArray = reinterpret_cast<MainObject*>(cpuDrawBuffers.mainObjectsBuffer->getPointer());
	
	if (currentMainObjectCount >= MaxIndexableMainObjects)
		return InvalidMainObjectIdx;
	if (currentMainObjectCount >= maxMainObjects)
		return InvalidMainObjectIdx;

	void* dst = mainObjsArray + currentMainObjectCount;
	memcpy(dst, &mainObject, sizeof(MainObject));
	uint32_t ret = currentMainObjectCount;
	currentMainObjectCount++;
	return ret;
}

uint32_t DrawResourcesFiller::addLineStyle_Internal(const LineStyleInfo& lineStyleInfo)
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

uint64_t DrawResourcesFiller::acquireCurrentClipProjectionAddress(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (clipProjectionAddresses.empty())
		return InvalidClipProjectionAddress;

	if (clipProjectionAddresses.back() == InvalidClipProjectionAddress)
		clipProjectionAddresses.back() = addClipProjectionData_SubmitIfNeeded(clipProjections.back(), intendedNextSubmit);
	
	return clipProjectionAddresses.back();
}

uint64_t DrawResourcesFiller::addClipProjectionData_SubmitIfNeeded(const ClipProjectionData& clipProjectionData, SIntendedSubmitInfo& intendedNextSubmit)
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

uint64_t DrawResourcesFiller::addClipProjectionData_Internal(const ClipProjectionData& clipProjectionData)
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
	const auto maxGeometryBufferConnectors = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(PolylineConnector);

	uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
	uploadableObjects = std::min(static_cast<uint64_t>(uploadableObjects), maxGeometryBufferConnectors);
	uploadableObjects = std::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

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

void DrawResourcesFiller::addLines_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
{
	assert(section.count >= 1u);
	assert(section.type == ObjectType::LINE);

	const auto maxGeometryBufferPoints = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(LinePointInfo);
	const auto maxGeometryBufferLines = (maxGeometryBufferPoints <= 1u) ? 0u : maxGeometryBufferPoints - 1u;

	uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
	uploadableObjects = std::min(static_cast<uint64_t>(uploadableObjects), maxGeometryBufferLines);
	uploadableObjects = std::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);

	const auto lineCount = section.count;
	const auto remainingObjects = lineCount - currentObjectInSection;
	uint32_t objectsToUpload = std::min(uploadableObjects, remainingObjects);

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

void DrawResourcesFiller::addQuadBeziers_Internal(const CPolylineBase& polyline, const CPolylineBase::SectionInfo& section, uint32_t& currentObjectInSection, uint32_t mainObjIdx)
{
	constexpr uint32_t CagesPerQuadBezier = getCageCountPerPolylineObject(ObjectType::QUAD_BEZIER);
	assert(section.type == ObjectType::QUAD_BEZIER);

	const auto maxGeometryBufferBeziers = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(QuadraticBezierInfo);
	
	uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
	uploadableObjects = std::min(static_cast<uint64_t>(uploadableObjects), maxGeometryBufferBeziers);
	uploadableObjects = std::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
	uploadableObjects /= CagesPerQuadBezier;

	const auto beziersCount = section.count;
	const auto remainingObjects = beziersCount - currentObjectInSection;
	uint32_t objectsToUpload = std::min(uploadableObjects, remainingObjects);

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

void DrawResourcesFiller::addHatch_Internal(const Hatch& hatch, uint32_t& currentObjectInSection, uint32_t mainObjIndex)
{
	const auto maxGeometryBufferHatchBoxes = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(Hatch::CurveHatchBox);
	
	uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
	uploadableObjects = std::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
	uploadableObjects = std::min(static_cast<uint64_t>(uploadableObjects), maxGeometryBufferHatchBoxes);

	uint32_t remainingObjects = hatch.getHatchBoxCount() - currentObjectInSection;
	uploadableObjects = std::min(uploadableObjects, remainingObjects);

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

bool DrawResourcesFiller::addFontGlyph_Internal(const GlyphInfo& glyphInfo, uint32_t mainObjIdx)
{
	const auto maxGeometryBufferFontGlyphs = (maxGeometryBufferSize - currentGeometryBufferSize) / sizeof(GlyphInfo);
	
	uint32_t uploadableObjects = (maxIndexCount / 6u) - currentDrawObjectCount;
	uploadableObjects = std::min(uploadableObjects, maxDrawObjects - currentDrawObjectCount);
	uploadableObjects = std::min(static_cast<uint64_t>(uploadableObjects), maxGeometryBufferFontGlyphs);

	if (uploadableObjects >= 1u)
	{
		void* geomDst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
		memcpy(geomDst, &glyphInfo, sizeof(GlyphInfo));
		uint64_t fontGlyphAddr = geometryBufferAddress + currentGeometryBufferSize;
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
			// `submitCurrentObjectsAndReset` function handles the above + updating clipProjectionData and making sure the mainObjectIdx references to the correct clipProj data after reseting geometry buffer
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