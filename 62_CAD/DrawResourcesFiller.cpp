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
	// size = 368u;
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

	setActiveLineStyle(lineStyleInfo);
	
	beginMainObject(MainObjectType::POLYLINE);
	drawPolyline(polyline, intendedNextSubmit);
	endMainObject();
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

void DrawResourcesFiller::drawTriangleMesh(const CTriangleMesh& mesh, CTriangleMesh::DrawData& drawData, const DTMSettingsInfo& dtmSettingsInfo, SIntendedSubmitInfo& intendedNextSubmit)
{
	setActiveDTMSettings(dtmSettingsInfo);
	uint32_t mainObjectIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
	drawData.pushConstants.triangleMeshMainObjectIndex = mainObjectIdx;

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
	if (color.a == 0.0f) // not visible
		return;

	uint32_t textureIdx = InvalidTextureIdx;
	if (fillPattern != HatchFillPattern::SOLID_FILL)
	{
		MSDFInputInfo msdfInfo = MSDFInputInfo(fillPattern);
		textureIdx = getMSDFIndexFromInputInfo(msdfInfo, intendedNextSubmit);
		if (textureIdx == InvalidTextureIdx)
			textureIdx = addMSDFTexture(msdfInfo, getHatchFillPatternMSDF(fillPattern), intendedNextSubmit);
		_NBL_DEBUG_BREAK_IF(textureIdx == InvalidTextureIdx); // probably getHatchFillPatternMSDF returned nullptr
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
	uint32_t textureIdx = InvalidTextureIdx;
	const MSDFInputInfo msdfInput = MSDFInputInfo(fontFace->getHash(), glyphIdx);
	textureIdx = getMSDFIndexFromInputInfo(msdfInput, intendedNextSubmit);
	if (textureIdx == InvalidTextureIdx)
		textureIdx = addMSDFTexture(msdfInput, getGlyphMSDF(fontFace, glyphIdx), intendedNextSubmit);

	uint32_t mainObjIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
	assert(mainObjIdx != InvalidMainObjectIdx);

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

void DrawResourcesFiller::_test_addImageObject(float64_t2 topLeftPos, float32_t2 size, float32_t rotation, SIntendedSubmitInfo& intendedNextSubmit)
{
	auto addImageObject_Internal = [&](const ImageObjectInfo& imageObjectInfo, uint32_t mainObjIdx) -> bool
		{
			const size_t remainingResourcesSize = calculateRemainingResourcesSize();
			
			const uint32_t uploadableObjects = (remainingResourcesSize) / (sizeof(ImageObjectInfo) + sizeof(DrawObject) + sizeof(uint32_t) * 6u);
			// TODO[ERFAN]: later take into account, our limit of max index buffer and vettex buffer size or constrainst other than mem
	
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
			drawObj.type_subsectionIdx = uint32_t(static_cast<uint16_t>(ObjectType::IMAGE) | (0 << 16)); // TODO: use custom pack/unpack function
			drawObj.geometryAddress = geometryBufferOffset;
			drawObjectsToBeFilled[0u] = drawObj;

			return true;
		};

	beginMainObject(MainObjectType::IMAGE);

	uint32_t mainObjIdx = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);

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

	endMainObject();
}

bool DrawResourcesFiller::finalizeAllCopiesToGPU(SIntendedSubmitInfo& intendedNextSubmit)
{
	bool success = true;
	success &= finalizeBufferCopies(intendedNextSubmit);
	success &= finalizeTextureCopies(intendedNextSubmit);
	return success;
}

void DrawResourcesFiller::setActiveLineStyle(const LineStyleInfo& lineStyle)
{
	activeLineStyle = lineStyle;
	activeLineStyleIndex = InvalidStyleIdx;
}

void DrawResourcesFiller::setActiveDTMSettings(const DTMSettingsInfo& dtmSettings)
{
	activeDTMSettings = dtmSettings;
	activeDTMSettingsIndex = InvalidDTMSettingsIdx;
}

void DrawResourcesFiller::beginMainObject(MainObjectType type)
{
	activeMainObjectType = type;
	activeMainObjectIndex = InvalidMainObjectIdx;
}

void DrawResourcesFiller::endMainObject()
{
	activeMainObjectType = MainObjectType::NONE;
	activeMainObjectIndex = InvalidMainObjectIdx;
}

void DrawResourcesFiller::pushClipProjectionData(const ClipProjectionData& clipProjectionData)
{
	activeClipProjections.push_back(clipProjectionData);
	activeClipProjectionIndices.push_back(InvalidClipProjectionIndex);
}

void DrawResourcesFiller::popClipProjectionData()
{
	if (activeClipProjections.empty())
		return;

	activeClipProjections.pop_back();
	activeClipProjectionIndices.pop_back();
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

void DrawResourcesFiller::submitCurrentDrawObjectsAndReset(SIntendedSubmitInfo& intendedNextSubmit, uint32_t& mainObjectIndex)
{
	finalizeAllCopiesToGPU(intendedNextSubmit);
	submitDraws(intendedNextSubmit);
	reset(); // resets everything, things referenced through mainObj and other shit will be pushed again through acquireXXX_SubmitIfNeeded
	mainObjectIndex = acquireActiveMainObjectIndex_SubmitIfNeeded(intendedNextSubmit);
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

	return resourcesCollection.lineStyles.addAndGetOffset(gpuLineStyle); // this will implicitly increase total resource consumption and reduce remaining size --> no need for mem size trackers
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
	
	resourcesCollection.dtmSettings.addAndGetOffset(dtmSettings); // this will implicitly increase total resource consumption and reduce remaining size --> no need for mem size trackers
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

uint32_t DrawResourcesFiller::acquireActiveClipProjectionIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (activeClipProjectionIndices.empty())
		return InvalidClipProjectionIndex;

	if (activeClipProjectionIndices.back() == InvalidClipProjectionIndex)
		activeClipProjectionIndices.back() = addClipProjectionData_SubmitIfNeeded(activeClipProjections.back(), intendedNextSubmit);
	
	return activeClipProjectionIndices.back();
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
	const bool needsDTMSettings = (activeMainObjectType == MainObjectType::DTM);
	const bool needsCustomClipProjection = (!activeClipProjectionIndices.empty());

	const size_t remainingResourcesSize = calculateRemainingResourcesSize();
	// making sure MainObject and everything it references fits into remaining resources mem
	size_t memRequired = sizeof(MainObject);
	if (needsLineStyle) memRequired += sizeof(LineStyle);
	if (needsDTMSettings) memRequired += sizeof(DTMSettings);
	if (needsCustomClipProjection) memRequired += sizeof(ClipProjectionData);

	const bool enoughMem = remainingResourcesSize >= memRequired; // enough remaining memory for 1 more dtm settings with 2 referenced line styles?
	const bool needToOverflowSubmit = (!enoughMem) || (resourcesCollection.mainObjects.vector.size() >= MaxIndexableMainObjects);
	
	if (needToOverflowSubmit)
	{
		// failed to fit into remaining resources mem or exceeded max indexable mainobj
		finalizeAllCopiesToGPU(intendedNextSubmit);
		submitDraws(intendedNextSubmit);
		reset(); // resets everything! be careful!
	}
	
	MainObject mainObject = {};
	// These 3 calls below shouldn't need to Submit because we made sure there is enough memory for all of them.
	// if something here triggers a auto-submit it's a possible bug, TODO: assert that somehow?
	mainObject.styleIdx = (needsLineStyle) ? acquireActiveLineStyleIndex_SubmitIfNeeded(intendedNextSubmit) : InvalidStyleIdx;
	mainObject.dtmSettingsIdx = (needsDTMSettings) ? acquireActiveDTMSettingsIndex_SubmitIfNeeded(intendedNextSubmit) : InvalidDTMSettingsIdx;
	mainObject.clipProjectionIndex = (needsCustomClipProjection) ? acquireActiveClipProjectionIndex_SubmitIfNeeded(intendedNextSubmit) : InvalidClipProjectionIndex;
	activeMainObjectIndex = resourcesCollection.mainObjects.addAndGetOffset(mainObject);
	return activeMainObjectIndex;
}

uint32_t DrawResourcesFiller::addLineStyle_SubmitIfNeeded(const LineStyleInfo& lineStyle, SIntendedSubmitInfo& intendedNextSubmit)
{
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
		// resets everything! be careful!
		reset();

		outDTMSettingIdx = addDTMSettings_Internal(dtmSettings, intendedNextSubmit);
		assert(outDTMSettingIdx != InvalidDTMSettingsIdx);
	}
	return outDTMSettingIdx;
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
		// resets everything! be careful!
		reset();
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

void DrawResourcesFiller::addPolylineConnectors_Internal(const CPolylineBase& polyline, uint32_t& currentPolylineConnectorObj, uint32_t mainObjIdx)
{
	const size_t remainingResourcesSize = calculateRemainingResourcesSize();

	const uint32_t uploadableObjects = (remainingResourcesSize) / (sizeof(PolylineConnector) + sizeof(DrawObject) + sizeof(uint32_t) * 6u);
	// TODO[ERFAN]: later take into account, our limit of max index buffer and vettex buffer size or constrainst other than mem
	
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
	// TODO[ERFAN]: later take into account, our limit of max index buffer and vettex buffer size or constrainst other than mem
	
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
	// TODO[ERFAN]: later take into account, our limit of max index buffer and vettex buffer size or constrainst other than mem
	
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

void DrawResourcesFiller::setGlyphMSDFTextureFunction(const GetGlyphMSDFTextureFunc& func)
{
	getGlyphMSDF = func;
}

void DrawResourcesFiller::setHatchFillMSDFTextureFunction(const GetHatchFillPatternMSDFTextureFunc& func)
{
	getHatchFillPatternMSDF = func;
}

uint32_t DrawResourcesFiller::addMSDFTexture(const MSDFInputInfo& msdfInput, core::smart_refctd_ptr<ICPUImage>&& cpuImage, SIntendedSubmitInfo& intendedNextSubmit)
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
			finalizeAllCopiesToGPU(intendedNextSubmit);
			submitDraws(intendedNextSubmit);
			reset(); // resets everything, things referenced through mainObj and other shit will be pushed again through acquireXXX_SubmitIfNeeded
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