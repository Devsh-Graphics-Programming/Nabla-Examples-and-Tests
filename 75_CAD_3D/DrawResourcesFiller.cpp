#include "DrawResourcesFiller.h"

using namespace nbl;

DrawResourcesFiller::DrawResourcesFiller()
{}

DrawResourcesFiller::DrawResourcesFiller(smart_refctd_ptr<video::ILogicalDevice>&& device, smart_refctd_ptr<IUtilities>&& bufferUploadUtils, IQueue* copyQueue, core::smart_refctd_ptr<system::ILogger>&& logger) :
	m_device(std::move(device)),
	m_bufferUploadUtils(std::move(bufferUploadUtils)),
	m_copyQueue(copyQueue),
	m_logger(std::move(logger))
{
}

// function is called when buffer is filled and we should submit draws and clear the buffers and continue filling
void DrawResourcesFiller::setSubmitDrawsFunction(const SubmitFunc& func)
{
	submitDraws = func;
}

// TODO: redo it completely
bool DrawResourcesFiller::allocateDrawResources(ILogicalDevice* logicalDevice, size_t requiredImageMemorySize, size_t requiredBufferMemorySize, std::span<uint32_t> memoryTypeIndexTryOrder)
{
	const size_t adjustedBuffersMemorySize = requiredBufferMemorySize;
	const size_t totalResourcesSize = adjustedBuffersMemorySize;

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

	buffersMemoryArena = {
		.memory = allocation.memory,
		.offset = core::alignUp(allocation.offset, GPUStructsMaxNaturalAlignment), // first natural alignment after images section of the memory allocation
	};

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
	const size_t minimumAcceptableSize = MinimumDrawResourcesMemorySize;

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

void DrawResourcesFiller::drawTriangleMesh(
	const CTriangleMesh& mesh,
	const DTMSettingsInfo& dtmSettingsInfo,
	SIntendedSubmitInfo& intendedNextSubmit)
{
	setActiveDTMSettings(dtmSettingsInfo);
	beginMainObject(MainObjectType::DTM);

	uint32_t mainObjectIdx = acquireActiveMainObjectIndex(intendedNextSubmit);
	if (mainObjectIdx == InvalidMainObjectIdx)
	{
		m_logger.log("drawTriangleMesh: acquireActiveMainObjectIndex returned invalid index", nbl::system::ILogger::ELL_ERROR);
		assert(false);
		return;
	}

	// TODO: for now we add whole mesh at once, instead we should add triangle by triangle and see check if we overflow memory

	const size_t vertexBuffByteSize = mesh.getVertexBuffByteSize();
	const size_t indexBuffByteSize = mesh.getIndexBuffByteSize();
	const size_t triangleDataByteSize = vertexBuffByteSize + indexBuffByteSize;
	const auto& indexBuffer = mesh.getIndices();
	const auto& vertexBuffer = mesh.getVertices();
	assert(indexBuffer.size() == vertexBuffer.size()); // TODO: figure out why it was needed then decide if this constraint needs to be kept

	DrawCallData drawCallData = {};

	// Copy VertexBuffer
	size_t geometryBufferOffset = resourcesCollection.geometryInfo.increaseSizeAndGetOffset(triangleDataByteSize, alignof(CTriangleMesh::vertex_t));
	drawCallData.triangleMeshVerticesBaseAddress = geometryBufferOffset;
	void* dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
	memcpy(dst, vertexBuffer.data(), vertexBuffByteSize);
	geometryBufferOffset += vertexBuffByteSize;

	// Copy IndexBuffer
	dst = resourcesCollection.geometryInfo.data() + geometryBufferOffset;
	drawCallData.indexBufferOffset = geometryBufferOffset;
	memcpy(dst, indexBuffer.data(), indexBuffByteSize);

	drawCallData.triangleMeshMainObjectIndex = mainObjectIdx;
	drawCallData.indexCount = mesh.getIndexCount();
	drawCalls.push_back(drawCallData);

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
	success &= pushBufferUploads(intendedNextSubmit, resourcesCollection);

	return success;
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
			SBufferRange<IGPUBuffer> copyRange = { copiedResourcesSize, drawBuffer.getStorageSize(), resourcesGPUBuffer };

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

	copyCPUFilledDrawBuffer(resources.mainObjects);
	copyCPUFilledDrawBuffer(resources.dtmSettings);
	copyCPUFilledDrawBuffer(resources.drawObjects);
	copyCPUFilledDrawBuffer(resources.indexBuffer);
	copyCPUFilledDrawBuffer(resources.geometryInfo);

	return true;
}

void DrawResourcesFiller::markFrameUsageComplete(uint64_t drawSubmitWaitValue)
{
	// m_logger.log(std::format("Finished Frame Idx = {}", currentFrameIndex).c_str(), nbl::system::ILogger::ELL_INFO);
	currentFrameIndex++;
	// TODO[LATER]: take into account that currentFrameIndex was submitted with drawSubmitWaitValue; Use that value when deallocating the resources marked with this frame index
	//				Currently, for evictions the worst case value will be waited for, as there is no way yet to know which semaphoroe value will signal the completion of the (to be evicted) resource's usage
}

uint32_t DrawResourcesFiller::acquireActiveMainObjectIndex(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (activeMainObjectIndex != InvalidMainObjectIdx)
		return activeMainObjectIndex;

	if (activeMainObjectType == MainObjectType::NONE)
	{
		assert(false); // You're probably trying to acquire mainObjectIndex outside of startMainObject, endMainObject scope
		return InvalidMainObjectIdx;
	}

	const bool needsDTMSettings = activeMainObjectType == MainObjectType::DTM;

	MainObject mainObject = {};
	mainObject.dtmSettingsIdx = (needsDTMSettings) ? acquireActiveDTMSettingsIndex_SubmitIfNeeded(intendedNextSubmit) : InvalidDTMSettingsIdx;
	activeMainObjectIndex = resourcesCollection.mainObjects.addAndGetOffset(mainObject);
	return activeMainObjectIndex;
}

uint32_t DrawResourcesFiller::acquireActiveDTMSettingsIndex_SubmitIfNeeded(SIntendedSubmitInfo& intendedNextSubmit)
{
	if (activeDTMSettingsIndex == InvalidDTMSettingsIdx)
		activeDTMSettingsIndex = addDTMSettings_SubmitIfNeeded(activeDTMSettings, intendedNextSubmit);

	return activeDTMSettingsIndex;
}

uint32_t DrawResourcesFiller::addDTMSettings_SubmitIfNeeded(const DTMSettingsInfo& dtmSettings, SIntendedSubmitInfo& intendedNextSubmit)
{
	// before calling `addDTMSettings_Internal` we have made sute we have enough mem for 
	uint32_t outDTMSettingIdx = addDTMSettings_Internal(dtmSettings, intendedNextSubmit);
	return outDTMSettingIdx;
}

uint32_t DrawResourcesFiller::addDTMSettings_Internal(const DTMSettingsInfo& dtmSettingsInfo, SIntendedSubmitInfo& intendedNextSubmit)
{
	DTMSettings dtmSettings;

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

	for (uint32_t i = 0u; i < resourcesCollection.dtmSettings.vector.size(); ++i)
	{
		const DTMSettings& itr = resourcesCollection.dtmSettings.vector[i];
		if (itr == dtmSettings)
			return i;
	}

	return resourcesCollection.dtmSettings.addAndGetOffset(dtmSettings); // this will implicitly increase total resource consumption and reduce remaining size --> no need for mem size trackers
}