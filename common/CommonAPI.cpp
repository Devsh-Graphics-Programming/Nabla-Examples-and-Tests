
#include "CommonAPI.h"

std::vector<CommonAPI::GPUInfo> CommonAPI::extractGPUInfos(
	nbl::core::SRange<nbl::video::IPhysicalDevice* const> gpus,
	nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface,
	const bool headlessCompute = false)
{
	using namespace nbl;
	using namespace nbl::video;

	std::vector<GPUInfo> extractedInfos = std::vector<GPUInfo>(gpus.size());

	for (size_t i = 0ull; i < gpus.size(); ++i)
	{
		auto& extractedInfo = extractedInfos[i];
		extractedInfo = {};
		auto gpu = gpus.begin()[i];

		// Find queue family indices
		{
			const auto& queueFamilyProperties = gpu->getQueueFamilyProperties();

			std::vector<uint32_t> remainingQueueCounts = std::vector<uint32_t>(queueFamilyProperties.size(), 0u);

			for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
			{
				const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];
				remainingQueueCounts[familyIndex] = familyProperty.queueCount;
			}

			// Select Graphics Queue Family Index
			if (!headlessCompute)
			{
				// Select Graphics Queue Family Index
				for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
				{
					const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];
					auto& outFamilyProp = extractedInfo.queueFamilyProps;

					const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
					if (currentFamilyQueueCount <= 0)
						continue;

					bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(gpu, familyIndex);
					bool hasGraphicsFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_GRAPHICS_BIT).value != 0;
					bool hasComputeFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_COMPUTE_BIT).value != 0;
					bool hasTransferFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_TRANSFER_BIT).value != 0;
					bool hasSparseBindingFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_SPARSE_BINDING_BIT).value != 0;
					bool hasProtectedFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_PROTECTED_BIT).value != 0;

					const uint32_t remainingQueueCount = remainingQueueCounts[familyIndex];
					const bool hasEnoughQueues = remainingQueueCount >= 1u;

					/*
					* Examples:
					*	-> score is 0 for every queueFam with no Graphics support
					*	-> If both queue families !hasEnoughtQueues -> score will be equal but this doesn't/shouldn't happen -> there should be a queueFamily with "enoughQueues" for graphics.
					*	-> if both queue families hasEnoughQueues and have similar support for present and compute: Queue Family with more remainingQueueCount is preferred.
					*	-> if both queue families hasEnoughQueues with the same number of remainingQueueCount -> "QueueFamily with present and no compute" >>>> "QueueFamily with compute and no present"
					*	-> if both queue families hasEnoughQueues -> "QueueFamily with compute and no present and 16 remainingQueues" ==== "QueueFamily with present and no compute and 1 remaining Queue"
					*	-> if both queue families hasEnoughQueues -> "QueueFamily with present and compute and 1 remaining Queue" ==== "QueueFamily with no compute and no present and 34 remaining Queues xD"
					*/
					uint32_t score = 0u;
					if (hasGraphicsFlag) {
						score++;
						if (hasEnoughQueues) {
							score += 1u * remainingQueueCount;

							if (supportsPresent)
							{
								score += 32u; // more important to have present than compute (presentSupport is larger in scoring to 16 extra compute queues)
							}

							if (hasComputeFlag)
							{
								score += 1u * remainingQueueCount;
							}
						}
					}

					if (score > outFamilyProp.graphics.score)
					{
						outFamilyProp.graphics.index = familyIndex;
						outFamilyProp.graphics.supportsGraphics = hasGraphicsFlag;
						outFamilyProp.graphics.supportsCompute = hasComputeFlag;
						outFamilyProp.graphics.supportsTransfer = true; // Reporting this is optional for Vk Graphics-Capable QueueFam, but Its support is guaranteed.
						outFamilyProp.graphics.supportsSparseBinding = hasSparseBindingFlag;
						outFamilyProp.graphics.supportsPresent = supportsPresent;
						outFamilyProp.graphics.supportsProtected = hasProtectedFlag;
						outFamilyProp.graphics.dedicatedQueueCount = 1u;
						outFamilyProp.graphics.score = score;
					}
				}
				assert(extractedInfo.queueFamilyProps.graphics.index != QueueFamilyProps::InvalidIndex);
				remainingQueueCounts[extractedInfo.queueFamilyProps.graphics.index] -= extractedInfo.queueFamilyProps.graphics.dedicatedQueueCount;
			}

			// Select Compute Queue Family Index
			for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
			{
				const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];
				auto& outFamilyProp = extractedInfo.queueFamilyProps;

				const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
				if (currentFamilyQueueCount <= 0)
					continue;

				bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(gpu, familyIndex);
				bool hasGraphicsFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_GRAPHICS_BIT).value != 0;
				bool hasComputeFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_COMPUTE_BIT).value != 0;
				bool hasTransferFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_TRANSFER_BIT).value != 0;
				bool hasSparseBindingFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_SPARSE_BINDING_BIT).value != 0;
				bool hasProtectedFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_PROTECTED_BIT).value != 0;

				const uint32_t remainingQueueCount = remainingQueueCounts[familyIndex];
				const bool hasExtraQueues = remainingQueueCount >= 1u;

				/*
				* Examples:
				*	-> If both !hasEnoughExtraQueues: "queue family that supports graphics" >>>> "queue family that doesn't support graphics"
				*	-> If both queueFams supports Graphics and hasEnoughExtraQueues: "Graphics-capable QueueFamily equal to the selected Graphics QueueFam" >>>> "Any other Graphics-capable QueueFamily"
				*	-> If both support Graphics (not equal to graphicsQueueFamIndex): "queue family that hasEnoughExtraQueues" >>>> "queue family that !hasEnoughExtraQueues"
				*	-> If both support Graphics and hasEnoughExtraQueues (not equal to graphicsQueueFamIndex):  both are adequate enough, depends on the order of the queueFams.
				*	-> "Compute-capable QueueFam with hasEnoughExtraQueues" >>>> "Compute-capable QueueFam with graphics capability and ==graphicsQueueFamIdx with no extra dedicated queues"
				*/
				uint32_t score = 0u;
				if (hasComputeFlag) {
					score++;

					if (hasExtraQueues) {
						score += 3;
					}

					if (!headlessCompute && hasGraphicsFlag) {
						score++;
						if (familyIndex == outFamilyProp.graphics.index) {
							score++;
						}
					}
				}

				if (score > outFamilyProp.compute.score)
				{
					outFamilyProp.compute.index = familyIndex;
					outFamilyProp.compute.supportsGraphics = hasGraphicsFlag;
					outFamilyProp.compute.supportsCompute = hasComputeFlag;
					outFamilyProp.compute.supportsTransfer = true; // Reporting this is optional for Vk Compute-Capable QueueFam, but Its support is guaranteed.
					outFamilyProp.compute.supportsSparseBinding = hasSparseBindingFlag;
					outFamilyProp.compute.supportsPresent = supportsPresent;
					outFamilyProp.compute.supportsProtected = hasProtectedFlag;
					outFamilyProp.compute.dedicatedQueueCount = (hasExtraQueues) ? 1u : 0u;
					outFamilyProp.compute.score = score;
				}
			}
			assert(extractedInfo.queueFamilyProps.compute.index != QueueFamilyProps::InvalidIndex);
			remainingQueueCounts[extractedInfo.queueFamilyProps.compute.index] -= extractedInfo.queueFamilyProps.compute.dedicatedQueueCount;

			// Select Transfer Queue Family Index
			for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
			{
				const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];
				auto& outFamilyProp = extractedInfo.queueFamilyProps;

				const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
				if (currentFamilyQueueCount <= 0)
					continue;

				bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(gpu, familyIndex);
				bool hasGraphicsFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_GRAPHICS_BIT).value != 0;
				bool hasComputeFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_COMPUTE_BIT).value != 0;
				bool hasTransferFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_TRANSFER_BIT).value != 0;
				bool hasSparseBindingFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_SPARSE_BINDING_BIT).value != 0;
				bool hasProtectedFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_PROTECTED_BIT).value != 0;

				const uint32_t extraQueueCount = nbl::core::min(remainingQueueCounts[familyIndex], 2u); // UP + DOWN
				const bool hasExtraQueues = extraQueueCount >= 1u;

				/*
				* Examples:
				*	-> score is 0 for every queueFam with no Transfer support
				*	-> If both have similar hasEnoughExtraQueues, compute and graphics support: the one with more remainingQueueCount is preferred
				*	-> If both support Transfer: "QueueFam with >=1 extra queues and graphics and compute support" >>>> (less probable)"QueueFam with no extra queues and transfer-only(no compute and graphics support)"
				*	-> If both support Transfer: "QueueFam with >=0 extra queues and only compute" >>>> "QueueFam with >=0 extra queues and only graphics"
				*/
				uint32_t score = 0u;
				if (hasTransferFlag) {
					score += 1u;

					uint32_t notHavingComputeScore = 1u;
					uint32_t notHavingGraphicsScore = 2u;

					if (hasExtraQueues) { // Having extra queues to have seperate up/down transfer queues is more important
						score += 4u * extraQueueCount;
						notHavingComputeScore *= extraQueueCount;
						notHavingGraphicsScore *= extraQueueCount;
					}

					if (!hasGraphicsFlag) {
						score += notHavingGraphicsScore;
					}

					if (!hasComputeFlag) {
						score += notHavingComputeScore;
					}

				}

				if (score > outFamilyProp.transfer.score)
				{
					outFamilyProp.transfer.index = familyIndex;
					outFamilyProp.transfer.supportsGraphics = hasGraphicsFlag;
					outFamilyProp.transfer.supportsCompute = hasComputeFlag;
					outFamilyProp.transfer.supportsTransfer = hasTransferFlag;
					outFamilyProp.transfer.supportsSparseBinding = hasSparseBindingFlag;
					outFamilyProp.transfer.supportsPresent = supportsPresent;
					outFamilyProp.transfer.supportsProtected = hasProtectedFlag;
					outFamilyProp.transfer.dedicatedQueueCount = extraQueueCount;
					outFamilyProp.transfer.score = score;
				}
			}
			assert(extractedInfo.queueFamilyProps.transfer.index != QueueFamilyProps::InvalidIndex);
			remainingQueueCounts[extractedInfo.queueFamilyProps.transfer.index] -= extractedInfo.queueFamilyProps.transfer.dedicatedQueueCount;

			// Select Present Queue Family Index
			if (!headlessCompute)
			{
				if (extractedInfo.queueFamilyProps.graphics.supportsPresent && extractedInfo.queueFamilyProps.graphics.index != QueueFamilyProps::InvalidIndex)
				{
					extractedInfo.queueFamilyProps.present = extractedInfo.queueFamilyProps.graphics;
					extractedInfo.queueFamilyProps.present.dedicatedQueueCount = 0u;
				}
				else
				{
					const uint32_t maxNeededQueueCountForPresent = 1u;
					for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
					{
						const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];
						auto& outFamilyProp = extractedInfo.queueFamilyProps;

						const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
						if (currentFamilyQueueCount <= 0)
							continue;

						bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(gpu, familyIndex);
						bool hasGraphicsFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_GRAPHICS_BIT).value != 0;
						bool hasComputeFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_COMPUTE_BIT).value != 0;
						bool hasTransferFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_TRANSFER_BIT).value != 0;
						bool hasSparseBindingFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_SPARSE_BINDING_BIT).value != 0;
						bool hasProtectedFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_PROTECTED_BIT).value != 0;

						const uint32_t remainingQueueCount = remainingQueueCounts[familyIndex];
						const bool hasEnoughExtraQueues = remainingQueueCount >= 1u;

						/* this will only lead here if selected graphics queue can't support present
						* Examples:
						*	-> score is 0 for every queueFam with no Present support
						*	-> If both queue families support Present -> "graphics support is preferred rather than extra dedicated queues"
						*		-> graphics support is equal in scoring to 100 extra queues with no graphics support
						*	-> If both queue families !hasEnoughExtraQueues -> "graphics support is preferred"
						*	-> If both queue families hasEnoughExtraQueues and have similar support for graphics -> "queue family with more remainingQueueCount is preferred"
						*/
						uint32_t score = 0u;
						if (supportsPresent) {
							score += 1u;

							uint32_t graphicsSupportScore = 100u;
							if (hasEnoughExtraQueues) {
								score += 1u * remainingQueueCount;
								graphicsSupportScore *= remainingQueueCount;
							}

							if (hasGraphicsFlag) {
								score += graphicsSupportScore; // graphics support is larger in scoring than 100 extra queues with no graphics support
							}
						}

						if (score > outFamilyProp.present.score)
						{
							outFamilyProp.present.index = familyIndex;
							outFamilyProp.present.supportsGraphics = hasGraphicsFlag;
							outFamilyProp.present.supportsCompute = hasComputeFlag;
							outFamilyProp.present.supportsTransfer = hasTransferFlag;
							outFamilyProp.present.supportsSparseBinding = hasSparseBindingFlag;
							outFamilyProp.present.supportsPresent = supportsPresent;
							outFamilyProp.present.supportsProtected = hasProtectedFlag;
							outFamilyProp.present.dedicatedQueueCount = (hasEnoughExtraQueues) ? 1u : 0u;
							outFamilyProp.present.score = score;
						}
					}
				}
				assert(extractedInfo.queueFamilyProps.present.index != QueueFamilyProps::InvalidIndex);
				remainingQueueCounts[extractedInfo.queueFamilyProps.present.index] -= extractedInfo.queueFamilyProps.present.dedicatedQueueCount;
			}

			if (!headlessCompute)
				assert(extractedInfo.queueFamilyProps.graphics.supportsTransfer && "This shouldn't happen");
			assert(extractedInfo.queueFamilyProps.compute.supportsTransfer && "This shouldn't happen");
		}

		extractedInfo.isSwapChainSupported = gpu->isSwapchainSupported();

		// Check if the surface is adequate
		if (surface)
		{
			uint32_t surfaceFormatCount;
			surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, nullptr);
			extractedInfo.availableSurfaceFormats = std::vector<nbl::video::ISurface::SFormat>(surfaceFormatCount);
			surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, extractedInfo.availableSurfaceFormats.data());

			extractedInfo.availablePresentModes = surface->getAvailablePresentModesForPhysicalDevice(gpu);

			// TODO: @achal OpenGL shouldn't fail this
			extractedInfo.surfaceCapabilities = {};
			if (surface->getSurfaceCapabilitiesForPhysicalDevice(gpu, extractedInfo.surfaceCapabilities))
				extractedInfo.hasSurfaceCapabilities = true;
		}
	}

	return extractedInfos;
}

uint32_t CommonAPI::findSuitableGPU(const std::vector<GPUInfo>& extractedInfos, const bool headlessCompute)
{
	uint32_t ret = ~0u;
	for (uint32_t i = 0; i < extractedInfos.size(); ++i)
	{
		bool isGPUSuitable = false;
		const auto& extractedInfo = extractedInfos[i];

		if (!headlessCompute)
		{
			if ((extractedInfo.queueFamilyProps.graphics.index != QueueFamilyProps::InvalidIndex) &&
				(extractedInfo.queueFamilyProps.compute.index != QueueFamilyProps::InvalidIndex) &&
				(extractedInfo.queueFamilyProps.transfer.index != QueueFamilyProps::InvalidIndex) &&
				(extractedInfo.queueFamilyProps.present.index != QueueFamilyProps::InvalidIndex))
				isGPUSuitable = true;

			if (extractedInfo.isSwapChainSupported == false)
				isGPUSuitable = false;

			if (extractedInfo.hasSurfaceCapabilities == false)
				isGPUSuitable = false;
		}
		else
		{
			if ((extractedInfo.queueFamilyProps.compute.index != QueueFamilyProps::InvalidIndex) &&
				(extractedInfo.queueFamilyProps.transfer.index != QueueFamilyProps::InvalidIndex))
				isGPUSuitable = true;
		}

		if (isGPUSuitable)
		{
			// find the first suitable GPU
			ret = i;
			break;
		}
	}

	if (ret == ~0u)
	{
		//_NBL_DEBUG_BREAK_IF(true);
		ret = 0;
	}

	return ret;
}

nbl::video::ISwapchain::SCreationParams CommonAPI::computeSwapchainCreationParams(
	const GPUInfo& gpuInfo, uint32_t& imageCount,
	const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
	const nbl::core::smart_refctd_ptr<nbl::video::ISurface>& surface,
	nbl::asset::IImage::E_USAGE_FLAGS imageUsage,
	// Acceptable settings, ordered by preference.
	const nbl::asset::E_FORMAT* acceptableSurfaceFormats, uint32_t acceptableSurfaceFormatCount,
	const nbl::asset::E_COLOR_PRIMARIES* acceptableColorPrimaries, uint32_t acceptableColorPrimaryCount,
	const nbl::asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION* acceptableEotfs, uint32_t acceptableEotfCount,
	const nbl::video::ISurface::E_PRESENT_MODE* acceptablePresentModes, uint32_t acceptablePresentModeCount,
	const nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS* acceptableSurfaceTransforms, uint32_t acceptableSurfaceTransformsCount
)
{
	using namespace nbl;

	asset::E_SHARING_MODE imageSharingMode;
	if (gpuInfo.queueFamilyProps.graphics.index == gpuInfo.queueFamilyProps.present.index)
		imageSharingMode = asset::ESM_EXCLUSIVE;
	else
		imageSharingMode = asset::ESM_CONCURRENT;

	nbl::video::ISurface::SFormat surfaceFormat;
	nbl::video::ISurface::E_PRESENT_MODE presentMode = nbl::video::ISurface::EPM_UNKNOWN;
	nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS surfaceTransform = nbl::video::ISurface::EST_FLAG_BITS_MAX_ENUM;

	if (device->getAPIType() == nbl::video::EAT_VULKAN)
	{
		// Deduce format features from imageUsage param
		nbl::video::IPhysicalDevice::SFormatImageUsage requiredFormatUsages = {};
		if (imageUsage & asset::IImage::EUF_STORAGE_BIT)
			requiredFormatUsages.storageImage = 1;

		nbl::video::ISurface::SCapabilities capabilities;
		surface->getSurfaceCapabilitiesForPhysicalDevice(device->getPhysicalDevice(), capabilities);

		for (uint32_t i = 0; i < acceptableSurfaceFormatCount; i++)
		{
			auto testSurfaceTransform = acceptableSurfaceTransforms[i];
			if (capabilities.currentTransform == testSurfaceTransform)
			{
				surfaceTransform = testSurfaceTransform;
				break;
			}
		}
		assert(surfaceTransform != nbl::video::ISurface::EST_FLAG_BITS_MAX_ENUM); // currentTransform must be supported in acceptableSurfaceTransforms

		auto availablePresentModes = surface->getAvailablePresentModesForPhysicalDevice(device->getPhysicalDevice());
		for (uint32_t i = 0; i < acceptablePresentModeCount; i++)
		{
			auto testPresentMode = acceptablePresentModes[i];
			if ((availablePresentModes & testPresentMode) == testPresentMode)
			{
				presentMode = testPresentMode;
				break;
			}
		}
		assert(presentMode != nbl::video::ISurface::EST_FLAG_BITS_MAX_ENUM);

		constexpr uint32_t MAX_SURFACE_FORMAT_COUNT = 1000u;
		uint32_t availableFormatCount;
		nbl::video::ISurface::SFormat availableFormats[MAX_SURFACE_FORMAT_COUNT];
		surface->getAvailableFormatsForPhysicalDevice(device->getPhysicalDevice(), availableFormatCount, availableFormats);

		for (uint32_t i = 0; i < availableFormatCount; ++i)
		{
			// TODO verify if acceptableSurfaceFormats, acceptableColorPrimaries & acceptableEotfs
			// allow for supportedFormat
			const auto& supportedFormat = availableFormats[i];
			if (true)
			{
				surfaceFormat = supportedFormat;
				break;
			}
		}
		// Require at least one of the acceptable options to be present
		assert(surfaceFormat.format != nbl::asset::EF_UNKNOWN &&
			surfaceFormat.colorSpace.primary != nbl::asset::ECP_COUNT &&
			surfaceFormat.colorSpace.eotf != nbl::asset::EOTF_UNKNOWN);
	}
	else
	{
		// Temporary path until OpenGL reports properly!
		surfaceFormat = nbl::video::ISurface::SFormat(acceptableSurfaceFormats[0], acceptableColorPrimaries[0], acceptableEotfs[0]);
		presentMode = nbl::video::ISurface::EPM_IMMEDIATE;
		surfaceTransform = nbl::video::ISurface::EST_HORIZONTAL_MIRROR_ROTATE_180_BIT;
	}

	nbl::video::ISwapchain::SCreationParams sc_params = {};
	sc_params.arrayLayers = 1u;
	sc_params.minImageCount = imageCount;
	sc_params.presentMode = presentMode;
	sc_params.imageUsage = imageUsage;
	sc_params.surface = surface;
	sc_params.imageSharingMode = imageSharingMode;
	sc_params.preTransform = surfaceTransform;
	sc_params.compositeAlpha = nbl::video::ISurface::ECA_OPAQUE_BIT;
	sc_params.surfaceFormat = surfaceFormat;

	return sc_params;
}

nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> CommonAPI::createRenderpass(const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device, nbl::asset::E_FORMAT colorAttachmentFormat, nbl::asset::E_FORMAT baseDepthFormat)
{
	using namespace nbl;

	bool useDepth = baseDepthFormat != nbl::asset::EF_UNKNOWN;
	nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN;
	if (useDepth)
	{
		depthFormat = baseDepthFormat;
		//depthFormat = device->getPhysicalDevice()->promoteImageFormat(
		//	{ baseDepthFormat, nbl::video::IPhysicalDevice::SFormatImageUsage(nbl::asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT) },
		//	nbl::asset::IImage::ET_OPTIMAL
		//);
		// TODO error reporting
		assert(depthFormat != nbl::asset::EF_UNKNOWN);
	}

	nbl::video::IGPURenderpass::SCreationParams::SAttachmentDescription attachments[2];
	attachments[0].initialLayout = asset::EIL_UNDEFINED;
	attachments[0].finalLayout = asset::EIL_PRESENT_SRC;
	attachments[0].format = colorAttachmentFormat;
	attachments[0].samples = asset::IImage::ESCF_1_BIT;
	attachments[0].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
	attachments[0].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

	attachments[1].initialLayout = asset::EIL_UNDEFINED;
	attachments[1].finalLayout = asset::EIL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachments[1].format = depthFormat;
	attachments[1].samples = asset::IImage::ESCF_1_BIT;
	attachments[1].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
	attachments[1].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

	nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
	colorAttRef.attachment = 0u;
	colorAttRef.layout = asset::EIL_COLOR_ATTACHMENT_OPTIMAL;

	nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef depthStencilAttRef;
	depthStencilAttRef.attachment = 1u;
	depthStencilAttRef.layout = asset::EIL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

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

	return device->createRenderpass(rp_params);
}

auto createFBOWithSwapchainImages(
	size_t imageCount, uint32_t width, uint32_t height,
	const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
	nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain,
	nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass,
	nbl::asset::E_FORMAT baseDepthFormat = nbl::asset::EF_UNKNOWN
) -> nbl::core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>>
{
	using namespace nbl;

	bool useDepth = baseDepthFormat != nbl::asset::EF_UNKNOWN;
	nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN;
	if (useDepth)
	{
		depthFormat = baseDepthFormat;
		//depthFormat = device->getPhysicalDevice()->promoteImageFormat(
		//	{ baseDepthFormat, nbl::video::IPhysicalDevice::SFormatImageUsage(nbl::asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT) },
		//	nbl::asset::IImage::ET_OPTIMAL
		//);
		// TODO error reporting
		assert(depthFormat != nbl::asset::EF_UNKNOWN);
	}

	auto sc_images = swapchain->getImages();
	assert(sc_images.size() == imageCount);
	auto fbo = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>>>(imageCount);
	for (uint32_t i = 0u; i < imageCount; ++i)
	{
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> view[2] = {};

		auto img = sc_images.begin()[i];
		{
			nbl::video::IGPUImageView::SCreationParams view_params;
			view_params.format = img->getCreationParameters().format;
			view_params.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
			view_params.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			view_params.subresourceRange.baseMipLevel = 0u;
			view_params.subresourceRange.levelCount = 1u;
			view_params.subresourceRange.baseArrayLayer = 0u;
			view_params.subresourceRange.layerCount = 1u;
			view_params.image = std::move(img);

			view[0] = device->createImageView(std::move(view_params));
			assert(view[0]);
		}

		if (useDepth) {
			nbl::video::IGPUImage::SCreationParams imgParams;
			imgParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
			imgParams.type = asset::IImage::ET_2D;
			imgParams.format = depthFormat;
			imgParams.extent = { width, height, 1 };
			imgParams.usage = asset::IImage::E_USAGE_FLAGS::EUF_DEPTH_STENCIL_ATTACHMENT_BIT;
			imgParams.mipLevels = 1u;
			imgParams.arrayLayers = 1u;
			imgParams.samples = asset::IImage::ESCF_1_BIT;

			auto depthImg = device->createImage(std::move(imgParams));
			auto depthImgMemReqs = depthImg->getMemoryReqs();
			depthImgMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto depthImgMem = device->allocate(depthImgMemReqs, depthImg.get());

			nbl::video::IGPUImageView::SCreationParams view_params;
			view_params.format = depthFormat;
			view_params.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
			view_params.subresourceRange.aspectMask = asset::IImage::EAF_DEPTH_BIT;
			view_params.subresourceRange.baseMipLevel = 0u;
			view_params.subresourceRange.levelCount = 1u;
			view_params.subresourceRange.baseArrayLayer = 0u;
			view_params.subresourceRange.layerCount = 1u;
			view_params.image = std::move(depthImg);

			view[1] = device->createImageView(std::move(view_params));
			assert(view[1]);
		}

		nbl::video::IGPUFramebuffer::SCreationParams fb_params;
		fb_params.width = width;
		fb_params.height = height;
		fb_params.layers = 1u;
		fb_params.renderpass = renderpass;
		fb_params.flags = static_cast<nbl::video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
		fb_params.attachmentCount = (useDepth) ? 2u : 1u;
		fb_params.attachments = view;

		fbo->begin()[i] = device->createFramebuffer(std::move(fb_params));
		assert(fbo->begin()[i]);
	}
	return fbo;
}
