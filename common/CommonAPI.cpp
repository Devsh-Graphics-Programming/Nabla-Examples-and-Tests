
#include "CommonAPI.h"

nbl::video::IPhysicalDevice* const CommonAPI::CDefaultPhysicalDeviceSelector::selectPhysicalDevice(nbl::core::set<nbl::video::IPhysicalDevice* const> suitablePhysicalDevices)
{
	if(suitablePhysicalDevices.empty())
		return nullptr;
	auto firstPhysDev = *suitablePhysicalDevices.begin();
	return firstPhysDev;
}

nbl::core::set<nbl::video::IPhysicalDevice* const> getFilteredPhysicalDevices(nbl::core::SRange<nbl::video::IPhysicalDevice* const> physicalDevices, const nbl::video::SDefaultPhysicalDeviceFilter& filter)
{
	using namespace nbl;
	using namespace nbl::video;

	core::set<IPhysicalDevice* const> ret;

	for (size_t i = 0ull; i < physicalDevices.size(); ++i)
	{
		auto physDev = physicalDevices[i];
		if(filter.meetsRequirements(physDev))
			ret.insert(physDev);
	}
	return ret;
}

// Used to help with queue selection
struct QueueFamilyProps
{
	static constexpr uint32_t InvalidIndex = ~0u;
	uint32_t index					= InvalidIndex;
	uint32_t dedicatedQueueCount	= 0u;
	uint32_t score					= 0u;
	bool supportsGraphics			: 1;
	bool supportsCompute			: 1;
	bool supportsTransfer			: 1;
	bool supportsSparseBinding		: 1;
	bool supportsPresent			: 1;
	bool supportsProtected			: 1;
};

struct PhysicalDeviceQueuesInfo
{
	QueueFamilyProps graphics;
	QueueFamilyProps compute;
	QueueFamilyProps transfer;
	QueueFamilyProps present;
};

PhysicalDeviceQueuesInfo extractPhysicalDeviceQueueInfos(
	nbl::video::IPhysicalDevice* const physicalDevice,
	nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface,
	bool headlessCompute)
{
	using namespace nbl;
	using namespace nbl::video;

	PhysicalDeviceQueuesInfo queuesInfo = {};

	// Find queue family indices
	{
		const auto& queueFamilyProperties = physicalDevice->getQueueFamilyProperties();

		std::vector<uint32_t> remainingQueueCounts = std::vector<uint32_t>(queueFamilyProperties.size(), 0u);

		for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
		{
			const auto& familyProperty = queueFamilyProperties[familyIndex];
			remainingQueueCounts[familyIndex] = familyProperty.queueCount;
		}

		// Select Graphics Queue Family Index
		if (!headlessCompute)
		{
			// Select Graphics Queue Family Index
			for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
			{
				const auto& familyProperty = queueFamilyProperties[familyIndex];

				const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
				if (currentFamilyQueueCount <= 0)
					continue;

				bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(physicalDevice, familyIndex);
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

				if (score > queuesInfo.graphics.score)
				{
					queuesInfo.graphics.index = familyIndex;
					queuesInfo.graphics.supportsGraphics = hasGraphicsFlag;
					queuesInfo.graphics.supportsCompute = hasComputeFlag;
					queuesInfo.graphics.supportsTransfer = true; // Reporting this is optional for Vk Graphics-Capable QueueFam, but Its support is guaranteed.
					queuesInfo.graphics.supportsSparseBinding = hasSparseBindingFlag;
					queuesInfo.graphics.supportsPresent = supportsPresent;
					queuesInfo.graphics.supportsProtected = hasProtectedFlag;
					queuesInfo.graphics.dedicatedQueueCount = 1u;
					queuesInfo.graphics.score = score;
				}
			}
			assert(queuesInfo.graphics.index != QueueFamilyProps::InvalidIndex);
			remainingQueueCounts[queuesInfo.graphics.index] -= queuesInfo.graphics.dedicatedQueueCount;
		}

		// Select Compute Queue Family Index
		for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
		{
			const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];

			const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
			if (currentFamilyQueueCount <= 0)
				continue;

			bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(physicalDevice, familyIndex);
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
					if (familyIndex == queuesInfo.graphics.index) {
						score++;
					}
				}
			}

			if (score > queuesInfo.compute.score)
			{
				queuesInfo.compute.index = familyIndex;
				queuesInfo.compute.supportsGraphics = hasGraphicsFlag;
				queuesInfo.compute.supportsCompute = hasComputeFlag;
				queuesInfo.compute.supportsTransfer = true; // Reporting this is optional for Vk Compute-Capable QueueFam, but Its support is guaranteed.
				queuesInfo.compute.supportsSparseBinding = hasSparseBindingFlag;
				queuesInfo.compute.supportsPresent = supportsPresent;
				queuesInfo.compute.supportsProtected = hasProtectedFlag;
				queuesInfo.compute.dedicatedQueueCount = (hasExtraQueues) ? 1u : 0u;
				queuesInfo.compute.score = score;
			}
		}
		assert(queuesInfo.compute.index != QueueFamilyProps::InvalidIndex);
		remainingQueueCounts[queuesInfo.compute.index] -= queuesInfo.compute.dedicatedQueueCount;

		// Select Transfer Queue Family Index
		for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
		{
			const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];

			const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
			if (currentFamilyQueueCount <= 0)
				continue;

			bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(physicalDevice, familyIndex);
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

			if (score > queuesInfo.transfer.score)
			{
				queuesInfo.transfer.index = familyIndex;
				queuesInfo.transfer.supportsGraphics = hasGraphicsFlag;
				queuesInfo.transfer.supportsCompute = hasComputeFlag;
				queuesInfo.transfer.supportsTransfer = hasTransferFlag;
				queuesInfo.transfer.supportsSparseBinding = hasSparseBindingFlag;
				queuesInfo.transfer.supportsPresent = supportsPresent;
				queuesInfo.transfer.supportsProtected = hasProtectedFlag;
				queuesInfo.transfer.dedicatedQueueCount = extraQueueCount;
				queuesInfo.transfer.score = score;
			}
		}
		assert(queuesInfo.transfer.index != QueueFamilyProps::InvalidIndex);
		remainingQueueCounts[queuesInfo.transfer.index] -= queuesInfo.transfer.dedicatedQueueCount;

		// Select Present Queue Family Index
		if (!headlessCompute)
		{
			if (queuesInfo.graphics.supportsPresent && queuesInfo.graphics.index != QueueFamilyProps::InvalidIndex)
			{
				queuesInfo.present = queuesInfo.graphics;
				queuesInfo.present.dedicatedQueueCount = 0u;
			}
			else
			{
				const uint32_t maxNeededQueueCountForPresent = 1u;
				for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
				{
					const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];

					const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
					if (currentFamilyQueueCount <= 0)
						continue;

					bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(physicalDevice, familyIndex);
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

					if (score > queuesInfo.present.score)
					{
						queuesInfo.present.index = familyIndex;
						queuesInfo.present.supportsGraphics = hasGraphicsFlag;
						queuesInfo.present.supportsCompute = hasComputeFlag;
						queuesInfo.present.supportsTransfer = hasTransferFlag;
						queuesInfo.present.supportsSparseBinding = hasSparseBindingFlag;
						queuesInfo.present.supportsPresent = supportsPresent;
						queuesInfo.present.supportsProtected = hasProtectedFlag;
						queuesInfo.present.dedicatedQueueCount = (hasEnoughExtraQueues) ? 1u : 0u;
						queuesInfo.present.score = score;
					}
				}
			}
			assert(queuesInfo.present.index != QueueFamilyProps::InvalidIndex);
			remainingQueueCounts[queuesInfo.present.index] -= queuesInfo.present.dedicatedQueueCount;
		}

		if (!headlessCompute)
			assert(queuesInfo.graphics.supportsTransfer && "This shouldn't happen");
		assert(queuesInfo.compute.supportsTransfer && "This shouldn't happen");
	}

	return queuesInfo;
}

nbl::video::ISwapchain::SCreationParams CommonAPI::computeSwapchainCreationParams(
	uint32_t& imageCount,
	const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
	const nbl::core::smart_refctd_ptr<nbl::video::ISurface>& surface,
	nbl::asset::IImage::E_USAGE_FLAGS imageUsage,
	// Acceptable settings, ordered by preference.
	const nbl::asset::E_FORMAT* acceptableSurfaceFormats, uint32_t acceptableSurfaceFormatCount,
	const nbl::asset::E_COLOR_PRIMARIES* acceptableColorPrimaries, uint32_t acceptableColorPrimaryCount,
	const nbl::asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION* acceptableEotfs, uint32_t acceptableEotfCount,
	const nbl::video::ISurface::E_PRESENT_MODE* acceptablePresentModes, uint32_t acceptablePresentModeCount,
	const nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS* acceptableSurfaceTransforms, uint32_t acceptableSurfaceTransformCount
)
{
	using namespace nbl;

	nbl::video::ISurface::SFormat surfaceFormat;
	nbl::video::ISurface::E_PRESENT_MODE presentMode = nbl::video::ISurface::EPM_UNKNOWN;
	nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS surfaceTransform = nbl::video::ISurface::EST_FLAG_BITS_MAX_ENUM;

	if (device->getAPIType() == nbl::video::EAT_VULKAN)
	{
		nbl::video::ISurface::SCapabilities capabilities;
		surface->getSurfaceCapabilitiesForPhysicalDevice(device->getPhysicalDevice(), capabilities);

		for (uint32_t i = 0; i < acceptableSurfaceTransformCount; i++)
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
			auto testsformat = availableFormats[i];
			bool supportsFormat = false;
			bool supportsEotf = false;
			bool supportsPrimary = false;

			for (uint32_t i = 0; i < acceptableSurfaceFormatCount; i++)
			{
				if (testsformat.format == acceptableSurfaceFormats[i])
				{
					supportsFormat = true;
					break;
				}
			}
			for (uint32_t i = 0; i < acceptableEotfCount; i++)
			{
				if (testsformat.colorSpace.eotf == acceptableEotfs[i])
				{
					supportsEotf = true;
					break;
				}
			}
			for (uint32_t i = 0; i < acceptableColorPrimaryCount; i++)
			{
				if (testsformat.colorSpace.primary == acceptableColorPrimaries[i])
				{
					supportsPrimary = true;
					break;
				}
			}

			if (supportsFormat && supportsEotf && supportsPrimary)
			{
				surfaceFormat = testsformat;
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
	sc_params.preTransform = surfaceTransform;
	sc_params.compositeAlpha = nbl::video::ISurface::ECA_OPAQUE_BIT;
	sc_params.surfaceFormat = surfaceFormat;

	return sc_params;
}

void CommonAPI::dropRetiredSwapchainResources(nbl::core::deque<IRetiredSwapchainResources*>& qRetiredSwapchainResources, const uint64_t completedFrameId)
{
	while (!qRetiredSwapchainResources.empty() && qRetiredSwapchainResources.front()->retiredFrameId < completedFrameId)
	{
		std::cout << "Dropping resource scheduled at " << qRetiredSwapchainResources.front()->retiredFrameId << " with completedFrameId " << completedFrameId << "\n";
		delete(qRetiredSwapchainResources.front());
		qRetiredSwapchainResources.pop_front();
	}
}

void CommonAPI::retireSwapchainResources(nbl::core::deque<IRetiredSwapchainResources*>& qRetiredSwapchainResources, IRetiredSwapchainResources* retired)
{
	qRetiredSwapchainResources.push_back(retired);
}

nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> CommonAPI::createRenderpass(const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device, nbl::asset::E_FORMAT colorAttachmentFormat, nbl::asset::E_FORMAT baseDepthFormat)
{
	using namespace nbl;

	bool useDepth = baseDepthFormat != nbl::asset::EF_UNKNOWN;
	nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN;
	if (useDepth)
	{
		depthFormat = device->getPhysicalDevice()->promoteImageFormat(
			{ baseDepthFormat, nbl::video::IPhysicalDevice::SFormatImageUsages::SUsage(nbl::asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT) },
			nbl::video::IGPUImage::ET_OPTIMAL
		);
		assert(depthFormat != nbl::asset::EF_UNKNOWN);
	}

	nbl::video::IGPURenderpass::SCreationParams::SAttachmentDescription attachments[2];
	attachments[0].initialLayout = asset::IImage::EL_UNDEFINED;
	attachments[0].finalLayout = asset::IImage::EL_PRESENT_SRC;
	attachments[0].format = colorAttachmentFormat;
	attachments[0].samples = asset::IImage::ESCF_1_BIT;
	attachments[0].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
	attachments[0].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

	attachments[1].initialLayout = asset::IImage::EL_UNDEFINED;
	attachments[1].finalLayout = asset::IImage::EL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachments[1].format = depthFormat;
	attachments[1].samples = asset::IImage::ESCF_1_BIT;
	attachments[1].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
	attachments[1].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

	nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
	colorAttRef.attachment = 0u;
	colorAttRef.layout = asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL;

	nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef depthStencilAttRef;
	depthStencilAttRef.attachment = 1u;
	depthStencilAttRef.layout = asset::IImage::EL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

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

nbl::core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>> CommonAPI::createFBOWithSwapchainImages(
	size_t imageCount, uint32_t width, uint32_t height,
	const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
	nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain,
	nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass,
	nbl::asset::E_FORMAT baseDepthFormat
) {
	using namespace nbl;

	bool useDepth = baseDepthFormat != nbl::asset::EF_UNKNOWN;
	nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN;
	if (useDepth)
	{
		depthFormat = baseDepthFormat;
		//depthFormat = device->getPhysicalDevice()->promoteImageFormat(
		//	{ baseDepthFormat, nbl::video::IPhysicalDevice::SFormatImageUsages::SUsage(nbl::asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT) },
		//	nbl::asset::IImage::ET_OPTIMAL
		//);
		// TODO error reporting
		assert(depthFormat != nbl::asset::EF_UNKNOWN);
	}

	auto fbo = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>>>(imageCount);
	for (uint32_t i = 0u; i < imageCount; ++i)
	{
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> view[2] = {};

		auto img = swapchain->createImage(i);
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

bool CommonAPI::createSwapchain(
	const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>&& device,
	nbl::video::ISwapchain::SCreationParams& params,
	uint32_t width, uint32_t height,
	// nullptr for initial creation, old swapchain for eventual resizes
	nbl::core::smart_refctd_ptr<nbl::video::ISwapchain>& swapchain
)
{
	auto oldSwapchain = swapchain;

	nbl::video::ISurface::SCapabilities surfaceCapabilities;
	bool didGetCaps = params.surface->getSurfaceCapabilitiesForPhysicalDevice(device->getPhysicalDevice(), surfaceCapabilities);
	assert(didGetCaps);

	uint32_t w = std::max(std::min(width, surfaceCapabilities.maxImageExtent.width), surfaceCapabilities.minImageExtent.width);
	uint32_t h = std::max(std::min(width, surfaceCapabilities.maxImageExtent.height), surfaceCapabilities.minImageExtent.height);

	nbl::video::ISwapchain::SCreationParams paramsCp = params;
	paramsCp.width = w;
	paramsCp.height = h;
	paramsCp.oldSwapchain = oldSwapchain;

	if (device->getAPIType() == nbl::video::EAT_VULKAN)
	{
		swapchain = nbl::video::CVulkanSwapchain::create(std::move(device), std::move(paramsCp));
	}
	else if (device->getAPIType() == nbl::video::EAT_OPENGL)
	{
		swapchain = nbl::video::COpenGLSwapchain::create(std::move(device), std::move(paramsCp));
	}
	else if (device->getAPIType() == nbl::video::EAT_OPENGL_ES)
	{
		swapchain = nbl::video::COpenGLESSwapchain::create(std::move(device), std::move(paramsCp));
	}
	else
	{
		_NBL_TODO();
	}

	assert(swapchain);
	assert(swapchain != oldSwapchain);

	return true;
}

void CommonAPI::performGpuInit(InitParams& params, InitOutput& result)
{
	using namespace nbl;
	using namespace nbl::video;

	bool headlessCompute = params.isHeadlessCompute();

	if (params.apiType == EAT_VULKAN)
	{
		auto _apiConnection = nbl::video::CVulkanConnection::create(
			nbl::core::smart_refctd_ptr(result.system),
			0,
			params.appName.data(),
			nbl::core::smart_refctd_ptr(result.logger),
			params.apiFeaturesToEnable);

		if (!headlessCompute)
		{
#ifdef _NBL_PLATFORM_WINDOWS_
			result.surface = nbl::video::CSurfaceVulkanWin32::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowWin32>(static_cast<nbl::ui::IWindowWin32*>(params.window.get())));
#elif defined(_NBL_PLATFORM_ANDROID_)
			////result.surface = nbl::video::CSurfaceVulkanAndroid::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowAndroid>(static_cast<nbl::ui::IWindowAndroid*>(params.window.get())));
#endif
		}
		result.apiConnection = _apiConnection;
	}
	else if (params.apiType == EAT_OPENGL)
	{
		auto _apiConnection = nbl::video::COpenGLConnection::create(nbl::core::smart_refctd_ptr(result.system), 0, params.appName.data(), nbl::video::COpenGLDebugCallback(nbl::core::smart_refctd_ptr(result.logger)));

		if (!headlessCompute)
		{
#ifdef _NBL_PLATFORM_WINDOWS_
			result.surface = nbl::video::CSurfaceGLWin32::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowWin32>(static_cast<nbl::ui::IWindowWin32*>(params.window.get())));
#elif defined(_NBL_PLATFORM_ANDROID_)
			result.surface = nbl::video::CSurfaceGLAndroid::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowAndroid>(static_cast<nbl::ui::IWindowAndroid*>(params.window.get())));
#endif
		}

		result.apiConnection = _apiConnection;
	}
	else if (params.apiType == EAT_OPENGL_ES)
	{
		auto _apiConnection = nbl::video::COpenGLESConnection::create(nbl::core::smart_refctd_ptr(result.system), 0, params.appName.data(), nbl::video::COpenGLDebugCallback(nbl::core::smart_refctd_ptr(result.logger)));

		if (!headlessCompute)
		{
#ifdef _NBL_PLATFORM_WINDOWS_
			result.surface = nbl::video::CSurfaceGLWin32::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowWin32>(static_cast<nbl::ui::IWindowWin32*>(params.window.get())));
#elif defined(_NBL_PLATFORM_ANDROID_)
			result.surface = nbl::video::CSurfaceGLAndroid::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowAndroid>(static_cast<nbl::ui::IWindowAndroid*>(params.window.get())));
#endif
		}

		result.apiConnection = _apiConnection;
	}
	else
	{
		_NBL_TODO();
	}

	auto gpus = result.apiConnection->getPhysicalDevices();
	assert(!gpus.empty());
	SDefaultPhysicalDeviceFilter defaultFilter = {};
	defaultFilter.minApiVersion.major = 1u; 
	defaultFilter.minApiVersion.minor = 1u; 
	defaultFilter.minApiVersion.patch = 0u; 
	auto filteredPhysicalDevices = getFilteredPhysicalDevices(gpus, params.physicalDeviceFilter);

	CDefaultPhysicalDeviceSelector defaultPhysicalDeviceSelector = {};
	if(params.physicalDeviceSelector == nullptr)
		params.physicalDeviceSelector = &defaultPhysicalDeviceSelector;

	auto selectedPhysicalDevice = params.physicalDeviceSelector->selectPhysicalDevice(filteredPhysicalDevices);
	auto queuesInfo = extractPhysicalDeviceQueueInfos(selectedPhysicalDevice, result.surface, headlessCompute);

	// Fill QueueCreationParams
	constexpr uint32_t MaxQueuesInFamily = 32;
	float queuePriorities[MaxQueuesInFamily];
	std::fill(queuePriorities, queuePriorities + MaxQueuesInFamily, IGPUQueue::DEFAULT_QUEUE_PRIORITY);

	constexpr uint32_t MaxQueueFamilyCount = 4;
	nbl::video::ILogicalDevice::SQueueCreationParams qcp[MaxQueueFamilyCount] = {};

	uint32_t actualQueueParamsCount = 0u;

	uint32_t queuesIndexInFamily[InitOutput::EQT_COUNT];
	uint32_t presentQueueIndexInFamily = 0u;

	// Graphics Queue
	if (!headlessCompute)
	{
		uint32_t dedicatedQueuesInFamily = queuesInfo.graphics.dedicatedQueueCount;
		assert(dedicatedQueuesInFamily >= 1u);

		qcp[0].familyIndex = queuesInfo.graphics.index;
		qcp[0].count = dedicatedQueuesInFamily;
		qcp[0].flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
		qcp[0].priorities = queuePriorities;
		queuesIndexInFamily[InitOutput::EQT_GRAPHICS] = 0u;
		actualQueueParamsCount++;
	}

	// Compute Queue
	bool foundComputeInOtherFamily = false;
	for (uint32_t i = 0; i < actualQueueParamsCount; ++i)
	{
		auto& otherQcp = qcp[i];
		uint32_t dedicatedQueuesInFamily = queuesInfo.compute.dedicatedQueueCount;
		if (otherQcp.familyIndex == queuesInfo.compute.index)
		{
			if (dedicatedQueuesInFamily >= 1)
			{
				queuesIndexInFamily[InitOutput::EQT_COMPUTE] = otherQcp.count + 0u;
			}
			else
			{
				queuesIndexInFamily[InitOutput::EQT_COMPUTE] = 0u;
			}
			otherQcp.count += dedicatedQueuesInFamily;
			foundComputeInOtherFamily = true;
			break; // If works correctly no need to check other family indices as they are unique
		}
	}
	if (!foundComputeInOtherFamily)
	{
		uint32_t dedicatedQueuesInFamily = queuesInfo.compute.dedicatedQueueCount;
		assert(dedicatedQueuesInFamily == 1u);

		queuesIndexInFamily[InitOutput::EQT_COMPUTE] = 0u;

		auto& computeQcp = qcp[actualQueueParamsCount];
		computeQcp.familyIndex = queuesInfo.compute.index;
		computeQcp.count = dedicatedQueuesInFamily;
		computeQcp.flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
		computeQcp.priorities = queuePriorities;
		actualQueueParamsCount++;
	}

	// Transfer Queue
	bool foundTransferInOtherFamily = false;
	for (uint32_t i = 0; i < actualQueueParamsCount; ++i)
	{
		auto& otherQcp = qcp[i];
		uint32_t dedicatedQueuesInFamily = queuesInfo.transfer.dedicatedQueueCount;
		if (otherQcp.familyIndex == queuesInfo.transfer.index)
		{
			if (dedicatedQueuesInFamily >= 2u)
			{
				queuesIndexInFamily[InitOutput::EQT_TRANSFER_UP] = otherQcp.count + 0u;
				queuesIndexInFamily[InitOutput::EQT_TRANSFER_DOWN] = otherQcp.count + 1u;
			}
			else if (dedicatedQueuesInFamily >= 1u)
			{
				queuesIndexInFamily[InitOutput::EQT_TRANSFER_UP] = otherQcp.count + 0u;
				queuesIndexInFamily[InitOutput::EQT_TRANSFER_DOWN] = otherQcp.count + 0u;
			}
			else if (dedicatedQueuesInFamily == 0u)
			{
				queuesIndexInFamily[InitOutput::EQT_TRANSFER_UP] = 0u;
				queuesIndexInFamily[InitOutput::EQT_TRANSFER_DOWN] = 0u;
			}
			otherQcp.count += dedicatedQueuesInFamily;
			foundTransferInOtherFamily = true;
			break; // If works correctly no need to check other family indices as they are unique
		}
	}
	if (!foundTransferInOtherFamily)
	{
		uint32_t dedicatedQueuesInFamily = queuesInfo.transfer.dedicatedQueueCount;
		assert(dedicatedQueuesInFamily >= 1u);

		if (dedicatedQueuesInFamily >= 2u)
		{
			queuesIndexInFamily[InitOutput::EQT_TRANSFER_UP] = 0u;
			queuesIndexInFamily[InitOutput::EQT_TRANSFER_DOWN] = 1u;
		}
		else if (dedicatedQueuesInFamily >= 1u)
		{
			queuesIndexInFamily[InitOutput::EQT_TRANSFER_UP] = 0u;
			queuesIndexInFamily[InitOutput::EQT_TRANSFER_DOWN] = 0u;
		}
		else
		{
			assert(false);
		}

		auto& transferQcp = qcp[actualQueueParamsCount];
		transferQcp.familyIndex = queuesInfo.transfer.index;
		transferQcp.count = dedicatedQueuesInFamily;
		transferQcp.flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
		transferQcp.priorities = queuePriorities;
		actualQueueParamsCount++;
	}

	// Present Queue
	if (!headlessCompute)
	{
		bool foundPresentInOtherFamily = false;
		for (uint32_t i = 0; i < actualQueueParamsCount; ++i)
		{
			auto& otherQcp = qcp[i];
			if (otherQcp.familyIndex == queuesInfo.present.index)
			{
				if (otherQcp.familyIndex == queuesInfo.graphics.index)
				{
					presentQueueIndexInFamily = 0u;
				}
				else
				{
					uint32_t dedicatedQueuesInFamily = queuesInfo.present.dedicatedQueueCount;

					if (dedicatedQueuesInFamily >= 1u)
					{
						presentQueueIndexInFamily = otherQcp.count + 0u;
					}
					else if (dedicatedQueuesInFamily == 0u)
					{
						presentQueueIndexInFamily = 0u;
					}
					otherQcp.count += dedicatedQueuesInFamily;
				}
				foundPresentInOtherFamily = true;
				break; // If works correctly no need to check other family indices as they are unique
			}
		}
		if (!foundPresentInOtherFamily)
		{
			uint32_t dedicatedQueuesInFamily = queuesInfo.present.dedicatedQueueCount;
			assert(dedicatedQueuesInFamily == 1u);
			presentQueueIndexInFamily = 0u;

			auto& presentQcp = qcp[actualQueueParamsCount];
			presentQcp.familyIndex = queuesInfo.present.index;
			presentQcp.count = dedicatedQueuesInFamily;
			presentQcp.flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
			presentQcp.priorities = queuePriorities;
			actualQueueParamsCount++;
		}
	}

	nbl::video::ILogicalDevice::SCreationParams dev_params;
	dev_params.queueParamsCount = actualQueueParamsCount;
	dev_params.queueParams = qcp;
	dev_params.featuresToEnable = params.physicalDeviceFilter.requiredFeatures;
	result.logicalDevice = selectedPhysicalDevice->createLogicalDevice(std::move(dev_params));

	result.utilities = nbl::core::make_smart_refctd_ptr<nbl::video::IUtilities>(nbl::core::smart_refctd_ptr(result.logicalDevice));

	if (!headlessCompute)
		result.queues[InitOutput::EQT_GRAPHICS] = result.logicalDevice->getQueue(queuesInfo.graphics.index, queuesIndexInFamily[InitOutput::EQT_GRAPHICS]);
	result.queues[InitOutput::EQT_COMPUTE] = result.logicalDevice->getQueue(queuesInfo.compute.index, queuesIndexInFamily[InitOutput::EQT_COMPUTE]);

	// TEMP_FIX
#ifdef EXAMPLES_CAN_HANDLE_TRANSFER_WITHOUT_GRAPHICS 
	result.queues[InitOutput::EQT_TRANSFER_UP] = result.logicalDevice->getQueue(queuesInfo.transfer.index, queuesIndexInFamily[EQT_TRANSFER_UP]);
	result.queues[InitOutput::EQT_TRANSFER_DOWN] = result.logicalDevice->getQueue(queuesInfo.transfer.index, queuesIndexInFamily[EQT_TRANSFER_DOWN]);
#else
	if (!headlessCompute)
	{
		result.queues[InitOutput::EQT_TRANSFER_UP] = result.logicalDevice->getQueue(queuesInfo.graphics.index, 0u);
		result.queues[InitOutput::EQT_TRANSFER_DOWN] = result.logicalDevice->getQueue(queuesInfo.graphics.index, 0u);
	}
	else
	{
		result.queues[InitOutput::EQT_TRANSFER_UP] = result.logicalDevice->getQueue(queuesInfo.compute.index, queuesIndexInFamily[InitOutput::EQT_COMPUTE]);
		result.queues[InitOutput::EQT_TRANSFER_DOWN] = result.logicalDevice->getQueue(queuesInfo.compute.index, queuesIndexInFamily[InitOutput::EQT_COMPUTE]);
	}
#endif
	if (!headlessCompute)
	{
		result.swapchainCreationParams = computeSwapchainCreationParams(
			params.swapchainImageCount,
			result.logicalDevice,
			result.surface,
			params.swapchainImageUsage,
			params.acceptableSurfaceFormats, params.acceptableSurfaceFormatCount,
			params.acceptableColorPrimaries, params.acceptableColorPrimaryCount,
			params.acceptableEotfs, params.acceptableEotfCount,
			params.acceptablePresentModes, params.acceptablePresentModeCount,
			params.acceptableSurfaceTransforms, params.acceptableSurfaceTransformCount
		);

		nbl::asset::E_FORMAT swapChainFormat = result.swapchainCreationParams.surfaceFormat.format;
		result.renderToSwapchainRenderpass = createRenderpass(result.logicalDevice, swapChainFormat, params.depthFormat);
	}

	uint32_t commandPoolsToCreate = core::max(params.framesInFlight, 1u);
	for (uint32_t i = 0; i < InitOutput::EQT_COUNT; ++i)
	{
		const IGPUQueue* queue = result.queues[i];
		if (queue != nullptr)
		{
			for (size_t j = 0; j < commandPoolsToCreate; j++)
			{
				result.commandPools[i][j] = result.logicalDevice->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
				assert(result.commandPools[i][j]);
			}
		}
	}

	result.physicalDevice = selectedPhysicalDevice;

	uint32_t mainQueueFamilyIndex = (headlessCompute) ? queuesInfo.compute.index : queuesInfo.graphics.index;
	result.cpu2gpuParams.assetManager = result.assetManager.get();
	result.cpu2gpuParams.device = result.logicalDevice.get();
	result.cpu2gpuParams.finalQueueFamIx = mainQueueFamilyIndex;
	result.cpu2gpuParams.limits = result.physicalDevice->getLimits();
	result.cpu2gpuParams.pipelineCache = nullptr;
	result.cpu2gpuParams.utilities = result.utilities.get();

	result.cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = result.queues[InitOutput::EQT_TRANSFER_UP];
	result.cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = result.queues[InitOutput::EQT_COMPUTE];

	const uint32_t transferUpQueueFamIndex = result.queues[InitOutput::EQT_TRANSFER_UP]->getFamilyIndex();
	const uint32_t computeQueueFamIndex = result.queues[InitOutput::EQT_COMPUTE]->getFamilyIndex();

	auto pool_transfer = result.logicalDevice->createCommandPool(transferUpQueueFamIndex, IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
	nbl::core::smart_refctd_ptr<IGPUCommandPool> pool_compute;
	if (transferUpQueueFamIndex == computeQueueFamIndex)
		pool_compute = pool_transfer;
	else
		pool_compute = result.logicalDevice->createCommandPool(result.queues[InitOutput::EQT_COMPUTE]->getFamilyIndex(), IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);

	nbl::core::smart_refctd_ptr<IGPUCommandBuffer> transferCmdBuffer;
	nbl::core::smart_refctd_ptr<IGPUCommandBuffer> computeCmdBuffer;

	result.logicalDevice->createCommandBuffers(pool_transfer.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &transferCmdBuffer);
	result.logicalDevice->createCommandBuffers(pool_compute.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &computeCmdBuffer);

	result.cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_TRANSFER].cmdbuf = transferCmdBuffer;
	result.cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_COMPUTE].cmdbuf = computeCmdBuffer;
}
