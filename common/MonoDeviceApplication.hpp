// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_MONO_DEVICE_APPLICATION_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_MONO_DEVICE_APPLICATION_HPP_INCLUDED_

// Build on top of the previous one
#include "../common/MonoSystemMonoLoggerApplication.hpp"

namespace nbl::examples
{

// Virtual Inheritance because apps might end up doing diamond inheritance
class MonoDeviceApplication : public virtual MonoSystemMonoLoggerApplication
{
		using base_t = MonoSystemMonoLoggerApplication;

	public:
		using base_t::base_t;

	protected:
		// need this one for skipping passing all args into ApplicationFramework
		MonoDeviceApplication() = default;

		// This time we build upon the Mono-System and Mono-Logger application and add the choice of a single physical device
		virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			using namespace nbl::core;
			using namespace nbl::video;
			// TODO: specify version of the app
			m_api = CVulkanConnection::create(smart_refctd_ptr(m_system),0,_NBL_APP_NAME_,smart_refctd_ptr(base_t::m_logger),getAPIFeaturesToEnable());
			if (!m_api)
				return logFail("Failed to crate an IAPIConnection!");

			// declaring as auto so we can migrate to span easily later
			auto gpus = m_api->getPhysicalDevices();
			if (gpus.empty())
				return logFail("Failed to find any Nabla Core Profile Vulkan devices!");

			const core::set<video::IPhysicalDevice*> suitablePhysicalDevices = filterDevices(gpus);
			if (suitablePhysicalDevices.empty())
				return logFail("No PhysicalDevice met the feature requirements of the application!");

			// we're very constrained by the physical device selection so there's nothing to override here
			{
				IPhysicalDevice* selectedDevice = selectPhysicalDevice(suitablePhysicalDevices);

				ILogicalDevice::SCreationParams params = {};

				const auto queueParams = getQueueCreationParameters(selectedDevice->getQueueFamilyProperties());
				if (queueParams.empty())
					return logFail("Failed to compute queue creation parameters for a Logical Device!");

				params.queueParamsCount = queueParams.size();
				std::copy_n(queueParams.begin(),params.queueParamsCount,params.queueParams.begin());
				
				params.featuresToEnable = getRequiredDeviceFeatures(); // .union(getPreferredDeviceFeatures().intersect(selectedDevice->getFeatures()))
				
				m_device = selectedDevice->createLogicalDevice(std::move(params));
				if (!m_device)
					return logFail("Failed to create a Logical Device!");
			}

			return true;
		}

		// virtual function so you can override as needed for some example father down the line
		virtual video::IAPIConnection::SFeatures getAPIFeaturesToEnable()
		{
			video::IAPIConnection::SFeatures retval = {};
			retval.validations = true;
			retval.synchronizationValidation = true;
			retval.debugUtils = true;
			return retval;
		}

		// a device filter helps you create a set of physical devices that satisfy your requirements in terms of features, limits etc.
		virtual core::set<video::IPhysicalDevice*> filterDevices(const core::SRange<video::IPhysicalDevice* const>& physicalDevices) const
		{
			video::SPhysicalDeviceFilter deviceFilter = {};

			deviceFilter.minApiVersion = { 1,3,0 };
			deviceFilter.minConformanceVersion = {1,3,0,0};

			deviceFilter.minimumLimits = getRequiredDeviceLimits();
			deviceFilter.requiredFeatures = getRequiredDeviceFeatures();

			deviceFilter.requiredImageFormatUsagesOptimalTiling = getRequiredOptimalTilingImageUsages();

			const auto memoryReqs = getMemoryRequirements();
			deviceFilter.memoryRequirements = memoryReqs.data();
			deviceFilter.memoryRequirementsCount = memoryReqs.size();

			const auto queueReqs = getQueueRequirements();
			deviceFilter.queueRequirements = queueReqs.data();
			deviceFilter.queueRequirementsCount = queueReqs.size();
			
			return deviceFilter(physicalDevices);
		}

		// virtual function so you can override as needed for some example father down the line
		virtual video::SPhysicalDeviceLimits getRequiredDeviceLimits() const
		{
			video::SPhysicalDeviceLimits retval = {};

			// TODO: remove most on vulkan_1_3 branch as most will be required
			retval.subgroupOpsShaderStages = asset::IShader::ESS_COMPUTE;
			retval.shaderSubgroupBasic = true;
			retval.shaderSubgroupVote = true;
			retval.shaderSubgroupBallot = true;
			retval.shaderSubgroupShuffle = true;
			retval.shaderSubgroupShuffleRelative = true;
			retval.shaderInt64 = true;
			retval.shaderInt16 = true;
			retval.samplerAnisotropy = true;
			retval.storageBuffer16BitAccess = true;
			retval.uniformAndStorageBuffer16BitAccess = true;
			retval.storageBuffer8BitAccess = true;
			retval.uniformAndStorageBuffer8BitAccess = true;
			retval.shaderInt8 = true;
			retval.workgroupSizeFromSpecConstant = true;
			retval.externalFence = true;
			retval.externalMemory = true;
			retval.externalSemaphore = true;
			retval.spirvVersion = asset::IShaderCompiler::E_SPIRV_VERSION::ESV_1_5; // TODO: erm why is 1.6 not supported by my driver?

			return retval;
		}

		// virtual function so you can override as needed for some example father down the line
		virtual video::SPhysicalDeviceFeatures getRequiredDeviceFeatures() const
		{
			video::SPhysicalDeviceFeatures retval = {};

			// TODO: remove most on vulkan_1_3 branch as most will be required
			retval.fullDrawIndexUint32 = true;
			retval.multiDrawIndirect = true;
			retval.drawIndirectFirstInstance = true;
			retval.shaderStorageImageExtendedFormats = true;
			retval.shaderStorageImageWriteWithoutFormat = true;
			retval.scalarBlockLayout = true;
			retval.uniformBufferStandardLayout = true;
			retval.shaderSubgroupExtendedTypes = true;
			retval.separateDepthStencilLayouts = true;
			retval.bufferDeviceAddress = true;
			retval.vulkanMemoryModel = true;
			retval.subgroupBroadcastDynamicId = true;
			retval.subgroupSizeControl = true;

			return retval;
		}

		// Lets declare a few common usages of images
		struct CommonFormatImageUsages
		{
			using usages_t = video::IPhysicalDevice::SFormatImageUsages;
			using format_usage_t = usages_t::SUsage;
			using image_t = nbl::asset::IImage;

			constexpr static inline format_usage_t sampling = format_usage_t(image_t::EUF_SAMPLED_BIT);
			constexpr static inline format_usage_t transferUpAndDown = format_usage_t(image_t::EUF_TRANSFER_DST_BIT|image_t::EUF_TRANSFER_SRC_BIT);
			constexpr static inline format_usage_t shaderStorage = format_usage_t(image_t::EUF_STORAGE_BIT);
			constexpr static inline format_usage_t shaderStorageAtomic = shaderStorage|[]()->auto {format_usage_t tmp; tmp.storageImageAtomic = true; return tmp;}();
			constexpr static inline format_usage_t attachment = []()->auto {format_usage_t tmp; tmp.attachment = true; return tmp; }();
			constexpr static inline format_usage_t attachmentBlend = []()->auto {format_usage_t tmp; tmp.attachmentBlend = true; return tmp; }();
			constexpr static inline format_usage_t blitSrc = []()->auto {format_usage_t tmp; tmp.blitSrc = true; return tmp; }();
			constexpr static inline format_usage_t blitDst = []()->auto {format_usage_t tmp; tmp.blitDst = true; return tmp; }();
			// TODO: redo when we incorporate blits into the asset converter (just sampling then)
			constexpr static inline format_usage_t mipmapGeneration = sampling|blitSrc|blitDst;
			constexpr static inline format_usage_t opaqueRendering = sampling|transferUpAndDown|attachment|mipmapGeneration;
			constexpr static inline format_usage_t genericRendering = opaqueRendering|attachmentBlend|mipmapGeneration;
			constexpr static inline format_usage_t renderingAndStorage = genericRendering|shaderStorage;
		};

		// virtual function so you can override as needed for some example father down the line
		virtual video::IPhysicalDevice::SFormatImageUsages getRequiredOptimalTilingImageUsages() const
		{
			video::IPhysicalDevice::SFormatImageUsages retval = {};
			
			using namespace nbl::asset;
			// we care that certain "basic" formats are usable in some "basic" ways
			retval[EF_R32_UINT] = CommonFormatImageUsages::shaderStorageAtomic;
			retval[EF_R8_UNORM] = CommonFormatImageUsages::genericRendering;
			retval[EF_R8G8_UNORM] = CommonFormatImageUsages::genericRendering;
			retval[EF_R8G8B8A8_UNORM] = CommonFormatImageUsages::genericRendering;
			retval[EF_R8G8B8A8_SRGB] = CommonFormatImageUsages::genericRendering;
			retval[EF_R16_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;
			retval[EF_R16G16_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;
			retval[EF_R16G16B16A16_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;
			retval[EF_R32_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;
			retval[EF_R32G32_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;
			retval[EF_R32G32B32A32_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;

			return retval;
		}

		// virtual function so you can override as needed for some example father down the line
		virtual core::vector<video::SPhysicalDeviceFilter::MemoryRequirement> getMemoryRequirements() const
		{
			using namespace core;
			using namespace video;
			
			vector<SPhysicalDeviceFilter::MemoryRequirement> retval;
			using memory_flags_t = IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS;
			// at least 512 MB of Device Local Memory
			retval.push_back({.size=512<<20,.memoryFlags=memory_flags_t::EMPF_DEVICE_LOCAL_BIT});

			return retval;
		}

		// virtual function so you can override as needed for some example father down the line
		using queue_req_t = video::SPhysicalDeviceFilter::QueueRequirement;
		virtual core::vector<queue_req_t> getQueueRequirements() const
		{
			core::vector<queue_req_t> retval;
			
			using flags_t = video::IPhysicalDevice::E_QUEUE_FLAGS;
			// The Graphics Queue should be able to do Compute and image transfers of any granularity (transfer only queue families can have problems with that)
			retval.push_back({.requiredFlags=flags_t::EQF_COMPUTE_BIT,.disallowedFlags=flags_t::EQF_NONE,.queueCount=1,.maxImageTransferGranularity={1,1,1}});

			return retval;
		}

		// These features are features you'll enable if present but won't interfere with your choice of device
		// There's no intersection operator (yet) on the features, so its not used yet!
		// virtual function so you can override as needed for some example father down the line
		virtual video::SPhysicalDeviceFeatures getPreferredDeviceFeatures() const
		{
			video::SPhysicalDeviceFeatures retval = {};

			retval.shaderFloat64 = true;
			retval.shaderDrawParameters = true;
			retval.drawIndirectCount = true;

			return retval;
		}

		// This will get called after all physical devices go through filtering via `InitParams::physicalDeviceFilter`
		virtual video::IPhysicalDevice* selectPhysicalDevice(const core::set<video::IPhysicalDevice*>& suitablePhysicalDevices)
		{
			using namespace nbl::video;

			using driver_id_enum = IPhysicalDevice::E_DRIVER_ID;
			// from least to most buggy
			const core::vector<driver_id_enum> preference = {
				driver_id_enum::EDI_NVIDIA_PROPRIETARY,
				driver_id_enum::EDI_INTEL_OPEN_SOURCE_MESA,
				driver_id_enum::EDI_MESA_RADV,
				driver_id_enum::EDI_AMD_OPEN_SOURCE,
				driver_id_enum::EDI_MOLTENVK,
				driver_id_enum::EDI_MESA_LLVMPIPE,
				driver_id_enum::EDI_INTEL_PROPRIETARY_WINDOWS,
				driver_id_enum::EDI_AMD_PROPRIETARY,
				driver_id_enum::EDI_GOOGLE_SWIFTSHADER
			};
			// @Hazardu you'll probably want to add an override from cmdline for GPU choice here
			for (auto driver_id : preference)
			for (auto device : suitablePhysicalDevices)
			if (device->getProperties().driverID==driver_id)
				return device;

			return nullptr;
		}

		// This will most certainly be overriden
		using queue_family_range_t = core::SRange<const video::IPhysicalDevice::SQueueFamilyProperties>;
		virtual core::vector<video::ILogicalDevice::SQueueCreationParams> getQueueCreationParameters(const queue_family_range_t& familyProperties)
		{
			using namespace video;
			core::vector<ILogicalDevice::SQueueCreationParams> retval(1);

			retval[0].count = 1;
			// since we requested a device that has such a capable queue family (unless `getQueueRequirements` got overriden) we're sure we'll get at least one family
			for (auto i=0u; i<familyProperties.size(); i++)
			if (familyProperties[i].queueFlags.hasFlags(getQueueRequirements().front().requiredFlags))
				retval[0].familyIndex = i;

			return retval;
		}

		// virtual to allow aliasing and total flexibility
		virtual video::IGPUQueue* getComputeQueue() const
		{
			// In the default implementation of everything I asked only for one queue from first compute family
			const auto familyProperties = m_device->getPhysicalDevice()->getQueueFamilyProperties();
			for (auto i=0u; i<familyProperties.size(); i++)
			if (familyProperties[i].queueFlags.hasFlags(video::IPhysicalDevice::E_QUEUE_FLAGS::EQF_COMPUTE_BIT))
				return m_device->getQueue(i,0);

			return nullptr;
		}


		core::smart_refctd_ptr<video::CVulkanConnection> m_api;
		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
};

}

#endif // _CAMERA_IMPL_