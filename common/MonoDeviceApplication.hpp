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
		// This time we build upon the Mono-System and Mono-Logger application and add the choice of a single physical device and creation of utilities
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
				ILogicalDevice::SCreationParams params = {};

				const auto queueParams = getQueueCreationParameters();
				params.queueParamsCount = queueParams.size();
				params.queueParams = queueParams.data();
				
				IPhysicalDevice* selectedDevice = selectPhysicalDevice(suitablePhysicalDevices);
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

		// virtual function so you can override as needed for some example father down the line
		virtual video::IPhysicalDevice::SFormatImageUsages getRequiredOptimalTilingImageUsages() const
		{
			using usages_t = video::IPhysicalDevice::SFormatImageUsages;
			usages_t retval = {};
			
			using format_usage_t = usages_t::SUsage;
			using namespace nbl::asset;
			// Lets declare a few common usages of images
			const format_usage_t sampling(IImage::EUF_SAMPLED_BIT);
			const format_usage_t transferUpAndDown(IImage::EUF_TRANSFER_DST_BIT|IImage::EUF_TRANSFER_SRC_BIT);
			const format_usage_t shaderStorage(IImage::EUF_STORAGE_BIT);
			const format_usage_t shaderStorageAtomic = shaderStorage | []()->auto {format_usage_t tmp; tmp.storageImageAtomic = true; return tmp;}();
			const format_usage_t attachment = []()->auto {format_usage_t tmp; tmp.attachment = true; return tmp; }();
			const format_usage_t attachmentBlend = []()->auto {format_usage_t tmp; tmp.attachmentBlend = true; return tmp; }();
			const format_usage_t blitSrc = []()->auto {format_usage_t tmp; tmp.blitSrc = true; return tmp; }();
			const format_usage_t blitDst = []()->auto {format_usage_t tmp; tmp.blitDst = true; return tmp; }();
			// TODO: redo when we incorporate blits into the asset converter (just sampling then)
			const format_usage_t mipmapGeneration = sampling|blitSrc|blitDst;
			// we care that certain "basic" formats are usable in some "basic" ways
			retval[EF_R32_UINT] = shaderStorageAtomic;
			const format_usage_t opaqueRendering = sampling|transferUpAndDown|attachment|mipmapGeneration;
			const format_usage_t genericRendering = opaqueRendering|attachmentBlend|mipmapGeneration;
			retval[EF_R8_UNORM] = genericRendering;
			retval[EF_R8G8_UNORM] = genericRendering;
			retval[EF_R8G8B8A8_UNORM] = genericRendering;
			retval[EF_R8G8B8A8_SRGB] = genericRendering;
			const format_usage_t renderingAndStorage = genericRendering|shaderStorage;
			retval[EF_R16_SFLOAT] = renderingAndStorage;
			retval[EF_R16G16_SFLOAT] = renderingAndStorage;
			retval[EF_R16G16B16A16_SFLOAT] = renderingAndStorage;
			retval[EF_R32_SFLOAT] = renderingAndStorage;
			retval[EF_R32G32_SFLOAT] = renderingAndStorage;
			retval[EF_R32G32B32A32_SFLOAT] = renderingAndStorage;

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
		virtual core::vector<video::SPhysicalDeviceFilter::QueueRequirement> getQueueRequirements() const
		{
			using namespace core;
			using namespace video;
			vector<SPhysicalDeviceFilter::QueueRequirement> retval;
			
			using flags_t = IPhysicalDevice::E_QUEUE_FLAGS;
			// The Graphics Queue should be able to do Compute and image transfers of any granularity (transfer only queue families can have problems with that)
			retval.push_back({.requiredFlags=flags_t::EQF_GRAPHICS_BIT|flags_t::EQF_COMPUTE_BIT,.disallowedFlags=flags_t::EQF_NONE,.queueCount=1,.maxImageTransferGranularity={1,1,1}});

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

		// Queue choice is a bit complicated, for basic usage 1 graphics, 1 async compute, and 2 async transfer queues are enough.
		virtual core::vector<video::ILogicalDevice::SQueueCreationParams> getQueueCreationParameters()
		{
			using namespace video;
			core::vector<ILogicalDevice::SQueueCreationParams> retval;

			// TODO: redo this
			retval.push_back({.flags=IGPUQueue::ECF_NONE,.familyIndex=0,.count=1});

			return retval;
		}

		// However some devices might not have that many queues so here we'll "alias" a queue to multiple uses.
		virtual video::IGPUQueue* getComputeQueue() const
		{
			return m_queue;
		}


		core::smart_refctd_ptr<video::CVulkanConnection> m_api;
		core::smart_refctd_ptr<video::ILogicalDevice> m_device;

	private:
		video::IGPUQueue* m_queue = nullptr;
};

}

#endif // _CAMERA_IMPL_