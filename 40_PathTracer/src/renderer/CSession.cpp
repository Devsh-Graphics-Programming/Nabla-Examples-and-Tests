// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "renderer/CRenderer.h"

namespace nbl::this_example
{
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::hlsl;
using namespace nbl::video;

//
bool CSession::init(SIntendedSubmitInfo& info)
{
	auto renderer = m_params.scene->getRenderer();
	auto& logger = renderer->getCreationParams().logger;
	auto device = renderer->getDevice();

	auto& immutables = m_active.immutables;

	// create the descriptors
	core::vector<IGPUDescriptorSet::SDescriptorInfo> infos;
	core::vector<IGPUDescriptorSet::SWriteDescriptorSet> writes;
	{
		auto addWrite = [&](const uint32_t binding, IGPUDescriptorSet::SDescriptorInfo&& info)->void
		{
			writes.emplace_back() = {
				.binding = binding,
				.arrayElement = 0,
				.count = 1,
				.info = reinterpret_cast<const IGPUDescriptorSet::SDescriptorInfo*>(infos.size())
			};
			infos.push_back(std::move(info));
		};

		//
		auto dedicatedAllocate = [&](IDeviceMemoryBacked* memBacked, const std::string_view debugName)->bool
		{
			if (!memBacked)
			{
				logger.log("Failed to create Sensor \"%s\"'s \"%s\" in CSession::init()",ILogger::ELL_ERROR,m_params.name.c_str(),debugName.data());
				return false;
			}
			memBacked->setObjectDebugName(debugName.data());

			auto mreqs = memBacked->getMemoryReqs();
			mreqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			using flags_e = IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS;
			core::bitflag<flags_e> flags = flags_e::EMAF_NONE;
			if (memBacked->getObjectType()==IDeviceMemoryBacked::E_OBJECT_TYPE::EOT_BUFFER &&
				static_cast<IGPUBuffer*>(memBacked)->getCreationParams().usage.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT))
				flags |= flags_e::EMAF_DEVICE_ADDRESS_BIT;
			if (!device->allocate(mreqs,memBacked,flags).isValid())
			{
				logger.log("Could not allocate memory for Sensor \"%s\"'s \"%s\" in CSession::init()",ILogger::ELL_ERROR,m_params.name.c_str(),debugName.data());
				return false;
			}
			return true;
		};

		// create UBO
		{
			IGPUBuffer::SCreationParams params = {};
			params.size = sizeof(m_params.uniforms);
			using usage_flags_e = IGPUBuffer::E_USAGE_FLAGS;
			params.usage = usage_flags_e::EUF_UNIFORM_BUFFER_BIT | usage_flags_e::EUF_TRANSFER_DST_BIT | usage_flags_e::EUF_INLINE_UPDATE_VIA_CMDBUF;
			auto ubo = device->createBuffer(std::move(params));
			if (!dedicatedAllocate(ubo.get(),"Sensor UBO"))
				return false;
			// pipeline barrier in `reset` will take care of sync for this
			info.getCommandBufferForRecording()->cmdbuf->updateBuffer({.size=sizeof(m_params.uniforms),.buffer=ubo},&m_params.uniforms);
			addWrite(SensorDSBindings::UBO,SBufferRange<IGPUBuffer>{.offset=0,.size=sizeof(m_params.uniforms),.buffer=ubo});
		}

		const auto allowedFormatUsages = device->getPhysicalDevice()->getImageFormatUsagesOptimalTiling();
		auto createImage = [&](
			const std::string_view debugName, const E_FORMAT format, const uint16_t2 resolution, const uint16_t layers,
			const IGPUImage::E_CREATE_FLAGS extraFlags=IGPUImage::E_CREATE_FLAGS::ECF_NONE, std::bitset<E_FORMAT::EF_COUNT> viewFormats={},
			const IGPUImage::E_USAGE_FLAGS extraUsages=IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT|IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT
		) -> SImageWithViews
		{
				SImageWithViews retval = {};
			{
				{
					IGPUImage::SCreationParams params = {};
					params.type = IGPUImage::E_TYPE::ET_2D;
					params.samples = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
					params.format = format;
					params.extent.width = resolution[0];
					params.extent.height = resolution[1];
					params.extent.depth = 1;
					params.mipLevels = 1;
					params.arrayLayers = layers;
					params.flags |= extraFlags;
					using image_usage_e = IGPUImage::E_USAGE_FLAGS;
					params.usage = image_usage_e::EUF_TRANSFER_DST_BIT|image_usage_e::EUF_TRANSFER_SRC_BIT|extraUsages;
					viewFormats.set(format);
					if (viewFormats.count()>1)
					{
						params.flags |= IGPUImage::E_CREATE_FLAGS::ECF_MUTABLE_FORMAT_BIT;
						params.flags |= IGPUImage::E_CREATE_FLAGS::ECF_EXTENDED_USAGE_BIT;
					}
					params.viewFormats = viewFormats;
					retval.image = device->createImage(std::move(params));
					if (!dedicatedAllocate(retval.image.get(),debugName))
						return {};
				}
				const auto& params = retval.image->getCreationParameters();
				for (uint8_t f=0; f<viewFormats.size(); f++)
				if (viewFormats.test(f))
				{
					const auto viewFormat = static_cast<E_FORMAT>(f);
					const auto thisFormatUsages = static_cast<core::bitflag<IGPUImage::E_USAGE_FLAGS>>(allowedFormatUsages[viewFormat]);
					auto view = device->createImageView({
						.subUsages = retval.image->getCreationParameters().usage & thisFormatUsages,
						.image = retval.image,
						.viewType = IGPUImageView::E_TYPE::ET_2D_ARRAY,
						.format = viewFormat
					});
					string viewDebugName = string(debugName)+" "+to_string(viewFormat)+" View";
					if (!view)
					{
						logger.log("Failed to create Sensor \"%s\"'s \"%s\" in CSession::init()",ILogger::ELL_ERROR,m_params.name.c_str(),viewDebugName.c_str());
						return {};
					}
					view->setObjectDebugName(viewDebugName.c_str());
					retval.views[viewFormat] = std::move(view);
				}
			}
			return retval;
		};
		auto addImageWrite = [&](const uint32_t binding, const smart_refctd_ptr<IGPUImageView>& view)->void
		{
			IGPUDescriptorSet::SDescriptorInfo info = {};
			info.desc = view;
			info.info.image.imageLayout = IGPUImage::LAYOUT::GENERAL;
			addWrite(binding,std::move(info));
		};

		// create Scramble Key image
		{
			const auto layers = 1u; // for now, until the crazy Heitz 2019 thing, or if we choose to save 8 bytes in ray payload and read a premade scramble at every depth 
			immutables.scrambleKey = createImage("Scramble Dimension Keys",E_FORMAT::EF_R32G32_UINT,hlsl::promote<hlsl::uint16_t2>(SSensorUniforms::ScrambleKeyTextureSize),layers);
		}
		//
		auto scrambleKeyView = immutables.scrambleKey.views[E_FORMAT::EF_R32G32_UINT];
		addImageWrite(SensorDSBindings::ScrambleKey,scrambleKeyView);

		// create the render-sized images
		auto createScreenSizedImage = [&]<typename... Args>(const std::string_view debugName, const E_FORMAT format, uint16_t layers=1, Args&&... args)->SImageWithViews
		{
			using create_flags_e = IGPUImage::E_CREATE_FLAGS;
			create_flags_e flags = create_flags_e::ECF_NONE;
			if (m_params.type==sensor_type_e::Env)
			{
				layers *= 6;
				flags = IGPUImage::E_CREATE_FLAGS::ECF_CUBE_COMPATIBLE_BIT;
			}
			return createImage(debugName,format,m_params.uniforms.renderSize,layers,flags,std::forward<Args>(args)...);
		};
		immutables.sampleCount = createScreenSizedImage("Current Sample Count",E_FORMAT::EF_R16_UINT);
		auto sampleCountView = immutables.sampleCount.views[E_FORMAT::EF_R16_UINT];
		addImageWrite(SensorDSBindings::SampleCount,sampleCountView);
		immutables.rwmcCascades = createScreenSizedImage("RWMC Cascades",E_FORMAT::EF_R32G32_UINT,m_params.uniforms.lastCascadeIndex+1,std::bitset<E_FORMAT::EF_COUNT>().set(E_FORMAT::EF_R16G16B16A16_SFLOAT));
		addImageWrite(SensorDSBindings::RWMCCascades,immutables.rwmcCascades.views[E_FORMAT::EF_R32G32_UINT]);
		immutables.beauty = createScreenSizedImage("Beauty",E_FORMAT::EF_R16G16B16A16_SFLOAT);
		addImageWrite(SensorDSBindings::Beauty,immutables.beauty.views[E_FORMAT::EF_R16G16B16A16_SFLOAT]);
		immutables.albedo = createScreenSizedImage("Albedo",E_FORMAT::EF_R16G16B16A16_SFLOAT);
		auto albedoView = immutables.albedo.views[E_FORMAT::EF_R16G16B16A16_SFLOAT];
		addImageWrite(SensorDSBindings::Albedo,albedoView);
		immutables.normal = createScreenSizedImage("Normal",E_FORMAT::EF_R16G16B16A16_SFLOAT);
		auto normalView = immutables.normal.views[E_FORMAT::EF_R16G16B16A16_SFLOAT];
		addImageWrite(SensorDSBindings::Normal,normalView);
		immutables.motion = createScreenSizedImage("Motion",E_FORMAT::EF_A2B10G10R10_UNORM_PACK32);
		auto motionView = immutables.motion.views[E_FORMAT::EF_A2B10G10R10_UNORM_PACK32];
		addImageWrite(SensorDSBindings::Motion,motionView);
		immutables.mask = createScreenSizedImage("Mask",E_FORMAT::EF_R16_UNORM);
		auto maskView = immutables.mask.views[E_FORMAT::EF_R16_UNORM];
		addImageWrite(SensorDSBindings::Mask,maskView);
		// shorthand a little bit
		addImageWrite(SensorDSBindings::AsSampledImages,scrambleKeyView);
		writes.back().count = SensorDSBindingCounts::AsSampledImages;
		{
			const auto oldSize = infos.size();
			infos.resize(oldSize +SensorDSBindingCounts::AsSampledImages,infos.back());
			const auto viewInfos = infos.data()+oldSize-1;
			using index_e = SensorDSBindings::SampledImageIndex;
			viewInfos[uint8_t(index_e::ScrambleKey)].desc = scrambleKeyView;
			viewInfos[uint8_t(index_e::SampleCount)].desc = sampleCountView;
			viewInfos[uint8_t(index_e::RWMCCascades)].desc = immutables.rwmcCascades.views[E_FORMAT::EF_R16G16B16A16_SFLOAT];
			viewInfos[uint8_t(index_e::Beauty)].desc = immutables.beauty.views[E_FORMAT::EF_R16G16B16A16_SFLOAT];
			viewInfos[uint8_t(index_e::Albedo)].desc = albedoView;
			viewInfos[uint8_t(index_e::Normal)].desc = normalView;
			viewInfos[uint8_t(index_e::Motion)].desc = motionView;
			viewInfos[uint8_t(index_e::Mask)].desc = maskView;
		}
	}

	// create descriptor set
	{
		auto layout = renderer->getConstructionParams().sensorDSLayout;
		auto pool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT,{&layout.get(),1});
		immutables.ds = pool->createDescriptorSet(std::move(layout));
		const char* DebugName = "Sensor Descriptor Set";
		if (!immutables.ds)
		{
			logger.log("Failed to create Sensor \"%s\"'s \"%s\" in CSession::init()",ILogger::ELL_ERROR,m_params.name.c_str(),DebugName);
			return false;
		}
		immutables.ds->setObjectDebugName(DebugName);
		for (auto& write : writes)
		{
			write.dstSet = immutables.ds.get();
			write.info = infos.data()+reinterpret_cast<const uint64_t&>(write.info);
		}
		if (!device->updateDescriptorSets(writes,{}))
		{
			logger.log("Failed to write Sensor \"%s\"'s \"%s\" in CSession::init()",ILogger::ELL_ERROR,m_params.name.c_str(),DebugName);
			return false;
		}
	}

	bool success = immutables;

	// transition image layouts instead of barriering in Reset
	if (success)
	{
		// slam the barriers as big as possible, it wont happen frequently
		using image_barrier_t = IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t;
		core::vector<image_barrier_t> barr;
		{
			constexpr image_barrier_t base = {
				.barrier = {
					.dep = {
						.dstStageMask = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT,
						.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS|ACCESS_FLAGS::SHADER_WRITE_BITS
					}
				},
				.subresourceRange = {},
				.newLayout = IGPUImage::LAYOUT::GENERAL
			};
			barr.reserve(SensorDSBindingCounts::AsSampledImages);

			auto enqueueBarrier = [&barr,base](const SImageWithViews& img)->void
			{
				auto& out = barr.emplace_back(base);
				out.image = img.image.get();
				out.subresourceRange = {
					.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
					.levelCount = 1,
					.layerCount = out.image->getCreationParameters().arrayLayers
				};
			};
			enqueueBarrier(immutables.sampleCount);
			enqueueBarrier(immutables.beauty); // TODO: who will clear this? Resolver or denoiser?
			enqueueBarrier(immutables.rwmcCascades);
			enqueueBarrier(immutables.albedo);
			enqueueBarrier(immutables.normal);
			enqueueBarrier(immutables.motion);
			enqueueBarrier(immutables.mask);
		}
		success = info.getCommandBufferForRecording()->cmdbuf->pipelineBarrier(asset::EDF_NONE,{.imgBarriers=barr});
	}

	if (!success || !reset(m_params.initDynamics,info))
	{
		logger.log("Could not Init Session for sensor \"%s\" failed to reset!",ILogger::ELL_ERROR,m_params.name.c_str());
		deinit();
		return false;
	}

	return true;
}

bool CSession::reset(const SSensorDynamics& newVal, video::SIntendedSubmitInfo& info)
{
	if (!isInitialized())
		return false;

	auto* const renderer = m_params.scene->getRenderer();
	auto* const device = renderer->getDevice();
	const auto& scrambleImage = m_active.immutables.scrambleKey.image;
	const auto& params = scrambleImage->getCreationParameters();
	const IGPUImage::SSubresourceRange subresources = {
		.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
		.levelCount = 1,
		.layerCount = params.arrayLayers
	};

	bool success = true;
	constexpr auto RegularScrambleAccesses = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT|PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
	// slam the barriers as big as possible, it wont happen frequently
	using image_barrier_t = IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t;
	{
		const image_barrier_t before = {
			.barrier = {
				.dep = {
					.srcStageMask = RegularScrambleAccesses,
					.srcAccessMask = ACCESS_FLAGS::NONE, // because we don't care about reading previously written values
					.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
					.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
				}
			},
			.image = scrambleImage.get(),
			.subresourceRange = subresources,
			.newLayout = IGPUImage::LAYOUT::GENERAL
		};
		success = success && info.getCommandBufferForRecording()->cmdbuf->pipelineBarrier(asset::EDF_NONE,{.imgBarriers={&before,1}});
	}

	// fill scramble with noise
	{
		auto* const utils = renderer->getCreationParams().utilities.get();
		core::vector<hlsl::uint32_t2> data(params.extent.width*params.extent.height*params.arrayLayers);
		{
			core::RandomSampler rng(0xbadc0ffeu);
			for (auto& el : data)
				el = {rng.nextSample(),rng.nextSample()};
		}
		const ICPUImage::SBufferCopy region = {
			.bufferRowLength = params.extent.width,
			.bufferImageHeight = params.extent.height,
			.imageSubresource = {
				.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
				.mipLevel = 0u,
				.baseArrayLayer = 0u,
				.layerCount = params.arrayLayers
			},
			.imageExtent = params.extent
		};
		utils->updateImageViaStagingBuffer(info,data.data(),params.format,scrambleImage.get(),IGPUImage::LAYOUT::GENERAL,{&region,1});
	}

	{
		const image_barrier_t after = {
			.barrier = {
				.dep = {
					.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
					.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
					.dstStageMask = RegularScrambleAccesses,
					.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS|ACCESS_FLAGS::SHADER_WRITE_BITS
				}
			},
			.image = scrambleImage.get(),
			.subresourceRange = subresources,
			.newLayout = IGPUImage::LAYOUT::GENERAL
		};
		success = success && info.getCommandBufferForRecording()->cmdbuf->pipelineBarrier(asset::EDF_NONE,{.imgBarriers={&after,1}});
	}

	if (success)
	{
		m_samplesDispatched = 0u;
		m_active.currentSensorState = newVal;
		m_active.currentSensorState.keepAccumulating = false;
		m_active.prevSensorState = m_active.currentSensorState;
	}
	return success;
}

bool CSession::update(const SSensorDynamics& newVal)
{
	if (!isInitialized())
		return false;

	m_active.prevSensorState = m_active.currentSensorState;
	m_active.currentSensorState = newVal;
	// TODO: reset m_framesDispatched to 0 every time camera moves considerable amount
	m_active.currentSensorState.keepAccumulating = true;
	return true;
}

}
