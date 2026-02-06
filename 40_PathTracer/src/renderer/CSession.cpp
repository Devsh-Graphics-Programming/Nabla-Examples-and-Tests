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
	bool CSession::init(video::IGPUCommandBuffer* cb)
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
						logger.log("Failed to create Sensor \"%s\"'s \"%s\" in CSession::init()", ILogger::ELL_ERROR, m_params.name.c_str(), debugName.data());
						return false;
					}
					memBacked->setObjectDebugName(debugName.data());

					auto mreqs = memBacked->getMemoryReqs();
					mreqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
					if (!device->allocate(mreqs, memBacked, IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE).isValid())
					{
						logger.log("Could not allocate memory for Sensor \"%s\"'s \"%s\" in CSession::init()", ILogger::ELL_ERROR, m_params.name.c_str(), debugName.data());
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
				if (!dedicatedAllocate(ubo.get(), "Sensor UBO"))
					return false;
				// pipeline barrier in `reset` will take care of sync for this
				cb->updateBuffer({ .size = sizeof(m_params.uniforms),.buffer = ubo }, &m_params.uniforms);
				addWrite(SensorDSBindings::UBO, SBufferRange<IGPUBuffer>{.offset = 0, .size = sizeof(m_params.uniforms), .buffer = ubo});
			}

			const auto allowedFormatUsages = device->getPhysicalDevice()->getImageFormatUsagesOptimalTiling();
			auto createImage = [&](
				const std::string_view debugName, const E_FORMAT format, const uint16_t2 resolution, const uint16_t layers, std::bitset<E_FORMAT::EF_COUNT> viewFormats = {},
				const IGPUImage::E_USAGE_FLAGS extraUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT | IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT
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
							using image_usage_e = IGPUImage::E_USAGE_FLAGS;
							params.usage = image_usage_e::EUF_TRANSFER_DST_BIT | extraUsages;
							if (m_params.type == sensor_type_e::Env)
							{
								params.arrayLayers *= 6;
								params.flags |= IGPUImage::E_CREATE_FLAGS::ECF_CUBE_COMPATIBLE_BIT;
							}
							viewFormats.set(format);
							if (viewFormats.count() > 1)
							{
								params.flags |= IGPUImage::E_CREATE_FLAGS::ECF_MUTABLE_FORMAT_BIT;
								params.flags |= IGPUImage::E_CREATE_FLAGS::ECF_EXTENDED_USAGE_BIT;
							}
							params.viewFormats = viewFormats;
							retval.image = device->createImage(std::move(params));
							if (!dedicatedAllocate(retval.image.get(), debugName))
								return {};
						}
						const auto& params = retval.image->getCreationParameters();
						for (uint8_t f = 0; f < viewFormats.size(); f++)
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
								string viewDebugName = string(debugName) + " " + to_string(viewFormat) + " View";
								if (!view)
								{
									logger.log("Failed to create Sensor \"%s\"'s \"%s\" in CSession::init()", ILogger::ELL_ERROR, m_params.name.c_str(), viewDebugName.c_str());
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
					addWrite(binding, std::move(info));
				};
			immutables.scrambleKey = createImage("Scramble Key", E_FORMAT::EF_R32G32_UINT, promote<uint16_t2>(SSensorUniforms::ScrambleKeyTextureSize), 1);
			auto scrambleKeyView = immutables.scrambleKey.views[E_FORMAT::EF_R32G32_UINT];
			addImageWrite(SensorDSBindings::ScrambleKey, scrambleKeyView);

			// create the render-sized images
			auto createScreenSizedImage = [&]<typename... Args>(const std::string_view debugName, const E_FORMAT format, Args&&... args)->SImageWithViews
			{
				return createImage(debugName, format, m_params.uniforms.renderSize, std::forward<Args>(args)...);
			};
			immutables.sampleCount = createScreenSizedImage("Current Sample Count", E_FORMAT::EF_R16_UINT, 1);
			auto sampleCountView = immutables.sampleCount.views[E_FORMAT::EF_R16_UINT];
			addImageWrite(SensorDSBindings::SampleCount, sampleCountView);
			immutables.rwmcCascades = createScreenSizedImage("RWMC Cascades", E_FORMAT::EF_R32G32_UINT, m_params.uniforms.lastCascadeIndex + 1);
			auto rwmcCascadesView = immutables.rwmcCascades.views[E_FORMAT::EF_R32G32_UINT];
			addImageWrite(SensorDSBindings::RWMCCascades, rwmcCascadesView);
			immutables.beauty = createScreenSizedImage("Beauty", E_FORMAT::EF_E5B9G9R9_UFLOAT_PACK32, 1, std::bitset<E_FORMAT::EF_COUNT>().set(E_FORMAT::EF_R32_UINT));
			addImageWrite(SensorDSBindings::Beauty, immutables.beauty.views[E_FORMAT::EF_R32_UINT]);
			immutables.albedo = createScreenSizedImage("Albedo", E_FORMAT::EF_A2B10G10R10_UNORM_PACK32, 1);
			auto albedoView = immutables.albedo.views[E_FORMAT::EF_A2B10G10R10_UNORM_PACK32];
			addImageWrite(SensorDSBindings::Albedo, albedoView);
			// Normal and Albedo should have used `EF_A2B10G10R10_SNORM_PACK32` but Nvidia doesn't support
			immutables.normal = createScreenSizedImage("Normal", E_FORMAT::EF_A2B10G10R10_UNORM_PACK32, 1);
			auto normalView = immutables.normal.views[E_FORMAT::EF_A2B10G10R10_UNORM_PACK32];
			addImageWrite(SensorDSBindings::Normal, normalView);
			immutables.motion = createScreenSizedImage("Motion", E_FORMAT::EF_A2B10G10R10_UNORM_PACK32, 1);
			auto motionView = immutables.motion.views[E_FORMAT::EF_A2B10G10R10_UNORM_PACK32];
			addImageWrite(SensorDSBindings::Motion, motionView);
			immutables.mask = createScreenSizedImage("Mask", E_FORMAT::EF_R16_UNORM, 1);
			auto maskView = immutables.mask.views[E_FORMAT::EF_R16_UNORM];
			addImageWrite(SensorDSBindings::Mask, maskView);
			// shorthand a little bit
			addImageWrite(SensorDSBindings::AsSampledImages, scrambleKeyView);
			writes.back().count = SensorDSBindingCounts::AsSampledImages;
			{
				const auto oldSize = infos.size();
				infos.resize(oldSize + SensorDSBindingCounts::AsSampledImages, infos.back());
				const auto viewInfos = infos.data() + oldSize - 1;
				using index_e = SensorDSBindings::SampledImageIndex;
				viewInfos[uint8_t(index_e::ScrambleKey)].desc = scrambleKeyView;
				viewInfos[uint8_t(index_e::SampleCount)].desc = sampleCountView;
				viewInfos[uint8_t(index_e::RWMCCascades)].desc = rwmcCascadesView;
				viewInfos[uint8_t(index_e::Beauty)].desc = immutables.beauty.views[E_FORMAT::EF_E5B9G9R9_UFLOAT_PACK32];
				viewInfos[uint8_t(index_e::Albedo)].desc = albedoView;
				viewInfos[uint8_t(index_e::Normal)].desc = normalView;
				viewInfos[uint8_t(index_e::Motion)].desc = motionView;
				viewInfos[uint8_t(index_e::Mask)].desc = maskView;
			}
		}

		// create descriptor set
		{
			auto layout = renderer->getConstructionParams().sensorDSLayout;
			auto pool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT, { &layout.get(),1 });
			immutables.ds = pool->createDescriptorSet(std::move(layout));
			const char* DebugName = "Sensor Descriptor Set";
			if (!immutables.ds)
			{
				logger.log("Failed to create Sensor \"%s\"'s \"%s\" in CSession::init()", ILogger::ELL_ERROR, m_params.name.c_str(), DebugName);
				return false;
			}
			immutables.ds->setObjectDebugName(DebugName);
			for (auto& write : writes)
			{
				write.dstSet = immutables.ds.get();
				write.info = infos.data() + reinterpret_cast<const uint64_t&>(write.info);
			}
			if (!device->updateDescriptorSets(writes, {}))
			{
				logger.log("Failed to write Sensor \"%s\"'s \"%s\" in CSession::init()", ILogger::ELL_ERROR, m_params.name.c_str(), DebugName);
				return false;
			}
		}

		if (!immutables || !reset(m_params.initDynamics, cb))
		{
			logger.log("Could not Init Session for sensor \"%s\" failed to reset!", ILogger::ELL_ERROR, m_params.name.c_str());
			deinit();
			return false;
		}

		// TODO: fill scramble Key with noise

		return true;
	}

	bool CSession::reset(const SSensorDynamics& newVal, IGPUCommandBuffer* cb)
	{
		if (!isInitialized())
			return false;

		auto* const renderer = m_params.scene->getRenderer();
		auto* const device = renderer->getDevice();
		const auto& immutables = m_active.immutables;

		bool success = true;
		// slam the barriers as big as possible, it wont happen frequently
		using image_barrier_t = IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t;
		core::vector<image_barrier_t> before;
		{
			constexpr image_barrier_t beforeBase = {
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
						.srcAccessMask = ACCESS_FLAGS::NONE, // because we don't care about reading previously written values
						.dstStageMask = PIPELINE_STAGE_FLAGS::CLEAR_BIT,
						.dstAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS
					}
				},
				.subresourceRange = {},
				.newLayout = IGPUImage::LAYOUT::GENERAL
			};
			before.reserve(SensorDSBindingCounts::AsSampledImages);

			auto enqueueClear = [&before, beforeBase](const SImageWithViews& img)->void
				{
					auto& out = before.emplace_back(beforeBase);
					out.image = img.image.get();
					out.subresourceRange = {
						.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
						.levelCount = 1,
						.layerCount = out.image->getCreationParameters().arrayLayers
					};
				};
			enqueueClear(immutables.sampleCount);
			enqueueClear(immutables.beauty);
			enqueueClear(immutables.rwmcCascades);
			enqueueClear(immutables.albedo);
			enqueueClear(immutables.normal);
			enqueueClear(immutables.motion);
			enqueueClear(immutables.mask);
			success = success && cb->pipelineBarrier(asset::EDF_NONE, { .imgBarriers = before });
		}

		{
			IGPUCommandBuffer::SClearColorValue color;
			memset(&color, 0, sizeof(color));
			for (const auto& entry : before)
			{
				success = success && cb->clearColorImage(const_cast<IGPUImage*>(entry.image), IGPUImage::LAYOUT::GENERAL, &color, 1, &entry.subresourceRange);
			}
		}

		const SMemoryBarrier after[] = {
			{
				.srcStageMask = PIPELINE_STAGE_FLAGS::CLEAR_BIT,
				.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS,
				.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
				.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS | ACCESS_FLAGS::SHADER_WRITE_BITS
			}
		};
		success = success && cb->pipelineBarrier(asset::EDF_NONE, { .memBarriers = after });

		if (success)
			m_active.prevSensorState = m_active.currentSensorState = newVal;
		return success;
	}

	bool CSession::update(const SSensorDynamics& newVal)
	{
		if (!isInitialized())
			return false;

		m_active.prevSensorState = m_active.currentSensorState;
		m_active.currentSensorState = newVal;
		return true;
	}

}