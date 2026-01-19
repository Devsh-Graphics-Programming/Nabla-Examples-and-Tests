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
				logger.log("Failed to create Sensor \"%s\"'s \"%s\" in CSession::init()",ILogger::ELL_ERROR,m_params.name.c_str(),debugName.data());
				return false;
			}
			memBacked->setObjectDebugName(debugName.data());

			auto mreqs = memBacked->getMemoryReqs();
			mreqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			if (!device->allocate(mreqs,memBacked,IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE).isValid())
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
			params.usage = IGPUBuffer::E_USAGE_FLAGS::EUF_UNIFORM_BUFFER_BIT | IGPUBuffer::E_USAGE_FLAGS::EUF_INLINE_UPDATE_VIA_CMDBUF;
			auto ubo = device->createBuffer(std::move(params));
			if (!dedicatedAllocate(ubo.get(),"Sensor UBO"))
				return false;
			addWrite(SensorDSBindings::UBO,SBufferRange<IGPUBuffer>{.offset=0,.size=sizeof(m_params.uniforms),.buffer=ubo});
		}

		const auto allowedFormatUsages = device->getPhysicalDevice()->getImageFormatUsagesOptimalTiling();
		auto createImage = [&](
			const std::string_view debugName, const E_FORMAT format, const uint16_t2 resolution, const uint16_t layers, std::bitset<E_FORMAT::EF_COUNT> viewFormats={},
			const IGPUImage::E_USAGE_FLAGS extraUsages=IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT
		) -> SActiveResources::SImageWithViews
		{
			SActiveResources::SImageWithViews retval = {};
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
					params.usage = image_usage_e::EUF_TRANSFER_DST_BIT|extraUsages;
					if (m_params.type==sensor_type_e::Env)
					{
						params.arrayLayers *= 6;
						params.flags |= IGPUImage::E_CREATE_FLAGS::ECF_CUBE_COMPATIBLE_BIT;
					}
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
		immutables.scrambleKey = createImage("Scramble Key",E_FORMAT::EF_R32G32_UINT,promote<uint16_t2>(SSensorUniforms::ScrambleKeyTextureSize),1);
		addImageWrite(SensorDSBindings::ScrambleKey,immutables.scrambleKey.views[E_FORMAT::EF_R32G32_UINT]);

		// create the render-sized images
		auto createScreenSizedImage = [&]<typename... Args>(const std::string_view debugName, const E_FORMAT format, Args&&... args)->SActiveResources::SImageWithViews
		{
			return createImage(debugName,format,m_params.uniforms.renderSize,std::forward<Args>(args)...);
		};
		immutables.sampleCount = createScreenSizedImage("Current Sample Count",E_FORMAT::EF_R16_UINT,1);
		addImageWrite(SensorDSBindings::SampleCount,immutables.sampleCount.views[E_FORMAT::EF_R16_UINT]);
		immutables.beauty = createScreenSizedImage("Beauty",E_FORMAT::EF_E5B9G9R9_UFLOAT_PACK32,1,std::bitset<E_FORMAT::EF_COUNT>().set(E_FORMAT::EF_R32_UINT));
		addImageWrite(SensorDSBindings::Beauty,immutables.beauty.views[E_FORMAT::EF_R32_UINT]);
		immutables.rwmcCascades = createScreenSizedImage("RWMC Cascades",E_FORMAT::EF_R32G32_UINT,m_params.uniforms.lastCascadeIndex+1);
		addImageWrite(SensorDSBindings::RWMCCascades,immutables.rwmcCascades.views[E_FORMAT::EF_R32G32_UINT]);
		immutables.albedo = createScreenSizedImage("Albedo",E_FORMAT::EF_A2B10G10R10_UNORM_PACK32,1);
		addImageWrite(SensorDSBindings::Albedo,immutables.albedo.views[E_FORMAT::EF_A2B10G10R10_UNORM_PACK32]);
		// Normal and Albedo should have used `EF_A2B10G10R10_SNORM_PACK32` but Nvidia doesn't support
		immutables.normal = createScreenSizedImage("Normal",E_FORMAT::EF_A2B10G10R10_UNORM_PACK32,1);
		addImageWrite(SensorDSBindings::Normal,immutables.normal.views[E_FORMAT::EF_A2B10G10R10_UNORM_PACK32]);
		immutables.motion = createScreenSizedImage("Motion",E_FORMAT::EF_A2B10G10R10_UNORM_PACK32,1);
		addImageWrite(SensorDSBindings::Motion,immutables.motion.views[E_FORMAT::EF_A2B10G10R10_UNORM_PACK32]);
		immutables.mask = createScreenSizedImage("Mask",E_FORMAT::EF_R16_UNORM,1);
		addImageWrite(SensorDSBindings::Mask,immutables.mask.views[E_FORMAT::EF_R16_UNORM]);
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

	if (!immutables || !reset(m_params.initDynamics,cb))
	{
		logger.log("Could not Init Session for sensor \"%s\" failed to reset!",ILogger::ELL_ERROR,m_params.name.c_str());
		deinit();
		return false;
	}

// TODO: fill scramble Key with noise

	return true;
}

bool CSession::reset(const SSensorDynamics& newVal, IGPUCommandBuffer* cb)
{
	auto* const renderer = m_params.scene->getRenderer();
	auto* const device = renderer->getDevice();
	const auto& immutables = m_active.immutables;

	// slam the barriers as big as possible, it wont happen frequently
	bool success = true;
	const SMemoryBarrier before[] = {
		{
			.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
			.srcAccessMask = ACCESS_FLAGS::NONE, // because we don't care about reading previously written values
			.dstStageMask = PIPELINE_STAGE_FLAGS::CLEAR_BIT,
			.dstAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS
		}
	};
	success = success && cb->pipelineBarrier(asset::EDF_NONE,{.memBarriers=before});
	auto clearImage = [cb,&success](const SActiveResources::SImageWithViews& img)->void
	{
		const IGPUImage::SSubresourceRange subresRng = {
			.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
			.levelCount = 1,
			.layerCount = img.image->getCreationParameters().arrayLayers
		};
		IGPUCommandBuffer::SClearColorValue color;
		memset(&color,0,sizeof(color));
		success = success && cb->clearColorImage(img.image.get(),IGPUImage::LAYOUT::GENERAL,&color,1,&subresRng);
	};
	clearImage(immutables.sampleCount);
	clearImage(immutables.beauty);
	clearImage(immutables.rwmcCascades);
	clearImage(immutables.albedo);
	clearImage(immutables.normal);
	clearImage(immutables.motion);
	clearImage(immutables.mask);
	const SMemoryBarrier after[] = {
		{
			.srcStageMask = PIPELINE_STAGE_FLAGS::CLEAR_BIT,
			.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS,
			.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
			.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS|ACCESS_FLAGS::SHADER_WRITE_BITS
		}
	};
	success = success && cb->pipelineBarrier(asset::EDF_NONE,{.memBarriers=after});

	if (success)
		m_active.prevSensorState = newVal;
	return success;
}

}