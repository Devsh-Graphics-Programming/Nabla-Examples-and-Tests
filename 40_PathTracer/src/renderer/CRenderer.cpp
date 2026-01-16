// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "renderer/CRenderer.h"
#include "renderer/SAASequence.h"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include <array>

namespace nbl::this_example
{
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::system;
using namespace nbl::video;

//
smart_refctd_ptr<CRenderer> CRenderer::create(SCreationParams&& _params)
{
	if (!_params)
		return nullptr;
	SConstructorParams params = {std::move(_params)};

	//
	if (!params.logger.get())
		params.logger = smart_refctd_ptr<ILogger>(params.utilities->getLogger());
	auto checkNullObject = [&params](auto& obj, const std::string_view debugName)->bool
	{
		if (!obj)
		{
			params.logger.log("Failed to Create %s Object!",ILogger::ELL_ERROR,debugName.data());
			return true;
		}
		obj->setObjectDebugName(debugName.data());
		return false;
	};

	//
	ILogicalDevice* device = params.utilities->getLogicalDevice();
	// limits

	// basic samplers
	const auto samplerDefaultRepeat = device->createSampler({});

	// create the layouts
	smart_refctd_ptr<IGPUPipelineLayout> renderingLayouts[uint8_t(RenderMode::Count)];
	{
		constexpr auto RTStages = hlsl::ShaderStage::ESS_ALL_RAY_TRACING | hlsl::ShaderStage::ESS_COMPUTE;
		// descriptor
		{
			using binding_create_flags_t = IDescriptorSetLayoutBase::SBindingBase::E_CREATE_FLAGS;
			constexpr IGPUDescriptorSetLayout::SBinding UBOBinding = {
				.binding = SensorDSBindings::UBO,
				.type = IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
				.createFlags = binding_create_flags_t::ECF_NONE,
				.stageFlags = hlsl::ShaderStage::ESS_ALL_OR_LIBRARY,
				.count = 1
			};
			// the generic single-UBO
			{
				params.uboDSLayout = device->createDescriptorSetLayout({&UBOBinding,1});
				if (checkNullObject(params.uboDSLayout,"Generic Single UBO Layout"))
					return nullptr;
			}
			constexpr auto DescriptorIndexingFlags = binding_create_flags_t::ECF_UPDATE_AFTER_BIND_BIT | binding_create_flags_t::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT | binding_create_flags_t::ECF_PARTIALLY_BOUND_BIT;
			//
			auto singleStorageImage = [](const uint32_t binding)->IGPUDescriptorSetLayout::SBinding
			{
				return {
					.binding = binding,
					.type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					.createFlags = binding_create_flags_t::ECF_NONE,
					.stageFlags = RTStages,
					.count = 1
				};
			};
			// TODO: provide these two samplers from Envmap Importance sampling extension
			const auto samplerNearestRepeat = device->createSampler({
				{
					.MinFilter = ISampler::E_TEXTURE_FILTER::ETF_NEAREST,
					.MaxFilter = ISampler::E_TEXTURE_FILTER::ETF_NEAREST,
					.MipmapMode = ISampler::E_SAMPLER_MIPMAP_MODE::ESMM_NEAREST,
					.AnisotropicFilter = 0,
				},
				0.f,
				0.f,
				0.f
			});
			// bindless everything
			{
				// TODO: provide these two samplers from Envmap Importance sampling extension
				const auto samplerEnvmapPDF = samplerNearestRepeat;
				const auto samplerEnvmapWarpmap = device->createSampler({
					{
						.MinFilter = ISampler::E_TEXTURE_FILTER::ETF_LINEAR,
						.MaxFilter = ISampler::E_TEXTURE_FILTER::ETF_LINEAR,
						.MipmapMode = ISampler::E_SAMPLER_MIPMAP_MODE::ESMM_NEAREST,
						.AnisotropicFilter = 0,
					},
					0.f,
					0.f,
					0.f
				});
				std::initializer_list<const IGPUDescriptorSetLayout::SBinding> bindings = {
					UBOBinding,
					{
						.binding = SceneDSBindings::Envmap,
						.type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
						.createFlags = binding_create_flags_t::ECF_NONE,
						.stageFlags = RTStages,
						.count = 1,
						.immutableSamplers = &samplerDefaultRepeat
					},
					{
						.binding = SceneDSBindings::TLASes,
						.type = IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE,
						.createFlags = DescriptorIndexingFlags,
						.stageFlags = RTStages,
						.count = SceneDSBindingCounts::TLASes
					},
					{
						.binding = SceneDSBindings::Samplers,
						.type = IDescriptor::E_TYPE::ET_SAMPLER,
						.createFlags = DescriptorIndexingFlags,
						.stageFlags = RTStages,
						.count = SceneDSBindingCounts::Samplers
					},
					{
						.binding = SceneDSBindings::SampledImages,
						.type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
						.createFlags = DescriptorIndexingFlags,
						.stageFlags = RTStages,
						.count = SceneDSBindingCounts::SampledImages
					},
					{
						.binding = SceneDSBindings::EnvmapPDF,
						.type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
						.createFlags = DescriptorIndexingFlags,
						.stageFlags = RTStages,
						.count = 1,
						.immutableSamplers = &samplerEnvmapPDF
					},
					{
						.binding = SceneDSBindings::EnvmapWarpMap,
						.type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
						.createFlags = DescriptorIndexingFlags,
						.stageFlags = RTStages,
						.count = 1,
						.immutableSamplers = &samplerEnvmapWarpmap
					}
				};
				params.sceneDSLayout = device->createDescriptorSetLayout(bindings);
				if (checkNullObject(params.sceneDSLayout,"Scene Descriptor Layout"))
					return nullptr;
			}
			// the sensor layout
			{
				std::initializer_list<const IGPUDescriptorSetLayout::SBinding> bindings = {
					UBOBinding,
					singleStorageImage(SensorDSBindings::ScrambleKey),
					singleStorageImage(SensorDSBindings::SampleCount),
					singleStorageImage(SensorDSBindings::RWMCCascades),
					singleStorageImage(SensorDSBindings::Albedo),
					singleStorageImage(SensorDSBindings::Normal),
					singleStorageImage(SensorDSBindings::Motion),
					singleStorageImage(SensorDSBindings::Mask)
				};
				params.sensorDSLayout = device->createDescriptorSetLayout(bindings);
				if (checkNullObject(params.sensorDSLayout,"Sensor Descriptor Layout"))
					return nullptr;
			}
		}

		// but many push constant ranges
		SPushConstantRange pcRanges[uint8_t(RenderMode::Count)];
		auto setPCRange = [&pcRanges]<typename T>(const RenderMode mode)->void
		{
			pcRanges[uint8_t(mode)] = {.stageFlags=RTStages,.offset=0,.size=sizeof(T)};
		};
		setPCRange.operator()<SPrevisPushConstants>(RenderMode::Previs);
		setPCRange.operator()<SBeautyPushConstants>(RenderMode::Beauty);
		setPCRange.operator()<SDebugPushConstants>(RenderMode::DebugIDs);
		for (uint8_t t=0; t<uint8_t(RenderMode::Count); t++)
		{
			renderingLayouts[t] = device->createPipelineLayout({pcRanges+t,1},params.sceneDSLayout,params.sensorDSLayout);
			string debugName = to_string(static_cast<RenderMode>(t))+"Rendering Pipeline Layout";
			if (checkNullObject(renderingLayouts[t],debugName))
				return nullptr;
		}
	}

	// create the pipelines
	{
		// TODO
	}

	// the renderpass: custom dependencies, but everything else fixed from outside (format, and number of subpasses)
	{
//		params.presentRenderpass = device->createRenderpass();
	}

	// present pipelines
	{
		// TODO
	}

	// command buffers
	for (uint8_t i=0; i<SConstructorParams::FramesInFlight; i++)
	{
		auto pool = device->createCommandPool(params.graphicsQueue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::NONE);
		if (pool)
			pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,1,params.commandBuffers+i,smart_refctd_ptr(params.logger.get()));
		if (checkNullObject(params.commandBuffers[i],"Graphics Command Buffer "+to_string(i)))
			return nullptr;
	}

	return core::smart_refctd_ptr<CRenderer>(new CRenderer(std::move(params)),core::dont_grab);
}


core::smart_refctd_ptr<CScene> CRenderer::createScene(CScene::SCreationParams&& _params)
{
	if (!_params)
		return nullptr;

	auto* const device = getDevice();
	auto converter = core::smart_refctd_ptr<CAssetConverter>(_params.converter);

	CScene::SConstructorParams params = {std::move(_params)};
	params.sensors = std::move(_params.load.sensors);
	params.renderer = smart_refctd_ptr<CRenderer>(this);
	{
		auto pool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT,{&m_construction.sceneDSLayout.get(),1});
		auto ds = pool->createDescriptorSet(smart_refctd_ptr(m_construction.sceneDSLayout));
		if (!ds)
		{
			m_creation.logger.log("Failed to create a scene - failed descriptor set allocation!",ILogger::ELL_ERROR);
			return nullptr;
		}
		params.sceneDS = make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(ds));
	}
	
	// new cache if none provided
	if (!converter)
		converter = CAssetConverter::create({.device=device,.optimizer={}});
	
	//
//	converter->reserve();
	// build the BLAS and TLAS
	{
		// TODO
	}
	core::smart_refctd_ptr<IGPUBuffer> ubo;

	// write into DS
	{
		vector<IGPUDescriptorSet::SDescriptorInfo> infos;
		vector<IGPUDescriptorSet::SWriteDescriptorSet> writes;
		auto* const ds = params.sceneDS->getDescriptorSet();
		auto addWrite = [&](const uint32_t binding, IGPUDescriptorSet::SDescriptorInfo&& info)->void
		{
			writes.emplace_back() = {
				.dstSet = ds,
				.binding = binding,
				.arrayElement = 0,
				.count = 1,
				.info = reinterpret_cast<const IGPUDescriptorSet::SDescriptorInfo*>(infos.size())
			};
			infos.push_back(std::move(info));
		};
		addWrite(SceneDSBindings::UBO,SBufferRange<IGPUBuffer>{.offset=0,.size=sizeof(SSceneUniforms),.buffer=ubo});
		// TODO: Envmap
		// TODO: TLASes
		// TODO: Samplers
		// TODO: Sampled Images
		// TODO: Envmap PDF
		// TODO: Envmap Warp Map
		for (auto& write : writes)
			write.info = infos.data()+reinterpret_cast<const uint64_t&>(write.info);
//		device->updateDescriptorSets(writes,{});
	}

#if 0
	float m_maxAreaLightLuma;
	// Resources used for envmap sampling
	nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_finalEnvmap;
#endif

	//
	if (!params)
	{
		m_creation.logger.log("Failed to create a scene!",ILogger::ELL_ERROR);
		return nullptr;
	}
	return core::smart_refctd_ptr<CScene>(new CScene(std::move(params)),core::dont_grab);
}

}