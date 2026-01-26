// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "renderer/CRenderer.h"
#include "renderer/SAASequence.h"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include <array>
#include <thread>
#include <future>
#include <filesystem>


namespace nbl::this_example
{
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::system;
using namespace nbl::video;


smart_refctd_ptr<IShader> CRenderer::loadPrecompiledShader_impl(IAssetManager* assMan, const core::string& key, logger_opt_ptr logger)
{
	IAssetLoader::SAssetLoadParams lp = {};
	lp.logger = logger;
	lp.workingDirectory = "app_resources"; // virtual root
	auto assetBundle = assMan->getAsset(key,lp);
	const auto assets = assetBundle.getContents();
	if (!assets.empty())
	if (auto shader = IAsset::castDown<IShader>(*assets.begin()); shader)
		return shader;

	logger.log("Failed to load precompiled shader %s", ILogger::ELL_ERROR, key.c_str());
	return nullptr;
}

//
smart_refctd_ptr<CRenderer> CRenderer::create(SCreationParams&& _params)
{
	if (!_params)
		return nullptr;
	SConstructorParams params = {std::move(_params)};

	//
	if (!params.logger.get())
		params.logger = smart_refctd_ptr<ILogger>(params.utilities->getLogger());
	logger_opt_ptr logger = params.logger.get().get();

	//
	auto checkNullObject = [&params,logger](auto& obj, const std::string_view debugName)->bool
	{
		if (!obj)
		{
			logger.log("Failed to Create %s Object!",ILogger::ELL_ERROR,debugName.data());
			return true;
		}
		obj->setObjectDebugName(debugName.data());
		return false;
	};

	//
	ILogicalDevice* device = params.utilities->getLogicalDevice();
	// limits

	//
	params.semaphore = device->createSemaphore(0);
	if (checkNullObject(params.semaphore,"CRenderer Semaphore"))
		return nullptr;

	// basic samplers
	const auto samplerDefaultRepeat = device->createSampler({});

	using render_mode_e = CSession::RenderMode;
	// create the layouts
	smart_refctd_ptr<IGPUPipelineLayout> renderingLayouts[uint8_t(CSession::RenderMode::Count)];
	{
		constexpr auto RTStages = hlsl::ShaderStage::ESS_ALL_RAY_TRACING;// | hlsl::ShaderStage::ESS_COMPUTE;
		constexpr auto RenderingStages = RTStages | hlsl::ShaderStage::ESS_COMPUTE;
		// descriptor
		{
			using binding_create_flags_t = IDescriptorSetLayoutBase::SBindingBase::E_CREATE_FLAGS;
			constexpr IGPUDescriptorSetLayout::SBinding UBOBinding = {
				.binding = SensorDSBindings::UBO,
				.type = IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
				.createFlags = binding_create_flags_t::ECF_NONE,
				.stageFlags = RenderingStages,
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
					.createFlags = binding_create_flags_t::ECF_PARTIALLY_BOUND_BIT,
					.stageFlags = RenderingStages,
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
				constexpr auto ResolveAndPresentStages = hlsl::ShaderStage::ESS_COMPUTE | hlsl::ShaderStage::ESS_FRAGMENT;
				const auto defaultSampler = device->createSampler({
					{
						.AnisotropicFilter = 0
					},
					0.f,
					0.f,
					0.f
				});
				std::initializer_list<const IGPUDescriptorSetLayout::SBinding> bindings = {
					UBOBinding,
					singleStorageImage(SensorDSBindings::ScrambleKey),
					singleStorageImage(SensorDSBindings::SampleCount),
					singleStorageImage(SensorDSBindings::Beauty),
					singleStorageImage(SensorDSBindings::RWMCCascades),
					singleStorageImage(SensorDSBindings::Albedo),
					singleStorageImage(SensorDSBindings::Normal),
					singleStorageImage(SensorDSBindings::Motion),
					singleStorageImage(SensorDSBindings::Mask),
					{
						.binding = SensorDSBindings::Samplers,
						.type = IDescriptor::E_TYPE::ET_SAMPLER,
						.createFlags = binding_create_flags_t::ECF_NONE,
						.stageFlags = ResolveAndPresentStages,
						.count = SensorDSBindingCounts::Samplers,
						.immutableSamplers = &defaultSampler
					},
					{
						.binding = SensorDSBindings::AsSampledImages,
						.type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
						.createFlags = binding_create_flags_t::ECF_PARTIALLY_BOUND_BIT,
						.stageFlags = ResolveAndPresentStages,
						.count = SensorDSBindingCounts::AsSampledImages
					}
				};
				params.sensorDSLayout = device->createDescriptorSetLayout(bindings);
				if (checkNullObject(params.sensorDSLayout,"Sensor Descriptor Layout"))
					return nullptr;
			}
		}

		// but many push constant ranges
		SPushConstantRange pcRanges[uint8_t(render_mode_e::Count)];
		auto setPCRange = [&pcRanges]<typename T>(const render_mode_e mode)->void
		{
			pcRanges[uint8_t(mode)] = {.stageFlags=RTStages,.offset=0,.size=sizeof(T)};
		};
		setPCRange.operator()<SPrevisPushConstants>(render_mode_e::Previs);
		setPCRange.operator()<SBeautyPushConstants>(render_mode_e::Beauty);
		setPCRange.operator()<SDebugPushConstants>(render_mode_e::Debug);
		for (uint8_t t=0; t<uint8_t(render_mode_e::Count); t++)
		{
			renderingLayouts[t] = device->createPipelineLayout({pcRanges+t,1},params.sceneDSLayout,params.sensorDSLayout);
			string debugName = to_string(static_cast<render_mode_e>(t))+"Rendering Pipeline Layout";
			if (checkNullObject(renderingLayouts[t],debugName))
				return nullptr;
		}
	}

	// create the pipelines
	{

		IGPURayTracingPipeline::SCreationParams creationParams[uint8_t(render_mode_e::Count)] = {};
		using creation_flags_e = IGPURayTracingPipeline::SCreationParams::FLAGS;
		auto flags = creation_flags_e::NO_NULL_MISS_SHADERS;
		{
			smart_refctd_ptr<IShader> raygenShaders[uint8_t(render_mode_e::Count)] = {};
			raygenShaders[uint8_t(render_mode_e::Previs)] = loadPrecompiledShader<"pathtrace_previs">(_params.assMan,device,logger);
			raygenShaders[uint8_t(render_mode_e::Beauty)] = loadPrecompiledShader<"pathtrace_beauty">(_params.assMan,device,logger);
			raygenShaders[uint8_t(render_mode_e::Debug)] = loadPrecompiledShader<"pathtrace_debug">(_params.assMan,device,logger);
			IGPURayTracingPipeline::SShaderSpecInfo missShaders[uint8_t(render_mode_e::Count)] = {};
			for (uint8_t m=0; m<uint8_t(render_mode_e::Count); m++)
			{
				missShaders[m] = {.shader=raygenShaders[m].get(),.entryPoint="miss"};
				creationParams[m] = {
					.layout = renderingLayouts[m].get(),
					.shaderGroups = {
						.raygen = {.shader=raygenShaders[m].get(),.entryPoint="raygen"},
						.misses = {missShaders+m,1}
					},
					.cached = {
						.flags = flags
					}
				};
			}
		}
		if (!device->createRayTracingPipelines(nullptr,creationParams,params.renderingPipelines.data()))
		{
			logger.log("Failed to create Path Tracing Pipelines",ILogger::ELL_ERROR);
			return nullptr;
		}
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


auto CRenderer::render(CSession* session) -> SSubmit
{
	if (!session || !session->isInitialized())
		return {};
	const auto& sessionParams = session->getConstructionParams();
	auto* const device = getDevice();

	if (m_frameIx>=SCachedConstructionParams::FramesInFlight)
	{
		const ISemaphore::SWaitInfo cbDonePending[] =
		{
			{
				.semaphore = m_construction.semaphore.get(),
				.value = m_frameIx+1-SCachedConstructionParams::FramesInFlight
			}
		};
		if (device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
			return {};
	}
	const auto resourceIx = m_frameIx % SCachedConstructionParams::FramesInFlight;

	auto* const cb = m_construction.commandBuffers[resourceIx].get();
	cb->getPool()->reset();
	if (!cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
		return {};

	const auto mode = sessionParams.mode;
	const auto& sessionResources = session->getActiveResources();
	const auto* const pipeline = m_construction.renderingPipelines[static_cast<uint8_t>(mode)].get();
	
	bool success;
	// push constants
	{
		switch (mode)
		{
			case CSession::RenderMode::Debug:
			{
				SDebugPushConstants pc = {sessionResources.currentSensorState};
				success = cb->pushConstants(pipeline->getLayout(),hlsl::ShaderStage::ESS_ALL_RAY_TRACING,0,sizeof(pc),&pc);
				break;
			}
			default:
				getLogger().log("Unimplemented RenderMode::%s !",ILogger::ELL_ERROR,system::to_string(mode).c_str());
				return {};
		}
	}
	// bind pipelines
	success = success && cb->bindRayTracingPipeline(pipeline);
	{
		const IGPUDescriptorSet* sets[2] = {sessionParams.scene->getDescriptorSet(),sessionResources.immutables.ds.get()};
		success = success && cb->bindDescriptorSets(EPBP_RAY_TRACING,pipeline->getLayout(),0,2,sets);
	}

	const auto renderSize = sessionParams.uniforms.renderSize;
	success = success && cb->traceRays({},{},0,{},0,{},0,renderSize.x,renderSize.y,sessionParams.type!=CSession::sensor_type_e::Env ? 1:6);

	if (success)
		return SSubmit(this,cb);
	else
		return {};
}

IQueue::SSubmitInfo::SSemaphoreInfo CRenderer::SSubmit::operator()(std::span<const video::IQueue::SSubmitInfo::SSemaphoreInfo> extraWaits)
{
	if (cb)
		return {};

	const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
	{
		{
			.semaphore = renderer->m_construction.semaphore.get(),
			.value = ++renderer->m_frameIx,
			.stageMask = stageMask
		}
	};
	const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] = {{.cmdbuf=cb}};
	const IQueue::SSubmitInfo infos[] =
	{
		{
			.waitSemaphores = extraWaits,
			.commandBuffers = commandBuffers,
			.signalSemaphores = rendered
		}
	};
	if (renderer->getCreationParams().graphicsQueue->submit(infos)!=IQueue::RESULT::SUCCESS)
	{
		renderer->m_frameIx--;
		return {};
	}
	return rendered[0];
}

}