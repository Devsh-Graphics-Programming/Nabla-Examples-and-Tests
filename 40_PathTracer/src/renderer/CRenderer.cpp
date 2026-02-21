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

	//
	params.semaphore = device->createSemaphore(0);
	if (checkNullObject(params.semaphore,"CRenderer Semaphore"))
		return nullptr;

	// basic samplers
	const auto samplerDefaultRepeat = device->createSampler({});

	using render_mode_e = CSession::RenderMode;
	// create the layouts
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
			params.renderingLayouts[t] = device->createPipelineLayout({pcRanges+t,1},params.sceneDSLayout,params.sensorDSLayout);
			string debugName = to_string(static_cast<render_mode_e>(t))+"Rendering Pipeline Layout";
			if (checkNullObject(params.renderingLayouts[t],debugName))
				return nullptr;
		}
	}

	// TODO: create the generic pipelines
	params.shaders[uint8_t(render_mode_e::Previs)] = loadPrecompiledShader<"pathtrace_previs">(_params.assMan,device,logger);
	params.shaders[uint8_t(render_mode_e::Beauty)] = loadPrecompiledShader<"pathtrace_beauty">(_params.assMan,device,logger);
	params.shaders[uint8_t(render_mode_e::Debug)] = loadPrecompiledShader<"pathtrace_debug">(_params.assMan,device,logger);
	for (auto i=0; i<params.shaders.size(); i++)
	if (!params.shaders[i])
	{
		logger.log("Failed to Load %s Shader!",ILogger::ELL_ERROR,system::to_string(static_cast<render_mode_e>(i)));
		return nullptr;
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
//	params.sceneBound = ;
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

	auto* const cpuScene = _params.load.scene.get();

	constexpr auto RenderModeCount = uint8_t(CSession::RenderMode::Count);
	// create the pipelines
	{
		IGPURayTracingPipeline::SCreationParams creationParams[RenderModeCount] = {};
		using creation_flags_e = IGPURayTracingPipeline::SCreationParams::FLAGS;
		auto flags = creation_flags_e::NO_NULL_MISS_SHADERS;
		IGPURayTracingPipeline::SShaderSpecInfo missShaders[RenderModeCount] = {};
		{
			for (uint8_t m=0; m<RenderModeCount; m++)
			{
				const auto* const shader = m_construction.shaders[m].get();
				missShaders[m] = {.shader=shader,.entryPoint="miss"};
				creationParams[m] = {
					.layout = m_construction.renderingLayouts[m].get(),
					.shaderGroups = {
						.raygen = {.shader=shader,.entryPoint="raygen"},
						.misses = {missShaders+m,1}
						// TODO: use Material Compiler to get callables for us
					},
					.cached = {
						.flags = flags
					}
				};
			}
		}
		if (!device->createRayTracingPipelines(nullptr,creationParams,params.pipelines))
		{
			m_creation.logger.log("Failed to create Path Tracing Pipelines",ILogger::ELL_ERROR);
			return nullptr;
		}
	}

	// TODO: make this configurable
	constexpr bool movableInstances = false;

	smart_refctd_ptr<IGPUBuffer> ubo;
	core::vector<asset_cached_t<ICPUTopLevelAccelerationStructure>> TLASes;
	{
		// construct the TLASes
		core::vector<smart_refctd_ptr<ICPUTopLevelAccelerationStructure>> tmpTLASes;
		// main TLAS
		{
			using tlas_build_f = ICPUTopLevelAccelerationStructure::BUILD_FLAGS;
			constexpr auto baseTLASFlags = tlas_build_f::PREFER_FAST_TRACE_BIT | tlas_build_f::ALLOW_COMPACTION_BIT;

			auto& main = tmpTLASes.emplace_back();
			main = make_smart_refctd_ptr<ICPUTopLevelAccelerationStructure>();
			{
				auto& cpuInstances = cpuScene->getInstances();
				// need to convert weird topologies to lists
				for (auto i=0u; i<cpuInstances.size(); i++)
				if (const auto* const targets=cpuInstances.getMorphTargets()[i].get(); targets)
				for (auto& target : targets->getTargets())
				{
					auto* const collection = target.geoCollection.get();
					for (auto& ref : *collection->getGeometries())
					{
                        const auto* geo = ref.geometry.get();
                        if (geo->getPrimitiveType()!=IGeometryBase::EPrimitiveType::Polygon)
                            continue;
						// TODO: test without and see why asset converter complains!
						ref.geometry = CPolygonGeometryManipulator::createTriangleListIndexing(static_cast<const ICPUPolygonGeometry*>(geo));
					}
				}
				ICPUScene::CDefaultTLASExporter exporter(cpuInstances);
				auto exported = exporter();
				if (!exported)
				{
					m_creation.logger.log("Failed to convert TLAS instances!",ILogger::ELL_ERROR);
					return nullptr;
				}
				if (!exported.allInstancesValid)
					m_creation.logger.log("Some instances in the scene are invisible!",ILogger::ELL_ERROR);
				for (auto& pair : exporter.m_blasCache)
				{
					using blas_build_f = ICPUBottomLevelAccelerationStructure::BUILD_FLAGS;
					auto flags = pair.second->getBuildFlags();
					flags |= blas_build_f::ALLOW_DATA_ACCESS;
					pair.second->setBuildFlags(flags);
				}
				for (auto& instance : *exported.instances)
				{
					// TODO: for now, we need material compiler and knowing how we'll orgarnise our SBT
					instance.getBase().instanceShaderBindingTableRecordOffset = 0;
				}
				main->setInstances(std::move(exported.instances));
			}
			// de-instancing and welding BLASes
			{
				// TODO: de-instancing step, need AS memory budget and a heuristic of which instances to eliminate (out of the ones we don't want to be movable)
				// probably need OOBs of Geometry Collections, best heuristic would be "total surface of all intersections with other instances" divided by build memory.
				// In the meantime can do own OBB/AABB surface area divded by build size.
				// NOTE: can only "weld" BLASes which have the same build flags
			}
			{
				auto flags = baseTLASFlags;
				if (movableInstances)
					flags |= tlas_build_f::ALLOW_UPDATE_BIT;
				main->setBuildFlags(flags);
			}
		}

		struct Buffers final
		{
			using render_mode_e = CSession::RenderMode;
			inline operator std::span<const ICPUBuffer* const>() const {return {&ubo.get(),1+RenderModeCount};}

			smart_refctd_ptr<ICPUBuffer> ubo;
			smart_refctd_ptr<ICPUBuffer> sbts[RenderModeCount];
		} tmpBuffers;
		//
		using buffer_usage_e = IGPUBuffer::E_USAGE_FLAGS;
		constexpr auto BasicBufferUsages = buffer_usage_e::EUF_SHADER_DEVICE_ADDRESS_BIT;
		{
			tmpBuffers.ubo = ICPUBuffer::create({{.size=sizeof(SSceneUniforms),.usage=BasicBufferUsages|buffer_usage_e::EUF_UNIFORM_BUFFER_BIT},nullptr});
			auto& uniforms = *reinterpret_cast<SSceneUniforms*>(tmpBuffers.ubo->getPointer());
			uniforms.init = {}; // TODO: fill with stuff
			tmpBuffers.ubo->setContentHash(tmpBuffers.ubo->computeContentHash());
		}
		// SBT
		const auto& limits = device->getPhysicalDevice()->getLimits();
		assert(limits.shaderGroupBaseAlignment>=limits.shaderGroupHandleAlignment);
		constexpr auto HandleSize = SPhysicalDeviceLimits::ShaderGroupHandleSize;
		const auto handleSizeAligned = nbl::core::alignUp(HandleSize,limits.shaderGroupHandleAlignment);
		for (uint8_t i=0; i<RenderModeCount; i++)
		{
			auto* const pipeline = params.pipelines[i].get();
			const auto hitHandles = pipeline->getHitHandles();
			const auto missHandles = pipeline->getMissHandles();
			const auto callableHandles = pipeline->getCallableHandles();
			//
			{
				class CVectorBacked final : public core::refctd_memory_resource
				{
					public:
						inline CVectorBacked(const size_t reservation)
						{
							storage.reserve(reservation*HandleSize);
						}

						inline void* allocate(size_t bytes, size_t alignment) override
						{
							assert(bytes==storage.size());
							return storage.data();
						}
						inline void deallocate(void* p, size_t bytes, size_t alignment) override {storage = {};}

						core::vector<uint8_t> storage;
				};
				auto memRsc = core::make_smart_refctd_ptr<CVectorBacked>(hitHandles.size()+missHandles.size()+callableHandles.size()+1);
				{
					core::LinearAddressAllocatorST<uint32_t> allocator(nullptr,0,0,limits.shaderGroupBaseAlignment,0x7fff0000u);
					auto copyShaderHandles = [&](const std::span<const IGPURayTracingPipeline::SShaderGroupHandle> handles)->SBufferRange<const IGPUBuffer>
					{
						SBufferRange<const IGPUBuffer> range = {.size=handles.size()*handleSizeAligned};
						range.offset = allocator.alloc_addr(range.size,limits.shaderGroupBaseAlignment);
						memRsc->storage.resize(allocator.get_allocated_size());
						uint8_t* out = memRsc->storage.data()+range.offset;
						for (const auto& handle : handles)
						{
							memcpy(out,&handle,HandleSize);
							out += handleSizeAligned;
						}
						return range;
					};
					auto& sbt = params.sbts[i];
					sbt.raygen = copyShaderHandles({&pipeline->getRaygen(),1});
					sbt.miss.range = copyShaderHandles(pipeline->getMissHandles());
					// TODO: the material compiler with an RT pipeline backend should give 3 or 4 hitgroups depending on opacity and other funny things
					// problem is that due to how TLAS instances and their Geometries call into hitgroups, we need to spam duplicates around the SBT
					// also de-dup stuff that has the same hash (array of hitgroups) so two instances can happily point at the same material
					sbt.hit.range = copyShaderHandles(pipeline->getHitHandles());
					// TODO: material compiler will give us callables and we need to turn those into materials
					sbt.callable.range = copyShaderHandles(pipeline->getCallableHandles());
					// TODO: futhermore different rays (NEE vs BxDF) should use different SBTs using big offsets so it becomes a really funny mess 
					sbt.miss.stride = sbt.hit.stride = sbt.callable.stride = handleSizeAligned;
				}
				auto& sbtBuff = tmpBuffers.sbts[i];
				sbtBuff = ICPUBuffer::create({
					{
						.size=memRsc->storage.size(),.usage=BasicBufferUsages|buffer_usage_e::EUF_SHADER_BINDING_TABLE_BIT
					},
					/*.data = */memRsc->storage.data(),
					/*.memoryResource = */memRsc
				},core::adopt_memory);
				sbtBuff->setContentHash(sbtBuff->computeContentHash());
			}
		}
	
		// new cache if none provided
		if (!converter)
			converter = CAssetConverter::create({.device=device,.optimizer={}});

		// customized setup
		struct MyInputs : CAssetConverter::SInputs
		{
			// For the GPU Buffers to be directly writeable and so that we don't need a Transfer Queue submit at all
			inline uint32_t constrainMemoryTypeBits(const size_t groupCopyID, const IAsset* canonicalAsset, const blake3_hash_t& contentHash, const IDeviceMemoryBacked* memoryBacked) const override
			{
				assert(memoryBacked);
				return memoryBacked->getObjectType()!=IDeviceMemoryBacked::EOT_BUFFER ? (~0u):rebarMemoryTypes;
			}

			uint32_t rebarMemoryTypes;
		} inputs = {};
		inputs.logger = m_creation.logger.get().get();
		inputs.rebarMemoryTypes = device->getPhysicalDevice()->getDirectVRAMAccessMemoryTypeBits();
		// the allocator needs to be overriden to hand out memory ranges which have already been mapped so that the ReBAR fast-path can kick in
		// (multiple buffers can be bound to same memory, but memory can only be mapped once at one place, so Asset Converter can't do it)
		struct MyAllocator final : public IDeviceMemoryAllocator
		{
			ILogicalDevice* getDeviceForAllocations() const override {return device;}

			SAllocation allocate(const SAllocateInfo& info) override
			{
				auto retval = device->allocate(info);
				// map what is mappable by default so ReBAR checks succeed
				if (retval.isValid() && retval.memory->isMappable())
					retval.memory->map({.offset=0,.length=info.size});
				return retval;
			}

			ILogicalDevice* device;
		} myalloc;
		myalloc.device = device;
		inputs.allocator = &myalloc;

		// assign inputs
		{
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUBuffer>>(inputs.assets) = tmpBuffers;
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUTopLevelAccelerationStructure>>(inputs.assets) = {&tmpTLASes.front().get(),tmpTLASes.size()};
		}
		CAssetConverter::SReserveResult reservation = converter->reserve(inputs);
		{
			bool success = true;
			auto check = [&]<typename asset_type_t>(const CAssetConverter::SInputs::asset_span_t<asset_type_t> references)->void
			{
				auto objects = reservation.getGPUObjects<asset_type_t>();
				auto referenceIt = references.begin();
				for (auto& object : objects)
				{
					auto* reference = *(referenceIt++);
					if (!reference)
						continue;

					success = bool(object.value);
					if (!success)
					{
						inputs.logger.log("Failed to convert a CPU object to GPU of type %s!",ILogger::ELL_ERROR,system::to_string(reference->getAssetType()));
						return;
					}
				}
			};
			check.template operator()<ICPUBuffer>(tmpBuffers);
			check.template operator()<ICPUTopLevelAccelerationStructure>({&tmpTLASes.front().get(),tmpTLASes.size()});
			if (!success)
				return nullptr;
		}

		// convert
		{
			smart_refctd_ptr<CAssetConverter::SConvertParams::scratch_for_device_AS_build_t> scratchAlloc;
			{
				constexpr auto scratchUsages = IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT|IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT|IGPUBuffer::EUF_STORAGE_BUFFER_BIT;

				constexpr uint16_t MaxAlignment = 256;
				constexpr uint64_t MinAllocationSize = 1024;
				const auto scratchSize = core::alignUp(hlsl::max(reservation.getMaxASBuildScratchSize(false),MinAllocationSize),MaxAlignment);

				auto scratchBuffer = device->createBuffer({{.size=scratchSize,.usage=scratchUsages}});

				auto reqs = scratchBuffer->getMemoryReqs();
				reqs.memoryTypeBits &= device->getPhysicalDevice()->getDirectVRAMAccessMemoryTypeBits();

				auto allocation = device->allocate(reqs,scratchBuffer.get(),IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
				allocation.memory->map({.offset=0,.length=reqs.size});

				scratchAlloc = make_smart_refctd_ptr<CAssetConverter::SConvertParams::scratch_for_device_AS_build_t>(
					SBufferRange<video::IGPUBuffer>{0ull,scratchSize,std::move(scratchBuffer)},
					core::allocator<uint8_t>(), MaxAlignment, MinAllocationSize
				);
			}

			constexpr auto CompBufferCount = 2;

			std::array<smart_refctd_ptr<IGPUCommandBuffer>,CompBufferCount> compBufs = {};
			std::array<IQueue::SSubmitInfo::SCommandBufferInfo,CompBufferCount> compBufInfos = {};
			{
				constexpr auto RequiredFlags = IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT|IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT;
				auto pool = device->createCommandPool(m_creation.computeQueue->getFamilyIndex(),RequiredFlags);
				if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, compBufs))
				{
					inputs.logger.log("Failed to create Command Buffers for the Compute Queue!",ILogger::ELL_ERROR);
					return nullptr;
				}
				compBufs.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				for (auto i=0; i<CompBufferCount; i++)
					compBufInfos[i].cmdbuf = compBufs[i].get();
			}
			auto compSema = device->createSemaphore(0u);

			// TODO: `SIntendedSubmitInfo transfer` as well, because of images
			SIntendedSubmitInfo compute = {};
			compute.queue = m_creation.computeQueue;
			compute.scratchCommandBuffers = compBufInfos;
			compute.scratchSemaphore = {
				.semaphore = compSema.get(),
				.value = 0u,
				.stageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT|PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_COPY_BIT|PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
			};
			struct MyParams final : CAssetConverter::SConvertParams
			{
				inline uint32_t getFinalOwnerQueueFamily(const IGPUBuffer* buffer, const core::blake3_hash_t& createdFrom) override
				{
					return finalUser;
				}
				inline uint32_t getFinalOwnerQueueFamily(const IGPUAccelerationStructure* image, const core::blake3_hash_t& createdFrom) override
				{
					return finalUser;
				}

				uint8_t finalUser;
			} cvtParam = {};
			cvtParam.utilities = m_creation.utilities.get();
			cvtParam.compute = &compute;
			cvtParam.scratchForDeviceASBuild = scratchAlloc.get();
			cvtParam.finalUser = m_creation.graphicsQueue->getFamilyIndex();
			
			auto future = reservation.convert(cvtParam);
			// release the memory
			{
				for (auto& tmpTLAS : tmpTLASes)
				{
					IAsset* const asAsset = tmpTLAS.get();
					IPreHashed::discardDependantsContents({&asAsset,1});
				}
				tmpTLASes.clear();
				tmpBuffers = {};
			}
			if (future.copy()!=IQueue::RESULT::SUCCESS)
			{
				inputs.logger.log("Failed to await `CAssetConverter::SReserveResult::convert(...)` submission semaphore!",ILogger::ELL_ERROR);
				return nullptr;
			}


			const auto buffers = reservation.getGPUObjects<ICPUBuffer>();
			ubo = buffers[0].value;
			for (uint8_t i=0; i<RenderModeCount; i++)
			{
				const auto& buffer = buffers[i+1].value;
				auto setSBTBuffer = [&buffer](SStridedRange<const IGPUBuffer>& stRange)->void
				{
					stRange.range.buffer = stRange.range.size ? buffer:nullptr;
				};
				params.sbts[i].raygen.buffer = buffer;
				setSBTBuffer(params.sbts[i].miss);
				setSBTBuffer(params.sbts[i].hit);
				setSBTBuffer(params.sbts[i].callable);
			}

			const bool success = reservation.moveGPUObjects<ICPUTopLevelAccelerationStructure>(TLASes);
			assert(success);
			params.TLAS = TLASes[0].value;
		}
	}

	// write into DS
	{
		vector<IGPUDescriptorSet::SDescriptorInfo> infos;
		vector<IGPUDescriptorSet::SWriteDescriptorSet> writes;
		auto* const ds = params.sceneDS->getDescriptorSet();
		auto addWrite = [&](const uint32_t binding)->void
		{
			writes.emplace_back() = {
				.dstSet = ds,
				.binding = binding,
				.arrayElement = 0,
				.count = 1,
				.info = reinterpret_cast<const IGPUDescriptorSet::SDescriptorInfo*>(infos.size())
			};
		};
		addWrite(SceneDSBindings::UBO);
		infos.push_back(SBufferRange<IGPUBuffer>{.offset=0,.size=sizeof(SSceneUniforms),.buffer=std::move(ubo)});
		// TODO: Envmap
		{
			addWrite(SceneDSBindings::TLASes);
			infos.reserve(infos.size()+TLASes.size());
			for (auto& tlas : TLASes)
				infos.emplace_back().desc = tlas.value;
		}
		// TODO: Samplers
		// TODO: Sampled Images
		// TODO: Envmap PDF
		// TODO: Envmap Warp Map
		for (auto& write : writes)
			write.info = infos.data()+reinterpret_cast<const uint64_t&>(write.info);
		device->updateDescriptorSets(writes,{});
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

	const auto* const scene = session->getConstructionParams().scene.get();
	const auto mode = sessionParams.mode;
	const auto& sessionResources = session->getActiveResources();
	const auto* const pipeline = scene->getPipeline(mode);
	
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
	success = success && cb->traceRays(scene->getSBT(mode),renderSize.x,renderSize.y,sessionParams.type!=CSession::sensor_type_e::Env ? 1:6);

	if (success)
		return SSubmit(this,cb);
	else
		return {};
}

IQueue::SSubmitInfo::SSemaphoreInfo CRenderer::SSubmit::operator()(std::span<const video::IQueue::SSubmitInfo::SSemaphoreInfo> extraWaits)
{
	if (!cb || !cb->end())
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