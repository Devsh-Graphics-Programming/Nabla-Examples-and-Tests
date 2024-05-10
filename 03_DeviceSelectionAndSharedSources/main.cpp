// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

// TODO[Przemek]: update comments

// This is the most nuts thing you'll ever see, a header of HLSL included both in C++ and HLSL
#include "app_resources/common.hlsl"
#include "Testers.h"

constexpr bool ENABLE_TESTS = true;

// This time we create the device in the base class and also use a base class to give us an Asset Manager and an already mounted built-in resource archive
class DeviceSelectionAndSharedSourcesApp final : public application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	DeviceSelectionAndSharedSourcesApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	// we stuff all our work here because its a "single shot" app
	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(core::smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		// Just a check that out specialization info will match
		//if (!introspection->canSpecializationlesslyCreateDescSetFrom())
			//return logFail("Someone changed the shader and some descriptor binding depends on a specialization constant!");

		if constexpr (ENABLE_TESTS)
		{
			MergeTester mergeTester("CSPIRVIntrospector::CPipelineIntrospectionData::merge");
			mergeTester.performTests(m_physicalDevice, m_device.get(), m_logger.get(), m_assetMgr.get());

			// testing creation of compute pipeline layouts compatible for multiple shaders
			PredefinedLayoutTester layoutTester("CPSIRVIntrospector::createApproximateComputePipelineFromIntrospection");
			layoutTester.performTests(m_physicalDevice, m_device.get(), m_logger.get(), m_assetMgr.get());

			SandboxTester sandboxTester("unknown");
			sandboxTester.performTests(m_physicalDevice, m_device.get(), m_logger.get(), m_assetMgr.get());
		}

		CSPIRVIntrospector introspector;
		auto compiledShader = this->compileShaderAndTestIntrospection("app_resources/shader.comp.hlsl", introspector);
		auto source = compiledShader.first;
		auto shaderIntrospection = compiledShader.second;

		//shaderIntrospection->debugPrint(m_logger.get());

		// We've now skipped the manual creation of a descriptor set layout, pipeline layout
		ICPUShader::SSpecInfo specInfo;
		specInfo.entryPoint = "main";
		specInfo.shader = source.get();

		// TODO: cast to IGPUComputePipeline
		smart_refctd_ptr<nbl::asset::ICPUComputePipeline> cpuPipeline = introspector.createApproximateComputePipelineFromIntrospection(specInfo);

		smart_refctd_ptr<IGPUShader> shader = m_device->createShader(source.get());

		std::array<std::vector<IGPUDescriptorSetLayout::SBinding>, IGPUPipelineLayout::DESCRIPTOR_SET_COUNT> bindings;
		for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
		{
			const auto& introspectionBindings = shaderIntrospection->getDescriptorSetInfo(i);
			bindings[i].resize(introspectionBindings.size());

			for (const auto& introspectionBinding : introspectionBindings)
			{
				auto& binding = bindings[i].emplace_back();

				binding.binding = introspectionBinding.binding;
				binding.type = introspectionBinding.type;
				binding.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE;
				binding.stageFlags = IGPUShader::ESS_COMPUTE;
				assert(introspectionBinding.count.countMode == CSPIRVIntrospector::CIntrospectionData::SDescriptorArrayInfo::DESCRIPTOR_COUNT::STATIC);
				binding.count = introspectionBinding.count.count;
			}
		}

		const std::array<core::smart_refctd_ptr<IGPUDescriptorSetLayout>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> dsLayouts = { 
			bindings[0].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[0]),
			bindings[1].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[1]),
			bindings[2].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[2]),
			bindings[3].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[3])
		};

		// Nabla actually has facilities for SPIR-V Reflection and "guessing" pipeline layouts for a given SPIR-V which we'll cover in a different example
		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout = m_device->createPipelineLayout(
			{},
			core::smart_refctd_ptr(dsLayouts[0]),
			core::smart_refctd_ptr(dsLayouts[1]),
			core::smart_refctd_ptr(dsLayouts[2]),
			core::smart_refctd_ptr(dsLayouts[3])
		);
		if (!pplnLayout)
			return logFail("Failed to create a Pipeline Layout!\n");

		// We use strong typing on the pipelines (Compute, Graphics, Mesh, RT), since there's no reason to polymorphically switch between different pipelines
		smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
		{
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = pplnLayout.get();
			// Theoretically a blob of SPIR-V can contain multiple named entry points and one has to be chosen, in practice most compilers only support outputting one (and glslang used to require it be called "main")
			params.shader.entryPoint = "main";
			params.shader.shader = shader.get();
			// we'll cover the specialization constant API in another example
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline))
				return logFail("Failed to create pipelines (compile & link shaders)!\n");
		}

		// Nabla hardcodes the maximum descriptor set count to 4
		constexpr uint32_t MaxDescriptorSets = ICPUPipelineLayout::DESCRIPTOR_SET_COUNT;
		const std::array<IGPUDescriptorSetLayout*, MaxDescriptorSets> dscLayoutPtrs = { 
			!dsLayouts[0] ? nullptr : dsLayouts[0].get(),
			!dsLayouts[1] ? nullptr : dsLayouts[1].get(),
			!dsLayouts[2] ? nullptr : dsLayouts[2].get(),
			!dsLayouts[3] ? nullptr : dsLayouts[3].get()
		};
		std::array<smart_refctd_ptr<IGPUDescriptorSet>, MaxDescriptorSets> ds;
		auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()));
		pool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), ds.data());

		// Need to know input and output sizes by ourselves obviously
		constexpr size_t WorkgroupCount = 4096;
		constexpr size_t BufferSize = sizeof(uint32_t) * WorkgroupSize * WorkgroupCount;

		using BuffAllocPair = std::pair<smart_refctd_ptr<IGPUBuffer>, IDeviceMemoryAllocator::SAllocation>;
		auto allocateBuffer = [&]() -> BuffAllocPair
			{
				IGPUBuffer::SCreationParams params = {};
				params.size = BufferSize;
				params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
				auto buff = m_device->createBuffer(std::move(params));
				if (!buff)
				{
					m_logger->log("Failed to create a GPU Buffer for an SSBO of size %d!\n", ILogger::ELL_ERROR, params.size);
					return BuffAllocPair(nullptr, {});
				}

				auto reqs = buff->getMemoryReqs();
				// same as last time we make the buffers mappable for easy access
				reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getHostVisibleMemoryTypeBits();
				// and we make the allocation dedicated
				auto allocation = m_device->allocate(reqs, buff.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);

				return std::make_pair(buff, allocation);
			};
		BuffAllocPair inputBuff[2] = { allocateBuffer(), allocateBuffer() };
		BuffAllocPair outputBuff = allocateBuffer();

		// Make a utility since we have to touch 3 buffers
		auto mapBuffer = [&](const BuffAllocPair& buffAllocPair, bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> accessHint) -> uint32_t*
		{
			const auto& buff = buffAllocPair.first;
			const auto& buffMem = buffAllocPair.second.memory;
			void* ptr = buffMem->map({ 0ull, buffMem->getAllocationSize() }, accessHint);
			if (!ptr)
				m_logger->log("Failed to map the whole Memory Allocation of a GPU Buffer %p for access %d!\n", ILogger::ELL_ERROR, buff, accessHint);

			const ILogicalDevice::MappedMemoryRange memoryRange(buffMem.get(), 0ull, buffMem->getAllocationSize());
			if (accessHint.hasFlags(IDeviceMemoryAllocation::EMCAF_READ))
				if (!buffMem->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
					m_device->invalidateMappedMemoryRanges(1, &memoryRange);

			return static_cast<uint32_t*>(ptr);
		};

		// We'll fill the buffers in a way they always add up to WorkgrouCount*WorkgroupSize
		constexpr auto DWORDCount = WorkgroupSize * WorkgroupCount;
		{
			constexpr auto writeFlag = IDeviceMemoryAllocation::EMCAF_WRITE;
			uint32_t* const inputs[2] = { mapBuffer(inputBuff[0],writeFlag),mapBuffer(inputBuff[1],writeFlag) };
			for (auto i = 0; i < DWORDCount; i++)
			{
				inputs[0][i] = i;
				inputs[1][i] = DWORDCount - i;
			}
			for (auto j = 0; j < 2; j++)
			{
				const auto& buffMem = inputBuff[j].second.memory;
				// New thing to learn, if the mapping is not coherent and you write it, you need to flush!
				const ILogicalDevice::MappedMemoryRange memoryRange(buffMem.get(), 0ull, buffMem->getAllocationSize());
				m_device->flushMappedMemoryRanges(1, &memoryRange);
				buffMem->unmap();
			}
		}

		{
			IGPUDescriptorSet::SDescriptorInfo inputBuffersInfo[2];
			inputBuffersInfo[0].desc = smart_refctd_ptr(inputBuff[0].first);
			inputBuffersInfo[0].info.buffer = { .offset = 0,.size = inputBuff[0].first->getSize() };
			inputBuffersInfo[1].desc = smart_refctd_ptr(inputBuff[1].first);
			inputBuffersInfo[1].info.buffer = { .offset = 0,.size = inputBuff[1].first->getSize() };

			IGPUDescriptorSet::SDescriptorInfo outputBufferInfo;
			outputBufferInfo.desc = smart_refctd_ptr(outputBuff.first);
			outputBufferInfo.info.buffer = { .offset = 0,.size = outputBuff.first->getSize() };

			IGPUDescriptorSet::SWriteDescriptorSet writes[2] = {
				{.dstSet = ds[1].get(),.binding = 2,.arrayElement = 0,.count = 2,.info = inputBuffersInfo},
				{.dstSet = ds[3].get(),.binding = 6,.arrayElement = 0,.count = 1,.info = &outputBufferInfo}
			};

			m_device->updateDescriptorSets(std::span(writes, 2), {});
		}

		// create, record, submit and await commandbuffers
		constexpr auto StartedValue = 0;
		constexpr auto FinishedValue = 45;
		smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore(StartedValue);
		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
		{
			IQueue* const queue = getComputeQueue();

			{
				smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf))
				{
					logFail("Failed to create Command Buffers!\n");
					return false;
				}

				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				cmdbuf->bindComputePipeline(pipeline.get());
				cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, pipeline->getLayout(), 0, ds.size(), &ds.begin()->get());
				cmdbuf->dispatch(WorkgroupCount, 1, 1);
				cmdbuf->end();
			}
			
			IQueue::SSubmitInfo submitInfo = {};
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
			submitInfo.commandBuffers = cmdbufs;
			const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = progress.get(),.value = FinishedValue,.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
			submitInfo.signalSemaphores = signals;

			// To keep the sample renderdoc-able
			queue->startCapture();
			queue->submit({{submitInfo}});
			queue->endCapture();
		}

		// As the name implies this function will not progress until the fence signals or repeated waiting returns an error.
		const ISemaphore::SWaitInfo waitInfos[] = { {
			.semaphore = progress.get(),
			.value = FinishedValue
		} };
		m_device->blockForSemaphores(waitInfos);

		// a simple test to check we got the right thing back
		auto outputData = mapBuffer(outputBuff, IDeviceMemoryAllocation::EMCAF_READ);
		for (auto i = 0; i < WorkgroupSize * WorkgroupCount; i++)
			if (outputData[i] != DWORDCount)
				return logFail("DWORD at position %d doesn't match!\n", i);
		outputBuff.second.memory->unmap();

		return true;
	}

	// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
	void workLoopBody() override {}

	// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
	bool keepRunning() override { return false; }

	std::pair<smart_refctd_ptr<ICPUShader>, smart_refctd_ptr<const CSPIRVIntrospector::CStageIntrospectionData>> compileShaderAndTestIntrospection(
		const std::string& shaderPath, CSPIRVIntrospector& introspector)
	{
		IAssetLoader::SAssetLoadParams lp = {};
		lp.logger = m_logger.get();
		lp.workingDirectory = ""; // virtual root
		// this time we load a shader directly from a file
		auto assetBundle = m_assetMgr->getAsset(shaderPath, lp);
		const auto assets = assetBundle.getContents();
		if (assets.empty())
		{
			logFail("Could not load shader!");
			assert(0);
		}

		// It would be super weird if loading a shader from a file produced more than 1 asset
		assert(assets.size() == 1);
		smart_refctd_ptr<ICPUShader> source = IAsset::castDown<ICPUShader>(assets[0]);
		
		smart_refctd_ptr<const CSPIRVIntrospector::CStageIntrospectionData> introspection;
		{
			// The Asset Manager has a Default Compiler Set which contains all built-in compilers (so it can try them all)
			auto* compilerSet = m_assetMgr->getCompilerSet();

			// This time we use a more "generic" option struct which works with all compilers
			nbl::asset::IShaderCompiler::SCompilerOptions options = {};
			// The Shader Asset Loaders deduce the stage from the file extension,
			// if the extension is generic (.glsl or .hlsl) the stage is unknown.
			// But it can still be overriden from within the source with a `#pragma shader_stage` 
			options.stage = source->getStage() == IShader::ESS_COMPUTE ? source->getStage() : IShader::ESS_VERTEX; // TODO: do smth with it
			options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
			// we need to perform an unoptimized compilation with source debug info or we'll lose names of variable sin the introspection
			options.spirvOptimizer = nullptr;
			options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
			// The nice thing is that when you load a shader from file, it has a correctly set `filePathHint`
			// so it plays nicely with the preprocessor, and finds `#include`s without intervention.
			options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
			options.preprocessorOptions.logger = m_logger.get();
			options.preprocessorOptions.includeFinder = compilerSet->getShaderCompiler(source->getContentType())->getDefaultIncludeFinder();

			auto spirvUnspecialized = compilerSet->compileToSPIRV(source.get(), options);
			const CSPIRVIntrospector::CStageIntrospectionData::SParams inspctParams = { .entryPoint = "main", .shader = spirvUnspecialized };

			introspection = introspector.introspect(inspctParams);
			introspection->debugPrint(m_logger.get());

			if (!introspection)
			{
				logFail("SPIR-V Introspection failed, probably the required SPIR-V compilation failed first!");
				return std::pair(nullptr, nullptr);
			}

			{
				auto* srcContent = spirvUnspecialized->getContent();

				system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
				m_physicalDevice->getSystem()->createFile(future, system::path("../app_resources/compiled.spv"), system::IFileBase::ECF_WRITE);
				if (auto file = future.acquire(); file && bool(*file))
				{
					system::IFile::success_t succ;
					(*file)->write(succ, srcContent->getPointer(), 0, srcContent->getSize());
					succ.getBytesProcessed(true);
				}
			}

			// now we need to swap out the HLSL for SPIR-V
			source = std::move(spirvUnspecialized);
		}

		return std::pair(source, introspection);
	}
};

NBL_MAIN_FUNC(DeviceSelectionAndSharedSourcesApp)
