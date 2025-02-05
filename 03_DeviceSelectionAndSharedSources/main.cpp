// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "CommonPCH/PCH.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

// TODO[Przemek]: update comments

// This is the most nuts thing you'll ever see, a header of HLSL included both in C++ and HLSL
#include "app_resources/common.hlsl"
#include "Testers.h"

constexpr bool ENABLE_TESTS = false;

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

		smart_refctd_ptr<nbl::asset::ICPUComputePipeline> cpuPipeline = introspector.createApproximateComputePipelineFromIntrospection(specInfo);

		smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
		// Nabla hardcodes the Max number of Descriptor Sets to 4
		std::array<smart_refctd_ptr<IGPUDescriptorSet>,IGPUPipelineLayout::DESCRIPTOR_SET_COUNT> ds;
		// You can automate some of the IGPU object creation from ICPU using the Asset Converter
		{
			// The Asset Converter keeps a local cache of already converted GPU objects.
			// Because the asset converter converts "by content" and not "by handle" (content hashes are compared,
			// functionally identical objects will convert to the same GPU Object, so you get free duplicate removal.
			smart_refctd_ptr<nbl::video::CAssetConverter> converter = nbl::video::CAssetConverter::create({.device=m_device.get(),.optimizer={}});
			CAssetConverter::SInputs inputs = {};
			inputs.logger = m_logger.get();
			// All dependant assets will be converted (or found in the `inputs.readCache`) 
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUComputePipeline>>(inputs.assets) = {&cpuPipeline.get(),1};
			// Simple Objects that don't require any queue submissions (such as trasfer operations or Acceleration Structure builds) are created right away
			CAssetConverter::SReserveResult reservation = converter->reserve(inputs);
			// There's a 1:1 mapping between `SInputs::assets` and `SReservation::m_gpuObjects`.
			const auto pipelines = reservation.getGPUObjects<ICPUComputePipeline>();
			// Anything that fails to convert is a nullptr in the span of GPU Objects
			pipeline = pipelines[0].value;
			if (!pipeline)
				return logFail("Failed to convert CPU pipeline to GPU pipeline");

			// Create Descriptor Sets for the Layouts manually
			const auto dscLayoutPtrs = pipeline->getLayout()->getDescriptorSetLayouts();
			auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,dscLayoutPtrs);
			pool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), ds.data());
		}

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
			m_api->startCapture();
			queue->submit({{submitInfo}});
			m_api->endCapture();
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
			options.stage = source->getStage() == IShader::E_SHADER_STAGE::ESS_COMPUTE ? source->getStage() : IShader::E_SHADER_STAGE::ESS_VERTEX; // TODO: do smth with it
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
