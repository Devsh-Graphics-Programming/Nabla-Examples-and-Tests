// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/MonoDeviceApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


// This is the most nuts thing you'll ever see, a header of HLSL included both in C++ and HLSL
#include "app_resources/common.hlsl"


// This time we create the device in the base class and also use a base class to give us an Asset Manager and an already mounted built-in resource archive
class DeviceSelectionAndSharedSourcesApp final : public examples::MonoDeviceApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::MonoDeviceApplication;
		using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;
	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		DeviceSelectionAndSharedSourcesApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(std::move(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			// this time we load a shader directly from a file
			auto assetBundle = m_assetMgr->getAsset("app_resources/shader.comp.hlsl",lp);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
				return logFail("Could not load shader!");

			// It would be super weird if loading a shader from a file produced more than 1 asset
			assert(assets.size()==1);
			smart_refctd_ptr<ICPUSpecializedShader> source = IAsset::castDown<ICPUSpecializedShader>(assets[0]);

			// Now is the time to introduce the SPIR-V introspector which will let you "guess" Layouts and Pipelines Creation Parameters!
			auto introspector = std::make_unique<CSPIRVIntrospector>();

			smart_refctd_ptr<const CSPIRVIntrospector::CIntrospectionData> introspection;
			{
				// Unfortunately introspection only works on SPIR-V so we still have to compile the shader by hand
				const ICPUShader* unspecialized = source->getUnspecialized();

				// The Asset Manager has a Default Compiler Set which contains all built-in compilers (so it can try them all)
				auto* compilerSet = m_assetMgr->getCompilerSet();

				// This time we use a more "generic" option struct which works with all compilers
				nbl::asset::IShaderCompiler::SCompilerOptions options = {};
				// The Shader Asset Loaders deduce the stage from the file extension,
				// if the extension is generic (.glsl or .hlsl) the stage is unknown.
				// But it can still be overriden from within the source with a `#pragma shader_stage`
				options.stage = unspecialized->getStage();
				options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
				// we need to perform an unoptimized compilation with source debug info or we'll lose names of variable sin the introspection
				options.spirvOptimizer = nullptr;
				options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
				// The nice thing is that when you load a shader from file, it has a correctly set `filePathHint`
				// so it plays nicely with the preprocessor, and finds `#include`s without intervention.
				options.preprocessorOptions.sourceIdentifier = unspecialized->getFilepathHint();
				options.preprocessorOptions.logger = m_logger.get();
				options.preprocessorOptions.includeFinder = compilerSet->getShaderCompiler(unspecialized->getContentType())->getDefaultIncludeFinder();

				auto spirvUnspecialized = compilerSet->compileToSPIRV(unspecialized,options);
				const CSPIRVIntrospector::SIntrospectionParams inspctParams = {.entryPoint=source->getSpecializationInfo().entryPoint,.cpuShader=spirvUnspecialized};

				introspection = introspector->introspect(inspctParams);
				if (!introspection)
					return logFail("SPIR-V Introspection failed, probably the required SPIR-V compilation failed first!");

				// now we need to swap out the HLSL for SPIR-V
				source = make_smart_refctd_ptr<ICPUSpecializedShader>(std::move(spirvUnspecialized),ISpecializedShader::SInfo(nullptr,nullptr,source->getSpecializationInfo().entryPoint));
			}

			// Just a check that out specialization info will match
			if (!introspection->canSpecializationlesslyCreateDescSetFrom())
				return logFail("Someone changed the shader and some descriptor binding depends on a specialization constant!");

			// We've now skipped the manual creation of a descriptor set layout, pipeline layout
			smart_refctd_ptr<nbl::asset::ICPUComputePipeline> cpuPipeline = introspector->createApproximateComputePipelineFromIntrospection(source.get());

			// And now I show you how to save 100 lines of code to create all objects between a Compute Pipeline and a Shader
			smart_refctd_ptr<IGPUComputePipeline> pipeline;
			{
				// This is the main workhorse of our asset system, it automates the creation of matching IGPU objects for ICPU object while
				// making sure that if two ICPU objects reference the same dependency, that structure carries over to their associated IGPU objects.
				// We're working on a new improved Hash Tree based system which will catch identically defined ICPU objects even if their pointers differ.
				auto assetConverter = std::make_unique<IGPUObjectFromAssetConverter>();

				// Thankfully because we're not doing any conversions on Memory Backed objects we don't need to set up the queue parameters
				IGPUObjectFromAssetConverter::SParams conversionParams = {};
				conversionParams.device = m_device.get();
				// The asset manager is necessary in the conversion process to provide the CPU2GPU associative cache,
				// basically if we detect an ICPUAsset in the DAG already has an associated IGPUAsset resulting from
				// a previous conversion, that IGPUAsset will be used in place of creating a new ICPUAsset.
				// There's a new an improved system coming which will use Merkle Trees to catch duplicate subgraphs better.
				// In that system the cache will be split off from the Asset Manager and be its own thing to improve the API.
				// However Asset Manager will still be needed for "restore from dummy" operations (more on that later).
				conversionParams.assetManager = m_assetMgr.get();

				// Note that the Conversion Parameters are reusable between multiple runs
				nbl::video::created_gpu_object_array<ICPUComputePipeline> convertedGPUObjects = assetConverter->getGPUObjectsFromAssets(&cpuPipeline,&cpuPipeline+1,conversionParams);
				assert(convertedGPUObjects->size() == 1);
				// What does it mean that an asset is a "dummy"? It means it has been "hollowed out", degraded to a husk with enough metadata to allow us to hash and compare it for equality.
				assert(cpuPipeline->isADummyObjectForCache());
				// The default asset converter turns every converted asset into a dummy, releasing memory backing `IAsset`s, which are mostly the allocations backing an `ICPUBuffer`
				// Why doesn't the Asset simply get deleted?
				// We need a stable key (the old pointer) and ability to compare for them for true equality, so that our CPU->GPU hash map prevents the creation of duplicate GPU objects.
				// Also this allows us to retain some information in-case we need to "restore" a dummy by reloading it, for example when one wishes to create a derivative map from a heightmap.
				assert(source->getUnspecialized()->getContent()->getPointer() == nullptr);

				pipeline = convertedGPUObjects->front();
			}

			// Nabla hardcodes the maximum descriptor set count to 4
			constexpr uint32_t MaxDescriptorSets = 4; //SPhysicalDeviceLimits::MaxDescriptorSets;
			smart_refctd_ptr<IGPUDescriptorSet> ds[MaxDescriptorSets];

			// Aside from being mutable, ICPU Assets are reflectable too! And that's how we'll create the Descriptor Sets.
			{
				smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayouts[MaxDescriptorSets];
				for (auto i=0; i<MaxDescriptorSets; i++)
				{
					ICPUDescriptorSetLayout* cpuDSLayout = cpuPipeline->getLayout()->getDescriptorSetLayout(i);
					if (cpuDSLayout)
					{
						// This is bad design which we'll fix by splitting the CPU2GPU cache off from the Asset Manager
						// Unfortunate side-effect of current design is that the returned entry is typeless,
						// because of a ban on introducing dependencies from `video` namespace into `asset`.
						smart_refctd_ptr<IReferenceCounted> found = m_assetMgr->findGPUObject(cpuDSLayout);
						// We converted our original CPU Assets without an override that disables de-duplication,
						// so associated GPU Object must have been found
						assert(found);
						//
						dsLayouts[i] = nbl::core::move_and_dynamic_cast<IGPUDescriptorSetLayout>(found);
					}
				}
				// The signature is `T* const& smart_refctd_ptr<T>::get()` which makes it possible to neatly
				// get a raw pointer to a constant array of raw pointers from an array of smart pointers
				// by taking the address of the first element's get() method return value
				auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,&dsLayouts->get(),&dsLayouts->get()+MaxDescriptorSets);
				pool->createDescriptorSets(MaxDescriptorSets,&dsLayouts->get(),ds);
			}

			// Need to know input and output sizes by ourselves obviously
			constexpr size_t WorkgroupCount = 4096;
			constexpr size_t BufferSize = sizeof(uint32_t)*WorkgroupSize*WorkgroupCount;

			// Lets make the buffers for our shader, but lets fill them later
			auto allocateBuffer = [&]() -> smart_refctd_ptr<IGPUBuffer>
			{
				IGPUBuffer::SCreationParams params = {};
				params.size = BufferSize;
				params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
				auto buff = m_device->createBuffer(std::move(params));
				if (!buff)
				{
					m_logger->log("Failed to create a GPU Buffer for an SSBO of size %d!\n",ILogger::ELL_ERROR,params.size);
					return nullptr;
				}

				auto reqs = buff->getMemoryReqs();
				// same as last time we make the buffers mappable for easy access
				reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getHostVisibleMemoryTypeBits();
				// and we make the allocation dedicated
				m_device->allocate(reqs,buff.get(),nbl::video::IDeviceMemoryAllocation::EMAF_NONE);

				return buff;
			};
			smart_refctd_ptr<IGPUBuffer> inputBuff[2] = {allocateBuffer(),allocateBuffer()};
			auto outputBuff = allocateBuffer();

			// Now lets get to filling the descriptor sets from the introspection data
			{
				vector<IGPUDescriptorSet::SWriteDescriptorSet> writes;
				vector<IGPUDescriptorSet::SDescriptorInfo> infos;
				for (auto i=0; i<MaxDescriptorSets; i++)
				{
					for (const auto& binding : introspection->descriptorSetBindings[i])
					{
						// We know to always expect a buffer for this shader.
						if (binding.type!=nbl::asset::E_SHADER_RESOURCE_TYPE::ESRT_STORAGE_BUFFER)
						{
							m_logger->log("Unexpected Type of (in set %d) Descriptor Binding %d (count %d) detected by SPIR-V Reflection!",ILogger::ELL_ERROR,i,binding.binding,binding.descriptorCount);
							continue;
						}
						// This is why I'm not a fan of connecting stuff up based on reflection, because you need to know what to expect anyway (e.g. name string).
						// I prefer to keep Host and Device code in-sync w.r.t. Descriptor Binding mapping by using a shared HLSL header with NBL_CONSTEXPR
						const smart_refctd_ptr<IGPUBuffer>* buffers = nullptr;
						if (binding.name=="inputs")
							buffers = inputBuff;
						else if (binding.name=="output")
							buffers = &outputBuff;
						else
						{
							m_logger->log("Unexpected Named of Descriptor %s in Set %d, Binding %d detected by SPIR-V Reflection!",ILogger::ELL_ERROR,binding.name.c_str(),i,binding.binding,binding.descriptorCount);
							continue;
						}
						// Introspection of SPIR-V cannot tell you whether an UBO or SSBO is a DYNAMIC OFFSET kind or not,
						// it had to be "guessed" for the Approximate Pipeline Layout created by the introspection
						const nbl::asset::IDescriptor::E_TYPE type = CSPIRVIntrospector::resType2descType(binding.type);
						// Don't want to fall victim to `infos` reallocating after a push_back, so will patch up the pointers later
						writes.push_back({ds[i].get(),binding.binding,0,binding.descriptorCount,type,reinterpret_cast<IGPUDescriptorSet::SDescriptorInfo*>(infos.size())});
						for (auto j=0; j<binding.descriptorCount; j++)
						{
							auto& info = infos.emplace_back();
							info.desc = buffers[j];
							info.info.buffer = {0u,buffers[j]->getSize()};
						}
					}
				}
				// fix up the info pointers
				for (auto& write : writes)
					write.info = infos.data()+ptrdiff_t(write.info);
				if (!m_device->updateDescriptorSets(writes.size(),writes.data(),0u,nullptr))
					return logFail("Failed to write Descriptor Sets");
			}

			// Make a utility since we have to touch 3 buffers
			auto mapBuffer = [&]<typename PtrT>(const PtrT& buff, bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> accessHint) -> uint32_t*
			{
				auto* const memory = buff->getBoundMemory();
				const IDeviceMemoryAllocation::MappedMemoryRange memoryRange(memory,0ull,memory->getAllocationSize());

				void* ptr = m_device->mapMemory(memoryRange,accessHint);
				if (!ptr)
					m_logger->log("Failed to map the whole Memory Allocation of a GPU Buffer %p for access %d!\n",ILogger::ELL_ERROR,buff,accessHint);
				
				if (accessHint.hasFlags(IDeviceMemoryAllocation::EMCAF_READ))
				if (!memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
					m_device->invalidateMappedMemoryRanges(1,&memoryRange);

				return reinterpret_cast<uint32_t*>(ptr);
			};
			
			// We'll fill the buffers in a way they always add up to WorkgrouCount*WorkgroupSize
			constexpr auto DWORDCount = WorkgroupSize*WorkgroupCount;
			{
				constexpr auto writeFlag = IDeviceMemoryAllocation::EMCAF_WRITE;
				uint32_t* const inputs[2] = {mapBuffer(inputBuff[0],writeFlag),mapBuffer(inputBuff[1],writeFlag)};
				for (auto i=0; i<DWORDCount; i++)
				{
					inputs[0][i] = i;
					inputs[1][i] = DWORDCount - i;
				}
				for (auto j=0; j<2; j++)
				{
					auto* const memory = inputBuff[j]->getBoundMemory();
					// New thing to learn, if the mapping is not coherent and you write it, you need to flush!
					const IDeviceMemoryAllocation::MappedMemoryRange memoryRange(memory,0ull,memory->getAllocationSize());
					m_device->flushMappedMemoryRanges(1,&memoryRange);
					m_device->unmapMemory(memory);
				}
			}

			// create, record, submit and await commandbuffers
			{
				IGPUQueue* const queue = getComputeQueue();

				smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
				{
					smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::ECF_TRANSIENT_BIT);
					if (!m_device->createCommandBuffers(cmdpool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf))
					{
						logFail("Failed to create Command Buffers!\n");
						return false;
					}
				
					cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
					cmdbuf->bindComputePipeline(pipeline.get());
					cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE,pipeline->getLayout(),0,MaxDescriptorSets,&ds->get());
					cmdbuf->dispatch(WorkgroupCount,1,1);
					cmdbuf->end();
				}

				// A sidenote: Waiting for the Device or Queue to become idle does not give the same memory visibility guarantees as a signalled fence or semaphore
				smart_refctd_ptr<IGPUFence> done = m_device->createFence(IGPUFence::ECF_UNSIGNALED);
				{
					IGPUQueue::SSubmitInfo submitInfo = {};
					submitInfo.commandBufferCount = 1;
					submitInfo.commandBuffers = &cmdbuf.get();

					// To keep the sample renderdoc-able
					queue->startCapture();
					queue->submit(1u,&submitInfo,done.get());
					queue->endCapture();
				}
				m_device->blockForFences(1,&done.get());
			}

			// a simple test to check we got the right thing back
			auto outputData = mapBuffer(outputBuff,IDeviceMemoryAllocation::EMCAF_READ);
			for (auto i=0; i<WorkgroupSize*WorkgroupCount; i++)
			if (outputData[i]!=DWORDCount)
				return logFail("DWORD at position %d doesn't match!\n",i);
			m_device->unmapMemory(outputBuff->getBoundMemory());

			return true;
		}

		// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
		void workLoopBody() override {}

		// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
		bool keepRunning() override {return false;}

};


NBL_MAIN_FUNC(DeviceSelectionAndSharedSourcesApp)