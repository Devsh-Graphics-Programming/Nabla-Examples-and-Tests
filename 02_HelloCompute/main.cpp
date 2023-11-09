// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// always include nabla first before std:: headers
#include "nabla.h"

#include "nbl/system/CColoredStdoutLoggerANSI.h"
#include "nbl/system/IApplicationFramework.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

// this time instead of defining our own `int main()` 
class HelloComputeApp : public IApplicationFramework
{
	public:
		// Generally speaking because certain platforms delay initialization from main object construction you should just forward and not do anything in the ctor
		using IApplicationFramework::IApplicationFramework;

		// we stuff all our work here
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// This is a weird pattern, basically on some platforms all file & system operations need to go through a "God Object" only handed to you in some plaform specific way
			// On "normal" platforms like win32 and Linux we can just create system objects at will and there's no special state we need to find.
			if (!system)
				system = IApplicationFramework::createSystem();

			// create a logger with default logging level masks
			auto logger = make_smart_refctd_ptr<CColoredStdoutLoggerANSI>();
			auto logFail = [&logger]<typename... Args>(const char* msg, Args&&... args) -> void {logger->log(msg,ILogger::ELL_ERROR,std::forward<Args>(args)...);};

			// You should already know Vulkan and come here to save on the boilerplate, if you don't know what instances and instance extensions are, then find out.
			smart_refctd_ptr<nbl::video::CVulkanConnection> api;
			{
				// You generally want to default initialize any parameter structs
				nbl::video::IAPIConnection::SFeatures apiFeaturesToEnable = {};
				// generally you want to make your life easier during development
				apiFeaturesToEnable.validations = true;
				apiFeaturesToEnable.synchronizationValidation = true;
				// want to make sure we have this so we can name resources for vieweing in RenderDoc captures
				apiFeaturesToEnable.debugUtils = true;
				// create our Vulkan instance
				if (!(api=CVulkanConnection::create(smart_refctd_ptr(system),0,_NBL_APP_NAME_,smart_refctd_ptr(logger),apiFeaturesToEnable)))
				{
					logFail("Failed to crate an IAPIConnection!");
					return false;
				}
			}

			// We won't go deep into performing physical device selection in this example, we'll take any device with a compute queue.
			// Nabla has its own set of required baseline Vulkan features anyway, it won't report any device that doesn't meet them.
			nbl::video::IPhysicalDevice* physDev = nullptr;
			nbl::video::ILogicalDevice::SQueueCreationParams queueParams = {};
			// we will only deal with a single queue in this example
			queueParams.count = 1;
			for (auto physDevIt=api->getPhysicalDevices().begin(); physDevIt!=api->getPhysicalDevices().end(); physDevIt++)
			{
				const auto familyProps = (*physDevIt)->getQueueFamilyProperties();
				// this is the only "complicated" part, we want to create a queue that supports compute pipelines
				for (auto i=0; i<familyProps.size(); i++)
				if (familyProps[i].queueFlags.hasFlags(IPhysicalDevice::E_QUEUE_FLAGS::EQF_COMPUTE_BIT))
				{
					physDev = *physDevIt;
					queueParams.familyIndex = i;
					break;
				}
			}
			if (!physDev)
			{
				logFail("Failed to find any Physical Devices with Compute capable Queue Families!");
				return false;
			}

			// logical devices need to be created form physical devices which will actually let us create vulkan objects and use the physical device
			smart_refctd_ptr<ILogicalDevice> device;
			{
				float priority = 1;
				queueParams.flags = static_cast<IGPUQueue::E_CREATE_FLAGS>(0); // TODO: make this go away on the `vulkan_1_3` branch
				queueParams.priorities = &priority; // TODO: make this go away on the `vulkan_1_3` branch
				ILogicalDevice::SCreationParams params = {};
				params.queueParamsCount = 1;
				params.queueParams = &queueParams;
				if (!(device=physDev->createLogicalDevice(std::move(params))))
				{
					logFail("Failed to create a Logical Device!");
					return false;
				}
			}

			constexpr uint32_t WorkgroupSize = 256;
			constexpr uint32_t WorkgroupCount = 2048;
			//!
			smart_refctd_ptr<nbl::asset::ICPUShader> cpuShader;
			{
				// Normally we'd use the ISystem and the IAssetManager to load shaders flexibly from (virtual) files for ease of development (syntax highlighting and Intellisense),
				// but I want to show the full process of assembling a shader from raw source code at least once.
				smart_refctd_ptr<nbl::asset::IShaderCompiler> compiler = make_smart_refctd_ptr<nbl::asset::CHLSLCompiler>(smart_refctd_ptr(system));

				// A simple shader that writes out the Global Invocation Index to the position it corresponds to in the buffer
				// Note the injection of a define from C++ to keep the workgroup size in sync.
				const string source = "#define WORKGROUP_SIZE "+std::to_string(WorkgroupSize)+R"===(
					[[vk::binding(0,0)]] StructuredBuffer<uint32_t> buff;

					[numthreads(WORKGROUP_SIZE,1,1)]
					void main(uint32_t3 ID : SV_DispatchThreadID)
					{
						buff = ID.x;
					}
				)===";

				CHLSLCompiler::SOptions options = {};
				options.stage = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE;
				// want as much debug as possible
				options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
				// this lets you source-level debug/step shaders in renderdoc
				if (physDev->getLimits().shaderNonSemanticInfo)
					options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_NON_SEMANTIC_BIT;
				// if you don't set the logger and source identifier you'll have no meaningful errors
				options.preprocessorOptions.sourceIdentifier = "embedded.comp.hlsl";
				options.preprocessorOptions.logger = logger.get();
				cpuShader = compiler->compileToSPIRV(source.c_str(), options);

				if (!cpuShader)
				{
					logFail("Failed to compile following HLSL Shader:\n%s\n",source);
					return false;
				}
			}

			// Note how each ILogicalDevice method takes a smart-pointer r-value, si that the GPU objects refcount their dependencies
			smart_refctd_ptr<nbl::video::IGPUShader> shader = device->createShader(std::move(cpuShader));
			if (!shader)
			{
				logFail("Failed to create a GPU Shader, seems the Driver doesn't like the SPIR-V we're feeding it!\n");
				return false;
			}

			// the simplest example would have used push constants and BDA, but RenderDoc's debugging of that sucks, so I'll demonstrate "classical" binding of buffers with descriptors
			nbl::video::IGPUDescriptorSetLayout::SBinding bindings[1] = {
				{
					.binding=0,
					.type=nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags=IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, // not is not the time for descriptor indexing
					.stageFlags=IGPUShader::ESS_COMPUTE,
					.count=1,
					.samplers=nullptr // irrelevant for a buffer
				}
			};
			smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = device->createDescriptorSetLayout(bindings,bindings+1);
			if (!dsLayout)
			{
				logFail("Failed to create a Descriptor Layout!\n");
				return false;
			}

			// Nabla actually has facilities for SPIR-V Reflection and "guessing" pipeline layouts for a given SPIR-V which we'll cover in a different example
			smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout = device->createPipelineLayout(nullptr,nullptr,std::move(dsLayout));
			if (!pplnLayout)
			{
				logFail("Failed to create a Pipeline Layout!\n");
				return false;
			}

			// we use strong typing on the pipelines, since there's no reason to polymorphically switch between different pipelines
			smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline = device->createComputePipeline(nullptr,smart_refctd_ptr(pplnLayout),std::move(shader));

			// Device Memory is just an untyped memory allocation that can back a GPU Buffer or a GPU Image
			nbl::video::IDeviceMemoryAllocator::SMemoryOffset allocation = {};
			{
				constexpr size_t BufferSize = sizeof(uint32_t)*WorkgroupSize*WorkgroupCount;

				//!
				nbl::video::IGPUBuffer::SCreationParams params = {};
				params.size = BufferSize;
				params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
				smart_refctd_ptr<IGPUBuffer> outputBuff = device->createBuffer(std::move(params));
				if (!outputBuff)
				{
					logFail("Failed to create a GPU Buffer of size %d!\n",params.size);
					return false;
				}

				// This is a cool utility you can use instead of counting up how much of each descriptor type you need to N_i allocate descriptor sets with layout L_i from a single pool
				smart_refctd_ptr<nbl::video::IDescriptorPool> pool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,&dsLayout.get(),&dsLayout.get()+1);

				auto ds = pool->createDescriptorSet(std::move(dsLayout));
				// we still use Vulkan 1.0 descriptor update style, could move to Update Templates but Descriptor Buffer seems just around the corner
				{
					IGPUDescriptorSet::SDescriptorInfo info[1];
					info[0].desc = smart_refctd_ptr(outputBuff); // bad API, too late to change, should just take raw-pointers since not consumed
					info[0].info.buffer = {.offset=0,.size=BufferSize};
					IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
						{.dstSet=ds.get(),.binding=0,.arrayElement=0,.count=1,.descriptorType=IDescriptor::E_TYPE::ET_STORAGE_BUFFER,.info=info}
					};
					// we actually refcount all descriptors use by a descriptor set and you can pretty much fire and forget
					device->updateDescriptorSets(1u,writes,0u,nullptr);
				}

				//!
				nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = outputBuff->getMemoryReqs();
				reqs.memoryTypeBits &= physDev->getHostVisibleMemoryTypeBits();

				//!
				allocation = device->allocate(reqs,outputBuff.get(),IDeviceMemoryAllocation::EMAF_NONE);
				if (!allocation.isValid())
				{
					logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");
					return false;
				}
			}

			// To be able to read the contents of the buffer we need to map its memory
			// P.S. Nabla mandates Persistent Memory Mappings on all backends (but not coherent memory types)
			const IDeviceMemoryAllocation::MappedMemoryRange memoryRange(allocation.memory.get(),0ull,allocation.memory->getAllocationSize());
			auto ptr = device->mapMemory(memoryRange,IDeviceMemoryAllocation::EMCAF_READ);
			if (!ptr)
			{
				logFail("Failed to map the Device Memory!\n");
				return false;
			}

			//!
			smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
			{
				smart_refctd_ptr<IGPUCommandPool> cmdpool = device->createCommandPool(famix,IGPUCommandPool::ECF_TRANSIENT_BIT);
				if (!device->createCommandBuffers(cmdpool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf))
				{
					logFail("Failed to create Command Buffers!\n");
					return false;
				}
			}

			// if the mapping is not coherent the range needs to be invalidated
			if (!allocation.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
				device->invalidateMappedMemoryRanges(1,&memoryRange);

			// a simple test to check we got the right thing back
			auto buffData = reinterpret_cast<const uint32_t*>(ptr);
			for (auto i=0; i<WorkgroupSize*WorkgroupCount; i++)
			if (buffData[i]!=i)
			{
				logFail("DWORD at position %d doesn't match!\n",i);
				return false;
			}
			device->unmapMemory(allocation.memory.get());

			return true;
		}

		// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
		void workLoopBody() override {}

		// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
		bool keepRunning() override {return false;}

		// normally you'd deinitialize everything here but this is a single shot application
		bool onAppTerminated() override {return true;}

};


NBL_MAIN_FUNC(HelloComputeApp)