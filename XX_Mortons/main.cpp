// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include "app_resources/common.hlsl"
#include <bitset>

// Right now the test only checks that HLSL compiles the file
constexpr bool TestHLSL = true;

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

// this time instead of defining our own `int main()` we derive from `nbl::system::IApplicationFramework` to play "nice" wil all platforms
class MortonTestApp final : public application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = application_templates::MonoDeviceApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

		using morton_t = nbl::hlsl::morton::code<int32_t, 3>;
		using vector_t = nbl::hlsl::vector<int32_t, 3>;
		using unsigned_morton_t = nbl::hlsl::morton::code<uint32_t, 3>;
		using unsigned_vector_t = nbl::hlsl::vector<uint32_t, 3>;
		using bool_vector_t = nbl::hlsl::vector<bool, 3>;

		inline core::smart_refctd_ptr<video::IGPUShader> createShader(
			const char* includeMainName)
		{
			std::string prelude = "#include \"";
			auto CPUShader = core::make_smart_refctd_ptr<ICPUShader>((prelude + includeMainName + "\"\n").c_str(), IShader::E_SHADER_STAGE::ESS_COMPUTE, IShader::E_CONTENT_TYPE::ECT_HLSL, includeMainName);
			assert(CPUShader);
			return m_device->createShader(CPUShader.get());
		}
	public:
		MortonTestApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			// ----------------------------------------------- CPP TESTS ----------------------------------------------------------------------
			
			// Coordinate extraction and whole vector decode tests
			{
				morton_t morton(vector_t(-1011, 765, 248));
				unsigned_morton_t unsignedMorton(unsigned_vector_t(154, 789, 1011));

				assert(morton.getCoordinate(0) == -1011 && morton.getCoordinate(1) == 765 && morton.getCoordinate(2) == 248);
				assert(unsignedMorton.getCoordinate(0) == 154u && unsignedMorton.getCoordinate(1) == 789u && unsignedMorton.getCoordinate(2) == 1011u);

				assert(static_cast<vector_t>(morton) == vector_t(-1011, 765, 248) && static_cast<unsigned_vector_t>(unsignedMorton) == unsigned_vector_t(154, 789, 1011));
			}

			// ***********************************************************************************************************************************
			// ************************************************* Arithmetic operator tests *******************************************************
			// ***********************************************************************************************************************************
			
			//  ----------------------------------------------------------------------------------------------------
			//  --------------------------------------- ADDITION ---------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// ---------------------------------------- Signed -----------------------------------------------------
			
			// No overflow
			assert(static_cast<vector_t>(morton_t(vector_t(-1011, 765, 248)) + morton_t(vector_t(1000, -985, 200))) == vector_t(-11, -220, 448));
			
			// Type 1 overflow: Addition of representable coordinates goes out of range
			assert(static_cast<vector_t>(morton_t(vector_t(-900, 70, 500)) + morton_t(vector_t(-578, -50, 20))) == vector_t(570, 20, -504));

			// Type 2 overflow: Addition of irrepresentable range gives correct result
			assert(static_cast<vector_t>(morton_t(vector_t(54, 900, -475)) + morton_t(vector_t(46, -1437, 699))) == vector_t(100, -537, 224));

			// ---------------------------------------- Unsigned -----------------------------------------------------

			// No overflow
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(382, 910, 543)) + unsigned_morton_t(unsigned_vector_t(1563, 754, 220))) == unsigned_vector_t(1945, 1664, 763));

			// Type 1 overflow: Addition of representable coordinates goes out of range
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(382, 910, 543)) + unsigned_morton_t(unsigned_vector_t(2000, 2000, 1000))) == unsigned_vector_t(334, 862, 519));

			// Type 2 overflow: Addition of irrepresentable range gives correct result
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(382, 910, 543)) + unsigned_morton_t(unsigned_vector_t(-143, -345, -233))) == unsigned_vector_t(239, 565, 310));

			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- SUBTRACTION -------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// ---------------------------------------- Signed -----------------------------------------------------

			// No overflow
			assert(static_cast<vector_t>(morton_t(vector_t(1000, 764, -365)) - morton_t(vector_t(834, -243, 100))) == vector_t(166, 1007, -465));

			// Type 1 overflow: Subtraction of representable coordinates goes out of range
			assert(static_cast<vector_t>(morton_t(vector_t(-900, 70, 500)) - morton_t(vector_t(578, -50, -20))) == vector_t(570, 120, -504));

			// Type 2 overflow: Subtraction of irrepresentable range gives correct result
			assert(static_cast<vector_t>(morton_t(vector_t(54, 900, -475)) - morton_t(vector_t(-46, 1437, -699))) == vector_t(100, -537, 224));

			// ---------------------------------------- Unsigned -----------------------------------------------------

			// No overflow
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(382, 910, 543)) - unsigned_morton_t(unsigned_vector_t(322, 564, 299))) == unsigned_vector_t(60, 346, 244));

			// Type 1 overflow: Subtraction of representable coordinates goes out of range
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(382, 910, 543)) - unsigned_morton_t(unsigned_vector_t(2000, 2000, 1000))) == unsigned_vector_t(430, 958, 567));

			// Type 2 overflow: Subtraction of irrepresentable range gives correct result
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(54, 900, 475)) - unsigned_morton_t(unsigned_vector_t(-865, -100, -10))) == unsigned_vector_t(919, 1000, 485));


			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- UNARY NEGATION ----------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// Only makes sense for signed
			assert(static_cast<vector_t>(- morton_t(vector_t(-1024, 543, -475))) == vector_t(-1024, -543, 475));

			// ***********************************************************************************************************************************
			// ************************************************* Comparison operator tests *******************************************************
			// ***********************************************************************************************************************************

			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- OPERATOR< ---------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// Signed
			
			// Same sign, negative
			assert(morton_t(vector_t(-954, -455, -333)) < morton_t(vector_t(-433, -455, -433)) == bool_vector_t(true, false, false));
			// Same sign, positive
			assert(morton_t(vector_t(954, 455, 333)) < morton_t(vector_t(433, 455, 433)) == bool_vector_t(false, false, true));
			// Differing signs
			assert(morton_t(vector_t(954, -32, 0)) < morton_t(vector_t(-44, 0, -1)) == bool_vector_t(false, true, false));

			// Unsigned
			assert(unsigned_morton_t(unsigned_vector_t(239, 435, 66)) < unsigned_morton_t(unsigned_vector_t(240, 435, 50)) == bool_vector_t(true, false, false));

			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- OPERATOR<= --------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// Signed

			// Same sign, negative
			assert(morton_t(vector_t(-954, -455, -333)) <= morton_t(vector_t(-433, -455, -433)) == bool_vector_t(true, true, false));
			// Same sign, positive
			assert(morton_t(vector_t(954, 455, 333)) <= morton_t(vector_t(433, 455, 433)) == bool_vector_t(false, true, true));
			// Differing signs
			assert(morton_t(vector_t(954, -32, 0)) <= morton_t(vector_t(-44, 0, -1)) == bool_vector_t(false, true, false));

			// Unsigned
			assert(unsigned_morton_t(unsigned_vector_t(239, 435, 66)) <= unsigned_morton_t(unsigned_vector_t(240, 435, 50)) == bool_vector_t(true, true, false));

			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- OPERATOR> ---------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// Signed

			// Same sign, negative
			assert(morton_t(vector_t(-954, -455, -333)) > morton_t(vector_t(-433, -455, -433)) == bool_vector_t(false, false, true));
			// Same sign, positive
			assert(morton_t(vector_t(954, 455, 333)) > morton_t(vector_t(433, 455, 433)) == bool_vector_t(true, false, false));
			// Differing signs
			assert(morton_t(vector_t(954, -32, 0)) > morton_t(vector_t(-44, 0, -1)) == bool_vector_t(true, false, true));

			// Unsigned
			assert(unsigned_morton_t(unsigned_vector_t(239, 435, 66)) > unsigned_morton_t(unsigned_vector_t(240, 435, 50)) == bool_vector_t(false, false, true));

			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- OPERATOR>= --------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// Signed

			// Same sign, negative
			assert(morton_t(vector_t(-954, -455, -333)) >= morton_t(vector_t(-433, -455, -433)) == bool_vector_t(false, true, true));
			// Same sign, positive
			assert(morton_t(vector_t(954, 455, 333)) >= morton_t(vector_t(433, 455, 433)) == bool_vector_t(true, true, false));
			// Differing signs
			assert(morton_t(vector_t(954, -32, 0)) >= morton_t(vector_t(-44, 0, -1)) == bool_vector_t(true, false, true));

			// Unsigned
			assert(unsigned_morton_t(unsigned_vector_t(239, 435, 66)) >= unsigned_morton_t(unsigned_vector_t(240, 435, 50)) == bool_vector_t(false, true, true));


			if(!TestHLSL)
				return true;









			// ----------------------------------------------- HLSL COMPILATION + OPTIONAL TESTS ----------------------------------------------
			auto shader = createShader("app_resources/shader.hlsl");

			// Create massive upload/download buffers
			constexpr uint32_t DownstreamBufferSize = sizeof(unsigned_scalar_t) << 23;

			m_utils = make_smart_refctd_ptr<IUtilities>(smart_refctd_ptr(m_device), smart_refctd_ptr(m_logger), DownstreamBufferSize);
			if (!m_utils)
				return logFail("Failed to create Utilities!");
			m_downStreamingBuffer = m_utils->getDefaultDownStreamingBuffer();
			m_downStreamingBufferAddress = m_downStreamingBuffer->getBuffer()->getDeviceAddress();

			// Create device-local buffer
			{
				IGPUBuffer::SCreationParams deviceLocalBufferParams = {};

				IQueue* const queue = getComputeQueue();
				uint32_t queueFamilyIndex = queue->getFamilyIndex();

				deviceLocalBufferParams.queueFamilyIndexCount = 1;
				deviceLocalBufferParams.queueFamilyIndices = &queueFamilyIndex;
				deviceLocalBufferParams.size = sizeof(unsigned_scalar_t) * bufferSize;
				deviceLocalBufferParams.usage = nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT | nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;

				m_deviceLocalBuffer = m_device->createBuffer(std::move(deviceLocalBufferParams));
				auto mreqs = m_deviceLocalBuffer->getMemoryReqs();
				mreqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto gpubufMem = m_device->allocate(mreqs, m_deviceLocalBuffer.get(), IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_DEVICE_ADDRESS_BIT);

				m_deviceLocalBufferAddress = m_deviceLocalBuffer.get()->getDeviceAddress();
			}

			const nbl::asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,.offset = 0,.size = sizeof(PushConstantData) };

			{
				auto layout = m_device->createPipelineLayout({ &pcRange,1 });
				IGPUComputePipeline::SCreationParams params = {};
				params.layout = layout.get();
				params.shader.shader = shader.get();
				params.shader.requiredSubgroupSize = static_cast<IGPUShader::SSpecInfo::SUBGROUP_SIZE>(hlsl::findMSB(m_physicalDevice->getLimits().maxSubgroupSize));
				params.shader.requireFullSubgroups = true;
				if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_pipeline))
					return logFail("Failed to create compute pipeline!\n");
			}

			const auto& deviceLimits = m_device->getPhysicalDevice()->getLimits();
			// The ranges of non-coherent mapped memory you flush or invalidate need to be aligned. You'll often see a value of 64 reported by devices
			// which just happens to coincide with a CPU cache line size. So we ask our streaming buffers during allocation to give us properly aligned offsets.
			// Sidenote: For SSBOs, UBOs, BufferViews, Vertex Buffer Bindings, Acceleration Structure BDAs, Shader Binding Tables, Descriptor Buffers, etc.
			// there is also a requirement to bind buffers at offsets which have a certain alignment. Memory binding to Buffers and Images also has those.
			// We'll align to max of coherent atom size even if the memory is coherent,
			// and we also need to take into account BDA shader loads need to be aligned to the type being loaded.
			m_alignment = core::max(deviceLimits.nonCoherentAtomSize, alignof(float));

			// Semaphor used here to know the FFT is done before download
			m_timeline = m_device->createSemaphore(semaphorValue);

			IQueue* const queue = getComputeQueue();

			const uint32_t inputSize = sizeof(unsigned_scalar_t) * bufferSize;

			// Just need a single suballocation in this example
			const uint32_t AllocationCount = 1;

			// We always just wait till an allocation becomes possible (during allocation previous "latched" frees get their latch conditions polled)
			// Freeing of Streaming Buffer Allocations can and should be deferred until an associated polled event signals done (more on that later).
			std::chrono::steady_clock::time_point waitTill(std::chrono::years(45));

			// finally allocate our output range
			const uint32_t outputSize = inputSize;

			auto outputOffset = m_downStreamingBuffer->invalid_value;
			m_downStreamingBuffer->multi_allocate(waitTill, AllocationCount, &outputOffset, &outputSize, &m_alignment);

			smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
			{
				smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf)) {
					return logFail("Failed to create Command Buffers!\n");
				}
				cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmdbuf,1 }, core::smart_refctd_ptr(m_logger));
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				cmdbuf->bindComputePipeline(m_pipeline.get());
				// This is the new fun part, pushing constants
				const PushConstantData pc = { .deviceBufferAddress = m_deviceLocalBufferAddress };
				cmdbuf->pushConstants(m_pipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
				// Remember we do a single workgroup per 1D array in these parts
				cmdbuf->dispatch(1, 1, 1);

				// Pipeline barrier: wait for FFT shader to be done before copying to downstream buffer 
				IGPUCommandBuffer::SPipelineBarrierDependencyInfo pipelineBarrierInfo = {};

				decltype(pipelineBarrierInfo)::buffer_barrier_t barrier = {};
				pipelineBarrierInfo.bufBarriers = { &barrier, 1u };

				barrier.range.buffer = m_deviceLocalBuffer;

				barrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				barrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
				barrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
				barrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS;

				cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), pipelineBarrierInfo);

				IGPUCommandBuffer::SBufferCopy copyInfo = {};
				copyInfo.srcOffset = 0;
				copyInfo.dstOffset = 0;
				copyInfo.size = m_deviceLocalBuffer->getSize();
				cmdbuf->copyBuffer(m_deviceLocalBuffer.get(), m_downStreamingBuffer->getBuffer(), 1, &copyInfo);
				cmdbuf->end();
			}

			semaphorValue++;
			{
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo =
				{
					.cmdbuf = cmdbuf.get()
				};
				const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo =
				{
					.semaphore = m_timeline.get(),
					.value = semaphorValue,
					.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
				};

				const IQueue::SSubmitInfo submitInfo = {
					.waitSemaphores = {},
					.commandBuffers = {&cmdbufInfo,1},
					.signalSemaphores = {&signalInfo,1}
				};

				m_api->startCapture();
				queue->submit({ &submitInfo,1 });
				m_api->endCapture();
			}

			// We let all latches know what semaphore and counter value has to be passed for the functors to execute
			const ISemaphore::SWaitInfo futureWait = { m_timeline.get(),semaphorValue };

			// Now a new and even more advanced usage of the latched events, we make our own refcounted object with a custom destructor and latch that like we did the commandbuffer.
			// Instead of making our own and duplicating logic, we'll use one from IUtilities meant for down-staging memory.
			// Its nice because it will also remember to invalidate our memory mapping if its not coherent.
			auto latchedConsumer = make_smart_refctd_ptr<IUtilities::CDownstreamingDataConsumer>(
				IDeviceMemoryAllocation::MemoryRange(outputOffset, outputSize),
				// Note the use of capture by-value [=] and not by-reference [&] because this lambda will be called asynchronously whenever the event signals
				[=](const size_t dstOffset, const void* bufSrc, const size_t size)->void
				{
					// The unused variable is used for letting the consumer know the subsection of the output we've managed to download
					// But here we're sure we can get the whole thing in one go because we allocated the whole range ourselves.
					assert(dstOffset == 0 && size == outputSize);

					std::cout << "Begin array GPU\n";
					unsigned_scalar_t* const data = reinterpret_cast<unsigned_scalar_t*>(const_cast<void*>(bufSrc));
					std::cout << std::bitset<32>(data[0]) << "\n";
					/*
					for (auto i = 0u; i < bufferSize; i++) {
						std::cout << std::bitset<32>(data[i]) << "\n";
					}
					*/
					std::cout << "\nEnd array GPU\n";
				},
				// Its also necessary to hold onto the commandbuffer, even though we take care to not reset the parent pool, because if it
				// hits its destructor, our automated reference counting will drop all references to objects used in the recorded commands.
				// It could also be latched in the upstreaming deallocate, because its the same fence.
				std::move(cmdbuf), m_downStreamingBuffer
			);
			// We put a function we want to execute 
			m_downStreamingBuffer->multi_deallocate(AllocationCount, &outputOffset, &outputSize, futureWait, &latchedConsumer.get());

			return true;
		}

		// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
		void workLoopBody() override {}

		// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
		bool keepRunning() override {return false;}

		// Cleanup
		bool onAppTerminated() override
		{
			// Need to make sure that there are no events outstanding if we want all lambdas to eventually execute before `onAppTerminated`
			// (the destructors of the Command Pool Cache and Streaming buffers will still wait for all lambda events to drain)
			if (TestHLSL)
			{
				while (m_downStreamingBuffer->cull_frees()) {}
			}
			return device_base_t::onAppTerminated();
		}

	private:
		smart_refctd_ptr<IGPUComputePipeline> m_pipeline;

		smart_refctd_ptr<nbl::video::IUtilities> m_utils;

		StreamingTransientDataBufferMT<>* m_downStreamingBuffer;
		smart_refctd_ptr<nbl::video::IGPUBuffer> m_deviceLocalBuffer;

		// These are Buffer Device Addresses
		uint64_t m_downStreamingBufferAddress;
		uint64_t m_deviceLocalBufferAddress;

		uint32_t m_alignment;

		smart_refctd_ptr<ISemaphore> m_timeline;
		uint64_t semaphorValue = 0;
};


NBL_MAIN_FUNC(MortonTestApp)