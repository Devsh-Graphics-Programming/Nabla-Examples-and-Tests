// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/examples/examples.hpp"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;

#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"


// Simple showcase of how to run Bitonic Sort on a 1D array using workgroup operations
class BitonicSortApp final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t = BuiltinResourcesApplication;

	smart_refctd_ptr<IGPUComputePipeline> m_pipeline;
	smart_refctd_ptr<IGPUPipelineLayout> m_layout;

	smart_refctd_ptr<nbl::video::IUtilities> m_utils;

	nbl::video::StreamingTransientDataBufferMT<>* m_upStreamingBuffer;
	StreamingTransientDataBufferMT<>* m_downStreamingBuffer;
	smart_refctd_ptr<nbl::video::IGPUBuffer> m_deviceLocalBuffer;

	// These are Buffer Device Addresses
	uint64_t m_upStreamingBufferAddress;
	uint64_t m_downStreamingBufferAddress;
	uint64_t m_deviceLocalBufferAddress;

	uint32_t m_alignment;

	smart_refctd_ptr<ISemaphore> m_timeline;
	uint64_t semaphoreValue = 0;

public:

	BitonicSortApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		// Load shader
		auto prepShader = [&](const core::string& path) -> smart_refctd_ptr<IShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = "";
				auto assetBundle = m_assetMgr->getAsset(path, lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
				{
					logFail("Could not load shader!");
					return nullptr;
				}

				auto source = IAsset::castDown<IShader>(assets[0]);
				assert(source);

				auto shader = m_device->compileShader({ source.get() });
				if (!shader)
				{
					logFail("Creation of Bitonic Sort Shader failed!");
					return nullptr;
				}
				return shader;
			};

		auto bitonicSortShader = prepShader("app_resources/bitonic_sort_shader.comp.hlsl");
		if (!bitonicSortShader)
			return logFail("Failed to compile bitonic sort shader!");

		m_utils = video::IUtilities::create(smart_refctd_ptr(m_device), smart_refctd_ptr(m_logger));
		if (!m_utils)
			return logFail("Failed to create Utilities!");
		m_upStreamingBuffer = m_utils->getDefaultUpStreamingBuffer();
		m_downStreamingBuffer = m_utils->getDefaultDownStreamingBuffer();
		m_upStreamingBufferAddress = m_upStreamingBuffer->getBuffer()->getDeviceAddress();
		m_downStreamingBufferAddress = m_downStreamingBuffer->getBuffer()->getDeviceAddress();

		// Create device-local buffer
		{
			IGPUBuffer::SCreationParams deviceLocalBufferParams = {};

			IQueue* const queue = getComputeQueue();
			uint32_t queueFamilyIndex = queue->getFamilyIndex();

			deviceLocalBufferParams.queueFamilyIndexCount = 1;
			deviceLocalBufferParams.queueFamilyIndices = &queueFamilyIndex;
			deviceLocalBufferParams.size = sizeof(uint32_t) * elementCount * 2;  // *2 because we store (key, value) pairs
			deviceLocalBufferParams.usage = nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT | nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | nbl::asset::IBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;

			m_deviceLocalBuffer = m_device->createBuffer(std::move(deviceLocalBufferParams));
			auto mreqs = m_deviceLocalBuffer->getMemoryReqs();
			mreqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto gpubufMem = m_device->allocate(mreqs, m_deviceLocalBuffer.get(), IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_DEVICE_ADDRESS_BIT);

			m_deviceLocalBufferAddress = m_deviceLocalBuffer.get()->getDeviceAddress();
		}

		const nbl::asset::SPushConstantRange pcRange = {
			.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
			.offset = 0,
			.size = sizeof(PushConstantData)
		};

		{
			m_layout = m_device->createPipelineLayout({ &pcRange,1 });
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = m_layout.get();
			params.shader.shader = bitonicSortShader.get();
			params.shader.entryPoint = "main";
			params.shader.requiredSubgroupSize = static_cast<IPipelineBase::SUBGROUP_SIZE>(hlsl::findMSB(m_physicalDevice->getLimits().maxSubgroupSize));
			params.cached.requireFullSubgroups = true;
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_pipeline))
				return logFail("Failed to create compute pipeline!\n");
		}

		const auto& deviceLimits = m_device->getPhysicalDevice()->getLimits();
		m_alignment = core::max(deviceLimits.nonCoherentAtomSize, alignof(uint32_t));

		m_timeline = m_device->createSemaphore(semaphoreValue);

		IQueue* const queue = getComputeQueue();

		const uint32_t inputSize = sizeof(uint32_t) * elementCount * 2;  // *2 because we store (key, value) pairs

		const uint32_t AllocationCount = 1;

		auto inputOffset = m_upStreamingBuffer->invalid_value;

		std::chrono::steady_clock::time_point waitTill(std::chrono::years(45));
		m_upStreamingBuffer->multi_allocate(waitTill, AllocationCount, &inputOffset, &inputSize, &m_alignment);

		{
			auto* const inputPtr = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(m_upStreamingBuffer->getBufferPointer()) + inputOffset);

			// Generate random input data
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::mt19937 g(seed);

			std::cout << "Input array: ";
			for (uint32_t i = 0; i < elementCount; i++) {
				uint32_t key = g() % 10000;
				uint32_t value = i; // Use index as value for stable sorting
				inputPtr[i * 2] = key;
				inputPtr[i * 2 + 1] = value;
				std::cout << "(" << key << "," << value << "), ";
				if ((i + 1) % 20 == 0) {
					std::cout << "\n";
				}
			}
			std::cout << "\nElement count: " << elementCount << "\n";

			// Always remember to flush!
			if (m_upStreamingBuffer->needsManualFlushOrInvalidate())
			{
				const auto bound = m_upStreamingBuffer->getBuffer()->getBoundMemory();
				const ILogicalDevice::MappedMemoryRange range(bound.memory, bound.offset + inputOffset, inputSize);
				m_device->flushMappedMemoryRanges(1, &range);
			}
		}

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

			const PushConstantData pc = { .deviceBufferAddress = m_deviceLocalBufferAddress };

			IGPUCommandBuffer::SBufferCopy copyInfo = {};
			copyInfo.srcOffset = inputOffset;
			copyInfo.dstOffset = 0;
			copyInfo.size = m_deviceLocalBuffer->getSize();
			cmdbuf->copyBuffer(m_upStreamingBuffer->getBuffer(), m_deviceLocalBuffer.get(), 1, &copyInfo);

			IGPUCommandBuffer::SPipelineBarrierDependencyInfo pipelineBarrierInfo1 = {};
			decltype(pipelineBarrierInfo1)::buffer_barrier_t barrier1 = {};
			pipelineBarrierInfo1.bufBarriers = { &barrier1, 1u };
			barrier1.range.buffer = m_deviceLocalBuffer;
			barrier1.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
			barrier1.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
			barrier1.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			barrier1.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS;
			cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), pipelineBarrierInfo1);

			cmdbuf->pushConstants(m_pipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);

			cmdbuf->dispatch(1, 1, 1);

			IGPUCommandBuffer::SPipelineBarrierDependencyInfo pipelineBarrierInfo2 = {};
			decltype(pipelineBarrierInfo2)::buffer_barrier_t barrier2 = {};
			pipelineBarrierInfo2.bufBarriers = { &barrier2, 1u };
			barrier2.range.buffer = m_deviceLocalBuffer;
			barrier2.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			barrier2.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
			barrier2.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
			barrier2.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS;
			cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS(0), pipelineBarrierInfo2);

			copyInfo.srcOffset = 0;
			copyInfo.dstOffset = outputOffset;
			cmdbuf->copyBuffer(m_deviceLocalBuffer.get(), m_downStreamingBuffer->getBuffer(), 1, &copyInfo);
			cmdbuf->end();
		}

		semaphoreValue++;
		{
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo =
			{
				.cmdbuf = cmdbuf.get()
			};
			const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo =
			{
				.semaphore = m_timeline.get(),
				.value = semaphoreValue,
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

		const ISemaphore::SWaitInfo futureWait = { m_timeline.get(),semaphoreValue };

		m_upStreamingBuffer->multi_deallocate(AllocationCount, &inputOffset, &inputSize, futureWait);

		auto latchedConsumer = make_smart_refctd_ptr<IUtilities::CDownstreamingDataConsumer>(
			IDeviceMemoryAllocation::MemoryRange(outputOffset, outputSize),
			[=](const size_t dstOffset, const void* bufSrc, const size_t size)->void
			{
				assert(dstOffset == 0 && size == outputSize);

				std::cout << "Sorted array: ";
				const uint32_t* const data = reinterpret_cast<const uint32_t*>(bufSrc);
				for (auto i = 0u; i < elementCount; i++) {
					uint32_t key = data[i * 2];
					uint32_t value = data[i * 2 + 1];
					std::cout << "(" << key << "," << value << "), ";
					if ((i + 1) % 20 == 0) {
						std::cout << "\n";
					}
				}
				std::cout << "\nElement count: " << elementCount << "\n";

				bool is_sorted = true;
				int32_t error_index = -1;
				for (uint32_t i = 1; i < elementCount; i++) {
					uint32_t prevKey = data[(i - 1) * 2];
					uint32_t currKey = data[i * 2];
					if (currKey < prevKey) {
						is_sorted = false;
						error_index = i;
						break;
					}
				}

				if (is_sorted) {
					std::cout << "Array is correctly sorted!\n";
				}
				else {
					std::cout << "Array is NOT sorted correctly!\n";
					std::cout << "Error at index " << error_index << ":\n";
					std::cout << "  Previous key [" << (error_index - 1) << "] = " << data[(error_index - 1) * 2] << "\n";
					std::cout << "  Current key  [" << error_index << "] = " << data[error_index * 2] << "\n";
					std::cout << "  (" << data[error_index * 2] << " < " << data[(error_index - 1) * 2] << " is WRONG!)\n";
				}
			},
			std::move(cmdbuf), m_downStreamingBuffer
		);
		m_downStreamingBuffer->multi_deallocate(AllocationCount, &outputOffset, &outputSize, futureWait, &latchedConsumer.get());

		return true;
	}

	bool keepRunning() override { return false; }

	void workLoopBody() override {}

	bool onAppTerminated() override
	{
		while (m_downStreamingBuffer->cull_frees()) {}
		return device_base_t::onAppTerminated();
	}
};

NBL_MAIN_FUNC(BitonicSortApp)