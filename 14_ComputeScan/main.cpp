#include "nbl/application_templates/BasicMultiQueueApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include <chrono>
#include <random>

using namespace nbl;
using namespace core;
using namespace asset;
using namespace system;
using namespace video;

class ComputeScanApp final : public application_templates::BasicMultiQueueApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::BasicMultiQueueApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

public:
	ComputeScanApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!device_base_t::onAppInitialized(std::move(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		computeQueue = getComputeQueue();

		// Create (an almost) 128MB input buffer
		constexpr auto in_size = 128u << 5u;
		constexpr auto in_count = 1418;//in_size / sizeof(uint32_t) - 23u;

		m_logger->log("Input element count: %d", ILogger::ELL_PERFORMANCE, in_count);

		inputData = new uint32_t[in_count];
		{
			std::random_device random_device;
			std::mt19937 generator(random_device());
			std::uniform_int_distribution<uint32_t> distribution(0u, ~0u);
			for (auto i = 0u; i < in_count; i++)
				inputData[i] = distribution(generator) % 100000;
		}
		auto minSSBOAlign = m_physicalDevice->getLimits().minSSBOAlignment;
		constexpr auto begin = in_count / 4 + 118;
		assert(((begin * sizeof(uint32_t)) & (minSSBOAlign - 1u)) == 0u);
		constexpr auto end = in_count * 3 / 4 - 78;
		assert(((end * sizeof(uint32_t)) & (minSSBOAlign - 1u)) == 0u);
		constexpr auto elementCount = end - begin;

		smart_refctd_ptr<IGPUBuffer> gpuinputDataBuffer;
		{
			IGPUBuffer::SCreationParams inputDataBufferCreationParams = {};
			inputDataBufferCreationParams.size = sizeof(uint32_t) * in_count; // TODO Declare the element data type in the shader?
			inputDataBufferCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_SRC_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
			gpuinputDataBuffer = m_utils->createFilledDeviceLocalBufferOnDedMem(
				{ .queue = getTransferUpQueue() },
				std::move(inputDataBufferCreationParams),
				inputData
			);
		}
		SBufferRange<IGPUBuffer> in_gpu_range = { begin * sizeof(uint32_t), elementCount * sizeof(uint32_t), gpuinputDataBuffer };

		const auto scanType = video::CScanner::EST_EXCLUSIVE;
		auto scanner = m_utils->getDefaultScanner();

		CScanner::DefaultPushConstants scan_push_constants;
		CScanner::DispatchInfo scan_dispatch_info;
		scanner->buildParameters(elementCount, scan_push_constants, scan_dispatch_info);

		IGPUBuffer::SCreationParams params = { scan_push_constants.scanParams.getScratchSize(), bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT };
		SBufferRange<IGPUBuffer> scratch_gpu_range = {0u, params.size, m_device->createBuffer(std::move(params)) };
		{
			auto memReqs = scratch_gpu_range.buffer->getMemoryReqs();
			memReqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			auto scratchMem = m_device->allocate(memReqs, scratch_gpu_range.buffer.get());
		}

		auto scan_pipeline = scanner->getDefaultPipeline(scanType, CScanner::EDT_UINT, CScanner::EO_ADD, scan_push_constants.scanParams.getScratchSize());
		auto dsLayout = scanner->getDefaultDescriptorSetLayout();
		auto dsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &dsLayout, 1 });
		auto ds = dsPool->createDescriptorSet(core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayout));
		scanner->updateDescriptorSet(m_device.get(), ds.get(), in_gpu_range, scratch_gpu_range);
		
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(computeQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmdbuf,1 }))
			{
				logFail("Failed to create Command Buffers!\n");
				return false;
			}
		}

		cmdbuf->begin(IGPUCommandBuffer::USAGE::SIMULTANEOUS_USE_BIT); // (REVIEW): not sure about this
		cmdbuf->fillBuffer(scratch_gpu_range, 0u);
		cmdbuf->bindComputePipeline(scan_pipeline);
		auto pipeline_layout = scan_pipeline->getLayout();
		cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, pipeline_layout, 0u, 1u, &ds.get());
		scanner->dispatchHelper(cmdbuf.get(), pipeline_layout, scan_push_constants, scan_dispatch_info, 0u, nullptr, 0u, nullptr);
		cmdbuf->end();

		core::smart_refctd_ptr<ISemaphore> semaphore = m_device->createSemaphore(0);
		// submit
		IQueue::SSubmitInfo::SSemaphoreInfo semInfo[1] = { {
			.semaphore = semaphore.get(),
			.value = 1,
			// just as we've outputted all pixels, signal
			.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
		} };
		{
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = { {
				.cmdbuf = cmdbuf.get()
			} };
				
			const IQueue::SSubmitInfo infos[1] = { {
				.commandBuffers = commandBuffers,
				.signalSemaphores = semInfo
			} };

			computeQueue->startCapture();
			if (computeQueue->submit(infos) != IQueue::RESULT::SUCCESS) {
				m_logger->log("Submission failure", system::ILogger::ELL_ERROR);
			}
			computeQueue->endCapture();
		}

		// cpu counterpart
		auto cpu_begin = inputData + begin;
		m_logger->log("CPU scan begin", system::ILogger::ELL_PERFORMANCE);

		auto start = std::chrono::high_resolution_clock::now();
		switch (scanType)
		{
		case video::CScanner::EST_INCLUSIVE:
			std::inclusive_scan(cpu_begin, inputData + end, cpu_begin);
			break;
		case video::CScanner::EST_EXCLUSIVE:
			std::exclusive_scan(cpu_begin, inputData + end, cpu_begin, 0u);
			break;
		default:
			assert(false);
			exit(0xdeadbeefu);
			break;
		}
		auto stop = std::chrono::high_resolution_clock::now();

		m_logger->log("CPU scan end. Time taken: %d us", system::ILogger::ELL_PERFORMANCE, std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());

		// wait for the gpu impl to complete
		const ISemaphore::SWaitInfo cmdbufDonePending[] = {{
			.semaphore = semaphore.get(),
			.value = 1
		}};
		if (m_device->blockForSemaphores(cmdbufDonePending) != ISemaphore::WAIT_RESULT::SUCCESS) {
			m_logger->log("Blocking for operation semaphore failed", ILogger::ELL_ERROR);
			return false;
		}
		
		{
			IGPUBuffer::SCreationParams params = {};
			params.size = in_gpu_range.size;
			params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT;
			// (REVIEW): Check if this new download_buffer is needed or if we can directly read from the gpu_input buffer
			auto downloaded_buffer = m_device->createBuffer(std::move(params));
			auto memReqs = downloaded_buffer->getMemoryReqs();
			memReqs.memoryTypeBits &= m_physicalDevice->getDownStreamingMemoryTypeBits();
			auto queriesMem = m_device->allocate(memReqs, downloaded_buffer.get());
			{
				// (REVIEW): Maybe we can just reset the cmdbuf we already have?
				core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
				{
					auto cmdPool = m_device->createCommandPool(computeQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::NONE);
					cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmdbuf , 1}, core::smart_refctd_ptr(m_logger));
				}
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);  // TODO: Reset Frame's CommandPool
				IGPUCommandBuffer::SBufferCopy region;
				region.srcOffset = in_gpu_range.offset;
				region.dstOffset = 0u;
				region.size = in_gpu_range.size;
				cmdbuf->copyBuffer(in_gpu_range.buffer.get(), downloaded_buffer.get(), 1u, &region);
				cmdbuf->end();

				{
					const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = { {
						.cmdbuf = cmdbuf.get()
					} };

					semInfo[0].value = 2;
					const IQueue::SSubmitInfo infos[1] = { {
						.commandBuffers = commandBuffers,
						.signalSemaphores = semInfo
					} };

					if (computeQueue->submit(infos) != IQueue::RESULT::SUCCESS) {
						m_logger->log("Download submission failure", system::ILogger::ELL_ERROR);
					}

					const ISemaphore::SWaitInfo cmdbufDonePending[] = { {
						.semaphore = semaphore.get(),
						.value = 2
					} };
					if (m_device->blockForSemaphores(cmdbufDonePending) != ISemaphore::WAIT_RESULT::SUCCESS) {
						m_logger->log("Blocking for download semaphore failed", ILogger::ELL_ERROR);
						return false;
					}
				}
			}

			auto mem = const_cast<video::IDeviceMemoryAllocation*>(downloaded_buffer->getBoundMemory().memory);
			{
				ILogicalDevice::MappedMemoryRange range;
				{
					range.memory = mem;
					range.offset = 0u;
					range.length = in_gpu_range.size;
				}
				mem->map({ .offset = range.offset, .length = range.length }, video::IDeviceMemoryAllocation::EMCAF_READ);
			}
			auto gpu_begin = reinterpret_cast<uint32_t*>(mem->getMappedPointer());
			for (auto i = 0u; i < elementCount; i++)
			{
				if (gpu_begin[i] != cpu_begin[i])
					_NBL_DEBUG_BREAK_IF(true);
			}
			m_logger->log("Result Comparison Test Passed", system::ILogger::ELL_PERFORMANCE);
			scanSuccess = true;
		}

		delete[] inputData;

		return true;
	}
	
	//virtual video::SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	//{
	//	video::SPhysicalDeviceFeatures retval = {};

	//	retval.bufferDeviceAddress = true;
	//	retval.subgroupBroadcastDynamicId = true;
	//	retval.shaderSubgroupExtendedTypes = true;
	//	// TODO: actually need to implement this and set it on the pipelines
	//	retval.computeFullSubgroups = true;
	//	retval.subgroupSizeControl = true;

	//	return retval;
	//}

	virtual bool onAppTerminated() override
	{
		m_logger->log("==========Result==========", ILogger::ELL_INFO);
		m_logger->log("Scan Success: %s", ILogger::ELL_INFO, scanSuccess?"true":"false");
		delete[] inputData;
		return true;
	}

	// the unit test is carried out on init
	void workLoopBody() override {}

	bool keepRunning() override { return false; }

private:
	void logTestOutcome(bool passed, uint32_t workgroupSize)
	{
		if (passed)
			m_logger->log("Passed test #%u", ILogger::ELL_INFO, workgroupSize);
		else
		{
			m_logger->log("Failed test #%u", ILogger::ELL_ERROR, workgroupSize);
		}
	}

	IQueue* computeQueue;
	uint32_t* inputData = nullptr;
	smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
	smart_refctd_ptr<IGPUPipelineLayout> pipelineLayout;
	smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
	smart_refctd_ptr<ICPUBuffer> resultsBuffer;

	bool scanSuccess = false;
};

NBL_MAIN_FUNC(ComputeScanApp)