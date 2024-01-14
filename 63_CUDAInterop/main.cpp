// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/video/CCUDAHandler.h"
#include "nbl/video/CCUDASharedMemory.h"
#include "nbl/video/CCUDASharedSemaphore.h"

#include "../common/MonoDeviceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

/*
The start of the main function starts like in most other example. We ask the
user for the desired renderer and start it up.
*/

#define ASSERT_SUCCESS(expr) \
if (auto re = expr; CUDA_SUCCESS != re) { \
	const char* name = 0, *str = 0; \
	cu.pcuGetErrorName(re, &name); \
	cu.pcuGetErrorString(re, &str); \
	printf("%s:%d %s:\n\t%s\n", __FILE__, __LINE__, name, str); \
	abort(); \
}

#define ASSERT_SUCCESS_NV(expr) \
if (auto re = expr; NVRTC_SUCCESS != re) { \
	const char* str = cudaHandler->getNVRTCFunctionTable().pnvrtcGetErrorString(re); \
	printf("%s:%d %s\n", __FILE__, __LINE__, str); \
	abort(); \
}

constexpr uint32_t gridDim[3] = { 4096,1,1 };
constexpr uint32_t blockDim[3] = { 1024,1,1 };
size_t numElements = gridDim[0] * blockDim[0];
size_t size = sizeof(float) * numElements;

#ifndef _NBL_COMPILE_WITH_CUDA_
static_assert(false);
#endif

class CUDA2VKApp : public examples::MonoDeviceApplication
{
	using base_t = examples::MonoDeviceApplication;
public:
	// Generally speaking because certain platforms delay initialization from main object construction you should just forward and not do anything in the ctor
	using base_t::base_t;

	smart_refctd_ptr<CCUDAHandler> cudaHandler;
	smart_refctd_ptr<CCUDADevice> cudaDevice;

	IQueue* queue;

	std::array<smart_refctd_ptr<ICPUBuffer>, 2> cpubuffers;
	std::array<smart_refctd_ptr<CCUDASharedMemory>, 3> mem = {};
	smart_refctd_ptr<CCUDASharedSemaphore> cusema;

	smart_refctd_ptr<IGPUBuffer> importedbuf, stagingbuf, stagingbuf2;
	smart_refctd_ptr<IGPUImage> importedimg;
	smart_refctd_ptr<ISemaphore> sema;
	smart_refctd_ptr<IGPUCommandPool> commandPool;
	smart_refctd_ptr<IGPUCommandBuffer> cmd;


	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!base_t::onAppInitialized(std::move(system)))
			return false;

		{
			auto& limits = m_physicalDevice->getLimits();
			if (!limits.externalMemoryWin32 || !limits.externalFenceWin32 || !limits.externalSemaphoreWin32)
				return logFail("Physical device does not support the required extensions");
			
			cudaHandler = CCUDAHandler::create(system.get(), smart_refctd_ptr<ILogger>(m_logger));
			assert(cudaHandler);
			cudaDevice = cudaHandler->createDevice(smart_refctd_ptr_dynamic_cast<CVulkanConnection>(m_api), m_physicalDevice);
		}

		
		queue = base_t::getComputeQueue();
		
		createResources();

		smart_refctd_ptr<ICPUBuffer> ptx;
		{
			ISystem::future_t<smart_refctd_ptr<IFile>> fut;
			m_system->createFile(fut, "../vectorAdd_kernel.cu", IFileBase::ECF_READ);
			auto [ptx_, res] = cudaHandler->compileDirectlyToPTX(fut.copy().get(), cudaDevice->geDefaultCompileOptions());
			ASSERT_SUCCESS_NV(res);
			ptx = std::move(ptx_);
		}
		CUmodule   module;
		CUfunction kernel;
		CUstream   stream;

		auto& cu = cudaHandler->getCUDAFunctionTable();

		ASSERT_SUCCESS(cu.pcuModuleLoadDataEx(&module, ptx->getPointer(), 0u, nullptr, nullptr));
		ASSERT_SUCCESS(cu.pcuModuleGetFunction(&kernel, module, "vectorAdd"));
		ASSERT_SUCCESS(cu.pcuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

		launchKernel(kernel, stream);

		ASSERT_SUCCESS(cu.pcuStreamSynchronize(stream));
		ASSERT_SUCCESS(cu.pcuModuleUnload(module));
		ASSERT_SUCCESS(cu.pcuStreamDestroy_v2(stream));

		m_device->waitIdle();
		return true;
	}

	void createResources()
	{
		auto& cu = cudaHandler->getCUDAFunctionTable();

		for (auto& buf : cpubuffers)
			buf = make_smart_refctd_ptr<ICPUBuffer>(size);

		for (auto j = 0; j < 2; j++)
			for (auto i = 0; i < numElements; i++)
				reinterpret_cast<float*>(cpubuffers[j]->getPointer())[i] = rand() / float(RAND_MAX);



		ASSERT_SUCCESS(cudaDevice->createSharedMemory(&mem[0], { .size = size, .alignment = sizeof(float), .location = CU_MEM_LOCATION_TYPE_DEVICE }));
		ASSERT_SUCCESS(cudaDevice->createSharedMemory(&mem[1], { .size = size, .alignment = sizeof(float), .location = CU_MEM_LOCATION_TYPE_DEVICE }));
		ASSERT_SUCCESS(cudaDevice->createSharedMemory(&mem[2], { .size = size, .alignment = sizeof(float), .location = CU_MEM_LOCATION_TYPE_DEVICE }));
		
		sema = m_device->createSemaphore({ .externalHandleTypes = ISemaphore::EHT_OPAQUE_WIN32 });
		ASSERT_SUCCESS(cudaDevice->importGPUSemaphore(&cusema, sema.get()));
		{
			auto devmemory = mem[2]->exportAsMemory(m_device.get());
			assert(devmemory);
			IGPUBuffer::SCreationParams params = {};
			params.size = devmemory->getAllocationSize();
			params.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_SRC_BIT;
			params.externalHandleTypes = CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE;
			importedbuf = m_device->createBuffer(std::move(params));
			assert(importedbuf);
			ILogicalDevice::SBindBufferMemoryInfo bindInfo = { .buffer = importedbuf.get(), .binding = {.memory = devmemory.get() } };
			bool re = m_device->bindBufferMemory(1, &bindInfo);
			assert(re);
		}

		{
			
			IGPUImage::SCreationParams params = {};
			params.type = IGPUImage::ET_2D;
			params.samples = IGPUImage::ESCF_1_BIT;
			params.format = EF_R32_SFLOAT;
			params.extent = { gridDim[0], blockDim[0], 1 };
			params.mipLevels = 1;
			params.arrayLayers = 1;
			params.usage = IGPUImage::EUF_TRANSFER_SRC_BIT;
			params.externalHandleTypes = CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE;
			params.tiling = IGPUImage::TILING::LINEAR;
			importedimg = mem[2]->createAndBindImage(m_device.get(), std::move(params));
			assert(importedimg);
		}
		
		commandPool = m_device->createCommandPool(queue->getFamilyIndex(), {});
		bool re = commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1, &cmd, smart_refctd_ptr(m_logger));
		assert(re);

		auto createStaging = [logicalDevice= m_device]()
		{
			auto buf = logicalDevice->createBuffer({ {.size = size, .usage = asset::IBuffer::EUF_TRANSFER_DST_BIT} });
			auto req = buf->getMemoryReqs();
			req.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDownStreamingMemoryTypeBits();
			auto allocation = logicalDevice->allocate(req, buf.get());
			assert(allocation.isValid() && buf->getBoundMemory().memory->isMappable());
			
			bool re = allocation.memory->map(IDeviceMemoryAllocation::MemoryRange(0, req.size), IDeviceMemoryAllocation::EMCAF_READ);
			assert(re && allocation.memory->getMappedPointer());
			memset(allocation.memory->getMappedPointer(), 0, req.size);
			return buf;
		};

		stagingbuf = createStaging();
		stagingbuf2 = createStaging();
	}

	void launchKernel(CUfunction kernel, CUstream stream)
	{
		auto& cu = cudaHandler->getCUDAFunctionTable();
		// Launch kernel
		{
			CUdeviceptr ptrs[] = {
				mem[0]->getDeviceptr(),
				mem[1]->getDeviceptr(),
				mem[2]->getDeviceptr(),
			};
			void* parameters[] = { &ptrs[0], &ptrs[1], &ptrs[2], &numElements };
			ASSERT_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(ptrs[0], cpubuffers[0]->getPointer(), size, stream));
			ASSERT_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(ptrs[1], cpubuffers[1]->getPointer(), size, stream));
			ASSERT_SUCCESS(cu.pcuLaunchKernel(kernel, gridDim[0], gridDim[1], gridDim[2], blockDim[0], blockDim[1], blockDim[2], 0, stream, parameters, nullptr));
			CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams = { .params = {.fence = {.value = 1 } } };
			auto semaphore = cusema->getInternalObject();
			ASSERT_SUCCESS(cu.pcuSignalExternalSemaphoresAsync(&semaphore, &signalParams, 1, stream)); // Signal the imported semaphore
		}
		
		// After the cuda kernel has signalled our exported vk semaphore, we will download the results through the buffer imported from CUDA
		{
			IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> bufBarrier = {
				.barrier = {
					.dep = {
						.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
						.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS,
					},
					.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
					.otherQueueFamilyIndex = queue->getFamilyIndex(),
				},
				.range = { .buffer = importedbuf, },
			};

			bool re = true;
			re &= cmd->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			re &= cmd->pipelineBarrier(EDF_NONE, { .bufBarriers = std::span{&bufBarrier,&bufBarrier + 1} });

			IGPUCommandBuffer::SBufferCopy region = { .size = size };
			re &= cmd->copyBuffer(importedbuf.get(), stagingbuf.get(), 1, &region);

			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarrier = {
				.barrier = { 
					.dep = { 
						.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS, 
						.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS,
					},
					.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
					.otherQueueFamilyIndex = queue->getFamilyIndex(),
				},
				.image = importedimg.get(),
				.subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.levelCount = 1u,
					.layerCount = 1u,
				},
				.oldLayout = IImage::LAYOUT::UNDEFINED,
				.newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
			};

			re &= cmd->pipelineBarrier(EDF_NONE, { .imgBarriers = {&imgBarrier,&imgBarrier + 1} });

			IImage::SBufferCopy imgRegion = {
				.imageSubresource = {
					.aspectMask = imgBarrier.subresourceRange.aspectMask,
					.layerCount = imgBarrier.subresourceRange.layerCount,
				},
				.imageExtent = importedimg->getCreationParameters().extent,
			};

			re &= cmd->copyImageToBuffer(importedimg.get(), imgBarrier.newLayout, stagingbuf2.get(), 1, &imgRegion);
			re &= cmd->end();
			
			auto waitSemaphores = std::array{IQueue::SSubmitInfo::SSemaphoreInfo{.semaphore = sema.get(), .value = 1, .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS, }};
			auto signalSemaphores = std::array{IQueue::SSubmitInfo::SSemaphoreInfo{.semaphore = sema.get(), .value = 2, .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS, }};
			auto commandBuffers = std::array{IQueue::SSubmitInfo::SCommandBufferInfo{cmd.get()}};
			auto submitInfo = std::array{IQueue::SSubmitInfo {
				.waitSemaphores = waitSemaphores,
				.commandBuffers = commandBuffers,
				.signalSemaphores = signalSemaphores,
			}};
			auto submitRe = queue->submit(submitInfo);
			re &= IQueue::RESULT::SUCCESS == submitRe;
			assert(re);
		}
        
		ASSERT_SUCCESS(cu.pcuLaunchHostFunc(stream, [](void* userData) { decltype(this)(userData)->kernelCallback(); }, this));
	}

	void kernelCallback()
	{
		// Make sure we are also done with the readback
		auto wait = std::array{ISemaphore::SWaitInfo{.semaphore = sema.get(), .value = 2}};
		m_device->waitForSemaphores(wait, true, -1);

		float* A = reinterpret_cast<float*>(cpubuffers[0]->getPointer());
		float* B = reinterpret_cast<float*>(cpubuffers[1]->getPointer());
		float* CBuf = reinterpret_cast<float*>(stagingbuf->getBoundMemory().memory->getMappedPointer());
		float* CImg = reinterpret_cast<float*>(stagingbuf2->getBoundMemory().memory->getMappedPointer());

		assert(!memcmp(CBuf, CImg, size));

		for (auto i = 0; i < numElements; i++)
		{
			bool re = (abs(CBuf[i] - A[i] - B[i]) < 0.01f) && (abs(CImg[i] - A[i] - B[i]) < 0.01f);
			assert(re);
		}
		
		std::cout << "Success\n";
	}

	// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
	bool keepRunning() override { return false; }

	// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
	void workLoopBody() override {}
};

NBL_MAIN_FUNC(CUDA2VKApp)