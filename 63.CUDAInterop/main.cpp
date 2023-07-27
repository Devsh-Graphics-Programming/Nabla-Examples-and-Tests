// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/video/CCUDAHandler.h"
#include "nbl/video/CCUDASharedMemory.h"
#include "nbl/video/CCUDASharedSemaphore.h"

#include "../common/CommonAPI.h"

/**
This example just shows a screen which clears to red,
nothing fancy, just to show that Irrlicht links fine
**/
using namespace nbl;


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

constexpr uint32_t gridDim[3] = { 256,1,1 };
constexpr uint32_t blockDim[3] = { 256,1,1 };
size_t numElements = gridDim[0] * blockDim[0];
size_t size = sizeof(float) * numElements;

struct CUDA2VK
{
	core::smart_refctd_ptr<video::CCUDAHandler> cudaHandler;
	core::smart_refctd_ptr<video::CCUDADevice> cudaDevice;
	video::IUtilities* util;
	video::ILogicalDevice* logicalDevice;
	nbl::video::IGPUQueue** queues;

	std::array<core::smart_refctd_ptr<asset::ICPUBuffer>, 2> cpubuffers;
	std::array<core::smart_refctd_ptr<video::CCUDASharedMemory>, 3> mem = {};
	core::smart_refctd_ptr<video::CCUDASharedSemaphore> cusema;
	core::smart_refctd_ptr<video::IGPUBuffer> importedbuf, stagingbuf;
	core::smart_refctd_ptr<video::IGPUImage> importedimg, stagingimg;
	core::smart_refctd_ptr<video::IGPUSemaphore> sema;
	core::smart_refctd_ptr<video::IGPUCommandPool> commandPool;
	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmd;
	core::smart_refctd_ptr<video::IGPUFence> fence;

	CUDA2VK(
		core::smart_refctd_ptr<video::CCUDAHandler> _cudaHandler,
		core::smart_refctd_ptr<video::CCUDADevice> _cudaDevice,
		video::IUtilities* _util,
		video::ILogicalDevice* _logicalDevice,
		video::IGPUQueue** _queues)
		: cudaHandler(std::move(_cudaHandler))
		, cudaDevice(std::move(_cudaDevice))
		, util(_util)
		, logicalDevice(_logicalDevice)
		, queues(_queues)
	{
		createResources();
	}

	void createResources()
	{
		auto& cu = cudaHandler->getCUDAFunctionTable();

		for (auto& buf : cpubuffers)
			buf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(size);

		//for (auto j = 0; j < 2; j++)
		//	for (auto i = 0; i < numElements; i++)
		//		reinterpret_cast<float*>(cpubuffers[j]->getPointer())[i] = rand() / float(RAND_MAX);

		memset(cpubuffers[1]->getPointer(), 0, size);
		uint16_t (*ptr)[2] = reinterpret_cast<uint16_t(*)[2]>(cpubuffers[0]->getPointer());
		for (auto i = 0; i < gridDim[0]; i++)
			for (auto j = 0; j < blockDim[0]; j++)
			{
				ptr[i * blockDim[0] + j][0] = i;
				ptr[i * blockDim[0] + j][1] = j;
			}

		sema = logicalDevice->createSemaphore({ .externalHandleTypes = video::IGPUSemaphore::EHT_OPAQUE_WIN32 });
		ASSERT_SUCCESS(cudaDevice->importGPUSemaphore(&cusema, sema.get()));

		ASSERT_SUCCESS(cudaDevice->createSharedMemory(&mem[0], { size, sizeof(float), CU_MEM_LOCATION_TYPE_DEVICE }));
		ASSERT_SUCCESS(cudaDevice->createSharedMemory(&mem[1], { size, sizeof(float), CU_MEM_LOCATION_TYPE_DEVICE }));
		ASSERT_SUCCESS(cudaDevice->createSharedMemory(&mem[2], { size, sizeof(float), CU_MEM_LOCATION_TYPE_DEVICE }));

		importedbuf = mem[2]->exportAsBuffer(logicalDevice, asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_SRC_BIT);
		importedimg = mem[2]->exportAsImage(logicalDevice,
			asset::IImage::SCreationParams{
				.type = asset::IImage::ET_2D,
				.samples = asset::IImage::ESCF_1_BIT,
				.format  = asset::EF_R32_SFLOAT,
				.extent = { gridDim[0], blockDim[0], 1 },
				.mipLevels = 1,
				.arrayLayers = 1,
				.usage = asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_SRC_BIT,
			});

		assert(importedbuf);
		fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
		commandPool = logicalDevice->createCommandPool(queues[CommonAPI::InitOutput::EQT_COMPUTE]->getFamilyIndex(), {});
		bool re = logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1, &cmd);
		assert(re);

		auto createStaging = [logicalDevice=logicalDevice]()
		{
			auto buf = logicalDevice->createBuffer({ {.size = size, .usage = asset::IBuffer::EUF_TRANSFER_DST_BIT} });
			auto req = buf->getMemoryReqs();
			req.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDownStreamingMemoryTypeBits();
			auto allocation = logicalDevice->allocate(req, buf.get());
			assert(allocation.memory && allocation.offset != video::ILogicalDevice::InvalidMemoryOffset);
			assert(buf->getBoundMemory()->isMappable());
			logicalDevice->mapMemory(video::IDeviceMemoryAllocation::MappedMemoryRange(buf->getBoundMemory(), buf->getBoundMemoryOffset(), req.size), video::IDeviceMemoryAllocation::EMCAF_READ);
			assert(buf->getBoundMemory()->getMappedPointer());
			memset(buf->getBoundMemory()->getMappedPointer(), 0, req.size);
			return buf;
		};

		auto createStagingImg = [logicalDevice = logicalDevice]()
		{
			video::IGPUImage::SCreationParams params = {{
				.type = asset::IImage::ET_2D,
				.samples = asset::IImage::ESCF_1_BIT,
				.format = asset::EF_R32_SFLOAT,
				.extent = { gridDim[0], blockDim[0], 1 },
				.mipLevels = 1,
				.arrayLayers = 1,
				.usage = asset::IImage::EUF_TRANSFER_DST_BIT 
			}};
			params.tiling = video::IGPUImage::ET_LINEAR;

			auto img = logicalDevice->createImage(std::move(params));
			auto req = img->getMemoryReqs();
			req.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDownStreamingMemoryTypeBits();
			auto allocation = logicalDevice->allocate(req, img.get());
			assert(allocation.memory && allocation.offset != video::ILogicalDevice::InvalidMemoryOffset);
			assert(img->getBoundMemory()->isMappable());
			size_t sz = img->getImageDataSizeInBytes();
			logicalDevice->mapMemory(video::IDeviceMemoryAllocation::MappedMemoryRange(img->getBoundMemory(), img->getBoundMemoryOffset(), req.size), video::IDeviceMemoryAllocation::EMCAF_READ);
			assert(img->getBoundMemory()->getMappedPointer());
			memset(img->getBoundMemory()->getMappedPointer(), 0, req.size);
			return img;
		};

		stagingbuf = createStaging();
		stagingimg = createStagingImg();
	}

	void launchKernel(CUfunction kernel, CUstream stream)
	{
		auto queue = queues[CommonAPI::InitOutput::EQT_COMPUTE];

		auto& cu = cudaHandler->getCUDAFunctionTable();
		// Launch kernel
		{
			CUdeviceptr ptrs[] = { mem[0]->getDevicePtr(), mem[1]->getDevicePtr(), mem[2]->getDevicePtr() };
			void* parameters[] = { &ptrs[0], &ptrs[1], &ptrs[2], &numElements };
			ASSERT_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(ptrs[0], cpubuffers[0]->getPointer(), size, stream));
			ASSERT_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(ptrs[1], cpubuffers[1]->getPointer(), size, stream));
			ASSERT_SUCCESS(cu.pcuLaunchKernel(kernel, gridDim[0], gridDim[1], gridDim[2], blockDim[0], blockDim[1], blockDim[2], 0, stream, parameters, nullptr));
			CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams = {};
			auto semaphore = cusema->getInternalObject();
			ASSERT_SUCCESS(cu.pcuSignalExternalSemaphoresAsync(&semaphore, &signalParams, 1, stream)); // Signal the imported semaphore
		}

		// After the cuda kernel has signalled our exported vk semaphore, we will download the results through the buffer imported from CUDA
		{
			video::IGPUSemaphore* waitSemaphores[] = { sema.get() };
			asset::E_PIPELINE_STAGE_FLAGS waitStages[] = { asset::EPSF_ALL_COMMANDS_BIT };
			video::IGPUCommandBuffer* cmdBuffers[] = { cmd.get() };

			video::IGPUCommandBuffer::SBufferMemoryBarrier bufBarrier = {
				.barrier = { .dstAccessMask = asset::E_ACCESS_FLAGS::EAF_ALL_ACCESSES_BIT_DEVSH },
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL_KHR,
				.dstQueueFamilyIndex = queue->getFamilyIndex(),
				.buffer = importedbuf,
				.offset = 0,
				.size = VK_WHOLE_SIZE,
			};

			video::IGPUCommandBuffer::SImageMemoryBarrier imgBarriers[2] = {{
				.barrier = { .dstAccessMask = asset::E_ACCESS_FLAGS::EAF_ALL_ACCESSES_BIT_DEVSH },
				.newLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL_KHR,
				.dstQueueFamilyIndex = queue->getFamilyIndex(),
				.image = importedimg,
				.subresourceRange = {
					.aspectMask = asset::IImage::EAF_COLOR_BIT,
					.levelCount = 1u,
					.layerCount = 1u,
				}}, {
				.barrier = {.dstAccessMask = asset::E_ACCESS_FLAGS::EAF_ALL_ACCESSES_BIT_DEVSH },
				.newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL,
				.image = stagingimg,
				.subresourceRange = {
					.aspectMask = asset::IImage::EAF_COLOR_BIT,
					.levelCount = 1u,
					.layerCount = 1u,
				}}
			};

			bool re = true;
			re &= cmd->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
			/*Acquire?*/
			re &= cmd->pipelineBarrier(asset::EPSF_ALL_COMMANDS_BIT, asset::EPSF_ALL_COMMANDS_BIT, asset::EDF_NONE, 0u, nullptr, 1u, &bufBarrier, 2u, imgBarriers); 	// Ownership transfer?
			asset::SBufferCopy region = { .size = importedbuf->getSize() };
			re &= cmd->copyBuffer(importedbuf.get(), stagingbuf.get(), 1, &region);
			asset::IImage::SImageCopy imgRegion = { 
				.srcSubresource = {
					.aspectMask = imgBarriers[0].subresourceRange.aspectMask,
					.layerCount = imgBarriers[0].subresourceRange.layerCount,
				},
				.dstSubresource = {
					.aspectMask = imgBarriers[1].subresourceRange.aspectMask,
					.layerCount = imgBarriers[1].subresourceRange.layerCount,
				},
				.extent = importedimg->getCreationParameters().extent
			};
			re &= cmd->copyImage(importedimg.get(), imgBarriers[0].newLayout, stagingimg.get(), imgBarriers[1].newLayout, 1, &imgRegion);
			re &= cmd->end();

			video::IGPUQueue::SSubmitInfo submitInfo = {
				.waitSemaphoreCount = 1,
				.pWaitSemaphores = waitSemaphores,
				.pWaitDstStageMask = waitStages,
				.commandBufferCount = 1,
				.commandBuffers = cmdBuffers
			};

			re &= queue->submit(1, &submitInfo, fence.get());
			assert(re);
		}
        
		ASSERT_SUCCESS(cu.pcuLaunchHostFunc(stream, [](void* userData) { decltype(this)(userData)->kernelCallback(); }, this));
	}

	void kernelCallback()
	{
		// Make sure we are also done with the readback
		{
			video::IGPUFence* fences[] = { fence.get() };
			auto status = logicalDevice->waitForFences(1, fences, true, -1);
			assert(video::IGPUFence::ES_SUCCESS == status);
		}

		float* A = reinterpret_cast<float*>(cpubuffers[0]->getPointer());
		float* B = reinterpret_cast<float*>(cpubuffers[1]->getPointer());
		float* CBuf = reinterpret_cast<float*>(stagingbuf->getBoundMemory()->getMappedPointer());
		float* CImg = reinterpret_cast<float*>(stagingimg->getBoundMemory()->getMappedPointer());

		assert(!memcmp(CBuf, CImg, size));

		for (auto i = 0; i < numElements; i++)
		{
			assert(abs(CBuf[i] - A[i] - B[i]) < 0.01f);
			assert(abs(CImg[i] - A[i] - B[i]) < 0.01f);
		}
		
		std::cout << "Success\n";

		delete this;
	}
};

int main(int argc, char** argv)
{
	auto initOutput = CommonAPI::InitWithDefaultExt(CommonAPI::InitParams{
		.appName = { "63.CUDAInterop" },
		.apiType = video::EAT_VULKAN, 
		.swapchainImageUsage = nbl::asset::IImage::EUF_NONE,
	});

	auto& system = initOutput.system;
	auto& apiConnection = initOutput.apiConnection;
	auto& physicalDevice = initOutput.physicalDevice;
	auto& logicalDevice = initOutput.logicalDevice;
	auto& utilities = initOutput.utilities;
	auto& queues = initOutput.queues;
	auto& logger = initOutput.logger;
	

	assert(physicalDevice->getLimits().externalMemory);
	auto cudaHandler = video::CCUDAHandler::create(system.get(), core::smart_refctd_ptr<system::ILogger>(logger));
	assert(cudaHandler);
	auto cudaDevice = cudaHandler->createDevice(core::smart_refctd_ptr_dynamic_cast<video::CVulkanConnection>(apiConnection), physicalDevice);
	auto& cu = cudaHandler->getCUDAFunctionTable();	

	core::smart_refctd_ptr<asset::ICPUBuffer> ptx;
	CUmodule   module;
	CUfunction kernel;
	CUstream   stream;

	{
		system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> fut;
		system->createFile(fut, "../vectorAdd_kernel.cu", system::IFileBase::ECF_READ);
		auto [ptx_, res] = cudaHandler->compileDirectlyToPTX(fut.copy().get(), cudaDevice->geDefaultCompileOptions());
		ASSERT_SUCCESS_NV(res);
		ptx = std::move(ptx_);
	}

	ASSERT_SUCCESS(cu.pcuModuleLoadDataEx(&module, ptx->getPointer(), 0u, nullptr, nullptr));
	ASSERT_SUCCESS(cu.pcuModuleGetFunction(&kernel, module, "vectorAdd"));
	ASSERT_SUCCESS(cu.pcuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

	(new CUDA2VK(cudaHandler, cudaDevice, utilities.get(), logicalDevice.get(), queues.data()))->launchKernel(kernel, stream);

	ASSERT_SUCCESS(cu.pcuStreamSynchronize(stream));

	ASSERT_SUCCESS(cu.pcuModuleUnload(module));
	ASSERT_SUCCESS(cu.pcuStreamDestroy_v2(stream));
	return 0;
}
