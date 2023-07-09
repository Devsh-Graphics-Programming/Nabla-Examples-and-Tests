// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/video/CCUDAHandler.h"

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

constexpr uint32_t gridDim[3] = { 4096,1,1 };
constexpr uint32_t blockDim[3] = { 1024,1,1 };
size_t numElements = gridDim[0] * blockDim[0];
size_t size = sizeof(float) * numElements;

void vk2cuda(
	core::smart_refctd_ptr<video::CCUDAHandler> cudaHandler, 
	core::smart_refctd_ptr<video::CCUDADevice> cudaDevice, 
	video::IUtilities * util,
	video::ILogicalDevice* logicalDevice,
	nbl::video::IGPUQueue ** queues,
	CUfunction kernel,
	CUstream stream,
	int=0)
{
	auto& cu = cudaHandler->getCUDAFunctionTable();
	core::smart_refctd_ptr<asset::ICPUBuffer> cpubuffers[3] = { core::make_smart_refctd_ptr<asset::ICPUBuffer>(size),
																core::make_smart_refctd_ptr<asset::ICPUBuffer>(size),
																core::make_smart_refctd_ptr<asset::ICPUBuffer>(size) };
	for (auto j = 0; j < 2; j++)
		for (auto i = 0; i < numElements; i++)
			reinterpret_cast<float*>(cpubuffers[j]->getPointer())[i] = rand();

	{
		auto createBuffer = [&](core::smart_refctd_ptr<asset::ICPUBuffer>const& cpuBuf) {
			auto buf = util->createFilledDeviceLocalBufferOnDedMem(queues[CommonAPI::InitOutput::EQT_COMPUTE],
				{ {.size = size, .usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT},
				{{.externalMemoryHandType = video::IDeviceMemoryBacked::EHT_OPAQUE_WIN32}} },
				cpuBuf->getPointer());
			assert(buf.get());
			return buf;
		};

		auto buf0 = createBuffer(cpubuffers[0]);
		auto buf1 = createBuffer(cpubuffers[1]);
		auto buf2 = createBuffer(cpubuffers[2]);
		
		video::CCUDADevice::SSharedCUDAMemory mem0, mem1, mem2;

		ASSERT_SUCCESS(cudaDevice->importGPUBuffer(buf0.get(), &mem0));
		ASSERT_SUCCESS(cudaDevice->importGPUBuffer(buf1.get(), &mem1));
		ASSERT_SUCCESS(cudaDevice->importGPUBuffer(buf2.get(), &mem2));

		void* parameters[] = { &mem0.ptr, &mem1.ptr, &mem2.ptr, &numElements };
		ASSERT_SUCCESS(cu.pcuLaunchKernel(kernel, gridDim[0], gridDim[1], gridDim[2], blockDim[0], blockDim[1], blockDim[2], 0, stream, parameters, nullptr));
		ASSERT_SUCCESS(cu.pcuMemcpyDtoHAsync_v2(cpubuffers[2]->getPointer(), mem2.ptr, size, stream));
		ASSERT_SUCCESS(cu.pcuCtxSynchronize());

		float* A = reinterpret_cast<float*>(cpubuffers[0]->getPointer());
		float* B = reinterpret_cast<float*>(cpubuffers[1]->getPointer());
		float* C = reinterpret_cast<float*>(cpubuffers[2]->getPointer());

		for (auto i = 0; i < numElements; i++)
			assert(abs(C[i] - A[i] - B[i]) < 0.01f);
	}
}


void cuda2vk(
	core::smart_refctd_ptr<video::CCUDAHandler> cudaHandler,
	core::smart_refctd_ptr<video::CCUDADevice> cudaDevice,
	video::IUtilities* util,
	video::ILogicalDevice* logicalDevice,
	nbl::video::IGPUQueue** queues,
	CUfunction kernel,
	CUstream stream,
	int = 0)
{
	auto& cu = cudaHandler->getCUDAFunctionTable();

	core::smart_refctd_ptr<asset::ICPUBuffer> cpubuffers[3] = { core::make_smart_refctd_ptr<asset::ICPUBuffer>(size),
																core::make_smart_refctd_ptr<asset::ICPUBuffer>(size),
																core::make_smart_refctd_ptr<asset::ICPUBuffer>(size) };
	for (auto j = 0; j < 2; j++)
		for (auto i = 0; i < numElements; i++)
			reinterpret_cast<float*>(cpubuffers[j]->getPointer())[i] = rand();

	{
		video::CCUDADevice::SSharedCUDAMemory mem0, mem1, mem2;
		ASSERT_SUCCESS(cudaDevice->createExportableMemory(size, sizeof(float), &mem0));
		ASSERT_SUCCESS(cudaDevice->createExportableMemory(size, sizeof(float), &mem1));
		ASSERT_SUCCESS(cudaDevice->createExportableMemory(size, sizeof(float), &mem2));

		void* parameters[] = { &mem0.ptr, &mem1.ptr, &mem2.ptr, &numElements };
		ASSERT_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(mem0.ptr, cpubuffers[0]->getPointer(), size, stream));
		ASSERT_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(mem1.ptr, cpubuffers[1]->getPointer(), size, stream));
		ASSERT_SUCCESS(cu.pcuCtxSynchronize());
		ASSERT_SUCCESS(cu.pcuLaunchKernel(kernel, gridDim[0], gridDim[1], gridDim[2], blockDim[0], blockDim[1], blockDim[2], 0, stream, parameters, nullptr));
		ASSERT_SUCCESS(cu.pcuCtxSynchronize());

		auto buf = cudaDevice->exportGPUBuffer(mem2, logicalDevice);
		util->downloadBufferRangeViaStagingBufferAutoSubmit(asset::SBufferRange<video::IGPUBuffer>{.offset = 0, .size = size, .buffer = buf}, cpubuffers[2]->getPointer(), queues[CommonAPI::InitOutput::EQT_COMPUTE]);

		float* A = reinterpret_cast<float*>(cpubuffers[0]->getPointer());
		float* B = reinterpret_cast<float*>(cpubuffers[1]->getPointer());
		float* C = reinterpret_cast<float*>(cpubuffers[2]->getPointer());
		
		for (auto i = 0; i < numElements; i++)
			assert(abs(C[i] - A[i] - B[i]) < 0.01f);
	}
}

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

	vk2cuda(cudaHandler, cudaDevice, utilities.get(), logicalDevice.get(), queues.data(), kernel, stream);

	cuda2vk(cudaHandler, cudaDevice, utilities.get(), logicalDevice.get(), queues.data(), kernel, stream);
	
	ASSERT_SUCCESS(cu.pcuModuleUnload(module));
	ASSERT_SUCCESS(cu.pcuStreamDestroy_v2(stream));

	return 0;
}
