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
	CUmodule module;
	CUfunction kernel;
	CUstream stream;

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
	
	constexpr uint32_t gridDim[3]  = { 4096,1,1 };
	constexpr uint32_t blockDim[3] = { 1024,1,1 };
	int numElements = gridDim[0] * blockDim[0];
	auto _size = sizeof(float) * numElements;

	core::smart_refctd_ptr<asset::ICPUBuffer> cpubuffers[3] = { core::make_smart_refctd_ptr<asset::ICPUBuffer>(_size),
																core::make_smart_refctd_ptr<asset::ICPUBuffer>(_size),
																core::make_smart_refctd_ptr<asset::ICPUBuffer>(_size) };
	for (auto j = 0; j < 2; j++)
		for (auto i = 0; i < numElements; i++)
			reinterpret_cast<float*>(cpubuffers[j]->getPointer())[i] = rand();
	
	{
		auto createBuffer = [&](core::smart_refctd_ptr<asset::ICPUBuffer>const& cpuBuf) {
			struct CUCleaner : video::ICleanup
			{
				CUexternalMemory mem = nullptr;
				CUdeviceptr ptr = {};
				core::smart_refctd_ptr <video::CCUDAHandler> cudaHandler = nullptr;
				core::smart_refctd_ptr <video::CCUDADevice> cudaDevice = nullptr;

				~CUCleaner()
				{
					auto& cu = cudaHandler->getCUDAFunctionTable();
					ASSERT_SUCCESS(cu.pcuMemFree_v2(ptr));
					ASSERT_SUCCESS(cu.pcuDestroyExternalMemory(mem));
				}
			};

			auto cleaner = std::make_unique<CUCleaner>();
			cleaner->cudaHandler = cudaHandler;
			cleaner->cudaDevice = cudaDevice;
			CUexternalMemory* mem = &cleaner->mem;
			CUdeviceptr* ptr = &cleaner->ptr;
			auto buf = utilities->createFilledDeviceLocalBufferOnDedMem(queues[CommonAPI::InitOutput::EQT_COMPUTE],
				{{.size = _size, .usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT},
				{{.preDestroyCleanup = std::move(cleaner), .externalMemoryHandType = video::IDeviceMemoryBacked::EHT_OPAQUE_WIN32}}},
				cpuBuf->getPointer());
			assert(buf.get());
			CUDA_EXTERNAL_MEMORY_HANDLE_DESC handleDesc = {
				.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
				.handle = {.win32 = {.handle = buf->getExternalHandle()}},
				.size = buf->getMemoryReqs().size,
			};
			CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = { .size = buf->getMemoryReqs().size };
			ASSERT_SUCCESS(cu.pcuImportExternalMemory(mem, &handleDesc));
			ASSERT_SUCCESS(cu.pcuExternalMemoryGetMappedBuffer(ptr, *mem, &bufferDesc));
			return std::tuple< core::smart_refctd_ptr<video::IGPUBuffer>, CUexternalMemory, CUdeviceptr>{std::move(buf), *mem, *ptr};
		};

		auto [buf0, mem0, ptr0] = createBuffer(cpubuffers[0]);
		auto [buf1, mem1, ptr1] = createBuffer(cpubuffers[1]);
		auto [buf2, mem2, ptr2] = createBuffer(cpubuffers[2]);
		
		void* parameters[] = { &ptr0, &ptr1, &ptr2, &numElements };
		
		ASSERT_SUCCESS(cu.pcuLaunchKernel(kernel, gridDim[0], gridDim[1], gridDim[2], blockDim[0], blockDim[1], blockDim[2], 0, stream, parameters, nullptr));
		ASSERT_SUCCESS(cu.pcuMemcpyDtoHAsync_v2(cpubuffers[2]->getPointer(), ptr2, _size, stream));
		ASSERT_SUCCESS(cu.pcuCtxSynchronize());

		float* A = reinterpret_cast<float*>(cpubuffers[0]->getPointer());
		float* B = reinterpret_cast<float*>(cpubuffers[1]->getPointer());
		float* C = reinterpret_cast<float*>(cpubuffers[2]->getPointer());

		for (auto i = 0; i < numElements; i++)
			assert(abs(C[i] - A[i] - B[i]) < 0.01f);
	}

	ASSERT_SUCCESS(cu.pcuModuleUnload(module));
	ASSERT_SUCCESS(cu.pcuStreamDestroy_v2(stream));
	

	return 0;
}
