// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
#include "nbl/system/IApplicationFramework.h"
#include "app_resources/common.h"

#include <iostream>
#include <cstdio>
#include <assert.h>


using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;

//using namespace glm;

static constexpr uint32_t TestCaseIndices[] = {
#include "testCaseData.h"
};


void cpu_tests();

class CooperativeBinarySearch final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
    using device_base_t = application_templates::MonoDeviceApplication;
    using asset_base_t = BuiltinResourcesApplication;
public:
    CooperativeBinarySearch(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        // Remember to call the base class initialization!
        if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;
        if (!asset_base_t::onAppInitialized(std::move(system)))
            return false;

        m_queue = m_device->getQueue(0, 0);
        m_commandPool = m_device->createCommandPool(m_queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        m_commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &m_cmdbuf,1 }, smart_refctd_ptr(m_logger));

        smart_refctd_ptr<IShader> shader;
        {
            IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = m_logger.get();
            lp.workingDirectory = ""; // virtual root
            auto assetBundle = m_assetMgr->getAsset("app_resources/binarySearch.comp.hlsl", lp);
            const auto assets = assetBundle.getContents();
            if (assets.empty())
                return logFail("Could not load shader!");

            auto source = IAsset::castDown<IShader>(assets[0]);
            // The down-cast should not fail!
            assert(source);

            // this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
            shader = m_device->compileShader({ source.get() });
            if (!shader)
                return logFail("Creation of a GPU Shader to from CPU Shader source failed!");
        }

		const uint32_t bindingCount = 2u;
		IGPUDescriptorSetLayout::SBinding bindings[bindingCount] = {};
		bindings[0].type = IDescriptor::E_TYPE::ET_STORAGE_BUFFER; // [[vk::binding(0)]] StructuredBuffer<uint> Histogram;
		bindings[1].type = IDescriptor::E_TYPE::ET_STORAGE_BUFFER; // [[vk::binding(1)]] RWStructuredBuffer<uint> Output;
        
        for(int i = 0; i < bindingCount; ++i)
        {
            bindings[i].stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE;
            bindings[i].count = 1;
            bindings[i].binding = i;
        }
		m_descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);
        {
		    SPushConstantRange pcRange = {};
		    pcRange.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE;
		    pcRange.offset = 0u;
		    pcRange.size = 2 * sizeof(uint32_t);
            auto layout = m_device->createPipelineLayout({ &pcRange,1 }, smart_refctd_ptr(m_descriptorSetLayout));
            IGPUComputePipeline::SCreationParams params = {};
            params.layout = layout.get();
            params.shader.shader = shader.get();
            params.shader.entryPoint = "main";
            if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_pipeline))
                return logFail("Failed to create compute pipeline!\n");
        }

        for (uint32_t i = 0; i < bindingCount; i++)
        {
            m_buffers[i] = m_device->createBuffer(IGPUBuffer::SCreationParams {
                {.size = 500000, .usage = 
                    IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT | 
                    IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT,
                }
            });

            auto reqs = m_buffers[i]->getMemoryReqs();
            reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getHostVisibleMemoryTypeBits();

            m_allocations[i] = m_device->allocate(reqs, m_buffers[i].get());
            
            auto allocationType = i == 0 ? IDeviceMemoryAllocation::EMCAF_WRITE : IDeviceMemoryAllocation::EMCAF_READ;
            auto mapResult = m_allocations[i].memory->map({ 0ull,m_allocations[i].memory->getAllocationSize() }, allocationType);
            assert(mapResult);
        }

		smart_refctd_ptr<IDescriptorPool> descriptorPool = nullptr;
		{
            IDescriptorPool::SCreateInfo createInfo = {};
            createInfo.maxSets = 1;
            createInfo.maxDescriptorCount[static_cast<uint32_t>(IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] = bindingCount;
            descriptorPool = m_device->createDescriptorPool(std::move(createInfo));
        }

        m_descriptorSet = descriptorPool->createDescriptorSet(smart_refctd_ptr(m_descriptorSetLayout));

        IGPUDescriptorSet::SDescriptorInfo descriptorInfos[bindingCount] = {};
        IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[bindingCount] = {};
        
        for(int i = 0; i < bindingCount; ++i)
        {
            writeDescriptorSets[i].info = &descriptorInfos[i];
            writeDescriptorSets[i].dstSet = m_descriptorSet.get();
            writeDescriptorSets[i].binding = i;
            writeDescriptorSets[i].count = bindings[i].count;

			descriptorInfos[i].desc = m_buffers[i];
			descriptorInfos[i].info.buffer.size = ~0ull;
        }

        m_device->updateDescriptorSets(bindingCount, writeDescriptorSets, 0u, nullptr);
       
        // Write test data to the m_buffers[0]
        auto outPtr = m_allocations[0].memory->getMappedPointer();
        assert(outPtr);
        memcpy(
            reinterpret_cast<void*>(outPtr), 
            reinterpret_cast<const void*>(&TestCaseIndices[0]), 
            sizeof(TestCaseIndices));

        // In contrast to fences, we just need one semaphore to rule all dispatches
        return true;
    }

    void onAppTerminated_impl() override
    {
        m_device->waitIdle();
    }

    void workLoopBody() override
    {
        cpu_tests();

        constexpr auto StartedValue = 0;

        smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore(StartedValue);

        m_cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
        m_cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);


        IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t layoutBufferBarrier[1] = { {
            .barrier = {
                .dep = {
                    .srcStageMask = PIPELINE_STAGE_FLAGS::HOST_BIT,
                    .srcAccessMask = ACCESS_FLAGS::HOST_WRITE_BIT,
                    .dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
                    .dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
                }
            },
			// whole buffer because we transferred the contents into it
			.range = {.offset = 0,.size = m_buffers[1]->getCreationParams().size,.buffer = m_buffers[1]}
        } };

        const IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo = { .bufBarriers = layoutBufferBarrier };
        m_cmdbuf->pipelineBarrier(EDF_NONE, depInfo);
        

        const uint32_t pushConstants[2] = { 1920, 1080 };
        const IGPUDescriptorSet* set = m_descriptorSet.get();
        m_cmdbuf->bindComputePipeline(m_pipeline.get());
        m_cmdbuf->bindDescriptorSets(EPBP_COMPUTE, m_pipeline->getLayout(), 0u, 1u, &set);
        m_cmdbuf->dispatch(240, 135, 1u);

		layoutBufferBarrier[0].barrier.dep = layoutBufferBarrier[0].barrier.dep.nextBarrier(PIPELINE_STAGE_FLAGS::COPY_BIT,ACCESS_FLAGS::TRANSFER_READ_BIT);
        m_cmdbuf->pipelineBarrier(EDF_NONE,depInfo);
        
        m_cmdbuf->end();

        {
            constexpr auto FinishedValue = 69;
            IQueue::SSubmitInfo submitInfos[1] = {};
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = m_cmdbuf.get()} };
            submitInfos[0].commandBuffers = cmdbufs;
            const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = progress.get(),.value = FinishedValue,.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
            submitInfos[0].signalSemaphores = signals;
            m_api->startCapture();
            m_queue->submit(submitInfos);
            m_api->endCapture();
            const ISemaphore::SWaitInfo waitInfos[] = { {
                    .semaphore = progress.get(),
                    .value = FinishedValue
                } };
            m_device->blockForSemaphores(waitInfos);
        }

        auto ptr = m_allocations[1].memory->getMappedPointer();
        assert(ptr);
        printf("readback ptr %p\n", ptr);

        m_keepRunning = false;
    }

    bool keepRunning() override
    {
        return m_keepRunning;
    }


private:
    smart_refctd_ptr<IGPUComputePipeline> m_pipeline = nullptr;
    smart_refctd_ptr<IGPUDescriptorSetLayout> m_descriptorSetLayout;
    smart_refctd_ptr<IGPUDescriptorSet> m_descriptorSet;

    smart_refctd_ptr<IGPUBuffer> m_buffers[2];
	nbl::video::IDeviceMemoryAllocator::SAllocation m_allocations[2] = {};
    smart_refctd_ptr<IGPUCommandBuffer> m_cmdbuf = nullptr;
    IQueue* m_queue;
    smart_refctd_ptr<IGPUCommandPool> m_commandPool;
    uint64_t m_iteration = 0;
    constexpr static inline uint64_t MaxIterations = 200;

    bool m_keepRunning = true;
};

NBL_MAIN_FUNC(CooperativeBinarySearch)

void cpu_tests()
{
}
