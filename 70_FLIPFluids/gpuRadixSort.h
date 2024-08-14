#pragma once

#include <nabla.h>

#include "app_resources/common.hlsl"

using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;

class GPUPrefixSum
{
public:
    void initialize(smart_refctd_ptr<ILogicalDevice> device, smart_refctd_ptr<asset::IAssetManager> assetManager)
    {
        m_device = device;

        // create buffers
        {
            video::IGPUBuffer::SCreationParams params = {};
		    params.size = sizeof(SPrefixSumParams);
		    params.usage = IGPUBuffer::EUF_UNIFORM_BUFFER_BIT;
            paramsBuffer = m_device->createBuffer(std::move(params));

		    video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = paramsBuffer->getMemoryReqs();
		    reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

		    auto bufMem = m_device->allocate(reqs, paramsBuffer.get());
        }
        {
            video::IGPUBuffer::SCreationParams params = {};
		    params.size = sizeof(uint32_t); // 1 element
		    params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
            m_totalSumBuffer = m_device->createBuffer(std::move(params));

		    video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = m_totalSumBuffer->getMemoryReqs();
		    reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

		    m_totalSumAlloc = m_device->allocate(reqs, m_totalSumBuffer.get());

            m_totalSumAlloc.memory->map({0ull, m_totalSumAlloc.memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ);
        }

        // create pipelines
		auto bundle = assetManager->getAsset("app_resources/compute/radix_sort/prefixSum.comp.hlsl", {});
		const auto assets = bundle.getContents();
		assert(assets.size() == 1);
		smart_refctd_ptr<ICPUShader> shaderSrc = IAsset::castDown<ICPUShader>(assets[0]);
        smart_refctd_ptr<video::IGPUShader> shader = m_device->createShader(shaderSrc.get());

        nbl::video::IGPUDescriptorSetLayout::SBinding bindingsSet0[] = {
			{
				.binding=0,
				.type=nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
				.createFlags=IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags=IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,
				.count=1u,
			},
		};
		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout0 = device->createDescriptorSetLayout(bindingsSet0);

        nbl::video::IGPUDescriptorSetLayout::SBinding bindingsSet1[] = {
			{
				.binding=0,
				.type=nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
				.createFlags=IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags=IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,
				.count=1u,
			},
            {
				.binding=1,
				.type=nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
				.createFlags=IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags=IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,
				.count=1u,
			}
		};
		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout1 = device->createDescriptorSetLayout(bindingsSet1);

		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout = device->createPipelineLayout({}, smart_refctd_ptr(dsLayout0), smart_refctd_ptr(dsLayout1));
        {
            IGPUComputePipeline::SCreationParams params = {};
            params.shader.entryPoint = "prefixSum";
            params.shader.shader = shader.get();

            m_device->createComputePipelines(nullptr, {&params, 1}, &m_prefixSumPipeline);
        }
        {
            IGPUComputePipeline::SCreationParams params = {};
            params.shader.entryPoint = "addGroupSum";
            params.shader.shader = shader.get();

            m_device->createComputePipelines(nullptr, {&params, 1}, &m_groupSumPipeline);
        }

        const std::array<IGPUDescriptorSetLayout*, 2> dscLayoutPtrs = {
				dsLayout0.get(),
				dsLayout1.get()
			};
        dsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()));
		dsPool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), sets.data());
    }

    void doSum(smart_refctd_ptr<IGPUCommandBuffer> cmdbuf, smart_refctd_ptr<IGPUBuffer> buffer, uint32_t numElements)
    {
        doSumRecurse(cmdbuf, buffer, nullptr, numElements, 0, false, 0);
    }

private:
    void doSumRecurse(smart_refctd_ptr<IGPUCommandBuffer> cmdbuf, smart_refctd_ptr<IGPUBuffer> dataBuffer, smart_refctd_ptr<IGPUBuffer> totalSumBuffer,
        uint32_t numElements, uint32_t bufferOffset, bool returnTotal, int bufferIdx)
    {
        if (!totalSumBuffer)
            totalSumBuffer = m_totalSumBuffer;

        uint32_t numThreadsPerGroup = WorkgroupSize;  // fixed for now
        uint32_t numElemsPerGroup = 2 * numThreadsPerGroup;
        uint32_t numWorkgroups = (numElements + numElemsPerGroup - 1) / numElemsPerGroup;

        smart_refctd_ptr<video::IGPUBuffer> groupSumBuffer;
        if (groupSumBuffers.size() == bufferIdx)
        {
            video::IGPUBuffer::SCreationParams params = {};
		    params.size = numWorkgroups * sizeof(uint32_t);
		    params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
            groupSumBuffer = m_device->createBuffer(std::move(params));

		    video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = groupSumBuffer->getMemoryReqs();
		    reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

		    auto bufMem = m_device->allocate(reqs, groupSumBuffer.get());

            groupSumBuffers.push_back(groupSumBuffer);
        }
        else if (groupSumBuffers.size() > bufferIdx)
        {
            groupSumBuffer = groupSumBuffers[bufferIdx];
            uint32_t bufferSize = numWorkgroups * sizeof(uint32_t);
            if (groupSumBuffer->getSize() != bufferSize)
            {
                video::IGPUBuffer::SCreationParams params = {};
		        params.size = bufferSize;
		        params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
                groupSumBuffer = m_device->createBuffer(std::move(params));

		        video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = groupSumBuffer->getMemoryReqs();
		        reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

		        auto bufMem = m_device->allocate(reqs, groupSumBuffer.get());

                groupSumBuffers[bufferIdx] = groupSumBuffer;    // should replace and destroy the obsolete buffer
            }
        }

        {
			IGPUDescriptorSet::SDescriptorInfo infos[3];
			infos[0].desc = smart_refctd_ptr(paramsBuffer);
			infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
			infos[1].desc = smart_refctd_ptr(dataBuffer);
			infos[1].info.buffer = {.offset = 0, .size = dataBuffer->getSize()};
			infos[2].desc = smart_refctd_ptr(groupSumBuffer);
			infos[2].info.buffer = {.offset = 0, .size = groupSumBuffer->getSize()};
			IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
				{.dstSet = sets[0].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[0]},
				{.dstSet = sets[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[1]},
				{.dstSet = sets[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[2]},
			};
			m_device->updateDescriptorSets(std::span(writes, 3), {});
		}

        SPrefixSumParams localSumsParams;    // does this need to remain in scope?
        localSumsParams.numElements = numElements;
        localSumsParams.groupSumOffset = 0;
        SBufferRange<IGPUBuffer> range;
        for (uint32_t i = 0; i < numWorkgroups; i += 65535)
        {
            localSumsParams.groupOffset = i;

            range.buffer = paramsBuffer;
            range.size = paramsBuffer->getSize();
            cmdbuf->updateBuffer(range, &localSumsParams);

            {
                uint32_t bufferBarriersCount = 0u;
			    IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t bufferBarriers[6u];
			    {
				    auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				    bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				    bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				    bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				    bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				    bufferBarrier.range =
				    {
					    .offset = 0u,
					    .size = paramsBuffer->getSize(),
					    .buffer = paramsBuffer,
				    };
			    }
                cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.bufBarriers = {bufferBarriers, bufferBarriersCount}});
            }

            cmdbuf->bindComputePipeline(m_prefixSumPipeline.get());
		    cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_prefixSumPipeline->getLayout(), 0, sets.size(), &sets.begin()->get());
		    cmdbuf->dispatch(min(numWorkgroups - i, 65535), 1, 1);
        }

        if (numWorkgroups <= numElemsPerGroup)
        {
            {
			    IGPUDescriptorSet::SDescriptorInfo infos[3];
			    infos[0].desc = smart_refctd_ptr(paramsBuffer);
			    infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
			    infos[1].desc = smart_refctd_ptr(groupSumBuffer);
			    infos[1].info.buffer = {.offset = 0, .size = groupSumBuffer->getSize()};
			    infos[2].desc = smart_refctd_ptr(totalSumBuffer);
			    infos[2].info.buffer = {.offset = 0, .size = totalSumBuffer->getSize()};
			    IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
				    {.dstSet = sets[0].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[0]},
				    {.dstSet = sets[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[1]},
				    {.dstSet = sets[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[2]},
			    };
			    m_device->updateDescriptorSets(std::span(writes, 3), {});
		    }

            SPrefixSumParams totalSumsParams;    // does this need to remain in scope?
            totalSumsParams.numElements = numElements;
            totalSumsParams.groupSumOffset = bufferOffset;
            totalSumsParams.groupOffset = 0;

            range.buffer = paramsBuffer;
            range.size = paramsBuffer->getSize();
            cmdbuf->updateBuffer(range, &totalSumsParams);

            {
                uint32_t bufferBarriersCount = 0u;
			    IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t bufferBarriers[6u];
			    {
				    auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				    bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				    bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				    bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				    bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				    bufferBarrier.range =
				    {
					    .offset = 0u,
					    .size = paramsBuffer->getSize(),
					    .buffer = paramsBuffer,
				    };
			    }
                cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.bufBarriers = {bufferBarriers, bufferBarriersCount}});
            }

            cmdbuf->bindComputePipeline(m_prefixSumPipeline.get());
		    cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_prefixSumPipeline->getLayout(), 0, sets.size(), &sets.begin()->get());
		    cmdbuf->dispatch(1, 1, 1);

            /* not used
            if (returnTotal)
            {
                const ILogicalDevice::MappedMemoryRange memRange(m_totalSumAlloc.memory.get(), 0ull, m_totalSumAlloc.memory->getAllocationSize());
                if (!m_totalSumAlloc.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
                    m_device->invalidateMappedMemoryRanges(1, &memRange);

                auto totalData = reinterpret_cast<const uint32_t*>(m_totalSumAlloc.memory->getMappedPointer());
                uint32_t totalSum = totalData[0];
            }
            */
        }
        else
        {
            doSumRecurse(cmdbuf, groupSumBuffer, totalSumBuffer, bufferOffset, numWorkgroups, returnTotal, bufferIdx + 1);
        }

        // final sum from each group to output
        {
			IGPUDescriptorSet::SDescriptorInfo infos[3];
			infos[0].desc = smart_refctd_ptr(paramsBuffer);
			infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
			infos[1].desc = smart_refctd_ptr(dataBuffer);
			infos[1].info.buffer = {.offset = 0, .size = dataBuffer->getSize()};
			infos[2].desc = smart_refctd_ptr(groupSumBuffer);
			infos[2].info.buffer = {.offset = 0, .size = groupSumBuffer->getSize()};
			IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
				{.dstSet = sets[0].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[0]},
				{.dstSet = sets[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[1]},
				{.dstSet = sets[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[2]},
			};
			m_device->updateDescriptorSets(std::span(writes, 3), {});
		}

        SPrefixSumParams finalSumParams;    // does this need to remain in scope?
        finalSumParams.numElements = numElements;
        finalSumParams.groupSumOffset = 0;
        SBufferRange<IGPUBuffer> range;
        for (uint32_t i = 0; i < numWorkgroups; i += 65535)
        {
            finalSumParams.groupOffset = i;

            range.buffer = paramsBuffer;
            range.size = paramsBuffer->getSize();
            cmdbuf->updateBuffer(range, &finalSumParams);

            {
                uint32_t bufferBarriersCount = 0u;
			    IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t bufferBarriers[6u];
			    {
				    auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				    bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				    bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				    bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				    bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				    bufferBarrier.range =
				    {
					    .offset = 0u,
					    .size = paramsBuffer->getSize(),
					    .buffer = paramsBuffer,
				    };
			    }
                cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.bufBarriers = {bufferBarriers, bufferBarriersCount}});
            }

            cmdbuf->bindComputePipeline(m_prefixSumPipeline.get());
		    cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_prefixSumPipeline->getLayout(), 0, sets.size(), &sets.begin()->get());
		    cmdbuf->dispatch(min(numWorkgroups - i, 65535), 1, 1);
        }
    }

    struct SPrefixSumParams
    {
        uint32_t numElements;
        uint32_t groupOffset;
        uint32_t groupSumOffset;
        uint32_t pad;
    };

    smart_refctd_ptr<ILogicalDevice> m_device;

    smart_refctd_ptr<IGPUComputePipeline> m_prefixSumPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_groupSumPipeline;

    smart_refctd_ptr<IDescriptorPool> dsPool;
    std::array<smart_refctd_ptr<IGPUDescriptorSet>, 2> sets;

    smart_refctd_ptr<video::IGPUBuffer> m_totalSumBuffer;
    video::IDeviceMemoryAllocator::SAllocation m_totalSumAlloc;
    smart_refctd_ptr<video::IGPUBuffer> paramsBuffer;
    std::vector<smart_refctd_ptr<video::IGPUBuffer>> groupSumBuffers;

    uint32_t numWorkgroups;
};

class GPURadixSort
{
public:
    void initialize(smart_refctd_ptr<ILogicalDevice> device);

    void sort();

private:
    smart_refctd_ptr<ILogicalDevice> m_device;

    smart_refctd_ptr<IGPUComputePipeline> m_localSortPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_globalMergePipeline;

    GPUPrefixSum prefixSum;

    // buffers
};
