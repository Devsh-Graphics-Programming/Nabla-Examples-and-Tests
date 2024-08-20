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
    void initialize(smart_refctd_ptr<ILogicalDevice> device, smart_refctd_ptr<system::ISystem> system, smart_refctd_ptr<asset::IAssetManager> assetManager, smart_refctd_ptr<system::ILogger> logger)
    {
        m_device = device;

        // create buffers
        {
            video::IGPUBuffer::SCreationParams params = {};
		    params.size = sizeof(SPrefixSumParams);
		    params.usage = IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;;
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

        nbl::video::IGPUDescriptorSetLayout::SBinding bindingsSet0[] = {
			{
				.binding=1,
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

        auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(system));

		CHLSLCompiler::SOptions options = {};
		options.stage = shaderSrc->getStage();
		if (!(options.stage == IShader::E_SHADER_STAGE::ESS_COMPUTE || options.stage == IShader::E_SHADER_STAGE::ESS_FRAGMENT))
			options.stage = IShader::E_SHADER_STAGE::ESS_VERTEX;
		options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
		smart_refctd_ptr<const asset::ISPIRVOptimizer> optimizer = {};
		options.spirvOptimizer = optimizer.get();
		options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
		options.preprocessorOptions.sourceIdentifier = shaderSrc->getFilepathHint();
		options.preprocessorOptions.logger = logger.get();
		options.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder();

        {
            const std::string entryPoint = "prefixSum";
            std::string dxcOptionStr[] = {"-E " + entryPoint};
            options.dxcOptions = std::span(dxcOptionStr);

            auto shaderSpv = compiler->compileToSPIRV((const char*)shaderSrc->getContent()->getPointer(), options);
            smart_refctd_ptr<video::IGPUShader> shader = m_device->createShader(shaderSpv.get());

            IGPUComputePipeline::SCreationParams params = {};
			params.layout = pplnLayout.get();
            params.shader.entryPoint = entryPoint;
            params.shader.shader = shader.get();

            m_device->createComputePipelines(nullptr, {&params, 1}, &m_prefixSumPipeline);
        }
        {
            const std::string entryPoint = "addGroupSum";
            std::string dxcOptionStr[] = {"-E " + entryPoint};
            options.dxcOptions = std::span(dxcOptionStr);

            auto shaderSpv = compiler->compileToSPIRV((const char*)shaderSrc->getContent()->getPointer(), options);
            smart_refctd_ptr<video::IGPUShader> shader = m_device->createShader(shaderSpv.get());

            IGPUComputePipeline::SCreationParams params = {};
			params.layout = pplnLayout.get();
            params.shader.entryPoint = entryPoint;
            params.shader.shader = shader.get();

            m_device->createComputePipelines(nullptr, {&params, 1}, &m_groupSumPipeline);
        }

        dsLayouts[0] = dsLayout0;
		dsLayouts[1] = dsLayout1;

		const std::array<const IGPUDescriptorSetLayout*, dsLayoutCount> dsLayoutPtrs = {
			dsLayout0.get(),
			dsLayout1.get()
		};
		uint32_t setCounts[2] = { recurseLimit * 3u, recurseLimit * 3u };
        dsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dsLayoutPtrs.begin(), dsLayoutPtrs.end()), setCounts);
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

		if (bufferIdx > recurseLimit)
			;	// check recursion levels?

        uint32_t numThreadsPerGroup = WorkgroupSize;  // fixed for now
        uint32_t numElemsPerGroup = 2 * numThreadsPerGroup;
        uint32_t numWorkgroups = (numElements + numElemsPerGroup - 1) / numElemsPerGroup;

		const std::array<const IGPUDescriptorSetLayout*, dsLayoutCount> dsLayoutPtrs = {
			dsLayouts[0].get(),
			dsLayouts[1].get()
		};

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

		// create and update descriptor sets
		if (!m_prefixSumDs[bufferIdx][0] || !m_prefixSumDs[bufferIdx][1])	// check nullptr, so scuffed
			dsPool->createDescriptorSets(dsLayoutPtrs.size(), dsLayoutPtrs.data(), m_prefixSumDs[bufferIdx].data());

        {
			IGPUDescriptorSet::SDescriptorInfo infos[3];
			infos[0].desc = smart_refctd_ptr(paramsBuffer);
			infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
			infos[1].desc = smart_refctd_ptr(dataBuffer);
			infos[1].info.buffer = {.offset = 0, .size = dataBuffer->getSize()};
			infos[2].desc = smart_refctd_ptr(groupSumBuffer);
			infos[2].info.buffer = {.offset = 0, .size = groupSumBuffer->getSize()};
			IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
				{.dstSet = m_prefixSumDs[bufferIdx][0].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[0]},
				{.dstSet = m_prefixSumDs[bufferIdx][1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[1]},
				{.dstSet = m_prefixSumDs[bufferIdx][1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[2]},
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
		    cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_prefixSumPipeline->getLayout(), 0, m_prefixSumDs[bufferIdx].size(), &m_prefixSumDs[bufferIdx].begin()->get());
		    cmdbuf->dispatch(min(numWorkgroups - i, 65535), 1, 1);
        }

        if (numWorkgroups <= numElemsPerGroup)
        {
			if (!m_groupSumDs[bufferIdx][0] || !m_groupSumDs[bufferIdx][1])
			dsPool->createDescriptorSets(dsLayoutPtrs.size(), dsLayoutPtrs.data(), m_groupSumDs[bufferIdx].data());

            {
			    IGPUDescriptorSet::SDescriptorInfo infos[3];
			    infos[0].desc = smart_refctd_ptr(paramsBuffer);
			    infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
			    infos[1].desc = smart_refctd_ptr(groupSumBuffer);
			    infos[1].info.buffer = {.offset = 0, .size = groupSumBuffer->getSize()};
			    infos[2].desc = smart_refctd_ptr(totalSumBuffer);
			    infos[2].info.buffer = {.offset = 0, .size = totalSumBuffer->getSize()};
			    IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
				    {.dstSet = m_groupSumDs[bufferIdx][0].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[0]},
				    {.dstSet = m_groupSumDs[bufferIdx][1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[1]},
				    {.dstSet = m_groupSumDs[bufferIdx][1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[2]},
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
		    cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_prefixSumPipeline->getLayout(), 0, m_groupSumDs[bufferIdx].size(), &m_groupSumDs[bufferIdx].begin()->get());
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
		if (!m_totalGroupSumsDs[bufferIdx][0] || !m_totalGroupSumsDs[bufferIdx][1])	// check nullptr, so scuffed
			dsPool->createDescriptorSets(dsLayoutPtrs.size(), dsLayoutPtrs.data(), m_totalGroupSumsDs[bufferIdx].data());

        {
			IGPUDescriptorSet::SDescriptorInfo infos[3];
			infos[0].desc = smart_refctd_ptr(paramsBuffer);
			infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
			infos[1].desc = smart_refctd_ptr(dataBuffer);
			infos[1].info.buffer = {.offset = 0, .size = dataBuffer->getSize()};
			infos[2].desc = smart_refctd_ptr(groupSumBuffer);
			infos[2].info.buffer = {.offset = 0, .size = groupSumBuffer->getSize()};
			IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
				{.dstSet = m_totalGroupSumsDs[bufferIdx][0].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[0]},
				{.dstSet = m_totalGroupSumsDs[bufferIdx][1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[1]},
				{.dstSet = m_totalGroupSumsDs[bufferIdx][1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[2]},
			};
			m_device->updateDescriptorSets(std::span(writes, 3), {});
		}

        SPrefixSumParams finalSumParams;    // does this need to remain in scope?
        finalSumParams.numElements = numElements;
        finalSumParams.groupSumOffset = 0;
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

            cmdbuf->bindComputePipeline(m_groupSumPipeline.get());
		    cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_groupSumPipeline->getLayout(), 0, m_totalGroupSumsDs[bufferIdx].size(), &m_totalGroupSumsDs[bufferIdx].begin()->get());
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

	static constexpr uint32_t dsLayoutCount = 2;
	static constexpr uint32_t recurseLimit = 128;
	std::array<smart_refctd_ptr<IGPUDescriptorSetLayout>, dsLayoutCount> dsLayouts;

    smart_refctd_ptr<IDescriptorPool> dsPool;
    std::array<std::array<smart_refctd_ptr<IGPUDescriptorSet>, dsLayoutCount>, recurseLimit> m_prefixSumDs;
	std::array<std::array<smart_refctd_ptr<IGPUDescriptorSet>, dsLayoutCount>, recurseLimit> m_groupSumDs;
	std::array<std::array<smart_refctd_ptr<IGPUDescriptorSet>, dsLayoutCount>, recurseLimit> m_totalGroupSumsDs;

    smart_refctd_ptr<video::IGPUBuffer> m_totalSumBuffer;
    video::IDeviceMemoryAllocator::SAllocation m_totalSumAlloc;
    smart_refctd_ptr<video::IGPUBuffer> paramsBuffer;
    std::vector<smart_refctd_ptr<video::IGPUBuffer>> groupSumBuffers;

    uint32_t numWorkgroups;
};

class GPURadixSort
{
public:
    void initialize(smart_refctd_ptr<ILogicalDevice> device, smart_refctd_ptr<system::ISystem> system, smart_refctd_ptr<asset::IAssetManager> assetManager, smart_refctd_ptr<system::ILogger> logger)
    {
        m_device = device;

        // create buffers
        {
            video::IGPUBuffer::SCreationParams params = {};
		    params.size = sizeof(SSortParams);
		    params.usage = IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;;
            paramsBuffer = m_device->createBuffer(std::move(params));

		    video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = paramsBuffer->getMemoryReqs();
		    reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

		    auto bufMem = m_device->allocate(reqs, paramsBuffer.get());
        }
        // allocate remaining buffers on compute


        // create pipelines
        auto bundle = assetManager->getAsset("app_resources/compute/radix_sort/radixSort.comp.hlsl", {});
		const auto assets = bundle.getContents();
		assert(assets.size() == 1);
		smart_refctd_ptr<ICPUShader> shaderSrc = IAsset::castDown<ICPUShader>(assets[0]);

        nbl::video::IGPUDescriptorSetLayout::SBinding bindingsSet0[] = {
			{
				.binding=1,
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
			},
            {
				.binding=2,
				.type=nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
				.createFlags=IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags=IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,
				.count=1u,
			},
            {
				.binding=3,
				.type=nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
				.createFlags=IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags=IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,
				.count=1u,
			},
            {
				.binding=4,
				.type=nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
				.createFlags=IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags=IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,
				.count=1u,
			}
		};
		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout1 = device->createDescriptorSetLayout(bindingsSet1);

		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout = device->createPipelineLayout({}, smart_refctd_ptr(dsLayout0), smart_refctd_ptr(dsLayout1), nullptr, nullptr);

        auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(system));

		CHLSLCompiler::SOptions options = {};
		options.stage = shaderSrc->getStage();
		if (!(options.stage == IShader::E_SHADER_STAGE::ESS_COMPUTE || options.stage == IShader::E_SHADER_STAGE::ESS_FRAGMENT))
			options.stage = IShader::E_SHADER_STAGE::ESS_VERTEX;
		options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
		smart_refctd_ptr<const asset::ISPIRVOptimizer> optimizer = {};
		options.spirvOptimizer = optimizer.get();
		options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
		options.preprocessorOptions.sourceIdentifier = shaderSrc->getFilepathHint();
		options.preprocessorOptions.logger = logger.get();
		options.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder();

        {
            const std::string entryPoint = "localRadixSort";
            std::string dxcOptionStr[] = {"-E " + entryPoint};
            options.dxcOptions = std::span(dxcOptionStr);

            auto shaderSpv = compiler->compileToSPIRV((const char*)shaderSrc->getContent()->getPointer(), options);
            smart_refctd_ptr<video::IGPUShader> shader = m_device->createShader(shaderSpv.get());

            IGPUComputePipeline::SCreationParams params = {};
			params.layout = pplnLayout.get();
            params.shader.entryPoint = entryPoint;
            params.shader.shader = shader.get();

            m_device->createComputePipelines(nullptr, {&params, 1}, &m_localSortPipeline);
        }
        {
            const std::string entryPoint = "globalMerge";
            std::string dxcOptionStr[] = {"-E " + entryPoint};
            options.dxcOptions = std::span(dxcOptionStr);

            auto shaderSpv = compiler->compileToSPIRV((const char*)shaderSrc->getContent()->getPointer(), options);
            smart_refctd_ptr<video::IGPUShader> shader = m_device->createShader(shaderSpv.get());

            IGPUComputePipeline::SCreationParams params = {};
			params.layout = pplnLayout.get();
            params.shader.entryPoint = entryPoint;
            params.shader.shader = shader.get();

            m_device->createComputePipelines(nullptr, {&params, 1}, &m_globalMergePipeline);
        }

        const std::array<IGPUDescriptorSetLayout*, 2> dscLayoutPtrs = {
				dsLayout0.get(),
				dsLayout1.get()
			};
		const uint32_t setCounts[2u] = { 2u, 2u };
        dsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()), setCounts);
		dsPool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), m_localSortDs.data());
		dsPool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), m_globalMergeDs.data());

        prefixSum.initialize(device, system, assetManager, logger);
    }

    void sort(smart_refctd_ptr<IGPUCommandBuffer> cmdbuf, smart_refctd_ptr<IGPUBuffer> dataBuffer, uint32_t numElements)
    {
        uint32_t numWorkgroups = (numElements + numElemsPerGroup - 1) / numElemsPerGroup;

        updateBuffers(numElements, numWorkgroups, dataBuffer->getSize() / numElements);

		{
			IGPUDescriptorSet::SDescriptorInfo infos[5];
			infos[0].desc = smart_refctd_ptr(paramsBuffer);
			infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
			infos[1].desc = smart_refctd_ptr(dataBuffer);
			infos[1].info.buffer = {.offset = 0, .size = dataBuffer->getSize()};
			infos[2].desc = smart_refctd_ptr(tempDataBuffer);
			infos[2].info.buffer = {.offset = 0, .size = tempDataBuffer->getSize()};
            infos[3].desc = smart_refctd_ptr(firstIdxBuffer);
			infos[3].info.buffer = {.offset = 0, .size = firstIdxBuffer->getSize()};
			infos[4].desc = smart_refctd_ptr(groupSumBuffer);
			infos[4].info.buffer = {.offset = 0, .size = groupSumBuffer->getSize()};
			IGPUDescriptorSet::SWriteDescriptorSet writes[5] = {
				{.dstSet = m_localSortDs[0].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[0]},
				{.dstSet = m_localSortDs[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[1]},
				{.dstSet = m_localSortDs[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[2]},
                {.dstSet = m_localSortDs[1].get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &infos[3]},
				{.dstSet = m_localSortDs[1].get(), .binding = 3, .arrayElement = 0, .count = 1, .info = &infos[4]},
			};
			m_device->updateDescriptorSets(std::span(writes, 5), {});
		}
		{
			IGPUDescriptorSet::SDescriptorInfo infos[5];
			infos[0].desc = smart_refctd_ptr(paramsBuffer);
			infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
			infos[1].desc = smart_refctd_ptr(tempDataBuffer);
			infos[1].info.buffer = {.offset = 0, .size = tempDataBuffer->getSize()};
			infos[2].desc = smart_refctd_ptr(dataBuffer);
			infos[2].info.buffer = {.offset = 0, .size = dataBuffer->getSize()};
            infos[3].desc = smart_refctd_ptr(firstIdxBuffer);
			infos[3].info.buffer = {.offset = 0, .size = firstIdxBuffer->getSize()};
			infos[4].desc = smart_refctd_ptr(groupSumBuffer);
			infos[4].info.buffer = {.offset = 0, .size = groupSumBuffer->getSize()};
			IGPUDescriptorSet::SWriteDescriptorSet writes[5] = {
				{.dstSet = m_globalMergeDs[0].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[0]},
				{.dstSet = m_globalMergeDs[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[1]},
				{.dstSet = m_globalMergeDs[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[2]},
                {.dstSet = m_globalMergeDs[1].get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &infos[3]},
				{.dstSet = m_globalMergeDs[1].get(), .binding = 4, .arrayElement = 0, .count = 1, .info = &infos[4]},
			};
			m_device->updateDescriptorSets(std::span(writes, 5), {});
		}

        SSortParams params;
        params.numElements = numElements;
        params.numGroups = numWorkgroups;
        params.keyType = 0; // unused for now
        params.sortingOrder = 0;

        int firstBitHigh = 32;
        for (int bitShift = 0; bitShift < firstBitHigh; bitShift += 4)  // increment by (length of nWay in binary - 1)
        {
            params.bitShift = bitShift;

            for (int i = 0; i < numWorkgroups; i += 65535)
            {
                params.groupOffset = i;

                SBufferRange<IGPUBuffer> range;
                range.buffer = paramsBuffer;
                range.size = paramsBuffer->getSize();
                cmdbuf->updateBuffer(range, &params);

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

                cmdbuf->bindComputePipeline(m_localSortPipeline.get());
		        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_localSortPipeline->getLayout(), 0, m_localSortDs.size(), &m_localSortDs.begin()->get());
		        cmdbuf->dispatch(min(numWorkgroups - i, 65535), 1, 1);
            }

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
					    .size = groupSumBuffer->getSize(),
					    .buffer = groupSumBuffer,
				    };
			    }
                cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.bufBarriers = {bufferBarriers, bufferBarriersCount}});
            }

			prefixSum.doSum(cmdbuf, groupSumBuffer, numElements);

			for (int i = 0; i < numWorkgroups; i += 65535)
            {
                params.groupOffset = i;

                SBufferRange<IGPUBuffer> range;
                range.buffer = paramsBuffer;
                range.size = paramsBuffer->getSize();
                cmdbuf->updateBuffer(range, &params);

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

                cmdbuf->bindComputePipeline(m_globalMergePipeline.get());
		        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_globalMergePipeline->getLayout(), 0, m_globalMergeDs.size(), &m_globalMergeDs.begin()->get());
		        cmdbuf->dispatch(min(numWorkgroups - i, 65535), 1, 1);
            }
        }
    }

private:
    void updateBuffers(int numElements, int numWorkgroups, int dataTypeSize)
    {
        if (!tempDataBuffer || tempDataBuffer->getSize() < numElements * dataTypeSize)
        {
            //tempDataBuffer->drop();

            video::IGPUBuffer::SCreationParams params = {};
		    params.size = numElements * dataTypeSize;
		    params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
            tempDataBuffer = m_device->createBuffer(std::move(params));

		    video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = tempDataBuffer->getMemoryReqs();
		    reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

		    auto bufMem = m_device->allocate(reqs, tempDataBuffer.get());
        }

        uint32_t bufSize = nWay * numWorkgroups * sizeof(uint32_t);
        if (!firstIdxBuffer || firstIdxBuffer->getSize() < bufSize)
        {
            //firstIdxBuffer->drop();

            video::IGPUBuffer::SCreationParams params = {};
		    params.size = bufSize;
		    params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
            firstIdxBuffer = m_device->createBuffer(std::move(params));

		    video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = firstIdxBuffer->getMemoryReqs();
		    reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

		    auto bufMem = m_device->allocate(reqs, firstIdxBuffer.get());
        }
        if (!groupSumBuffer || groupSumBuffer->getSize() != bufSize)
        {
            //groupSumBuffer->drop();

            video::IGPUBuffer::SCreationParams params = {};
		    params.size = bufSize;
		    params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
            groupSumBuffer = m_device->createBuffer(std::move(params));

		    video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = groupSumBuffer->getMemoryReqs();
		    reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

		    auto bufMem = m_device->allocate(reqs, groupSumBuffer.get());
        }
    }

    struct SSortParams
    {
        uint32_t numElements;
        uint32_t numGroups;

        uint32_t groupOffset;
        uint32_t bitShift;
    
        uint32_t keyType;
        uint32_t sortingOrder;
    };

    const uint32_t nWay = 16;
    const uint32_t numElemsPerGroup = WorkgroupSize;

    smart_refctd_ptr<ILogicalDevice> m_device;

    smart_refctd_ptr<IGPUComputePipeline> m_localSortPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_globalMergePipeline;

    smart_refctd_ptr<IDescriptorPool> dsPool;
    std::array<smart_refctd_ptr<IGPUDescriptorSet>, 2> m_localSortDs;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, 2> m_globalMergeDs;

    GPUPrefixSum prefixSum;

    // buffers
    smart_refctd_ptr<video::IGPUBuffer> paramsBuffer;
    smart_refctd_ptr<video::IGPUBuffer> tempDataBuffer;
    smart_refctd_ptr<video::IGPUBuffer> firstIdxBuffer;
    smart_refctd_ptr<video::IGPUBuffer> groupSumBuffer;
};
