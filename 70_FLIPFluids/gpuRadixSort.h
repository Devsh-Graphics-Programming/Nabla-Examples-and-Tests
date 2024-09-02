#pragma once

#include <nabla.h>

#include "app_resources/compute/radix_sort/sort_common.hlsl"

using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;

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


        nbl::video::IGPUDescriptorSetLayout::SBinding bindingsSet1[] = {
			{
				.binding=0,
				.type=nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
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
			}
		};
		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout1 = device->createDescriptorSetLayout(bindingsSet1);

		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> histogramPplnLayout = device->createPipelineLayout({}, nullptr, smart_refctd_ptr(dsLayout1), nullptr, nullptr);

		nbl::video::IGPUDescriptorSetLayout::SBinding bindingsSet2[] = {
			{
				.binding=0,
				.type=nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
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
			}
		};
		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout2 = device->createDescriptorSetLayout(bindingsSet2);

		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> sortPplnLayout = device->createPipelineLayout({}, nullptr, smart_refctd_ptr(dsLayout2), nullptr, nullptr);

		// create pipelines
		auto compileShader = [&](const std::string& path) -> smart_refctd_ptr<IGPUShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = logger.get();
				lp.workingDirectory = "";
				auto bundle = assetManager->getAsset(path, lp);
				const auto assets = bundle.getContents();
				assert(assets.size() == 1);
				smart_refctd_ptr<ICPUShader> shaderSrc = IAsset::castDown<ICPUShader>(assets[0]);
				shaderSrc->setShaderStage(IShader::E_SHADER_STAGE::ESS_COMPUTE);
				if (!shaderSrc)
					return nullptr;

				return m_device->createShader(shaderSrc.get());
			};

        {
            smart_refctd_ptr<video::IGPUShader> shader = compileShader("app_resources/compute/radix_sort/radixSort.comp.hlsl");

            IGPUComputePipeline::SCreationParams params = {};
			params.layout = sortPplnLayout.get();
            params.shader.shader = shader.get();

            m_device->createComputePipelines(nullptr, {&params, 1}, &m_radixSortPipeline);
        }
        {
            smart_refctd_ptr<video::IGPUShader> shader = compileShader("app_resources/compute/radix_sort/buildHistogram.comp.hlsl");

            IGPUComputePipeline::SCreationParams params = {};
			params.layout = histogramPplnLayout.get();
            params.shader.shader = shader.get();

            m_device->createComputePipelines(nullptr, {&params, 1}, &m_buildHistogramPipeline);
        }

        std::array<IGPUDescriptorSetLayout*, 2> dscLayoutPtrs = {
				nullptr,
				dsLayout1.get()
			};
		const uint32_t setCounts[2u] = { 0u, 2u };
        histogramDsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()), setCounts);
		m_histogramDs[0] = histogramDsPool->createDescriptorSet(dsLayout1);
		m_histogramDs[1] = histogramDsPool->createDescriptorSet(dsLayout1);

		dscLayoutPtrs = {
				nullptr,
				dsLayout2.get()
			};
		sortDsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()), setCounts);
		m_radixSortDs[0] = sortDsPool->createDescriptorSet(dsLayout2);
		m_radixSortDs[1] = sortDsPool->createDescriptorSet(dsLayout2);
    }

    void sort(IGPUCommandBuffer* cmdbuf, smart_refctd_ptr<IGPUBuffer> dataBuffer, uint32_t numElements)
    {
        uint32_t numWorkgroups = (numElements + numThreadsPerGroup - 1) / numThreadsPerGroup;

        bool shouldWriteDs = updateBuffers(numElements, numWorkgroups, dataBuffer->getSize() / numElements);

		if (shouldWriteDs)
		{
			{	// iteration 0 and 2
				IGPUDescriptorSet::SDescriptorInfo infos[4];
				infos[0].desc = smart_refctd_ptr(paramsBuffer);
				infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
				infos[1].desc = smart_refctd_ptr(dataBuffer);
				infos[1].info.buffer = {.offset = 0, .size = dataBuffer->getSize()};
				infos[2].desc = smart_refctd_ptr(tempDataBuffer);
				infos[2].info.buffer = {.offset = 0, .size = tempDataBuffer->getSize()};
				infos[3].desc = smart_refctd_ptr(histogramBuffer);
				infos[3].info.buffer = {.offset = 0, .size = histogramBuffer->getSize()};
				IGPUDescriptorSet::SWriteDescriptorSet writes[4] = {
					{.dstSet = m_radixSortDs[0].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
					{.dstSet = m_radixSortDs[0].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]},
					{.dstSet = m_radixSortDs[0].get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &infos[2]},
					{.dstSet = m_radixSortDs[0].get(), .binding = 3, .arrayElement = 0, .count = 1, .info = &infos[3]},
				};
				m_device->updateDescriptorSets(std::span(writes, 4), {});
			}
			{	// iteration 1 and 3
				IGPUDescriptorSet::SDescriptorInfo infos[4];
				infos[0].desc = smart_refctd_ptr(paramsBuffer);
				infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
				infos[1].desc = smart_refctd_ptr(tempDataBuffer);
				infos[1].info.buffer = {.offset = 0, .size = tempDataBuffer->getSize()};
				infos[2].desc = smart_refctd_ptr(dataBuffer);
				infos[2].info.buffer = {.offset = 0, .size = dataBuffer->getSize()};			
				infos[3].desc = smart_refctd_ptr(histogramBuffer);
				infos[3].info.buffer = {.offset = 0, .size = histogramBuffer->getSize()};
				IGPUDescriptorSet::SWriteDescriptorSet writes[4] = {
					{.dstSet = m_radixSortDs[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
					{.dstSet = m_radixSortDs[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]},
					{.dstSet = m_radixSortDs[1].get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &infos[2]},
					{.dstSet = m_radixSortDs[1].get(), .binding = 3, .arrayElement = 0, .count = 1, .info = &infos[3]},
				};
				m_device->updateDescriptorSets(std::span(writes, 4), {});
			}
			{	// iteration 0 and 2
				IGPUDescriptorSet::SDescriptorInfo infos[3];
				infos[0].desc = smart_refctd_ptr(paramsBuffer);
				infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
				infos[1].desc = smart_refctd_ptr(dataBuffer);
				infos[1].info.buffer = {.offset = 0, .size = dataBuffer->getSize()};
				infos[2].desc = smart_refctd_ptr(histogramBuffer);
				infos[2].info.buffer = {.offset = 0, .size = histogramBuffer->getSize()};
				IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
					{.dstSet = m_histogramDs[0].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
					{.dstSet = m_histogramDs[0].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]},
					{.dstSet = m_histogramDs[0].get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &infos[2]},
				};
				m_device->updateDescriptorSets(std::span(writes, 3), {});
			}
			{	// iteration 1 and 3
				IGPUDescriptorSet::SDescriptorInfo infos[3];
				infos[0].desc = smart_refctd_ptr(paramsBuffer);
				infos[0].info.buffer = {.offset = 0, .size = paramsBuffer->getSize()};
				infos[1].desc = smart_refctd_ptr(tempDataBuffer);
				infos[1].info.buffer = {.offset = 0, .size = tempDataBuffer->getSize()};
				infos[2].desc = smart_refctd_ptr(histogramBuffer);
				infos[2].info.buffer = {.offset = 0, .size = histogramBuffer->getSize()};
				IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
					{.dstSet = m_histogramDs[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
					{.dstSet = m_histogramDs[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]},
					{.dstSet = m_histogramDs[1].get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &infos[2]},
				};
				m_device->updateDescriptorSets(std::span(writes, 3), {});
			}
		}

		SBufferRange<IGPUBuffer> range;
		range.buffer = histogramBuffer;
		range.size = histogramBuffer->getSize();
		cmdbuf->fillBuffer(range, 0ull);

		SMemoryBarrier memBarrier;
		memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
		memBarrier.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
		memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
		memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
		cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});

        SSortParams params;
        params.numElements = numElements;
        params.numWorkgroups = numWorkgroups;
        params.numThreadsPerGroup = numThreadsPerGroup;

		const uint32_t numIterations = 4;

        for (uint32_t i = 0; i < numIterations; i++)
        {
            params.bitShift = 8 * i;

			SBufferRange<IGPUBuffer> range;
            range.buffer = paramsBuffer;
            range.size = paramsBuffer->getSize();
            cmdbuf->updateBuffer(range, &params);

            {
				SMemoryBarrier memBarrier;
				memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
				memBarrier.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
				memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
            }

            cmdbuf->bindComputePipeline(m_buildHistogramPipeline.get());
			const IGPUDescriptorSet* histSet = m_histogramDs[i % 2].get();
		    cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_buildHistogramPipeline->getLayout(), 1, 1, &histSet);
		    cmdbuf->dispatch(numWorkgroups, 1, 1);

            {
				SMemoryBarrier memBarrier;
				memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
				memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
            }

			cmdbuf->bindComputePipeline(m_radixSortPipeline.get());
			const IGPUDescriptorSet* sortSet = m_radixSortDs[i % 2].get();
		    cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_radixSortPipeline->getLayout(), 1, 1, &sortSet);
		    cmdbuf->dispatch(numWorkgroups, 1, 1);

			{
				SMemoryBarrier memBarrier;
				memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
				memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
            }
        }
    }

private:
    bool updateBuffers(int numElements, int numWorkgroups, int dataTypeSize)
    {
		bool updated = false;
        if (!tempDataBuffer || tempDataBuffer->getSize() != numElements * dataTypeSize)
        {
            video::IGPUBuffer::SCreationParams params = {};
		    params.size = numElements * dataTypeSize;
		    params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
            tempDataBuffer = m_device->createBuffer(std::move(params));

		    video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = tempDataBuffer->getMemoryReqs();
		    reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

		    auto bufMem = m_device->allocate(reqs, tempDataBuffer.get());
			updated = true;
        }
		uint32_t bufSize = sizeof(uint32_t) * numWorkgroups * NUM_SORT_BINS;
		if (!histogramBuffer || histogramBuffer->getSize() != bufSize)
		{
			video::IGPUBuffer::SCreationParams params = {};
		    params.size = bufSize;
		    params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
            histogramBuffer = m_device->createBuffer(std::move(params));

		    video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = histogramBuffer->getMemoryReqs();
		    reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

		    auto bufMem = m_device->allocate(reqs, histogramBuffer.get());
			updated = true;
		}

		return updated;
    }

    struct SSortParams
    {
        uint32_t numElements;
        uint32_t bitShift;
		uint32_t numWorkgroups;
		uint32_t numThreadsPerGroup;
    };

	const uint32_t numThreadsPerGroup = 32;

    smart_refctd_ptr<ILogicalDevice> m_device;

    smart_refctd_ptr<IGPUComputePipeline> m_radixSortPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_buildHistogramPipeline;

    smart_refctd_ptr<IDescriptorPool> sortDsPool;
    std::array<smart_refctd_ptr<IGPUDescriptorSet>, 2> m_radixSortDs;
	smart_refctd_ptr<IDescriptorPool> histogramDsPool;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, 2> m_histogramDs;

    // buffers
    smart_refctd_ptr<video::IGPUBuffer> paramsBuffer;
    smart_refctd_ptr<video::IGPUBuffer> tempDataBuffer;
	smart_refctd_ptr<video::IGPUBuffer> histogramBuffer;
};
