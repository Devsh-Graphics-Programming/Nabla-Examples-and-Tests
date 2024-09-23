// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/video/alloc/SubAllocatedDescriptorSet.h"

#include "../common/BasicMultiQueueApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"

using namespace nbl;
using namespace core;
using namespace system;
using namespace ui;
using namespace asset;
using namespace video;

class SubAllocatedDescriptorSetApp final : public examples::MonoDeviceApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::MonoDeviceApplication;
		using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

		smart_refctd_ptr<nbl::video::ICommandPoolCache> m_poolCache;
		smart_refctd_ptr<nbl::video::SubAllocatedDescriptorSet> m_subAllocDescriptorSet;

		smart_refctd_ptr<ISemaphore> m_timeline;
		uint64_t m_iteration = 0;
		constexpr static inline uint64_t MaxIterations = 200;
		constexpr static inline uint64_t MaxDescriptors = 512;
		constexpr static inline uint64_t MaxAllocPerFrame = 10;
		constexpr static uint32_t AllocatedBinding = 0;
		smart_refctd_ptr<IGPUImageView> m_descriptorImages[MaxDescriptors];
		smart_refctd_ptr<IGPUBuffer> m_descriptorBuffers[MaxDescriptors];

	public:
		SubAllocatedDescriptorSetApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

		bool writeDescriptors(uint32_t count, uint32_t* valueIndices, uint32_t* allocationIndex)
		{
			auto createImageDescriptor = [&](uint32_t width, uint32_t height)
			{
				auto image = m_device->createImage(nbl::video::IGPUImage::SCreationParams {
					{
						.type = nbl::video::IGPUImage::E_TYPE::ET_2D,
						.samples = nbl::video::IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
						.format = nbl::asset::E_FORMAT::EF_R8G8B8A8_UNORM,
						.extent = { width, height, 1 },
						.mipLevels = 1,
						.arrayLayers = 1,
						.usage = nbl::video::IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT 
							| nbl::video::IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT
							| nbl::video::IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT,
					}, {}, nbl::video::IGPUImage::TILING::LINEAR,
				});

				auto reqs = image->getMemoryReqs();
				reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				m_device->allocate(reqs, image.get());

				auto imageView = m_device->createImageView(nbl::video::IGPUImageView::SCreationParams {
					.image = image,
						.viewType = nbl::video::IGPUImageView::E_TYPE::ET_2D,
						.format = nbl::asset::E_FORMAT::EF_R8G8B8A8_UNORM,
						// .subresourceRange = { nbl::video::IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT, 0, 1, 0, 1 },
				});

				return imageView;
			};

			auto createBufferDescriptor = [&](uint32_t size)
			{
				nbl::video::IGPUBuffer::SCreationParams params;
				{
					params.size = size;
					params.usage = nbl::video::IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT
						| nbl::video::IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT
						| nbl::video::IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT;
				}
				auto buffer = m_device->createBuffer(std::move(params));

				auto reqs = buffer->getMemoryReqs();
				reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				m_device->allocate(reqs, buffer.get());

				return buffer;
			};


			std::vector<video::IGPUDescriptorSet::SWriteDescriptorSet> descriptorWrites;
			descriptorWrites.reserve(count);
			std::vector<video::IGPUDescriptorSet::SDescriptorInfo> descriptorInfos;
			{
				for (uint32_t i = 0; i < count; i++)
				{
					auto index = valueIndices[i];
					m_logger->log("writeDescriptors[%d]: allocation[%d]: %d", system::ILogger::ELL_INFO, i, index, allocationIndex[i]);
					if (allocationIndex[i] == core::PoolAddressAllocator<uint32_t>::invalid_address)
						return logFail("value at %d wasn't allocated", i);

					auto allocationIdx = allocationIndex[i];

					video::IGPUDescriptorSet::SDescriptorInfo descriptorInfo;

					// Storage image
					{
						m_descriptorImages[index] = createImageDescriptor(256, 256);
						descriptorInfo.desc = core::smart_refctd_ptr<IGPUImageView>(m_descriptorImages[index]);
						descriptorInfo.info.image.imageLayout = asset::IImage::LAYOUT::GENERAL;
					}
					// Storage buffer
					//{
					//	m_descriptorBuffers[index] = createBufferDescriptor(1024);
					//	descriptorInfo.desc = core::smart_refctd_ptr<IGPUBuffer>(m_descriptorBuffers[index]);
					//	descriptorInfo.info.buffer.offset = 0u;
					//	descriptorInfo.info.buffer.size = 1024u;
					//}

					descriptorInfos.push_back(descriptorInfo);
				}
				for (uint32_t i = 0; i < count; i++)
				{
					auto index = valueIndices[i];
					auto allocationIdx = allocationIndex[i];

					video::IGPUDescriptorSet::SWriteDescriptorSet write;
					write.dstSet = m_subAllocDescriptorSet->getDescriptorSet();
					write.binding = AllocatedBinding;
					write.arrayElement = index;
					write.count = 1u;
					write.info = &descriptorInfos[i];
					descriptorWrites.push_back(write);
				}
			}

			m_device->updateDescriptorSets(descriptorWrites, {});
		}

		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			using nbl::video::IGPUDescriptorSetLayout;

			if (!device_base_t::onAppInitialized(std::move(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;


			constexpr auto MaxConcurrency = 64;

			m_poolCache = ICommandPoolCache::create(core::smart_refctd_ptr(m_device),getComputeQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::NONE,MaxConcurrency);

			m_timeline = m_device->createSemaphore(m_iteration);

			// Descriptor set sub allocator

			video::IGPUDescriptorSetLayout::SBinding bindings[12];
			{
				for (uint32_t i = 0; i < 12; i++)
				{
					bindings[i].binding = i;
					bindings[i].count = MaxDescriptors;
					bindings[i].createFlags = core::bitflag(IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT) 
						| IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT 
						| IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_PARTIALLY_BOUND_BIT;
					if (i % 2 == 0) bindings[i].type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
					else if (i % 2 == 1) bindings[i].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
					bindings[i].stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE;
				}
			}

			std::span<video::IGPUDescriptorSetLayout::SBinding> bindingsSpan(bindings);

			auto descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);

			video::IDescriptorPool::SCreateInfo poolParams = {};
			{
				poolParams.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)] = 512 * 6;
				poolParams.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] = 512 * 6;
				poolParams.maxSets = 1;
				poolParams.flags = core::bitflag(video::IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT);
			}

			auto descriptorPool = m_device->createDescriptorPool(std::move(poolParams));
			auto descriptorSet = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(descriptorSetLayout));


			// TODO: I don't think these are needed for sub allocated descriptor sets (alignment isn't needed, and min size is 1)
			auto subAllocatedDescriptorSet = core::make_smart_refctd_ptr<nbl::video::SubAllocatedDescriptorSet>(core::smart_refctd_ptr(descriptorSet), core::smart_refctd_ptr(m_device)); 
			//std::vector<uint32_t> allocation(MaxDescriptors, core::PoolAddressAllocator<uint32_t>::invalid_address);

			//std::vector<uint32_t> indices;
			//indices.reserve(MaxDescriptors);
			//for (uint32_t i = 0; i < MaxDescriptors; i++)
			//	indices.push_back(i);

			//auto allocNum = subAllocatedDescriptorSet->multi_allocate(AllocatedBinding, allocation.size(), allocation.data());
			//assert(allocNum == 0);
			m_subAllocDescriptorSet = std::move(subAllocatedDescriptorSet);

			//bool response = writeDescriptors(allocation.size(), indices.data(), allocation.data());
			//if (!response) return false;
			
			return true;
		}

		bool keepRunning() override { return m_iteration<MaxIterations; }

		void workLoopBody() override
		{
			IQueue* const queue = getComputeQueue();

			// Similar idea to example 05 (streaming buffers)
			// We will be allocating and freeing stuff, latched on previous frame's timeline semaphore
			auto rng = nbl::hlsl::Xoroshiro64StarStar::construct({ m_iteration ^ 0xdeadbeefu,std::hash<string>()(_NBL_APP_NAME_) });
			const auto elementCount = rng() % MaxAllocPerFrame;
			m_logger->log("elementCount: %d", system::ILogger::ELL_INFO, elementCount);

			std::vector<SubAllocatedDescriptorSet::value_type> values(elementCount, SubAllocatedDescriptorSet::invalid_value);

			{
				std::chrono::steady_clock::time_point waitTill(std::chrono::years(45));
				m_subAllocDescriptorSet->multi_allocate(waitTill, AllocatedBinding, elementCount, values.data());

				std::vector<SubAllocatedDescriptorSet::value_type> indices;
				indices.reserve(elementCount);
				for (uint32_t i = 0; i < elementCount; i++)
					indices.push_back(i);
			
				bool response = writeDescriptors(elementCount, indices.data(), values.data());
				assert(response);
			}

			uint32_t poolIx;
			do
			{
				poolIx = m_poolCache->acquirePool();
			} while (poolIx==ICommandPoolCache::invalid_index);

			smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
			{
				m_poolCache->getPool(poolIx)->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&cmdbuf,1},core::smart_refctd_ptr(m_logger));
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

				// COMMAND RECORDING
				// Here we would hipothetically use the descriptors created above

				auto result = cmdbuf->end();
				assert(result);
			}


			const auto savedIterNum = m_iteration++;
			{
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo =
				{
					.cmdbuf = cmdbuf.get()
				};
				const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo =
				{
					.semaphore = m_timeline.get(),
					.value = m_iteration,
					.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
				};
				const IQueue::SSubmitInfo submitInfo = {
					.waitSemaphores = {},
					.commandBuffers = {&cmdbufInfo,1},
					.signalSemaphores = {&signalInfo,1}
				};

				queue->startCapture();
				auto statusCode = queue->submit({ &submitInfo,1 });
				queue->endCapture();
				assert(statusCode == IQueue::RESULT::SUCCESS);
			}

			const ISemaphore::SWaitInfo futureWait = {m_timeline.get(),m_iteration};
			m_poolCache->releasePool(futureWait,poolIx);
			m_subAllocDescriptorSet->multi_deallocate(AllocatedBinding, elementCount, values.data(), futureWait);
		}

		bool onAppTerminated() override
		{
			return device_base_t::onAppTerminated();
		}
};

NBL_MAIN_FUNC(SubAllocatedDescriptorSetApp)