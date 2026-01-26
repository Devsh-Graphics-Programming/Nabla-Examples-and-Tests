// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_I_PRESENTER_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_I_PRESENTER_H_INCLUDED_


#include "renderer/CScene.h"
#include "renderer/CSession.h"

#include "renderer/shaders/pathtrace/push_constants.hlsl"


namespace nbl::this_example
{

class IPresenter : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
		constexpr static inline uint8_t CircularBufferSize = 4;

		struct SCachedCreationParams
		{
			core::smart_refctd_ptr<asset::IAssetManager> assMan = nullptr;
			system::logger_opt_smart_ptr logger = nullptr;
		};
		//
		inline const SCachedCreationParams& getCreationParams() const {return m_creation;}

		//
		inline bool init(CRenderer* renderer)
		{
			if (!m_queue)
				return isInitialized();

			auto& logger = m_creation.logger;
			auto* device = renderer->getDevice();
			m_queue = renderer->getCreationParams().graphicsQueue;

			bool success = false;
			auto deinit = core::makeRAIIExiter([&]()->void{
				if (success)
					return;
				m_semaphore = nullptr;
				std::fill(m_cmdbufs.begin(),m_cmdbufs.end(),nullptr);
			});

			using namespace nbl::system;
			if (!(m_semaphore=device->createSemaphore(m_presentCount)))
			{
				logger.log("`IPresenter::init` failed to create a semaphore!",ILogger::ELL_ERROR);
				return false;
			}

			for (auto& cmdbuf : m_cmdbufs)
			{
				using namespace nbl::video;
				auto pool=device->createCommandPool(m_queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&cmdbuf,1},core::smart_refctd_ptr(logger.get())))
				{
					logger.log("`IPresenter::init` failed to create Command Buffer!",ILogger::ELL_ERROR);
					return false;
				}
			}

			return success = init_impl(renderer);
		}
		inline bool isInitialized() const {return bool(m_semaphore);}
		
		//
		inline video::IQueue* getQueue() const {return m_queue;}
		//
		inline video::ILogicalDevice* getDevice() const {return const_cast<video::ILogicalDevice*>(m_semaphore->getOriginDevice());}

		//
		virtual bool irrecoverable() const {return false;}

		// returns expected presentation time for frame pacing
		using clock_t = std::chrono::steady_clock;
		inline clock_t::time_point acquire(const CSession* background)
		{
			auto expectedPresent = clock_t::time_point::min(); // invalid value
			m_currentImageAcquire = {};
			if (!background)
			{
				m_currentSessionDS = nullptr;
				return expectedPresent;
			}
			m_currentSessionDS = background->getActiveResources().immutables.ds;
			return acquire_impl(background,&m_currentImageAcquire);
		}

		//
		inline video::IGPUCommandBuffer* beginRenderpass()
		{
			if (!isInitialized() || !m_currentImageAcquire.semaphore)
				return nullptr;

			using namespace nbl::video;
			if (m_presentCount>=CircularBufferSize)
			{
				const ISemaphore::SWaitInfo cbDonePending[] =
				{
					{
						.semaphore = m_semaphore.get(),
						.value = m_presentCount+1-CircularBufferSize
					}
				};
				if (getDevice()->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return {};
			}
			
			auto* const cb = getCurrentCmdBuffer();
			cb->getPool()->reset();
			if (!cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
				return nullptr;

			if (!beginRenderpass_impl())
			{
				cb->end();
				return nullptr;
			}
			return cb;
		}

		//
		inline bool endRenderpassAndPresent(const video::IQueue::SSubmitInfo::SSemaphoreInfo& extraSubmitWait)
		{
			using namespace nbl::asset;
			using namespace nbl::video;
			auto* const cb = getCurrentCmdBuffer();
			if (cb->getState()!=IGPUCommandBuffer::STATE::RECORDING)
				return false;

			if (!endRenderpass() || !cb->end())
				return false;
			
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
			{
				{
					.semaphore = m_semaphore.get(),
					.value = ++m_presentCount,
					.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
				}
			};
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
			{
				{.cmdbuf = cb}
			};
			const IQueue::SSubmitInfo::SSemaphoreInfo wait[] =
			{
				{
					.semaphore = const_cast<ISemaphore*>(m_currentImageAcquire.semaphore),
					.value = m_currentImageAcquire.value,
					.stageMask = PIPELINE_STAGE_FLAGS::NONE
				},
				extraSubmitWait
			};
			IQueue::SSubmitInfo infos[] =
			{
				{
					.waitSemaphores = wait,
					.commandBuffers = commandBuffers,
					.signalSemaphores = rendered
				}
			};
			if (!extraSubmitWait.semaphore)
				infos->waitSemaphores;
	
			if (m_queue->submit(infos)!=IQueue::RESULT::SUCCESS)
			{
				m_presentCount--;
				return false;
			}
			return present(*rendered);
		}

	protected:
		inline IPresenter(SCachedCreationParams&& _params) : m_creation(std::move(_params)) {}
		virtual bool init_impl(CRenderer* renderer) = 0;

		virtual clock_t::time_point acquire_impl(const CSession* background, video::ISemaphore::SWaitInfo* p_currentImageAcquire) = 0;
		virtual bool beginRenderpass_impl() = 0;
		virtual bool endRenderpass()
		{
			return getCurrentCmdBuffer()->endRenderPass();
		}
		virtual bool present(const video::IQueue::SSubmitInfo::SSemaphoreInfo& readyToPresent) = 0;

		inline video::IGPUDescriptorSet* getCurrentSessionDS() const {return m_currentSessionDS.get();}
		inline video::IGPUCommandBuffer* getCurrentCmdBuffer() const {return m_cmdbufs[m_presentCount % CircularBufferSize].get();}

	private:
		SCachedCreationParams m_creation;
		video::CThreadSafeQueueAdapter* m_queue;
		core::smart_refctd_ptr<video::ISemaphore> m_semaphore;
		std::array<core::smart_refctd_ptr<video::IGPUCommandBuffer>,CircularBufferSize> m_cmdbufs;
		video::ISemaphore::SWaitInfo m_currentImageAcquire = {};
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_currentSessionDS;
		uint64_t m_presentCount = 0;
};

}
#endif
