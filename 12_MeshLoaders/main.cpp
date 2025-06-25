// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "common.hpp"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"


class MeshLoadersApp final : public MonoWindowApplication, public BuiltinResourcesApplication
{
		using device_base_t = MonoWindowApplication;
		using asset_base_t = BuiltinResourcesApplication;

	public:
		inline MeshLoadersApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
			device_base_t({1280,720}, EF_UNKNOWN, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;

			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			for (auto i=0u; i<MaxFramesInFlight; i++)
			{
				if (!pool)
					return logFail("Couldn't create Command Pool!");
				if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdBufs.data()+i,1}))
					return logFail("Couldn't create Command Buffer!");
			}
			
			//! cache results -- speeds up mesh generation on second run
			m_qnc = make_smart_refctd_ptr<CQuantNormalCache>();
			m_qnc->loadCacheFromFile<EF_R8G8B8_SNORM>(m_system.get(),sharedOutputCWD/"../../tmp/normalCache888.sse");

			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			m_renderer = CSimpleDebugRenderer::create(m_assetMgr.get(),scRes->getRenderpass(),0,{});
			if (!m_renderer)
				return logFail("Failed to create renderer!");

			//
			if (!reloadModel())
				return false;

			camera.mapKeysToArrows();

			onAppInitializedFinish();
			return true;
		}

		inline IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override
		{
			m_inputSystem->getDefaultMouse(&mouse);
			m_inputSystem->getDefaultKeyboard(&keyboard);

			//
			const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

			auto* const cb = m_cmdBufs.data()[resourceIx].get();
			cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			// clear to black for both things
			{
				// begin renderpass
				{
					auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
					auto* framebuffer = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex);
					const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {1.f,0.f,1.f,1.f} };
					const IGPUCommandBuffer::SClearDepthStencilValue depthValue = { .depth = 0.f };
					const VkRect2D currentRenderArea =
					{
						.offset = {0,0},
						.extent = {framebuffer->getCreationParameters().width,framebuffer->getCreationParameters().height}
					};
					const IGPUCommandBuffer::SRenderpassBeginInfo info =
					{
						.framebuffer = framebuffer,
						.colorClearValues = &clearValue,
						.depthStencilClearValues = &depthValue,
						.renderArea = currentRenderArea
					};
					cb->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

					const SViewport viewport = {
						.x = static_cast<float>(currentRenderArea.offset.x),
						.y = static_cast<float>(currentRenderArea.offset.y),
						.width = static_cast<float>(currentRenderArea.extent.width),
						.height = static_cast<float>(currentRenderArea.extent.height)
					};
					cb->setViewport(0u,1u,&viewport);
		
					cb->setScissor(0u,1u,&currentRenderArea);
				}
				// late latch input
				{
					camera.beginInputProcessing(nextPresentationTimestamp);
					mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, m_logger.get());
					keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
						{
							camera.keyboardProcess(events);
						},
						m_logger.get()
					);
					camera.endInputProcessing(nextPresentationTimestamp);
				}
				// draw scene
				{
					float32_t3x4 viewMatrix;
					float32_t4x4 viewProjMatrix;
					// TODO: get rid of legacy matrices
					{
						memcpy(&viewMatrix,camera.getViewMatrix().pointer(),sizeof(viewMatrix));
						memcpy(&viewProjMatrix,camera.getConcatenatedMatrix().pointer(),sizeof(viewProjMatrix));
					}
 					m_renderer->render(cb,CSimpleDebugRenderer::SViewParams(viewMatrix,viewProjMatrix));
				}
				cb->endRenderPass();
			}
			cb->end();

			//updateGUIDescriptorSet();

			IQueue::SSubmitInfo::SSemaphoreInfo retval =
			{
				.semaphore = m_semaphore.get(),
				.value = ++m_realFrameIx,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS
			};
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
			{
				{.cmdbuf = cb }
			};
			const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = {
				{
					.semaphore = device_base_t::getCurrentAcquire().semaphore,
					.value = device_base_t::getCurrentAcquire().acquireCount,
					.stageMask = PIPELINE_STAGE_FLAGS::NONE
				}
			};
			const IQueue::SSubmitInfo infos[] =
			{
				{
					.waitSemaphores = acquired,
					.commandBuffers = commandBuffers,
					.signalSemaphores = {&retval,1}
				}
			};
			
			if (getGraphicsQueue()->submit(infos) != IQueue::RESULT::SUCCESS)
			{
				retval.semaphore = nullptr; // so that we don't wait on semaphore that will never signal
				m_realFrameIx--;
			}

			std::string caption = "[Nabla Engine] Mesh Loaders";
			{
				caption += ", displaying [";
				caption += m_modelPath;
				caption += "]";
				m_window->setCaption(caption);
			}
			return retval;
		}

	protected:
		const video::IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const override
		{
			// Subsequent submits don't wait for each other, hence its important to have External Dependencies which prevent users of the depth attachment overlapping.
			const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
				// wipe-transition of Color to ATTACHMENT_OPTIMAL and depth
				{
					.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.dstSubpass = 0,
					.memoryBarrier = {
						// last place where the depth can get modified in previous frame, `COLOR_ATTACHMENT_OUTPUT_BIT` is implicitly later
						.srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,
						// don't want any writes to be available, we'll clear 
						.srcAccessMask = ACCESS_FLAGS::NONE,
						// destination needs to wait as early as possible
						// TODO: `COLOR_ATTACHMENT_OUTPUT_BIT` shouldn't be needed, because its a logically later stage, see TODO in `ECommonEnums.h`
						.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						// because depth and color get cleared first no read mask
						.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
					// leave view offsets and flags default
				},
				// color from ATTACHMENT_OPTIMAL to PRESENT_SRC
				{
					.srcSubpass = 0,
					.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.memoryBarrier = {
						// last place where the color can get modified, depth is implicitly earlier
						.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						// only write ops, reads can't be made available
						.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
						// spec says nothing is needed when presentation is the destination
					}
					// leave view offsets and flags default
				},
				IGPURenderpass::SCreationParams::DependenciesEnd
			};
			return dependencies;
		}

	private:
		// TODO: standardise this across examples, and take from `argv`
		bool m_nonInteractiveTest = true;

		inline bool reloadModel()
		{
			if (m_nonInteractiveTest) // TODO: maybe also take from argv and argc
				m_modelPath = (sharedInputCWD/"ply/Spanner-ply.ply").string();
			else
			{
				pfd::open_file file("Choose a supported Model File", sharedInputCWD.string(),
					{
						"All Supported Formats", "*.ply *.stl *.serialized *.obj",
						"TODO (.ply)", "*.ply",
						"TODO (.stl)", "*.stl",
						"Mitsuba 0.6 Serialized (.serialized)", "*.serialized",
						"Wavefront Object (.obj)", "*.obj"
					},
					false
				);
				if (file.result().empty())
					return false;
				m_modelPath = file.result()[0];
			}

			// free up
			m_renderer->m_instances.clear();
			m_renderer->clearGeometries({.semaphore=m_semaphore.get(),.value=m_realFrameIx});
			m_assetMgr->clearAllAssetCache();

			//! load the geometry
			IAssetLoader::SAssetLoadParams params = {};
			params.meshManipulatorOverride = nullptr; // TODO
			auto bundle = m_assetMgr->getAsset(m_modelPath,params);
			if (bundle.getContents().empty())
				return false;

			// 
			core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
			switch (bundle.getAssetType())
			{
				case IAsset::E_TYPE::ET_GEOMETRY:
					for (const auto& item : bundle.getContents())
					if (auto polyGeo=IAsset::castDown<ICPUPolygonGeometry>(item); polyGeo)
						geometries.push_back(polyGeo);
					break;
				default:
					m_logger->log("Asset loaded but not a supported type (ET_GEOMETRY,ET_GEOMETRY_COLLECTION)",ILogger::ELL_ERROR);
					break;
			}
			if (geometries.empty())
				return false;

			//! cache results -- speeds up mesh generation on second run
			m_qnc->saveCacheToFile<EF_R8G8B8_SNORM>(m_system.get(),sharedOutputCWD/"../../tmp/normalCache888.sse");
			
			// convert the geometries
			{
				smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({.device=m_device.get()});

				const auto transferFamily = getTransferUpQueue()->getFamilyIndex();

				struct SInputs : CAssetConverter::SInputs
				{
					virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUBuffer* buffer, const CAssetConverter::patch_t<asset::ICPUBuffer>& patch) const
					{
						return sharedBufferOwnership;
					}

					core::vector<uint32_t> sharedBufferOwnership;
				} inputs = {};
				core::vector<CAssetConverter::patch_t<ICPUPolygonGeometry>> patches(geometries.size(),CSimpleDebugRenderer::DefaultPolygonGeometryPatch);
				{
					inputs.logger = m_logger.get();
					std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = {&geometries.front().get(),geometries.size()};
					std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = patches;
					// set up shared ownership so we don't have to 
					core::unordered_set<uint32_t> families;
					families.insert(transferFamily);
					families.insert(getGraphicsQueue()->getFamilyIndex());
					if (families.size()>1)
					for (const auto fam : families)
						inputs.sharedBufferOwnership.push_back(fam);
				}
				
				// reserve
				auto reservation = converter->reserve(inputs);
				if (!reservation)
				{
					m_logger->log("Failed to reserve GPU objects for CPU->GPU conversion!",ILogger::ELL_ERROR);
					return false;
				}

				// convert
				{
					auto semaphore = m_device->createSemaphore(0u);

					constexpr auto MultiBuffering = 2;
					std::array<smart_refctd_ptr<IGPUCommandBuffer>,MultiBuffering> commandBuffers = {};
					{
						auto pool = m_device->createCommandPool(transferFamily,IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT|IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
						pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,commandBuffers,smart_refctd_ptr(m_logger));
					}
					commandBuffers.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

					std::array<IQueue::SSubmitInfo::SCommandBufferInfo,MultiBuffering> commandBufferSubmits;
					for (auto i=0; i<MultiBuffering; i++)
						commandBufferSubmits[i].cmdbuf = commandBuffers[i].get();

					SIntendedSubmitInfo transfer = {};
					transfer.queue = getTransferUpQueue();
					transfer.scratchCommandBuffers = commandBufferSubmits;
					transfer.scratchSemaphore = {
						.semaphore = semaphore.get(),
						.value = 0u,
						.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
					};

					CAssetConverter::SConvertParams cpar = {};
					cpar.utilities = m_utils.get();
					cpar.transfer = &transfer;

					// basically it records all data uploads and submits them right away
					auto future = reservation.convert(cpar);
					if (future.copy()!=IQueue::RESULT::SUCCESS)
					{
						m_logger->log("Failed to await submission feature!", ILogger::ELL_ERROR);
						return false;
					}
				}

				const auto& converted = reservation.getGPUObjects<ICPUPolygonGeometry>();
				return m_renderer->addGeometries({&converted.front().get(),converted.size()});
			}
		}

		// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
		constexpr static inline uint32_t MaxFramesInFlight = 3u;
		//
		smart_refctd_ptr<CQuantNormalCache> m_qnc;
		smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;
		//
		smart_refctd_ptr<ISemaphore> m_semaphore;
		uint64_t m_realFrameIx = 0;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,MaxFramesInFlight> m_cmdBufs;
		//
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
		//
		Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
		// mutables
		std::string m_modelPath;
};

NBL_MAIN_FUNC(MeshLoadersApp)