// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nabla.h>
#include "nbl/asset/utils/CGeometryCreator.h"
#include "nbl/video/utilities/CSimpleResizeSurface.h"

#include "../common/SimpleWindowedApplication.hpp"
#include "../common/InputSystem.hpp"

#include "nbl/api/CCamera.hpp"
#include "nbl/api/hlsl/SBasicViewParameters.hlsl"

#include "geometry/creator/spirv/builtin/CArchive.h"
#include "geometry/creator/spirv/builtin/builtinResources.h"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

class CSwapchainFramebuffersAndDepth final : public nbl::video::CDefaultSwapchainFramebuffers
{
	using base_t = CDefaultSwapchainFramebuffers;

public:
	template<typename... Args>
	inline CSwapchainFramebuffersAndDepth(ILogicalDevice* device, const asset::E_FORMAT _desiredDepthFormat, Args&&... args) : CDefaultSwapchainFramebuffers(device, std::forward<Args>(args)...)
	{
		const IPhysicalDevice::SImageFormatPromotionRequest req = {
			.originalFormat = _desiredDepthFormat,
			.usages = {IGPUImage::EUF_RENDER_ATTACHMENT_BIT}
		};
		m_depthFormat = m_device->getPhysicalDevice()->promoteImageFormat(req, IGPUImage::TILING::OPTIMAL);

		const static IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] = {
			{{
				{
					.format = m_depthFormat,
					.samples = IGPUImage::ESCF_1_BIT,
					.mayAlias = false
				},
			/*.loadOp = */{IGPURenderpass::LOAD_OP::CLEAR},
			/*.storeOp = */{IGPURenderpass::STORE_OP::STORE},
			/*.initialLayout = */{IGPUImage::LAYOUT::UNDEFINED}, // because we clear we don't care about contents
			/*.finalLayout = */{IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL} // transition to presentation right away so we can skip a barrier
		}},
		IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd
		};
		m_params.depthStencilAttachments = depthAttachments;

		static IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
			m_params.subpasses[0],
			IGPURenderpass::SCreationParams::SubpassesEnd
		};
		subpasses[0].depthStencilAttachment.render = { .attachmentIndex = 0,.layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL };
		m_params.subpasses = subpasses;
	}

protected:
	inline bool onCreateSwapchain_impl(const uint8_t qFam) override
	{
		auto device = const_cast<ILogicalDevice*>(m_renderpass->getOriginDevice());

		const auto depthFormat = m_renderpass->getCreationParameters().depthStencilAttachments[0].format;
		const auto& sharedParams = getSwapchain()->getCreationParameters().sharedParams;
		auto image = device->createImage({ IImage::SCreationParams{
			.type = IGPUImage::ET_2D,
			.samples = IGPUImage::ESCF_1_BIT,
			.format = depthFormat,
			.extent = {sharedParams.width,sharedParams.height,1},
			.mipLevels = 1,
			.arrayLayers = 1,
			.depthUsage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT
		} });

		device->allocate(image->getMemoryReqs(), image.get());

		m_depthBuffer = device->createImageView({
			.flags = IGPUImageView::ECF_NONE,
			.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
			.image = std::move(image),
			.viewType = IGPUImageView::ET_2D,
			.format = depthFormat,
			.subresourceRange = {IGPUImage::EAF_DEPTH_BIT,0,1,0,1}
			});

		const auto retval = base_t::onCreateSwapchain_impl(qFam);
		m_depthBuffer = nullptr;
		return retval;
	}

	inline smart_refctd_ptr<IGPUFramebuffer> createFramebuffer(IGPUFramebuffer::SCreationParams&& params) override
	{
		params.depthStencilAttachments = &m_depthBuffer.get();
		return m_device->createFramebuffer(std::move(params));
	}

	E_FORMAT m_depthFormat;
	// only used to pass a parameter from `onCreateSwapchain_impl` to `createFramebuffer`
	smart_refctd_ptr<IGPUImageView> m_depthBuffer;
};

class CEventCallback : public ISimpleManagedSurface::ICallback
{
public:
	CEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(m_inputSystem)), m_logger(std::move(logger)) {}
	CEventCallback() {}

	void setLogger(nbl::system::logger_opt_smart_ptr& logger)
	{
		m_logger = logger;
	}
	void setInputSystem(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem)
	{
		m_inputSystem = std::move(m_inputSystem);
	}
private:

	void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
	{
		m_logger.log("A mouse %p has been connected", nbl::system::ILogger::ELL_INFO, mch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_mouse, std::move(mch));
	}
	void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
	{
		m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse, mch);
	}
	void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
	{
		m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard, std::move(kbch));
	}
	void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
	{
		m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard, kbch);
	}

private:
	nbl::core::smart_refctd_ptr<InputSystem> m_inputSystem = nullptr;
	nbl::system::logger_opt_smart_ptr m_logger = nullptr;
};

class GeometryCreatorApp final : public examples::SimpleWindowedApplication
{
		using device_base_t = examples::SimpleWindowedApplication;
		using clock_t = std::chrono::steady_clock;

		constexpr static inline uint32_t WIN_W = 1280, WIN_H = 720, SC_IMG_COUNT = 3u, FRAMES_IN_FLIGHT = 5u;
		static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

		constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

	public:
		inline GeometryCreatorApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		virtual SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
		{
			auto retval = device_base_t::getRequiredDeviceFeatures();
			retval.geometryShader = true;
			return retval;
		}

		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			if (!m_surface)
			{
				{
					auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
					IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
					params.width = WIN_W;
					params.height = WIN_H;
					params.x = 32;
					params.y = 32;
					params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
					params.windowCaption = "GeometryCreatorApp";
					params.callback = windowCallback;
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}

				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>::create(std::move(surface));
			}

			if (m_surface)
				return { {m_surface->getSurface()/*,EQF_NONE*/} };

			return {};
		}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;

			m_semaphore = m_device->createSemaphore(m_submitIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			ISwapchain::SCreationParams swapchainParams = { .surface = m_surface->getSurface() };
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");

			// Subsequent submits don't wait for each other, hence its important to have External Dependencies which prevent users of the depth attachment overlapping.
			const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
				// wipe-transition of Color to ATTACHMENT_OPTIMAL
				{
					.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.dstSubpass = 0,
					.memoryBarrier = {
					// last place where the depth can get modified in previous frame
					.srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,
					// only write ops, reads can't be made available
					.srcAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
					// destination needs to wait as early as possible
					.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
					// because of depth test needing a read and a write
					.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_READ_BIT
				}
				// leave view offsets and flags default
			},
				// color from ATTACHMENT_OPTIMAL to PRESENT_SRC
				{
					.srcSubpass = 0,
					.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.memoryBarrier = {
					// last place where the depth can get modified
					.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					// only write ops, reads can't be made available
					.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					// spec says nothing is needed when presentation is the destination
				}
				// leave view offsets and flags default
			},
			IGPURenderpass::SCreationParams::DependenciesEnd
			};

			auto scResources = std::make_unique<CSwapchainFramebuffersAndDepth>(m_device.get(), EF_D16_UNORM, swapchainParams.surfaceFormat.format, dependencies);
			auto* renderpass = scResources->getRenderpass();

			if (!renderpass)
				return logFail("Failed to create Renderpass!");

			auto gQueue = getGraphicsQueue();
			if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");

			m_maxFramesInFlight = m_surface->getMaxFramesInFlight();
			if (FRAMES_IN_FLIGHT < m_maxFramesInFlight)
			{
				m_logger->log("Lowering frames in flight!", ILogger::ELL_WARNING);
				m_maxFramesInFlight = FRAMES_IN_FLIGHT;
			}

			m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

			for (auto i = 0u; i < m_maxFramesInFlight; i++)
			{
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");
				if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
					return logFail("Couldn't create Command Buffer!");
			}

			m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
			m_surface->recreateSwapchain();

			auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(system));
			auto* geometry = assetManager->getGeometryCreator();

			nbl::video::IGPUDescriptorSetLayout::SBinding bindings[] = {
				{
					.binding = 0u,
					.type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_VERTEX | asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
				}
			};

			auto descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);
			{
				const video::IGPUDescriptorSetLayout* const layouts[] = { nullptr, descriptorSetLayout.get() };
				const uint32_t setCounts[] = { 0u, 1u };
				m_descriptorPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
				if (!m_descriptorPool)
					return logFail("Failed to Create Descriptor Pool");
			}

			m_gpuDescriptorSet = m_descriptorPool->createDescriptorSet(descriptorSetLayout);

			if (!m_gpuDescriptorSet)
				return logFail("Could not create Descriptor Set!");

			auto pipelineLayout = m_device->createPipelineLayout({}, nullptr, std::move(descriptorSetLayout));

			if (!pipelineLayout)
				return logFail("Could not create Pipeline Layout!");

			struct
			{
				const IGeometryCreator* gc;

				const std::vector<O_DATA>
				basic = 
				{ 
					std::make_pair(gc->createCubeMesh(vector3df(1.f, 1.f, 1.f)), "Cube Mesh"),

					std::make_pair(gc->createSphereMesh(2, 16, 16), "Sphere Mesh"),
					std::make_pair(gc->createCylinderMesh(2, 2, 20), "Cylinder Mesh"),
					std::make_pair(gc->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3)), "Rectangle Mesh"),
					std::make_pair(gc->createDiskMesh(2, 30), "Disk Mesh"),
					std::make_pair(gc->createArrowMesh(), "Arrow Mesh")
				},
				cone =
				{
					std::make_pair(gc->createConeMesh(2, 3, 10), "Cone Mesh")
				}, 
				ico =
				{
					std::make_pair(gc->createIcoSphere(1, 3, true), "Icoshpere Mesh")
				}, 
				grid =
				{
					std::make_pair(gc->createRectangleMesh(vector2df_SIMD(999.f, 999.f)), "Grid on Scene")
				};
			} geometries{.gc = geometry };

			auto createBundlePassData = [&]<E_PASS_TYPE ept, nbl::core::StringLiteral vPath, nbl::core::StringLiteral fPath>(const auto& objects)
			{
				for (auto& object : objects)
					if (!createPassData<vPath,fPath>(ept, object, pipelineLayout.get(), renderpass))
						return logFail("Could not create pass data!");

				return true;
			};

			if (!createBundlePassData.template operator() < EPT_GEOMETRY_CREATOR, NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.vertex.spv"), NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv")> (geometries.basic))
				return false;

			if (!createBundlePassData.template operator() < EPT_GEOMETRY_CREATOR, NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.cone.vertex.spv"), NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (geometries.cone)) // note we reuse basic fragment shader
				return false;

			if (!createBundlePassData.template operator() < EPT_GEOMETRY_CREATOR, NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.ico.vertex.spv"), NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (geometries.ico)) // note we reuse basic fragment shader
				return false;

			if (!createBundlePassData.template operator() < EPT_GRID, NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/grid.vertex.spv"), NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/grid.fragment.spv") > (geometries.grid))
				return false;

			// gpu resources
			{
				const auto mask = m_device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

				m_ubo = m_device->createBuffer({{.size = sizeof(SBasicViewParameters), .usage = core::bitflag(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT) | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF} });

				for (auto it : { m_ubo })
				{
					IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = it->getMemoryReqs();
					reqs.memoryTypeBits &= mask;

					m_device->allocate(reqs, it.get());
				}

				{
					video::IGPUDescriptorSet::SWriteDescriptorSet write;
					write.dstSet = m_gpuDescriptorSet.get();
					write.binding = 0;
					write.arrayElement = 0u;
					write.count = 1u;
					video::IGPUDescriptorSet::SDescriptorInfo info;
					{
						info.desc = core::smart_refctd_ptr(m_ubo);
						info.info.buffer.offset = 0ull;
						info.info.buffer.size = m_ubo->getSize();
					}
					write.info = &info;
					m_device->updateDescriptorSets(1u, &write, 0u, nullptr);
				}
			}

			// camera
			{
				core::vectorSIMDf cameraPosition(-5.81655884, 2.58630896, -4.23974705);
				core::vectorSIMDf cameraTarget(-0.349590302, -0.213266611, 0.317821503);
				matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.1, 10000);
				camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 1.069f, 0.4f);
			}

			m_winMgr->show(m_window.get());
			oracle.reportBeginFrameRecord();

			return true;
		}

		inline void workLoopBody() override
		{
			const auto resourceIx = m_realFrameIx % m_maxFramesInFlight;

			if (m_realFrameIx >= m_maxFramesInFlight)
			{
				const ISemaphore::SWaitInfo cbDonePending[] =
				{
					{
						.semaphore = m_semaphore.get(),
						.value = m_realFrameIx + 1 - m_maxFramesInFlight
					}
				};
				if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

			m_inputSystem->getDefaultMouse(&mouse);
			m_inputSystem->getDefaultKeyboard(&keyboard);

			auto updatePresentationTimestamp = [&]()
			{
				m_currentImageAcquire = m_surface->acquireNextImage();

				oracle.reportEndFrameRecord();
				const auto timestamp = oracle.getNextPresentationTimeStamp();
				oracle.reportBeginFrameRecord();

				return timestamp;
			};

			const auto nextPresentationTimestamp = updatePresentationTimestamp();

			if (!m_currentImageAcquire)
				return;

			auto* const cb = m_cmdBufs.data()[resourceIx].get();
			cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cb->beginDebugMarker("GeometryCreatorApp Frame");
			{
				camera.beginInputProcessing(nextPresentationTimestamp);
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); mouseProcess(events); }, m_logger.get());
				keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, m_logger.get());
				camera.endInputProcessing(nextPresentationTimestamp);
			}

			const auto viewMatrix = camera.getViewMatrix();
			const auto viewProjectionMatrix = camera.getConcatenatedMatrix();

			core::matrix3x4SIMD modelMatrix;
			modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
			modelMatrix.setRotation(quaternion(0, 0, 0));

			core::matrix3x4SIMD modelViewMatrix = core::concatenateBFollowedByA(viewMatrix, modelMatrix);
			core::matrix4SIMD modelViewProjectionMatrix = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

			core::matrix3x4SIMD normalMatrix;
			modelViewMatrix.getSub3x3InverseTranspose(normalMatrix);

			SBasicViewParameters uboData;
			memcpy(uboData.MVP, modelViewProjectionMatrix.pointer(), sizeof(uboData.MVP));
			memcpy(uboData.MV, modelViewMatrix.pointer(), sizeof(uboData.MV));
			memcpy(uboData.NormalMat, normalMatrix.pointer(), sizeof(uboData.NormalMat));
			{

				SBufferRange<IGPUBuffer> range;
				range.buffer = core::smart_refctd_ptr(m_ubo);
				range.size = m_ubo->getSize();

				cb->updateBuffer(range, &uboData);
			}

			auto* queue = getGraphicsQueue();

			asset::SViewport viewport;
			{
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = m_window->getWidth();
				viewport.height = m_window->getHeight();
			}
			cb->setViewport(0u, 1u, &viewport);
		
			VkRect2D scissor =
			{
				.offset = { 0, 0 },
				.extent = { m_window->getWidth(), m_window->getHeight() },
			};
			cb->setScissor(0u, 1u, &scissor);

			{
				const VkRect2D currentRenderArea =
				{
					.offset = {0,0},
					.extent = {m_window->getWidth(),m_window->getHeight()}
				};

				const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
				const IGPUCommandBuffer::SClearDepthStencilValue depthValue = { .depth = 0.f };
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				const IGPUCommandBuffer::SRenderpassBeginInfo info =
				{
					.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = &depthValue,
					.renderArea = currentRenderArea
				};

				cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
			}

			auto render = [&]<E_PASS_TYPE ept>(uint16_t index = 0) -> void
			{
				auto& hook = gpu.pass[ept][index];

				auto* rawPipeline = hook.pipeline.get();
				cb->bindGraphicsPipeline(rawPipeline);
				cb->bindDescriptorSets(EPBP_GRAPHICS, rawPipeline->getLayout(), 1, 1, &m_gpuDescriptorSet.get());

				const asset::SBufferBinding<const IGPUBuffer> bVertices[] = { {.offset = 0, .buffer = hook.m_vertexBuffer} };
				const asset::SBufferBinding<const IGPUBuffer> bIndices = { .offset = 0, .buffer = hook.m_indexBuffer };

				cb->bindVertexBuffers(0, 1, bVertices);

				if (bIndices.buffer && hook.indexType != EIT_UNKNOWN)
				{
					cb->bindIndexBuffer(bIndices, hook.indexType);
					cb->drawIndexed(hook.indexCount, 1, 0, 0, 0);
				}
				else
					cb->draw(hook.indexCount, 1, 0, 0);
			};

			render.template operator() < EPT_GEOMETRY_CREATOR > (gcIndex);
			render.template operator() < EPT_GRID > ();

			cb->endRenderPass();
			cb->end();
			{
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
				{
					{
						.semaphore = m_semaphore.get(),
						.value = ++m_submitIx,
						.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
					}
				};
				{
					{
						const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
						{
							{.cmdbuf = cb }
						};

						const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] =
						{
							{
								.semaphore = m_currentImageAcquire.semaphore,
								.value = m_currentImageAcquire.acquireCount,
								.stageMask = PIPELINE_STAGE_FLAGS::NONE
							}
						};
						const IQueue::SSubmitInfo infos[] =
						{
							{
								.waitSemaphores = acquired,
								.commandBuffers = commandBuffers,
								.signalSemaphores = rendered
							}
						};

						if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
							m_submitIx--;
					}
				}

				std::string caption = "[Nabla Engine] Geometry Creator";
				{
					caption += ", displaying [" + gpu.pass[EPT_GEOMETRY_CREATOR][gcIndex].displayName + "]";
					m_window->setCaption(caption);
				}
				m_surface->present(m_currentImageAcquire.imageIndex, rendered);
			}

			m_realFrameIx++;
		}

		inline bool keepRunning() override
		{
			if (m_surface->irrecoverable())
				return false;

			return true;
		}

		inline bool onAppTerminated() override
		{
			return device_base_t::onAppTerminated();
		}

	private:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>> m_surface;
		smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;
		smart_refctd_ptr<ISemaphore> m_semaphore;
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		uint64_t m_realFrameIx : 59 = 0;
		uint64_t m_submitIx : 59 = 0;
		uint64_t m_maxFramesInFlight : 5;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
		video::CDumbPresentationOracle oracle;

		core::smart_refctd_ptr<video::IDescriptorPool> m_descriptorPool;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_gpuDescriptorSet;

		using O_DATA = std::pair<IGeometryCreator::return_type, std::string>;

		enum E_PASS_TYPE
		{
			EPT_GEOMETRY_CREATOR,
			EPT_GRID,
			EPT_COUNT
		};

		struct Pass {
			core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
			core::smart_refctd_ptr<video::IGPUBuffer> m_vertexBuffer, m_indexBuffer;
			E_INDEX_TYPE indexType;
			uint32_t indexCount;
			std::string displayName;
		};

		struct GPUPData
		{
			std::array<std::vector<Pass>, EPT_COUNT> pass = {};
		} gpu;

		uint16_t gcIndex = {};
		core::smart_refctd_ptr<video::IGPUBuffer> m_ubo;

		template<nbl::core::StringLiteral vPath, nbl::core::StringLiteral fPath>
		bool createPassData(E_PASS_TYPE ept, const O_DATA& oData, const video::IGPUPipelineLayout* pl, const video::IGPURenderpass* rp)
		{
			const auto& geo = oData.first;

			struct
			{
				core::smart_refctd_ptr<video::IGPUShader> vertex, geometry, fragment;
			} shaders;

			{
				struct
				{
					const system::SBuiltinFile vertex = ::geometry::creator::spirv::builtin::get_resource<vPath>();
					const system::SBuiltinFile fragment = ::geometry::creator::spirv::builtin::get_resource<fPath>();
				} spirv;

				auto createShader = [&](const system::SBuiltinFile& in, asset::IShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<video::IGPUShader>
				{
					const auto buffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>, true> >(in.size, (void*)in.contents, core::adopt_memory);
					const auto shader = make_smart_refctd_ptr<ICPUShader>(core::smart_refctd_ptr(buffer), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, "");

					// also first should look for cached/already created to not duplicate
					return m_device->createShader(shader.get());
				};

				shaders.vertex = createShader(spirv.vertex, IShader::E_SHADER_STAGE::ESS_VERTEX);
				shaders.fragment = createShader(spirv.fragment, IShader::E_SHADER_STAGE::ESS_FRAGMENT);
			}

			SBlendParams blendParams{};
			{
				blendParams.logicOp = ELO_NO_OP;

				auto& param = blendParams.blendParams[0];
				param.srcColorFactor = EBF_SRC_ALPHA;//VK_BLEND_FACTOR_SRC_ALPHA;
				param.dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
				param.colorBlendOp = EBO_ADD;//VK_BLEND_OP_ADD;
				param.srcAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
				param.dstAlphaFactor = EBF_ZERO;//VK_BLEND_FACTOR_ZERO;
				param.alphaBlendOp = EBO_ADD;//VK_BLEND_OP_ADD;
				param.colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);//VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			}

			SRasterizationParams rasterizationParams{};
			rasterizationParams.faceCullingMode = EFCM_NONE;

			{
				const IGPUShader::SSpecInfo specs[] =
				{
					{.entryPoint = "VSMain", .shader = shaders.vertex.get() },
					{.entryPoint = "PSMain", .shader = shaders.fragment.get() }
				};

				IGPUGraphicsPipeline::SCreationParams params[1];
				{
					auto& param = params[0];
					param.layout = pl;
					param.shaders = specs;
					param.renderpass = rp;
					param.cached = { .vertexInput = geo.inputParams, .primitiveAssembly = geo.assemblyParams, .rasterization = rasterizationParams, .blend = blendParams, .subpassIx = 0u };
				};

				auto& hook = gpu.pass[ept].emplace_back();

				hook.indexCount = geo.indexCount;
				hook.indexType = geo.indexType;
				hook.displayName = oData.second;

				// first should look for cached pipeline to not duplicate but lets leave how it is now
				if (!m_device->createGraphicsPipelines(nullptr, params, &hook.pipeline))
					return false;

				if (!createVIBuffers(hook, geo))
					return false;

				return true;
			}
		}

		bool createVIBuffers(Pass& hook, const CGeometryCreator::return_type& oData)
		{
			const auto mask = m_device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

			auto vBuffer = core::smart_refctd_ptr(oData.bindings[0].buffer); // no offset
			auto iBuffer = core::smart_refctd_ptr(oData.indexBuffer.buffer); // no offset

			hook.m_vertexBuffer = m_device->createBuffer({ {.size = vBuffer->getSize(), .usage = core::bitflag(asset::IBuffer::EUF_VERTEX_BUFFER_BIT) | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF}});
			hook.m_indexBuffer = iBuffer ? m_device->createBuffer({ {.size = iBuffer->getSize(), .usage = core::bitflag(asset::IBuffer::EUF_INDEX_BUFFER_BIT) | asset::IBuffer::EUF_VERTEX_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF}}) : nullptr;

			if (!hook.m_vertexBuffer)
				return false;

			if (oData.indexType != EIT_UNKNOWN)
				if (!hook.m_indexBuffer)
					return false;

			for (auto it : { hook.m_vertexBuffer , hook.m_indexBuffer })
			{
				if (it)
				{
					IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = it->getMemoryReqs();
					reqs.memoryTypeBits &= mask;

					m_device->allocate(reqs, it.get());
				}
			}

			{
				auto fillGPUBuffer = [&m_logger = m_logger](smart_refctd_ptr<ICPUBuffer> cBuffer, smart_refctd_ptr<IGPUBuffer> gBuffer)
				{
					auto binding = gBuffer->getBoundMemory();

					if (!binding.memory->map({ 0ull, binding.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
					{
						m_logger->log("Could not map device memory", system::ILogger::ELL_ERROR);
						return false;
					}

					if (!binding.memory->isCurrentlyMapped())
					{
						m_logger->log("Buffer memory is not mapped!", system::ILogger::ELL_ERROR);
						return false;
					}

					auto* mPointer = binding.memory->getMappedPointer();
					memcpy(mPointer, cBuffer->getPointer(), gBuffer->getSize());
					binding.memory->unmap();

					return true;
				};

				if (!fillGPUBuffer(vBuffer, hook.m_vertexBuffer))
					return false;

				if(hook.m_indexBuffer)
					if (!fillGPUBuffer(iBuffer, hook.m_indexBuffer))
						return false;
			}

			return true;
		}

		void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
		{
			for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
			{
				auto ev = *eventIt;

				if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL)
					gcIndex = std::clamp<uint16_t>(int16_t(gcIndex) + int16_t(core::sign(ev.scrollEvent.verticalScroll)), int64_t(0), int64_t( gpu.pass[EPT_GEOMETRY_CREATOR].size() - 1));
			}
		}
};

NBL_MAIN_FUNC(GeometryCreatorApp)