// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nabla.h>
#include "nbl/video/utilities/CSimpleResizeSurface.h"
#include "../common/SimpleWindowedApplication.hpp"
#include "../common/InputSystem.hpp"
#include "../common/Camera.hpp"
#include "this_example/spirv/builtin/CArchive.h"
#include "this_example/spirv/builtin/builtinResources.h"
#include "shaders/common.hlsl"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

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

class GizmoApp final : public examples::SimpleWindowedApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using clock_t = std::chrono::steady_clock;

	_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_W = 1280, WIN_H = 720, SC_IMG_COUNT = 3u, FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

public:
	inline GizmoApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

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
				params.windowCaption = "GizmoApp";
				params.callback = windowCallback;
				const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
			}

			auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<nbl::video::CDefaultSwapchainFramebuffers>::create(std::move(surface));
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

		const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] =
		{
			{
				.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.dstSubpass = 0,
				.memoryBarrier =
				{
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
					.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
					.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
				}
			},
			{
				.srcSubpass = 0,
				.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.memoryBarrier =
				{
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
				}
			},
			IGPURenderpass::SCreationParams::DependenciesEnd
		};

		auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(), swapchainParams.surfaceFormat.format, dependencies);
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

		{
			static constexpr int TotalSetCount = 1;
			IDescriptorPool::SCreateInfo createInfo = {};
			createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)] = TotalSetCount;
			createInfo.maxSets = 1;
			createInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_NONE;

			m_descriptorPool = m_device->createDescriptorPool(std::move(createInfo));
			if (!m_descriptorPool)
			{
				m_logger->log("Could not create Descriptor Pool!", system::ILogger::ELL_ERROR);
				return false;
			}
		}

		SPushConstantRange pushConstantRanges[] = {
			{
				.stageFlags = IShader::ESS_VERTEX,
				.offset = 0,
				.size = sizeof(PushConstants)
			}
		};

		nbl::video::IGPUDescriptorSetLayout::SBinding bindings[] = {
			{
				.binding = 0u,
				.type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT,
				.count = 1u,
			}
		};

		auto descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);

		m_gpuDescriptorSet = m_descriptorPool->createDescriptorSet(descriptorSetLayout);

		auto pipelineLayout = m_device->createPipelineLayout(pushConstantRanges, nullptr, std::move(descriptorSetLayout));

		struct
		{
			core::smart_refctd_ptr<video::IGPUShader> vertex, fragment;
		} shaders;

		{
			struct
			{
				const system::SBuiltinFile vertex = ::this_example::spirv::builtin::get_resource<"gizmo/spirv/vertex.spv">();
				const system::SBuiltinFile fragment = ::this_example::spirv::builtin::get_resource<"gizmo/spirv/fragment.spv">();
			} spirv;

			auto createShader = [&](const system::SBuiltinFile& in, asset::IShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<video::IGPUShader>
			{
				const auto buffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>, true> >(in.size, (void*)in.contents, core::adopt_memory);
				const auto shader = make_smart_refctd_ptr<ICPUShader>(core::smart_refctd_ptr(buffer), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, "");

				return m_device->createShader(shader.get());
			};

			shaders.vertex = createShader(spirv.vertex, IShader::ESS_VERTEX);
			shaders.fragment = createShader(spirv.fragment, IShader::ESS_FRAGMENT);
		}

		SVertexInputParams vertexInputParams{};
		{
			vertexInputParams.enabledBindingFlags = 0b1u;
			vertexInputParams.enabledAttribFlags = 0b11u;

			vertexInputParams.bindings[0].inputRate = asset::SVertexInputBindingParams::EVIR_PER_VERTEX;
			vertexInputParams.bindings[0].stride = sizeof(VSInput);

			auto& position = vertexInputParams.attributes[0];

			position.format = EF_R32G32B32A32_SFLOAT;
			position.relativeOffset = offsetof(VSInput, position);
			position.binding = 0u;

			auto& color = vertexInputParams.attributes[1];
			color.format = EF_R8G8B8A8_UNORM;
			color.relativeOffset = offsetof(VSInput, color);
			color.binding = 0u;
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
		{
			rasterizationParams.faceCullingMode = EFCM_NONE;
			rasterizationParams.depthWriteEnable = false;
			rasterizationParams.depthBoundsTestEnable = false;
		}

		SPrimitiveAssemblyParams primitiveAssemblyParams{};
		{
			primitiveAssemblyParams.primitiveType = EPT_TRIANGLE_LIST;
		}

		{
			const IGPUShader::SSpecInfo specs[] =
			{
				{.entryPoint = "VSMain", .shader = shaders.vertex.get() },
				{.entryPoint = "PSMain", .shader = shaders.fragment.get() }
			};

			IGPUGraphicsPipeline::SCreationParams params[1];
			{
				auto& param = params[0];
				param.layout = pipelineLayout.get();
				param.shaders = specs;
				param.renderpass = renderpass;
				param.cached = { .vertexInput = vertexInputParams, .primitiveAssembly = primitiveAssemblyParams, .rasterization = rasterizationParams, .blend = blendParams, .subpassIx = 0u };
			};

			if (!m_device->createGraphicsPipelines(nullptr, params, &pipeline))
			{
				m_logger->log("Could not create pipeline!", system::ILogger::ELL_ERROR);
				assert(false);
			}
		}

		auto getRandomColor = []
		{
			static std::random_device rd;
			static std::mt19937 gen(rd());
			static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

			return dis(gen);
		};

		// cube
		auto vertices = std::to_array(
		{
			VSInput{{-1.0f, -1.0f, -1.0f, 1.0f}, {getRandomColor(), getRandomColor(), getRandomColor(), 1.0f}}, // vertex 0
			VSInput{{ 1.0f, -1.0f, -1.0f, 1.0f}, {getRandomColor(), getRandomColor(), getRandomColor(), 1.0f}}, // vertex 1
			VSInput{{ 1.0f,  1.0f, -1.0f, 1.0f}, {getRandomColor(), getRandomColor(), getRandomColor(), 1.0f}}, // vertex 2
			VSInput{{-1.0f,  1.0f, -1.0f, 1.0f}, {getRandomColor(), getRandomColor(), getRandomColor(), 1.0f}}, // vertex 3
			VSInput{{-1.0f, -1.0f,  1.0f, 1.0f}, {getRandomColor(), getRandomColor(), getRandomColor(), 1.0f}}, // vertex 4
			VSInput{{ 1.0f, -1.0f,  1.0f, 1.0f}, {getRandomColor(), getRandomColor(), getRandomColor(), 1.0f}}, // vertex 5
			VSInput{{ 1.0f,  1.0f,  1.0f, 1.0f}, {getRandomColor(), getRandomColor(), getRandomColor(), 1.0f}}, // vertex 6
			VSInput{{-1.0f,  1.0f,  1.0f, 1.0f}, {getRandomColor(), getRandomColor(), getRandomColor(), 1.0f}}  // vertex 7
		});

		// indices of the cube
		_NBL_STATIC_INLINE_CONSTEXPR auto indices = std::to_array<uint16_t>(
		{
			// Front face
			0, 1, 2, 2, 3, 0,
			// Back face
			4, 5, 6, 6, 7, 4,
			// Left face
			0, 3, 7, 7, 4, 0,
			// Right face
			1, 5, 6, 6, 2, 1,
			// Top face
			3, 2, 6, 6, 7, 3,
			// Bottom face
			0, 1, 5, 5, 4, 0
		});

		// buffers
		{
			m_ubo = m_device->createBuffer({{.size = sizeof(SBasicViewParameters), .usage = core::bitflag(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT) | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF} });
			m_vertexBuffer = m_device->createBuffer({{.size = sizeof(vertices), .usage = core::bitflag(asset::IBuffer::EUF_VERTEX_BUFFER_BIT) | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF} });
			m_indexBuffer = m_device->createBuffer({{.size = sizeof(indices), .usage = core::bitflag(asset::IBuffer::EUF_INDEX_BUFFER_BIT) | asset::IBuffer::EUF_VERTEX_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF} });

			const auto mask = m_device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
			for (auto it : { m_ubo , m_vertexBuffer , m_indexBuffer })
			{
				IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = it->getMemoryReqs();
				reqs.memoryTypeBits &= mask;

				m_device->allocate(reqs, it.get());
			}

			{
				auto vBinding = m_vertexBuffer->getBoundMemory();
				auto iBinding = m_indexBuffer->getBoundMemory();

				{
					if (!vBinding.memory->map({ 0ull, vBinding.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
						m_logger->log("Could not map device memory for vertex buffer data!", system::ILogger::ELL_ERROR);

					assert(vBinding.memory->isCurrentlyMapped());
				}

				{
					if (!iBinding.memory->map({ 0ull, iBinding.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
						m_logger->log("Could not map device memory for index buffer data!", system::ILogger::ELL_ERROR);

					assert(iBinding.memory->isCurrentlyMapped());
				}

				auto* vPointer = static_cast<VSInput*>(vBinding.memory->getMappedPointer());
				auto* iPointer = static_cast<uint16_t*>(iBinding.memory->getMappedPointer());

				memcpy(vPointer, vertices.data(), m_vertexBuffer->getSize());
				memcpy(iPointer, indices.data(), m_indexBuffer->getSize());

				vBinding.memory->unmap();
				iBinding.memory->unmap();
			}
		}

		// camera
		{
			core::vectorSIMDf cameraPosition(-250.0f, 177.0f, 1.69f);
			core::vectorSIMDf cameraTarget(50.0f, 125.0f, -3.0f);
			matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.1, 10000);
			camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 10.f, 1.f);
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
		cb->beginDebugMarker("GizmoApp Frame");
		{
			camera.beginInputProcessing(nextPresentationTimestamp);
			mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, m_logger.get());
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
			viewport.width = WIN_W;
			viewport.height = WIN_H;
		}
		cb->setViewport(0u, 1u, &viewport);
		{
			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};

			const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			const IGPUCommandBuffer::SRenderpassBeginInfo info =
			{
				.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
				.colorClearValues = &clearValue,
				.depthStencilClearValues = nullptr,
				.renderArea = currentRenderArea
			};

			cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
		}

		auto* rawPipeline = pipeline.get();
		cb->bindGraphicsPipeline(rawPipeline);
		cb->bindDescriptorSets(EPBP_GRAPHICS, rawPipeline->getLayout(), 1, 1, &m_gpuDescriptorSet.get());
		cb->pushConstants(pipeline->getLayout(), IShader::ESS_VERTEX, 0, sizeof(PushConstants), &m_pc);

		const asset::SBufferBinding<const IGPUBuffer> bVertices[] = { {.offset = 0, .buffer = m_vertexBuffer} };
		const asset::SBufferBinding<const IGPUBuffer> bIndex = { .offset = 0, .buffer = m_indexBuffer };

		cb->bindVertexBuffers(0, 1, bVertices);
		cb->bindIndexBuffer(bIndex, EIT_16BIT);
		cb->drawIndexed(m_indexBuffer->getSize() / sizeof(uint16_t), 1, 0, 0, 0);
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

			m_window->setCaption("[Nabla Engine] Gizmo App Test Demo");
			m_surface->present(m_currentImageAcquire.imageIndex, rendered);
		}
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
	smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;
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
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
	core::smart_refctd_ptr<video::IGPUBuffer> m_vertexBuffer, m_indexBuffer, m_ubo;
	PushConstants m_pc = {.withGizmo = true};
};

NBL_MAIN_FUNC(GizmoApp)