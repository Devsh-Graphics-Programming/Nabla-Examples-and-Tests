#include <nabla.h>

#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "../common/SimpleWindowedApplication.hpp"
#include "../common/InputSystem.hpp"
#include "../common/Camera.hpp"

#include "gpuRadixSort.h"

#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

#include "app_resources/common.hlsl"

using namespace nbl::hlsl;
using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

// struct Particle defined in shader???
struct Particle
{
	float32_t4 position;
    float32_t4 velocity;

    uint32_t id;
    uint32_t pad[3];
};

struct SGridData
{
    float gridCellSize;
    float gridInvCellSize;
    float pad0[2];

    int32_t4 particleInitMin;
    int32_t4 particleInitMax;
    int32_t4 particleInitSize;

    float32_t4 worldMin;
    float32_t4 worldMax;
    int32_t4 gridSize;
};

struct SMVPParams
{
	float cameraPosition[4];

	float MVP[4*4];
	float M[4*4];
    float V[4*4];
	float P[4*4];
};

struct SParticleRenderParams
{
    float radius;
    float zNear;
    float zFar;
	float pad;
};

struct VertexInfo
{
	float32_t4 position;
	float32_t4 vsSpherePos;

    float radius;
    float pad;

    float32_t4 color;
	float32_t2 uv;
};

class CSwapchainFramebuffersAndDepth final : public nbl::video::CDefaultSwapchainFramebuffers
{
	using scbase_t = CDefaultSwapchainFramebuffers;

public:
	template<typename... Args>
	inline CSwapchainFramebuffersAndDepth(ILogicalDevice* device, const asset::E_FORMAT _desiredDepthFormat, Args&&... args)
		: CDefaultSwapchainFramebuffers(device, std::forward<Args>(args)...)
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

		const auto retval = scbase_t::onCreateSwapchain_impl(qFam);
		m_depthBuffer = nullptr;
		return retval;
	}

	inline smart_refctd_ptr<IGPUFramebuffer> createFramebuffer(IGPUFramebuffer::SCreationParams&& params) override
	{
		params.depthStencilAttachments = &m_depthBuffer.get();
		return m_device->createFramebuffer(std::move(params));
	}

	E_FORMAT m_depthFormat;
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

class FLIPFluidsApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
	using clock_t = std::chrono::steady_clock;

	_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_WIDTH = 1280, WIN_HEIGHT = 720, SC_IMG_COUNT = 3u, FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

	using IGPUDescriptorSetLayoutArray = std::array<core::smart_refctd_ptr<IGPUDescriptorSetLayout>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT>;

public:
	inline FLIPFluidsApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	inline virtual video::IAPIConnection::SFeatures getAPIFeaturesToEnable()
	{
		auto retval = device_base_t::getAPIFeaturesToEnable();
		//retval.synchronizationValidation = true;
		return retval;
	}

	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		if (!m_surface)
		{
			{
				auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
				IWindow::SCreationParams params{
					.callback = core::make_smart_refctd_ptr<ISimpleManagedSurface::ICallback>(),
					.x = 32,
					.y = 32,
					.width = WIN_WIDTH,
					.height = WIN_HEIGHT,
					.flags = IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE,
					.windowCaption = "FLIPFluidsApp"
				};
				params.callback = windowCallback;
				const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
			}

			auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>::create(std::move(surface));
		}

		if (m_surface)
			return { { m_surface->getSurface() } };

		return {};
	}

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

		if (!device_base_t::onAppInitialized(std::move(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		// init grid params
		m_gridData.gridCellSize = 0.25f;
		m_gridData.gridInvCellSize = 1.f / m_gridData.gridCellSize;
		m_gridData.gridSize = int32_t4{128, 128, 128, 0};
		m_gridData.particleInitMin = int32_t4{2, 2, 2, 0};
		m_gridData.particleInitMax = int32_t4{10, 10, 10, 0};
		m_gridData.particleInitSize = m_gridData.particleInitMax - m_gridData.particleInitMin;
		float32_t4 simAreaSize = m_gridData.gridSize;
		simAreaSize *= m_gridData.gridCellSize;
		m_gridData.worldMin = float32_t4(0.f);
		m_gridData.worldMax = simAreaSize;
		numParticles = m_gridData.particleInitSize.x * m_gridData.particleInitSize.y * m_gridData.particleInitSize.z * particlesPerCell;
		
		WorkgroupCount = numParticles / WorkgroupSize;
		if (numParticles % WorkgroupSize > 0)
			WorkgroupCount++;

		{
			float zNear = 0.1f, zFar = 10000.f;
			core::vectorSIMDf cameraPosition(10, 5, 8);
			core::vectorSIMDf cameraTarget(0, 0, 0);
			matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_WIDTH) / WIN_HEIGHT, zNear, zFar);
			camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 1.069f, 0.4f);

			m_pRenderParams.zNear = zNear;
			m_pRenderParams.zFar = zFar;
		}
		m_pRenderParams.radius = m_gridData.gridCellSize * 0.4f;

		// create buffers
		video::IGPUBuffer::SCreationParams params = {};
		params.size = sizeof(SGridData);
		params.usage = IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
		createBuffer(gridDataBuffer, params);
		
		params.size = sizeof(SMVPParams);
		createBuffer(cameraBuffer, params);

		params.size = sizeof(SParticleRenderParams);
		createBuffer(pParamsBuffer, params);

		params.size = numParticles * sizeof(Particle);
		params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
		createBuffer(particleBuffer, params);

		int32_t numGridCells = m_gridData.gridSize.x * m_gridData.gridSize.y * m_gridData.gridSize.z;
		params.size = numGridCells * sizeof(uint32_t2);
		createBuffer(gridParticleIDBuffer, params);

		params.size = numGridCells * sizeof(uint32_t);
		createBuffer(gridCellMaterialBuffer, params);

		params.size = numGridCells * sizeof(float32_t4);
		createBuffer(velocityFieldBuffer, params);
		createBuffer(prevVelocityFieldBuffer, params);

		params.size = numParticles * 6 * sizeof(VertexInfo);
		createBuffer(particleVertexBuffer, params);

		// init render pipeline
		if (!initGraphicsPipeline())
			return logFail("Failed to initialize render pipeline!\n");

		{
			// init particles pipeline
			auto pipeline = createComputePipelineFromShader("app_resources/compute/particlesInit.comp.hlsl");
			m_initParticlePipeline = pipeline.first;
			IGPUDescriptorSetLayoutArray initParticleDsLayouts = pipeline.second;

			// init and write descriptor
			constexpr uint32_t maxDescriptorSets = ICPUPipelineLayout::DESCRIPTOR_SET_COUNT;
			const std::array<IGPUDescriptorSetLayout*, maxDescriptorSets> dscLayoutPtrs = {
				!initParticleDsLayouts[0] ? nullptr : initParticleDsLayouts[0].get(),
				!initParticleDsLayouts[1] ? nullptr : initParticleDsLayouts[1].get(),
				!initParticleDsLayouts[2] ? nullptr : initParticleDsLayouts[2].get(),
				!initParticleDsLayouts[3] ? nullptr : initParticleDsLayouts[3].get()
			};
			m_initParticlePool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()));
			m_initParticlePool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), m_initParticleDs.data());

			{
				IGPUDescriptorSet::SDescriptorInfo inputInfo;
				inputInfo.desc = smart_refctd_ptr(gridDataBuffer);
				inputInfo.info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
				IGPUDescriptorSet::SDescriptorInfo outputInfo;
				outputInfo.desc = smart_refctd_ptr(particleBuffer);
				outputInfo.info.buffer = {.offset = 0, .size = particleBuffer->getSize()};
				IGPUDescriptorSet::SWriteDescriptorSet writes[2] = {
					{.dstSet = m_initParticleDs[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &inputInfo},
					{.dstSet = m_initParticleDs[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &outputInfo}
				};
				m_device->updateDescriptorSets(std::span(writes, 2), {});
			}
		}
		{
			// generate particle vertex pipeline
			auto pipeline = createComputePipelineFromShader("app_resources/compute/genParticleVertices.comp.hlsl");
			m_genParticleVerticesPipeline = pipeline.first;
			IGPUDescriptorSetLayoutArray dsLayouts = pipeline.second;

			constexpr uint32_t maxDescriptorSets = ICPUPipelineLayout::DESCRIPTOR_SET_COUNT;
			const std::array<IGPUDescriptorSetLayout*, maxDescriptorSets> dscLayoutPtrs = {
				!dsLayouts[0] ? nullptr : dsLayouts[0].get(),
				!dsLayouts[1] ? nullptr : dsLayouts[1].get(),
				!dsLayouts[2] ? nullptr : dsLayouts[2].get(),
				!dsLayouts[3] ? nullptr : dsLayouts[3].get()
			};
			m_genVerticesPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()));
			m_genVerticesPool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), m_genVerticesDs.data());

			{
				IGPUDescriptorSet::SDescriptorInfo inputInfos[3];
				inputInfos[0].desc = smart_refctd_ptr(cameraBuffer);
				inputInfos[0].info.buffer = {.offset = 0, .size = cameraBuffer->getSize()};
				inputInfos[1].desc = smart_refctd_ptr(pParamsBuffer);
				inputInfos[1].info.buffer = {.offset = 0, .size = pParamsBuffer->getSize()};
				inputInfos[2].desc = smart_refctd_ptr(particleBuffer);
				inputInfos[2].info.buffer = {.offset = 0, .size = particleBuffer->getSize()};
				IGPUDescriptorSet::SDescriptorInfo outputInfo;
				outputInfo.desc = smart_refctd_ptr(particleVertexBuffer);
				outputInfo.info.buffer = {.offset = 0, .size = particleVertexBuffer->getSize()};
				IGPUDescriptorSet::SWriteDescriptorSet writes[4] = {
					{.dstSet = m_genVerticesDs[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &inputInfos[0]},
					{.dstSet = m_genVerticesDs[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &inputInfos[1]},
					{.dstSet = m_genVerticesDs[1].get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &inputInfos[2]},
					{.dstSet = m_genVerticesDs[2].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &outputInfo},
				};
				m_device->updateDescriptorSets(std::span(writes, 4), {});
			}
		}
		{
			// update fluid cells pipelines
			const uint32_t numkernels = 3;
			std::string kernels[numkernels] = { "updateFluidCells", "updateNeighborFluidCells", "addParticlesToCells" };
			smart_refctd_ptr<IGPUComputePipeline> pipelines[] = { m_updateFluidCellsPipeline, m_updateNeighborFluidCellsPipeline, m_particlesToCellsPipeline };
			smart_refctd_ptr<IDescriptorPool> pools[] = { m_updateCellsPool, m_updateNeighborCellsPool, m_particlesToCellsPool };
			std::array<smart_refctd_ptr<IGPUDescriptorSet>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> sets[] = { m_updateCellsDs, m_updateNeighborCellsDs, m_particlesToCellsDs };

			for (uint32_t i = 0; i < numkernels; i++)
			{
				auto& pipeline = pipelines[i];
				auto& pool = pools[i];
				auto& set = sets[i];

				auto pipelineDs = createComputePipelineFromShader("app_resources/compute/updateFluidCells.comp.hlsl", kernels[i]);
				pipeline = pipelineDs.first;
				IGPUDescriptorSetLayoutArray dsLayouts = pipelineDs.second;

				constexpr uint32_t maxDescriptorSets = ICPUPipelineLayout::DESCRIPTOR_SET_COUNT;
				const std::array<IGPUDescriptorSetLayout*, maxDescriptorSets> dscLayoutPtrs = {
					!dsLayouts[0] ? nullptr : dsLayouts[0].get(),
					!dsLayouts[1] ? nullptr : dsLayouts[1].get(),
					!dsLayouts[2] ? nullptr : dsLayouts[2].get(),
					!dsLayouts[3] ? nullptr : dsLayouts[3].get()
				};
				pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()));
				pool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), set.data());

				{
					IGPUDescriptorSet::SDescriptorInfo inputInfos[3];
					inputInfos[0].desc = smart_refctd_ptr(gridDataBuffer);
					inputInfos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
					inputInfos[1].desc = smart_refctd_ptr(particleBuffer);
					inputInfos[1].info.buffer = {.offset = 0, .size = particleBuffer->getSize()};
					inputInfos[2].desc = smart_refctd_ptr(gridParticleIDBuffer);
					inputInfos[2].info.buffer = {.offset = 0, .size = gridParticleIDBuffer->getSize()};

					IGPUDescriptorSet::SDescriptorInfo outputInfos[3];
					outputInfos[0].desc = smart_refctd_ptr(gridCellMaterialBuffer);
					outputInfos[0].info.buffer = {.offset = 0, .size = gridCellMaterialBuffer->getSize()};
					outputInfos[1].desc = smart_refctd_ptr(velocityFieldBuffer);
					outputInfos[1].info.buffer = {.offset = 0, .size = velocityFieldBuffer->getSize()};
					outputInfos[2].desc = smart_refctd_ptr(prevVelocityFieldBuffer);
					outputInfos[2].info.buffer = {.offset = 0, .size = prevVelocityFieldBuffer->getSize()};
					IGPUDescriptorSet::SWriteDescriptorSet writes[6] = {
						{.dstSet = set[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &inputInfos[0]},
						{.dstSet = set[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &inputInfos[1]},
						{.dstSet = set[1].get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &inputInfos[2]},
						{.dstSet = set[1].get(), .binding = 3, .arrayElement = 0, .count = 1, .info = &outputInfos[0]},
						{.dstSet = set[1].get(), .binding = 4, .arrayElement = 0, .count = 1, .info = &outputInfos[1]},
						{.dstSet = set[1].get(), .binding = 5, .arrayElement = 0, .count = 1, .info = &outputInfos[2]},
					};
					m_device->updateDescriptorSets(std::span(writes, 6), {});
				}
			}
		}
		{
			// extrapolate velocities pipeline
			auto pipeline = createComputePipelineFromShader("app_resources/compute/extrapolateVelocities.comp.hlsl");
			m_extrapolateVelPipeline = pipeline.first;
			IGPUDescriptorSetLayoutArray dsLayouts = pipeline.second;

			constexpr uint32_t maxDescriptorSets = ICPUPipelineLayout::DESCRIPTOR_SET_COUNT;
			const std::array<IGPUDescriptorSetLayout*, maxDescriptorSets> dscLayoutPtrs = {
				!dsLayouts[0] ? nullptr : dsLayouts[0].get(),
				!dsLayouts[1] ? nullptr : dsLayouts[1].get(),
				!dsLayouts[2] ? nullptr : dsLayouts[2].get(),
				!dsLayouts[3] ? nullptr : dsLayouts[3].get()
			};
			m_extrapolateVelPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()));
			m_extrapolateVelPool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), m_extrapolateVelDs.data());

			{
				IGPUDescriptorSet::SDescriptorInfo inputInfos[2];
				inputInfos[0].desc = smart_refctd_ptr(gridDataBuffer);
				inputInfos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
				inputInfos[1].desc = smart_refctd_ptr(particleBuffer);
				inputInfos[1].info.buffer = {.offset = 0, .size = particleBuffer->getSize()};
				IGPUDescriptorSet::SDescriptorInfo outputInfos[2];
				outputInfos[0].desc = smart_refctd_ptr(velocityFieldBuffer);
				outputInfos[0].info.buffer = {.offset = 0, .size = velocityFieldBuffer->getSize()};
				outputInfos[1].desc = smart_refctd_ptr(prevVelocityFieldBuffer);
				outputInfos[1].info.buffer = {.offset = 0, .size = prevVelocityFieldBuffer->getSize()};
				IGPUDescriptorSet::SWriteDescriptorSet writes[4] = {
					{.dstSet = m_extrapolateVelDs[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &inputInfos[0]},
					{.dstSet = m_extrapolateVelDs[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &inputInfos[1]},
					{.dstSet = m_extrapolateVelDs[1].get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &outputInfos[0]},
					{.dstSet = m_extrapolateVelDs[1].get(), .binding = 3, .arrayElement = 0, .count = 1, .info = &outputInfos[1]},
				};
				m_device->updateDescriptorSets(std::span(writes, 4), {});
			}
		}
		{
			// extrapolate velocities pipeline
			auto pipeline = createComputePipelineFromShader("app_resources/compute/advectParticles.comp.hlsl");
			m_advectParticlesPipeline = pipeline.first;
			IGPUDescriptorSetLayoutArray dsLayouts = pipeline.second;

			constexpr uint32_t maxDescriptorSets = ICPUPipelineLayout::DESCRIPTOR_SET_COUNT;
			const std::array<IGPUDescriptorSetLayout*, maxDescriptorSets> dscLayoutPtrs = {
				!dsLayouts[0] ? nullptr : dsLayouts[0].get(),
				!dsLayouts[1] ? nullptr : dsLayouts[1].get(),
				!dsLayouts[2] ? nullptr : dsLayouts[2].get(),
				!dsLayouts[3] ? nullptr : dsLayouts[3].get()
			};
			m_advectParticlesPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()));
			m_advectParticlesPool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), m_advectParticlesDs.data());

			{
				IGPUDescriptorSet::SDescriptorInfo infos[3];
				infos[0].desc = smart_refctd_ptr(gridDataBuffer);
				infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
				infos[1].desc = smart_refctd_ptr(velocityFieldBuffer);
				infos[1].info.buffer = {.offset = 0, .size = velocityFieldBuffer->getSize()};
				infos[2].desc = smart_refctd_ptr(prevVelocityFieldBuffer);
				infos[2].info.buffer = {.offset = 0, .size = prevVelocityFieldBuffer->getSize()};
				IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
					{.dstSet = m_advectParticlesDs[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
					{.dstSet = m_advectParticlesDs[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]},
					{.dstSet = m_advectParticlesDs[1].get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &infos[2]},
				};
				m_device->updateDescriptorSets(std::span(writes, 3), {});
			}
		}

		radixSort.initialize(m_device, m_system, m_assetMgr, m_logger);

		testSort();

		m_winMgr->show(m_window.get());

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
					.semaphore = m_renderSemaphore.get(),
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

		auto* const cmdbuf = m_cmdBufs.data()[resourceIx].get();
		cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cmdbuf->beginDebugMarker("Frame Debug FLIP sim begin");
		{
			camera.beginInputProcessing(nextPresentationTimestamp);
			mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); mouseProcess(events); }, m_logger.get());
			keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, m_logger.get());
			camera.endInputProcessing(nextPresentationTimestamp);
		}

		SMVPParams camData;
		SBufferRange<IGPUBuffer> camDataRange;
		{
			const auto viewMatrix = camera.getViewMatrix();
			const auto projectionMatrix = camera.getProjectionMatrix();
			const auto viewProjectionMatrix = camera.getConcatenatedMatrix();

			core::matrix3x4SIMD modelMatrix;
			modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
			modelMatrix.setRotation(quaternion(0, 0, 0));

			core::matrix3x4SIMD modelViewMatrix = core::concatenateBFollowedByA(viewMatrix, modelMatrix);
			core::matrix4SIMD modelViewProjectionMatrix = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

			auto modelMat = core::concatenateBFollowedByA(core::matrix4SIMD(), modelMatrix);

			const core::vector3df camPos = camera.getPosition().getAsVector3df();

			camPos.getAs4Values(camData.cameraPosition);
			memcpy(camData.MVP, modelViewProjectionMatrix.pointer(), sizeof(camData.MVP));
			memcpy(camData.M, modelMat.pointer(), sizeof(camData.M));
			memcpy(camData.V, viewMatrix.pointer(), sizeof(camData.V));
			memcpy(camData.P, projectionMatrix.pointer(), sizeof(camData.P));
			{
				camDataRange.buffer = cameraBuffer;
				camDataRange.size = cameraBuffer->getSize();

				cmdbuf->updateBuffer(camDataRange, &camData);
			}
		}

		bool bCaptureTestInitParticles = false;
		SBufferRange<IGPUBuffer> gridDataRange;
		SBufferRange<IGPUBuffer> pParamsRange;
		if (m_shouldInitParticles)
		{
			bCaptureTestInitParticles = true;

			{
				gridDataRange.size = gridDataBuffer->getSize();
				gridDataRange.buffer = gridDataBuffer;
			}
			cmdbuf->updateBuffer(gridDataRange, &m_gridData);

			{
				pParamsRange.size = pParamsBuffer->getSize();
				pParamsRange.buffer = pParamsBuffer;
			}
			cmdbuf->updateBuffer(pParamsRange, &m_pRenderParams);

			initializeParticles(cmdbuf);
		}

		/*
		for (uint32_t i = 0; i < m_substepsPerFrame; i++)
		{
			dispatchUpdateFluidCells();			// particle to grid
			dispatchApplyBodyForces(i == 0);	// external forces, e.g. gravity
			dispatchApplyDiffusion();
			dispatchApplyPressure();
			dispatchExtrapolateVelocities();	// grid -> particle vel
			dispatchAdvection();				// update/advect fluid
		}
		*/

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
					.size = cameraBuffer->getSize(),
					.buffer = cameraBuffer,
				};
			}
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = pParamsBuffer->getSize(),
					.buffer = pParamsBuffer,
				};
			}
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = particleBuffer->getSize(),
					.buffer = particleBuffer,
				};
			}
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = particleVertexBuffer->getSize(),
					.buffer = particleVertexBuffer,
				};
			}
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.bufBarriers = {bufferBarriers, bufferBarriersCount}});
		}

		// prepare particle vertices for render
		cmdbuf->bindComputePipeline(m_genParticleVerticesPipeline.get());
		cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_genParticleVerticesPipeline->getLayout(), 0, m_genVerticesDs.size(), &m_genVerticesDs.begin()->get());
		cmdbuf->dispatch(WorkgroupCount, 1, 1);

		// draw particles
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
		cmdbuf->setViewport(0u, 1u, &viewport);

		VkRect2D scissor{
			.offset = { 0, 0 },
			.extent = { m_window->getWidth(), m_window->getHeight() }
		};
		cmdbuf->setScissor(0u, 1u, &scissor);		

		IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		VkRect2D currentRenderArea;
		const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
		const IGPUCommandBuffer::SClearDepthStencilValue depthValue = { .depth = 0.f };
		{
			currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};
	
			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			beginInfo =
			{
				.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
				.colorClearValues = &clearValue,
				.depthStencilClearValues = &depthValue,
				.renderArea = currentRenderArea
			};
		}
		cmdbuf->beginRenderPass(beginInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

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
					.size = particleVertexBuffer->getSize(),
					.buffer = particleVertexBuffer,
				};
			}
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.bufBarriers = {bufferBarriers, bufferBarriersCount}});
		}

		// put into renderFluid();		// TODO: mesh or particles?
		cmdbuf->bindGraphicsPipeline(m_graphicsPipeline.get());
		cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, m_graphicsPipeline->getLayout(), 0, m_renderDs.size(), &m_renderDs.begin()->get());

		cmdbuf->draw(numParticles * 6, 1, 0, 0);

		cmdbuf->endRenderPass();

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
					.size = particleBuffer->getSize(),
					.buffer = particleBuffer,
				};
			}
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.bufBarriers = {bufferBarriers, bufferBarriersCount}});
		}

		cmdbuf->endDebugMarker();
		cmdbuf->end();

		// submit
		const IQueue::SSubmitInfo::SSemaphoreInfo rendered[1] = {{
			.semaphore = m_renderSemaphore.get(),
			.value = ++m_submitIx,
			.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
		}};

		{
			{
				const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = {{
						.cmdbuf = cmdbuf
					}};
				const IQueue::SSubmitInfo::SSemaphoreInfo acquired[1] = {{
						.semaphore = m_currentImageAcquire.semaphore,
						.value = m_currentImageAcquire.acquireCount,
						.stageMask = PIPELINE_STAGE_FLAGS::NONE
					}};
				const IQueue::SSubmitInfo infos[1] = {{
					.waitSemaphores = acquired,
					.commandBuffers = commandBuffers,
					.signalSemaphores = rendered
				}};
				if (bCaptureTestInitParticles)
					queue->startCapture();
				if (queue->submit(infos)!=IQueue::RESULT::SUCCESS)
					m_submitIx--;
				if (bCaptureTestInitParticles)
					queue->endCapture();
			}
		}

		m_surface->present(m_currentImageAcquire.imageIndex, rendered);
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

	void dispatchUpdateFluidCells()
	{
	}
	
	void dispatchApplyBodyForces(bool isFirstSubstep)
	{
	}
	
	void dispatchApplyDiffusion()
	{
	}
	
	void dispatchApplyPressure()
	{
	}
	
	void dispatchExtrapolateVelocities()
	{
	}
			
	void dispatchAdvection()
	{
	}

private:
	std::pair<smart_refctd_ptr<ICPUShader>, smart_refctd_ptr<const CSPIRVIntrospector::CStageIntrospectionData>> compileShaderAndIntrospect(
		const std::string& filePath, CSPIRVIntrospector& introspector, const std::string& entryPoint)
	{
		IAssetLoader::SAssetLoadParams lparams = {};
		lparams.logger = m_logger.get();
		lparams.workingDirectory = "";
		auto bundle = m_assetMgr->getAsset(filePath, lparams);
		if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
		{
			m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
			exit(-1);
		}
		
		const auto assets = bundle.getContents();
		assert(assets.size() == 1);
		smart_refctd_ptr<ICPUShader> shaderSrc = IAsset::castDown<ICPUShader>(assets[0]);

		smart_refctd_ptr<const CSPIRVIntrospector::CStageIntrospectionData> introspection;
		{
			auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(m_system));

			CHLSLCompiler::SOptions options = {};
			options.stage = shaderSrc->getStage();
			if (!(options.stage == IShader::E_SHADER_STAGE::ESS_COMPUTE || options.stage == IShader::E_SHADER_STAGE::ESS_FRAGMENT))
				options.stage = IShader::E_SHADER_STAGE::ESS_VERTEX;
			options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
			options.spirvOptimizer = nullptr;
			options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
			options.preprocessorOptions.sourceIdentifier = shaderSrc->getFilepathHint();
			options.preprocessorOptions.logger = m_logger.get();
			options.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder();

			std::string dxcOptionStr[] = {"-E " + entryPoint};
			if (entryPoint != "main")
			{
				options.dxcOptions = std::span(dxcOptionStr);
			}

			auto spirvUnspecialized = compiler->compileToSPIRV((const char*)shaderSrc->getContent()->getPointer(), options);
			const CSPIRVIntrospector::CStageIntrospectionData::SParams inspectParams = {
				.entryPoint = entryPoint,
				.shader = spirvUnspecialized
			};

			introspection = introspector.introspect(inspectParams);
			introspection->debugPrint(m_logger.get());

			if (!introspection)
			{
				logFail("SPIR-V Introspection failed, probably the required SPIR-V compilation failed first!");
				return std::pair(nullptr, nullptr);
			}

			{
				auto* shaderContent = spirvUnspecialized->getContent();

				system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
				m_physicalDevice->getSystem()->createFile(future, system::path("../shaders/compiled.spv"), system::IFileBase::ECF_WRITE);
				if (auto file = future.acquire(); file && bool(*file))
				{
					system::IFile::success_t success;
					(*file)->write(success, shaderContent->getPointer(), 0, shaderContent->getSize());
					success.getBytesProcessed(true);
				}
			}

			shaderSrc = std::move(spirvUnspecialized);
		}
		return std::pair(shaderSrc, introspection);
	}

	bool createBuffer(smart_refctd_ptr<IGPUBuffer>& buffer, video::IGPUBuffer::SCreationParams& params)
	{
		buffer = m_device->createBuffer(std::move(params));
		if (!buffer)
			return logFail("Failed to create GPU buffer of size %d!\n", params.size);

		video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = buffer->getMemoryReqs();
		reqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();

		auto bufMem = m_device->allocate(reqs, buffer.get());
		if (!bufMem.isValid())
			return logFail("Failed to allocate device memory compatible with gpu buffer!\n");

		return true;
	}

	std::pair<smart_refctd_ptr<video::IGPUComputePipeline>, const IGPUDescriptorSetLayoutArray> createComputePipelineFromShader(
		const std::string& filePath, const std::string& entryPoint = "main")
	{
		CSPIRVIntrospector introspector;
		auto compiledShader = compileShaderAndIntrospect(filePath, introspector, entryPoint);
		auto source = compiledShader.first;
		auto shaderIntrospection = compiledShader.second;

		ICPUShader::SSpecInfo specInfo;
		specInfo.entryPoint = entryPoint;
		specInfo.shader = source.get();

		smart_refctd_ptr<ICPUComputePipeline> cpuPipeline = introspector.createApproximateComputePipelineFromIntrospection(specInfo); ///< what does this do?

		smart_refctd_ptr<nbl::video::IGPUShader> shader = m_device->createShader(source.get());
		if (!shader)
		{
			m_logger->log("Failed to create a GPU Shader, seems the Driver doesn't like the SPIR-V we're feeding it!\n", ILogger::ELL_ERROR);
			exit(-1);
		}

		std::array<std::vector<IGPUDescriptorSetLayout::SBinding>, IGPUPipelineLayout::DESCRIPTOR_SET_COUNT> bindings;
		for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
		{
			const auto& introspectionBindings = shaderIntrospection->getDescriptorSetInfo(i);
			bindings[i].reserve(introspectionBindings.size());

			for (const auto& introspectionBinding : introspectionBindings)
			{
				auto& binding = bindings[i].emplace_back();

				binding.binding = introspectionBinding.binding;
				binding.type = introspectionBinding.type;
				binding.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE;
				binding.stageFlags = IGPUShader::E_SHADER_STAGE::ESS_COMPUTE;
				assert(introspectionBinding.count.countMode == CSPIRVIntrospector::CIntrospectionData::SDescriptorArrayInfo::DESCRIPTOR_COUNT::STATIC);
				binding.count = introspectionBinding.count.count;
			}
		}

		const std::array<core::smart_refctd_ptr<IGPUDescriptorSetLayout>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> dsLayouts = {
			bindings[0].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[0]),
			bindings[1].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[1]),
			bindings[2].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[2]),
			bindings[3].empty() ? nullptr : m_device->createDescriptorSetLayout(bindings[3]),
		};

		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pipelineLayout = m_device->createPipelineLayout(
			{},
			core::smart_refctd_ptr(dsLayouts[0]),
			core::smart_refctd_ptr(dsLayouts[1]),
			core::smart_refctd_ptr(dsLayouts[2]),
			core::smart_refctd_ptr(dsLayouts[3])
		);
		if (!pipelineLayout)
		{
			m_logger->log("Failed to create compute pipeline layout!\n", ILogger::ELL_ERROR);
			exit(-1);
		}

		// init pipeline(s)
		smart_refctd_ptr<video::IGPUComputePipeline> pipeline;
		{
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = pipelineLayout.get();
			params.shader.entryPoint = entryPoint;
			params.shader.shader = shader.get();
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline))
			{
				m_logger->log("Failed to create pipelines (compile & link shaders)!\n", ILogger::ELL_ERROR);
				exit(-1);
			}
		}

		return std::pair(pipeline, dsLayouts);
	}

	bool initGraphicsPipeline()
	{
		m_renderSemaphore = m_device->createSemaphore(m_submitIx);
		if (!m_renderSemaphore)
			return logFail("Failed to create render semaphore!\n");
			
		ISwapchain::SCreationParams swapchainParams{
			.surface = m_surface->getSurface()
		};
		if (!swapchainParams.deduceFormat(m_physicalDevice))
			return logFail("Could not choose a surface format for the swapchain!\n");

		const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
			{
				.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.dstSubpass = 0,
				.memoryBarrier = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,
				.srcAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
				.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_READ_BIT
			}
		},
			// color from ATTACHMENT_OPTIMAL to PRESENT_SRC
			{
				.srcSubpass = 0,
				.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.memoryBarrier = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
			}
		},
		IGPURenderpass::SCreationParams::DependenciesEnd
		};

		auto scResources = std::make_unique<CSwapchainFramebuffersAndDepth>(m_device.get(), EF_D16_UNORM, swapchainParams.surfaceFormat.format, dependencies);
		auto* renderpass = scResources->getRenderpass();
		if (!renderpass)
			return logFail("Failed to create renderpass!\n");

		auto queue = getGraphicsQueue();
		if (!m_surface || !m_surface->init(queue, std::move(scResources), swapchainParams.sharedParams))
			return logFail("Could not create window & surface or initialize surface\n");

		m_maxFramesInFlight = m_surface->getMaxFramesInFlight();
		if (FRAMES_IN_FLIGHT < m_maxFramesInFlight)
		{
			m_logger->log("Lowering frames in flight!\n", ILogger::ELL_WARNING);
			m_maxFramesInFlight = FRAMES_IN_FLIGHT;
		}

		m_cmdPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		for (auto i = 0u; i < m_maxFramesInFlight; i++)
		{
			if (!m_cmdPool)
				return logFail("Couldn't create command pool\n");

			if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
				return logFail("Couldn't create command buffer\n");
		}

		m_winMgr->setWindowSize(m_window.get(), WIN_WIDTH, WIN_HEIGHT);
		m_surface->recreateSwapchain();

		// init shaders and pipeline

		auto compileShader = [&](const std::string& filePath, IShader::E_SHADER_STAGE stage) -> smart_refctd_ptr<IGPUShader>
			{
				IAssetLoader::SAssetLoadParams lparams = {};
				lparams.logger = m_logger.get();
				lparams.workingDirectory = "";
				auto bundle = m_assetMgr->getAsset(filePath, lparams);
				if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
				{
					m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
					exit(-1);
				}
		
				const auto assets = bundle.getContents();
				assert(assets.size() == 1);
				smart_refctd_ptr<ICPUShader> shaderSrc = IAsset::castDown<ICPUShader>(assets[0]);
				shaderSrc->setShaderStage(stage);
				if (!shaderSrc)
					return nullptr;

				return m_device->createShader(shaderSrc.get());
			};
		auto vs = compileShader("app_resources/fluidParticles.vertex.hlsl", IShader::E_SHADER_STAGE::ESS_VERTEX);
		auto fs = compileShader("app_resources/fluidParticles.fragment.hlsl", IShader::E_SHADER_STAGE::ESS_FRAGMENT);

		smart_refctd_ptr<video::IGPUDescriptorSetLayout> descriptorSetLayout1, descriptorSetLayout2;
		{
			// init descriptors
			video::IGPUDescriptorSetLayout::SBinding bindingsSet1[] = {
				{
					.binding = 0u,
					.type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
				},
				{
					.binding = 1u,
					.type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_VERTEX,
					.count = 1u,
				}
			};
			descriptorSetLayout1 = m_device->createDescriptorSetLayout(bindingsSet1);
			if (!descriptorSetLayout1)
				return logFail("Failed to Create Render Descriptor Layout 1");

			const auto maxDescriptorSets = ICPUPipelineLayout::DESCRIPTOR_SET_COUNT;
			const std::array<IGPUDescriptorSetLayout*, maxDescriptorSets> dscLayoutPtrs = {
				nullptr,
				descriptorSetLayout1.get(),
				nullptr,
				nullptr
			};
			m_renderDsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()));
			m_renderDsPool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), m_renderDs.data());
		}

		// write descriptors
		{
			IGPUDescriptorSet::SDescriptorInfo camInfo;
			camInfo.desc = smart_refctd_ptr(cameraBuffer);
			camInfo.info.buffer = {.offset = 0, .size = cameraBuffer->getSize()};
			IGPUDescriptorSet::SDescriptorInfo verticesInfo;
			verticesInfo.desc = smart_refctd_ptr(particleVertexBuffer);
			verticesInfo.info.buffer = {.offset = 0, .size = particleVertexBuffer->getSize()};
			IGPUDescriptorSet::SWriteDescriptorSet writes[2] = {
				{.dstSet = m_renderDs[1].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &camInfo},
				{.dstSet = m_renderDs[1].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &verticesInfo},
			};
			m_device->updateDescriptorSets(std::span(writes, 2), {});
		}

		SBlendParams blendParams = {};
		blendParams.logicOp = ELO_NO_OP;
		blendParams.blendParams[0u].srcColorFactor = asset::EBF_SRC_ALPHA;
		blendParams.blendParams[0u].dstColorFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
		blendParams.blendParams[0u].colorBlendOp = asset::EBO_ADD;
		blendParams.blendParams[0u].srcAlphaFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
		blendParams.blendParams[0u].dstAlphaFactor = asset::EBF_ZERO;
		blendParams.blendParams[0u].alphaBlendOp = asset::EBO_ADD;
		blendParams.blendParams[0u].colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);

		{
			IGPUShader::SSpecInfo specInfo[3] = {
				{.shader = vs.get()},
				{.shader = fs.get()},
			};

			const auto pipelineLayout = m_device->createPipelineLayout({}, nullptr, smart_refctd_ptr(descriptorSetLayout1), smart_refctd_ptr(descriptorSetLayout2), nullptr);

			SRasterizationParams rasterizationParams{};
			rasterizationParams.faceCullingMode = EFCM_NONE;
			rasterizationParams.depthWriteEnable = false;

			IGPUGraphicsPipeline::SCreationParams params[1] = {};
			params[0].layout = pipelineLayout.get();
			params[0].shaders = specInfo;
			params[0].cached = {
				.vertexInput = {
				},
				.primitiveAssembly = {
					.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST,
				},
				.rasterization = rasterizationParams,
				.blend = blendParams,
			};
			params[0].renderpass = renderpass;

			if (!m_device->createGraphicsPipelines(nullptr, params, &m_graphicsPipeline))
				return logFail("Graphics pipeline creation failed");
		}

		return true;
	}

	void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
	{
		for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
		{
			auto ev = *eventIt;

			// do nothing
		}
	}


	// in-loop functions
	void initializeParticles(IGPUCommandBuffer* cmdbuf)
	{
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
					.size = gridDataBuffer->getSize(),
					.buffer = gridDataBuffer,
				};
			}
			{
				auto& bufferBarrier = bufferBarriers[bufferBarriersCount++];
				bufferBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				bufferBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				bufferBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
				bufferBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;
				bufferBarrier.range =
				{
					.offset = 0u,
					.size = particleBuffer->getSize(),
					.buffer = particleBuffer,
				};
			}
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.bufBarriers = {bufferBarriers, bufferBarriersCount}});
		}
		
		cmdbuf->bindComputePipeline(m_initParticlePipeline.get());
		cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_initParticlePipeline->getLayout(), 0, m_initParticleDs.size(), &m_initParticleDs.begin()->get());
		cmdbuf->dispatch(WorkgroupCount, 1, 1);

		m_shouldInitParticles = false;
	}

	bool testSort()
	{
		const uint32_t numTestElements = 64;
		smart_refctd_ptr<IGPUBuffer> testbuf;
		IDeviceMemoryAllocator::SAllocation testbufAlloc;

		void* testbufMem;

		IGPUBuffer::SCreationParams params;
		params.size = 2 * numTestElements * sizeof(uint32_t);
		params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
		{
			testbuf = m_device->createBuffer(std::move(params));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = testbuf->getMemoryReqs();
			reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

			testbufAlloc = m_device->allocate(reqs, testbuf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

			testbufMem = testbufAlloc.memory->map({0ull, testbufAlloc.memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ);
		}

		// generate random data
		uint32_t bufferData[2 * numTestElements];
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937 g(seed);

		for (int i = 0; i < 2 * numTestElements; i++)
		{
			bufferData[i] = g() % 128;
		}
		memcpy(testbufMem, bufferData, 2 * numTestElements * sizeof(uint32_t));

		std::string outBuffer;
		for (auto i = 0; i < numTestElements; i++) {
			outBuffer.append("{");
			outBuffer.append(std::to_string(bufferData[2 * i]));
			outBuffer.append(",");
			outBuffer.append(std::to_string(bufferData[2 * i + 1]));
			outBuffer.append("} ");
		}
		outBuffer.append("\n");
		outBuffer.append("Count: ");
		outBuffer.append(std::to_string(numTestElements));
		outBuffer.append("\n");
		m_logger->log("Your input array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdBuf;
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(getComputeQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdBuf))
				return logFail("Failed to create Command Buffers!\n");
		}

		// run sort
		constexpr uint64_t started_value = 0;
		uint64_t timeline = started_value;
		smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore(started_value);

		cmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cmdBuf->beginDebugMarker("Radix sort Dispatch", core::vectorSIMDf(0, 1, 0, 1));
		
		radixSort.sort(cmdBuf, testbuf, numTestElements);

		cmdBuf->endDebugMarker();
		cmdBuf->end();

		// block till end
		{
			auto queue = getComputeQueue();

			IQueue::SSubmitInfo submit_infos[1];
			IQueue::SSubmitInfo::SCommandBufferInfo cmdBufs[] = {
				{
					.cmdbuf = cmdBuf.get()
				}
			};
			submit_infos[0].commandBuffers = cmdBufs;
			IQueue::SSubmitInfo::SSemaphoreInfo signals[] = {
				{
					.semaphore = progress.get(),
					.value = ++timeline,
					.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
				}
			};
			submit_infos[0].signalSemaphores = signals;

			queue->startCapture();
			queue->submit(submit_infos);
			queue->endCapture();
		}

		const ISemaphore::SWaitInfo wait_infos[] = { {
				.semaphore = progress.get(),
				.value = timeline
			} };
		m_device->blockForSemaphores(wait_infos);

		// check values
		const ILogicalDevice::MappedMemoryRange range = ILogicalDevice::MappedMemoryRange(testbufAlloc.memory.get(), 0ull, testbufAlloc.memory->getAllocationSize());
		if (!testbufAlloc.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &range);


		const uint32_t* mappedBufData = (const uint32_t*)testbufAlloc.memory->getMappedPointer();
		
		outBuffer.clear();
		for (auto i = 0; i < numTestElements; i++) {
			outBuffer.append("{");
			outBuffer.append(std::to_string(mappedBufData[2 * i]));
			outBuffer.append(",");
			outBuffer.append(std::to_string(mappedBufData[2 * i + 1]));
			outBuffer.append("} ");
		}
		outBuffer.append("\n");
		outBuffer.append("Count: ");
		outBuffer.append(std::to_string(numTestElements));
		outBuffer.append("\n");
		m_logger->log("Your output array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

		return true;
	}


	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>> m_surface;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_graphicsPipeline;
	smart_refctd_ptr<ISemaphore> m_renderSemaphore;
	smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
	uint64_t m_realFrameIx : 59 = 0;
	uint64_t m_submitIx : 59 = 0;
	uint64_t m_maxFramesInFlight : 5;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	smart_refctd_ptr<video::IDescriptorPool> m_renderDsPool;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> m_renderDs;

	// simulation compute shaders
	smart_refctd_ptr<IGPUComputePipeline> m_initParticlePipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_updateFluidCellsPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_updateNeighborFluidCellsPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_particlesToCellsPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_applyBodyForcesPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_diffusionPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_applyPressurePipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_extrapolateVelPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_advectParticlesPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_densityProjectPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_genParticleVerticesPipeline;

	smart_refctd_ptr<video::IDescriptorPool> m_initParticlePool;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> m_initParticleDs;
	smart_refctd_ptr<video::IDescriptorPool> m_updateCellsPool;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> m_updateCellsDs;
	smart_refctd_ptr<video::IDescriptorPool> m_updateNeighborCellsPool;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> m_updateNeighborCellsDs;
	smart_refctd_ptr<video::IDescriptorPool> m_particlesToCellsPool;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> m_particlesToCellsDs;
	smart_refctd_ptr<video::IDescriptorPool> m_extrapolateVelPool;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> m_extrapolateVelDs;
	smart_refctd_ptr<video::IDescriptorPool> m_advectParticlesPool;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> m_advectParticlesDs;
	smart_refctd_ptr<video::IDescriptorPool> m_genVerticesPool;
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> m_genVerticesDs;

	// input system
	smart_refctd_ptr<InputSystem> m_inputSystem;
	InputSystem::ChannelReader<IMouseEventChannel> mouse;
	InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	Camera camera = Camera(core::vectorSIMDf(0,0,0), core::vectorSIMDf(0,0,0), core::matrix4SIMD());
	video::CDumbPresentationOracle oracle;

	GPURadixSort radixSort;

	bool m_shouldInitParticles = true;

	// simulation constants
	size_t WorkgroupCount;
	uint32_t m_substepsPerFrame = 1;
	SGridData m_gridData;
	SParticleRenderParams m_pRenderParams;
	uint32_t particlesPerCell = 8;
	uint32_t numParticles;

	// buffers
	smart_refctd_ptr<IGPUBuffer> cameraBuffer;

	smart_refctd_ptr<IGPUBuffer> particleBuffer;		// Particle
	smart_refctd_ptr<IGPUBuffer> pParamsBuffer;			// SParticleRenderParams
	smart_refctd_ptr<IGPUBuffer> particleVertexBuffer;	// VertexInfo * 6 vertices

	smart_refctd_ptr<IGPUBuffer> gridDataBuffer;		// SGridData
	smart_refctd_ptr<IGPUBuffer> gridParticleIDBuffer;	// uint2
	smart_refctd_ptr<IGPUBuffer> gridCellMaterialBuffer;	// uint, fluid or solid
	smart_refctd_ptr<IGPUBuffer> velocityFieldBuffer;	// float4
	smart_refctd_ptr<IGPUBuffer> prevVelocityFieldBuffer;// float4
	smart_refctd_ptr<IGPUBuffer> gridDiffusionBuffer;	// float3
	smart_refctd_ptr<IGPUBuffer> gridAxisTypeBuffer;	// uint3
	smart_refctd_ptr<IGPUBuffer> divergenceBuffer;		// float
	smart_refctd_ptr<IGPUBuffer> pressureBuffer;		// float
	smart_refctd_ptr<IGPUBuffer> gridWeightBuffer;		// float
	smart_refctd_ptr<IGPUBuffer> gridUintWeightBuffer;	// uint
	smart_refctd_ptr<IGPUBuffer> gridDensityPressureBuffer;// float
	smart_refctd_ptr<IGPUBuffer> positionModifyBuffer;	// float3
	smart_refctd_ptr<IGPUBuffer> zeroBuffer;			// float

};

NBL_MAIN_FUNC(FLIPFluidsApp)