// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "common.hpp"
#include "app_resources/common.hlsl"

#include "nbl/this_example/builtin/build/spirv/keys.hpp" // this is for the shader keys embedded in this application for asset manager to find them
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/examples/common/BuiltinResourcesApplication.hpp"
#include <nbl/builtin/hlsl/math/linalg/transform.hlsl> // math
#include <nbl/builtin/hlsl/math/quaternions.hlsl> // math
#include <nbl/builtin/hlsl/math/thin_lens_projection.hlsl> // for perspective matrix utils

// Build command:
// cmake -S C:/work/Nabla -B C:/work/Nabla/build -D_NBL_JOBS_AMOUNT_=2 -DNBL_MEMORY_CONSUMPTION_CHECK_SKIP=ON -DNBL_UPDATE_GIT_SUBMODULE=OFF -DNBL_BUILD_EXAMPLES=ON -DNBL_BUILD_IMGUI=ON

// just bored
namespace nbl
{
	template <typename T> struct nbl_remove_reference { using type = T; };
	template <typename T> struct nbl_remove_reference<T&> { using type = T; };
	template <typename T> struct nbl_remove_reference<T&&> { using type = T; };

	template <typename T>
	using nbl_remove_reference_t = nbl_remove_reference<T>::type;

	template <typename T>
	nbl_remove_reference_t<T>&& nbl_move(T&& value)
	{
		return static_cast<nbl_remove_reference_t<T>&&>(value);
	}
}

class PhotonCausticsApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
	using device_base_t = SimpleWindowedApplication;
	using asset_base_t = BuiltinResourcesApplication;

	constexpr static inline uint32_t WIN_W = 1920;
	constexpr static inline uint32_t WIN_H = 1080;
	constexpr static inline uint32_t MaxFramesInFlight = 3;
	constexpr static inline uint8_t  MaxUITextureCount = 1;
	constexpr static inline uint32_t MaxPhotonInScene = 8 * 1024;;
	constexpr static inline float EmitterHalfExtent = 0.5f;
	constexpr static inline float SphereRadius = 0.75f;
	constexpr static inline float SceneHalfExtent = 2.f;
	static inline const core::vectorSIMDf InitialCamPos = core::vectorSIMDf(0, 3, 8);
	static inline const core::vectorSIMDf InitialCamTarget = core::vectorSIMDf(0, 1, 0);

public:
	inline PhotonCausticsApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		:IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD)
	{
	}

	inline SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	{
		auto retval = device_base_t::getRequiredDeviceFeatures();
		retval.rayTracingPipeline = true;
		retval.accelerationStructure = true;
		// <nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl> pulls in the ray-query
		// intrinsic declarations, which carry [[vk::ext_capability(RayQueryKHR)]],
		// and DXC emits that capability into the module regardless. 
		// So this features needs to be enabled nonetheless when using TraceRays.
		retval.rayQuery = true;
		return retval;
	}

	inline SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
	{
		auto retval = device_base_t::getPreferredDeviceFeatures();
		retval.accelerationStructureHostCommands = true; // we can build accl stuff on CPU, instead of GPU lol
		return retval;
	}

	inline core::vector<queue_req_t> getQueueRequirements() const override
	{
		// share one queue between the asset converter's AS builds and rendering
		auto reqs = device_base_t::getQueueRequirements();
		reqs.front().requiredFlags |= IQueue::FAMILY_FLAGS::COMPUTE_BIT;
		return reqs;
	}

	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		if (!m_surface)
		{
			{
				auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
				IWindow::SCreationParams params = {};
				params.callback = core::make_smart_refctd_ptr<ISimpleManagedSurface::ICallback>(); // shouldnt we able to pass in the callback here itself? check why later.
				params.width = WIN_W;
				params.height = WIN_H;
				params.x = 32;
				params.y = 32;
				params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
				params.windowCaption = "PhotonCaustics";
				params.callback = windowCallback;
				const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(nbl_move(params));
			}

			// create the Vulkan surface and VkSurfaceKHR managed handles
			auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>::create(std::move(surface));
		}

		// weird
		if (m_surface)
			return { {m_surface->getSurface()} };

		return {};
	}

	//-----------------------------------------------------------------------------

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

		// call some super functions
		if (!device_base_t::onAppInitialized(nbl_move(system)))
			return false;
		if (!asset_base_t::onAppInitialized(nbl_move(system)))
			return false;

		//-------------------------------------
		// Load Shaders 
		//-------------------------------------
		// util lambda, turn this into nbl core function? add it to IAssetManager?
		auto loadPreCompiledShader = [&]<core::StringLiteral ShaderKey>() -> smart_refctd_ptr<IShader>
		{
			IAssetLoader::SAssetLoadParams loadParams = {};
			loadParams.logger = m_logger.get();
			loadParams.workingDirectory = "app_resources";

			auto key = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_device.get());
			auto assetBundle = m_assetMgr->getAsset(key.data(), loadParams);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
				return nullptr;

			auto shader = IAsset::castDown<IShader>(assets[0]);
			if (!shader)
			{
				m_logger->log("Failed to load a precompiled shader.", ILogger::ELL_ERROR);
				return nullptr;
			}
			return shader;
		};

		// load some custom shaders
		const auto fragmentShader = loadPreCompiledShader.operator() < "present_frag" > ();
		const auto raygenShader = loadPreCompiledShader.operator() < "raytrace_rgen" > ();
		const auto closestHitShader = loadPreCompiledShader.operator() < "raytrace_rchit" > ();
		const auto missShader = loadPreCompiledShader.operator() < "raytrace_rmiss" > ();
		const auto photonShader = loadPreCompiledShader.operator() < "photon_rgen" > ();
		if (!fragmentShader)
		{
			logFail("Could not load present fragment shader");
			return false;
		};
		if (!raygenShader)
		{
			logFail("Could not load raygen shader");
			return false;
		};
		if (!closestHitShader)
		{
			logFail("Could not load closest hit shader");
			return false;
		};
		if (!missShader)
		{
			logFail("Could not load miss shader");
			return false;
		};
		if (!photonShader)
		{
			logFail("Could not load photon shader");
			return false;
		};

		//-------------------------------------
		// Create some default GPU resources 
		//-------------------------------------
		m_semaphore = m_device->createSemaphore(m_realFrameIx);
		if (!m_semaphore)
			return logFail("Failed to Create a Semaphore!");

		auto gfxQueue = getGraphicsQueue();

		// Swapchain presentation render pass
		nbl::video::IGPURenderpass* renderpass;
		{
			ISwapchain::SCreationParams swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");

			const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] =
			{
				// for Blit
				{
					.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.dstSubpass = 0,
					.memoryBarrier = {
						.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
						.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
						.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
				},
				// for COLOR_ATTACHMENT_OUTPUT to PRESENT_SRC
				{
					.srcSubpass = 0,
					.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.memoryBarrier = {
						.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
				},
				IGPURenderpass::SCreationParams::DependenciesEnd
			};

			auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(), swapchainParams.surfaceFormat.format, dependencies);
			renderpass = scResources->getRenderpass();
			if (!renderpass)
				return logFail("Failed to create Renderpass!");

			if (!m_surface || !m_surface->init(gfxQueue, nbl_move(scResources), swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");
		}

		auto pool = m_device->createCommandPool(gfxQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		for (auto i = 0u; i < MaxFramesInFlight; i++)
		{
			if (!pool) return logFail("Couldn't create Command Pool!");
			if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
				return logFail("Couldn't create Command Buffer!");
		}

		m_converter = CAssetConverter::create({ .device = m_device.get(), .optimizer = {} });

		m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
		m_surface->recreateSwapchain();

		//-------------------------------------
		// Create some App stuff 
		//-------------------------------------
		auto createStorageImage = [&](E_FORMAT format, smart_refctd_ptr<IGPUImage>& image, smart_refctd_ptr<IGPUImageView>& view) -> bool
			{
				image = m_device->createImage({
					{
						.type = IGPUImage::ET_2D,
						.samples = ICPUImage::ESCF_1_BIT,
						.format = format,
						.extent = { WIN_W, WIN_H, 1 },
						.mipLevels = 1,
						.arrayLayers = 1,
						.flags = IImage::ECF_NONE,
						.usage = bitflag(IImage::EUF_STORAGE_BIT) | IImage::EUF_SAMPLED_BIT | IImage::EUF_TRANSFER_DST_BIT
					}
					});
				if (!image || !m_device->allocate(image->getMemoryReqs(), { image.get() }).isValid())
					return false;

				view = m_device->createImageView({
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = IGPUImage::EUF_STORAGE_BIT | IGPUImage::EUF_SAMPLED_BIT,
					.image = image,
					.viewType = IGPUImageView::E_TYPE::ET_2D,
					.format = format
					});
				return bool(view);
			};

		// The average (sum / count) for this frame, needs to be tonemapped for swapchain
		if (!createStorageImage(EF_R16G16B16A16_SFLOAT, m_hdrImage, m_hdrImageView))
			return logFail("Could not create HDR image");

		// The running sum of all samples ever taken, plus a sample count in .w
		if (!createStorageImage(EF_R32G32B32A32_SFLOAT, m_accumImage, m_accumImageView))
			return logFail("Could not create accumulation image");

		//-------------------------------------
		// Create Scene geometry
		//-------------------------------------
		if (!createScene())
			return logFail("Could not build the scene");

		//-------------------------------------
		// Photon mapping resources
		//-------------------------------------
		if (!createPhotonResources())
			return logFail("Could not create photon mapping resources");

		//-------------------------------------
		// Create RT stuff
		//-------------------------------------
		if (!createRayTracingPipeline(raygenShader, closestHitShader, missShader, nbl_move(m_rayTracingPipeline), m_shaderBindingTable, m_rayTracingStackSize))
			return logFail("Could not create ray tracing pipeline");

		if (!createRayTracingPipeline(photonShader, closestHitShader, missShader, nbl_move(m_photonRayTracingPipeline), m_photonSBT, m_photonRTStackSize))
			return logFail("Could not create Photon ray tracing pipeline");

		//-------------------------------------
		// ImGui
		//-------------------------------------
		if (!createImGui(renderpass))
			return logFail("Could not create ImGui UI");

		//-------------------------------------
		// Create Presentation stuff, must be last thing
		//-------------------------------------
		{
			// Create binding sets and the PP

			ISampler::SParams samplerParams = { .AnisotropicFilter = 0 };
			auto defaultSampler = m_device->createSampler(samplerParams);

			//-------------------------------------

			const IGPUDescriptorSetLayout::SBinding bindings[] = {
				{
					.binding = 0u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
					.immutableSamplers = &defaultSampler
				}
			};
			auto presentDsLayout = m_device->createDescriptorSetLayout(bindings);

			const video::IGPUDescriptorSetLayout* const layouts[] = { presentDsLayout.get() };
			const uint32_t setCounts[] = { 1u };
			m_presentDsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
			m_presentDs = m_presentDsPool->createDescriptorSet(presentDsLayout);

			//-------------------------------------

			const SPushConstantRange presentPcRange = {
				.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
				.offset = 0u,
				.size = sizeof(SPresentPushConstants)
			};
			auto presentLayout = m_device->createPipelineLayout({ &presentPcRange, 1 }, core::smart_refctd_ptr(presentDsLayout), nullptr, nullptr, nullptr);

			const IGPUPipelineBase::SShaderSpecInfo fragSpec = { .shader = fragmentShader.get(),
				.entryPoint = "main"
			};

			ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
			if (!fsTriProtoPPln)
				return logFail("Failed to create Full Screen Triangle protopipeline!");

			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			m_presentPipeline = fsTriProtoPPln.createPipeline(fragSpec, presentLayout.get(), scRes->getRenderpass());
			if (!m_presentPipeline)
				return logFail("Could not create Graphics Pipeline!");

			//-------------------------------------

			IGPUDescriptorSet::SDescriptorInfo info = {};
			info.desc = m_hdrImageView;
			info.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
				{.dstSet = m_presentDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &info }
			};
			m_device->updateDescriptorSets(std::span(writes), {});
		}

		//-------------------------------------
		// Camera Init position and matrices
		//-------------------------------------
		{
			hlsl::float32_t4x4 proj = hlsl::math::thin_lens::rhPerspectiveFovMatrix(
				core::radians(m_cameraSetting.fov),
				float(WIN_W) / float(WIN_H),
				m_cameraSetting.zNear, m_cameraSetting.zFar);
			// Initial position of the camera
			m_camera = Camera(InitialCamPos, InitialCamTarget, proj);
			// cache this to reset it
			m_InitialMVP = m_camera.getConcatenatedMatrix();
			m_camera.mapKeysToWASD();
		}

		// show the window (and start presentation time recording?)
		m_winMgr->show(m_window.get());
		m_oracle.reportBeginFrameRecord();
		return true;
	}

	bool updateGUIDescriptorSet()
	{
		static std::array<IGPUDescriptorSet::SDescriptorInfo, MaxUITextureCount> descriptorInfo;
		static IGPUDescriptorSet::SWriteDescriptorSet writes[MaxUITextureCount];

		descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
		descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].desc = smart_refctd_ptr<IGPUImageView>(m_ui.manager->getFontAtlasView());

		for (uint32_t i = 0; i < descriptorInfo.size(); ++i)
		{
			writes[i].dstSet = m_ui.descriptorSet.get();
			writes[i].binding = 0u;
			writes[i].arrayElement = i;
			writes[i].count = 1u;
		}
		writes[nbl::ext::imgui::UI::FontAtlasTexId].info = descriptorInfo.data() + nbl::ext::imgui::UI::FontAtlasTexId;

		return m_device->updateDescriptorSets(writes, {});
	}

	static inline IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarrier(
		IGPUImage* image,
		PIPELINE_STAGE_FLAGS srcStage, ACCESS_FLAGS srcAccess, IImage::LAYOUT oldLayout,
		PIPELINE_STAGE_FLAGS dstStage, ACCESS_FLAGS dstAccess, IImage::LAYOUT newLayout)
	{
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t b = {};
		b.barrier = { .dep = {.srcStageMask = srcStage, .srcAccessMask = srcAccess, .dstStageMask = dstStage, .dstAccessMask = dstAccess } };
		b.image = image;
		b.subresourceRange = { .aspectMask = IImage::EAF_COLOR_BIT, .baseMipLevel = 0u, .levelCount = 1u, .baseArrayLayer = 0u, .layerCount = 1u };
		b.oldLayout = oldLayout;
		b.newLayout = newLayout;
		return b;
	}

	static inline void bufferBarrier(IGPUCommandBuffer* cmdbuf, IGPUBuffer* buffer,
		PIPELINE_STAGE_FLAGS srcStage, ACCESS_FLAGS srcAccess,
		PIPELINE_STAGE_FLAGS dstStage, ACCESS_FLAGS dstAccess)
	{
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t b = {};
		b.barrier.dep = { .srcStageMask = srcStage, .srcAccessMask = srcAccess,
						  .dstStageMask = dstStage, .dstAccessMask = dstAccess };
		b.range = { .offset = 0, .size = buffer->getSize(), .buffer = smart_refctd_ptr<IGPUBuffer>(buffer) };
		cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .bufBarriers = {&b, 1} });
	}

	inline void workLoopBody() override
	{
		const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
		// wait if we are way too ahead
		if (m_realFrameIx >= framesInFlight)
		{
			const ISemaphore::SWaitInfo cbDonePending[] = { {.semaphore = m_semaphore.get(), .value = m_realFrameIx + 1 - framesInFlight } };
			if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
				return;
		}
		const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

		m_api->startCapture();
		update();

		auto queue = getGraphicsQueue();
		auto cmdbuf = m_cmdBufs[resourceIx].get();
		if (!keepRunning())
			return;

		cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cmdbuf->beginDebugMarker("PhotonCaustics Frame");

		const auto& viewProjectionMatrix = m_camera.getConcatenatedMatrix();
		// reset accumulation when we move the camera
		if (m_cachedMVP != viewProjectionMatrix)
		{
			m_accumulatedFrames = 0;
			m_cachedMVP = viewProjectionMatrix;
		}
		const hlsl::float32_t4x4 invMVP = hlsl::inverse(viewProjectionMatrix);
		const bool restarting = (m_accumulatedFrames == 0);


		// Build photon once map per static light
		{
			const uint32_t emittedPhotons = uint32_t(m_photonsEmitCount);
			if (m_enablePhotonCaustics && !m_photonMapBuilt && m_lightCount > 0)
			{
				// photon map bounds and the slot counter live in the head of the photon buffer
				{
					SPhotonMapHeader header = {};
					header.cellCountsAddr = m_cellCountsBuffer->getDeviceAddress();
					header.cellPhotonsAddr = m_cellPhotonsBuffer->getDeviceAddress();
					header.photonMapCenter = m_photonMapCenter;
					header.photonMapRadius = m_photonMapRadius;
					header.gridMin = m_gridMin;
					header.gridInvCellSize = m_gridInvCellSize;
					header.photonCounter = 0u;
					cmdbuf->updateBuffer({ .offset = 0, .size = sizeof(SPhotonMapHeader), .buffer = m_photonBuffer }, &header);
					cmdbuf->fillBuffer({ .offset = 0, .size = sizeof(uint32_t) * GRID_CELLS, .buffer = m_cellCountsBuffer }, 0u);
					bufferBarrier(cmdbuf, m_cellCountsBuffer.get(),
						PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS, ACCESS_FLAGS::TRANSFER_WRITE_BIT,
						PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, ACCESS_FLAGS::SHADER_READ_BITS | ACCESS_FLAGS::SHADER_WRITE_BITS);
					bufferBarrier(cmdbuf, m_photonBuffer.get(),
						PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS, ACCESS_FLAGS::TRANSFER_WRITE_BIT,
						PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, ACCESS_FLAGS::SHADER_READ_BITS | ACCESS_FLAGS::SHADER_WRITE_BITS);
				}

				// trace rays from lights
				{
					SPushConstants pc = {};
					pc.geomInfoBuffer = m_geomInfoBuffer->getDeviceAddress();
					pc.lightBuffer = m_lightBuffer->getDeviceAddress();
					pc.photonBuffer = m_photonBuffer->getDeviceAddress();
					pc.lightCount = m_lightCount;
					pc.photonCount = emittedPhotons;
					pc.photonScale = 1.0f / float(emittedPhotons);
					pc.debugFlags = m_disableSpecularConcentration ? DEBUG_PHOTON_MAP_DISABLE_SPECULAR_CONCENTRATION : 0u;
					pc.accumulatedFrames = 0;

					cmdbuf->bindRayTracingPipeline(m_photonRayTracingPipeline.get());
					cmdbuf->setRayTracingPipelineStackSize(m_photonRTStackSize);
					cmdbuf->pushConstants(m_photonRayTracingPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING, 0, sizeof(SPushConstants), &pc);
					cmdbuf->bindDescriptorSets(EPBP_RAY_TRACING, m_photonRayTracingPipeline->getLayout(), 0, 1, &m_rayTracingDs.get());
					cmdbuf->traceRays(m_photonSBT, emittedPhotons, 1, 1);
				}

				bufferBarrier(cmdbuf, m_photonBuffer.get(),
					PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, ACCESS_FLAGS::SHADER_WRITE_BITS,
					PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, ACCESS_FLAGS::SHADER_READ_BITS);
				bufferBarrier(cmdbuf, m_cellCountsBuffer.get(),
					PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, ACCESS_FLAGS::SHADER_WRITE_BITS,
					PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, ACCESS_FLAGS::SHADER_READ_BITS);
				bufferBarrier(cmdbuf, m_cellPhotonsBuffer.get(),
					PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, ACCESS_FLAGS::SHADER_WRITE_BITS,
					PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, ACCESS_FLAGS::SHADER_READ_BITS);

				m_photonMapBuilt = true;
				m_needPhotonCountReadback = true;
			}
		}

		// PathTracing the entire scene
		{
			{
				// Make both HDR + Accumm images write ready
				const auto srcStage = restarting ? PIPELINE_STAGE_FLAGS::NONE : PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT;
				const auto srcAccess = restarting ? ACCESS_FLAGS::NONE : ACCESS_FLAGS::SHADER_READ_BITS;
				const auto oldHdr = restarting ? IImage::LAYOUT::UNDEFINED : IImage::LAYOUT::READ_ONLY_OPTIMAL;

				IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barriers[] = {
					imageBarrier(m_hdrImage.get(), srcStage, srcAccess, oldHdr,
						PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, ACCESS_FLAGS::SHADER_WRITE_BITS, IImage::LAYOUT::GENERAL),
				};
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = barriers });
			}

			// clear accumulation image if we moved the camera
			if (restarting)
			{
				{
					const auto oldAcc = restarting ? IImage::LAYOUT::UNDEFINED : IImage::LAYOUT::GENERAL;
					IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t acc_b0[] = {
						imageBarrier(m_accumImage.get(), PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, ACCESS_FLAGS::SHADER_WRITE_BITS, oldAcc,
							restarting ? PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS : PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT,
							restarting ? ACCESS_FLAGS::TRANSFER_WRITE_BIT : (ACCESS_FLAGS::SHADER_READ_BITS | ACCESS_FLAGS::SHADER_WRITE_BITS),
							IImage::LAYOUT::GENERAL),
					};
					cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = acc_b0 });
				}
				const IGPUCommandBuffer::SClearColorValue zero = { .float32 = {0.f,0.f,0.f,0.f} };
				const IGPUImage::SSubresourceRange fullRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = 1u,
					.baseArrayLayer = 0u,
					.layerCount = 1u };
				cmdbuf->clearColorImage(m_accumImage.get(), IImage::LAYOUT::GENERAL, &zero, 1u, &fullRange);
				{
					auto b = imageBarrier(m_accumImage.get(),
						PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS, ACCESS_FLAGS::TRANSFER_WRITE_BIT, IImage::LAYOUT::GENERAL,
						PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, ACCESS_FLAGS::SHADER_READ_BITS | ACCESS_FLAGS::SHADER_WRITE_BITS, IImage::LAYOUT::GENERAL);
					cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&b, 1} });
				}
			}

			SPushConstants pc = {};
			pc.invMVP = invMVP;
			const auto camPos = m_camera.getPosition().getAsVector3df();
			pc.camPos = float32_t3(camPos.X, camPos.Y, camPos.Z);
			pc.accumulatedFrames = m_accumulatedFrames;
			pc.geomInfoBuffer = m_geomInfoBuffer->getDeviceAddress();
			pc.lightBuffer = m_lightBuffer->getDeviceAddress();
			pc.photonBuffer = m_photonBuffer->getDeviceAddress();
			pc.lightCount = m_lightCount;
			pc.photonCount = (m_enablePhotonCaustics && m_photonMapBuilt) ? uint32_t(m_photonsEmitCount) : 0u;
			pc.gatherRadius = m_gatherRadius;
			pc.debugFlags = m_debugPhotonView ? DEBUG_PHOTONS_BIT : 0u;

			cmdbuf->bindRayTracingPipeline(m_rayTracingPipeline.get());
			cmdbuf->setRayTracingPipelineStackSize(m_rayTracingStackSize);
			cmdbuf->pushConstants(m_rayTracingPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING, 0, sizeof(SPushConstants), &pc);
			cmdbuf->bindDescriptorSets(EPBP_RAY_TRACING, m_rayTracingPipeline->getLayout(), 0, 1, &m_rayTracingDs.get());
			cmdbuf->traceRays(m_shaderBindingTable, WIN_W, WIN_H, 1);
		}

		// Tonemapping pass render HDR image to swapchain 
		{
			{
				// Make HDR image read ready for Tonemapping shader
				auto b = imageBarrier(m_hdrImage.get(),
					PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT, // Src
					ACCESS_FLAGS::SHADER_WRITE_BITS, // Src
					IImage::LAYOUT::GENERAL, // Src
					PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT, // Dst
					ACCESS_FLAGS::SHADER_READ_BITS, // Dst
					IImage::LAYOUT::READ_ONLY_OPTIMAL); // Dst
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&b, 1} });
			}

			asset::SViewport viewport;
			viewport.minDepth = 1.f;
			viewport.maxDepth = 0.f;
			viewport.x = 0u; viewport.y = 0u;
			viewport.width = WIN_W; viewport.height = WIN_H;
			cmdbuf->setViewport(0u, 1u, &viewport);

			VkRect2D scissors[] = { {.offset = {0,0}, .extent = {WIN_W, WIN_H} } };
			cmdbuf->setScissor(scissors);

			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			const IGPUCommandBuffer::SClearColorValue clearColor = { .float32 = {0.f,0.f,0.f,1.f} };
			const IGPUCommandBuffer::SRenderpassBeginInfo info = {
				.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
				.colorClearValues = &clearColor,
				.depthStencilClearValues = nullptr,
				.renderArea = {.offset = {0,0}, .extent = {m_window->getWidth(), m_window->getHeight()} }
			};
			nbl::video::ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };

			cmdbuf->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

			cmdbuf->bindGraphicsPipeline(m_presentPipeline.get());
			cmdbuf->pushConstants(m_presentPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_FRAGMENT, 0, sizeof(m_present), &m_present);
			cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, m_presentPipeline->getLayout(), 0, 1u, &m_presentDs.get());
			ext::FullScreenTriangle::recordDrawCall(cmdbuf);

			const auto uiParams = m_ui.manager->getCreationParameters();
			auto* uiPipeline = m_ui.manager->getPipeline();
			cmdbuf->bindGraphicsPipeline(uiPipeline);
			cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, uiPipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get());
			m_ui.manager->render(cmdbuf, waitInfo);

			cmdbuf->endRenderPass();
		}

		cmdbuf->endDebugMarker();
		cmdbuf->end();

		// Submit and present to the scene
		{
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] = { {.semaphore = m_semaphore.get(), .value = ++m_realFrameIx, .stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS } };
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] = { {.cmdbuf = cmdbuf } };
			const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = { {.semaphore = m_currentImageAcquire.semaphore, .value = m_currentImageAcquire.acquireCount, .stageMask = PIPELINE_STAGE_FLAGS::NONE } };
			const IQueue::SSubmitInfo infos[] = { {.waitSemaphores = acquired, .commandBuffers = commandBuffers, .signalSemaphores = rendered } };

			updateGUIDescriptorSet();

			if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
				m_realFrameIx--;

			m_window->setCaption("[Nabla Engine] Photon Caustics");
			m_surface->present(m_currentImageAcquire.imageIndex, rendered);
		}
		m_api->endCapture();
		m_accumulatedFrames++;

		if (m_needPhotonCountReadback)
		{
			m_needPhotonCountReadback = false;
			readbackPhotonCount();
		}
	}

	inline void update()
	{
		m_camera.setMoveSpeed(m_cameraSetting.moveSpeed);
		m_camera.setRotateSpeed(m_cameraSetting.rotateSpeed);

		static std::chrono::microseconds previousEventTimestamp{};

		m_inputSystem->getDefaultMouse(&m_mouse);
		m_inputSystem->getDefaultKeyboard(&m_keyboard);

		auto updatePresentationTimestamp = [&]()
			{
				m_currentImageAcquire = m_surface->acquireNextImage();
				m_oracle.reportEndFrameRecord();
				const auto timestamp = m_oracle.getNextPresentationTimeStamp();
				m_oracle.reportBeginFrameRecord();
				return timestamp;
			};
		const auto nextPresentationTimestamp = updatePresentationTimestamp();

		struct {
			std::vector<SMouseEvent> mouse{};
			std::vector<SKeyboardEvent> keyboard{};
		} capturedEvents;

		m_camera.beginInputProcessing(nextPresentationTimestamp);
		{
			const auto& io = ImGui::GetIO();

			m_mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
				{
					// Note: When the cursor is over ImGui, the camera must not also see the event
					if (!io.WantCaptureMouse)
						m_camera.mouseProcess(events);
					for (const auto& e : events) {
						if (e.timeStamp < previousEventTimestamp) continue;
						previousEventTimestamp = e.timeStamp;
						capturedEvents.mouse.emplace_back(e);
					}
				}, m_logger.get());

			m_keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
				{
					if (!io.WantCaptureKeyboard)
						m_camera.keyboardProcess(events);
					for (const auto& e : events) {
						if (e.timeStamp < previousEventTimestamp) continue;
						previousEventTimestamp = e.timeStamp;
						capturedEvents.keyboard.emplace_back(e);

						//if (e.keyCode == EKC_ESCAPE)
						//{
						//	m_keepRunning = false;
						//}
					}
				}, m_logger.get());
		}
		m_camera.endInputProcessing(nextPresentationTimestamp);

		const core::SRange<const nbl::ui::SMouseEvent> mouseEvents(capturedEvents.mouse.data(), capturedEvents.mouse.data() + capturedEvents.mouse.size());
		const core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents(capturedEvents.keyboard.data(), capturedEvents.keyboard.data() + capturedEvents.keyboard.size());
		const auto cursorPosition = m_window->getCursorControl()->getPosition();
		const auto mousePosition = float32_t2(cursorPosition.x, cursorPosition.y) - float32_t2(m_window->getX(), m_window->getY());

		const ext::imgui::UI::SUpdateParameters params = {
			.mousePosition = mousePosition,
			.displaySize = { m_window->getWidth(), m_window->getHeight() },
			.mouseEvents = mouseEvents,
			.keyboardEvents = keyboardEvents
		};
		m_ui.manager->update(params);
	}

	inline bool keepRunning() override { return !m_surface->irrecoverable() && m_keepRunning; }
	inline bool onAppTerminated() override { return device_base_t::onAppTerminated(); }

private:
	struct SceneObject
	{
		core::smart_refctd_ptr<asset::ICPUPolygonGeometry> data;
		Material material;
		hlsl::float32_t3x4 transform;
		float boundsRadius = 0.f;
	};

	bool createScene()
	{
		auto geometryCreator = make_smart_refctd_ptr<CGeometryCreator>();

		auto placeQuad = [](const hlsl::float32_t3& axis, float degrees, const hlsl::float32_t3& translation)
			{
				hlsl::float32_t3x4 transform;
				if (degrees != 0.f)
				{
					const auto rotation = hlsl::math::quaternion<hlsl::float32_t>::create(axis, core::radians(degrees));
					transform = hlsl::math::linalg::promote_affine<3, 4, 3, 3>(hlsl::_static_cast<hlsl::float32_t3x3>(rotation));
				}
				else
					transform = hlsl::math::linalg::identity<hlsl::float32_t3x4>();

				hlsl::math::linalg::setTranslation(transform, translation);
				return transform;
			};
		auto placeAt = [&](float x, float y, float z)
			{
				return placeQuad(hlsl::float32_t3(1.f, 0.f, 0.f), 0.f, hlsl::float32_t3(x, y, z));
			};

		constexpr float R = SceneHalfExtent;
		const hlsl::float32_t3 X_AXIS(1.f, 0.f, 0.f);
		const hlsl::float32_t3 Y_AXIS(0.f, 1.f, 0.f);

		const Material white = { .albedo = {0.73f,0.73f,0.73f}, .emission = {0,0,0}, .metallic = 0.f, .roughness = 1.f, .ior = 1.f, .transmission = 0.f };
		const Material red = { .albedo = {0.65f,0.05f,0.05f}, .emission = {0,0,0}, .metallic = 0.6f, .roughness = 1.f, .ior = 1.f, .transmission = 0.f };
		const Material green = { .albedo = {0.12f,0.45f,0.15f}, .emission = {0,0,0}, .metallic = 0.f, .roughness = 1.f, .ior = 1.f, .transmission = 0.f };

		//-------------------------------------
		// Cornell Box
		//-------------------------------------
		{
			m_sceneObjects = {
				SceneObject{
					.data = geometryCreator->createRectangle({R, R}),
					.material = white,
					.transform = placeQuad(X_AXIS, -90.f, {0.f, 0.f, 0.f}),
				},
				SceneObject{
					.data = geometryCreator->createRectangle({R, R}),
					.material = white,
					.transform = placeQuad(X_AXIS, 90.f, {0.f, 2.f * R, 0.f}),
				},
				SceneObject{
					.data = geometryCreator->createRectangle({R, R}),
					.material = white,
					.transform = placeQuad(X_AXIS, 0.f, {0.f, R, -R}),
				},
				SceneObject{
					.data = geometryCreator->createRectangle({R, R}),
					.material = red,
					.transform = placeQuad(Y_AXIS, 90.f, {-R, R, 0.f}),
				},
				SceneObject{
					.data = geometryCreator->createRectangle({R, R}),
					.material = green,
					.transform = placeQuad(Y_AXIS, -90.f, {R, R, 0.f}),
				},
				SceneObject{
					.data = geometryCreator->createIcoSphere(SphereRadius, 4, true),
					.material = {.albedo = {1.f,1.f,1.f}, .emission = {0,0,0}, .metallic = 0.f, .roughness = 0.f, .ior = 1.5f, .transmission = 1.f },
					.transform = placeAt(-0.7f, 1.5f * SphereRadius, -0.2f),
					.boundsRadius = SphereRadius,
				},
				SceneObject{
					.data = geometryCreator->createCube({0.6f, 1.2f, 0.6f}),
					.material = white,
					.transform = placeAt(0.85f, 0.6f, -0.6f),
				},
				SceneObject{
					.data = geometryCreator->createRectangle({EmitterHalfExtent, EmitterHalfExtent}),
					.material = {.albedo = {0.f,0.f,0.f}, .emission = {22.f,19.f,14.f}, .metallic = 0.f, .roughness = 1.f, .ior = 1.f, .transmission = 0.f },
					.transform = placeQuad(X_AXIS, 90.f, {0.f, 2.f * R - 0.02f, 0.f}),
				},
			};
		}

		return createAccelerationStructuresAndGeomInfo();
	}

	bool createPhotonResources()
	{
		auto createRWBuffer = [&](size_t size, smart_refctd_ptr<IGPUBuffer>& out) -> bool
			{
				IGPUBuffer::SCreationParams p;
				p.size = size;
				p.usage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT)
					| IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT
					| IGPUBuffer::EUF_TRANSFER_DST_BIT
					| IGPUBuffer::EUF_TRANSFER_SRC_BIT
					| IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
				out = m_device->createBuffer(std::move(p));
				if (!out) return false;
				auto reqs = out->getMemoryReqs();
				reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				return m_device->allocate(reqs, { out.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT }).isValid();
			};

		if (!createRWBuffer(PHOTON_ARRAY_OFFSET + sizeof(SPhoton) * MaxPhotonInScene, m_photonBuffer))
			return logFail("Could not create Photon buffer");
		if (!createRWBuffer(sizeof(uint32_t) * GRID_CELLS, m_cellCountsBuffer))
			return logFail("Could not create photon grid cell counts buffer");
		if (!createRWBuffer(sizeof(uint32_t) * size_t(GRID_CELLS) * size_t(MaxPhotonInScene), m_cellPhotonsBuffer))
			return logFail("Could not create photon grid cell buffer");

		//-------------------------------------
		// Emissive triangles
		//-------------------------------------
		core::vector<SLight> lights;
		for (const auto& obj : m_sceneObjects)
		{
			const auto& e = obj.material.emission;
			if (e.x + e.y + e.z <= 0.f)
				continue;

			const hlsl::float32_t3 local[4] = {
				{-EmitterHalfExtent, 0.f, -EmitterHalfExtent},
				{ EmitterHalfExtent, 0.f, -EmitterHalfExtent},
				{ EmitterHalfExtent, 0.f,  EmitterHalfExtent},
				{-EmitterHalfExtent, 0.f,  EmitterHalfExtent},
			};
			const hlsl::float32_t3 origin(obj.transform[0][3], obj.transform[1][3], obj.transform[2][3]);

			hlsl::float32_t3 world[4];
			for (int i = 0; i < 4; i++)
				world[i] = origin + local[i];

			auto pushTri = [&](const hlsl::float32_t3& a, const hlsl::float32_t3& b, const hlsl::float32_t3& c)
				{
					SLight l = {};
					l.v0 = a;
					l.v1 = b;
					l.v2 = c;
					l.emission = e;
					l.area = 0.5f * hlsl::length(hlsl::cross(b - a, c - a));
					lights.push_back(l);
				};
			pushTri(world[0], world[1], world[2]);
			pushTri(world[0], world[2], world[3]);
		}

		m_lightCount = uint32_t(lights.size());
		if (m_lightCount == 0)
		{
			m_logger->log("No emissive geometry, photon pass will do nothing.", ILogger::ELL_WARNING);
			lights.push_back({});
		}

		{
			IGPUBuffer::SCreationParams params;
			params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT
				| IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
			params.size = lights.size() * sizeof(SLight);
			m_utils->createFilledDeviceLocalBufferOnDedMem(
				SIntendedSubmitInfo{ .queue = getGraphicsQueue() }, nbl_move(params), lights.data())
				.move_into(m_lightBuffer);
		}
		if (!m_lightBuffer)
			return logFail("Could not create light buffer");

		//-------------------------------------
		// Photon map bounds
		//-------------------------------------
		{
			m_photonMapCenter = hlsl::float32_t3(0.f, 0.f, 0.f);
			m_photonMapRadius = 0.f;

			uint32_t specularCount = 0;
			for (const auto& obj : m_sceneObjects)
			{
				const bool isSpecular = obj.material.metallic > 0.5f || obj.material.transmission > 0.5f;
				if (!isSpecular || obj.boundsRadius <= 0.f)
					continue;
				m_photonMapCenter = m_photonMapCenter + hlsl::float32_t3(obj.transform[0][3], obj.transform[1][3], obj.transform[2][3]);
				specularCount++;
			}

			if (specularCount > 0)
			{
				m_photonMapCenter = m_photonMapCenter / float(specularCount);
				for (const auto& obj : m_sceneObjects)
				{
					const bool isSpecular = obj.material.metallic > 0.5f || obj.material.transmission > 0.5f;
					if (!isSpecular || obj.boundsRadius <= 0.f)
						continue;
					const hlsl::float32_t3 c(obj.transform[0][3], obj.transform[1][3], obj.transform[2][3]);
					m_photonMapRadius = std::max(m_photonMapRadius, hlsl::length(c - m_photonMapCenter) + obj.boundsRadius);
				}
				m_logger->log("Photon map bounds: center (%.2f, %.2f, %.2f) radius %.2f", ILogger::ELL_INFO,
					m_photonMapCenter.x, m_photonMapCenter.y, m_photonMapCenter.z, m_photonMapRadius);
			}
			else
				m_logger->log("No specular geometry, photons will use hemisphere emission.", ILogger::ELL_WARNING);
		}

		//-------------------------------------
		// Uniform grid over the scene
		//-------------------------------------
		{
			const hlsl::float32_t3 gridMin(-SceneHalfExtent, 0.f, -SceneHalfExtent);
			const hlsl::float32_t3 gridMax( SceneHalfExtent, 2.f * SceneHalfExtent, SceneHalfExtent);

			m_gridMin = gridMin;
			const hlsl::float32_t3 extent = gridMax - gridMin;
			m_gridInvCellSize = hlsl::float32_t3(
				float(GRID_DIM) / extent.x, float(GRID_DIM) / extent.y, float(GRID_DIM) / extent.z);

			m_logger->log("Photon grid: %ux%ux%u cells over (%.2f, %.2f, %.2f)..(%.2f, %.2f, %.2f), cell size (%.3f, %.3f, %.3f)",
				ILogger::ELL_INFO, GRID_DIM, GRID_DIM, GRID_DIM,
				gridMin.x, gridMin.y, gridMin.z, gridMax.x, gridMax.y, gridMax.z,
				1.f / m_gridInvCellSize.x, 1.f / m_gridInvCellSize.y, 1.f / m_gridInvCellSize.z);
		}

		return true;
	}

	void readbackPhotonCount()
	{
		auto queue = getGraphicsQueue();
		m_device->waitIdle();

		smart_refctd_ptr<IGPUBuffer> hostBuffer;
		{
			IGPUBuffer::SCreationParams p;
			p.size = sizeof(uint32_t);
			p.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT;
			hostBuffer = m_device->createBuffer(std::move(p));
			if (!hostBuffer)
				return;
			auto reqs = hostBuffer->getMemoryReqs();
			reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDownStreamingMemoryTypeBits();
			if (!m_device->allocate(reqs, { hostBuffer.get() }).isValid())
				return;
		}

		auto pool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
		smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
		pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf);

		cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		const IGPUCommandBuffer::SBufferCopy region = { .srcOffset = PHOTON_COUNTER_OFFSET, .dstOffset = 0, .size = sizeof(uint32_t) };
		cmdbuf->copyBuffer(m_photonBuffer.get(), hostBuffer.get(), 1u, &region);
		cmdbuf->end();

		auto semaphore = m_device->createSemaphore(0u);
		{
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get() } };
			const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = semaphore.get(), .value = 1, .stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS } };
			const IQueue::SSubmitInfo infos[] = { {.commandBuffers = cmdbufs, .signalSemaphores = signals } };
			queue->submit(infos);
		}

		const ISemaphore::SWaitInfo waits[] = { {.semaphore = semaphore.get(), .value = 1 } };
		m_device->blockForSemaphores(waits);

		auto* memory = hostBuffer->getBoundMemory().memory;
		auto* mapped = memory->map({ 0ull, sizeof(uint32_t) }, IDeviceMemoryAllocation::EMCAF_READ);
		if (!mapped)
			return;

		const ILogicalDevice::MappedMemoryRange range(memory, 0ull, sizeof(uint32_t));
		if (memory->haveToMakeVisible())
			m_device->invalidateMappedMemoryRanges(1, &range);

		const uint32_t* counters = reinterpret_cast<const uint32_t*>(mapped);
		const uint32_t emitted = uint32_t(m_photonsEmitCount);
		m_storedPhotonsCount = std::min(counters[0], emitted);

		m_logger->log("Photon map: emitted %u, stored %u (%.1f%%)", ILogger::ELL_INFO,
			emitted, m_storedPhotonsCount, 100.0 * double(m_storedPhotonsCount) / double(emitted));
		memory->unmap();
	}

	bool createAccelerationStructuresAndGeomInfo()
	{
		auto queue = getGraphicsQueue();
		const auto objectCount = m_sceneObjects.size();

		//-------------------------------------
		// Build BLAS
		//-------------------------------------
		core::vector<smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>> cpuBlasList(objectCount);
		{
			for (uint32_t i = 0; i < objectCount; i++)
			{
				auto& blas = cpuBlasList[i];
				blas = make_smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>();

				auto triangles = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUBottomLevelAccelerationStructure::Triangles<ICPUBuffer>>>(1u);
				auto primitiveCounts = make_refctd_dynamic_array<smart_refctd_dynamic_array<uint32_t>>(1u);
				primitiveCounts->front() = m_sceneObjects[i].data->getPrimitiveCount();

				auto& tri = triangles->front();
				tri = m_sceneObjects[i].data->exportForBLAS();
				tri.geometryFlags = IGPUBottomLevelAccelerationStructure::GEOMETRY_FLAGS::OPAQUE_BIT;

				blas->setGeometries(nbl_move(triangles), nbl_move(primitiveCounts));
				blas->setBuildFlags(IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT);
				blas->setContentHash(blas->computeContentHash());
			}
		}

		//-------------------------------------
		// Build TLAS
		//-------------------------------------
		auto geomInstances = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUTopLevelAccelerationStructure::PolymorphicInstance>>(objectCount);
		{
			uint32_t i = 0;
			for (auto it = geomInstances->begin(); it != geomInstances->end(); it++, i++)
			{
				ICPUTopLevelAccelerationStructure::StaticInstance inst;
				inst.base.blas = cpuBlasList[i];
				inst.base.flags = static_cast<uint32_t>(IGPUTopLevelAccelerationStructure::INSTANCE_FLAGS::TRIANGLE_FACING_CULL_DISABLE_BIT);
				inst.base.instanceCustomIndex = i;
				inst.base.instanceShaderBindingTableRecordOffset = 0;
				inst.base.mask = 0xFF;
				inst.transform = m_sceneObjects[i].transform;
				it->instance = inst;
			}
		}

		auto cpuTlas = make_smart_refctd_ptr<ICPUTopLevelAccelerationStructure>();
		cpuTlas->setInstances(nbl_move(geomInstances));
		cpuTlas->setBuildFlags(IGPUTopLevelAccelerationStructure::BUILD_FLAGS::PREFER_FAST_TRACE_BIT);

		//-------------------------------------
		// Prepare For Conversion
		//-------------------------------------
		CAssetConverter::SInputs inputs = {};
		inputs.logger = m_logger.get();

		std::array<ICPUTopLevelAccelerationStructure*, 1> tmpTlas = { cpuTlas.get() };
		core::vector<ICPUPolygonGeometry*> tmpGeometries(objectCount);
		core::vector<CAssetConverter::patch_t<asset::ICPUPolygonGeometry>> tmpGeometryPatches(objectCount);
		{
			for (uint32_t i = 0; i < objectCount; i++)
			{
				tmpGeometries[i] = m_sceneObjects[i].data.get();
				tmpGeometryPatches[i].indexBufferUsages = IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;
			}

			std::get<CAssetConverter::SInputs::asset_span_t<ICPUTopLevelAccelerationStructure>>(inputs.assets) = tmpTlas;
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = tmpGeometries;
			std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = tmpGeometryPatches;
		}

		auto reservation = m_converter->reserve(inputs);

		//-------------------------------------
		// Scratch Buffer
		//-------------------------------------
		smart_refctd_ptr<CAssetConverter::SConvertParams::scratch_for_device_AS_build_t> scratchAlloc;
		{
			constexpr auto MaxAlignment = 256;
			constexpr auto MinAllocationSize = 1024;
			const auto scratchSize = core::alignUp(reservation.getMaxASBuildScratchSize(false), MaxAlignment);

			IGPUBuffer::SCreationParams cp = {};
			cp.size = scratchSize;
			cp.usage = IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
			auto scratchBuffer = m_device->createBuffer(nbl_move(cp));

			auto reqs = scratchBuffer->getMemoryReqs();
			reqs.memoryTypeBits &= m_physicalDevice->getDirectVRAMAccessMemoryTypeBits();
			auto allocation = m_device->allocate(reqs, { scratchBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT });
			allocation.memory->map({ .offset = 0, .length = reqs.size });

			scratchAlloc = make_smart_refctd_ptr<CAssetConverter::SConvertParams::scratch_for_device_AS_build_t>(
				SBufferRange<video::IGPUBuffer>{0ull, scratchSize, nbl_move(scratchBuffer)},
				core::allocator<uint8_t>(), MaxAlignment, MinAllocationSize
			);
		}

		//-------------------------------------
		// Command Buffers
		//-------------------------------------
		auto asPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);

		constexpr auto CompBufferCount = 2;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, CompBufferCount> computeBufs = {};
		std::array<IQueue::SSubmitInfo::SCommandBufferInfo, CompBufferCount> computeBufInfos = {};
		{
			asPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, computeBufs);
			computeBufs.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			for (auto i = 0; i < CompBufferCount; i++)
				computeBufInfos[i].cmdbuf = computeBufs[i].get();
		}

		std::array<smart_refctd_ptr<IGPUCommandBuffer>, CompBufferCount> transferBufs = {};
		std::array<IQueue::SSubmitInfo::SCommandBufferInfo, CompBufferCount> transferBufInfos = {};
		{
			asPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, transferBufs);
			transferBufs.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			for (auto i = 0; i < CompBufferCount; i++)
				transferBufInfos[i].cmdbuf = transferBufs[i].get();
		}

		//-------------------------------------
		// Submit Info
		//-------------------------------------
		auto computeSema = m_device->createSemaphore(0u);
		SIntendedSubmitInfo compute = {};
		{
			compute.queue = queue;
			compute.scratchCommandBuffers = computeBufInfos;
			compute.scratchSemaphore = {
				.semaphore = computeSema.get(),
				.value = 0u,
				.stageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT | PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_COPY_BIT
			};
		}

		auto transferSema = m_device->createSemaphore(0u);
		SIntendedSubmitInfo transfer = {};
		{
			transfer.queue = queue;
			transfer.scratchCommandBuffers = transferBufInfos;
			transfer.scratchSemaphore = {
				.semaphore = transferSema.get(),
				.value = 0u,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};
		}

		//-------------------------------------
		// Convert Acceleration Structures
		//-------------------------------------
		CAssetConverter::SConvertParams params = {};
		{
			params.utilities = m_utils.get();
			params.transfer = &transfer;
			params.compute = &compute;
			params.scratchForDeviceASBuild = scratchAlloc.get();
		}

		auto future = reservation.convert(params);
		if (future.copy() != IQueue::RESULT::SUCCESS)
		{
			m_logger->log("Failed to await acceleration structure build submission!", ILogger::ELL_ERROR);
			return false;
		}

		m_gpuTlas = reservation.getGPUObjects<ICPUTopLevelAccelerationStructure>()[0].value;
		if (!m_gpuTlas)
			return false;

		//-------------------------------------
		// Geometry Info
		//-------------------------------------
		auto geomInfoBuffer = ICPUBuffer::create({ objectCount * sizeof(SGeomInfo) });
		SGeomInfo* geomInfos = reinterpret_cast<SGeomInfo*>(geomInfoBuffer->getPointer());

		auto&& gpuPolygonGeometries = reservation.getGPUObjects<ICPUPolygonGeometry>();
		{
			for (uint32_t i = 0; i < gpuPolygonGeometries.size(); i++)
			{
				const auto& gpuPolygon = gpuPolygonGeometries[i].value;
				if (!gpuPolygon)
				{
					m_logger->log("Failed to convert a scene geometry to GPU!", ILogger::ELL_ERROR);
					return false;
				}
				const auto gpuTriangles = gpuPolygon->exportForBLAS();

				const auto& vertexBufferBinding = gpuTriangles.vertexData[0];
				const uint64_t vertexBufferAddress = vertexBufferBinding.buffer->getDeviceAddress() + vertexBufferBinding.offset;

				const auto& normalView = gpuPolygon->getNormalView();
				const uint64_t normalBufferAddress = normalView ? normalView.src.buffer->getDeviceAddress() + normalView.src.offset : 0;
				uint32_t normalType = 0;
				if (normalView && normalView.composed.format == EF_R8G8B8A8_SNORM)
					normalType = 1;

				const auto& indexBufferBinding = gpuTriangles.indexData;
				const uint64_t indexBufferAddress = indexBufferBinding.buffer ? indexBufferBinding.buffer->getDeviceAddress() + indexBufferBinding.offset : 0;

				geomInfos[i] = {
					.material = m_sceneObjects[i].material,
					.vertexBufferAddress = vertexBufferAddress,
					.indexBufferAddress = indexBufferAddress,
					.normalBufferAddress = normalBufferAddress,
					.indexType = static_cast<uint32_t>(gpuTriangles.indexType),
					.normalType = normalType,
				};
			}
		}

		//-------------------------------------
		// Upload Geometry Info Buffer
		//-------------------------------------
		{
			IGPUBuffer::SCreationParams bufParams;
			bufParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
			bufParams.size = geomInfoBuffer->getSize();
			m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = queue }, nbl_move(bufParams), geomInfos).move_into(m_geomInfoBuffer);
		}

		return bool(m_geomInfoBuffer);
	}

	bool createRayTracingPipeline(smart_refctd_ptr<IShader> rgen, smart_refctd_ptr<IShader> rchit, smart_refctd_ptr<IShader> rmiss, smart_refctd_ptr<IGPURayTracingPipeline>&& rayTracingPipeline, IGPURayTracingPipeline::SShaderBindingTable& sbt, uint64_t& rtStackSize)
	{
		//-------------------------------------
		// Descriptor Layout
		//-------------------------------------
		const auto bindings = std::array<ICPUDescriptorSetLayout::SBinding, 3>{
			ICPUDescriptorSetLayout::SBinding{
				.binding = 0,
				.type = asset::IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING,
				.count = 1,
			},
			{
				.binding = 1,
				.type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_RAYGEN,
				.count = 1
			},
			{
				.binding = 2,
				.type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_RAYGEN,
				.count = 1

			},
		};
		auto cpuDsLayout = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindings);

		const SPushConstantRange pcRange = {
			.stageFlags = IShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING,
			.offset = 0u,
			.size = sizeof(SPushConstants)
		};
		const auto cpuPipelineLayout = core::make_smart_refctd_ptr<ICPUPipelineLayout>(std::span<const asset::SPushConstantRange>({ pcRange }), nbl_move(cpuDsLayout), nullptr, nullptr, nullptr);

		//-------------------------------------
		// Create Pipeline
		//-------------------------------------
		const auto pipeline = ICPURayTracingPipeline::create(cpuPipelineLayout.get());
		{
			pipeline->getCachedCreationParams() = {
				.flags = IGPURayTracingPipeline::SCreationParams::FLAGS::NONE,
				.maxRecursionDepth = 1,
				.dynamicStackSize = true,
			};
		}

		//-------------------------------------
		// Setup Shaders
		//-------------------------------------
		pipeline->getSpecInfos(ESS_RAYGEN)[0] = { .shader = rgen, .entryPoint = "main" };

		pipeline->getSpecInfoVector(ESS_MISS)->resize(1);
		pipeline->getSpecInfos(ESS_MISS)[0] = { .shader = rmiss, .entryPoint = "main" };

		pipeline->getSpecInfoVector(ESS_CLOSEST_HIT)->resize(1);
		pipeline->getSpecInfoVector(ESS_ANY_HIT)->resize(1);
		pipeline->getSpecInfoVector(ESS_INTERSECTION)->resize(1);
		pipeline->getSpecInfos(ESS_CLOSEST_HIT)[0] = { .shader = rchit, .entryPoint = "main" };

		//-------------------------------------
		// Convert to GPU
		//-------------------------------------
		CAssetConverter::SInputs inputs = {};
		{
			inputs.logger = m_logger.get();
			const std::array cpuPipelines = { pipeline.get() };
			std::get<CAssetConverter::SInputs::asset_span_t<ICPURayTracingPipeline>>(inputs.assets) = cpuPipelines;
		}

		auto reservation = m_converter->reserve(inputs);
		CAssetConverter::SConvertParams params = {};
		{
			params.utilities = m_utils.get();
		}
		if (reservation.convert(params).copy() != IQueue::RESULT::SUCCESS)
			return logFail("Failed to convert ray tracing pipeline");

		rayTracingPipeline = reservation.getGPUObjects<ICPURayTracingPipeline>()[0].value;
		if (!rayTracingPipeline)
			return false;

		//-------------------------------------
		// Create Descriptor Set
		//-------------------------------------
		if (!m_rayTracingDs)
		{
			const auto* gpuDsLayout = rayTracingPipeline->getLayout()->getDescriptorSetLayouts()[0];
			const std::array<const IGPUDescriptorSetLayout*, 1> dsLayoutPtrs = { gpuDsLayout };
			m_rayTracingDsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dsLayoutPtrs));
			m_rayTracingDs = m_rayTracingDsPool->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(gpuDsLayout));

			//-------------------------------------
			// Write Descriptors
			//-------------------------------------
			IGPUDescriptorSet::SDescriptorInfo infos[3] = {};
			{
				infos[0].desc = m_gpuTlas;
				infos[1].desc = m_hdrImageView;
				infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
				infos[2].desc = m_accumImageView;
				infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
			}

			IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
				{.dstSet = m_rayTracingDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &infos[0]},
				{.dstSet = m_rayTracingDs.get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &infos[1]},
				{.dstSet = m_rayTracingDs.get(), .binding = 2, .arrayElement = 0, .count = 1, .info = &infos[2]},
			};
			m_device->updateDescriptorSets(std::span(writes), {});
		}

		rtStackSize = calculateRayTracingStackSize(std::forward<smart_refctd_ptr<IGPURayTracingPipeline>>(rayTracingPipeline));
		return createShaderBindingTable(std::forward<smart_refctd_ptr<IGPURayTracingPipeline>>(rayTracingPipeline), sbt);
	}

	uint32_t calculateRayTracingStackSize(smart_refctd_ptr<IGPURayTracingPipeline>&& rayTracingPipeline)
	{
		const auto raygenStackSize = rayTracingPipeline->getRaygenStackSize();
		auto getMaxSize = [&](auto ranges, auto valProj) -> uint16_t
			{
				uint16_t maxValue = 0;
				for (const auto& val : ranges)
					maxValue = std::max<uint16_t>(maxValue, std::invoke(valProj, val));
				return maxValue;
			};

		const auto closestHitMax = getMaxSize(rayTracingPipeline->getHitStackSizes(), &IGPURayTracingPipeline::SHitGroupStackSize::closestHit);
		const auto missMax = getMaxSize(rayTracingPipeline->getMissStackSizes(), std::identity{});
		return raygenStackSize + std::max(closestHitMax, missMax);
	}

	bool createShaderBindingTable(smart_refctd_ptr<IGPURayTracingPipeline>&& rayTracingPipeline, IGPURayTracingPipeline::SShaderBindingTable& shaderBindingTable)
	{
		const auto& limits = m_device->getPhysicalDevice()->getLimits();
		const auto handleSize = SPhysicalDeviceLimits::ShaderGroupHandleSize;
		const auto handleSizeAligned = nbl::core::alignUp(handleSize, limits.shaderGroupHandleAlignment);

		const auto missHandles = rayTracingPipeline->getMissHandles();
		const auto hitHandles = rayTracingPipeline->getHitHandles();

		//-------------------------------------
		// Calculate Ranges
		//-------------------------------------
		auto& raygenRange = shaderBindingTable.raygen;
		auto& missRange = shaderBindingTable.miss.range;
		auto& hitRange = shaderBindingTable.hit.range;

		{
			raygenRange = { .offset = 0, .size = core::alignUp(handleSizeAligned, limits.shaderGroupBaseAlignment) };

			missRange = { .offset = raygenRange.size, .size = core::alignUp(missHandles.size() * handleSizeAligned, limits.shaderGroupBaseAlignment) };
			shaderBindingTable.miss.stride = handleSizeAligned;

			hitRange = { .offset = missRange.offset + missRange.size, .size = core::alignUp(hitHandles.size() * handleSizeAligned, limits.shaderGroupBaseAlignment) };
			shaderBindingTable.hit.stride = handleSizeAligned;
		}

		const auto bufferSize = raygenRange.size + missRange.size + hitRange.size;

		//-------------------------------------
		// Fill CPU Buffer
		//-------------------------------------
		ICPUBuffer::SCreationParams cpuBufferParams;
		cpuBufferParams.size = bufferSize;
		auto cpuBuffer = ICPUBuffer::create(nbl_move(cpuBufferParams));
		uint8_t* pData = reinterpret_cast<uint8_t*>(cpuBuffer->getPointer());

		{
			memcpy(pData, &rayTracingPipeline->getRaygen(), handleSize);

			uint8_t* p = pData + missRange.offset;
			for (const auto& h : missHandles)
			{
				memcpy(p, &h, handleSize);
				p += shaderBindingTable.miss.stride;
			}

			p = pData + hitRange.offset;
			for (const auto& h : hitHandles)
			{
				memcpy(p, &h, handleSize);
				p += shaderBindingTable.hit.stride;
			}
		}

		//-------------------------------------
		// Create GPU Buffer
		//-------------------------------------
		smart_refctd_ptr<IGPUBuffer> buffer;
		{
			IGPUBuffer::SCreationParams params;
			params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | IGPUBuffer::EUF_SHADER_BINDING_TABLE_BIT;
			params.size = bufferSize;
			m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = getGraphicsQueue() }, nbl_move(params), pData).move_into(buffer);
		}

		raygenRange.buffer = smart_refctd_ptr(buffer);
		missRange.buffer = smart_refctd_ptr(buffer);
		hitRange.buffer = smart_refctd_ptr(buffer);
		return true;
	}

	bool createImGui(nbl::video::IGPURenderpass* renderpass)
	{
		{
			IGPUSampler::SParams params;
			params.AnisotropicFilter = 1u;
			params.TextureWrapU = ETC_REPEAT;
			params.TextureWrapV = ETC_REPEAT;
			params.TextureWrapW = ETC_REPEAT;
			m_ui.sampler = m_device->createSampler(params);
			m_ui.sampler->setObjectDebugName("Nabla IMGUI UI Sampler");
		}

		{
			nbl::ext::imgui::UI::SCreationParameters params;
			params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
			params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
			params.assetManager = m_assetMgr;
			params.pipelineCache = nullptr;
			params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, MaxUITextureCount);
			params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
			params.streamingBuffer = nullptr;
			params.subpassIx = 0u;
			params.transfer = getGraphicsQueue();
			params.utilities = m_utils;

			m_ui.manager = ext::imgui::UI::create(nbl_move(params));
			if (!m_ui.manager)
				return false;
		}

		{
			const auto* descriptorSetLayout = m_ui.manager->getPipeline()->getLayout()->getDescriptorSetLayout(0u);

			IDescriptorPool::SCreateInfo descriptorPoolInfo = {};
			descriptorPoolInfo.maxDescriptorCount[(uint32_t)asset::IDescriptor::E_TYPE::ET_SAMPLER] = (uint32_t)nbl::ext::imgui::UI::DefaultSamplerIx::COUNT;
			descriptorPoolInfo.maxDescriptorCount[(uint32_t)asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE] = MaxUITextureCount;
			descriptorPoolInfo.maxSets = 1u;
			descriptorPoolInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT;

			m_ui.descriptorPool = m_device->createDescriptorPool(nbl_move(descriptorPoolInfo));
			if (!m_ui.descriptorPool)
				return false;
			m_ui.descriptorPool->createDescriptorSets(1u, &descriptorSetLayout, &m_ui.descriptorSet);
			if (!m_ui.descriptorSet)
				return false;
		}

		// Build actual UI here
		m_ui.manager->registerListener([this]() -> void
			{
				ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
				ImGui::SetNextWindowSize(ImVec2(320, 260), ImGuiCond_Appearing);
				ImGui::Begin("Photon Caustics");

				ImGui::Text("%.1f FPS (%.2f ms)", ImGui::GetIO().Framerate, 1000.f / ImGui::GetIO().Framerate);
				ImGui::Text("Accumulated frames: %u", m_accumulatedFrames);
				ImGui::Separator();

				ImGui::SliderFloat("Move speed", &m_cameraSetting.moveSpeed, 0.1f, 10.f);
				ImGui::SliderFloat("Rotate speed", &m_cameraSetting.rotateSpeed, 0.1f, 10.f);
				if (ImGui::Button("Reset Camera"))
				{
					m_camera.setPosition(InitialCamPos);
					m_camera.setTarget(InitialCamTarget);
					m_cachedMVP = m_InitialMVP;
					m_accumulatedFrames = 0;
				}
				ImGui::Separator();

				bool causticsDirty = false;
				causticsDirty |= ImGui::Checkbox("Caustics", &m_enablePhotonCaustics);
				if (ImGui::Checkbox("Debug: photon density", &m_debugPhotonView))
				{
					causticsDirty = true;
					m_present.tonemapOperator = m_debugPhotonView ? 0u : 2u;
					m_present.exposure = 1.0f;
				}
				if (ImGui::Checkbox("Debug: no specular concentration", &m_disableSpecularConcentration))
				{
					m_photonMapBuilt = false;
					causticsDirty    = true;
				}
				if (ImGui::SliderFloat("Gather radius", &m_gatherRadius, 0.005f, 0.25f, "%.4f"))
					m_accumulatedFrames = 0;
				if (ImGui::SliderInt("Photons", &m_photonsEmitCount, 1024, MaxPhotonInScene))
				{
					m_photonMapBuilt = false;
					causticsDirty    = true;
				}

				if (causticsDirty)
					m_accumulatedFrames = 0;

				ImGui::Separator();
				ImGui::SliderFloat("Exposure", &m_present.exposure, 0.01f, 8.f);
				ImGui::Combo("Tonemap", (int*)&m_present.tonemapOperator, "None\0Reinhard\0ACES\0");

				ImGui::End();
			});

		return true;
	}

	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>> m_surface;
	smart_refctd_ptr<ISemaphore> m_semaphore;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
	video::CDumbPresentationOracle m_oracle; // maybe is it like VK_KHR_present_wait to get accurate frame time until presentation is also done?

	smart_refctd_ptr<InputSystem> m_inputSystem;
	InputSystem::ChannelReader<IMouseEventChannel> m_mouse;
	InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;

	smart_refctd_ptr<IGPUImage> m_hdrImage, m_accumImage;
	smart_refctd_ptr<IGPUImageView> m_hdrImageView, m_accumImageView;

	smart_refctd_ptr<IDescriptorPool> m_presentDsPool;
	smart_refctd_ptr<IGPUDescriptorSet> m_presentDs;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_presentPipeline;
	SPresentPushConstants m_present = { .exposure = 1.0f, .tonemapOperator = 2 };

	smart_refctd_ptr<CAssetConverter> m_converter;
	core::vector<SceneObject> m_sceneObjects;
	smart_refctd_ptr<IGPUTopLevelAccelerationStructure> m_gpuTlas;
	smart_refctd_ptr<IGPUBuffer> m_geomInfoBuffer;
	smart_refctd_ptr<IDescriptorPool> m_rayTracingDsPool;
	smart_refctd_ptr<IGPUDescriptorSet> m_rayTracingDs;
	smart_refctd_ptr<IGPURayTracingPipeline> m_rayTracingPipeline;
	uint64_t m_rayTracingStackSize = 0;
	IGPURayTracingPipeline::SShaderBindingTable m_shaderBindingTable;

	smart_refctd_ptr<IGPUBuffer> m_photonBuffer;
	smart_refctd_ptr<IGPUBuffer> m_cellCountsBuffer;
	smart_refctd_ptr<IGPUBuffer> m_cellPhotonsBuffer;
	hlsl::float32_t3 m_gridMin = {};
	hlsl::float32_t3 m_gridInvCellSize = {};
	smart_refctd_ptr<IGPURayTracingPipeline> m_photonRayTracingPipeline;
	IGPURayTracingPipeline::SShaderBindingTable m_photonSBT;
	uint64_t m_photonRTStackSize = 0;

	smart_refctd_ptr<IGPUBuffer> m_lightBuffer;
	uint32_t m_lightCount = 0;
	hlsl::float32_t3 m_photonMapCenter = {};
	float m_photonMapRadius = 0.f;

	int32_t m_photonsEmitCount{ MaxPhotonInScene };
	bool m_photonMapBuilt = false;
	bool m_needPhotonCountReadback = false;
	uint32_t m_storedPhotonsCount = 0;
	bool m_enablePhotonCaustics = true;
	bool m_debugPhotonView = false;
	bool m_disableSpecularConcentration = false;
	float m_gatherRadius = 0.06f;

	struct ImGuiRes
	{
		smart_refctd_ptr<nbl::ext::imgui::UI> manager;
		smart_refctd_ptr<IGPUSampler> sampler;
		smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
		smart_refctd_ptr<IDescriptorPool> descriptorPool;
	} m_ui;

	struct CameraSettings { float moveSpeed = 1.f, rotateSpeed = 1.f, fov = 60.f, zNear = 0.01f, zFar = 500.f; } m_cameraSetting;
	Camera m_camera;
	hlsl::float32_t4x4 m_cachedMVP;
	hlsl::float32_t4x4 m_InitialMVP;
	uint32_t m_accumulatedFrames = 0;
	bool m_keepRunning = true;
};

NBL_MAIN_FUNC(PhotonCausticsApp)
