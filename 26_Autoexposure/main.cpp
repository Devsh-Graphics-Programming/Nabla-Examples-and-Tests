// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "../common/SimpleWindowedApplication.hpp"

#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"


using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

//#include "app_resources/push_constants.hlsl"

class AutoexposureApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
	using clock_t = std::chrono::steady_clock;

	constexpr static inline std::string_view DefaultImagePathsFile = "../../media/noises/spp_benchmark_4k_512.exr";

public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	inline AutoexposureApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	// Will get called mid-initialization, via `filterDevices` between when the API Connection is created and Physical Device is chosen
	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		// So let's create our Window and Surface then!
		if (!m_surface)
		{
			{
				IWindow::SCreationParams params = {};
				params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
				params.width = 256;
				params.height = 256;
				params.x = 32;
				params.y = 32;
				// Don't want to have a window lingering about before we're ready so create it hidden.
				// Only programmatic resize, not regular.
				params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
				params.windowCaption = "AutoexposureApp";
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
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		/*
			* We'll be using a combined image sampler for this example, which lets us assign both a sampled image and a sampler to the same binding.
			* In this example we provide a sampler at descriptor set creation time, via the SBinding struct below. This specifies that the sampler for this binding is immutable,
			* as evidenced by the name of the field in the SBinding.
			* Samplers for combined image samplers can also be mutable, which for a binding of a descriptor set is specified also at creation time by leaving the immutableSamplers
			* field set to its default (nullptr).
			*/
		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
		{
			auto defaultSampler = m_device->createSampler({
				.AnisotropicFilter = 0
				});

			const IGPUDescriptorSetLayout::SBinding bindings[1] = { {
				.binding = 0,
				.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
				.count = 1,
				.immutableSamplers = &defaultSampler
			}
			};
			dsLayout = m_device->createDescriptorSetLayout(bindings);
			if (!dsLayout)
				return logFail("Failed to Create Descriptor Layout");

		}

		// create the descriptor set and with enough room for one image sampler
		{
			const uint32_t setCount = 1;
			auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, { &dsLayout.get(),1 }, &setCount);
			if (!pool)
				return logFail("Failed to Create Descriptor Pool");

			m_descriptorSets[0] = pool->createDescriptorSet(core::smart_refctd_ptr(dsLayout));
			if (!m_descriptorSets[0])
				return logFail("Could not create Descriptor Set!");
		}

		auto queue = getGraphicsQueue();

		// Gather swapchain resources
		std::unique_ptr<CDefaultSwapchainFramebuffers> scResources;
		ISwapchain::SCreationParams swapchainParams;
		{
			swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
			// Need to choose a surface format
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");
			// We actually need external dependencies to ensure ordering of the Implicit Layout Transitions relative to the semaphore signals
			constexpr IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
				// wipe-transition to ATTACHMENT_OPTIMAL
				{
					.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.dstSubpass = 0,
					.memoryBarrier = {
					// since we're uploading the image data we're about to draw
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
					.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
					.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					// because we clear and don't blend
					.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
					// leave view offsets and flags default
				},
				// ATTACHMENT_OPTIMAL to PRESENT_SRC
				{
					.srcSubpass = 0,
					.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.memoryBarrier = {
						.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
						// we can have NONE as the Destinations because the spec says so about presents
						}
					// leave view offsets and flags default
				},
				IGPURenderpass::SCreationParams::DependenciesEnd
			};
			scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(), swapchainParams.surfaceFormat.format, dependencies);
			if (!scResources->getRenderpass())
				return logFail("Failed to create Renderpass!");
		}

		// Load the shaders and create the pipeline
		{
			// Load FSTri Shader
			ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
			if (!fsTriProtoPPln)
				return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

			// Load Custom Shader
			auto loadCompileAndCreateShader = [&](const std::string& relPath) -> smart_refctd_ptr<IGPUShader>
				{
					IAssetLoader::SAssetLoadParams lp = {};
					lp.logger = m_logger.get();
					lp.workingDirectory = ""; // virtual root
					auto assetBundle = m_assetMgr->getAsset(relPath, lp);
					const auto assets = assetBundle.getContents();
					if (assets.empty())
						return nullptr;

					// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
					auto source = IAsset::castDown<ICPUShader>(assets[0]);
					if (!source)
						return nullptr;

					return m_device->createShader(source.get());
				};
			auto fragmentShader = loadCompileAndCreateShader("app_resources/present.frag.hlsl");
			if (!fragmentShader)
				return logFail("Failed to Load and Compile Fragment Shader!");

			auto layout = m_device->createPipelineLayout({}, nullptr, nullptr, nullptr, core::smart_refctd_ptr(dsLayout));
			const IGPUShader::SSpecInfo fragSpec = {
				.entryPoint = "main",
				.shader = fragmentShader.get()
			};
			m_pipeline = fsTriProtoPPln.createPipeline(fragSpec, layout.get(), scResources->getRenderpass());
			if (!m_pipeline)
				return logFail("Could not create Graphics Pipeline!");
		}

		// Init the surface and create the swapchain
		if (!m_surface || !m_surface->init(queue, std::move(scResources), swapchainParams.sharedParams))
			return logFail("Could not create Window & Surface or initialize the Surface!");

		// need resetttable commandbuffers for the upload utility
		{
			m_cmdPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			// create the commandbuffers
			if (!m_cmdPool)
				return logFail("Couldn't create Command Pool!");
			if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data(), 1 }))
				return logFail("Couldn't create Command Buffer!");
		}

		// things for IUtilities
		{
			m_scratchSemaphore = m_device->createSemaphore(0);
			if (!m_scratchSemaphore)
				return logFail("Could not create Scratch Semaphore");
			m_scratchSemaphore->setObjectDebugName("Scratch Semaphore");
			// we don't want to overcomplicate the example with multi-queue
			m_intendedSubmit.queue = queue;
			// wait for nothing before upload
			m_intendedSubmit.waitSemaphores = {};
			// fill later
			m_intendedSubmit.commandBuffers = {};
			m_intendedSubmit.scratchSemaphore = {
				.semaphore = m_scratchSemaphore.get(),
				.value = 0,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};
		}

		// Allocate and Leave 1/4 for image uploads, to test image copy with small memory remaining
		{
			uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_value;
			uint32_t maxFreeBlock = m_utils->getDefaultUpStreamingBuffer()->max_size();
			const uint32_t allocationAlignment = 64u;
			const uint32_t allocationSize = (maxFreeBlock / 4) * 3;
			m_utils->getDefaultUpStreamingBuffer()->multi_allocate(std::chrono::steady_clock::now() + std::chrono::microseconds(500u), 1u, &localOffset, &allocationSize, &allocationAlignment);
		}

		// Load exr file into gpu
		{
			IAssetLoader::SAssetLoadParams params;
			auto imageBundle = m_assetMgr->getAsset(DefaultImagePathsFile.data(), params);
			auto cpuImg = IAsset::castDown<ICPUImage>(imageBundle.getContents().begin()[0]);
			auto format = cpuImg->getCreationParameters().format;

			ICPUImageView::SCreationParams viewParams = {
				.flags = ICPUImageView::E_CREATE_FLAGS::ECF_NONE,
				.image = std::move(cpuImg),
				.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D,
				.format = format,
				.subresourceRange = {
					.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = ICPUImageView::remaining_mip_levels,
					.baseArrayLayer = 0u,
					.layerCount = ICPUImageView::remaining_array_layers
				}
			};

			const auto cpuImgView = ICPUImageView::create(std::move(viewParams));
			const auto& cpuImgParams = cpuImgView->getCreationParameters();

			// create matching size image
			IGPUImage::SCreationParams imageParams = {};
			imageParams = cpuImgParams.image->getCreationParameters();
			imageParams.usage |= IGPUImage::EUF_TRANSFER_DST_BIT | IGPUImage::EUF_SAMPLED_BIT | IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT;
			// promote format because RGB8 and friends don't actually exist in HW
			{
				const IPhysicalDevice::SImageFormatPromotionRequest request = {
					.originalFormat = imageParams.format,
					.usages = IPhysicalDevice::SFormatImageUsages::SUsage(imageParams.usage)
				};
				imageParams.format = m_physicalDevice->promoteImageFormat(request, imageParams.tiling);
			}
			if (imageParams.type == IGPUImage::ET_3D)
				imageParams.flags |= IGPUImage::ECF_2D_ARRAY_COMPATIBLE_BIT;
			m_gpuImg = m_device->createImage(std::move(imageParams));
			if (!m_gpuImg || !m_device->allocate(m_gpuImg->getMemoryReqs(), m_gpuImg.get()).isValid())
				return false;
			m_gpuImg->setObjectDebugName("Autoexposure Image");

			// we don't want to overcomplicate the example with multi-queue
			auto queue = getGraphicsQueue();
			auto cmdbuf = m_cmdBufs[0].get();
			IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = { cmdbuf };
			m_intendedSubmit.commandBuffers = { &cmdbufInfo, 1 };

			// there's no previous operation to wait for
			const SMemoryBarrier toTransferBarrier = {
				.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
				.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
			};

			// upload image and write to descriptor set
			queue->startCapture();
			auto ds = m_descriptorSets[0].get();

			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			// change the layout of the image
			const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers[] = { {
				.barrier = {
					.dep = toTransferBarrier
					// no ownership transfers
				},
				.image = m_gpuImg.get(),
				// transition the whole view
				.subresourceRange = cpuImgParams.subresourceRange,
				// a wiping transition
				.newLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL
			} };
			cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers });
			// upload contents and submit right away
			m_utils->updateImageViaStagingBufferAutoSubmit(
				m_intendedSubmit,
				cpuImgParams.image->getBuffer(),
				cpuImgParams.image->getCreationParameters().format,
				m_gpuImg.get(),
				IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				cpuImgParams.image->getRegions()
			);

			IGPUImageView::SCreationParams gpuImgViewParams = {
				.image = m_gpuImg,
				.viewType = IGPUImageView::ET_2D_ARRAY,
				.format = m_gpuImg->getCreationParameters().format
			};

			m_gpuImgView = m_device->createImageView(std::move(gpuImgViewParams));
		}

		return true;
	}

	// We do a very simple thing, display an image and wait `DisplayImageMs` to show it
	inline void workLoopBody() override
	{
	}

	inline bool keepRunning() override
	{
		return false;
	}

	inline bool onAppTerminated() override
	{
		return device_base_t::onAppTerminated();
	}

protected:
	smart_refctd_ptr<IGPUImage> m_gpuImg;
	smart_refctd_ptr<IGPUImageView> m_gpuImgView;

	// for image uploads
	smart_refctd_ptr<ISemaphore> m_scratchSemaphore;
	SIntendedSubmitInfo m_intendedSubmit;

	// Command Buffers and other resources
	std::array<smart_refctd_ptr<IGPUDescriptorSet>, ISwapchain::MaxImages> m_descriptorSets;
	smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;

	// window
	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;
};

NBL_MAIN_FUNC(AutoexposureApp)

#if 0

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include <iostream>
#include <cstdio>


#include "nbl/ext/ToneMapper/CToneMapper.h"

#include "../common/QToQuitEventReceiver.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;


int main()
{
	nbl::SIrrlichtCreationParameters deviceParams;
	deviceParams.Bits = 24; //may have to set to 32bit for some platforms
	deviceParams.ZBufferBits = 24; //we'd like 32bit here
	deviceParams.DriverType = EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	deviceParams.WindowSize = dimension2d<uint32_t>(1280, 720);
	deviceParams.Fullscreen = false;
	deviceParams.Vsync = true; //! If supported by target platform
	deviceParams.Doublebuffer = true;
	deviceParams.Stencilbuffer = false; //! This will not even be a choice soon

	auto device = createDeviceEx(deviceParams);
	if (!device)
		return 1; // could not create selected driver.

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	IVideoDriver* driver = device->getVideoDriver();
	
	nbl::io::IFileSystem* filesystem = device->getFileSystem();
	IAssetManager* am = device->getAssetManager();

	IAssetLoader::SAssetLoadParams lp;
	auto imageBundle = am->getAsset("../../media/noises/spp_benchmark_4k_512.exr", lp);

	auto glslCompiler = am->getCompilerSet();
	const auto inputColorSpace = std::make_tuple(inFormat,ECP_SRGB,EOTF_IDENTITY);

	using LumaMeterClass = ext::LumaMeter::CLumaMeter;
	constexpr auto MeterMode = LumaMeterClass::EMM_MEDIAN;
	const float minLuma = 1.f/2048.f;
	const float maxLuma = 65536.f;

	auto cpuLumaMeasureSpecializedShader = LumaMeterClass::createShader(glslCompiler,inputColorSpace,MeterMode,minLuma,maxLuma);
	auto gpuLumaMeasureShader = driver->createShader(smart_refctd_ptr<const ICPUShader>(cpuLumaMeasureSpecializedShader->getUnspecialized()));
	auto gpuLumaMeasureSpecializedShader = driver->createSpecializedShader(gpuLumaMeasureShader.get(), cpuLumaMeasureSpecializedShader->getSpecializationInfo());

	const float meteringMinUV[2] = { 0.1f,0.1f };
	const float meteringMaxUV[2] = { 0.9f,0.9f };
	LumaMeterClass::Uniforms_t<MeterMode> uniforms;
	auto lumaDispatchInfo = LumaMeterClass::buildParameters(uniforms, outImg->getCreationParameters().extent, meteringMinUV, meteringMaxUV);

	auto uniformBuffer = driver->createFilledDeviceLocalBufferOnDedMem(sizeof(uniforms),&uniforms);


	using ToneMapperClass = ext::ToneMapper::CToneMapper;
	constexpr auto TMO = ToneMapperClass::EO_ACES;
	constexpr bool usingLumaMeter = MeterMode<LumaMeterClass::EMM_COUNT;
	constexpr bool usingTemporalAdapatation = true;

	auto cpuTonemappingSpecializedShader = ToneMapperClass::createShader(am->getGLSLCompiler(),
		inputColorSpace,
		std::make_tuple(outFormat,ECP_SRGB,OETF_sRGB),
		TMO,usingLumaMeter,MeterMode,minLuma,maxLuma,usingTemporalAdapatation
	);
	auto gpuTonemappingShader = driver->createShader(smart_refctd_ptr<const ICPUShader>(cpuTonemappingSpecializedShader->getUnspecialized()));
	auto gpuTonemappingSpecializedShader = driver->createSpecializedShader(gpuTonemappingShader.get(),cpuTonemappingSpecializedShader->getSpecializationInfo());

	auto outImgStorage = ToneMapperClass::createViewForImage(driver,false,core::smart_refctd_ptr(outImg),{static_cast<IImage::E_ASPECT_FLAGS>(0u),0,1,0,1});

	auto parameterBuffer = driver->createDeviceLocalGPUBufferOnDedMem(ToneMapperClass::getParameterBufferSize<TMO,MeterMode>());
	constexpr float Exposure = 0.f;
	constexpr float Key = 0.18;
	auto params = ToneMapperClass::Params_t<TMO>(Exposure, Key, 0.85f);
	{
		params.setAdaptationFactorFromFrameDelta(0.f);
		driver->updateBufferRangeViaStagingBuffer(parameterBuffer.get(),0u,sizeof(params),&params);
	}

	auto commonPipelineLayout = ToneMapperClass::getDefaultPipelineLayout(driver,usingLumaMeter);

	auto lumaMeteringPipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(commonPipelineLayout),std::move(gpuLumaMeasureSpecializedShader));
	auto toneMappingPipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(commonPipelineLayout),std::move(gpuTonemappingSpecializedShader));

	auto commonDescriptorSet = driver->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(commonPipelineLayout->getDescriptorSetLayout(0u)));
	ToneMapperClass::updateDescriptorSet<TMO,MeterMode>(driver,commonDescriptorSet.get(),parameterBuffer,imgToTonemapView,outImgStorage,1u,2u,usingLumaMeter ? 3u:0u,uniformBuffer,0u,usingTemporalAdapatation);


	constexpr auto dynOffsetArrayLen = usingLumaMeter ? 2u : 1u;

	auto lumaDynamicOffsetArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(dynOffsetArrayLen,0u);
	lumaDynamicOffsetArray->back() = sizeof(ToneMapperClass::Params_t<TMO>);

	auto toneDynamicOffsetArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(dynOffsetArrayLen,0u);


	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImgView));

	uint32_t outBufferIx = 0u;
	auto lastPresentStamp = std::chrono::high_resolution_clock::now();
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		driver->bindComputePipeline(lumaMeteringPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE,commonPipelineLayout.get(),0u,1u,&commonDescriptorSet.get(),&lumaDynamicOffsetArray);
		driver->pushConstants(commonPipelineLayout.get(),IGPUSpecializedShader::ESS_COMPUTE,0u,sizeof(outBufferIx),&outBufferIx); outBufferIx ^= 0x1u;
		LumaMeterClass::dispatchHelper(driver,lumaDispatchInfo,true);

		driver->bindComputePipeline(toneMappingPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE,commonPipelineLayout.get(),0u,1u,&commonDescriptorSet.get(),&toneDynamicOffsetArray);
		ToneMapperClass::dispatchHelper(driver,outImgStorage.get(),true);

		driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();
		if (usingTemporalAdapatation)
		{
			auto thisPresentStamp = std::chrono::high_resolution_clock::now();
			auto microsecondsElapsedBetweenPresents = std::chrono::duration_cast<std::chrono::microseconds>(thisPresentStamp-lastPresentStamp);
			lastPresentStamp = thisPresentStamp;

			params.setAdaptationFactorFromFrameDelta(float(microsecondsElapsedBetweenPresents.count())/1000000.f);
			// dont override shader output
			constexpr auto offsetPastLumaHistory = offsetof(decltype(params),lastFrameExtraEVAsHalf)+sizeof(decltype(params)::lastFrameExtraEVAsHalf);
			auto* paramPtr = reinterpret_cast<const uint8_t*>(&params);
			driver->updateBufferRangeViaStagingBuffer(parameterBuffer.get(), offsetPastLumaHistory, sizeof(params)-offsetPastLumaHistory, paramPtr+offsetPastLumaHistory);
		}
	}

	return 0;
}

#endif