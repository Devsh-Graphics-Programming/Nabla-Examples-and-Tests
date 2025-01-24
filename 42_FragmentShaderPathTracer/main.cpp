// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "CCamera.hpp"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/video/utilities/CDumbPresentationOracle.h"

using namespace nbl;
using namespace core;
using namespace ui;


using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

smart_refctd_ptr<IGPUImageView> createHDRImageView(nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> device, asset::E_FORMAT colorFormat, uint32_t width, uint32_t height)
{
	smart_refctd_ptr<IGPUImageView> gpuImageViewColorBuffer;
	{
		IGPUImage::SCreationParams imgInfo;
		imgInfo.format = colorFormat;
		imgInfo.type = IGPUImage::ET_2D;
		imgInfo.extent.width = width;
		imgInfo.extent.height = height;
		imgInfo.extent.depth = 1u;
		imgInfo.mipLevels = 1u;
		imgInfo.arrayLayers = 1u;
		imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
		imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
		imgInfo.usage = core::bitflag(asset::IImage::EUF_STORAGE_BIT) | asset::IImage::EUF_TRANSFER_SRC_BIT;

		auto image = device->createImage(std::move(imgInfo));
		auto imageMemReqs = image->getMemoryReqs();
		imageMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		device->allocate(imageMemReqs, image.get());

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.image = std::move(image);
		imgViewInfo.format = colorFormat;
		imgViewInfo.viewType = IGPUImageView::ET_2D;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		imgViewInfo.subresourceRange.baseArrayLayer = 0u;
		imgViewInfo.subresourceRange.baseMipLevel = 0u;
		imgViewInfo.subresourceRange.layerCount = 1u;
		imgViewInfo.subresourceRange.levelCount = 1u;

		gpuImageViewColorBuffer = device->createImageView(std::move(imgViewInfo));
	}

	return gpuImageViewColorBuffer;
}

struct ShaderParameters
{
	const uint32_t MaxDepthLog2 = 4; //5
	const uint32_t MaxSamplesLog2 = 10; //18
} kShaderParameters;

enum E_LIGHT_GEOMETRY
{
	ELG_SPHERE,
	ELG_TRIANGLE,
	ELG_RECTANGLE
};

struct DispatchInfo_t
{
	uint32_t workGroupCount[3];
};

_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_SIZE = 16u;

DispatchInfo_t getDispatchInfo(uint32_t imgWidth, uint32_t imgHeight) {
	DispatchInfo_t ret = {};
	ret.workGroupCount[0] = (uint32_t)core::ceil<float>((float)imgWidth / (float)DEFAULT_WORK_GROUP_SIZE);
	ret.workGroupCount[1] = (uint32_t)core::ceil<float>((float)imgHeight / (float)DEFAULT_WORK_GROUP_SIZE);
	ret.workGroupCount[2] = 1;
	return ret;
}

int main()
{
	system::IApplicationFramework::GlobalsInit();

	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t FBO_COUNT = 2u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	constexpr bool LOG_TIMESTAMP = false;
	static_assert(FRAMES_IN_FLIGHT>FBO_COUNT);
	
	const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_TRANSFER_DST_BIT);
	CommonAPI::InitParams initParams;
	initParams.apiType = video::EAT_VULKAN;
	initParams.appName = { "Compute Shader PathTracer" };
	initParams.framesInFlight = FRAMES_IN_FLIGHT;
	initParams.windowWidth = WIN_W;
	initParams.windowHeight = WIN_H;
	initParams.swapchainImageCount = FBO_COUNT;
	initParams.swapchainImageUsage = swapchainImageUsage;
	initParams.depthFormat = asset::EF_D32_SFLOAT;
	auto initOutput = CommonAPI::InitWithDefaultExt(std::move(initParams));

	auto system = std::move(initOutput.system);
	auto window = std::move(initParams.window);
	auto windowCb = std::move(initParams.windowCb);
	auto gl = std::move(initOutput.apiConnection);
	auto surface = std::move(initOutput.surface);
	auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
	auto device = std::move(initOutput.logicalDevice);
	auto queues = std::move(initOutput.queues);
	auto graphicsQueue = queues[CommonAPI::InitOutput::EQT_GRAPHICS];
	auto transferUpQueue = queues[CommonAPI::InitOutput::EQT_TRANSFER_UP];
	auto computeQueue = queues[CommonAPI::InitOutput::EQT_COMPUTE];
	auto renderpass = std::move(initOutput.renderToSwapchainRenderpass);
	auto assetManager = std::move(initOutput.assetManager);
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	auto logger = std::move(initOutput.logger);
	auto inputSystem = std::move(initOutput.inputSystem);
	auto utilities = std::move(initOutput.utilities);
	auto graphicsCommandPools = std::move(initOutput.commandPools[CommonAPI::InitOutput::EQT_GRAPHICS]);
	auto computeCommandPools = std::move(initOutput.commandPools[CommonAPI::InitOutput::EQT_COMPUTE]);
	auto swapchainCreationParams = std::move(initOutput.swapchainCreationParams);

	core::smart_refctd_ptr<video::ISwapchain> swapchain = nullptr;
	CommonAPI::createSwapchain(std::move(device), swapchainCreationParams, WIN_W, WIN_H, swapchain);
	assert(swapchain);
	auto fbo = CommonAPI::createFBOWithSwapchainImages(
		swapchain->getImageCount(), WIN_W, WIN_H,
		device, swapchain, renderpass,
		asset::EF_D32_SFLOAT
	);

	auto graphicsCmdPoolQueueFamIdx = graphicsQueue->getFamilyIndex();

	nbl::video::IGPUObjectFromAssetConverter CPU2GPU;
	
	core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf[FRAMES_IN_FLIGHT];
	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		device->createCommandBuffers(graphicsCommandPools[i].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1, cmdbuf+i);	

	constexpr uint32_t maxDescriptorCount = 256u;
	constexpr uint32_t PoolSizesCount = 5u;

	nbl::video::IDescriptorPool::SCreateInfo createInfo;
	createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] = maxDescriptorCount * 1;
	createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)] = maxDescriptorCount * 8;
	createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)] = maxDescriptorCount * 2;
	createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER)] = maxDescriptorCount * 1;
	createInfo.maxDescriptorCount[static_cast<uint32_t>(nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] = maxDescriptorCount * 1;
	createInfo.maxSets = maxDescriptorCount;

	auto descriptorPool = device->createDescriptorPool(std::move(createInfo));

	const auto timestampQueryPool = device->createQueryPool({
		.queryType = video::IQueryPool::EQT_TIMESTAMP,
		.queryCount = 2u
	});

	// Camera 
	core::vectorSIMDf cameraPosition(0, 5, -10);
	matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(60.0f), video::ISurface::getTransformedAspectRatio(swapchain->getPreTransform(), WIN_W, WIN_H), 0.01f, 500.0f);
	Camera cam = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), proj);

	IGPUDescriptorSetLayout::SBinding descriptorSet0Bindings[] = {
		{ 0u, nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, IShader::ESS_COMPUTE, 1u, nullptr },
	};
	IGPUDescriptorSetLayout::SBinding uboBinding
	{ 0u, nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, IShader::ESS_COMPUTE, 1u, nullptr };
	IGPUDescriptorSetLayout::SBinding descriptorSet3Bindings[] = {
		{ 0u, nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, IShader::ESS_COMPUTE, 1u, nullptr },
		{ 1u, nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, IShader::ESS_COMPUTE, 1u, nullptr },
		{ 2u, nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, IShader::ESS_COMPUTE, 1u, nullptr },
	};
	
	auto gpuDescriptorSetLayout0 = device->createDescriptorSetLayout(descriptorSet0Bindings, descriptorSet0Bindings + 1u);
	auto gpuDescriptorSetLayout1 = device->createDescriptorSetLayout(&uboBinding, &uboBinding + 1u);
	auto gpuDescriptorSetLayout2 = device->createDescriptorSetLayout(descriptorSet3Bindings, descriptorSet3Bindings+3u);

	auto createGpuResources = [&](std::string pathToShader) -> core::smart_refctd_ptr<video::IGPUComputePipeline>
	{
		asset::IAssetLoader::SAssetLoadParams params{};
		params.logger = logger.get();
		//params.relativeDir = tmp.c_str();
		auto spec = assetManager->getAsset(pathToShader,params).getContents();
		
		if (spec.empty())
			assert(false);

		auto cpuComputeSpecializedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*spec.begin());

		ISpecializedShader::SInfo info = cpuComputeSpecializedShader->getSpecializationInfo();
		info.m_backingBuffer = ICPUBuffer::create({ sizeof(ShaderParameters) });
		memcpy(info.m_backingBuffer->getPointer(),&kShaderParameters,sizeof(ShaderParameters));
		info.m_entries = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ISpecializedShader::SInfo::SMapEntry>>(2u);
		for (uint32_t i=0; i<2; i++)
			info.m_entries->operator[](i) = {i,(uint32_t)(i*sizeof(uint32_t)),sizeof(uint32_t)};


		cpuComputeSpecializedShader->setSpecializationInfo(std::move(info));

		auto gpuComputeSpecializedShader = CPU2GPU.getGPUObjectsFromAssets(&cpuComputeSpecializedShader, &cpuComputeSpecializedShader + 1, cpu2gpuParams)->front();

		auto gpuPipelineLayout = device->createPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout0), core::smart_refctd_ptr(gpuDescriptorSetLayout1), core::smart_refctd_ptr(gpuDescriptorSetLayout2), nullptr);

		auto gpuPipeline = device->createComputePipeline(nullptr, std::move(gpuPipelineLayout), std::move(gpuComputeSpecializedShader));

		return gpuPipeline;
	};

	E_LIGHT_GEOMETRY lightGeom = ELG_SPHERE;
	constexpr const char* shaderPaths[] = {"../litBySphere.comp","../litByTriangle.comp","../litByRectangle.comp"};
	auto gpuComputePipeline = createGpuResources(shaderPaths[lightGeom]);

	DispatchInfo_t dispatchInfo = getDispatchInfo(WIN_W, WIN_H);

	auto createImageView = [&](std::string pathToOpenEXRHDRIImage)
	{
#ifndef _NBL_COMPILE_WITH_OPENEXR_LOADER_
		assert(false);
#endif

		auto pathToTexture = pathToOpenEXRHDRIImage;
		IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
		auto cpuTexture = assetManager->getAsset(pathToTexture, lp);
		auto cpuTextureContents = cpuTexture.getContents();
		assert(!cpuTextureContents.empty());
		auto cpuImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(*cpuTextureContents.begin());
		cpuImage->setImageUsageFlags(IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT);

		ICPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = cpuImage;
		viewParams.format = viewParams.image->getCreationParameters().format;
		viewParams.viewType = IImageView<ICPUImage>::ET_2D;
		viewParams.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = 1u;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = 1u;

		auto cpuImageView = ICPUImageView::create(std::move(viewParams));

		cpu2gpuParams.beginCommandBuffers();
		auto gpuImageView = CPU2GPU.getGPUObjectsFromAssets(&cpuImageView, &cpuImageView + 1u, cpu2gpuParams)->front();
		cpu2gpuParams.waitForCreationToComplete(false);

		return gpuImageView;
	};
	
	auto gpuEnvmapImageView = createImageView("../../media/envmap/envmap_0.exr");

	smart_refctd_ptr<IGPUBufferView> gpuSequenceBufferView;
	{
		const uint32_t MaxDimensions = 3u<<kShaderParameters.MaxDepthLog2;
		const uint32_t MaxSamples = 1u<<kShaderParameters.MaxSamplesLog2;

		auto sampleSequence = core::make_smart_refctd_ptr<asset::({ sizeof(uint32_t)*MaxDimensions*MaxSamples });
		
		core::OwenSampler sampler(MaxDimensions, 0xdeadbeefu);
		//core::SobolSampler sampler(MaxDimensions);

		auto out = reinterpret_cast<uint32_t*>(sampleSequence->getPointer());
		for (auto dim=0u; dim<MaxDimensions; dim++)
		for (uint32_t i=0; i<MaxSamples; i++)
		{
			out[i*MaxDimensions+dim] = sampler.sample(dim,i);
		}
		
		// TODO: Temp Fix because createFilledDeviceLocalBufferOnDedMem doesn't take in params
		// auto gpuSequenceBuffer = utilities->createFilledDeviceLocalBufferOnDedMem(graphicsQueue, sampleSequence->getSize(), sampleSequence->getPointer());
		core::smart_refctd_ptr<IGPUBuffer> gpuSequenceBuffer;
		{
			IGPUBuffer::SCreationParams params = {};
			const size_t size = sampleSequence->getSize();
			params.usage = core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT) | asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT; 
			params.size = size;
			gpuSequenceBuffer = device->createBuffer(std::move(params));
			auto gpuSequenceBufferMemReqs = gpuSequenceBuffer->getMemoryReqs();
			gpuSequenceBufferMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			device->allocate(gpuSequenceBufferMemReqs, gpuSequenceBuffer.get());
			utilities->updateBufferRangeViaStagingBufferAutoSubmit(asset::SBufferRange<IGPUBuffer>{0u,size,gpuSequenceBuffer},sampleSequence->getPointer(), graphicsQueue);
		}
		gpuSequenceBufferView = device->createBufferView(gpuSequenceBuffer.get(), asset::EF_R32G32B32_UINT);
	}

	smart_refctd_ptr<IGPUImageView> gpuScrambleImageView;
	{
		IGPUImage::SCreationParams imgParams;
		imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
		imgParams.type = IImage::ET_2D;
		imgParams.format = EF_R32G32_UINT;
		imgParams.extent = {WIN_W, WIN_H,1u};
		imgParams.mipLevels = 1u;
		imgParams.arrayLayers = 1u;
		imgParams.samples = IImage::ESCF_1_BIT;
		imgParams.usage = core::bitflag(IImage::EUF_SAMPLED_BIT) | IImage::EUF_TRANSFER_DST_BIT;
		imgParams.initialLayout = asset::IImage::EL_UNDEFINED;

		IGPUImage::SBufferCopy region = {};
		region.bufferOffset = 0u;
		region.bufferRowLength = 0u;
		region.bufferImageHeight = 0u;
		region.imageExtent = imgParams.extent;
		region.imageOffset = {0u,0u,0u};
		region.imageSubresource.layerCount = 1u;
		region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;

		constexpr auto ScrambleStateChannels = 2u;
		const auto renderPixelCount = imgParams.extent.width*imgParams.extent.height;
		core::vector<uint32_t> random(renderPixelCount*ScrambleStateChannels);
		{
			core::RandomSampler rng(0xbadc0ffeu);
			for (auto& pixel : random)
				pixel = rng.nextSample();
		}

		// TODO: Temp Fix because createFilledDeviceLocalBufferOnDedMem doesn't take in params
		// auto buffer = utilities->createFilledDeviceLocalBufferOnDedMem(graphicsQueue, random.size()*sizeof(uint32_t), random.data());
		core::smart_refctd_ptr<IGPUBuffer> buffer;
		{
			IGPUBuffer::SCreationParams params = {};
			const size_t size = random.size() * sizeof(uint32_t);
			params.usage = core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT) | asset::IBuffer::EUF_TRANSFER_SRC_BIT; 
			params.size = size;
			buffer = device->createBuffer(std::move(params));
			auto bufferMemReqs = buffer->getMemoryReqs();
			bufferMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			device->allocate(bufferMemReqs, buffer.get());
			utilities->updateBufferRangeViaStagingBufferAutoSubmit(asset::SBufferRange<IGPUBuffer>{0u,size,buffer},random.data(),graphicsQueue);
		}

		IGPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		// TODO: Replace this IGPUBuffer -> IGPUImage to using image upload utility
		viewParams.image = utilities->createFilledDeviceLocalImageOnDedMem(std::move(imgParams), buffer.get(), 1u, &region, graphicsQueue);
		viewParams.viewType = IGPUImageView::ET_2D;
		viewParams.format = EF_R32G32_UINT;
		viewParams.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		viewParams.subresourceRange.levelCount = 1u;
		viewParams.subresourceRange.layerCount = 1u;
		gpuScrambleImageView = device->createImageView(std::move(viewParams));
	}
	
	// Create Out Image TODO
	constexpr uint32_t MAX_FBO_COUNT = 4u;
	smart_refctd_ptr<IGPUImageView> outHDRImageViews[MAX_FBO_COUNT] = {};
	assert(MAX_FBO_COUNT >= swapchain->getImageCount());
	for(uint32_t i = 0; i < swapchain->getImageCount(); ++i) {
		outHDRImageViews[i] = createHDRImageView(device, asset::EF_R16G16B16A16_SFLOAT, WIN_W, WIN_H);
	}

	core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSets0[FBO_COUNT] = {};
	for(uint32_t i = 0; i < FBO_COUNT; ++i)
	{
		auto & descSet = descriptorSets0[i];
		descSet = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout0));
		video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSet;
		writeDescriptorSet.dstSet = descSet.get();
		writeDescriptorSet.binding = 0;
		writeDescriptorSet.count = 1u;
		writeDescriptorSet.arrayElement = 0u;
		writeDescriptorSet.descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = outHDRImageViews[i];
			info.info.image.sampler = nullptr;
			info.info.image.imageLayout = asset::IImage::EL_GENERAL;
		}
		writeDescriptorSet.info = &info;
		device->updateDescriptorSets(1u, &writeDescriptorSet, 0u, nullptr);
	}
	
	struct SBasicViewParametersAligned
	{
		SBasicViewParameters uboData;
	};

	IGPUBuffer::SCreationParams gpuuboParams = {};
	gpuuboParams.usage = core::bitflag(IGPUBuffer::EUF_UNIFORM_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuuboParams.size = sizeof(SBasicViewParametersAligned);
	auto gpuubo = device->createBuffer(std::move(gpuuboParams));
	auto gpuuboMemReqs = gpuubo->getMemoryReqs();
	gpuuboMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	device->allocate(gpuuboMemReqs, gpuubo.get());

	auto uboDescriptorSet1 = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout1));
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet uboWriteDescriptorSet;
		uboWriteDescriptorSet.dstSet = uboDescriptorSet1.get();
		uboWriteDescriptorSet.binding = 0;
		uboWriteDescriptorSet.count = 1u;
		uboWriteDescriptorSet.arrayElement = 0u;
		uboWriteDescriptorSet.descriptorType = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = gpuubo;
			info.info.buffer.offset = 0ull;
			info.info.buffer.size = sizeof(SBasicViewParametersAligned);
		}
		uboWriteDescriptorSet.info = &info;
		device->updateDescriptorSets(1u, &uboWriteDescriptorSet, 0u, nullptr);
	}

	ISampler::SParams samplerParams0 = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
	auto sampler0 = device->createSampler(samplerParams0);
	ISampler::SParams samplerParams1 = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
	auto sampler1 = device->createSampler(samplerParams1);
	
	auto descriptorSet2 = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout2));
	{
		constexpr auto kDescriptorCount = 3;
		IGPUDescriptorSet::SWriteDescriptorSet samplerWriteDescriptorSet[kDescriptorCount];
		IGPUDescriptorSet::SDescriptorInfo samplerDescriptorInfo[kDescriptorCount];
		for (auto i=0; i<kDescriptorCount; i++)
		{
			samplerWriteDescriptorSet[i].dstSet = descriptorSet2.get();
			samplerWriteDescriptorSet[i].binding = i;
			samplerWriteDescriptorSet[i].arrayElement = 0u;
			samplerWriteDescriptorSet[i].count = 1u;
			samplerWriteDescriptorSet[i].info = samplerDescriptorInfo+i;
		}
		samplerWriteDescriptorSet[0].descriptorType = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER;
		samplerWriteDescriptorSet[1].descriptorType = nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER;
		samplerWriteDescriptorSet[2].descriptorType = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER;

		samplerDescriptorInfo[0].desc = gpuEnvmapImageView;
		{
			// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
			samplerDescriptorInfo[0].info.image.sampler = sampler0;
			samplerDescriptorInfo[0].info.image.imageLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
		}
		samplerDescriptorInfo[1].desc = gpuSequenceBufferView;
		samplerDescriptorInfo[2].desc = gpuScrambleImageView;
		{
			// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
			samplerDescriptorInfo[2].info.image.sampler = sampler1;
			samplerDescriptorInfo[2].info.image.imageLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
		}

		device->updateDescriptorSets(kDescriptorCount, samplerWriteDescriptorSet, 0u, nullptr);
	}

	constexpr uint32_t FRAME_COUNT = 500000u;

	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	for (uint32_t i=0u; i<FRAMES_IN_FLIGHT; i++)
	{
		imageAcquire[i] = device->createSemaphore();
		renderFinished[i] = device->createSemaphore();
	}
	
	CDumbPresentationOracle oracle;
	oracle.reportBeginFrameRecord();
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	
	// polling for events!
	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	
	uint32_t resourceIx = 0;
	while(windowCb->isWindowOpen())
	{
		resourceIx++;
		if(resourceIx >= FRAMES_IN_FLIGHT) {
			resourceIx = 0;
		}
		
		oracle.reportEndFrameRecord();
		double dt = oracle.getDeltaTimeInMicroSeconds() / 1000.0;
		auto nextPresentationTimeStamp = oracle.getNextPresentationTimeStamp();
		oracle.reportBeginFrameRecord();

		// Input 
		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		cam.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { cam.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { cam.keyboardProcess(events); }, logger.get());
		cam.endInputProcessing(nextPresentationTimeStamp);
		
		auto& cb = cmdbuf[resourceIx];
		auto& fence = frameComplete[resourceIx];
		if (fence)
		while (device->waitForFences(1u,&fence.get(),false,MAX_TIMEOUT)==video::IGPUFence::ES_TIMEOUT)
		{
		}
		else
			fence = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		
		const auto viewMatrix = cam.getViewMatrix();
		const auto viewProjectionMatrix = matrix4SIMD::concatenateBFollowedByAPrecisely(
			video::ISurface::getSurfaceTransformationMatrix(swapchain->getPreTransform()),
			cam.getConcatenatedMatrix()
		);
				
		// safe to proceed
		cb->begin(IGPUCommandBuffer::EU_NONE);
		cb->resetQueryPool(timestampQueryPool.get(), 0u, 2u);

		// renderpass 
		uint32_t imgnum = 0u;
		swapchain->acquireNextImage(MAX_TIMEOUT,imageAcquire[resourceIx].get(),nullptr,&imgnum);
		{
			auto mv = viewMatrix;
			auto mvp = viewProjectionMatrix;
			core::matrix3x4SIMD normalMat;
			mv.getSub3x3InverseTranspose(normalMat);

			SBasicViewParametersAligned viewParams;
			memcpy(viewParams.uboData.MV, mv.pointer(), sizeof(mv));
			memcpy(viewParams.uboData.MVP, mvp.pointer(), sizeof(mvp));
			memcpy(viewParams.uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
			
			asset::SBufferRange<video::IGPUBuffer> range;
			range.buffer = gpuubo;
			range.offset = 0ull;
			range.size = sizeof(viewParams);
			utilities->updateBufferRangeViaStagingBufferAutoSubmit(range, &viewParams, graphicsQueue);
		}
				
		// TRANSITION outHDRImageViews[imgnum] to EIL_GENERAL (because of descriptorSets0 -> ComputeShader Writes into the image)
		{
			IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[3u] = {};
			imageBarriers[0].barrier.srcAccessMask = asset::EAF_NONE;
			imageBarriers[0].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_WRITE_BIT);
			imageBarriers[0].oldLayout = asset::IImage::EL_UNDEFINED;
			imageBarriers[0].newLayout = asset::IImage::EL_GENERAL;
			imageBarriers[0].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].image = outHDRImageViews[imgnum]->getCreationParameters().image;
			imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;

			imageBarriers[1].barrier.srcAccessMask = asset::EAF_NONE;
			imageBarriers[1].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT);
			imageBarriers[1].oldLayout = asset::IImage::EL_UNDEFINED;
			imageBarriers[1].newLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
			imageBarriers[1].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[1].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[1].image = gpuScrambleImageView->getCreationParameters().image;
			imageBarriers[1].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[1].subresourceRange.baseMipLevel = 0u;
			imageBarriers[1].subresourceRange.levelCount = 1;
			imageBarriers[1].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[1].subresourceRange.layerCount = 1;

			 imageBarriers[2].barrier.srcAccessMask = asset::EAF_NONE;
			 imageBarriers[2].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT);
			 imageBarriers[2].oldLayout = asset::IImage::EL_UNDEFINED;
			 imageBarriers[2].newLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
			 imageBarriers[2].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			 imageBarriers[2].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			 imageBarriers[2].image = gpuEnvmapImageView->getCreationParameters().image;
			 imageBarriers[2].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			 imageBarriers[2].subresourceRange.baseMipLevel = 0u;
			 imageBarriers[2].subresourceRange.levelCount = gpuEnvmapImageView->getCreationParameters().subresourceRange.levelCount;
			 imageBarriers[2].subresourceRange.baseArrayLayer = 0u;
			 imageBarriers[2].subresourceRange.layerCount = gpuEnvmapImageView->getCreationParameters().subresourceRange.layerCount;

			cb->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 3u, imageBarriers);
		}

		// cube envmap handle
		{
			cb->writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS::EPSF_TOP_OF_PIPE_BIT, timestampQueryPool.get(), 0u);
			cb->bindComputePipeline(gpuComputePipeline.get());
			cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 0u, 1u, &descriptorSets0[imgnum].get());
			cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 1u, 1u, &uboDescriptorSet1.get());
			cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 2u, 1u, &descriptorSet2.get());
			cb->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);
			cb->writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS::EPSF_BOTTOM_OF_PIPE_BIT, timestampQueryPool.get(), 1u);
		}
		// TODO: tone mapping and stuff

		// Copy HDR Image to SwapChain
		auto srcImgViewCreationParams = outHDRImageViews[imgnum]->getCreationParameters();
		auto dstImgViewCreationParams = fbo->begin()[imgnum]->getCreationParameters().attachments[0]->getCreationParameters();
		
		// Getting Ready for Blit
		// TRANSITION outHDRImageViews[imgnum] to EIL_TRANSFER_SRC_OPTIMAL
		// TRANSITION `fbo[imgnum]->getCreationParameters().attachments[0]` to EIL_TRANSFER_DST_OPTIMAL
		{
			IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[2u] = {};
			imageBarriers[0].barrier.srcAccessMask = asset::EAF_NONE;
			imageBarriers[0].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			imageBarriers[0].oldLayout = asset::IImage::EL_UNDEFINED;
			imageBarriers[0].newLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
			imageBarriers[0].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].image = srcImgViewCreationParams.image;
			imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;

			imageBarriers[1].barrier.srcAccessMask = asset::EAF_NONE;
			imageBarriers[1].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			imageBarriers[1].oldLayout = asset::IImage::EL_UNDEFINED;
			imageBarriers[1].newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
			imageBarriers[1].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[1].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[1].image = dstImgViewCreationParams.image;
			imageBarriers[1].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[1].subresourceRange.baseMipLevel = 0u;
			imageBarriers[1].subresourceRange.levelCount = 1;
			imageBarriers[1].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[1].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 2u, imageBarriers);
		}

		// Blit Image
		{
			SImageBlit blit = {};
			blit.srcOffsets[0] = {0, 0, 0};
			blit.srcOffsets[1] = {WIN_W, WIN_H, 1};
		
			blit.srcSubresource.aspectMask = srcImgViewCreationParams.subresourceRange.aspectMask;
			blit.srcSubresource.mipLevel = srcImgViewCreationParams.subresourceRange.baseMipLevel;
			blit.srcSubresource.baseArrayLayer = srcImgViewCreationParams.subresourceRange.baseArrayLayer;
			blit.srcSubresource.layerCount = srcImgViewCreationParams.subresourceRange.layerCount;
			blit.dstOffsets[0] = {0, 0, 0};
			blit.dstOffsets[1] = {WIN_W, WIN_H, 1};
			blit.dstSubresource.aspectMask = dstImgViewCreationParams.subresourceRange.aspectMask;
			blit.dstSubresource.mipLevel = dstImgViewCreationParams.subresourceRange.baseMipLevel;
			blit.dstSubresource.baseArrayLayer = dstImgViewCreationParams.subresourceRange.baseArrayLayer;
			blit.dstSubresource.layerCount = dstImgViewCreationParams.subresourceRange.layerCount;

			auto srcImg = srcImgViewCreationParams.image;
			auto dstImg = dstImgViewCreationParams.image;

			cb->blitImage(srcImg.get(), asset::IImage::EL_TRANSFER_SRC_OPTIMAL, dstImg.get(), asset::IImage::EL_TRANSFER_DST_OPTIMAL, 1u, &blit , ISampler::ETF_NEAREST);
		}
		
		// TRANSITION `fbo[imgnum]->getCreationParameters().attachments[0]` to EIL_PRESENT
		{
			IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
			imageBarriers[0].barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			imageBarriers[0].barrier.dstAccessMask = asset::EAF_NONE;
			imageBarriers[0].oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
			imageBarriers[0].newLayout = asset::IImage::EL_PRESENT_SRC;
			imageBarriers[0].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].image = dstImgViewCreationParams.image;
			imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TOP_OF_PIPE_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);
		}

		cb->end();
		device->resetFences(1, &fence.get());
		CommonAPI::Submit(device.get(), cb.get(), graphicsQueue, imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
		CommonAPI::Present(device.get(), swapchain.get(), graphicsQueue, renderFinished[resourceIx].get(), imgnum);
		
		if (LOG_TIMESTAMP)
		{
			std::array<uint64_t, 4> timestamps{};
			auto queryResultFlags = core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS>(video::IQueryPool::EQRF_WAIT_BIT) | video::IQueryPool::EQRF_WITH_AVAILABILITY_BIT | video::IQueryPool::EQRF_64_BIT;
			device->getQueryPoolResults(timestampQueryPool.get(), 0u, 2u, sizeof(timestamps), timestamps.data(), sizeof(uint64_t) * 2ull, queryResultFlags);
			const float timePassed = (timestamps[2] - timestamps[0]) * device->getPhysicalDevice()->getLimits().timestampPeriodInNanoSeconds;
			logger->log("Time Passed (Seconds) = %f", system::ILogger::ELL_INFO, (timePassed * 1e-9));
			logger->log("Timestamps availablity: %d, %d", system::ILogger::ELL_INFO, timestamps[1], timestamps[3]);
		}
	}
	
	const auto& fboCreationParams = fbo->begin()[0]->getCreationParameters();
	auto gpuSourceImageView = fboCreationParams.attachments[0];

	device->waitIdle();

	// bool status = ext::ScreenShot::createScreenShot(device.get(), queues[decltype(initOutput)::EQT_TRANSFER_UP], renderFinished[0].get(), gpuSourceImageView.get(), assetManager.get(), "ScreenShot.png");
	// assert(status);

	return 0;
}
