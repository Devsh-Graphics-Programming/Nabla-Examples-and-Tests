// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"

/*
	General namespaces. Entire engine consists of those bellow.
*/

using namespace nbl;
using namespace asset;
using namespace video;
using namespace core;


int main(int argc, char** argv)
{
	/*
		 SIrrlichtCreationParameters holds some specific initialization information 
		 about driver being used, size of window, stencil buffer or depth buffer.
		 Used to create a device.
	*/

	system::path CWD = system::path(argv[0]).parent_path().generic_string() + "/";
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	/*
		Most important objects to manage literally whole stuff are bellow.
		By their usage you can create for example GPU objects, load or write
		assets or manage objects on a scene.
	*/

	auto initOutput = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "NablaTutorialExample", nbl::asset::EF_D32_SFLOAT);
	auto window = std::move(initOutput.window);
	auto gl = std::move(initOutput.apiConnection);
	auto surface = std::move(initOutput.surface);
	auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
	auto logicalDevice = std::move(initOutput.logicalDevice);
	auto queues = std::move(initOutput.queues);
	auto swapchain = std::move(initOutput.swapchain);
	auto renderpass = std::move(initOutput.renderpass);
	auto fbos = std::move(initOutput.fbo);
	auto commandPool = std::move(initOutput.commandPool);
	auto assetManager = std::move(initOutput.assetManager);
	auto logger = std::move(initOutput.logger);
	auto inputSystem = std::move(initOutput.inputSystem);
	auto system = std::move(initOutput.system);
	auto windowCallback = std::move(initOutput.windowCb);
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	auto utilities = std::move(initOutput.utilities);

	auto gpuTransferFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
	auto gpuComputeFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
	{
		cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
		cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
	}

	auto createDescriptorPool = [&](const uint32_t textureCount)
	{
		constexpr uint32_t maxItemCount = 256u;
		{
			nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
			poolSize.count = textureCount;
			poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
			return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
		}
	};

	/*
		Helpfull class for managing basic geometry objects. 
		Thanks to it you can get half filled pipeline for your
		geometries such as cubes, cones or spheres.
	*/

	auto geometryCreator = assetManager->getGeometryCreator();
	auto rectangleGeometry = geometryCreator->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3));

	/*
		Loading an asset bundle. You can specify some flags 
		and parameters to have an impact on extraordinary 
		tasks while loading for example. 
	*/

	asset::IAssetLoader::SAssetLoadParams loadingParams;
	auto images_bundle = assetManager->getAsset("../../media/color_space_test/R8G8B8A8_1.png", loadingParams);
	assert(!images_bundle.getContents().empty());
	auto image = images_bundle.getContents().begin()[0];
	auto image_raw = static_cast<asset::ICPUImage*>(image.get());

	/*
		Specifing gpu image view parameters to create a gpu
		image view through the driver.
	*/

	auto gpuImage = cpu2gpu.getGPUObjectsFromAssets(&image_raw, &image_raw + 1, cpu2gpuParams)->front();
	cpu2gpuParams.waitForCreationToComplete();
	auto& gpuParams = gpuImage->getCreationParameters();

	IImageView<IGPUImage>::SCreationParams gpuImageViewParams = { static_cast<IGPUImageView::E_CREATE_FLAGS>(0), gpuImage, IImageView<IGPUImage>::ET_2D, gpuParams.format, {}, {static_cast<IImage::E_ASPECT_FLAGS>(0u), 0, gpuParams.mipLevels, 0, gpuParams.arrayLayers} };
	auto gpuImageView = logicalDevice->createGPUImageView(std::move(gpuImageViewParams));

	/*
		Specifying cache key to default exsisting cached asset bundle
		and specifying it's size where end is determined by 
		static_cast<IAsset::E_TYPE>(0u)
	*/

	const IAsset::E_TYPE types[]{ IAsset::E_TYPE::ET_SPECIALIZED_SHADER, IAsset::E_TYPE::ET_SPECIALIZED_SHADER, static_cast<IAsset::E_TYPE>(0u) };

	auto cpuVertexShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(assetManager->findAssets("nbl/builtin/material/lambertian/singletexture/specialized_shader.vert", types)->front().getContents().begin()[0]);
	auto cpuFragmentShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(assetManager->findAssets("nbl/builtin/material/lambertian/singletexture/specialized_shader.frag", types)->front().getContents().begin()[0]);

	auto gpuVertexShader = cpu2gpu.getGPUObjectsFromAssets(&cpuVertexShader.get(), &cpuVertexShader.get() + 1, cpu2gpuParams)->front();
	cpu2gpuParams.waitForCreationToComplete();
	auto gpuFragmentShader = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader.get(), &cpuFragmentShader.get() + 1, cpu2gpuParams)->front();
	cpu2gpuParams.waitForCreationToComplete();
	cpu2gpuParams.waitForCreationToComplete();
	std::array<IGPUSpecializedShader*, 2> gpuShaders = { gpuVertexShader.get(), gpuFragmentShader.get() };

	/*
		Creating helpull variables for descriptor sets.
		We are using to descriptor sets, one for the texture
		(sampler) and one for UBO holding basic view parameters.
		Each uses 0 as index of binding.
	*/

	size_t ds0SamplerBinding = 0, ds1UboBinding = 0;
	auto createAndGetUsefullData = [&](asset::IGeometryCreator::return_type& geometryObject)
	{
		/*
			SBinding for the texture (sampler). 
		*/

		IGPUDescriptorSetLayout::SBinding gpuSamplerBinding;
		gpuSamplerBinding.binding = ds0SamplerBinding;
		gpuSamplerBinding.type = EDT_COMBINED_IMAGE_SAMPLER;
		gpuSamplerBinding.count = 1u;
		gpuSamplerBinding.stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_FRAGMENT);
		gpuSamplerBinding.samplers = nullptr;	

		/*
			SBinding for UBO - basic view parameters.
		*/

		IGPUDescriptorSetLayout::SBinding gpuUboBinding;
		gpuUboBinding.count = 1u;
		gpuUboBinding.binding = ds1UboBinding;
		gpuUboBinding.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_VERTEX | asset::ICPUSpecializedShader::ESS_FRAGMENT);
		gpuUboBinding.type = asset::EDT_UNIFORM_BUFFER;

		/*
			Creating specific descriptor set layouts from specialized bindings.
			Those layouts needs to attached to pipeline layout if required by user.
			IrrlichtBaW provides 4 places for descriptor set layout usage.
		*/

		auto gpuDs1Layout = logicalDevice->createGPUDescriptorSetLayout(&gpuUboBinding, &gpuUboBinding + 1);
		auto gpuDs3Layout = logicalDevice->createGPUDescriptorSetLayout(&gpuSamplerBinding, &gpuSamplerBinding + 1);

		/*
			Creating gpu UBO with appropiate size.

			We know ahead of time that `SBasicViewParameters` struct is the expected structure of the only UBO block in the descriptor set nr. 1 of the shader.
		*/

		IGPUBuffer::SCreationParams creationParams;
		creationParams.usage = asset::IBuffer::EUF_UNIFORM_BUFFER_BIT;
		IDriverMemoryBacked::SDriverMemoryRequirements memReq;
		memReq.vulkanReqs.size = sizeof(SBasicViewParameters);
		auto gpuubo = logicalDevice->createGPUBufferOnDedMem(creationParams, memReq, true);

		/*
			Creating descriptor sets - texture (sampler) and basic view parameters (UBO).
			Specifying info and write parameters for updating certain descriptor set to the driver.

			We know ahead of time that `SBasicViewParameters` struct is the expected structure of the only UBO block in the descriptor set nr. 1 of the shader.
		*/

		auto descriptorPool = createDescriptorPool(1u);

		auto gpuDescriptorSet3 = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), gpuDs3Layout);
		{
			video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = gpuDescriptorSet3.get();
			write.binding = ds0SamplerBinding;
			write.count = 1u;
			write.arrayElement = 0u;
			write.descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
			IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = std::move(gpuImageView);
				ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETBC_FLOAT_OPAQUE_BLACK,ISampler::ETF_LINEAR,ISampler::ETF_LINEAR,ISampler::ESMM_LINEAR,0u,false,ECO_ALWAYS };
				info.image = { logicalDevice->createGPUSampler(samplerParams),EIL_SHADER_READ_ONLY_OPTIMAL };
			}
			write.info = &info;
			logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
		}

		auto gpuDescriptorSet1 = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), gpuDs1Layout);
		{
			video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = gpuDescriptorSet1.get();
			write.binding = ds1UboBinding;
			write.count = 1u;
			write.arrayElement = 0u;
			write.descriptorType = asset::EDT_UNIFORM_BUFFER;
			video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = gpuubo;
				info.buffer.offset = 0ull;
				info.buffer.size = sizeof(SBasicViewParameters);
			}
			write.info = &info;
			logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
		}

		auto gpuPipelineLayout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, nullptr, std::move(gpuDs1Layout), nullptr, std::move(gpuDs3Layout));

		/*
			Preparing required pipeline parameters and filling choosen one.
			Note that some of them are returned from geometry creator according 
			to what I mentioned in returning half pipeline parameters.
		*/

		asset::SBlendParams blendParams;
		asset::SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = asset::EFCM_NONE;

		/*
			Creating gpu pipeline with it's pipeline layout and specilized parameters.
			Attaching vertex shader and fragment shaders.
		*/

		auto gpuPipeline = logicalDevice->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), gpuShaders.data(), gpuShaders.data() + gpuShaders.size(), geometryObject.inputParams, blendParams, geometryObject.assemblyParams, rasterParams);

		nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
		graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(gpuPipeline.get());
		graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);
		auto gpuGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));

		/*
			Creating gpu meshbuffer from parameters fetched from geometry creator return value.
		*/

		constexpr auto MAX_ATTR_BUF_BINDING_COUNT = video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT;
		constexpr auto MAX_DATA_BUFFERS = MAX_ATTR_BUF_BINDING_COUNT + 1;
		core::vector<asset::ICPUBuffer*> cpubuffers;
		cpubuffers.reserve(MAX_DATA_BUFFERS);
		for (auto i = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
		{
			auto buf = geometryObject.bindings[i].buffer.get();
			if (buf)
				cpubuffers.push_back(buf);
		}
		auto cpuindexbuffer = geometryObject.indexBuffer.buffer.get();
		if (cpuindexbuffer)
			cpubuffers.push_back(cpuindexbuffer);

		auto gpubuffers = cpu2gpu.getGPUObjectsFromAssets(cpubuffers.data(), cpubuffers.data() + cpubuffers.size(), cpu2gpuParams);
		cpu2gpuParams.waitForCreationToComplete();

		asset::SBufferBinding<video::IGPUBuffer> bindings[MAX_DATA_BUFFERS];
		for (auto i = 0, j = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
		{
			if (!geometryObject.bindings[i].buffer)
				continue;
			auto buffPair = gpubuffers->operator[](j++);
			bindings[i].offset = buffPair->getOffset();
			bindings[i].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
		}
		if (cpuindexbuffer)
		{
			auto buffPair = gpubuffers->back();
			bindings[MAX_ATTR_BUF_BINDING_COUNT].offset = buffPair->getOffset();
			bindings[MAX_ATTR_BUF_BINDING_COUNT].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
		}

		auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(core::smart_refctd_ptr(gpuPipeline), nullptr, bindings, std::move(bindings[MAX_ATTR_BUF_BINDING_COUNT]));
		{
			mb->setIndexType(geometryObject.indexType);
			mb->setIndexCount(geometryObject.indexCount);
			mb->setBoundingBox(geometryObject.bbox);
		}

		return std::make_tuple(mb, gpuPipeline, gpuubo, gpuDescriptorSet1, gpuDescriptorSet3, gpuGraphicsPipeline);
	};

	auto gpuRectangle = createAndGetUsefullData(rectangleGeometry);
	auto gpuMeshBuffer = std::get<0>(gpuRectangle);
	auto gpuRenderpassIndependentPipeline = std::get<1>(gpuRectangle);
	auto gpuubo = std::get<2>(gpuRectangle);
	auto gpuDescriptorSet1 = std::get<3>(gpuRectangle);
	auto gpuDescriptorSet3 = std::get<4>(gpuRectangle);
	auto gpuGraphicsPipeline = std::get<5>(gpuRectangle);

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	core::vectorSIMDf cameraPosition(-5, 0, 0);
	matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.01, 1000);
	Camera camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);
	auto lastTime = std::chrono::system_clock::now();

	constexpr size_t NBL_FRAMES_TO_AVERAGE = 100ull;
	bool frameDataFilled = false;
	size_t frame_count = 0ull;
	double time_sum = 0;
	double dtList[NBL_FRAMES_TO_AVERAGE] = {};
	for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
		dtList[i] = 0.0;

	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];
	logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
	{
		imageAcquire[i] = logicalDevice->createSemaphore();
		renderFinished[i] = logicalDevice->createSemaphore();
	}

	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	uint32_t acquiredNextFBO = {};
	auto resourceIx = -1;

	/*
		Hot loop for rendering a scene.
	*/

	while (windowCallback->isWindowOpen())
	{
		++resourceIx;
		if (resourceIx >= FRAMES_IN_FLIGHT)
			resourceIx = 0;

		auto& commandBuffer = commandBuffers[resourceIx];
		auto& fence = frameComplete[resourceIx];

		if (fence)
			while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		auto renderStart = std::chrono::system_clock::now();
		const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - lastTime).count();
		lastTime = renderStart;
		{ // Calculate Simple Moving Average for FrameTime
			time_sum -= dtList[frame_count];
			time_sum += renderDt;
			dtList[frame_count] = renderDt;
			frame_count++;
			if (frame_count >= NBL_FRAMES_TO_AVERAGE)
			{
				frameDataFilled = true;
				frame_count = 0;
			}

		}
		const double averageFrameTime = frameDataFilled ? (time_sum / (double)NBL_FRAMES_TO_AVERAGE) : (time_sum / frame_count);

#ifdef NBL_MORE_LOGS
		logger->log("renderDt = %f ------ averageFrameTime = %f", system::ILogger::ELL_INFO, renderDt, averageFrameTime);
#endif // NBL_MORE_LOGS

		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		camera.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
		camera.endInputProcessing(nextPresentationTimeStamp);

		const auto& viewMatrix = camera.getViewMatrix();
		const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

		commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
		commandBuffer->begin(0);

		asset::SViewport viewport;
		viewport.minDepth = 1.f;
		viewport.maxDepth = 0.f;
		viewport.x = 0u;
		viewport.y = 0u;
		viewport.width = WIN_W;
		viewport.height = WIN_H;
		commandBuffer->setViewport(0u, 1u, &viewport);

		swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { WIN_W, WIN_H };
			asset::SClearValue clear[2] = {};
			clear[0].color.float32[0] = 0.f;
			clear[0].color.float32[1] = 0.f;
			clear[0].color.float32[2] = 0.f;
			clear[0].color.float32[3] = 1.f;
			clear[1].depthStencil.depth = 0.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = fbos[acquiredNextFBO];
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

		const auto viewProjection = camera.getConcatenatedMatrix();
		core::matrix3x4SIMD modelMatrix;
		modelMatrix.setRotation(nbl::core::quaternion(0, 1, 0));

		auto mv = core::concatenateBFollowedByA(camera.getViewMatrix(), modelMatrix);
		auto mvp = core::concatenateBFollowedByA(viewProjection, modelMatrix);
		core::matrix3x4SIMD normalMat;
		mv.getSub3x3InverseTranspose(normalMat);

		/*
			Updating UBO for basic view parameters and sending
			updated data to staging buffer that will redirect
			the data to graphics card - to vertex shader.
		*/

		SBasicViewParameters uboData;
		memcpy(uboData.MV, mv.pointer(), sizeof(mv));
		memcpy(uboData.MVP, mvp.pointer(), sizeof(mvp));
		memcpy(uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
		commandBuffer->updateBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), &uboData);

		/*
			Binding the most important objects needed to
			render anything on the screen with textures:

			- gpu pipeline
			- gpu descriptor sets
		*/

		commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());
		commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuDescriptorSet1.get(), nullptr);
		commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuDescriptorSet3.get(), nullptr);

		/*
			Drawing a mesh (created rectangle) with it's gpu mesh buffer usage.
		*/

		commandBuffer->drawMeshBuffer(gpuMeshBuffer.get());

		commandBuffer->endRenderPass();
		commandBuffer->end();

		CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
		CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[decltype(initOutput)::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
	}
}
