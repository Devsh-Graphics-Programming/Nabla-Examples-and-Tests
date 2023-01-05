#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

struct double4x4
{
	double _r1[4u];
	double _r2[4u];
	double _r3[4u];
	double _r4[4u];
};

struct double2
{
	double v[2];
};

struct uint2
{
	uint32_t v[2];
};

#define uint uint32_t
#define float4 nbl::core::vectorSIMDf
#include "common.hlsl"

static_assert(sizeof(DrawObject) == 16u);
static_assert(sizeof(LinePoints) == 64u);
static_assert(sizeof(EllipseInfo) == 48u);
static_assert(sizeof(Globals) == 160u);

using namespace nbl;

class Camera2D : public core::IReferenceCounted
{
public:
private:
};

class CADApp : public ApplicationBase
{
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	constexpr static uint32_t WIN_W = 1280u;
	constexpr static uint32_t WIN_H = 720u;

	core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	nbl::core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>> framebuffersDynArraySmartPtr;
	std::array<std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxFramesInFlight>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;
	core::smart_refctd_ptr<video::IGPUImage> m_swapchainImages[CommonAPI::InitOutput::MaxSwapChainImageCount];

	int32_t m_resourceIx = -1;

	core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf[FRAMES_IN_FLIGHT] = { nullptr };

	nbl::video::ISwapchain::SCreationParams m_swapchainCreationParams;

	// Related to Drawing Stuff
	uint32_t currentDrawObjectCount = 0u;
	uint64_t currentGeometryBufferAddress = 0u;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> m_device;
	core::smart_refctd_ptr<video::IGPUBuffer> indexBuffer;
	core::smart_refctd_ptr<video::IGPUBuffer> drawObjectsBuffer;
	core::smart_refctd_ptr<video::IGPUBuffer> geometryBuffer;
	core::smart_refctd_ptr<video::IGPUBuffer> globalsBuffer;

	constexpr size_t getMaxMemoryNeeded(uint32_t numberOfLines, uint32_t numberOfEllipses)
	{
		size_t mem = sizeof(Globals);
		uint32_t allObjectsCount = numberOfLines + numberOfEllipses;
		mem += allObjectsCount * 6u * sizeof(uint32_t); // Index Buffer 6 indices per object cage
		mem += allObjectsCount * sizeof(DrawObject); // One DrawObject struct per object
		mem += numberOfLines * 4u * sizeof(double2); // 4 points per line max (generated before/after for calculations)
		mem += numberOfEllipses * sizeof(EllipseInfo);
		return mem;
	}

	void initDrawObjects(uint32_t maxObjects = 128u)
	{
		{
			size_t indexBufferSize = maxObjects * 6u * sizeof(uint32_t);
			video::IGPUBuffer::SCreationParams indexBufferCreationParams = {};
			indexBufferCreationParams.size = indexBufferSize;
			indexBufferCreationParams.usage = core::bitflag(video::IGPUBuffer::EUF_INDEX_BUFFER_BIT) | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
			indexBuffer = logicalDevice->createBuffer(std::move(indexBufferCreationParams));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = indexBuffer->getMemoryReqs();
			memReq.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
			auto indexBufferMem = logicalDevice->allocate(memReq, indexBuffer.get());

			std::vector<uint32_t> indices(maxObjects * 6u);
			for (uint32_t i = 0u; i < maxObjects; ++i)
			{
				indices[i * 6]		= i * 4u + 0u;
				indices[i * 6 + 1u] = i * 4u + 1u;
				indices[i * 6 + 2u] = i * 4u + 2u;
				indices[i * 6 + 3u] = i * 4u + 2u;
				indices[i * 6 + 4u] = i * 4u + 1u;
				indices[i * 6 + 5u] = i * 4u + 3u;
			}

			asset::SBufferRange<video::IGPUBuffer> rangeToUpload = {0ull, indexBufferSize, indexBuffer};
			utilities->updateBufferRangeViaStagingBufferAutoSubmit(rangeToUpload, indices.data(), queues[CommonAPI::InitOutput::EQT_GRAPHICS]);
		}

		{
			size_t drawObjectsBufferSize = maxObjects * sizeof(DrawObject);
			video::IGPUBuffer::SCreationParams drawObjectsCreationParams = {};
			drawObjectsCreationParams.size = drawObjectsBufferSize;
			drawObjectsCreationParams.usage = core::bitflag(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
			drawObjectsBuffer = logicalDevice->createBuffer(std::move(drawObjectsCreationParams));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = drawObjectsBuffer->getMemoryReqs();
			memReq.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
			auto drawObjectsBufferMem = logicalDevice->allocate(memReq, drawObjectsBuffer.get());
		}

		{
			size_t geometryBufferSize = maxObjects * sizeof(EllipseInfo);
			video::IGPUBuffer::SCreationParams geometryCreationParams = {};
			geometryCreationParams.size = geometryBufferSize;
			geometryCreationParams.usage = core::bitflag(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
			geometryBuffer = logicalDevice->createBuffer(std::move(geometryCreationParams));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = geometryBuffer->getMemoryReqs();
			memReq.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
			auto geometryBufferMem = logicalDevice->allocate(memReq, geometryBuffer.get(), video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
			currentGeometryBufferAddress = logicalDevice->getBufferDeviceAddress(geometryBuffer.get());
		}

		{
			size_t globalsBufferSize = maxObjects * sizeof(EllipseInfo);
			video::IGPUBuffer::SCreationParams globalsCreationParams = {};
			globalsCreationParams.size = globalsBufferSize;
			globalsCreationParams.usage = core::bitflag(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
			globalsBuffer = logicalDevice->createBuffer(std::move(globalsCreationParams));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = globalsBuffer->getMemoryReqs();
			memReq.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
			auto globalsBufferMem = logicalDevice->allocate(memReq, globalsBuffer.get());
		}
	}

public:
	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) override
	{
		system = std::move(system);
	}
	video::IAPIConnection* getAPIConnection() override
	{
		return apiConnection.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return logicalDevice.get();
	}
	video::IGPURenderpass* getRenderpass() override
	{
		return renderpass.get();
	}
	void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
	{
		surface = std::move(s);
	}
	void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
	{
		for (int i = 0; i < f.size(); i++)
		{
			auto& fboDynArray = *(framebuffersDynArraySmartPtr.get());
			fboDynArray[i] = core::smart_refctd_ptr(f[i]);
		}
	}
	void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
	{
		swapchain = std::move(s);
	}
	uint32_t getSwapchainImageCount() override
	{
		return swapchain->getImageCount();
	}
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_D32_SFLOAT;
	}

	APP_CONSTRUCTOR(CADApp);

	void onAppInitialized_impl() override
	{
		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT);
		std::array<asset::E_FORMAT, 1> acceptableSurfaceFormats = { asset::EF_B8G8R8A8_UNORM };

		CommonAPI::InitParams initParams;
		initParams.windowCb = core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback>(this);
		initParams.window = core::smart_refctd_ptr(window);
		initParams.apiType = video::EAT_VULKAN;
		initParams.appName = { "62.CAD" };
		initParams.framesInFlight = FRAMES_IN_FLIGHT;
		initParams.windowWidth = WIN_W;
		initParams.windowHeight = WIN_H;
		initParams.swapchainImageCount = 3u;
		initParams.swapchainImageUsage = swapchainImageUsage;
		initParams.depthFormat = getDepthFormat();
		initParams.acceptableSurfaceFormats = acceptableSurfaceFormats.data();
		initParams.acceptableSurfaceFormatCount = acceptableSurfaceFormats.size();
		initParams.physicalDeviceFilter.requiredFeatures.bufferDeviceAddress = true;
		auto initOutput = CommonAPI::InitWithDefaultExt(std::move(initParams));

		system = std::move(initOutput.system);
		window = std::move(initParams.window);
		windowCb = std::move(initParams.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		physicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		utilities = std::move(initOutput.utilities);
		queues = std::move(initOutput.queues);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);
		windowManager = std::move(initOutput.windowManager);
		renderpass = std::move(initOutput.renderToSwapchainRenderpass);
		m_swapchainCreationParams = std::move(initOutput.swapchainCreationParams);

		CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, initParams.windowWidth, initParams.windowHeight, swapchain);

		commandPools = std::move(initOutput.commandPools);
		const auto& graphicsCommandPools = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS];

		const uint32_t swapchainImageCount = swapchain->getImageCount();
		framebuffersDynArraySmartPtr = CommonAPI::createFBOWithSwapchainImages(
			swapchain->getImageCount(), WIN_W, WIN_H,
			logicalDevice, swapchain, renderpass,
			getDepthFormat()
		);

		for (uint32_t i = 0; i < swapchainImageCount; ++i)
		{
			auto& fboDynArray = *(framebuffersDynArraySmartPtr.get());
			m_swapchainImages[i] = fboDynArray[i]->getCreationParameters().attachments[0u]->getCreationParameters().image;
		}

		video::IGPUObjectFromAssetConverter CPU2GPU;

		// Used to load SPIR-V directly, if HLSL Compiler doesn't work
		auto loadSPIRVShader = [&](const std::string& filePath, asset::IShader::E_SHADER_STAGE stage)
		{
			system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> shader_future;
			system->createFile(shader_future, filePath, core::bitflag(nbl::system::IFile::ECF_READ));
			auto shader_file = shader_future.get();
			auto shaderSizeInBytes = shader_file->getSize();
			auto vertexShaderSPIRVBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(shaderSizeInBytes);
			system::IFile::success_t succ;
			shader_file->read(succ, vertexShaderSPIRVBuffer->getPointer(), 0u, shaderSizeInBytes);
			const bool success = bool(succ);
			assert(success);
			return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(vertexShaderSPIRVBuffer), stage, asset::IShader::E_CONTENT_TYPE::ECT_SPIRV, std::string(filePath));
		};

		core::smart_refctd_ptr<video::IGPUSpecializedShader> shaders[2u] = {};
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			core::smart_refctd_ptr<asset::ICPUSpecializedShader> cpuShaders[2u] = {};
			cpuShaders[0u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset("../vertex_shader.hlsl", params).getContents().begin());
			cpuShaders[1u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset("../fragment_shader.hlsl", params).getContents().begin());
			cpuShaders[0u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "VSMain"));
			cpuShaders[1u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "PSMain"));
			auto gpuShaders = CPU2GPU.getGPUObjectsFromAssets(cpuShaders, cpuShaders + 2u, cpu2gpuParams);
			shaders[0u] = gpuShaders->begin()[0u];
			shaders[1u] = gpuShaders->begin()[1u];
		}

		initDrawObjects();

		// TODO:
		// Create DescriptorSetLayout
		// Create DescriptorSets
		// Create PipelineLayout from DescriptorSetLayout
		// Create Pipeline with correct params
		// Create OrthoCamera


		for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
		{
			logicalDevice->createCommandBuffers(
				graphicsCommandPools[i].get(),
				video::IGPUCommandBuffer::EL_PRIMARY,
				1,
				m_cmdbuf + i);
		}

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			m_frameComplete[i] = logicalDevice->createFence(video::IGPUFence::ECF_SIGNALED_BIT);
			m_imageAcquire[i] = logicalDevice->createSemaphore();
			m_renderFinished[i] = logicalDevice->createSemaphore();
		}
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	void workLoopBody() override
	{
		m_resourceIx++;
		if (m_resourceIx >= FRAMES_IN_FLIGHT)
			m_resourceIx = 0;

		auto& cb = m_cmdbuf[m_resourceIx];
		auto& commandPool = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS][m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];

		// TODO:
		// Input and change zoom based on camera
		// Update UniformBuffer/ConstantBuffer data with camera shit.

		logicalDevice->blockForFences(1u, &fence.get());
		logicalDevice->resetFences(1u, &fence.get());

		uint32_t imgnum = 0u;
		auto acquireResult = swapchain->acquireNextImage(m_imageAcquire[m_resourceIx].get(), nullptr, &imgnum);
		assert(acquireResult == video::ISwapchain::E_ACQUIRE_IMAGE_RESULT::EAIR_SUCCESS);

		core::smart_refctd_ptr<video::IGPUImage> swapchainImg = m_swapchainImages[imgnum];

		uint32_t windowWidth = swapchain->getCreationParameters().width;
		uint32_t windowHeight = swapchain->getCreationParameters().height;

		// safe to proceed
		cb->reset(video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT); // TODO: Begin doesn't release the resources in the command pool, meaning the old swapchains never get dropped
		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT); // TODO: Reset Frame's CommandPool

		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = windowWidth;
		vp.height = windowHeight;
		cb->setViewport(0u, 1u, &vp);

		VkRect2D scissor;
		scissor.extent = { windowWidth, windowHeight };
		scissor.offset = { 0, 0 };
		cb->setScissor(0u, 1u, &scissor);

		// SwapchainImage Transition to EL_COLOR_ATTACHMENT_OPTIMAL
		{
			nbl::video::IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
			imageBarriers[0].barrier.srcAccessMask = nbl::asset::EAF_NONE;
			imageBarriers[0].barrier.dstAccessMask = nbl::asset::EAF_COLOR_ATTACHMENT_WRITE_BIT;
			imageBarriers[0].oldLayout = nbl::asset::IImage::EL_UNDEFINED;
			imageBarriers[0].newLayout = nbl::asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL;
			imageBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].image = swapchainImg;
			imageBarriers[0].subresourceRange.aspectMask = nbl::asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(nbl::asset::EPSF_TOP_OF_PIPE_BIT, nbl::asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT, nbl::asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);
		}

		// TODO
		// Bind DescriptorSet, GraphicsPipeline
		// BindVertexBuffer, BindIndexBuffer
		// Issue cb->drawIndexed();

		// SwapchainImage Transition to EL_PRESENT_SRC
		{
			nbl::video::IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
			imageBarriers[0].barrier.srcAccessMask = nbl::asset::EAF_COLOR_ATTACHMENT_WRITE_BIT;
			imageBarriers[0].barrier.dstAccessMask = nbl::asset::EAF_NONE;
			imageBarriers[0].oldLayout = nbl::asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL;
			imageBarriers[0].newLayout = nbl::asset::IImage::EL_PRESENT_SRC;
			imageBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].image = swapchainImg;
			imageBarriers[0].subresourceRange.aspectMask = nbl::asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(nbl::asset::EPSF_ALL_GRAPHICS_BIT, nbl::asset::EPSF_BOTTOM_OF_PIPE_BIT, nbl::asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);
		}

		cb->end();

		CommonAPI::Submit(
			logicalDevice.get(),
			cb.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			m_imageAcquire[m_resourceIx].get(),
			m_renderFinished[m_resourceIx].get(),
			fence.get());


		CommonAPI::Present(
			logicalDevice.get(),
			swapchain.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			m_renderFinished[m_resourceIx].get(),
			imgnum);
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}
};

//NBL_COMMON_API_MAIN(CADApp)
int main(int argc, char** argv) {
	CommonAPI::main<CADApp>(argc, argv);
}