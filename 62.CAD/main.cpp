#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"


static constexpr bool DebugMode = false;

enum class ExampleMode
{
	CASE_0, // Zooming In/Out
	CASE_1, // Rotating Line
	CASE_2, // Straight Line Moving up and down
	CASE_3, // Ellipses
};

constexpr ExampleMode mode = ExampleMode::CASE_3;


struct double4x4
{
	double _r0[4u];
	double _r1[4u];
	double _r2[4u];
	double _r3[4u];
};

struct double2
{
	double x;
	double y;

	inline double2 operator-(const double2& other) const
	{
		return { x - other.x, y - other.y };
	}
	inline double2 operator+(const double2& other) const
	{
		return { x + other.x, y + other.y };
	}
};

struct uint2
{
	uint32_t x;
	uint32_t y;
};

#define float4 nbl::core::vectorSIMDf
#include "common.hlsl"

static_assert(sizeof(DrawObject) == 16u);
static_assert(sizeof(LinePoints) == 64u);
static_assert(sizeof(EllipseInfo) == 48u);
static_assert(sizeof(Globals) == 160u);

using namespace nbl;
using namespace ui;

class Camera2D : public core::IReferenceCounted
{
public:
	Camera2D()
	{}

	void setOrigin(const double2& origin)
	{
		m_origin = origin;
	}

	void setAspectRatio(const double& aspectRatio)
	{
		m_aspectRatio = aspectRatio;
	}

	void setSize(const double size)
	{
		m_size = { size * m_aspectRatio, size };
	}

	double4x4 constructViewProjection()
	{
		double4x4 ret = {};

		ret._r0[0] = 2.0 / m_size.x;
		ret._r1[1] = -2.0 / m_size.y;
		ret._r2[2] = 1.0;

		ret._r2[0] = (-2.0 * m_origin.x) / m_size.x;
		ret._r2[1] = (2.0 * m_origin.y) / m_size.y;

		return ret;
	}

	void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
	{
		for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
		{
			auto ev = *eventIt;

			if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL)
			{
				m_size = m_size + double2{ (double)ev.scrollEvent.verticalScroll * -0.1 * m_aspectRatio, (double)ev.scrollEvent.verticalScroll * -0.1};
				m_size = double2 {core::max(m_aspectRatio, m_size.x), core::max(1.0, m_size.y)};
			}
		}
	}

private:

	double m_aspectRatio = 0.0;
	double2 m_size = {};
	double2 m_origin = {};
};

class CADApp : public ApplicationBase
{
	constexpr static uint32_t FRAMES_IN_FLIGHT = 3u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	constexpr static uint32_t WIN_W = 1280u;
	constexpr static uint32_t WIN_H = 720u;

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	video::CDumbPresentationOracle oracle;

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
	Camera2D m_Camera;
	uint32_t currentDrawObjectCount = 0u;
	uint64_t geometryBufferAddress = 0u;
	uint64_t currentGeometryBufferOffset = 0u;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> m_device;
	core::smart_refctd_ptr<video::IGPUBuffer> indexBuffer;
	core::smart_refctd_ptr<video::IGPUBuffer> drawObjectsBuffer;
	core::smart_refctd_ptr<video::IGPUBuffer> geometryBuffer;
	core::smart_refctd_ptr<video::IGPUBuffer> globalsBuffer[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUDescriptorSet> descriptorSets[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> graphicsPipeline;
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> debugGraphicsPipeline;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> graphicsPipelineLayout;

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

	void addLines(std::vector<double2>&& linePoints)
	{
		if (linePoints.size() >= 2u)
		{
			const auto noLines = linePoints.size() - 1u;
			DrawObject drawObj = {};
			drawObj.type = ObjectType::LINE;
			drawObj.address = geometryBufferAddress + currentGeometryBufferOffset;
			for(uint32_t i = 0u; i < noLines; ++i)
			{
				asset::SBufferRange<video::IGPUBuffer> drawObjUpload = { currentDrawObjectCount * sizeof(DrawObject), sizeof(DrawObject), drawObjectsBuffer };
				utilities->updateBufferRangeViaStagingBufferAutoSubmit(drawObjUpload, &drawObj, queues[CommonAPI::InitOutput::EQT_TRANSFER_UP]);
				currentDrawObjectCount += 1u;
				drawObj.address += sizeof(double2);
			}

			const auto& firstPoint = linePoints[0u];
			const auto& secondPoint = linePoints[1u];
			const auto differenceStart = firstPoint - secondPoint;
			const double2 generatedStart = firstPoint + differenceStart;
			linePoints.emplace(linePoints.begin(), generatedStart);
			const auto& lastPoint = linePoints[linePoints.size() - 1u];
			const auto& oneToLastPoint = linePoints[linePoints.size() - 2u];
			const auto differenceEnd = lastPoint - oneToLastPoint;
			const double2 generatedEnd = lastPoint + differenceEnd;
			linePoints.push_back(generatedEnd);
			const auto pointsByteSize = sizeof(double2) * linePoints.size();

			assert(currentGeometryBufferOffset + pointsByteSize <= geometryBuffer->getSize());
			asset::SBufferRange<video::IGPUBuffer> geometryUpload = { currentGeometryBufferOffset, pointsByteSize, geometryBuffer };
			utilities->updateBufferRangeViaStagingBufferAutoSubmit(geometryUpload, linePoints.data(), queues[CommonAPI::InitOutput::EQT_TRANSFER_UP]);
			currentGeometryBufferOffset += pointsByteSize;
		}
	}

	void addEllipse(const EllipseInfo& ellipseInfo)
	{
		DrawObject drawObj = {};
		drawObj.type = ObjectType::ELLIPSE;
		drawObj.address = geometryBufferAddress + currentGeometryBufferOffset;
		asset::SBufferRange<video::IGPUBuffer> drawObjUpload = { currentDrawObjectCount * sizeof(DrawObject), sizeof(DrawObject), drawObjectsBuffer };
		utilities->updateBufferRangeViaStagingBufferAutoSubmit(drawObjUpload, &drawObj, queues[CommonAPI::InitOutput::EQT_TRANSFER_UP]);
		currentDrawObjectCount += 1u;

		const auto ellipseBytesize = sizeof(EllipseInfo);

		assert(currentGeometryBufferOffset + ellipseBytesize <= geometryBuffer->getSize());
		asset::SBufferRange<video::IGPUBuffer> geometryUpload = { currentGeometryBufferOffset, ellipseBytesize, geometryBuffer };
		utilities->updateBufferRangeViaStagingBufferAutoSubmit(geometryUpload, &ellipseInfo, queues[CommonAPI::InitOutput::EQT_TRANSFER_UP]);
		currentGeometryBufferOffset += sizeof(EllipseInfo);
	}

	void clearObjects()
	{
		currentDrawObjectCount = 0u;
		currentGeometryBufferOffset = 0u;
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
			utilities->updateBufferRangeViaStagingBufferAutoSubmit(rangeToUpload, indices.data(), queues[CommonAPI::InitOutput::EQT_TRANSFER_UP]);
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
			geometryBufferAddress = logicalDevice->getBufferDeviceAddress(geometryBuffer.get());
		}


		{
			size_t globalsBufferSize = sizeof(Globals);

			for(uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i)
			{
				video::IGPUBuffer::SCreationParams globalsCreationParams = {};
				globalsCreationParams.size = globalsBufferSize;
				globalsCreationParams.usage = core::bitflag(video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT) | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
				globalsBuffer[i] = logicalDevice->createBuffer(std::move(globalsCreationParams));

				video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = globalsBuffer[i]->getMemoryReqs();
				memReq.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
				auto globalsBufferMem = logicalDevice->allocate(memReq, globalsBuffer[i].get());
			}
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
		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
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
		initParams.physicalDeviceFilter.requiredFeatures.shaderFloat64 = true;
		initParams.physicalDeviceFilter.requiredFeatures.fillModeNonSolid = DebugMode;
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

		commandPools = std::move(initOutput.commandPools);
		const auto& graphicsCommandPools = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS];

		CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, WIN_W, WIN_H, swapchain);

		framebuffersDynArraySmartPtr = CommonAPI::createFBOWithSwapchainImages(
			swapchain->getImageCount(), WIN_W, WIN_H,
			logicalDevice, swapchain, renderpass,
			getDepthFormat()
		);

		const uint32_t swapchainImageCount = swapchain->getImageCount();
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

		core::smart_refctd_ptr<video::IGPUSpecializedShader> shaders[3u] = {};
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			core::smart_refctd_ptr<asset::ICPUSpecializedShader> cpuShaders[3u] = {};
			constexpr auto vertexShaderPath = "../vertex_shader.hlsl";
			constexpr auto fragmentShaderPath = "../fragment_shader.hlsl";
			constexpr auto debugfragmentShaderPath = "../fragment_shader_debug.hlsl";
			cpuShaders[0u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(vertexShaderPath, params).getContents().begin());
			cpuShaders[1u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(fragmentShaderPath, params).getContents().begin());
			cpuShaders[2u] = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(debugfragmentShaderPath, params).getContents().begin());
			cpuShaders[0u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			cpuShaders[1u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			cpuShaders[2u]->setSpecializationInfo(asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			auto gpuShaders = CPU2GPU.getGPUObjectsFromAssets(cpuShaders, cpuShaders + 3u, cpu2gpuParams);
			shaders[0u] = gpuShaders->begin()[0u];
			shaders[1u] = gpuShaders->begin()[1u];
			shaders[2u] = gpuShaders->begin()[2u];
		}

		initDrawObjects();

		video::IGPUDescriptorSetLayout::SBinding bindings[2u] = {};
		bindings[0u].binding = 0u;
		bindings[0u].type = asset::EDT_UNIFORM_BUFFER;
		bindings[0u].count = 1u;
		bindings[0u].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;
		bindings[1u].binding = 1u;
		bindings[1u].type = asset::EDT_STORAGE_BUFFER;
		bindings[1u].count = 1u;
		bindings[1u].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;
		auto descriptorSetLayout = logicalDevice->createDescriptorSetLayout(bindings, bindings+2u);

		nbl::video::IDescriptorPool::SDescriptorPoolSize poolSizes[2u] =
		{
			{ nbl::asset::EDT_UNIFORM_BUFFER, FRAMES_IN_FLIGHT },
			{ nbl::asset::EDT_STORAGE_BUFFER, FRAMES_IN_FLIGHT },
		};
		auto descriptorPool = logicalDevice->createDescriptorPool(nbl::video::IDescriptorPool::ECF_NONE, 128u, 2u, poolSizes);

		for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
		{
			descriptorSets[i] = logicalDevice->createDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(descriptorSetLayout));
			video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[2u] = {};
			descriptorInfos[0u].buffer.offset = 0u;
			descriptorInfos[0u].buffer.size = globalsBuffer[i]->getCreationParams().size;
			descriptorInfos[0u].desc = globalsBuffer[i];

			descriptorInfos[1u].buffer.offset = 0u;
			descriptorInfos[1u].buffer.size = drawObjectsBuffer->getCreationParams().size;
			descriptorInfos[1u].desc = drawObjectsBuffer;

			video::IGPUDescriptorSet::SWriteDescriptorSet descriptorUpdates[2u] = {};
			descriptorUpdates[0u].dstSet = descriptorSets[i].get();
			descriptorUpdates[0u].binding = 0u;
			descriptorUpdates[0u].arrayElement = 0u;
			descriptorUpdates[0u].count = 1u;
			descriptorUpdates[0u].descriptorType = asset::EDT_UNIFORM_BUFFER;
			descriptorUpdates[0u].info = &descriptorInfos[0];

			descriptorUpdates[1u].dstSet = descriptorSets[i].get();
			descriptorUpdates[1u].binding = 1u;
			descriptorUpdates[1u].arrayElement = 0u;
			descriptorUpdates[1u].count = 1u;
			descriptorUpdates[1u].descriptorType = asset::EDT_STORAGE_BUFFER;
			descriptorUpdates[1u].info = &descriptorInfos[1];

			logicalDevice->updateDescriptorSets(2u, descriptorUpdates, 0u, nullptr);
		}

		graphicsPipelineLayout = logicalDevice->createPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);

		video::IGPURenderpassIndependentPipeline::SCreationParams renderpassIndependantPipeInfo = {};
		renderpassIndependantPipeInfo.layout = graphicsPipelineLayout;
		renderpassIndependantPipeInfo.shaders[0u] = shaders[0u];
		renderpassIndependantPipeInfo.shaders[1u] = shaders[1u];
		// renderpassIndependantPipeInfo.vertexInput; no gpu vertex buffers
		renderpassIndependantPipeInfo.blend.blendParams[0u].blendEnable = true;
		renderpassIndependantPipeInfo.blend.blendParams[0u].srcColorFactor = asset::EBF_SRC_ALPHA;
		renderpassIndependantPipeInfo.blend.blendParams[0u].dstColorFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
		renderpassIndependantPipeInfo.blend.blendParams[0u].colorBlendOp = asset::EBO_ADD;
		renderpassIndependantPipeInfo.blend.blendParams[0u].srcAlphaFactor = asset::EBF_ONE;
		renderpassIndependantPipeInfo.blend.blendParams[0u].dstAlphaFactor = asset::EBF_ZERO;
		renderpassIndependantPipeInfo.blend.blendParams[0u].alphaBlendOp = asset::EBO_ADD;
		renderpassIndependantPipeInfo.blend.blendParams[0u].colorWriteMask = (1u << 4u) - 1u;

		renderpassIndependantPipeInfo.primitiveAssembly.primitiveType = asset::E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST;
		renderpassIndependantPipeInfo.rasterization.depthTestEnable = false;
		renderpassIndependantPipeInfo.rasterization.depthWriteEnable = false;
		renderpassIndependantPipeInfo.rasterization.stencilTestEnable = false;
		renderpassIndependantPipeInfo.rasterization.polygonMode = asset::EPM_FILL;
		renderpassIndependantPipeInfo.rasterization.faceCullingMode = asset::EFCM_NONE;

		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> renderpassIndependant;
		bool succ = logicalDevice->createRenderpassIndependentPipelines(
			nullptr,
			core::SRange<const video::IGPURenderpassIndependentPipeline::SCreationParams>(&renderpassIndependantPipeInfo, &renderpassIndependantPipeInfo + 1u),
			&renderpassIndependant);
		assert(succ);

		video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineCreateInfo = {};
		graphicsPipelineCreateInfo.renderpassIndependent = renderpassIndependant;
		graphicsPipelineCreateInfo.renderpass = renderpass;
		graphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(graphicsPipelineCreateInfo));

		if constexpr (DebugMode)
		{
			core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> renderpassIndependantDebug;
			renderpassIndependantPipeInfo.shaders[1u] = shaders[2u];
			renderpassIndependantPipeInfo.rasterization.polygonMode = asset::EPM_LINE;
			succ = logicalDevice->createRenderpassIndependentPipelines(
				nullptr,
				core::SRange<const video::IGPURenderpassIndependentPipeline::SCreationParams>(&renderpassIndependantPipeInfo, &renderpassIndependantPipeInfo + 1u),
				&renderpassIndependantDebug);
			assert(succ);

			video::IGPUGraphicsPipeline::SCreationParams debugGraphicsPipelineCreateInfo = {};
			debugGraphicsPipelineCreateInfo.renderpassIndependent = renderpassIndependantDebug;
			debugGraphicsPipelineCreateInfo.renderpass = renderpass;
			debugGraphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(debugGraphicsPipelineCreateInfo));
		}

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

		m_Camera.setOrigin({ 00.0, 0.0 });
		m_Camera.setAspectRatio((double)WIN_W / WIN_H);
		m_Camera.setSize(200.0);

		update(0.0);

		oracle.reportBeginFrameRecord();
	}

	void update(double timeElapsed)
	{
		utilities->getDefaultUpStreamingBuffer()->cull_frees();
		clearObjects();

		std::vector<double2> linePoints;

		if constexpr (mode == ExampleMode::CASE_0)
		{
			linePoints.push_back({ -50.0, 0.0 });
			linePoints.push_back({ 0.0, 0.0 });
			linePoints.push_back({ 80.0, 10.0 });
			linePoints.push_back({ 40.0, 40.0 });
			linePoints.push_back({ 0.0, 40.0 });
			linePoints.push_back({ 30.0, 80.0 });
			linePoints.push_back({ -30.0, 50.0 });
			linePoints.push_back({ -30.0, 110.0 });
			linePoints.push_back({ +30.0, -112.0 });
			addLines(std::move(linePoints));
		}
		else if (mode == ExampleMode::CASE_1)
		{
			linePoints.push_back({ 0.0, 0.0 });
			linePoints.push_back({ 30.0, 30.0 });
			addLines(std::move(linePoints));
		}
		else if (mode == ExampleMode::CASE_2)
		{
			linePoints.push_back({ -70.0, cos(timeElapsed * 0.00003) * 10 });
			linePoints.push_back({ 70.0, cos(timeElapsed * 0.00003) * 10 });
			addLines(std::move(linePoints));
		}
		else if (mode == ExampleMode::CASE_3)
		{
			constexpr double twoPi = core::PI<double>() * 2.0;
			EllipseInfo ellipse = {};
			const double a = timeElapsed * 0.001;
			// ellipse.majorAxis = { 40.0 * cos(a), 40.0 * sin(a) };
			ellipse.majorAxis = { 40.0, 0.0 };
			ellipse.center = { 0, 0 };
			ellipse.eccentricityPacked = (0.6 * UINT32_MAX);

			ellipse.angleBoundsPacked = {
				static_cast<uint32_t>(((0.0) / twoPi) * UINT32_MAX),
				static_cast<uint32_t>(((core::PI<double>() * 0.5) / twoPi) * UINT32_MAX)
			};
			addEllipse(ellipse);

			ellipse.angleBoundsPacked = {
				static_cast<uint32_t>(((core::PI<double>() * 0.5) / twoPi) * UINT32_MAX),
				static_cast<uint32_t>(((core::PI<double>()) / twoPi) * UINT32_MAX)
			};
			addEllipse(ellipse);
			ellipse.angleBoundsPacked = {
				static_cast<uint32_t>(((core::PI<double>()) / twoPi) * UINT32_MAX),
				static_cast<uint32_t>(((core::PI<double>() * 1.5) / twoPi) * UINT32_MAX)
			};
			addEllipse(ellipse);
			ellipse.angleBoundsPacked = {
				static_cast<uint32_t>(((core::PI<double>() * 1.5) / twoPi) * UINT32_MAX),
				static_cast<uint32_t>(((core::PI<double>() * 2) / twoPi) * UINT32_MAX)
			};
			addEllipse(ellipse);
		}
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	double dt = 0; //! render loop
	std::chrono::steady_clock::time_point lastTime;

	void workLoopBody() override
	{
		m_resourceIx++;
		if (m_resourceIx >= FRAMES_IN_FLIGHT)
			m_resourceIx = 0;

		auto& cb = m_cmdbuf[m_resourceIx];
		auto& commandPool = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS][m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];

		logicalDevice->blockForFences(1u, &fence.get());
		logicalDevice->resetFences(1u, &fence.get());

		auto now = std::chrono::high_resolution_clock::now();
		dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
		lastTime = now;
		static double timeElapsed = 0.0;
		timeElapsed += dt;

		update(timeElapsed);

		uint32_t imgnum = 0u;

		const auto nextPresentationTimestamp = oracle.acquireNextImage(swapchain.get(), m_imageAcquire[m_resourceIx].get(), nullptr, &imgnum);
		// auto acquireResult = swapchain->acquireNextImage(m_imageAcquire[m_resourceIx].get(), nullptr, &imgnum);
		// assert(acquireResult == video::ISwapchain::E_ACQUIRE_IMAGE_RESULT::EAIR_SUCCESS);

		core::smart_refctd_ptr<video::IGPUImage> swapchainImg = m_swapchainImages[imgnum];

		uint32_t windowWidth = swapchain->getCreationParameters().width;
		uint32_t windowHeight = swapchain->getCreationParameters().height;

		// safe to proceed
		cb->reset(video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT); // TODO: Begin doesn't release the resources in the command pool, meaning the old swapchains never get dropped
		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT); // TODO: Reset Frame's CommandPool

		if constexpr (mode == ExampleMode::CASE_0)
		{
			m_Camera.setSize(20.0 + abs(cos(timeElapsed * 0.0001)) * 7000);
		}

		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void 
			{
				m_Camera.mouseProcess(events);
			}
		, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void 
			{
				// TODO:
			}
		, logger.get());

		Globals globalData = {};
		globalData.color = core::vectorSIMDf(0.8f, 0.7f, 0.5f, 0.5f);
		globalData.lineWidth = 6.0f;
		globalData.antiAliasingFactor = 1.0f;// + abs(cos(timeElapsed * 0.0008))*20.0f;
		globalData.resolution = uint2{ WIN_W, WIN_H };
		globalData.viewProjection = m_Camera.constructViewProjection();
		cb->updateBuffer(globalsBuffer[m_resourceIx].get(), 0ull, sizeof(Globals), &globalData);

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

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { WIN_W, WIN_H };
			asset::SClearValue clear[2] = {};
			clear[0].color.float32[0] = 0.f;
			clear[0].color.float32[1] = 0.f;
			clear[0].color.float32[2] = 0.f;
			clear[0].color.float32[3] = 0.f;
			clear[1].depthStencil.depth = 1.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = framebuffersDynArraySmartPtr->begin()[imgnum];
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		cb->beginRenderPass(&beginInfo, asset::ESC_INLINE);

		cb->bindDescriptorSets(asset::EPBP_GRAPHICS, graphicsPipelineLayout.get(), 0u, 1u, &descriptorSets[m_resourceIx].get());
		cb->bindIndexBuffer(indexBuffer.get(), 0u, asset::EIT_32BIT);
		cb->bindGraphicsPipeline(graphicsPipeline.get());
		cb->drawIndexed(currentDrawObjectCount * 6u, 1u, 0u, 0u, 0u);

		if constexpr (DebugMode)
		{
			cb->bindGraphicsPipeline(debugGraphicsPipeline.get());
			cb->drawIndexed(currentDrawObjectCount * 6u, 1u, 0u, 0u, 0u);
		}

		cb->endRenderPass();

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