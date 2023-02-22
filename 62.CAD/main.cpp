#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"


static constexpr bool DebugMode = false;
static constexpr bool FragmentShaderPixelInterlock = true;

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

typedef nbl::core::vector2d<double> double2;
typedef nbl::core::vector2d<uint32_t> uint2;

#define float4 nbl::core::vectorSIMDf
#include "common.hlsl"

static_assert(sizeof(DrawObject) == 16u);
static_assert(sizeof(EllipseInfo) == 48u);
static_assert(sizeof(Globals) == 160u);

using namespace nbl;
using namespace ui;

// TODO: Use a math lib?
double dot(const double2& a, const double2& b)
{
	return a.X * b.X + a.Y * b.Y;
}
double2 normalize(const double2& x)
{
	double len = dot(x,x);
#ifdef __NBL_FAST_MATH
	return x * core::inversesqrt<double>(len);
#else
	return x / core::sqrt<double>(len);
#endif
}

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
		m_size = double2{ size * m_aspectRatio, size };
	}

	double4x4 constructViewProjection()
	{
		double4x4 ret = {};

		ret._r0[0] = 2.0 / m_size.X;
		ret._r1[1] = -2.0 / m_size.Y;
		ret._r2[2] = 1.0;

		ret._r2[0] = (-2.0 * m_origin.X) / m_size.X;
		ret._r2[1] = (2.0 * m_origin.Y) / m_size.Y;

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
				m_size = double2 {core::max(m_aspectRatio, m_size.X), core::max(1.0, m_size.Y)};
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
	core::smart_refctd_ptr<video::IGPUSemaphore> m_drawBufferUploadsFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> m_drawBufferUploadsComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_uploadCmdBuf[FRAMES_IN_FLIGHT] = { nullptr };

	nbl::video::ISwapchain::SCreationParams m_swapchainCreationParams;

	// Related to Drawing Stuff
	Camera2D m_Camera;

	core::smart_refctd_ptr<video::IGPUImageView> pseudoStencilImageView[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUBuffer> globalsBuffer[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUDescriptorSet> descriptorSets[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> graphicsPipeline;
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> debugGraphicsPipeline;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> graphicsPipelineLayout;

	template <typename BufferType>
	struct DrawBuffers
	{
		core::smart_refctd_ptr<BufferType> indexBuffer;
		core::smart_refctd_ptr<BufferType> drawObjectsBuffer;
		core::smart_refctd_ptr<BufferType> geometryBuffer;
	};

	struct DrawBuffersFiller
	{
		core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		core::smart_refctd_ptr<nbl::video::ILogicalDevice> device;

		DrawBuffers<asset::ICPUBuffer> cpuDrawBuffers;
		DrawBuffers<video::IGPUBuffer> gpuDrawBuffers;

		uint64_t geometryBufferAddress = 0u;

		uint32_t currentIndexCount = 0u;
		uint32_t maxIndices = 0u;
		
		uint32_t currentDrawObjectCount = 0u;
		uint32_t maxDrawObjects = 0u;

		uint64_t currentGeometryBufferSize = 0u;
		
		void allocateIndexBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, uint32_t indices)
		{
			maxIndices = indices;
			const size_t indexBufferSize = maxIndices * sizeof(uint32_t);

			video::IGPUBuffer::SCreationParams indexBufferCreationParams = {};
			indexBufferCreationParams.size = indexBufferSize;
			indexBufferCreationParams.usage = video::IGPUBuffer::EUF_INDEX_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
			gpuDrawBuffers.indexBuffer = logicalDevice->createBuffer(std::move(indexBufferCreationParams));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.indexBuffer->getMemoryReqs();
			memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto indexBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.indexBuffer.get());

			cpuDrawBuffers.indexBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(indexBufferSize);
		}

		void allocateDrawObjectsBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, uint32_t drawObjects)
		{
			maxDrawObjects = drawObjects;
			size_t drawObjectsBufferSize = drawObjects * sizeof(DrawObject);

			video::IGPUBuffer::SCreationParams drawObjectsCreationParams = {};
			drawObjectsCreationParams.size = drawObjectsBufferSize;
			drawObjectsCreationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
			gpuDrawBuffers.drawObjectsBuffer = logicalDevice->createBuffer(std::move(drawObjectsCreationParams));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.drawObjectsBuffer->getMemoryReqs();
			memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto drawObjectsBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.drawObjectsBuffer.get());

			cpuDrawBuffers.drawObjectsBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(drawObjectsBufferSize);
		}

		void allocateGeometryBuffer(core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice, size_t size)
		{
			video::IGPUBuffer::SCreationParams geometryCreationParams = {};
			geometryCreationParams.size = size;
			geometryCreationParams.usage = core::bitflag(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
			gpuDrawBuffers.geometryBuffer = logicalDevice->createBuffer(std::move(geometryCreationParams));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = gpuDrawBuffers.geometryBuffer->getMemoryReqs();
			memReq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto geometryBufferMem = logicalDevice->allocate(memReq, gpuDrawBuffers.geometryBuffer.get(), video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
			geometryBufferAddress = logicalDevice->getBufferDeviceAddress(gpuDrawBuffers.geometryBuffer.get());

			cpuDrawBuffers.geometryBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(size);
		}

		void clear()
		{
			currentIndexCount = 0u;
			currentDrawObjectCount = 0u;
			currentGeometryBufferSize = 0u;
		}

		void finalizeCopiesToGPU(
			core::smart_refctd_ptr<nbl::video::IUtilities> utils,
			video::IGPUQueue* submissionQueue,
			video::IGPUFence* submissionFence,
			video::IGPUQueue::SSubmitInfo intendedNextSubmit)
		{
			// Copy Indices
			asset::SBufferRange<video::IGPUBuffer> indicesRange = {0u, sizeof(uint32_t) * currentIndexCount, gpuDrawBuffers.indexBuffer };
			intendedNextSubmit = utils->updateBufferRangeViaStagingBuffer(indicesRange, cpuDrawBuffers.indexBuffer->getPointer(), submissionQueue, submissionFence, intendedNextSubmit);
			// Copy DrawBuffers
			asset::SBufferRange<video::IGPUBuffer> drawObjectsRange = { 0u, sizeof(DrawObject) * currentDrawObjectCount, gpuDrawBuffers.drawObjectsBuffer };
			intendedNextSubmit = utils->updateBufferRangeViaStagingBuffer(drawObjectsRange, cpuDrawBuffers.drawObjectsBuffer->getPointer(), submissionQueue, submissionFence, intendedNextSubmit);
			// Copy GeometryBuffer and AutoSubmit
			asset::SBufferRange<video::IGPUBuffer> geomRange = { 0u, currentGeometryBufferSize, gpuDrawBuffers.geometryBuffer };
			utils->updateBufferRangeViaStagingBufferAutoSubmit(geomRange, cpuDrawBuffers.geometryBuffer->getPointer(), submissionQueue, submissionFence, intendedNextSubmit);
		}

		void addLines(std::vector<double2>&& linePoints)
		{
			if (linePoints.size() < 2u)
				return;

			const auto noLines = linePoints.size() - 1u;

			// Indices for Objects
			bool isOpaque = false;
			addNewObjectsIndices(noLines, isOpaque);

			// DrawObj
			DrawObject drawObj = {};
			drawObj.type = ObjectType::LINE;
			drawObj.address = geometryBufferAddress + currentGeometryBufferSize;
			for (uint32_t i = 0u; i < noLines; ++i)
			{
				void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
				memcpy(dst, &drawObj, sizeof(DrawObject));
				currentDrawObjectCount += 1u;
				drawObj.address += sizeof(double2);
			}
			
			// Geom
			{
				const auto pointsByteSize = sizeof(double2) * linePoints.size();
				void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
				memcpy(dst, linePoints.data(), pointsByteSize);
				currentGeometryBufferSize += pointsByteSize;
			}
		}

		void addRoads(std::vector<double2>&& linePoints)
		{
			return;
			if (linePoints.size() < 2u)
				return;

			const auto noPoints = linePoints.size();
			const auto noLines = noPoints - 1u;

			// Indices for Objects
			bool isOpaque = false;
			addNewObjectsIndices(noLines, isOpaque);

			double2 prevLineVec;
			for (uint32_t i = 0u; i < noPoints; ++i)
			{
				void* geomBufferPointer = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
				void* drawObjPointer = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;

				// Add Geom
				{
					double2 lineVec;
					if (i < noPoints - 1u)
						lineVec = linePoints[i + 1] - linePoints[i];
					else 
						lineVec = prevLineVec;

					lineVec = normalize(lineVec);

					if (i == 0)
						prevLineVec = lineVec;

					const double2 normalToLine = double2{ -lineVec.Y, lineVec.X };
					const double2 normalToPrevLine = double2{ -prevLineVec.Y, prevLineVec.X};
					const double2 miter = normalize(normalToPrevLine + normalToLine);
					const double lineWidth = 2.0;
					double2 points[2u] = {};
					points[0u] = linePoints[i] + (miter * lineWidth * 0.5) / dot(normalToLine, miter);
					points[1u] = linePoints[i] - (miter * lineWidth * 0.5) / dot(normalToLine, miter);

					memcpy(geomBufferPointer, points, sizeof(double2) * 2u);
					prevLineVec = lineVec;
				}

				// Add Draw Obj
				if (i < noPoints - 1u)
				{
					// DrawObj
					DrawObject drawObj = {};
					drawObj.type = ObjectType::ROAD;
					drawObj.address = geometryBufferAddress + currentGeometryBufferSize;
					memcpy(drawObjPointer, &drawObj, sizeof(DrawObject));
					currentDrawObjectCount++;
				}

				currentGeometryBufferSize += sizeof(double2) * 2u;
			}
		}

		void addEllipse(const EllipseInfo& ellipseInfo)
		{
			// Indices for objects
			bool isOpaque = false;
			addNewObjectsIndices(1u, isOpaque);

			// Geom
			DrawObject drawObj = {};
			drawObj.type = ObjectType::ELLIPSE;
			drawObj.address = geometryBufferAddress + currentGeometryBufferSize;
			void* dst = reinterpret_cast<DrawObject*>(cpuDrawBuffers.drawObjectsBuffer->getPointer()) + currentDrawObjectCount;
			memcpy(dst, &drawObj, sizeof(DrawObject));
			currentDrawObjectCount += 1u;

			// Geom
			{
				void* dst = reinterpret_cast<char*>(cpuDrawBuffers.geometryBuffer->getPointer()) + currentGeometryBufferSize;
				memcpy(dst, &ellipseInfo, sizeof(EllipseInfo));
				currentGeometryBufferSize += sizeof(EllipseInfo);
			}
		}

		void addNewObjectsIndices(const uint32_t noOfObjects, const bool isOpaque)
		{
			uint32_t* indices = reinterpret_cast<uint32_t*>(cpuDrawBuffers.indexBuffer->getPointer()) + currentIndexCount;

			for (uint32_t i = 0u; i < noOfObjects; ++i)
			{
				uint32_t start = i + currentDrawObjectCount;
				indices[i * 6] = start * 4u + 0u;
				indices[i * 6 + 1u] = start * 4u + 1u;
				indices[i * 6 + 2u] = start * 4u + 2u;

				indices[i * 6 + 3u] = start * 4u + 2u;
				indices[i * 6 + 4u] = start * 4u + 1u;
				indices[i * 6 + 5u] = start * 4u + 3u;
			}
			if (!isOpaque)
			{
				// Transparent Objects (as a whole) need to draw twice, one for setting the alpha and another for clearing it and drawing it
				indices += noOfObjects * 6u;
				for (uint32_t i = 0u; i < noOfObjects; ++i)
				{
					uint32_t start = i + currentDrawObjectCount;
					indices[i * 6] = start * 4u + 1u;
					indices[i * 6 + 1u] = start * 4u + 0u;
					indices[i * 6 + 2u] = start * 4u + 2u;

					indices[i * 6 + 3u] = start * 4u + 1u;
					indices[i * 6 + 4u] = start * 4u + 2u;
					indices[i * 6 + 5u] = start * 4u + 3u;
				}
			}
			currentIndexCount += noOfObjects * 6u * (isOpaque ? 1u : 2u);
		}
	};

	DrawBuffersFiller drawBuffers[FRAMES_IN_FLIGHT];

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
		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			size_t maxIndices = maxObjects * 6u * 2u;
			drawBuffers[i].allocateIndexBuffer(logicalDevice, maxIndices);
			drawBuffers[i].allocateDrawObjectsBuffer(logicalDevice, maxObjects);

			size_t geometryBufferSize = maxObjects * sizeof(EllipseInfo);
			drawBuffers[i].allocateGeometryBuffer(logicalDevice, geometryBufferSize);
		}

		for(uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i)
		{
			video::IGPUBuffer::SCreationParams globalsCreationParams = {};
			globalsCreationParams.size = sizeof(Globals);
			globalsCreationParams.usage = core::bitflag(video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT) | video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
			globalsBuffer[i] = logicalDevice->createBuffer(std::move(globalsCreationParams));

			video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = globalsBuffer[i]->getMemoryReqs();
			memReq.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
			auto globalsBufferMem = logicalDevice->allocate(memReq, globalsBuffer[i].get());
		}

		// pseudoStencil

		asset::E_FORMAT pseudoStencilFormat = asset::EF_R32_UINT;

		video::IPhysicalDevice::SImageFormatPromotionRequest promotionRequest = {};
		promotionRequest.originalFormat = asset::EF_R8_UINT;
		promotionRequest.usages = {};
		promotionRequest.usages.storageImageAtomic = true;
		pseudoStencilFormat = physicalDevice->promoteImageFormat(promotionRequest, video::IGPUImage::ET_OPTIMAL);
		
		for(uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			video::IGPUImage::SCreationParams imgInfo;
			imgInfo.format = pseudoStencilFormat;
			imgInfo.type = video::IGPUImage::ET_2D;
			imgInfo.extent.width = WIN_W;
			imgInfo.extent.height = WIN_H;
			imgInfo.extent.depth = 1u;
			imgInfo.mipLevels = 1u;
			imgInfo.arrayLayers = 1u;
			imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
			imgInfo.flags = asset::IImage::E_CREATE_FLAGS::ECF_NONE;
			imgInfo.usage = asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT;
			imgInfo.initialLayout = video::IGPUImage::EL_UNDEFINED;
			imgInfo.tiling = video::IGPUImage::ET_OPTIMAL;
			
			auto image = logicalDevice->createImage(std::move(imgInfo));
			auto imageMemReqs = image->getMemoryReqs();
			imageMemReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			logicalDevice->allocate(imageMemReqs, image.get());

			image->setObjectDebugName("pseudoStencil Image");

			video::IGPUImageView::SCreationParams imgViewInfo;
			imgViewInfo.image = std::move(image);
			imgViewInfo.format = pseudoStencilFormat;
			imgViewInfo.viewType = video::IGPUImageView::ET_2D;
			imgViewInfo.flags = video::IGPUImageView::E_CREATE_FLAGS::ECF_NONE;
			imgViewInfo.subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			imgViewInfo.subresourceRange.baseArrayLayer = 0u;
			imgViewInfo.subresourceRange.baseMipLevel = 0u;
			imgViewInfo.subresourceRange.layerCount = 1u;
			imgViewInfo.subresourceRange.levelCount = 1u;

			pseudoStencilImageView[i] = logicalDevice->createImageView(std::move(imgViewInfo));
		}
	}

	void update(const double timeElapsed, const uint32_t resourceIdx)
	{
		utilities->getDefaultUpStreamingBuffer()->cull_frees();
		auto& currentDrawBuffers = drawBuffers[resourceIdx];
		currentDrawBuffers.clear();

		if constexpr (mode == ExampleMode::CASE_0)
		{
			std::vector<double2> linePoints;
			linePoints.push_back({ -50.0, 0.0 });
			linePoints.push_back({ 0.0, 0.0 });
			linePoints.push_back({ 80.0, 10.0 });
			linePoints.push_back({ 40.0, 40.0 });
			linePoints.push_back({ 0.0, 40.0 });
			linePoints.push_back({ 30.0, 80.0 });
			linePoints.push_back({ -30.0, 50.0 });
			linePoints.push_back({ -30.0, 110.0 });
			linePoints.push_back({ +30.0, -112.0 });
			currentDrawBuffers.addLines(std::move(linePoints));
		}
		else if (mode == ExampleMode::CASE_1)
		{
			std::vector<double2> linePoints;
			linePoints.push_back({ 0.0, 0.0 });
			linePoints.push_back({ 30.0, 30.0 });
			currentDrawBuffers.addLines(std::move(linePoints));
		}
		else if (mode == ExampleMode::CASE_2)
		{
			std::vector<double2> linePoints;
			linePoints.push_back({ -70.0, cos(timeElapsed * 0.00003) * 10 });
			linePoints.push_back({ 70.0, cos(timeElapsed * 0.00003) * 10 });
			currentDrawBuffers.addLines(std::move(linePoints));
		}
		else if (mode == ExampleMode::CASE_3)
		{
			constexpr double twoPi = core::PI<double>() * 2.0;
			EllipseInfo ellipse = {};
			const double a = timeElapsed * 0.001;
			// ellipse.majorAxis = { 40.0 * cos(a), 40.0 * sin(a) };
			ellipse.majorAxis = double2{ 30.0, 0.0 };
			ellipse.center = double2{ 0, 0 };
			ellipse.eccentricityPacked = (0.6 * UINT32_MAX);

			ellipse.angleBoundsPacked = uint2{
				static_cast<uint32_t>(((0.0) / twoPi) * UINT32_MAX),
				static_cast<uint32_t>(((core::PI<double>() * 0.5) / twoPi) * UINT32_MAX)
			};
			currentDrawBuffers.addEllipse(ellipse);

			ellipse.angleBoundsPacked = uint2{
				static_cast<uint32_t>(((core::PI<double>() * 0.5) / twoPi) * UINT32_MAX),
				static_cast<uint32_t>(((core::PI<double>()) / twoPi) * UINT32_MAX)
			};
			currentDrawBuffers.addEllipse(ellipse);
			ellipse.angleBoundsPacked = uint2{
				static_cast<uint32_t>(((core::PI<double>()) / twoPi) * UINT32_MAX),
				static_cast<uint32_t>(((core::PI<double>() * 1.5) / twoPi) * UINT32_MAX)
			};
			currentDrawBuffers.addEllipse(ellipse);
			ellipse.angleBoundsPacked = uint2{
				static_cast<uint32_t>(((core::PI<double>() * 1.5) / twoPi) * UINT32_MAX),
				static_cast<uint32_t>(((core::PI<double>() * 2) / twoPi) * UINT32_MAX)
			};
			currentDrawBuffers.addEllipse(ellipse);
			ellipse.majorAxis = double2{ 30.0 * sin(timeElapsed * 0.0005), 30.0 * cos(timeElapsed * 0.0005) };
			ellipse.center = double2{ 50, 50 };
			ellipse.angleBoundsPacked = uint2{
				static_cast<uint32_t>(((core::PI<double>() * 1.5) / twoPi) * UINT32_MAX),
				static_cast<uint32_t>(((core::PI<double>() * 2) / twoPi) * UINT32_MAX)
			};
			currentDrawBuffers.addEllipse(ellipse);

			std::vector<double2> linePoints;
			linePoints.push_back({ -50.0, 0.0 });
			linePoints.push_back({ sin(timeElapsed * 0.0005) * 20, cos(timeElapsed * 0.0005) * 20 });
			linePoints.push_back({ -sin(timeElapsed * 0.0005) * 20, -cos(timeElapsed * 0.0005) * 20 });
			linePoints.push_back({ 50.0, 0.0 });
			linePoints.push_back({ 80.0, 00.0 });
			linePoints.push_back({ 80.0, +40.0 });
			linePoints.push_back({ 60.0, -50.0 });
			linePoints.push_back({ 40.0, +30.0 });
			currentDrawBuffers.addLines(std::move(linePoints));

			std::vector<double2> linePoints2;
			linePoints2.push_back({ -50.0, 0.0 });
			linePoints2.push_back({ 50.0, 0.0 });
			linePoints2.push_back({ 50.0, 50.0 });
			linePoints2.push_back({ -40.0, -20.0 });
			currentDrawBuffers.addRoads(std::move(linePoints2));
		}


		auto& transferQueue = queues[CommonAPI::InitOutput::EQT_TRANSFER_UP];
		auto& cb = m_uploadCmdBuf[m_resourceIx];

		nbl::video::IGPUQueue::SSubmitInfo submit;
		submit.commandBufferCount = 1u;
		submit.commandBuffers = &cb.get();
		submit.signalSemaphoreCount = 1u;
		submit.pSignalSemaphores = &m_drawBufferUploadsFinished[m_resourceIx].get();
		submit.waitSemaphoreCount = 0u;
		submit.pWaitSemaphores = nullptr;
		submit.pWaitDstStageMask = nullptr;

		logicalDevice->resetFences(1, &m_drawBufferUploadsComplete[m_resourceIx].get());

		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
		currentDrawBuffers.finalizeCopiesToGPU(utilities, transferQueue, m_drawBufferUploadsComplete[m_resourceIx].get(), submit);
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
		initParams.physicalDeviceFilter.requiredFeatures.fragmentShaderPixelInterlock = FragmentShaderPixelInterlock;
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
		const auto& transferCommandPools = commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP];

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

		video::IGPUDescriptorSetLayout::SBinding bindings[3u] = {};
		bindings[0u].binding = 0u;
		bindings[0u].type = asset::EDT_UNIFORM_BUFFER;
		bindings[0u].count = 1u;
		bindings[0u].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

		bindings[1u].binding = 1u;
		bindings[1u].type = asset::EDT_STORAGE_BUFFER;
		bindings[1u].count = 1u;
		bindings[1u].stageFlags = asset::IShader::ESS_VERTEX | asset::IShader::ESS_FRAGMENT;

		bindings[2u].binding = 2u;
		bindings[2u].type = asset::EDT_STORAGE_IMAGE;
		bindings[2u].count = 1u;
		bindings[2u].stageFlags = asset::IShader::ESS_FRAGMENT;
		auto descriptorSetLayout = logicalDevice->createDescriptorSetLayout(bindings, bindings+3u);
		
		nbl::video::IDescriptorPool::SDescriptorPoolSize poolSizes[3u] =
		{
			{ nbl::asset::EDT_UNIFORM_BUFFER, FRAMES_IN_FLIGHT },
			{ nbl::asset::EDT_STORAGE_BUFFER, FRAMES_IN_FLIGHT },
			{ nbl::asset::EDT_STORAGE_IMAGE, FRAMES_IN_FLIGHT },
		};
		auto descriptorPool = logicalDevice->createDescriptorPool(nbl::video::IDescriptorPool::ECF_NONE, 128u, 3u, poolSizes);

		for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
		{
			descriptorSets[i] = logicalDevice->createDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(descriptorSetLayout));
			video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[3u] = {};
			descriptorInfos[0u].buffer.offset = 0u;
			descriptorInfos[0u].buffer.size = globalsBuffer[i]->getCreationParams().size;
			descriptorInfos[0u].desc = globalsBuffer[i];

			descriptorInfos[1u].buffer.offset = 0u;
			descriptorInfos[1u].buffer.size = drawBuffers[i].gpuDrawBuffers.drawObjectsBuffer->getCreationParams().size;
			descriptorInfos[1u].desc = drawBuffers[i].gpuDrawBuffers.drawObjectsBuffer;

			descriptorInfos[2u].image.imageLayout = asset::IImage::E_LAYOUT::EL_GENERAL;
			descriptorInfos[2u].image.sampler = nullptr;
			descriptorInfos[2u].desc = pseudoStencilImageView[i];

			video::IGPUDescriptorSet::SWriteDescriptorSet descriptorUpdates[3u] = {};
			descriptorUpdates[0u].dstSet = descriptorSets[i].get();
			descriptorUpdates[0u].binding = 0u;
			descriptorUpdates[0u].arrayElement = 0u;
			descriptorUpdates[0u].count = 1u;
			descriptorUpdates[0u].descriptorType = asset::EDT_UNIFORM_BUFFER;
			descriptorUpdates[0u].info = &descriptorInfos[0u];

			descriptorUpdates[1u].dstSet = descriptorSets[i].get();
			descriptorUpdates[1u].binding = 1u;
			descriptorUpdates[1u].arrayElement = 0u;
			descriptorUpdates[1u].count = 1u;
			descriptorUpdates[1u].descriptorType = asset::EDT_STORAGE_BUFFER;
			descriptorUpdates[1u].info = &descriptorInfos[1u];

			descriptorUpdates[2u].dstSet = descriptorSets[i].get();
			descriptorUpdates[2u].binding = 2u;
			descriptorUpdates[2u].arrayElement = 0u;
			descriptorUpdates[2u].count = 1u;
			descriptorUpdates[2u].descriptorType = asset::EDT_STORAGE_IMAGE;
			descriptorUpdates[2u].info = &descriptorInfos[2u];

			logicalDevice->updateDescriptorSets(3u, descriptorUpdates, 0u, nullptr);
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

			logicalDevice->createCommandBuffers(
				transferCommandPools[i].get(),
				video::IGPUCommandBuffer::EL_PRIMARY,
				1,
				m_uploadCmdBuf + i);
		}

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			m_frameComplete[i] = logicalDevice->createFence(video::IGPUFence::ECF_SIGNALED_BIT);
			m_drawBufferUploadsComplete[i] = logicalDevice->createFence(video::IGPUFence::ECF_SIGNALED_BIT);
			m_imageAcquire[i] = logicalDevice->createSemaphore();
			m_renderFinished[i] = logicalDevice->createSemaphore();
			m_drawBufferUploadsFinished[i] = logicalDevice->createSemaphore();
		}

		m_Camera.setOrigin({ 00.0, 0.0 });
		m_Camera.setAspectRatio((double)WIN_W / WIN_H);
		m_Camera.setSize(200.0);

		oracle.reportBeginFrameRecord();
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

		auto now = std::chrono::high_resolution_clock::now();
		dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
		lastTime = now;
		static double timeElapsed = 0.0;
		timeElapsed += dt;

		auto& cb = m_cmdbuf[m_resourceIx];
		auto& commandPool = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS][m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];
		logicalDevice->blockForFences(1u, &fence.get());
		logicalDevice->resetFences(1u, &fence.get());

		update(timeElapsed, m_resourceIx);

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
		globalData.lineWidth = 16.0f;
		globalData.antiAliasingFactor = 1.0f;// + abs(cos(timeElapsed * 0.0008))*20.0f;
		globalData.resolution = uint2{ WIN_W, WIN_H };
		globalData.viewProjection = m_Camera.constructViewProjection();
		cb->updateBuffer(globalsBuffer[m_resourceIx].get(), 0ull, sizeof(Globals), &globalData);

		// Clear pseudoStencil
		{
			auto pseudoStencilImage = pseudoStencilImageView[m_resourceIx]->getCreationParameters().image;

			nbl::video::IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
			imageBarriers[0].barrier.srcAccessMask = nbl::asset::EAF_NONE;
			imageBarriers[0].barrier.dstAccessMask = nbl::asset::EAF_NONE; // TODO?
			imageBarriers[0].oldLayout = nbl::asset::IImage::EL_UNDEFINED;
			imageBarriers[0].newLayout = nbl::asset::IImage::EL_GENERAL;
			imageBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageBarriers[0].image = pseudoStencilImage;
			imageBarriers[0].subresourceRange.aspectMask = nbl::asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(nbl::asset::EPSF_TOP_OF_PIPE_BIT, nbl::asset::EPSF_TRANSFER_BIT, nbl::asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);

			asset::SClearColorValue clear = {};
			clear.uint32[0] = 0u;
			
			asset::IImage::SSubresourceRange subresourceRange = {};
			subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			subresourceRange.baseArrayLayer = 0u;
			subresourceRange.baseMipLevel = 0u;
			subresourceRange.layerCount = 1u;
			subresourceRange.levelCount = 1u;

			cb->clearColorImage(pseudoStencilImage.get(), asset::IImage::EL_GENERAL, &clear, 1u, &subresourceRange);
		}

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
			clear[0].color.float32[0] = 0.8f;
			clear[0].color.float32[1] = 0.8f;
			clear[0].color.float32[2] = 0.8f;
			clear[0].color.float32[3] = 0.f;
			clear[1].depthStencil.depth = 1.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = framebuffersDynArraySmartPtr->begin()[imgnum];
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		cb->beginRenderPass(&beginInfo, asset::ESC_INLINE);

		const uint32_t currentIndexCount = drawBuffers[m_resourceIx].currentIndexCount;

		cb->bindDescriptorSets(asset::EPBP_GRAPHICS, graphicsPipelineLayout.get(), 0u, 1u, &descriptorSets[m_resourceIx].get());
		cb->bindIndexBuffer(drawBuffers[m_resourceIx].gpuDrawBuffers.indexBuffer.get(), 0u, asset::EIT_32BIT);
		cb->bindGraphicsPipeline(graphicsPipeline.get());
		cb->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);

		if constexpr (DebugMode)
		{
			cb->bindGraphicsPipeline(debugGraphicsPipeline.get());
			cb->drawIndexed(currentIndexCount, 1u, 0u, 0u, 0u);
		}

		cb->endRenderPass();

		cb->end();

		auto& graphicsQueue = queues[CommonAPI::InitOutput::EQT_GRAPHICS];

		nbl::video::IGPUQueue::SSubmitInfo submit;
		submit.commandBufferCount = 1u;
		submit.commandBuffers = &cb.get();
		submit.signalSemaphoreCount = 1u;
		submit.pSignalSemaphores = &m_renderFinished[m_resourceIx].get();
		nbl::video::IGPUSemaphore* waitSemaphores[2u] = { m_imageAcquire[m_resourceIx].get(), m_drawBufferUploadsFinished[m_resourceIx].get() };
		asset::E_PIPELINE_STAGE_FLAGS waitStages[2u] = { nbl::asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT, nbl::asset::EPSF_VERTEX_INPUT_BIT };
		submit.waitSemaphoreCount = 2u;
		submit.pWaitSemaphores = waitSemaphores;
		submit.pWaitDstStageMask = waitStages;
		graphicsQueue->submit(1u, &submit, fence.get());

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