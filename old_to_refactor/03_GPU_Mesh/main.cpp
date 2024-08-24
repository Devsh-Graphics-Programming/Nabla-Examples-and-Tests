
#include "CCamera.hpp"


#include "nbl/nblpack.h"
struct VertexStruct
{
    /// every member needs to be at location aligned to its type size for GLSL
    float Pos[3]; /// uses float hence need 4 byte alignment
    uint8_t Col[2]; /// same logic needs 1 byte alignment
    uint8_t uselessPadding[2]; /// so if there is a member with 4 byte alignment then whole struct needs 4 byte align, so pad it
} PACK_STRUCT;
#include "nbl/nblunpack.h"

const char* vertexSource = R"===(
#version 430 core

layout(location = 0) in vec4 vPos; //only a 3d position is passed from Nabla, but last (the W) coordinate gets filled with default 1.0
layout(location = 1) in vec4 vCol;

layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} PushConstants;

layout(location = 0) out vec4 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    gl_Position = PushConstants.modelViewProj*vPos; //only thing preventing the shader from being core-compliant
    Color = vCol;
}
)===";

const char* fragmentSource = R"===(
#version 430 core

layout(location = 0) in vec4 Color; //per vertex output color, will be interpolated across the triangle

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = Color;
}
)===";

class GPUMesh : public ApplicationBase
{

public:

	nbl::core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
	nbl::core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

	CommonAPI::InputSystem::ChannelReader<ui::IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<ui::IKeyboardEventChannel> keyboard;
	Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());

	int resourceIx = -1;
	uint32_t acquiredNextFBO = {};
	std::chrono::system_clock::time_point lastTime;
	bool frameDataFilled = false;
	size_t frame_count = 0ull;
	double time_sum = 0;
	double dtList[NBL_FRAMES_TO_AVERAGE] = {};

	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

	nbl::video::ISwapchain::SCreationParams m_swapchainCreationParams;




	void onAppInitialized_impl() override
	{

		for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
			dtList[i] = 0.0;

		matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.1, 1000);
		camera = Camera(core::vectorSIMDf(-4, 0, 0), core::vectorSIMDf(0, 0, 0), projectionMatrix);
	}

	void workLoopBody() override
	{

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
		mouse.consumeEvents([&](const ui::IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const ui::IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
		camera.endInputProcessing(nextPresentationTimeStamp);

		const auto& mvp = camera.getConcatenatedMatrix();








		asset::SViewport viewport;
		viewport.minDepth = 1.f;
		viewport.maxDepth = 0.f;
		viewport.x = 0u;
		viewport.y = 0u;
		viewport.width = WIN_W;
		viewport.height = WIN_H;
		commandBuffer->setViewport(0u, 1u, &viewport);






		//! Stress test for memleaks aside from demo how to create meshes that live on the GPU RAM
		{
			VertexStruct vertices[8];
			vertices[0] = VertexStruct{ {-1.f,-1.f,-1.f},{  0,  0} };
			vertices[1] = VertexStruct{ { 1.f,-1.f,-1.f},{127,  0} };
			vertices[2] = VertexStruct{ {-1.f, 1.f,-1.f},{255,  0} };
			vertices[3] = VertexStruct{ { 1.f, 1.f,-1.f},{  0,127} };
			vertices[4] = VertexStruct{ {-1.f,-1.f, 1.f},{127,127} };
			vertices[5] = VertexStruct{ { 1.f,-1.f, 1.f},{255,127} };
			vertices[6] = VertexStruct{ {-1.f, 1.f, 1.f},{  0,255} };
			vertices[7] = VertexStruct{ { 1.f, 1.f, 1.f},{127,255} };

			uint16_t indices_indexed16[] =
			{
				0,1,2,1,2,3,
				4,5,6,5,6,7,
				0,1,4,1,4,5,
				2,3,6,3,6,7,
				0,2,4,2,4,6,
				1,3,5,3,5,7
			};

			//	auto upStreamBuff = driver->getDefaultUpStreamingBuffer();
			//	core::smart_refctd_ptr<video::IGPUBuffer> upStreamRef(upStreamBuff->getBuffer());

			//	const void* dataToPlace[2] = { vertices,indices_indexed16 };
			//	uint32_t offsets[2] = { video::StreamingTransientDataBufferMT<>::invalid_address,video::StreamingTransientDataBufferMT<>::invalid_address };
			//	uint32_t alignments[2] = { sizeof(decltype(vertices[0u])),sizeof(decltype(indices_indexed16[0u])) };
			//	uint32_t sizes[2] = { sizeof(vertices),sizeof(indices_indexed16) };
			//	upStreamBuff->multi_place(2u, (const void* const*)dataToPlace, (uint32_t*)offsets, (uint32_t*)sizes, (uint32_t*)alignments);
			//	if (upStreamBuff->needsManualFlushOrInvalidate())
			//	{
			//		auto upStreamMem = upStreamBuff->getBuffer()->getBoundMemory();
			//		driver->flushMappedMemoryRanges({ video::IDeviceMemoryAllocation::MappedMemoryRange(upStreamMem,offsets[0],sizes[0]),video::IDeviceMemoryAllocation::MappedMemoryRange(upStreamMem,offsets[1],sizes[1]) });
			//	}

			//	asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };

			//	auto createSpecializedShaderFromSource = [=](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage)
			//	{
			//		auto spirv = device->getAssetManager()->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID");
			//		auto unspec = driver->createShader(std::move(spirv));
			//		return driver->createSpecializedShader(unspec.get(), { nullptr,nullptr,"main",stage });
			//	};
			//	// origFilepath is only relevant when you have filesystem #includes in your shader
			//	auto createSpecializedShaderFromSourceWithIncludes = [&](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage, const char* origFilepath)
			//	{
			//		auto resolved_includes = device->getAssetManager()->getGLSLCompiler()->resolveIncludeDirectives(source, stage, origFilepath);
			//		return createSpecializedShaderFromSource(reinterpret_cast<const char*>(resolved_includes->getContent()->getPointer()), stage);
			//	};
			//	core::smart_refctd_ptr<video::IGPUSpecializedShader> shaders[2] =
			//	{
			//		createSpecializedShaderFromSourceWithIncludes(vertexSource,asset::ISpecializedShader::ESS_VERTEX, "shader.vert"),
			//		createSpecializedShaderFromSource(fragmentSource,asset::ISpecializedShader::ESS_FRAGMENT)
			//	};
			//	auto shadersPtr = reinterpret_cast<video::IGPUSpecializedShader**>(shaders);

			//	asset::SVertexInputParams inputParams;
			//	inputParams.enabledAttribFlags = 0b11u;
			//	inputParams.enabledBindingFlags = 0b1u;
			//	inputParams.attributes[0].binding = 0u;
			//	inputParams.attributes[0].format = asset::EF_R32G32B32_SFLOAT;
			//	inputParams.attributes[0].relativeOffset = offsetof(VertexStruct, Pos[0]);
			//	inputParams.attributes[1].binding = 0u;
			//	inputParams.attributes[1].format = asset::EF_R8G8_UNORM;
			//	inputParams.attributes[1].relativeOffset = offsetof(VertexStruct, Col[0]);
			//	inputParams.bindings[0].stride = sizeof(VertexStruct);
			//	inputParams.bindings[0].inputRate = asset::EVIR_PER_VERTEX;

			//	asset::SBlendParams blendParams; // defaults are sane

			//	asset::SPrimitiveAssemblyParams assemblyParams = { asset::EPT_TRIANGLE_LIST,false,1u };

			//	asset::SStencilOpParams defaultStencil;
			//	asset::SRasterizationParams rasterParams;
			//	rasterParams.faceCullingMode = asset::EFCM_NONE;
			//	auto pipeline = driver->createRenderpassIndependentPipeline(nullptr, driver->createPipelineLayout(range, range + 1u, nullptr, nullptr, nullptr, nullptr),
			//		shadersPtr, shadersPtr + sizeof(shaders) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>),
			//		inputParams, blendParams, assemblyParams, rasterParams);

			//	asset::SBufferBinding<video::IGPUBuffer> bindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
			//	bindings[0u] = { offsets[0],upStreamRef };
			//	auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(std::move(pipeline), nullptr, bindings, asset::SBufferBinding<video::IGPUBuffer>{offsets[1], upStreamRef});
			//	{
			//		mb->setIndexType(asset::EIT_16BIT);
			//		mb->setIndexCount(2 * 3 * 6);
			//	}

			//	driver->bindGraphicsPipeline(mb->getPipeline());
			//	driver->pushConstants(mb->getPipeline()->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
			//	driver->drawMeshBuffer(mb.get());

			//	upStreamBuff->multi_free(2u, (uint32_t*)&offsets, (uint32_t*)&sizes, driver->placeFence());
			//}
			//driver->endScene();
		}
	}
};
