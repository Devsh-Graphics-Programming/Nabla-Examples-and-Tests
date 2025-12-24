#pragma once

#include "common.hpp"
#include "nbl/ui/ICursorControl.h"
#include "MeshRenderer.hpp"




struct MeshletPush {
	float32_t4x4 viewProj; //nbl::core::matrix4SIMD is 128bit??
	constexpr static uint8_t object_type_count_max = 16;//it can go up til this struct hits the limit for push size
	uint32_t objectInstanceCount[object_type_count_max]; //this data is going to cropped before pushing, if necessary
};

class MeshSampleApp final : public MonoWindowApplication, public BuiltinResourcesApplication
{
		using device_base_t = MonoWindowApplication;
		using asset_base_t = BuiltinResourcesApplication;

	public:
		MeshSampleApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) 
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
			device_base_t({1280,720}, EF_UNKNOWN, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) 
        {}

		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override;
		virtual bool onAppTerminated();
		IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override;

	protected:
		const video::IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const override;
	private:
		void UpdateScene(nbl::video::IGPUCommandBuffer* cb);
		void update(const std::chrono::microseconds nextPresentationTimestamp);
		void recreateFramebuffer(const uint16_t2 resolution);
		void beginRenderpass(IGPUCommandBuffer* cb, const IGPUCommandBuffer::SRenderpassBeginInfo& info);

		// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
		constexpr static inline uint32_t MaxFramesInFlight = 3u;
		constexpr static inline auto sceneRenderDepthFormat = EF_D32_SFLOAT;
		constexpr static inline auto finalSceneRenderFormat = EF_R8G8B8A8_SRGB;
		constexpr static inline auto TexturesImGUIBindingIndex = 0u;
		// we create the Descriptor Set with a few slots extra to spare, so we don't have to `waitIdle` the device whenever ImGUI virtual window resizes
		constexpr static inline auto MaxImGUITextures = 2u+MaxFramesInFlight;

		smart_refctd_ptr<IGPURenderpass> m_renderpass;
		smart_refctd_ptr<IGPUFramebuffer> m_framebuffer;

		//i PROBABLY need to replace the debug renderer
		smart_refctd_ptr<MeshDebugRenderer> m_renderer;
		//
		smart_refctd_ptr<ISemaphore> m_semaphore;
		uint64_t m_realFrameIx = 0;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,MaxFramesInFlight> m_cmdBufs;
		//
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		core::smart_refctd_ptr<video::SubAllocatedDescriptorSet> meshlet_subAllocDS;
		smart_refctd_ptr<IGPUPipelineLayout> meshletLayout;
		smart_refctd_ptr<IGPUMeshPipeline> meshletPipeline;


		smart_refctd_ptr<IGPUBuffer> meshGPUBuffer;
		nbl::video::IDeviceMemoryAllocator::SAllocation mesh_allocation;
		// UI stuff
		//i really hate interface beign it's own object
		struct CInterface
		{
			bool cameraControlSeparated = false;
			void DrawCameraControls();

			bool guizmoEnabled = true;
			void UpdateImguizmo();

			void operator()();
			
			smart_refctd_ptr<ext::imgui::UI> imGUI;
			// descriptor set
			smart_refctd_ptr<SubAllocatedDescriptorSet> subAllocDS;
			SubAllocatedDescriptorSet::value_type renderColorViewDescIndex = SubAllocatedDescriptorSet::invalid_value;

			core::matrix3x4SIMD model;

			Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());

			TransformRequestParams transformParams;
			uint16_t2 sceneResolution = {1280,720};
			uint16_t4 widgetBox;
			float fov = 60.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
			float viewWidth = 10.f;
			float camYAngle = 165.f / 180.f * 3.14159f; //wheres my pi constant
			float camXAngle = 32.f / 180.f * 3.14159f;
			uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed
			bool isPerspective = true, isLH = true, flipGizmoY = true, move = false;
			bool firstFrame = true;

			ILogicalDevice::MappedMemoryRange meshMemoryRange;
			void* mesh_mapped_memory = nullptr;

		} interface;
};
