#ifndef _NBL_THIS_EXAMPLE_APP_HPP_
#define _NBL_THIS_EXAMPLE_APP_HPP_

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include "argparse/argparse.hpp"

#include "common.hpp"
#include "app/AppSwapchainResources.hpp"
#include "keysmapping.hpp"
#include "app/AppTypes.hpp"
#include "app/AppViewportBindingUtilities.hpp"
#include "camera/CCubeProjection.hpp"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

namespace nbl::system
{
	struct SCameraAppResourceContext;
	struct SCameraConfigCollections;
	struct SCameraPlanarRuntimeBootstrap;
	struct CCameraScriptedInputParseResult;
}

class App final : public examples::SimpleWindowedApplication, public examples::BuiltinResourcesApplication
{
	using base_t = examples::SimpleWindowedApplication;
	using asset_base_t = examples::BuiltinResourcesApplication;
	using clock_t = std::chrono::steady_clock;

	struct SpaceEnvPushConstants
	{
		float32_t4x4 invProj = float32_t4x4(1.f);
		float32_t4x4 invViewRot = float32_t4x4(1.f);
		uint32_t orthoMode = 0u;
		uint32_t pad0 = 0u;
		uint32_t pad1 = 0u;
		uint32_t pad2 = 0u;
	};

	public:
	using base_t::base_t;

	inline App(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) 
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	// Will get called mid-initialization, via `filterDevices` between when the API Connection is created and Physical Device is chosen
	core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override;

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override;
	core::bitflag<nbl::system::ILogger::E_LOG_LEVEL> getLogLevelMask() override
	{
		return core::bitflag<nbl::system::ILogger::E_LOG_LEVEL>(nbl::system::ILogger::ELL_INFO) |
			nbl::system::ILogger::ELL_WARNING |
			nbl::system::ILogger::ELL_PERFORMANCE |
			nbl::system::ILogger::ELL_ERROR;
	}

		bool updateGUIDescriptorSet();

		void workLoopBody() override;

		void paceScriptedVisualDebugFrame();

		bool keepRunning() override;
		bool onAppTerminated() override;

		void update();
		bool runHeadlessCameraSmoke(argparse::ArgumentParser& program, smart_refctd_ptr<ISystem>&& system);

		private:
		bool initializeMountedCameraResources(smart_refctd_ptr<ISystem>&& system);
		nbl::hlsl::uint32_t2 getPresentationRenderExtent() const;
		bool shouldMaximizePresentationWindow() const;
		using CameraPreset = CCameraPreset;
		using CameraKeyframe = CCameraKeyframe;
		using CameraKeyframeTrack = CCameraKeyframeTrack;

		using PresetFilterMode = EPresetApplyPresentationFilter;
		using PresetUiAnalysis = SCameraGoalApplyPresentation;
		using CaptureUiAnalysis = SCameraCapturePresentation;

		using CameraConstraintSettings = SCameraConstraintSettings;

		ICamera* getActiveCamera();
		uint32_t getActivePlanarIx() const;
		inline std::span<smart_refctd_ptr<planar_projection_t>> getPlanarProjectionSpan()
		{
			return { m_planarProjections.data(), m_planarProjections.size() };
		}
		inline std::span<const smart_refctd_ptr<planar_projection_t>> getPlanarProjectionSpan() const
		{
			return { m_planarProjections.data(), m_planarProjections.size() };
		}
		nbl::system::SCameraAppResourceContext getCameraAppResourceContext() const;
		SCameraFollowConfig* getActiveFollowConfig();
		const SCameraFollowConfig* getActiveFollowConfig() const;
		SActiveViewportRuntimeState tryGetActiveViewportRuntimeState();
		bool tryBuildActiveCameraInputContext(SActiveCameraInputContext& outContext);
		bool tryBuildActiveProjectionTabContext(SActiveProjectionTabContext& outContext);
		bool tryBuildActiveScriptedCameraContext(SActiveScriptedCameraContext& outContext);

		uint32_t getManipulableObjectCount() const;
		bool isManipulableObjectFollowTarget(uint32_t objectIx) const;
		std::optional<uint32_t> getManipulableObjectPlanarIx(uint32_t objectIx) const;
		bool tryBuildManipulableObjectContext(uint32_t objectIx, SManipulableObjectContext& outContext) const;
		bool tryBuildActiveManipulatedObjectContext(SManipulableObjectContext& outContext) const;
		uint32_t getManipulatedObjectIx() const;
		void bindManipulatedModel();
		void bindManipulatedFollowTarget();
		void bindManipulatedCamera(uint32_t planarIx);
		void bindManipulatedObjectByIx(uint32_t objectIx);
		void bindManipulableObject(const SManipulableObjectContext& context);
		std::string getManipulableObjectLabel(uint32_t objectIx) const;
		float32_t4x4 getManipulableObjectTransform(uint32_t objectIx) const;
		float32_t3 getManipulableObjectWorldPosition(uint32_t objectIx) const;
		float32_t3x4 computeFollowTargetMarkerWorld() const;
		void applyManipulableObjectTransform(const SManipulableObjectContext& context, const float64_t4x4& transform);

		void setFollowTargetTransform(const float64_t4x4& transform);

		bool captureFollowOffsetsForPlanar(uint32_t planarIx);
		bool followConfigUsesCapturedOffset(const SCameraFollowConfig& config) const;
		void refreshFollowOffsetConfigForPlanar(uint32_t planarIx);
		void refreshFollowOffsetConfigsForCamera(ICamera* camera);
		void refreshAllFollowOffsetConfigs();
		float64_t3 getDefaultFollowTargetPosition() const;
		camera_quaternion_t<float64_t> getDefaultFollowTargetOrientation() const;
		SCameraFollowConfig makeExampleDefaultFollowConfig(const ICamera* camera) const;
		void resetFollowTargetToDefault();
		void snapFollowTargetToModel();
		void applyFollowToConfiguredCameras(bool allowDuringScriptedInput = false);
		bool isOrbitLikeCamera(ICamera* camera);
		void syncVisualDebugWindowBindings();
		void drawScriptVisualDebugOverlay(const ImVec2& displaySize);

		bool tryCaptureGoal(ICamera* camera, CCameraGoal& out) const;
		PresetUiAnalysis analyzePresetForUi(ICamera* camera, const CameraPreset& preset) const;
		CaptureUiAnalysis analyzeCameraCaptureForUi(ICamera* camera) const;
		CCameraGoalSolver::SCompatibilityResult analyzePresetCompatibility(ICamera* camera, const CameraPreset& preset) const;
		bool presetMatchesFilter(ICamera* camera, const CameraPreset& preset) const;
		CCameraGoalSolver::SApplyResult applyPresetFromUi(ICamera* camera, const CameraPreset& preset);
		void storeApplyStatusBanner(ApplyStatusBanner& banner, std::string summary, bool succeeded, bool approximate);
		void clearApplyStatusBanner(ApplyStatusBanner& banner);
		void storePlaybackApplySummary(const SCameraPresetApplySummary& summary);
		void appendVirtualEventLog(std::string_view source, std::string_view inputSource, uint32_t planarIx, ICamera* camera, const CVirtualGimbalEvent* events, uint32_t count);
		SCameraPresetApplySummary applyPresetToTargets(const CameraPreset& preset);
		bool tryBuildPlaybackPresetAtTime(float time, CameraPreset& preset);
		bool applyPlaybackAtTime(float time);
		void sortKeyframesByTime();
		void clampPlaybackTimeToKeyframes();
		int selectKeyframeNearestTime(float time);
		void normalizeSelectedKeyframe();
		CameraKeyframe* getSelectedKeyframe();
		const CameraKeyframe* getSelectedKeyframe() const;
		bool replaceSelectedKeyframeFromCamera(ICamera* camera);
		void updatePlayback(double dtSec);

		bool savePresetsToFile(const nbl::system::path& path);
		bool loadPresetsFromFile(const nbl::system::path& path);
		bool saveKeyframesToFile(const nbl::system::path& path);
		bool loadKeyframesFromFile(const nbl::system::path& path);

		void imguiListen();
		void drawWindowedViewportWindows(ImGuiIO& io, SImResourceInfo& info);
		void drawWindowedViewportWindow(uint32_t windowIx, ImGuiCond windowCond, bool hideSceneGizmos, size_t& gizmoIx, SImResourceInfo& info);
		void drawViewportWindowOverlay(
			ImDrawList& drawList,
			const nbl::ui::SViewportOverlayRect& viewportRect,
			uint32_t windowIx,
			const SWindowControlBinding& binding,
			const nbl::ui::SBoundViewportCameraState& viewportState) const;
		void updateActiveRenderWindowFromViewport(uint32_t windowIx, bool windowHovered, bool windowFocused);
		void drawViewportManipulationGizmos(
			uint32_t windowIx,
			SWindowControlBinding& binding,
			const nbl::ui::SBoundViewportCameraState& viewportState,
			size_t& gizmoIx);
		void drawManipulableObjectHoverOverlay(const SManipulableObjectContext& objectContext) const;
		void drawViewportSplitOverlayWindow(const ImVec2& displaySize);
		void drawFullscreenViewportWindow(ImGuiIO& io, SImResourceInfo& info);
		void refreshViewportBindingMatrices();
		void finalizeUiFrameState();
		void updatePresentationTiming();
		SCapturedUiEvents captureUiInputEvents();
		void buildCameraInputEvents(const SCapturedUiEvents& capturedEvents, std::vector<SKeyboardEvent>& outKeyboardEvents, std::vector<SMouseEvent>& outMouseEvents) const;
		nbl::ext::imgui::UI::SUpdateParameters buildUiUpdateParameters(const SCapturedUiEvents& capturedEvents) const;
		void prepareScriptedFrameState(SAppFrameUpdateState::SPreparedScriptedFrame& outState);
		void prepareCapturedCameraInput(const SAppFrameUpdateState::SPreparedScriptedFrame& scriptedState, SAppFrameUpdateState::SPreparedCapturedInput& outCameraInput);
		void prepareUiRuntimeState(const SAppFrameUpdateState::SPreparedCapturedInput& cameraInput, SAppFrameUpdateState::SUiRuntimeState& outUiState);
		void prepareCameraAndUiInput(const SAppFrameUpdateState::SPreparedScriptedFrame& scriptedState, SAppFrameUpdateState::SPreparedCapturedInput& outCameraInput, SAppFrameUpdateState::SUiRuntimeState& outUiState);
		SAppFrameUpdateState buildFrameUpdateState();
		void runCameraFramePasses(SAppFrameUpdateState& frameState);
		void applyPreparedCameraInput(const SAppFrameUpdateState::SPreparedCapturedInput& cameraInput, bool skipCameraInput);
		void runPreparedScriptedFrame(SAppFrameUpdateState::SPreparedScriptedFrame& scriptedState);
		void updateUiFrame(const SAppFrameUpdateState::SUiRuntimeState& uiState);
		void applyFrameRuntimeState(SAppFrameUpdateState& frameState);
		bool initializeCameraConfiguration(const argparse::ArgumentParser& program);
		bool tryBuildCameraConfigurationBootstrap(
			const argparse::ArgumentParser& program,
			nbl::system::SCameraPlanarRuntimeBootstrap& outRuntimeBootstrap,
			std::optional<CCameraSequenceScript>& outPendingScriptedSequence);
		bool initializePlanarRuntimeState(const nbl::system::SCameraPlanarRuntimeBootstrap& runtimeBootstrap, const std::optional<CCameraSequenceScript>& pendingScriptedSequence);
		void initializePlanarFollowConfigs();
		bool tryLoadConfiguredScriptedInput(const argparse::ArgumentParser& program, const nbl::system::SCameraConfigCollections& cameraCollections, std::optional<CCameraSequenceScript>& outPendingScriptedSequence);
		bool initializePresentationResources();
		bool initializeUiResources();
		bool initializeSceneResources();
		bool initializeGeometrySceneResources();
		bool initializeSceneRenderpass();
		bool initializeSpaceEnvironmentResources();
		bool initializeDebugSceneRendererResources();
		bool initializeWindowSceneFramebufferResources();
		uint32_t getFramesInFlight() const;
		bool waitForInflightFrameSlot();
		std::optional<SFrameSubmissionContext> tryBuildFrameSubmissionContext();
		bool recordFramePasses(const SFrameSubmissionContext& frameContext);
		bool submitAndPresentFrame(const SFrameSubmissionContext& frameContext);
		void resetScriptedInputRuntimeState();
		void finalizeScriptedInputRuntimeState();
		void applyParsedScriptedInput(nbl::system::CCameraScriptedInputParseResult parsed, std::optional<CCameraSequenceScript>& pendingScriptedSequence);
		std::optional<uint32_t> resolveSequenceSegmentPlanarIx(const CCameraSequenceSegment& segment) const;
		bool expandPendingScriptedSequence(const CCameraSequenceScript& sequence);
		void dequeueScriptedFrameInput(SScriptedFrameInputState& outFrame);
		void applyScriptedFrameActions(const CCameraScriptedFrameEvents& scriptedFrameEvents);
		void ensureScriptedVisualPlanarState();
		void updateScriptedMouseButtons(std::span<const SMouseEvent> scriptedMouse);
		void appendScriptedInputEvents(const SScriptedFrameInputState& scriptedFrame, SCapturedUiEvents& capturedEvents);
		void syncDynamicPerspectiveForPlanar(planar_projection_t* planar, ICamera* camera);
		void logScriptedVirtualEvents(const char* label, std::span<const CVirtualGimbalEvent> events) const;
		void applyActiveCameraInput(std::span<const SKeyboardEvent> keyboardEvents, std::span<const SMouseEvent> mouseEvents, bool skipCameraInput);
		void applyScriptedImguizmoInput(SScriptedFrameInputState& scriptedFrame, bool skipCameraInput);
		void applyScriptedGoals(const CCameraScriptedFrameEvents& scriptedFrameEvents, bool skipCameraInput);
		void logScriptedCameraPose(const char* label, ICamera* camera) const;
		void updateScriptedFollowVisualState(const CCameraScriptedFrameEvents& scriptedFrameEvents);
		void runActiveFrameScriptedChecks(const SScriptedFrameInputState& scriptedFrame);
		void updateSceneDebugInstances();
		void updateAuxSceneInstances(size_t geometryCount);
		bool recordSceneFramebufferPass(IGPUCommandBuffer* cmdbuf, SWindowControlBinding& binding, uint32_t bindingIx);
		bool recordUiRenderPass(IGPUCommandBuffer* cmdbuf, uint32_t resourceIx);
		void captureRenderedFrame(IGPUImage* frame, uint64_t renderedFrameIx, const nbl::system::path& outPath, const char* tag);
		void handleFrameCaptureRequests(IGPUImage* frame, uint64_t renderedFrameIx);

		bool shouldCaptureOSCursor();
		void UpdateBoundCameraMovement();
		void UpdateCursorVisibility();
		void UpdateUiMetrics();

		void DrawControlPanel();
		void drawControlPanelTabs(const nbl::ui::SCameraControlPanelStyle& panelStyle);
		void drawControlPanelHeader(const nbl::ui::SCameraControlPanelStyle& panelStyle);
		void drawControlPanelToggles(const nbl::ui::SCameraControlPanelStyle& panelStyle);
		void drawControlPanelStatusTab(const nbl::ui::SCameraControlPanelStyle& panelStyle);
		void drawControlPanelProjectionTab(const nbl::ui::SCameraControlPanelStyle& panelStyle);
		void drawControlPanelCameraTab(const nbl::ui::SCameraControlPanelStyle& panelStyle);
		void drawControlPanelPresetsTab(const nbl::ui::SCameraControlPanelStyle& panelStyle);
		void drawControlPanelPlaybackTab(const nbl::ui::SCameraControlPanelStyle& panelStyle);
		void drawControlPanelGizmoTab(const nbl::ui::SCameraControlPanelStyle& panelStyle);
		void drawControlPanelLogTab(const nbl::ui::SCameraControlPanelStyle& panelStyle);

		void TransformEditorContents();

		void addMatrixTable(const char* topText, const char* tableName, int rows, int columns, const float* pointer, bool withSeparator = true);

		std::chrono::seconds timeout = std::chrono::seconds(0x7fffFFFFu);
		clock_t::time_point start;

		/// @brief One window and surface.
		smart_refctd_ptr<CSmoothResizeSurface<CSwapchainResources>> m_surface;
		smart_refctd_ptr<IWindow> m_window;
		// We can't use the same semaphore for acquire and present, because that would disable "Frames in Flight" by syncing previous present against next acquire.
		// At least two timelines must be used.
		smart_refctd_ptr<ISemaphore> m_semaphore;
		// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
		constexpr static inline uint32_t MaxFramesInFlight = SCameraAppRuntimeDefaults::MaxFramesInFlight;
		// Use a separate counter to cycle through our resources because `getAcquireCount()` increases upon spontaneous resizes with immediate blit-presents 
		uint64_t m_realFrameIx = 0;
		// We'll write to the Triple Buffer with a Renderpass
		core::smart_refctd_ptr<IGPURenderpass> m_renderpass = {};
		// These are atomic counters where the Surface lets us know what's the latest Blit timeline semaphore value which will be signalled on the resource
		std::array<std::atomic_uint64_t, MaxFramesInFlight> m_blitWaitValues;
		// Enough Command Buffers and other resources for all frames in flight!
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
		// Our own persistent images that don't get recreated with the swapchain
		std::array<smart_refctd_ptr<IGPUImage>, MaxFramesInFlight> m_tripleBuffers;
		// Resources derived from the images
		std::array<core::smart_refctd_ptr<IGPUFramebuffer>, MaxFramesInFlight> m_framebuffers = {};
		// Input system for capturing system events
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		// Handles mouse events
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		// Handles keyboard events
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
		/// @brief Next presentation timestamp.
		std::chrono::microseconds m_nextPresentationTimestamp = {};

		core::smart_refctd_ptr<IDescriptorPool> m_descriptorSetPool;

		struct CRenderUI
		{
			nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> manager;

			struct
			{
				core::smart_refctd_ptr<video::IGPUSampler> gui, scene;
			} samplers;

			core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
		};

		SCameraAppSceneInteractionState m_sceneInteraction;

		std::vector<nbl::core::smart_refctd_ptr<planar_projection_t>> m_planarProjections;

		void syncWindowInputBinding(SWindowControlBinding& binding);
		void syncWindowInputBindingToProjection(SWindowControlBinding& binding);

		static constexpr inline auto MaxSceneFBOs = 2u;
		SCameraAppViewportSessionState<MaxSceneFBOs> m_viewports;

		// UI font atlas + viewport FBO color attachment textures
		constexpr static inline auto TotalUISampleTexturesAmount = 1u + MaxSceneFBOs;

		SCameraAppDebugSceneState m_debugScene;
		SCameraAppSpaceEnvironmentState m_spaceEnvironment;

		CRenderUI m_ui;
		video::CDumbPresentationOracle oracle;

		SCameraAppCliRuntimeState m_cliRuntime;
		SScriptedInputRuntimeState m_scriptedInput;
		CameraControlSettings m_cameraControls;
		CameraConstraintSettings m_cameraConstraints;
		core::smart_refctd_ptr<CUILogFormatter> m_logFormatter;
		SCameraAppEventLogState m_eventLog;
		SCameraAppPresetAuthoringState m_presetAuthoring;
		SCameraAppPlaybackAuthoringState m_playbackAuthoring;
		CCameraGoalSolver m_cameraGoalSolver;
		SCameraAppPresentationTimingState m_presentationTiming;
		SCameraAppUiMetricsState m_uiMetrics;
		SCameraAppGizmoState m_gizmoState;
};


#endif // _NBL_THIS_EXAMPLE_APP_HPP_

