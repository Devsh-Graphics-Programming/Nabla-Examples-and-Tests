#ifndef _NBL_THIS_EXAMPLE_APP_TYPES_HPP_
#define _NBL_THIS_EXAMPLE_APP_TYPES_HPP_

#include <chrono>
#include <array>
#include <algorithm>
#include <deque>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common.hpp"

using planar_projections_range_t = std::vector<IPlanarProjection::CProjection>;
using planar_projection_t = CPlanarProjection<planar_projections_range_t>;

struct ImGuizmoPlanarM16InOut
{
	float32_t4x4 view, projection;
};

struct ImGuizmoModelM16InOut
{
	float32_t4x4 inTRS, outTRS, outDeltaTRS;
};

struct SWindowControlBinding final
{
	static inline constexpr uint32_t InvalidPlanarIx = std::numeric_limits<uint32_t>::max();

	nbl::core::smart_refctd_ptr<IGPUFramebuffer> sceneFramebuffer;
	nbl::core::smart_refctd_ptr<IGPUImageView> sceneColorView;
	nbl::core::smart_refctd_ptr<IGPUImageView> sceneDepthView;
	float32_t3x4 viewMatrix = float32_t3x4(1.f);
	float32_t4x4 projectionMatrix = float32_t4x4(1.f);
	float32_t4x4 viewProjMatrix = float32_t4x4(1.f);

	uint32_t activePlanarIx = 0u;
	bool allowGizmoAxesToFlip = false;
	bool enableDebugGridDraw = true;
	bool isOrthographicProjection = false;
	float aspectRatio = 16.f / 9.f;
	bool leftHandedProjection = true;
	CGimbalInputBinder inputBinding;

	std::optional<uint32_t> boundProjectionIx = std::nullopt;
	std::optional<uint32_t> lastBoundPerspectivePresetProjectionIx = std::nullopt;
	std::optional<uint32_t> lastBoundOrthoPresetProjectionIx = std::nullopt;
	std::optional<uint32_t> inputBindingProjectionIx = std::nullopt;
	uint32_t inputBindingPlanarIx = InvalidPlanarIx;

	inline void pickDefaultProjections(const planar_projections_range_t& projections)
	{
		auto init = [&](std::optional<uint32_t>& presetix, IPlanarProjection::CProjection::ProjectionType requestedType) -> void
		{
			for (uint32_t i = 0u; i < projections.size(); ++i)
			{
				const auto& params = projections[i].getParameters();
				if (params.m_type == requestedType)
				{
					presetix = i;
					break;
				}
			}

			assert(presetix.has_value());
		};

		init(lastBoundPerspectivePresetProjectionIx = std::nullopt, IPlanarProjection::CProjection::Perspective);
		init(lastBoundOrthoPresetProjectionIx = std::nullopt, IPlanarProjection::CProjection::Orthographic);
		boundProjectionIx = lastBoundPerspectivePresetProjectionIx.value();
		inputBindingProjectionIx = std::nullopt;
		inputBindingPlanarIx = InvalidPlanarIx;
	}
};

struct SCameraAppSceneDefaults final
{
	static inline constexpr uint32_t ModelObjectIx = 0u;
	static inline constexpr uint32_t FollowTargetObjectIx = 1u;
	static inline constexpr uint32_t CameraObjectIxOffset = 2u;
	static inline constexpr float FollowTargetMarkerScale = 0.28f;
	static inline constexpr float FollowTargetMarkerScaleVisualDebug = 0.6f;
    static inline const float64_t3 DefaultFollowTargetPosition = float64_t3(6.0, -4.5, 2.25);
    static inline const camera_quaternion_t<float64_t> DefaultFollowTargetOrientation = CCameraMathUtilities::makeIdentityQuaternion<float64_t>();
};

inline float32_t3x4 buildFollowTargetMarkerWorldTransform(
    const CTrackedTarget& trackedTarget,
    const float markerScale)
{
    const auto& targetGimbal = trackedTarget.getGimbal();
    const auto position = getCastedVector<float32_t>(targetGimbal.getPosition());
    const auto orientation = getCastedVector<float32_t>(targetGimbal.getOrientation().data);
    const auto markerTransform = hlsl::CCameraMathUtilities::composeTransformMatrix(
        position,
        CCameraMathUtilities::makeQuaternionFromComponents<float32_t>(orientation.x, orientation.y, orientation.z, orientation.w),
        float32_t3(markerScale, markerScale, markerScale));
    return float32_t3x4(hlsl::transpose(markerTransform));
}

struct SCameraAppViewportDefaults final
{
	static inline constexpr ImVec2 MinWindowSize = ImVec2(69.0f, 69.0f);
	static inline constexpr ImVec2 MaxWindowSize = ImVec2(7680.0f, 4320.0f);
	static inline constexpr float DefaultGizmoWorldRadius = 0.22f;
	static inline constexpr float FollowTargetGizmoWorldRadius = 0.35f;
	static inline constexpr float MinPerspectiveGizmoDepth = 0.001f;
	static inline constexpr bool FlipGizmoY = true;
	static inline constexpr float32_t2 WindowPaddingOffset = float32_t2(10.0f, 10.0f);
};

struct SCameraAppRuntimeDefaults final
{
	static inline constexpr uint32_t CiFramesBeforeCapture = 10u;
	static inline constexpr auto CiMaxRuntime = std::chrono::minutes(2);
	static inline constexpr auto DisplayImageDuration = std::chrono::milliseconds(900);
	static inline constexpr size_t VirtualEventLogMax = 128u;
	static inline constexpr size_t UiMetricSamples = 96u;
	static inline constexpr uint32_t MaxFramesInFlight = 3u;
};

struct SCameraAppUiTextureSlots final
{
	static inline constexpr uint32_t FontAtlas = nbl::ext::imgui::UI::FontAtlasTexId;
	static inline constexpr uint32_t FirstViewport = FontAtlas + 1u;

	static inline constexpr uint32_t viewport(const uint32_t windowIx)
	{
		return FirstViewport + windowIx;
	}

	static inline SImResourceInfo makeDefaultViewportResourceInfo()
	{
		SImResourceInfo info = {};
		info.samplerIx = static_cast<uint16_t>(nbl::ext::imgui::UI::DefaultSamplerIx::USER);
		return info;
	}
};

struct SCameraAppRenderDefaults final
{
	static inline constexpr auto SceneDepthFormat = EF_D32_SFLOAT;
	static inline constexpr auto FinalSceneFormat = EF_R8G8B8A8_SRGB;
	static inline constexpr IGPUCommandBuffer::SClearColorValue SceneClearColor = { .float32 = { 0.014f, 0.018f, 0.030f, 1.0f } };
	static inline constexpr IGPUCommandBuffer::SClearDepthStencilValue SceneClearDepth = { .depth = 0.0f };
};

struct SCameraAppPresentationDefaults final
{
	static inline constexpr nbl::hlsl::uint32_t2 CiWindowExtent = nbl::hlsl::uint32_t2(1280u, 720u);
	static inline constexpr nbl::hlsl::uint32_t2 WindowOrigin = nbl::hlsl::uint32_t2(32u, 32u);
};

struct SCameraAppFrameRuntimeDefaults final
{
	static inline constexpr float ViewportMinDepth = 1.0f;
	static inline constexpr float ViewportMaxDepth = 0.0f;
	static inline constexpr float32_t4 InverseViewRotationXyzMask = float32_t4(1.0f, 1.0f, 1.0f, 0.0f);
	static inline constexpr float32_t4 InverseViewRotationHomogeneousRow = float32_t4(0.0f, 0.0f, 0.0f, 1.0f);
	static inline constexpr IGPUCommandBuffer::SClearColorValue UiClearColor = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } };
};

struct SCameraAppSceneDebugDefaults final
{
	static inline constexpr float GridExtent = 32.0f;
	static inline constexpr float GridVerticalOffset = -0.5f;
};

struct SCameraAppInputDefaults final
{
	static inline constexpr float KeyboardScale = 0.00625f;
	static inline constexpr float UnitScale = 1.0f;
};

struct SCameraAppCameraFactoryDefaults final
{
	static inline constexpr double DefaultMoveScale = 0.01;
	static inline constexpr double DefaultRotateScale = 0.003;
	static inline constexpr double TargetRigMoveScale = 0.5;
};

struct SCameraAppProjectionUiDefaults final
{
	static inline constexpr float NearPlaneMin = 0.1f;
	static inline constexpr float NearPlaneMax = 100.0f;
	static inline constexpr float FarPlaneMin = 110.0f;
	static inline constexpr float FarPlaneMax = 10000.0f;
	static inline constexpr float PerspectiveFovMinDeg = 20.0f;
	static inline constexpr float PerspectiveFovMaxDeg = 150.0f;
	static inline constexpr float OrthoWidthMin = 1.0f;
	static inline constexpr float OrthoWidthMax = 30.0f;
};

struct SCameraAppControlPanelRangeDefaults final
{
	static inline constexpr float MotionScaleMin = 0.0001f;
	static inline constexpr float MotionScaleMax = 10.0f;
	static inline constexpr float InputScaleMin = 0.01f;
	static inline constexpr float InputScaleMax = 10.0f;
	static inline constexpr float ConstraintDistanceMin = 0.01f;
	static inline constexpr float ConstraintMinDistanceMax = 1000.0f;
	static inline constexpr float ConstraintMaxDistanceMax = 10000.0f;
	static inline constexpr float ConstraintAngleMinDeg = -180.0f;
	static inline constexpr float ConstraintAngleMaxDeg = 180.0f;
};

struct SCameraAppViewportLayoutDefaults final
{
	static inline constexpr float ControlPanelWidthRatio = 0.33f;
	static inline constexpr float ControlPanelMinWidth = 380.0f;
	static inline constexpr float ControlPanelMaxWidthRatio = 0.48f;
	static inline constexpr float RenderPaddingX = 0.0f;
	static inline constexpr float RenderPaddingY = 0.0f;
	static inline constexpr float SplitGap = 4.0f;
};

struct SCameraAppScriptedVisualDefaults final
{
	static inline constexpr float TargetFps = 60.0f;
	static inline constexpr float HoldSeconds = 3.0f;
	static inline constexpr std::string_view DefaultCapturePrefix = "script";
};

struct SCameraAppBindingEditorUiDefaults final
{
	static inline constexpr float TableColumnWeight = 0.33f;
	static inline constexpr ImVec2 ActionButtonSize = ImVec2(100.0f, 30.0f);
	static inline constexpr ImVec2 WindowInitialSize = ImVec2(600.0f, 400.0f);
	static inline constexpr ImVec4 ActiveStatusColor = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
	static inline constexpr ImVec4 InactiveStatusColor = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
};

struct SCameraAppTransformEditorUiDefaults final
{
	static inline constexpr ImVec4 GizmoActiveStatusColor = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
	static inline constexpr ImVec4 GizmoIdleStatusColor = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
	static inline constexpr float32_t4x4 IdentityTransform = float32_t4x4(1.0f);
	static inline constexpr float32_t3 IdentityScale = float32_t3(1.0f);
	static inline constexpr float32_t3 ZeroRotation = float32_t3(0.0f);
};

struct SCameraAppCliRuntimeState final
{
	bool ciMode = false;
	bool ciScreenshotDone = false;
	uint32_t ciFrameCounter = 0u;
	nbl::system::path ciScreenshotPath;
	std::chrono::steady_clock::time_point ciStartedAt = std::chrono::steady_clock::time_point::min();
	bool scriptVisualDebugCli = false;
	bool disableScreenshotsCli = false;
	bool headlessCameraSmokeMode = false;
	bool headlessCameraSmokePassed = false;
};

struct SCameraAppUiMetricsState final
{
	std::array<float, SCameraAppRuntimeDefaults::UiMetricSamples> frameMs = {};
	std::array<float, SCameraAppRuntimeDefaults::UiMetricSamples> inputCounts = {};
	std::array<float, SCameraAppRuntimeDefaults::UiMetricSamples> virtualCounts = {};
	uint32_t sampleIndex = 0u;
	uint32_t inputEventsThisFrame = 0u;
	uint32_t virtualEventsThisFrame = 0u;
	uint32_t lastInputEvents = 0u;
	uint32_t lastVirtualEvents = 0u;
	float lastFrameMs = 0.0f;
};

struct SCameraAppPresentationTimingState final
{
	std::chrono::microseconds lastPresentationTimestamp = {};
	bool hasLastPresentationTimestamp = false;
	double frameDeltaSec = 0.0;
};

struct SCameraAppGizmoState final
{
	bool useSnap = false;
	ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
	ImGuizmo::MODE mode = ImGuizmo::LOCAL;
	float snap[3] = { 1.0f, 1.0f, 1.0f };
};

struct SScriptedVisualPlanarState final
{
	bool valid = false;
	uint32_t planarIx = 0u;
	uint64_t startFrame = 0u;
	std::string segmentLabel;
};

struct SScriptedMouseButtonState final
{
	bool leftDown = false;
	bool rightDown = false;
};

struct SScriptedFramePacerState final
{
	bool initialized = false;
	std::chrono::steady_clock::time_point nextFrame = {};
};

struct SScriptedInputRuntimeState final
{
	bool enabled = false;
	bool log = false;
	bool exclusive = false;
	bool hardFail = false;
	bool visualDebug = false;
	float visualTargetFps = 0.f;
	float visualCameraHoldSeconds = 0.f;
	CCameraScriptedTimeline timeline = {};
	size_t nextEventIndex = 0u;
	CCameraScriptedCheckRuntimeState checkRuntime = {};
	size_t nextCaptureIndex = 0u;
	std::string capturePrefix = "script";
	nbl::system::path captureOutputDir;
	bool failed = false;
	bool summaryReported = false;
	SScriptedVisualPlanarState visualPlanar = {};
	SCameraFollowVisualMetrics visualFollow = {};
	SScriptedMouseButtonState scriptedMouseButtons = {};
	SScriptedFramePacerState framePacer = {};
};

struct SCapturedUiEvents final
{
	std::vector<SMouseEvent> mouse;
	std::vector<SKeyboardEvent> keyboard;

	inline void clear()
	{
		mouse.clear();
		keyboard.clear();
	}

	inline uint32_t getEventCount() const
	{
		return static_cast<uint32_t>(mouse.size() + keyboard.size());
	}
};

struct CUILogFormatter final : public nbl::system::ILogger
{
	CUILogFormatter() : ILogger(ILogger::DefaultLogMask()) {}

	std::string format(const E_LOG_LEVEL level, const std::string_view fmt, ...)
	{
		va_list args;
		va_start(args, fmt);
		auto out = constructLogString(fmt, level, args);
		va_end(args);
		if (!out.empty() && out.back() == '\n')
			out.pop_back();
		return out;
	}

protected:
	void log_impl(const std::string_view&, const E_LOG_LEVEL, va_list) override {}
};

struct VirtualEventLogEntry final
{
	uint64_t frame = 0u;
	CVirtualGimbalEvent::VirtualEventType type = CVirtualGimbalEvent::None;
	float64_t magnitude = 0.0;
	std::string source;
	std::string inputSource;
	std::string camera;
	uint32_t planarIx = 0u;
	std::string line;
};

struct CameraPlaybackState : CCameraPlaybackCursor
{
	bool overrideInput = true;
};

struct ApplyStatusBanner final
{
	std::string summary;
	bool succeeded = false;
	bool approximate = false;

	inline bool visible() const
	{
		return !summary.empty();
	}
};

struct SCameraAppAuthoringDefaults final
{
	static inline constexpr std::string_view DefaultPresetName = "Preset";
	static inline constexpr std::string_view DefaultPresetPath = "camera_presets.json";
	static inline constexpr std::string_view DefaultKeyframePath = "camera_keyframes.json";
	static inline constexpr int PresetListVisibleEntries = 6;
	static inline constexpr size_t EventLogVisibleEntries = 200u;
	static inline constexpr float PlaybackSpeedMin = 0.1f;
	static inline constexpr float PlaybackSpeedMax = 4.0f;
	static inline constexpr float KeyframeTimeStep = 0.1f;
	static inline constexpr float KeyframeTimeFastStep = 1.0f;
};

struct SCameraAppEventLogState final
{
	std::deque<VirtualEventLogEntry> entries = {};
	bool showHud = true;
	bool showEventLog = false;
	bool autoScroll = true;
	bool wrap = true;
};

struct SCameraAppPresetAuthoringState final
{
	std::vector<nbl::core::CCameraPreset> presets = {};
	std::vector<nbl::core::CCameraPreset> initialPlanarPresets = {};
	ApplyStatusBanner applyBanner = {};
	nbl::ui::EPresetApplyPresentationFilter filterMode = nbl::ui::EPresetApplyPresentationFilter::All;
	int selectedPresetIx = -1;
	std::string presetName = std::string(SCameraAppAuthoringDefaults::DefaultPresetName);
	std::string presetPath = std::string(SCameraAppAuthoringDefaults::DefaultPresetPath);
};

struct SCameraAppPlaybackAuthoringState final
{
	nbl::core::CCameraKeyframeTrack keyframeTrack = {};
	CameraPlaybackState playback = {};
	ApplyStatusBanner applyBanner = {};
	bool affectsAll = false;
	float newKeyframeTime = 0.f;
	std::string keyframePath = std::string(SCameraAppAuthoringDefaults::DefaultKeyframePath);
};

enum class SceneManipulatedObjectKind : uint8_t
{
	Model,
	FollowTarget,
	Camera
};

struct SManipulableObjectContext final
{
	uint32_t objectIx = SCameraAppSceneDefaults::ModelObjectIx;
	SceneManipulatedObjectKind kind = SceneManipulatedObjectKind::Model;
	std::optional<uint32_t> planarIx = std::nullopt;
	ICamera* camera = nullptr;
	std::string label = "Model";
	float32_t4x4 transform = float32_t4x4(1.0f);
	float32_t3 worldPosition = float32_t3(0.0f);

	inline bool isCamera() const
	{
		return kind == SceneManipulatedObjectKind::Camera && camera;
	}

	inline bool isFollowTarget() const
	{
		return kind == SceneManipulatedObjectKind::FollowTarget;
	}
};

struct SCameraAppSceneInteractionState final
{
	float32_t3x4 model = float32_t3x4(1.0f);
	CTrackedTarget followTarget = {};
	std::vector<SCameraFollowConfig> planarFollowConfigs = {};
	bool followTargetVisible = true;
	SceneManipulatedObjectKind manipulatedObjectKind = SceneManipulatedObjectKind::Model;
	nbl::core::smart_refctd_ptr<ICamera> boundCameraToManipulate = nullptr;
	std::optional<uint32_t> boundPlanarCameraIxToManipulate = std::nullopt;
};

struct SCameraAppDebugSceneState final
{
	nbl::core::smart_refctd_ptr<CGeometryCreatorScene> scene = {};
	nbl::core::smart_refctd_ptr<IGPURenderpass> renderpass = {};
	nbl::core::smart_refctd_ptr<CSimpleDebugRenderer> renderer = {};
	std::optional<uint32_t> gridGeometryIx = std::nullopt;
	std::optional<uint32_t> followTargetGeometryIx = std::nullopt;
	uint16_t geometrySelectionIx = 0u;
};

struct SCameraAppSpaceEnvironmentState final
{
	core::smart_refctd_ptr<IGPUGraphicsPipeline> pipeline = {};
	core::smart_refctd_ptr<IGPUDescriptorSetLayout> descriptorSetLayout = {};
	core::smart_refctd_ptr<IDescriptorPool> descriptorPool = {};
	core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet = {};
	core::smart_refctd_ptr<IGPUImage> image = {};
	core::smart_refctd_ptr<IGPUImageView> imageView = {};
	core::smart_refctd_ptr<IGPUSampler> sampler = {};
};

struct CameraControlSettings final
{
	bool mirrorInput = false;
	bool worldTranslate = false;
	float keyboardScale = SCameraAppInputDefaults::KeyboardScale;
	float mouseMoveScale = SCameraAppInputDefaults::UnitScale;
	float mouseScrollScale = SCameraAppInputDefaults::UnitScale;
	float translationScale = SCameraAppInputDefaults::UnitScale;
	float rotationScale = SCameraAppInputDefaults::UnitScale;
};

struct SScriptedFrameInputState final
{
	CCameraScriptedFrameEvents frameEvents = {};
	std::vector<SMouseEvent> mouse;
	std::vector<SKeyboardEvent> keyboard;
	std::vector<CVirtualGimbalEvent> imguizmoVirtualEvents;

	inline bool hasRuntimePayload() const
	{
		return !keyboard.empty() ||
			!mouse.empty() ||
			!frameEvents.imguizmo.empty() ||
			!frameEvents.goals.empty() ||
			!frameEvents.trackedTargetTransforms.empty();
	}
};

struct SAppFrameUpdateState final
{
	struct SPreparedScriptedFrame final
	{
		bool skipCameraInput = false;
		SScriptedFrameInputState frame = {};
	};

	struct SPreparedCapturedInput final
	{
		SCapturedUiEvents capturedEvents = {};
		std::vector<SKeyboardEvent> keyboardEvents = {};
		std::vector<SMouseEvent> mouseEvents = {};
	};

	struct SUiRuntimeState final
	{
		nbl::ext::imgui::UI::SUpdateParameters updateParams = {};
	};

	SPreparedScriptedFrame scripted = {};
	SPreparedCapturedInput cameraInput = {};
	SUiRuntimeState ui = {};
};

struct SFrameSubmissionContext final
{
	uint32_t resourceIx = 0u;
	VkRect2D renderArea = {};
	IGPUImage* frame = nullptr;
	IGPUCommandBuffer* cmdbuf = nullptr;
	std::atomic_uint64_t* blitWaitValue = nullptr;
};

struct SViewportWindowFrame final
{
	ImVec2 contentRegionSize = {};
	ImVec2 cursorPos = {};
	nbl::ui::SViewportOverlayRect overlayRect = {};
	bool hovered = false;
	bool focused = false;
	bool mouseInside = false;
};

struct SImWindowInit final
{
	float32_t2 iPos = float32_t2(0.0f);
	float32_t2 iSize = float32_t2(0.0f);
};

template<size_t WindowCount>
struct SAppWindowInitState final
{
	SImWindowInit trsEditor = {};
	SImWindowInit planars = {};
	std::array<SImWindowInit, WindowCount> renderWindows = {};
};

template<size_t WindowCount>
struct SCameraAppViewportSessionState final
{
	bool enableActiveCameraMovement = false;
	bool captureCursorInMoveMode = false;
	bool resetCursorToCenter = true;
	bool useWindow = true;
	uint32_t activeRenderWindowIx = 0u;
	std::array<SWindowControlBinding, WindowCount> windowBindings = {};
	SAppWindowInitState<WindowCount> windowInit = {};
};

struct SActiveViewportRuntimeState final
{
	SWindowControlBinding* binding = nullptr;
	planar_projection_t* planar = nullptr;
	ICamera* camera = nullptr;

	inline bool valid() const
	{
		return binding && planar && camera;
	}

	inline SWindowControlBinding& requireBinding() const
	{
		assert(binding);
		return *binding;
	}

	inline planar_projection_t& requirePlanar() const
	{
		assert(planar);
		return *planar;
	}

	inline ICamera& requireCamera() const
	{
		assert(camera);
		return *camera;
	}
};

struct SActiveCameraInputContext final
{
	SActiveViewportRuntimeState viewport = {};

	inline bool valid() const
	{
		return viewport.valid();
	}
};

struct SActiveProjectionTabContext final
{
	SActiveViewportRuntimeState viewport = {};
	std::string activeRenderWindowIxString = {};
	std::string activePlanarIxString = {};

	inline bool valid() const
	{
		return viewport.valid();
	}

	inline SWindowControlBinding& requireBinding() const
	{
		return viewport.requireBinding();
	}

	inline planar_projection_t& requirePlanar() const
	{
		return viewport.requirePlanar();
	}

	inline ICamera& requireCamera() const
	{
		return viewport.requireCamera();
	}
};

struct SActiveScriptedCameraContext final
{
	SActiveViewportRuntimeState viewport = {};
	SCameraFollowConfig* followConfig = nullptr;
	nbl::system::SCameraProjectionContext projectionContext = {};
	bool hasProjectionContext = false;

	inline bool valid() const
	{
		return viewport.valid();
	}

	inline SWindowControlBinding& requireBinding() const
	{
		return viewport.requireBinding();
	}

	inline planar_projection_t& requirePlanar() const
	{
		return viewport.requirePlanar();
	}

	inline ICamera& requireCamera() const
	{
		return viewport.requireCamera();
	}

	inline const nbl::system::SCameraProjectionContext* getProjectionContext() const
	{
		return hasProjectionContext ? &projectionContext : nullptr;
	}
};

struct SActiveCameraInputTarget final
{
	ICamera* camera = nullptr;
	uint32_t planarIx = SWindowControlBinding::InvalidPlanarIx;

	inline bool valid() const
	{
		return camera && planarIx != SWindowControlBinding::InvalidPlanarIx;
	}
};

constexpr IGPUImage::SSubresourceRange TripleBufferUsedSubresourceRange =
{
	.aspectMask = IGPUImage::EAF_COLOR_BIT,
	.baseMipLevel = 0,
	.levelCount = 1,
	.baseArrayLayer = 0,
	.layerCount = 1
};

#endif // _NBL_THIS_EXAMPLE_APP_TYPES_HPP_
