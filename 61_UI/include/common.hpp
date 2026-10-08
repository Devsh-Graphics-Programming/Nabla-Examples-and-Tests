#ifndef _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_

#include <bitset>

#include "nbl/examples/examples.hpp"

// common api
#include "nbl/ext/Cameras/CFPSCamera.hpp"
#include "nbl/ext/Cameras/CFreeCamera.hpp"
#include "nbl/ext/Cameras/CSphericalTargetCamera.hpp"
#include "nbl/ext/Cameras/COrbitCamera.hpp"
#include "nbl/ext/Cameras/CArcballCamera.hpp"
#include "nbl/ext/Cameras/CTurntableCamera.hpp"
#include "nbl/ext/Cameras/CTopDownCamera.hpp"
#include "nbl/ext/Cameras/CIsometricCamera.hpp"
#include "nbl/ext/Cameras/CChaseCamera.hpp"
#include "nbl/ext/Cameras/CDollyCamera.hpp"
#include "nbl/ext/Cameras/CDollyZoomCamera.hpp"
#include "nbl/ext/Cameras/CPathCamera.hpp"
#include "camera/CCameraPreset.hpp"
#include "camera/CCameraPresetFlow.hpp"
#include "camera/CCameraKeyframeTrack.hpp"
#include "camera/CCameraPlaybackTimeline.hpp"
#include "camera/CCameraSequenceScript.hpp"
#include "camera/CCameraScriptedRuntime.hpp"
#include "camera/CCameraScriptedUiInputUtilities.hpp"
#include "camera/CCameraScriptedCheckRunner.hpp"
#include "camera/CCameraGoalAnalysis.hpp"
#include "camera/CCameraGoalSolver.hpp"
#include "camera/CCameraPresentationUtilities.hpp"
#include "camera/CCameraProjectionUtilities.hpp"
#include "nbl/ext/Cameras/CCameraKindUtilities.hpp"
#include "camera/CCameraFollowUtilities.hpp"
#include "camera/CCameraFollowRegressionUtilities.hpp"
#include "camera/CCameraConstraintUtilities.hpp"
#include "camera/CCameraScriptedActionUtilities.hpp"
#include "camera/CCameraScriptedRuntimePersistence.hpp"
#include "camera/CCameraSequenceScriptedBuilder.hpp"
#include "camera/CCameraControlPanelUiUtilities.hpp"
#include "camera/CCameraScriptVisualDebugOverlayUtilities.hpp"
#include "camera/CCameraViewportOverlayUtilities.hpp"
#include "camera/CCameraTextUtilities.hpp"
#include "nbl/ext/Cameras/SCameraControls.hpp"
#include "camera/CInputCodeNames.hpp"
#include "nbl/ext/Cameras/CCameraMouseKeyboardController.hpp"
#include "nbl/ext/Cameras/CCameraMouseKeyboardPresets.hpp"

#include "nbl/ext/Cameras/CCameraWithProjections.hpp"
// the example's headers
#include "nbl/ui/ICursorControl.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "imgui/imgui_internal.h"
#include "imguizmo/ImGuizmo.h"

namespace nbl::this_example
{

template<typename Tout, typename Tin, uint32_t N, uint32_t M>
inline hlsl::matrix<Tout, N, M> getCastedMatrix(const hlsl::matrix<Tin, N, M>& input)
{
	return hlsl::_static_cast<hlsl::matrix<Tout, N, M> >(input);
}

}

namespace core = nbl::core;
namespace asset = nbl::asset;
namespace ext = nbl::ext;
namespace ui = nbl::ui;
namespace video = nbl::video;
namespace examples = nbl::examples;
namespace hlsl = nbl::hlsl;
using nbl::core::bitflag;
using nbl::core::make_smart_refctd_ptr;
using nbl::core::smart_refctd_ptr;
using nbl::core::smart_refctd_ptr_static_cast;
using nbl::core::vector;
using nbl::system::path;
using nbl::system::ILogger;
using nbl::system::ISystem;
using nbl::system::IApplicationFramework;
using nbl::system::logger_opt_smart_ptr;
using nbl::asset::E_FORMAT;
using nbl::asset::EF_D32_SFLOAT;
using nbl::asset::EF_R16G16B16A16_SFLOAT;
using nbl::asset::EF_R8G8B8A8_SRGB;
using nbl::asset::EPBP_GRAPHICS;
using nbl::asset::ACCESS_FLAGS;
using nbl::asset::IAsset;
using nbl::asset::IAssetManager;
using nbl::asset::IAssetLoader;
using nbl::asset::IDescriptor;
using nbl::asset::IImage;
using nbl::asset::ISampler;
using nbl::asset::IShader;
using nbl::asset::PIPELINE_STAGE_FLAGS;
using nbl::asset::SBufferRange;
using nbl::asset::isDepthOrStencilFormat;
using nbl::ui::ICursorControl;
using nbl::ui::IKeyboardEventChannel;
using nbl::ui::IMouseEventChannel;
using nbl::ui::EKC_NONE;
using nbl::ui::E_KEY_CODE;
using nbl::ui::E_MOUSE_BUTTON;
using nbl::ui::EMB_COUNT;
using nbl::ui::SKeyboardEvent;
using nbl::ui::SMouseEvent;
using nbl::ui::IWindow;
using nbl::ui::IWindowWin32;
using nbl::video::CSurfaceVulkanWin32;
using nbl::video::CSmoothResizeSurface;
using nbl::video::IDescriptorPool;
using nbl::video::IDeviceMemoryAllocation;
using nbl::video::ILogicalDevice;
using nbl::video::IQueue;
using nbl::video::ISemaphore;
using nbl::video::ISwapchain;
using nbl::video::ISmoothResizeSurface;
using nbl::video::IGPUBuffer;
using nbl::video::IGPUCommandBuffer;
using nbl::video::IGPUCommandPool;
using nbl::video::IGPUDescriptorSet;
using nbl::video::IGPUDescriptorSetLayout;
using nbl::video::IGPUFramebuffer;
using nbl::video::IGPUGraphicsPipeline;
using nbl::video::IGPUImage;
using nbl::video::IGPUImageView;
using nbl::video::IGPUPipelineBase;
using nbl::video::IGPURenderpass;
using nbl::video::IGPUSampler;
using nbl::video::SIntendedSubmitInfo;
using nbl::examples::CGeometryCreatorScene;
using nbl::examples::InputSystem;
using nbl::examples::CSimpleDebugRenderer;
using nbl::ext::cameras::CCameraMathUtilities;
using nbl::ext::cameras::SCameraBasis;
using nbl::ext::cameras::ICamera;
using nbl::ext::cameras::CFPSCamera;
using nbl::ext::cameras::CFreeCamera;
using nbl::ext::cameras::CSphericalTargetCamera;
using nbl::ext::cameras::COrbitCamera;
using nbl::ext::cameras::CArcballCamera;
using nbl::ext::cameras::CTurntableCamera;
using nbl::ext::cameras::CTopDownCamera;
using nbl::ext::cameras::CIsometricCamera;
using nbl::ext::cameras::CChaseCamera;
using nbl::ext::cameras::CDollyCamera;
using nbl::ext::cameras::CDollyZoomCamera;
using nbl::ext::cameras::CPathCamera;
using nbl::this_example::ECameraScriptedActionCode;
using nbl::this_example::CCameraConstraintUtilities;
using nbl::this_example::CCameraScriptedActionUtilities;
using nbl::this_example::CCameraScriptedActionEvent;
using nbl::this_example::CCameraScriptedInputParseResult;
using nbl::this_example::CCameraScriptedRuntimePersistenceUtilities;
using nbl::this_example::CCameraSequenceScriptedSegmentBuildInfo;
using nbl::this_example::CCameraSequenceScriptedBuilderUtilities;
using nbl::this_example::SCameraConstraintSettings;
using nbl::ext::cameras::CPlanarProjection;
using nbl::ext::cameras::CCameraWithProjections;
using nbl::ext::cameras::SCameraControls;
using nbl::ext::cameras::ECameraControlAxis;
using nbl::ext::cameras::CameraControlAxisCount;
using nbl::ext::cameras::cameraControlAxisName;
using nbl::ext::cameras::cameraControlAxisFromIndex;
using nbl::ext::cameras::cameraControlAxisIndex;
using nbl::ext::cameras::stringToCameraControlAxis;
using nbl::ext::cameras::SMouseKeyboardAxisBinding;
using nbl::ext::cameras::SCameraMouseKeyboardBinding;
using nbl::ext::cameras::CCameraMouseKeyboardController;
using nbl::ext::cameras::CCameraMouseKeyboardPresets;
using nbl::hlsl::float32_t;
using nbl::hlsl::float32_t2;
using nbl::hlsl::float32_t3;
using nbl::hlsl::float32_t4;
using nbl::hlsl::float32_t3x3;
using nbl::hlsl::float32_t3x4;
using nbl::hlsl::float32_t4x4;
using nbl::hlsl::float64_t;
using nbl::hlsl::float64_t3;
using nbl::hlsl::float64_t4;
using nbl::hlsl::float64_t4x4;
using nbl::hlsl::uint16_t2;
using nbl::ext::cameras::CCameraMathUtilities;
using nbl::ui::CCameraControlPanelUiUtilities;
using nbl::ui::CCameraScriptVisualDebugOverlayUtilities;
using nbl::ui::CCameraViewportOverlayUtilities;
using nbl::this_example::getCastedMatrix;
using nbl::hlsl::mul;
using nbl::hlsl::math::quaternion;

#endif // _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_
