// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include "nbl/video/utilities/CSimpleResizeSurface.h"

#include "../common/SimpleWindowedApplication.hpp"
#include "../common/InputSystem.hpp"

#include "nbl/ext/ImGui/ImGui.h"
#include "nbl/ui/ICursorControl.h"

#include "imgui/imgui_internal.h"
#include "imguizmo/ImGuizmo.h"

#include "nbl/api/CCamera.hpp"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

// https://github.com/Devsh-Graphics-Programming/ImGuizmo/blob/master/example/main.cpp
// https://github.com/Devsh-Graphics-Programming/ImGuizmo/blob/master/LICENSE

bool useWindow = true;
float camDistance = 8.f;
static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::TRANSLATE);

float objectMatrix[4][16] = {
  { 1.f, 0.f, 0.f, 0.f,
	0.f, 1.f, 0.f, 0.f,
	0.f, 0.f, 1.f, 0.f,
	0.f, 0.f, 0.f, 1.f },

  { 1.f, 0.f, 0.f, 0.f,
  0.f, 1.f, 0.f, 0.f,
  0.f, 0.f, 1.f, 0.f,
  2.f, 0.f, 0.f, 1.f },

  { 1.f, 0.f, 0.f, 0.f,
  0.f, 1.f, 0.f, 0.f,
  0.f, 0.f, 1.f, 0.f,
  2.f, 0.f, 2.f, 1.f },

  { 1.f, 0.f, 0.f, 0.f,
  0.f, 1.f, 0.f, 0.f,
  0.f, 0.f, 1.f, 0.f,
  0.f, 0.f, 2.f, 1.f }
};

static const float identityMatrix[16] =
{ 1.f, 0.f, 0.f, 0.f,
	0.f, 1.f, 0.f, 0.f,
	0.f, 0.f, 1.f, 0.f,
	0.f, 0.f, 0.f, 1.f };

void EditTransform(float* cameraView, const float* cameraProjection, float* matrix, bool editTransformDecomposition)
{
	static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::LOCAL);
	static bool useSnap = false;
	static float snap[3] = { 1.f, 1.f, 1.f };
	static float bounds[] = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
	static float boundsSnap[] = { 0.1f, 0.1f, 0.1f };
	static bool boundSizing = false;
	static bool boundSizingSnap = false;

	if (editTransformDecomposition)
	{
		if (ImGui::IsKeyPressed(ImGuiKey_T))
			mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
		if (ImGui::IsKeyPressed(ImGuiKey_R))
			mCurrentGizmoOperation = ImGuizmo::ROTATE;
		if (ImGui::IsKeyPressed(ImGuiKey_S))
			mCurrentGizmoOperation = ImGuizmo::SCALE;
		if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
			mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
		ImGui::SameLine();
		if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
			mCurrentGizmoOperation = ImGuizmo::ROTATE;
		ImGui::SameLine();
		if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
			mCurrentGizmoOperation = ImGuizmo::SCALE;
		if (ImGui::RadioButton("Universal", mCurrentGizmoOperation == ImGuizmo::UNIVERSAL))
			mCurrentGizmoOperation = ImGuizmo::UNIVERSAL;
		float matrixTranslation[3], matrixRotation[3], matrixScale[3];
		ImGuizmo::DecomposeMatrixToComponents(matrix, matrixTranslation, matrixRotation, matrixScale);
		ImGui::InputFloat3("Tr", matrixTranslation);
		ImGui::InputFloat3("Rt", matrixRotation);
		ImGui::InputFloat3("Sc", matrixScale);
		ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, matrix);

		if (mCurrentGizmoOperation != ImGuizmo::SCALE)
		{
			if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
				mCurrentGizmoMode = ImGuizmo::LOCAL;
			ImGui::SameLine();
			if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
				mCurrentGizmoMode = ImGuizmo::WORLD;
		}
		if (ImGui::IsKeyPressed(ImGuiKey_S) && ImGui::IsKeyPressed(ImGuiKey_LeftShift))
			useSnap = !useSnap;
		ImGui::Checkbox("##UseSnap", &useSnap);
		ImGui::SameLine();

		switch (mCurrentGizmoOperation)
		{
		case ImGuizmo::TRANSLATE:
			ImGui::InputFloat3("Snap", &snap[0]);
			break;
		case ImGuizmo::ROTATE:
			ImGui::InputFloat("Angle Snap", &snap[0]);
			break;
		case ImGuizmo::SCALE:
			ImGui::InputFloat("Scale Snap", &snap[0]);
			break;
		}
		ImGui::Checkbox("Bound Sizing", &boundSizing);
		if (boundSizing)
		{
			ImGui::PushID(3);
			ImGui::Checkbox("##BoundSizing", &boundSizingSnap);
			ImGui::SameLine();
			ImGui::InputFloat3("Snap", boundsSnap);
			ImGui::PopID();
		}
	}

	ImGuiIO& io = ImGui::GetIO();
	float viewManipulateRight = io.DisplaySize.x;
	float viewManipulateTop = 0;
	static ImGuiWindowFlags gizmoWindowFlags = 0;
	if (useWindow)
	{
		ImGui::SetNextWindowSize(ImVec2(800, 400), ImGuiCond_Appearing);
		ImGui::SetNextWindowPos(ImVec2(400, 20), ImGuiCond_Appearing);
		ImGui::PushStyleColor(ImGuiCol_WindowBg, (ImVec4)ImColor(0.35f, 0.3f, 0.3f));
		ImGui::Begin("Gizmo", 0, gizmoWindowFlags);
		ImGuizmo::SetDrawlist();
		float windowWidth = (float)ImGui::GetWindowWidth();
		float windowHeight = (float)ImGui::GetWindowHeight();
		ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, windowWidth, windowHeight);
		viewManipulateRight = ImGui::GetWindowPos().x + windowWidth;
		viewManipulateTop = ImGui::GetWindowPos().y;
		ImGuiWindow* window = ImGui::GetCurrentWindow();
		gizmoWindowFlags = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(window->InnerRect.Min, window->InnerRect.Max) ? ImGuiWindowFlags_NoMove : 0;
	}
	else
	{
		ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
	}

	ImGuizmo::DrawGrid(cameraView, cameraProjection, identityMatrix, 100.f);
	ImGuizmo::DrawCubes(cameraView, cameraProjection, &objectMatrix[0][0], 1);
	ImGuizmo::Manipulate(cameraView, cameraProjection, mCurrentGizmoOperation, mCurrentGizmoMode, matrix, NULL, useSnap ? &snap[0] : NULL, boundSizing ? bounds : NULL, boundSizingSnap ? boundsSnap : NULL);

	ImGuizmo::ViewManipulate(cameraView, camDistance, ImVec2(viewManipulateRight - 128, viewManipulateTop), ImVec2(128, 128), 0x10101010);

	if (useWindow)
	{
		ImGui::End();
		ImGui::PopStyleColor(1);
	}
}

class CEventCallback : public ISimpleManagedSurface::ICallback
{
public:
	CEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(m_inputSystem)), m_logger(std::move(logger)) {}
	CEventCallback() {}

	void setLogger(nbl::system::logger_opt_smart_ptr& logger)
	{
		m_logger = logger;
	}
	void setInputSystem(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem)
	{
		m_inputSystem = std::move(m_inputSystem);
	}
private:

	void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
	{
		m_logger.log("A mouse %p has been connected", nbl::system::ILogger::ELL_INFO, mch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_mouse, std::move(mch));
	}
	void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
	{
		m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse, mch);
	}
	void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
	{
		m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard, std::move(kbch));
	}
	void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
	{
		m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard, kbch);
	}

private:
	nbl::core::smart_refctd_ptr<InputSystem> m_inputSystem = nullptr;
	nbl::system::logger_opt_smart_ptr m_logger = nullptr;
};

class UISampleApp final : public examples::SimpleWindowedApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using clock_t = std::chrono::steady_clock;

	_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_W = 1280, WIN_H = 720, SC_IMG_COUNT = 3u, FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

	public:
		inline UISampleApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) 
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			if (!m_surface)
			{
				{
					auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
					IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
					params.width = WIN_W;
					params.height = WIN_H;
					params.x = 32;
					params.y = 32;
					params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
					params.windowCaption = "UISampleApp";
					params.callback = windowCallback;
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}

				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<nbl::video::CDefaultSwapchainFramebuffers>::create(std::move(surface));
			}

			if (m_surface)
				return { {m_surface->getSurface()/*,EQF_NONE*/} };

			return {};
		}

		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;

			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			ISwapchain::SCreationParams swapchainParams = { .surface = m_surface->getSurface() };
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");

			const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = 
			{
				{
					.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.dstSubpass = 0,
					.memoryBarrier = 
					{
						.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
						.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
						.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
				},
				{
					.srcSubpass = 0,
					.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.memoryBarrier = 
					{
						.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
				},
				IGPURenderpass::SCreationParams::DependenciesEnd
			};

			auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(), swapchainParams.surfaceFormat.format, dependencies);
			auto* renderpass = scResources->getRenderpass();
			
			if (!renderpass)
				return logFail("Failed to create Renderpass!");

			auto gQueue = getGraphicsQueue();
			if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");

			m_maxFramesInFlight = m_surface->getMaxFramesInFlight();
			if (FRAMES_IN_FLIGHT < m_maxFramesInFlight)
			{
				m_logger->log("Lowering frames in flight!", ILogger::ELL_WARNING);
				m_maxFramesInFlight = FRAMES_IN_FLIGHT;
			}

			m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
	
			for (auto i = 0u; i < m_maxFramesInFlight; i++)
			{
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");
				if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
					return logFail("Couldn't create Command Buffer!");
			}

			ui = core::make_smart_refctd_ptr<nbl::ext::imgui::UI>(smart_refctd_ptr(m_device), (int)m_maxFramesInFlight, renderpass, nullptr, smart_refctd_ptr(m_window));
			ui->registerListener([this]() -> void 
				{
					ImGuiIO& io = ImGui::GetIO();

					if (isPerspective)
						camera.setProjectionMatrix(matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y, zNear, zFar));
					else
					{
						float viewHeight = viewWidth * io.DisplaySize.y / io.DisplaySize.x;
						camera.setProjectionMatrix(matrix4SIMD::buildProjectionMatrixOrthoLH(viewWidth, viewHeight, zNear, zFar));
					}

					ImGuizmo::SetOrthographic(false);
					ImGuizmo::BeginFrame();

					ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
					ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

					// create a window and insert the inspector
					ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
					ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
					ImGui::Begin("Editor");

					if (ImGui::RadioButton("Full view", !useWindow))
						useWindow = false;

					ImGui::SameLine();

					if (ImGui::RadioButton("Window", useWindow))
						useWindow = true;

					ImGui::Text("Camera");
					bool viewDirty = false;

					if (ImGui::RadioButton("Perspective", isPerspective))
						isPerspective = true;

					ImGui::SameLine();

					if (ImGui::RadioButton("Orthographic", !isPerspective))
						isPerspective = false;

					ImGui::Checkbox("Enable movement", &move);
					ImGui::SliderFloat("Move speed", &moveSpeed, 0.1f, 10.f);
					ImGui::SliderFloat("Rotate speed", &rotateSpeed, 0.1f, 10.f);

					ImGui::Checkbox("Flip Y", &flipY);

					if (isPerspective)
						ImGui::SliderFloat("Fov", &fov, 20.f, 150.f);
					else
						ImGui::SliderFloat("Ortho width", &viewWidth, 1, 20);

					ImGui::SliderFloat("zNear", &zNear, 0.1f, 100.f);
					ImGui::SliderFloat("zFar", &zFar, 110.f, 10000.f);

					viewDirty |= ImGui::SliderFloat("Distance", &camDistance, 1.f, 69.f);

					if (viewDirty || firstFrame)
					{
						core::vectorSIMDf cameraPosition(cosf(camYAngle)* cosf(camXAngle)* camDistance, sinf(camXAngle)* camDistance, sinf(camYAngle)* cosf(camXAngle)* camDistance);
						core::vectorSIMDf cameraTarget(0.f, 0.f, 0.f);
						const static core::vectorSIMDf up(0.f, 1.f, 0.f);

						camera.setPosition(cameraPosition);
						camera.setTarget(cameraTarget);
						camera.setBackupUpVector(up);

						camera.recomputeViewMatrix();

						firstFrame = false;
					}

					ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);
					if (ImGuizmo::IsUsing())
					{
						ImGui::Text("Using gizmo");
					}
					else
					{
						ImGui::Text(ImGuizmo::IsOver() ? "Over gizmo" : "");
						ImGui::SameLine();
						ImGui::Text(ImGuizmo::IsOver(ImGuizmo::TRANSLATE) ? "Over translate gizmo" : "");
						ImGui::SameLine();
						ImGui::Text(ImGuizmo::IsOver(ImGuizmo::ROTATE) ? "Over rotate gizmo" : "");
						ImGui::SameLine();
						ImGui::Text(ImGuizmo::IsOver(ImGuizmo::SCALE) ? "Over scale gizmo" : "");
					}
					ImGui::Separator();
					ImGuizmo::SetID(0);

					/*
					* ImGuizmo expects view & perspective matrix to be column major both with 4x4 layout
					* and Nabla uses row major matricies - 3x4 matrix for view & 4x4 for projection

					- VIEW:

						ImGuizmo

						|     X[0]          Y[0]          Z[0]         0.0f |
						|     X[1]          Y[1]          Z[1]         0.0f |
						|     X[2]          Y[2]          Z[2]         0.0f |
						| -Dot(X, eye)  -Dot(Y, eye)  -Dot(Z, eye)     1.0f |

						Nabla

						|     X[0]         X[1]           X[2]     -Dot(X, eye)  |
						|     Y[0]         Y[1]           Y[2]     -Dot(Y, eye)  |
						|     Z[0]         Z[1]           Z[2]     -Dot(Z, eye)  |

						<ImGuizmo View Matrix> = transpose(nbl::core::matrix4SIMD(<Nabla View Matrix>))

					- PERSPECTIVE [PROJECTION CASE]:

						ImGuizmo

						|      (temp / temp2)                 (0.0)                       (0.0)                   (0.0)  |
						|          (0.0)                  (temp / temp3)                  (0.0)                   (0.0)  |
						| ((right + left) / temp2)   ((top + bottom) / temp3)    ((-zfar - znear) / temp4)       (-1.0f) |
						|          (0.0)                      (0.0)               ((-temp * zfar) / temp4)        (0.0)  |

						Nabla

						|            w                        (0.0)                       (0.0)                   (0.0)               |
						|          (0.0)                       -h                         (0.0)                   (0.0)               |
						|          (0.0)                      (0.0)               (-zFar/(zFar-zNear))     (-zNear*zFar/(zFar-zNear)) |
						|          (0.0)                      (0.0)                      (-1.0)                   (0.0)               |

						<ImGuizmo Projection Matrix> = transpose(<Nabla Projection Matrix>)

					*
					* the ViewManipulate final call (inside EditTransform) returns world space column major matrix for an object,
					* note it also modifies input view matrix but projection matrix is immutable
					*/

					static struct
					{
						core::matrix4SIMD view, projection;
						float* world = nullptr;
					} imguizmoM16InOut;

					imguizmoM16InOut.view = core::transpose(matrix4SIMD(camera.getViewMatrix()));
					imguizmoM16InOut.projection = core::transpose(camera.getProjectionMatrix());
					imguizmoM16InOut.world = objectMatrix[0];
					{
						if(flipY)
							imguizmoM16InOut.projection[1][1] *= -1.f; // https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/

						EditTransform(imguizmoM16InOut.view.pointer(), imguizmoM16InOut.projection.pointer(), imguizmoM16InOut.world, true);
					}
					auto newView = core::transpose(imguizmoM16InOut.view).extractSub3x4(); // to Nabla view + update camera
					const_cast<core::matrix3x4SIMD&>(camera.getViewMatrix()) = newView; // a hack, correct way would be to use inverse matrix and get position + target because now it will bring you back to last position & target when switching from gizmo move to manual move (but from manual to gizmo is ok)

					ImGui::End();
				}
			);

			m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
			m_surface->recreateSwapchain();
			m_winMgr->show(m_window.get());
			oracle.reportBeginFrameRecord();
			camera.mapKeysToArrows();

			return true;
		}

		inline void workLoopBody() override
		{
			const auto resourceIx = m_realFrameIx % m_maxFramesInFlight;

			if (m_realFrameIx >= m_maxFramesInFlight)
			{
				const ISemaphore::SWaitInfo cbDonePending[] = 
				{
					{
						.semaphore = m_semaphore.get(),
						.value = m_realFrameIx + 1 - m_maxFramesInFlight
					}
				};
				if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

			m_inputSystem->getDefaultMouse(&mouse);
			m_inputSystem->getDefaultKeyboard(&keyboard);

			auto updatePresentationTimestamp = [&]()
			{
				m_currentImageAcquire = m_surface->acquireNextImage();

				oracle.reportEndFrameRecord();
				const auto timestamp = oracle.getNextPresentationTimeStamp();
				oracle.reportBeginFrameRecord();

				return timestamp;
			};

			const auto nextPresentationTimestamp = updatePresentationTimestamp();

			auto* const cb = m_cmdBufs.data()[resourceIx].get();
			cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cb->beginDebugMarker("UISampleApp Frame");

			auto* queue = getGraphicsQueue();

			asset::SViewport viewport;
			{
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = WIN_W;
				viewport.height = WIN_H;
			}
			cb->setViewport(0u, 1u, &viewport);
			{
				const VkRect2D currentRenderArea =
				{
					.offset = {0,0},
					.extent = {m_window->getWidth(),m_window->getHeight()}
				};

				const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				const IGPUCommandBuffer::SRenderpassBeginInfo info = 
				{
					.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = nullptr,
					.renderArea = currentRenderArea
				};
				cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
			}

			// TODO: Use real deltaTime instead
			float deltaTimeInSec = 0.1f;
			ui->render(cb, resourceIx);
			cb->endRenderPass();
			cb->end();
			{
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] = 
				{ 
					{
						.semaphore = m_semaphore.get(),
						.value = ++m_realFrameIx,
						.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
					} 
				};
				{
					{
						const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] = 
						{ 
							{ .cmdbuf = cb } 
						};

						const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = 
						{ 
							{
								.semaphore = m_currentImageAcquire.semaphore,
								.value = m_currentImageAcquire.acquireCount,
								.stageMask = PIPELINE_STAGE_FLAGS::NONE
							} 
						};
						const IQueue::SSubmitInfo infos[] = 
						{ 
							{
								.waitSemaphores = acquired,
								.commandBuffers = commandBuffers,
								.signalSemaphores = rendered
							} 
						};

						if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
							m_realFrameIx--;
					}
				}

				m_window->setCaption("[Nabla Engine] UI App Test Demo");
				m_surface->present(m_currentImageAcquire.imageIndex, rendered);
			}

			static std::chrono::microseconds previousEventTimestamp{};

			struct
			{
				std::vector<SMouseEvent> mouse{};
				std::vector<SKeyboardEvent> keyboard{};
			} capturedEvents;

			if (move) camera.beginInputProcessing(nextPresentationTimestamp);
			{
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
				{
					if(move)
						camera.mouseProcess(events); // don't capture the events, only let camera handle them with its impl

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						capturedEvents.mouse.emplace_back(e);
					}
				}, m_logger.get());

				keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
				{
					if (move)
						camera.keyboardProcess(events); // don't capture the events, only let camera handle them with its impl

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						capturedEvents.keyboard.emplace_back(e);
					}
				}, m_logger.get());
			}
			if (move) camera.endInputProcessing(nextPresentationTimestamp);

			const auto mousePosition = m_window->getCursorControl()->getPosition();
			core::SRange<const nbl::ui::SMouseEvent> mouseEvents(capturedEvents.mouse.data(), capturedEvents.mouse.data() + capturedEvents.mouse.size());
			core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents(capturedEvents.keyboard.data(), capturedEvents.keyboard.data() + capturedEvents.keyboard.size());

			ui->update(deltaTimeInSec, { mousePosition.x , mousePosition.y }, mouseEvents, keyboardEvents);
			camera.setMoveSpeed(moveSpeed);
			camera.setRotateSpeed(rotateSpeed);
		}

		inline bool keepRunning() override
		{
			if (m_surface->irrecoverable())
				return false;

			return true;
		}

		inline bool onAppTerminated() override
		{
			return device_base_t::onAppTerminated();
		}

	private:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;
		smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;
		smart_refctd_ptr<ISemaphore> m_semaphore;
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		uint64_t m_realFrameIx : 59 = 0;
		uint64_t m_maxFramesInFlight : 5;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

		nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> ui;
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
		video::CDumbPresentationOracle oracle;

		int lastUsing = 0;

		bool isPerspective = true, flipY = true, move = false;
		float fov = 60.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
		float viewWidth = 10.f; // for orthographic
		float camYAngle = 165.f / 180.f * 3.14159f;
		float camXAngle = 32.f / 180.f * 3.14159f;

		bool firstFrame = true;
};

NBL_MAIN_FUNC(UISampleApp)