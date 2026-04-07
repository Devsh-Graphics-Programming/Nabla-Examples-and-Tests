#include "app/App.hpp"

void App::update()
{
			m_inputSystem->getDefaultMouse(&mouse);
			m_inputSystem->getDefaultKeyboard(&keyboard);

			auto updatePresentationTimestamp = [&]()
			{
				oracle.reportEndFrameRecord();
				const auto timestamp = oracle.getNextPresentationTimeStamp();
				oracle.reportBeginFrameRecord();

				return timestamp;
			};

			m_nextPresentationTimestamp = updatePresentationTimestamp();
			if (m_haveLastPresentationTimestamp)
			{
				const auto delta = m_nextPresentationTimestamp - m_lastPresentationTimestamp;
				if (delta.count() < 0)
					m_frameDeltaSec = 0.0;
				else
					m_frameDeltaSec = static_cast<double>(delta.count()) / 1000000.0;
			}
			m_lastPresentationTimestamp = m_nextPresentationTimestamp;
			m_haveLastPresentationTimestamp = true;

			updatePlayback(m_frameDeltaSec);
			const bool skipCameraInput = m_playback.playing && m_playback.overrideInput;

			struct
			{
				std::vector<SMouseEvent> mouse {};
				std::vector<SKeyboardEvent> keyboard {};
			} capturedEvents;
			{
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
				{
					if (m_window->hasInputFocus())
						for (const auto& e : events)
							capturedEvents.mouse.emplace_back(e);
				}, m_logger.get());

				keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
				{
					if (m_window->hasInputFocus())
						for (const auto& e : events)
							capturedEvents.keyboard.emplace_back(e);
				}, m_logger.get());
			}

			if (m_scriptedInput.enabled && m_scriptedInput.exclusive)
			{
				capturedEvents.mouse.clear();
				capturedEvents.keyboard.clear();
			}

			std::vector<SMouseEvent> scriptedMouse;
			std::vector<SKeyboardEvent> scriptedKeyboard;
			CCameraScriptedFrameEvents scriptedFrameEvents;
			const CVirtualGimbalEvent* scriptedImguizmoVirtual = nullptr;
			uint32_t scriptedImguizmoVirtualCount = 0u;

			if (m_scriptedInput.enabled && m_scriptedInput.nextEventIndex < m_scriptedInput.timeline.events.size())
				nbl::hlsl::dequeueScriptedFrameEvents(
					m_scriptedInput.timeline.events,
					m_scriptedInput.nextEventIndex,
					m_realFrameIx,
					scriptedFrameEvents);

			for (const auto& authoredKeyboard : scriptedFrameEvents.keyboard)
			{
				SKeyboardEvent e(m_nextPresentationTimestamp);
				e.keyCode = authoredKeyboard.key;
				e.action = authoredKeyboard.action;
				e.window = m_window.get();
				scriptedKeyboard.emplace_back(e);
			}
			for (const auto& authoredMouse : scriptedFrameEvents.mouse)
			{
				SMouseEvent e(m_nextPresentationTimestamp);
				e.window = m_window.get();
				e.type = authoredMouse.type;
				if (authoredMouse.type == ui::SMouseEvent::EET_CLICK)
				{
					e.clickEvent.mouseButton = authoredMouse.button;
					e.clickEvent.action = authoredMouse.action;
					e.clickEvent.clickPosX = authoredMouse.x;
					e.clickEvent.clickPosY = authoredMouse.y;
				}
				else if (authoredMouse.type == ui::SMouseEvent::EET_SCROLL)
				{
					e.scrollEvent.verticalScroll = authoredMouse.v;
					e.scrollEvent.horizontalScroll = authoredMouse.h;
				}
				else if (authoredMouse.type == ui::SMouseEvent::EET_MOVEMENT)
				{
					e.movementEvent.relativeMovementX = authoredMouse.dx;
					e.movementEvent.relativeMovementY = authoredMouse.dy;
				}
				scriptedMouse.emplace_back(e);
			}

			if (!scriptedFrameEvents.segmentLabels.empty())
				m_scriptedInput.visualSegmentLabel = scriptedFrameEvents.segmentLabels.back();

			if (m_scriptedInput.enabled && scriptedFrameEvents.actions.size())
			{
				auto applyAction = [&](const CCameraScriptedInputEvent::ActionData& action) -> void
				{
					switch (action.kind)
					{
						case CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow:
						{
							if (action.value < 0 || static_cast<size_t>(action.value) >= windowBindings.size())
							{
								m_logger->log("[script][warn] action set_active_render_window out of range: %d", ILogger::ELL_WARNING, action.value);
								return;
							}
							activeRenderWindowIx = static_cast<uint32_t>(action.value);
						} break;

						case CCameraScriptedInputEvent::ActionData::Kind::SetActivePlanar:
						{
							if (action.value < 0 || static_cast<size_t>(action.value) >= m_planarProjections.size())
							{
								m_logger->log("[script][warn] action set_active_planar out of range: %d", ILogger::ELL_WARNING, action.value);
								return;
							}
							auto& binding = windowBindings[activeRenderWindowIx];
							binding.activePlanarIx = static_cast<uint32_t>(action.value);
							binding.pickDefaultProjections(m_planarProjections[binding.activePlanarIx]->getPlanarProjections());
							m_scriptedInput.visualActivePlanarValid = true;
							m_scriptedInput.visualActivePlanarIx = binding.activePlanarIx;
							m_scriptedInput.visualActivePlanarStartFrame = m_realFrameIx;
						} break;

						case CCameraScriptedInputEvent::ActionData::Kind::SetProjectionType:
						{
							auto& binding = windowBindings[activeRenderWindowIx];
							if (!binding.lastBoundPerspectivePresetProjectionIx.has_value() || !binding.lastBoundOrthoPresetProjectionIx.has_value())
								binding.pickDefaultProjections(m_planarProjections[binding.activePlanarIx]->getPlanarProjections());

							const auto type = static_cast<IPlanarProjection::CProjection::ProjectionType>(action.value);
							switch (type)
							{
								case IPlanarProjection::CProjection::Perspective:
									binding.boundProjectionIx = binding.lastBoundPerspectivePresetProjectionIx.value();
									break;
								case IPlanarProjection::CProjection::Orthographic:
									binding.boundProjectionIx = binding.lastBoundOrthoPresetProjectionIx.value();
									break;
								default:
									m_logger->log("[script][warn] action set_projection_type invalid value: %d", ILogger::ELL_WARNING, action.value);
									break;
							}
						} break;

						case CCameraScriptedInputEvent::ActionData::Kind::SetProjectionIndex:
						{
							auto& binding = windowBindings[activeRenderWindowIx];
							auto& projections = m_planarProjections[binding.activePlanarIx]->getPlanarProjections();
							if (action.value < 0 || static_cast<size_t>(action.value) >= projections.size())
							{
								m_logger->log("[script][warn] action set_projection_index out of range: %d", ILogger::ELL_WARNING, action.value);
								return;
							}
							const auto ix = static_cast<uint32_t>(action.value);
							const auto type = projections[ix].getParameters().m_type;
							binding.boundProjectionIx = ix;
							if (type == IPlanarProjection::CProjection::Perspective)
								binding.lastBoundPerspectivePresetProjectionIx = ix;
							else if (type == IPlanarProjection::CProjection::Orthographic)
								binding.lastBoundOrthoPresetProjectionIx = ix;
						} break;

						case CCameraScriptedInputEvent::ActionData::Kind::SetUseWindow:
						{
							useWindow = action.value != 0;
						} break;

						case CCameraScriptedInputEvent::ActionData::Kind::SetLeftHanded:
						{
							auto& binding = windowBindings[activeRenderWindowIx];
							binding.leftHandedProjection = action.value != 0;
						} break;

						case CCameraScriptedInputEvent::ActionData::Kind::ResetActiveCamera:
						{
							auto& binding = windowBindings[activeRenderWindowIx];
							if (binding.activePlanarIx >= m_planarProjections.size())
							{
								m_logger->log("[script][warn] action reset_active_camera active planar out of range: %u", ILogger::ELL_WARNING, binding.activePlanarIx);
								return;
							}
							if (binding.activePlanarIx >= m_initialPlanarPresets.size())
							{
								m_logger->log("[script][warn] action reset_active_camera missing initial preset for planar: %u", ILogger::ELL_WARNING, binding.activePlanarIx);
								return;
							}

							auto* camera = m_planarProjections[binding.activePlanarIx]->getCamera();
							if (!nbl::hlsl::applyPreset(m_cameraGoalSolver, camera, m_initialPlanarPresets[binding.activePlanarIx]))
								m_logger->log("[script][warn] action reset_active_camera failed for planar: %u", ILogger::ELL_WARNING, binding.activePlanarIx);
						} break;
					}
				};

				for (const auto& action : scriptedFrameEvents.actions)
					if (action.kind == CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow)
						applyAction(action);

				for (const auto& action : scriptedFrameEvents.actions)
					if (action.kind != CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow)
						applyAction(action);

				if (m_scriptedInput.log)
					m_logger->log("[script] frame %llu actions=%zu", ILogger::ELL_INFO, static_cast<unsigned long long>(m_realFrameIx), scriptedFrameEvents.actions.size());
			}

			if (m_scriptedInput.enabled && m_scriptedInput.visualDebug && !m_scriptedInput.visualActivePlanarValid)
			{
				if (activeRenderWindowIx < windowBindings.size())
				{
					m_scriptedInput.visualActivePlanarValid = true;
					m_scriptedInput.visualActivePlanarIx = windowBindings[activeRenderWindowIx].activePlanarIx;
					m_scriptedInput.visualActivePlanarStartFrame = m_realFrameIx;
				}
			}

			if (!m_scriptedInput.enabled)
			{
				m_scriptedInput.scriptedLeftMouseDown = false;
				m_scriptedInput.scriptedRightMouseDown = false;
			}
			else
			{
				for (const auto& ev : scriptedMouse)
				{
					if (ev.type != ui::SMouseEvent::EET_CLICK)
						continue;
					if (ev.clickEvent.mouseButton == ui::EMB_LEFT_BUTTON)
					{
						if (ev.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_PRESSED)
							m_scriptedInput.scriptedLeftMouseDown = true;
						else if (ev.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_RELEASED)
							m_scriptedInput.scriptedLeftMouseDown = false;
					}
					else if (ev.clickEvent.mouseButton == ui::EMB_RIGHT_BUTTON)
					{
						if (ev.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_PRESSED)
							m_scriptedInput.scriptedRightMouseDown = true;
						else if (ev.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_RELEASED)
							m_scriptedInput.scriptedRightMouseDown = false;
					}
				}
			}

			if (!scriptedMouse.empty())
				capturedEvents.mouse.insert(capturedEvents.mouse.end(), scriptedMouse.begin(), scriptedMouse.end());
			if (!scriptedKeyboard.empty())
				capturedEvents.keyboard.insert(capturedEvents.keyboard.end(), scriptedKeyboard.begin(), scriptedKeyboard.end());

			m_uiInputEventsThisFrame = static_cast<uint32_t>(capturedEvents.mouse.size() + capturedEvents.keyboard.size());

			auto cameraKeyboardEvents = capturedEvents.keyboard;
			auto cameraMouseEvents = capturedEvents.mouse;
			for (auto& ev : cameraMouseEvents)
			{
				if (ev.type == ui::SMouseEvent::EET_SCROLL)
				{
					ev.scrollEvent.verticalScroll *= m_cameraControls.mouseScrollScale;
					ev.scrollEvent.horizontalScroll *= m_cameraControls.mouseScrollScale;
				}
				else if (ev.type == ui::SMouseEvent::EET_MOVEMENT)
				{
					ev.movementEvent.relativeMovementX *= m_cameraControls.mouseMoveScale;
					ev.movementEvent.relativeMovementY *= m_cameraControls.mouseMoveScale;
				}
			}

			const auto cursorPosition = m_window->getCursorControl()->getPosition();

			nbl::ext::imgui::UI::SUpdateParameters params =
			{
				.mousePosition = nbl::hlsl::float32_t2(cursorPosition.x, cursorPosition.y) - nbl::hlsl::float32_t2(m_window->getX(), m_window->getY()),
				.displaySize = { m_window->getWidth(), m_window->getHeight() },
				.mouseEvents = { capturedEvents.mouse.data(), capturedEvents.mouse.size() },
				.keyboardEvents = { capturedEvents.keyboard.data(), capturedEvents.keyboard.size() }
			};

			if (m_scriptedInput.log && (scriptedKeyboard.size() || scriptedMouse.size() || scriptedFrameEvents.imguizmo.size() || scriptedFrameEvents.goals.size() || scriptedFrameEvents.trackedTargetTransforms.size()))
			{
				m_logger->log("[script] frame %llu input kb=%zu mouse=%zu imguizmo=%zu goals=%zu target=%zu", ILogger::ELL_INFO,
					static_cast<unsigned long long>(m_realFrameIx),
					scriptedKeyboard.size(),
					scriptedMouse.size(),
					scriptedFrameEvents.imguizmo.size(),
					scriptedFrameEvents.goals.size(),
					scriptedFrameEvents.trackedTargetTransforms.size());
			}

			if (enableActiveCameraMovement && !skipCameraInput)
			{
				auto& binding = windowBindings[activeRenderWindowIx];
				auto& planar = m_planarProjections[binding.activePlanarIx];
				auto* camera = planar->getCamera();

				assert(binding.boundProjectionIx.has_value());
				auto& projection = planar->getPlanarProjections()[binding.boundProjectionIx.value()];

				syncWindowInputBinding(binding);
				auto& inputBinder = binding.inputBinding;

				std::vector<SMouseEvent> filteredOrbitMouseEvents;
				std::span<const SMouseEvent> mouseInput = { cameraMouseEvents.data(), cameraMouseEvents.size() };
				if (isOrbitLikeCamera(camera))
				{
					const bool orbitLookDown = ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
						(m_scriptedInput.enabled && (m_scriptedInput.scriptedLeftMouseDown || m_scriptedInput.scriptedRightMouseDown));
					filteredOrbitMouseEvents.reserve(cameraMouseEvents.size());
					for (const auto& ev : cameraMouseEvents)
					{
						if (ev.type == ui::SMouseEvent::EET_MOVEMENT && !orbitLookDown)
							continue;
						filteredOrbitMouseEvents.emplace_back(ev);
					}
					mouseInput = { filteredOrbitMouseEvents.data(), filteredOrbitMouseEvents.size() };
				}

				auto collectedEvents = inputBinder.collectVirtualEvents(m_nextPresentationTimestamp, {
					.keyboardEvents = { cameraKeyboardEvents.data(), cameraKeyboardEvents.size() },
					.mouseEvents = mouseInput
				});
				auto& virtualEvents = collectedEvents.events;
				const uint32_t vCount = collectedEvents.totalCount();
				const uint32_t vKeyboardEventsCount = collectedEvents.keyboardCount;

				if (vCount)
				{
					for (uint32_t i = 0u; i < vKeyboardEventsCount; ++i)
						virtualEvents[i].magnitude *= m_cameraControls.keyboardScale;

					nbl::hlsl::scaleVirtualEvents(virtualEvents, vCount, m_cameraControls.translationScale, m_cameraControls.rotationScale);

					const char* bindingLabel = "Keyboard/Mouse";
					auto applyEventsToCamera = [&](ICamera* target, uint32_t planarIx)
					{
						if (!target)
							return;

						if (m_cameraControls.worldTranslate)
						{
							std::vector<CVirtualGimbalEvent> perCameraEvents(virtualEvents.begin(), virtualEvents.begin() + vCount);
							uint32_t perCount = vCount;
							nbl::hlsl::remapTranslationEventsFromWorldToCameraLocal(target, perCameraEvents, perCount);
							if (perCount)
								target->manipulate({ perCameraEvents.data(), perCount });
						}
						else
						{
							target->manipulate({ virtualEvents.data(), vCount });
						}

						nbl::hlsl::applyCameraConstraints(m_cameraGoalSolver, target, m_cameraConstraints);
						if (!m_scriptedInput.enabled)
							refreshFollowOffsetConfigForPlanar(planarIx);
						appendVirtualEventLog("input", bindingLabel, planarIx, target, virtualEvents.data(), vCount);
					};

					if (m_cameraControls.mirrorInput)
					{
						std::unordered_set<uintptr_t> visited;
						for (size_t bindingIx = 0u; bindingIx < windowBindings.size(); ++bindingIx)
						{
							auto& bindingIt = windowBindings[bindingIx];
							auto& planarIt = m_planarProjections[bindingIt.activePlanarIx];
							if (!planarIt)
								continue;
							auto* target = planarIt->getCamera();
							if (!target)
								continue;
							const auto id = target->getGimbal().getID();
							if (!visited.insert(id).second)
								continue;
							applyEventsToCamera(target, bindingIt.activePlanarIx);
						}
					}
					else
					{
						applyEventsToCamera(camera, binding.activePlanarIx);
					}

					if (m_scriptedInput.log)
					{
						for (uint32_t i = 0u; i < vCount; ++i)
						{
							const auto& ev = virtualEvents[i];
							m_logger->log("[script] virtual %s magnitude=%.6f", ILogger::ELL_INFO, CVirtualGimbalEvent::virtualEventToString(ev.type).data(), ev.magnitude);
						}

						const auto& gimbal = camera->getGimbal();
						const auto pos = gimbal.getPosition();
						const auto euler = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));
						m_logger->log("[script] gimbal pos=(%.3f, %.3f, %.3f) euler_deg=(%.3f, %.3f, %.3f)", ILogger::ELL_INFO,
							pos.x, pos.y, pos.z, euler.x, euler.y, euler.z);
					}
				}
			}

			if (m_scriptedInput.enabled && scriptedFrameEvents.imguizmo.size() && !skipCameraInput)
			{
				auto& binding = windowBindings[activeRenderWindowIx];
				auto& planar = m_planarProjections[binding.activePlanarIx];
				auto* camera = planar->getCamera();

				CGimbalInputBinder imguizmoBinding;
				camera->copyDefaultInputBindingPresetTo(imguizmoBinding);
				auto collectedEvents = imguizmoBinding.collectVirtualEvents(m_nextPresentationTimestamp, {
					.imguizmoEvents = { scriptedFrameEvents.imguizmo.data(), scriptedFrameEvents.imguizmo.size() }
				});
				auto& imguizmoEvents = collectedEvents.events;
				const uint32_t vCount = collectedEvents.imguizmoCount;

				if (vCount)
				{
					camera->manipulate({ imguizmoEvents.data(), vCount });
					appendVirtualEventLog("imguizmo", "ImGuizmo", binding.activePlanarIx, camera, imguizmoEvents.data(), vCount);

					if (m_scriptedInput.log)
					{
						for (uint32_t i = 0u; i < vCount; ++i)
						{
							const auto& ev = imguizmoEvents[i];
							m_logger->log("[script] imguizmo virtual %s magnitude=%.6f", ILogger::ELL_INFO, CVirtualGimbalEvent::virtualEventToString(ev.type).data(), ev.magnitude);
						}

						const auto& gimbal = camera->getGimbal();
						const auto pos = gimbal.getPosition();
						const auto euler = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));
						m_logger->log("[script] imguizmo gimbal pos=(%.3f, %.3f, %.3f) euler_deg=(%.3f, %.3f, %.3f)", ILogger::ELL_INFO,
							pos.x, pos.y, pos.z, euler.x, euler.y, euler.z);
					}
				}

				scriptedImguizmoVirtual = vCount ? imguizmoEvents.data() : nullptr;
				scriptedImguizmoVirtualCount = vCount;
			}

			if (m_scriptedInput.enabled && scriptedFrameEvents.goals.size() && !skipCameraInput)
			{
				auto& binding = windowBindings[activeRenderWindowIx];
				auto& planar = m_planarProjections[binding.activePlanarIx];
				auto* camera = planar->getCamera();

				auto logGoalFail = [&](const char* fmt, auto&&... args) -> void
				{
					m_scriptedInput.failed = true;
					m_logger->log(fmt, ILogger::ELL_ERROR, std::forward<decltype(args)>(args)...);
				};

				for (const auto& goalEvent : scriptedFrameEvents.goals)
				{
					const auto result = m_cameraGoalSolver.applyDetailed(camera, goalEvent.goal);
					if (!result.succeeded() || (goalEvent.requireExact && !result.exact))
					{
						logGoalFail("[script][fail] goal_apply frame=%llu status=%s exact=%d details=%s",
							static_cast<unsigned long long>(m_realFrameIx),
							result.succeeded() ? "inexact" : "failed",
							result.exact ? 1 : 0,
							describeApplyResult(result).c_str());
						continue;
					}
				}

				if (camera)
				{
					for (auto& projection : planar->getPlanarProjections())
						nbl::hlsl::syncDynamicPerspectiveProjection(camera, projection);
				}

				if (m_scriptedInput.log && camera)
				{
					const auto& gimbal = camera->getGimbal();
					const auto pos = gimbal.getPosition();
					const auto euler = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));
					m_logger->log("[script] goal_apply gimbal pos=(%.3f, %.3f, %.3f) euler_deg=(%.3f, %.3f, %.3f)", ILogger::ELL_INFO,
						pos.x, pos.y, pos.z, euler.x, euler.y, euler.z);
				}
			}

			auto tryBuildFollowViewProjForPlanar = [&](const uint32_t planarIx, float32_t4x4& outViewProjMatrix) -> bool
			{
				if (activeRenderWindowIx >= windowBindings.size())
					return false;

				const auto& binding = windowBindings[activeRenderWindowIx];
				if (binding.activePlanarIx != planarIx || !binding.boundProjectionIx.has_value())
					return false;
				if (planarIx >= m_planarProjections.size() || !m_planarProjections[planarIx])
					return false;

				auto& planar = m_planarProjections[planarIx];
				auto* camera = planar->getCamera();
				if (!camera)
					return false;

				const auto projectionIx = binding.boundProjectionIx.value();
				auto& projections = planar->getPlanarProjections();
				if (projectionIx >= projections.size())
					return false;

				const auto viewMatrix = getMatrix3x4As4x4(getCastedMatrix<float32_t>(camera->getGimbal().getViewMatrix()));
				const auto projectionMatrix = getCastedMatrix<float32_t>(projections[projectionIx].getProjectionMatrix());
				outViewProjMatrix = mul(projectionMatrix, viewMatrix);
				return true;
			};

			auto setScriptedVisualFollowState = [&](const SCameraFollowVisualMetrics& metrics) -> void
			{
				m_scriptedInput.visualFollowActive = metrics.active;
				m_scriptedInput.visualFollowMode = metrics.mode;
				m_scriptedInput.visualFollowLockValid = metrics.lockValid;
				m_scriptedInput.visualFollowLockAngleDeg = metrics.lockAngleDeg;
				m_scriptedInput.visualFollowTargetDistance = metrics.targetDistance;
				m_scriptedInput.visualFollowProjectedValid = metrics.projectedValid;
				m_scriptedInput.visualFollowTargetCenterNdcX = metrics.projectedNdcX;
				m_scriptedInput.visualFollowTargetCenterNdcY = metrics.projectedNdcY;
				m_scriptedInput.visualFollowTargetCenterNdcRadius = metrics.projectedNdcRadius;
			};

			if (!scriptedFrameEvents.trackedTargetTransforms.empty())
			{
				setFollowTargetTransform(scriptedFrameEvents.trackedTargetTransforms.back().transform);
				applyFollowToConfiguredCameras(true);
				SCameraFollowVisualMetrics followMetrics = {};
				if (activeRenderWindowIx < windowBindings.size())
				{
					const auto planarIx = windowBindings[activeRenderWindowIx].activePlanarIx;
					if (planarIx < m_planarFollowConfigs.size())
					{
						auto* activeCamera = planarIx < m_planarProjections.size() && m_planarProjections[planarIx] ?
							m_planarProjections[planarIx]->getCamera() : nullptr;
						float32_t4x4 viewProjMatrix = float32_t4x4(1.0f);
						const float32_t4x4* viewProjMatrixPtr = tryBuildFollowViewProjForPlanar(planarIx, viewProjMatrix) ? &viewProjMatrix : nullptr;
						followMetrics = nbl::hlsl::buildFollowVisualMetrics(
							activeCamera,
							m_followTarget,
							&m_planarFollowConfigs[planarIx],
							viewProjMatrixPtr);
					}
				}
				setScriptedVisualFollowState(followMetrics);
			}
			else
			{
				applyFollowToConfiguredCameras();
				setScriptedVisualFollowState({});
			}

			if (m_scriptedInput.enabled && m_scriptedInput.checkRuntime.nextCheckIndex < m_scriptedInput.timeline.checks.size())
			{
				auto* camera = [&]() -> ICamera*
				{
					if (m_planarProjections.empty())
						return nullptr;
					auto& binding = windowBindings[activeRenderWindowIx];
					if (binding.activePlanarIx >= m_planarProjections.size())
						return nullptr;
					return m_planarProjections[binding.activePlanarIx]->getCamera();
				}();

				auto logFail = [&](const char* fmt, auto&&... args) -> void
				{
					m_scriptedInput.failed = true;
					m_logger->log(fmt, ILogger::ELL_ERROR, std::forward<decltype(args)>(args)...);
				};

				auto logPass = [&](const char* fmt, auto&&... args) -> void
				{
					if (!m_scriptedInput.log)
						return;
					m_logger->log(fmt, ILogger::ELL_INFO, std::forward<decltype(args)>(args)...);
				};
				SCameraFollowConfig activeFollowConfig = {};
				bool hasActiveFollowConfig = false;
				bool hasFollowViewProjMatrix = false;
				float32_t4x4 followViewProjMatrix = float32_t4x4(1.0f);
				if (activeRenderWindowIx < windowBindings.size())
				{
					const auto& binding = windowBindings[activeRenderWindowIx];
					const auto planarIx = binding.activePlanarIx;
					if (planarIx < m_planarFollowConfigs.size())
					{
						activeFollowConfig = m_planarFollowConfigs[planarIx];
						hasActiveFollowConfig = true;
					}
					if (camera && planarIx < m_planarProjections.size() && m_planarProjections[planarIx] && binding.boundProjectionIx.has_value())
					{
						auto& planar = m_planarProjections[planarIx];
						const auto projectionIx = binding.boundProjectionIx.value();
						auto& projections = planar->getPlanarProjections();
						if (projectionIx < projections.size())
						{
							const auto viewMatrix = getMatrix3x4As4x4(getCastedMatrix<float32_t>(camera->getGimbal().getViewMatrix()));
							const auto projectionMatrix = getCastedMatrix<float32_t>(projections[projectionIx].getProjectionMatrix());
							followViewProjMatrix = mul(projectionMatrix, viewMatrix);
							hasFollowViewProjMatrix = true;
						}
					}
				}

				const auto checkResult = nbl::hlsl::evaluateScriptedChecksForFrame(
					m_scriptedInput.timeline.checks,
					m_scriptedInput.checkRuntime,
					{
						.frame = m_realFrameIx,
						.camera = camera,
						.imguizmoVirtual = scriptedImguizmoVirtual,
						.imguizmoVirtualCount = scriptedImguizmoVirtualCount,
						.trackedTarget = &m_followTarget,
						.followConfig = hasActiveFollowConfig ? &activeFollowConfig : nullptr,
						.followViewProjMatrix = hasFollowViewProjMatrix ? &followViewProjMatrix : nullptr,
						.goalSolver = &m_cameraGoalSolver
					});

				for (const auto& entry : checkResult.logs)
				{
					if (entry.failure)
					{
						m_scriptedInput.failed = true;
						logFail("%s", entry.text.c_str());
					}
					else
					{
						logPass("%s", entry.text.c_str());
					}
				}

				if (!m_scriptedInput.summaryReported && m_scriptedInput.checkRuntime.nextCheckIndex >= m_scriptedInput.timeline.checks.size())
				{
					m_scriptedInput.summaryReported = true;
					if (m_scriptedInput.failed)
						m_logger->log("[script] checks result: FAIL", ILogger::ELL_ERROR);
					else
						m_logger->log("[script] checks result: PASS", ILogger::ELL_INFO);
				}
			}

			UpdateUiMetrics();
			m_ui.manager->update(params);


}


