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
			std::vector<float32_t4x4> scriptedImguizmo;
			std::vector<ScriptedInputEvent::ActionData> scriptedActions;
			std::vector<ScriptedInputEvent::GoalData> scriptedGoals;
			const CVirtualGimbalEvent* scriptedImguizmoVirtual = nullptr;
			uint32_t scriptedImguizmoVirtualCount = 0u;

			if (m_scriptedInput.enabled && m_scriptedInput.nextEventIndex < m_scriptedInput.events.size())
			{
				const auto frame = m_realFrameIx;
				while (m_scriptedInput.nextEventIndex < m_scriptedInput.events.size() &&
					m_scriptedInput.events[m_scriptedInput.nextEventIndex].frame == frame)
				{
					const auto& ev = m_scriptedInput.events[m_scriptedInput.nextEventIndex];

					if (ev.type == ScriptedInputEvent::Type::Keyboard)
					{
						SKeyboardEvent e(m_nextPresentationTimestamp);
						e.keyCode = ev.keyboard.key;
						e.action = ev.keyboard.action;
						e.window = m_window.get();
						scriptedKeyboard.emplace_back(e);
					}
					else if (ev.type == ScriptedInputEvent::Type::Mouse)
					{
						SMouseEvent e(m_nextPresentationTimestamp);
						e.window = m_window.get();
						e.type = ev.mouse.type;
						if (ev.mouse.type == ui::SMouseEvent::EET_CLICK)
						{
							e.clickEvent.mouseButton = ev.mouse.button;
							e.clickEvent.action = ev.mouse.action;
							e.clickEvent.clickPosX = ev.mouse.x;
							e.clickEvent.clickPosY = ev.mouse.y;
						}
						else if (ev.mouse.type == ui::SMouseEvent::EET_SCROLL)
						{
							e.scrollEvent.verticalScroll = ev.mouse.v;
							e.scrollEvent.horizontalScroll = ev.mouse.h;
						}
						else if (ev.mouse.type == ui::SMouseEvent::EET_MOVEMENT)
						{
							e.movementEvent.relativeMovementX = ev.mouse.dx;
							e.movementEvent.relativeMovementY = ev.mouse.dy;
						}
						scriptedMouse.emplace_back(e);
					}
					else if (ev.type == ScriptedInputEvent::Type::Imguizmo)
					{
						scriptedImguizmo.emplace_back(ev.imguizmo);
					}
					else if (ev.type == ScriptedInputEvent::Type::Action)
					{
						scriptedActions.emplace_back(ev.action);
					}
					else if (ev.type == ScriptedInputEvent::Type::Goal)
					{
						scriptedGoals.emplace_back(ev.goal);
					}

					++m_scriptedInput.nextEventIndex;
				}
			}

			if (m_scriptedInput.enabled && scriptedActions.size())
			{
				auto applyAction = [&](const ScriptedInputEvent::ActionData& action) -> void
				{
					switch (action.kind)
					{
						case ScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow:
						{
							if (action.value < 0 || static_cast<size_t>(action.value) >= windowBindings.size())
							{
								m_logger->log("[script][warn] action set_active_render_window out of range: %d", ILogger::ELL_WARNING, action.value);
								return;
							}
							activeRenderWindowIx = static_cast<uint32_t>(action.value);
						} break;

						case ScriptedInputEvent::ActionData::Kind::SetActivePlanar:
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

						case ScriptedInputEvent::ActionData::Kind::SetProjectionType:
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

						case ScriptedInputEvent::ActionData::Kind::SetProjectionIndex:
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

						case ScriptedInputEvent::ActionData::Kind::SetUseWindow:
						{
							useWindow = action.value != 0;
						} break;

						case ScriptedInputEvent::ActionData::Kind::SetLeftHanded:
						{
							auto& binding = windowBindings[activeRenderWindowIx];
							binding.leftHandedProjection = action.value != 0;
						} break;

						case ScriptedInputEvent::ActionData::Kind::ResetActiveCamera:
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

				for (const auto& action : scriptedActions)
					if (action.kind == ScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow)
						applyAction(action);

				for (const auto& action : scriptedActions)
					if (action.kind != ScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow)
						applyAction(action);

				if (m_scriptedInput.log)
					m_logger->log("[script] frame %llu actions=%zu", ILogger::ELL_INFO, static_cast<unsigned long long>(m_realFrameIx), scriptedActions.size());
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

			if (m_scriptedInput.log && (scriptedKeyboard.size() || scriptedMouse.size() || scriptedImguizmo.size() || scriptedGoals.size()))
			{
				m_logger->log("[script] frame %llu input kb=%zu mouse=%zu imguizmo=%zu goals=%zu", ILogger::ELL_INFO,
					static_cast<unsigned long long>(m_realFrameIx),
					scriptedKeyboard.size(),
					scriptedMouse.size(),
					scriptedImguizmo.size(),
					scriptedGoals.size());
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

			if (m_scriptedInput.enabled && scriptedImguizmo.size() && !skipCameraInput)
			{
				auto& binding = windowBindings[activeRenderWindowIx];
				auto& planar = m_planarProjections[binding.activePlanarIx];
				auto* camera = planar->getCamera();

				CGimbalInputBinder imguizmoBinding;
				camera->copyDefaultInputBindingPresetTo(imguizmoBinding);
				auto collectedEvents = imguizmoBinding.collectVirtualEvents(m_nextPresentationTimestamp, {
					.imguizmoEvents = { scriptedImguizmo.data(), scriptedImguizmo.size() }
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

			if (m_scriptedInput.enabled && scriptedGoals.size() && !skipCameraInput)
			{
				auto& binding = windowBindings[activeRenderWindowIx];
				auto& planar = m_planarProjections[binding.activePlanarIx];
				auto* camera = planar->getCamera();

				auto logGoalFail = [&](const char* fmt, auto&&... args) -> void
				{
					m_scriptedInput.failed = true;
					m_logger->log(fmt, ILogger::ELL_ERROR, std::forward<decltype(args)>(args)...);
				};

				for (const auto& goalEvent : scriptedGoals)
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

			if (m_scriptedInput.enabled && m_scriptedInput.nextCheckIndex < m_scriptedInput.checks.size())
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

				auto angleDiffDeg = [](float a, float b) -> float
				{
					float d = std::fmod(a - b + 180.0f, 360.0f);
					if (d < 0.0f)
						d += 360.0f;
					return std::abs(d - 180.0f);
				};

				auto isFinite3 = [](const float32_t3& v) -> bool
				{
					return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
				};
				auto setStepReference = [&](const float32_t3& newPos, const float32_t3& newEuler) -> void
				{
					m_scriptedInput.stepValid = true;
					m_scriptedInput.stepPos = newPos;
					m_scriptedInput.stepEulerDeg = newEuler;
				};

				const auto frame = m_realFrameIx;
				while (m_scriptedInput.nextCheckIndex < m_scriptedInput.checks.size() &&
					m_scriptedInput.checks[m_scriptedInput.nextCheckIndex].frame == frame)
				{
					const auto& check = m_scriptedInput.checks[m_scriptedInput.nextCheckIndex];

					if (!camera)
					{
						logFail("[script][fail] check frame=%llu no active camera", static_cast<unsigned long long>(frame));
						++m_scriptedInput.nextCheckIndex;
						continue;
					}

					const auto& gimbal = camera->getGimbal();
					const auto pos = gimbal.getPosition();
					const auto euler = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));

					if (!isFinite3(pos) || !isFinite3(euler))
					{
						logFail("[script][fail] check frame=%llu non-finite gimbal state", static_cast<unsigned long long>(frame));
						++m_scriptedInput.nextCheckIndex;
						continue;
					}

					if (check.kind == ScriptedInputCheck::Kind::Baseline)
					{
						m_scriptedInput.baselineValid = true;
						m_scriptedInput.baselinePos = pos;
						m_scriptedInput.baselineEulerDeg = euler;
						setStepReference(pos, euler);
						logPass("[script][pass] baseline frame=%llu pos=(%.3f, %.3f, %.3f) euler_deg=(%.3f, %.3f, %.3f)",
							static_cast<unsigned long long>(frame),
							pos.x, pos.y, pos.z, euler.x, euler.y, euler.z);
					}
					else if (check.kind == ScriptedInputCheck::Kind::ImguizmoVirtual)
					{
						bool ok = true;
						if (!scriptedImguizmoVirtual || scriptedImguizmoVirtualCount == 0u)
						{
							ok = false;
						}
						else
						{
							for (const auto& expected : check.expectedVirtualEvents)
							{
								bool found = false;
								double actual = 0.0;
								for (uint32_t i = 0u; i < scriptedImguizmoVirtualCount; ++i)
								{
									if (scriptedImguizmoVirtual[i].type == expected.type)
									{
										found = true;
										actual = scriptedImguizmoVirtual[i].magnitude;
										break;
									}
								}
								if (!found || std::abs(actual - expected.magnitude) > check.tolerance)
								{
									ok = false;
									logFail("[script][fail] imguizmo_virtual frame=%llu type=%s expected=%.6f actual=%.6f tol=%.6f",
										static_cast<unsigned long long>(frame),
										CVirtualGimbalEvent::virtualEventToString(expected.type).data(),
										expected.magnitude,
										actual,
										check.tolerance);
								}
							}
						}

						if (ok)
							logPass("[script][pass] imguizmo_virtual frame=%llu events=%zu", static_cast<unsigned long long>(frame), check.expectedVirtualEvents.size());
					}
					else if (check.kind == ScriptedInputCheck::Kind::GimbalNear)
					{
						bool ok = true;
						if (check.hasExpectedPos)
						{
							const auto diff = float32_t3(pos.x - check.expectedPos.x, pos.y - check.expectedPos.y, pos.z - check.expectedPos.z);
							const auto d = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
							if (d > check.posTolerance)
							{
								ok = false;
								logFail("[script][fail] gimbal_near frame=%llu pos_diff=%.6f tol=%.6f",
									static_cast<unsigned long long>(frame), d, check.posTolerance);
							}
						}
						if (check.hasExpectedEuler)
						{
							const auto dx = angleDiffDeg(euler.x, check.expectedEulerDeg.x);
							const auto dy = angleDiffDeg(euler.y, check.expectedEulerDeg.y);
							const auto dz = angleDiffDeg(euler.z, check.expectedEulerDeg.z);
							const auto dmax = std::max(dx, std::max(dy, dz));
							if (dmax > check.eulerToleranceDeg)
							{
								ok = false;
								logFail("[script][fail] gimbal_near frame=%llu euler_diff=%.6f tol=%.6f",
									static_cast<unsigned long long>(frame), dmax, check.eulerToleranceDeg);
							}
						}

						if (ok)
							logPass("[script][pass] gimbal_near frame=%llu", static_cast<unsigned long long>(frame));
					}
					else if (check.kind == ScriptedInputCheck::Kind::GimbalDelta)
					{
						if (!m_scriptedInput.baselineValid)
						{
							logFail("[script][fail] gimbal_delta frame=%llu missing baseline", static_cast<unsigned long long>(frame));
						}
						else
						{
							const auto diff = float32_t3(pos.x - m_scriptedInput.baselinePos.x, pos.y - m_scriptedInput.baselinePos.y, pos.z - m_scriptedInput.baselinePos.z);
							const auto dpos = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
							const auto dx = angleDiffDeg(euler.x, m_scriptedInput.baselineEulerDeg.x);
							const auto dy = angleDiffDeg(euler.y, m_scriptedInput.baselineEulerDeg.y);
							const auto dz = angleDiffDeg(euler.z, m_scriptedInput.baselineEulerDeg.z);
							const auto dmax = std::max(dx, std::max(dy, dz));

							if (dpos > check.posTolerance || dmax > check.eulerToleranceDeg)
							{
								logFail("[script][fail] gimbal_delta frame=%llu pos_diff=%.6f tol=%.6f euler_diff=%.6f tol=%.6f",
									static_cast<unsigned long long>(frame),
									dpos, check.posTolerance,
									dmax, check.eulerToleranceDeg);
							}
							else
							{
								logPass("[script][pass] gimbal_delta frame=%llu pos_diff=%.6f euler_diff=%.6f",
									static_cast<unsigned long long>(frame), dpos, dmax);
							}
						}
					}
					else if (check.kind == ScriptedInputCheck::Kind::GimbalStep)
					{
						if (!m_scriptedInput.stepValid)
						{
							if (m_scriptedInput.baselineValid)
								setStepReference(m_scriptedInput.baselinePos, m_scriptedInput.baselineEulerDeg);
							else
							{
								logFail("[script][fail] gimbal_step frame=%llu missing step reference", static_cast<unsigned long long>(frame));
								setStepReference(pos, euler);
								++m_scriptedInput.nextCheckIndex;
								continue;
							}
						}

						const auto diff = float32_t3(pos.x - m_scriptedInput.stepPos.x, pos.y - m_scriptedInput.stepPos.y, pos.z - m_scriptedInput.stepPos.z);
						const auto dpos = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
						const auto dx = angleDiffDeg(euler.x, m_scriptedInput.stepEulerDeg.x);
						const auto dy = angleDiffDeg(euler.y, m_scriptedInput.stepEulerDeg.y);
						const auto dz = angleDiffDeg(euler.z, m_scriptedInput.stepEulerDeg.z);
						const auto dmax = std::max(dx, std::max(dy, dz));

						bool ok = true;
						bool requiresProgress = false;
						bool hasProgress = false;
						if (check.hasPosDeltaConstraint)
						{
							if (dpos > check.posTolerance)
							{
								ok = false;
								logFail("[script][fail] gimbal_step frame=%llu pos_delta=%.6f max=%.6f",
									static_cast<unsigned long long>(frame), dpos, check.posTolerance);
							}
							if (check.minPosDelta > 0.0f)
							{
								requiresProgress = true;
								hasProgress = hasProgress || dpos >= check.minPosDelta;
							}
						}
						if (check.hasEulerDeltaConstraint)
						{
							if (dmax > check.eulerToleranceDeg)
							{
								ok = false;
								logFail("[script][fail] gimbal_step frame=%llu euler_delta=%.6f max=%.6f",
									static_cast<unsigned long long>(frame), dmax, check.eulerToleranceDeg);
							}
							if (check.minEulerDeltaDeg > 0.0f)
							{
								requiresProgress = true;
								hasProgress = hasProgress || dmax >= check.minEulerDeltaDeg;
							}
						}
						if (requiresProgress && !hasProgress)
						{
							ok = false;
							logFail("[script][fail] gimbal_step frame=%llu missing progress pos_delta=%.6f euler_delta=%.6f",
								static_cast<unsigned long long>(frame), dpos, dmax);
						}

						if (ok)
							logPass("[script][pass] gimbal_step frame=%llu pos_delta=%.6f euler_delta=%.6f",
								static_cast<unsigned long long>(frame), dpos, dmax);
						setStepReference(pos, euler);
					}

					++m_scriptedInput.nextCheckIndex;
				}

				if (!m_scriptedInput.summaryReported && m_scriptedInput.nextCheckIndex >= m_scriptedInput.checks.size())
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


