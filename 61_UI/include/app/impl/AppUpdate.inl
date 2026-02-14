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

			if (m_scriptedInput.log && (scriptedKeyboard.size() || scriptedMouse.size() || scriptedImguizmo.size()))
			{
				m_logger->log("[script] frame %llu input kb=%zu mouse=%zu imguizmo=%zu", ILogger::ELL_INFO,
					static_cast<unsigned long long>(m_realFrameIx),
					scriptedKeyboard.size(),
					scriptedMouse.size(),
					scriptedImguizmo.size());
			}

			if (enableActiveCameraMovement && !skipCameraInput)
			{
				auto& binding = windowBindings[activeRenderWindowIx];
				auto& planar = m_planarProjections[binding.activePlanarIx];
				auto* camera = planar->getCamera();

				assert(binding.boundProjectionIx.has_value());
				auto& projection = planar->getPlanarProjections()[binding.boundProjectionIx.value()];

				static std::vector<CVirtualGimbalEvent> virtualEvents(0x45);
				uint32_t vCount = {};
				uint32_t vKeyboardEventsCount = {};
				uint32_t vMouseEventsCount = {};

				projection.beginInputProcessing(m_nextPresentationTimestamp);
				{
					projection.processKeyboard(nullptr, vKeyboardEventsCount, {});
					projection.processMouse(nullptr, vMouseEventsCount, {});

					const auto totalCount = vKeyboardEventsCount + vMouseEventsCount;
					if (virtualEvents.size() < totalCount)
						virtualEvents.resize(totalCount);

					auto* output = virtualEvents.data();
					projection.processKeyboard(output, vKeyboardEventsCount, { cameraKeyboardEvents.data(), cameraKeyboardEvents.size() });
					for (uint32_t i = 0u; i < vKeyboardEventsCount; ++i)
						output[i].magnitude *= m_cameraControls.keyboardScale;
					output += vKeyboardEventsCount;

					if (isOrbitLikeCamera(camera))
					{
						if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
							projection.processMouse(output, vMouseEventsCount, { cameraMouseEvents.data(), cameraMouseEvents.size() });
						else
							vMouseEventsCount = 0;
					}
					else
					{
						projection.processMouse(output, vMouseEventsCount, { cameraMouseEvents.data(), cameraMouseEvents.size() });
					}

					vCount = vKeyboardEventsCount + vMouseEventsCount;
				}
				projection.endInputProcessing();

				if (vCount)
				{
					applyVirtualEventScaling(virtualEvents, vCount);

					const char* controllerLabel = "Keyboard/Mouse";
					auto applyEventsToCamera = [&](ICamera* target, uint32_t planarIx)
					{
						if (!target)
							return;

						if (m_cameraControls.worldTranslate)
						{
							std::vector<CVirtualGimbalEvent> perCameraEvents(virtualEvents.begin(), virtualEvents.begin() + vCount);
							uint32_t perCount = vCount;
							remapTranslationToWorld(target, perCameraEvents, perCount);
							if (perCount)
								target->manipulate({ perCameraEvents.data(), perCount });
						}
						else
						{
							target->manipulate({ virtualEvents.data(), vCount });
						}

						applyConstraintsToCamera(target);
						appendVirtualEventLog("input", controllerLabel, planarIx, target, virtualEvents.data(), vCount);
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

				static std::vector<CVirtualGimbalEvent> imguizmoEvents(0x20);
				uint32_t vCount = 0u;

				camera->beginInputProcessing(m_nextPresentationTimestamp);
				{
					camera->processImguizmo(nullptr, vCount, {});
					if (imguizmoEvents.size() < vCount)
						imguizmoEvents.resize(vCount);

					camera->processImguizmo(imguizmoEvents.data(), vCount, { scriptedImguizmo.data(), scriptedImguizmo.size() });
				}
				camera->endInputProcessing();

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
						if (check.hasPosDeltaConstraint)
						{
							if (dpos < check.minPosDelta || dpos > check.posTolerance)
							{
								ok = false;
								logFail("[script][fail] gimbal_step frame=%llu pos_delta=%.6f expected=[%.6f, %.6f]",
									static_cast<unsigned long long>(frame), dpos, check.minPosDelta, check.posTolerance);
							}
						}
						if (check.hasEulerDeltaConstraint)
						{
							if (dmax < check.minEulerDeltaDeg || dmax > check.eulerToleranceDeg)
							{
								ok = false;
								logFail("[script][fail] gimbal_step frame=%llu euler_delta=%.6f expected=[%.6f, %.6f]",
									static_cast<unsigned long long>(frame), dmax, check.minEulerDeltaDeg, check.eulerToleranceDeg);
							}
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

