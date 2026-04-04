#include "app/App.hpp"

void App::imguiListen()
{
			ImGuiIO& io = ImGui::GetIO();
			if (m_ciMode)
			{
				io.IniFilename = nullptr;
				useWindow = true;
			}
			
			ImGuizmo::BeginFrame();
			{
				if (!m_ciMode)
				{
				}

				SImResourceInfo info;
				info.samplerIx = (uint16_t)nbl::ext::imgui::UI::DefaultSamplerIx::USER;

				// ORBIT CAMERA TEST
				{
					for (auto& planar : m_planarProjections)
					{
						auto* camera = planar->getCamera();
						if (camera)
						{
							const auto targetPosition = hlsl::transpose(getMatrix3x4As4x4(m_model))[3];
							if (camera->trySetSphericalTarget(float64_t3(targetPosition.x, targetPosition.y, targetPosition.z)))
								camera->manipulate({}, {});
						}
					}
				}

				// render bound planar camera views onto GUI windows
				if (useWindow)
				{
					syncVisualDebugWindowBindings();
					const bool hideSceneGizmos = enableActiveCameraMovement || (m_scriptedInput.enabled && m_scriptedInput.visualDebug);
					if(hideSceneGizmos)
						ImGuizmo::Enable(false);
					else
						ImGuizmo::Enable(true);

					size_t gizmoIx = {};
					size_t manipulationCounter = {};
					const std::optional<uint32_t> modelInUseIx = ImGuizmo::IsUsingAny() ? std::optional<uint32_t>(boundPlanarCameraIxToManipulate.has_value() ? 1u + boundPlanarCameraIxToManipulate.value() : 0u) : std::optional<uint32_t>(std::nullopt);

					for (uint32_t windowIx = 0; windowIx < windowBindings.size(); ++windowIx)
					{
						// setup
						{
							const auto& rw = wInit.renderWindows[windowIx];
							const ImGuiCond windowCond = m_ciMode ? ImGuiCond_Always : ImGuiCond_Appearing;
							ImGui::SetNextWindowPos({ rw.iPos.x, rw.iPos.y }, windowCond);
							ImGui::SetNextWindowSize({ rw.iSize.x, rw.iSize.y }, windowCond);
						}
						ImGui::SetNextWindowSizeConstraints(ImVec2(0x45, 0x45), ImVec2(7680, 4320));

						ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
						ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
						ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
						ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
						const std::string ident = "Render Window \"" + std::to_string(windowIx) + "\"";

						ImGui::Begin(ident.data(), 0, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);
						const ImVec2 contentRegionSize = ImGui::GetContentRegionAvail(), windowPos = ImGui::GetWindowPos(), cursorPos = ImGui::GetCursorScreenPos();

						ImGuiWindow* window = ImGui::GetCurrentWindow();
						{
							const auto mPos = ImGui::GetMousePos();

							if (mPos.x < cursorPos.x || mPos.y < cursorPos.y || mPos.x > cursorPos.x + contentRegionSize.x || mPos.y > cursorPos.y + contentRegionSize.y)
								window->Flags &= ~ImGuiWindowFlags_NoMove;
							else
								window->Flags |= ImGuiWindowFlags_NoMove;
						}

						// setup bound entities for the window like camera & projections
						auto& binding = windowBindings[windowIx];
						auto& planarBound = m_planarProjections[binding.activePlanarIx];
						assert(planarBound);

						binding.aspectRatio = contentRegionSize.x / contentRegionSize.y;
						auto* planarViewCameraBound = planarBound->getCamera();

						assert(planarViewCameraBound);
						assert(binding.boundProjectionIx.has_value());
						
						auto& projection = planarBound->getPlanarProjections()[binding.boundProjectionIx.value()];
						applyDollyZoomProjection(planarViewCameraBound, projection);
						projection.update(binding.leftHandedProjection, binding.aspectRatio);

						// TODO: 
						// would be nice to normalize imguizmo visual vectors (possible with styles)

						// first 0th texture is for UI texture atlas, then there are our window textures
						auto fboImguiTextureID = windowIx + 1u;
						info.textureID = fboImguiTextureID;

						if(binding.allowGizmoAxesToFlip)
							ImGuizmo::AllowAxisFlip(true);
						else
							ImGuizmo::AllowAxisFlip(false);

						if(projection.getParameters().m_type == IPlanarProjection::CProjection::Orthographic)
							ImGuizmo::SetOrthographic(true);
						else
							ImGuizmo::SetOrthographic(false);

						ImGuizmo::SetDrawlist();
						ImGui::Image(info, contentRegionSize);
						ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);
						{
							const char* projLabel = projection.getParameters().m_type == IPlanarProjection::CProjection::Perspective ? "Persp" : "Ortho";
							const std::string overlayText = "Planar " + std::to_string(binding.activePlanarIx) + " | " + projLabel + " | W" + std::to_string(windowIx);
							const std::string cameraText = std::string(getCameraTypeLabel(planarViewCameraBound)) + ": " + std::string(getCameraTypeDescription(planarViewCameraBound));
							const std::string frustumText = "Frustum: active camera (hidden in owner view)";
							const ImVec2 textSize = ImGui::CalcTextSize(overlayText.c_str());
							const ImVec2 descSize = ImGui::CalcTextSize(cameraText.c_str());
							const ImVec2 frustumSize = ImGui::CalcTextSize(frustumText.c_str());
							const ImVec2 pad = ImVec2(6.0f, 4.0f);
							const float lineGap = 2.0f;
							const float width = std::max(std::max(textSize.x, descSize.x), frustumSize.x);
							const float height = textSize.y + descSize.y + frustumSize.y + lineGap * 2.0f + pad.y * 2.0f;
							ImVec2 overlayPos = ImVec2(cursorPos.x + contentRegionSize.x - width - pad.x * 2.0f - 6.0f, cursorPos.y + 6.0f);
							overlayPos.x = std::max(overlayPos.x, cursorPos.x + 6.0f);
							ImVec2 overlayMax = ImVec2(overlayPos.x + width + pad.x * 2.0f, overlayPos.y + height);
							auto* drawList = ImGui::GetWindowDrawList();
							drawList->AddRectFilled(overlayPos, overlayMax, ImGui::ColorConvertFloat4ToU32(ImVec4(0.05f, 0.06f, 0.08f, 0.80f)), 6.0f);
							drawList->AddRect(overlayPos, overlayMax, ImGui::ColorConvertFloat4ToU32(ImVec4(0.60f, 0.66f, 0.76f, 0.80f)), 6.0f);
							drawList->AddText(ImVec2(overlayPos.x + pad.x, overlayPos.y + pad.y), ImGui::ColorConvertFloat4ToU32(ImVec4(0.96f, 0.98f, 1.0f, 1.0f)), overlayText.c_str());
							drawList->AddText(ImVec2(overlayPos.x + pad.x, overlayPos.y + pad.y + textSize.y + lineGap), ImGui::ColorConvertFloat4ToU32(ImVec4(0.78f, 0.82f, 0.90f, 1.0f)), cameraText.c_str());
							drawList->AddText(ImVec2(overlayPos.x + pad.x, overlayPos.y + pad.y + textSize.y + descSize.y + lineGap * 2.0f), ImGui::ColorConvertFloat4ToU32(ImVec4(0.96f, 0.90f, 0.36f, 1.0f)), frustumText.c_str());
						}

						const bool windowHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
						const bool windowFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
						if (!(m_scriptedInput.enabled && m_scriptedInput.exclusive))
						{
							if (!m_scriptedInput.enabled && windowHovered)
								activeRenderWindowIx = windowIx;
							else if (windowFocused)
								activeRenderWindowIx = windowIx;
						}

						// we render a scene from view of a camera bound to planar window
						ImGuizmoPlanarM16InOut imguizmoPlanar;
						imguizmoPlanar.view = getCastedMatrix<float32_t>(hlsl::transpose(getMatrix3x4As4x4(planarViewCameraBound->getGimbal().getViewMatrix())));
						imguizmoPlanar.projection = getCastedMatrix<float32_t>(hlsl::transpose(projection.getProjectionMatrix()));
						const auto viewMatrix = getMatrix3x4As4x4(getCastedMatrix<float32_t>(planarViewCameraBound->getGimbal().getViewMatrix()));
						const auto projectionMatrix = getCastedMatrix<float32_t>(projection.getProjectionMatrix());

						if (flipGizmoY) // note we allow to flip gizmo just to match our coordinates
							imguizmoPlanar.projection[1][1] *= -1.f; // https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/	

						if (!hideSceneGizmos)
						{
							for (uint32_t modelIx = 0; modelIx < 1u + m_planarProjections.size(); modelIx++)
							{
								ImGuizmo::PushID(gizmoIx); ++gizmoIx;

								const bool isCameraGimbalTarget = modelIx; // I assume scene demo model is 0th ix, left are planar cameras
								ICamera* const targetGimbalManipulationCamera = isCameraGimbalTarget ? m_planarProjections[modelIx - 1u]->getCamera() : nullptr;

							// if we try to manipulate a camera which appears to be the same camera we see scene from then obvsly it doesn't make sense to manipulate its gizmo so we skip it
							// EDIT: it actually makes some sense if you assume render planar view is rendered with ortho projection, but we would need to add imguizmo controller virtual map
							// to ban forward/backward in this mode if this condition is true
								if (targetGimbalManipulationCamera == planarViewCameraBound)
								{
									ImGuizmo::PopID();
									continue;
								}

								ImGuizmoModelM16InOut imguizmoModel;

								if (isCameraGimbalTarget)
								{
									assert(targetGimbalManipulationCamera);
									imguizmoModel.inTRS = getCastedMatrix<float32_t>(targetGimbalManipulationCamera->getGimbal().template operator() < float64_t4x4 > ());
								}
								else
									imguizmoModel.inTRS = hlsl::transpose(getMatrix3x4As4x4(m_model));

								const float gizmoWorldRadius = 0.22f;
								float32_t3 gizmoWorldPos = {};
								if (isCameraGimbalTarget)
									gizmoWorldPos = getCastedVector<float32_t>(targetGimbalManipulationCamera->getGimbal().getPosition());
								else
								{
									const auto modelPos = hlsl::transpose(getMatrix3x4As4x4(m_model))[3];
									gizmoWorldPos = float32_t3(modelPos.x, modelPos.y, modelPos.z);
								}

								const auto viewPos = mul(viewMatrix, float32_t4(gizmoWorldPos, 1.0f));
								const float depth = std::max(0.001f, std::abs(viewPos.z));
								float gizmoSizeClip = 0.1f;
								if (projection.getParameters().m_type == IPlanarProjection::CProjection::Perspective)
									gizmoSizeClip = (gizmoWorldRadius * projectionMatrix[1][1]) / depth;
								else
									gizmoSizeClip = gizmoWorldRadius * projectionMatrix[1][1];
								ImGuizmo::SetGizmoSizeClipSpace(gizmoSizeClip);

								imguizmoModel.outTRS = imguizmoModel.inTRS;
								{
									const bool success = ImGuizmo::Manipulate(&imguizmoPlanar.view[0][0], &imguizmoPlanar.projection[0][0], ImGuizmo::OPERATION::UNIVERSAL, mCurrentGizmoMode, &imguizmoModel.outTRS[0][0], &imguizmoModel.outDeltaTRS[0][0], useSnap ? &snap[0] : nullptr);

								if (success)
								{
									if (targetGimbalManipulationCamera)
									{
										const auto referenceFrame = getCastedMatrix<float64_t>(*reinterpret_cast<float32_t4x4*>(ImGuizmo::GetReferenceFrame()));

										boundCameraToManipulate = smart_refctd_ptr<ICamera>(targetGimbalManipulationCamera);
										boundPlanarCameraIxToManipulate = modelIx - 1u;

										// TODO: TO BE REMOVED, ONLY FOR TESTING ITS INCOMPLETE TYPE!
										const auto& imguizmoCtx = ImGuizmo::GetContext();

										struct
										{
											float32_t3 t, r, s;
										} out, delta;

										ImGuizmo::DecomposeMatrixToComponents(&imguizmoModel.outTRS[0][0], &out.t[0], &out.r[0], &out.s[0]);
										ImGuizmo::DecomposeMatrixToComponents(&imguizmoModel.outDeltaTRS[0][0], &delta.t[0], &delta.r[0], &delta.s[0]);
										{
											std::vector<CVirtualGimbalEvent> virtualEvents;
	
											auto requestMagnitudeUpdateWithScalar = [&](float signPivot, float dScalar, float dMagnitude, auto positive, auto negative)
											{
												if (dScalar != signPivot)
												{
													auto& ev = virtualEvents.emplace_back();
													auto code = (dScalar > signPivot) ? positive : negative;

													ev.type = code;
													ev.magnitude += dMagnitude;
												}
											};
		
											// TODO TESTING STUFF WITH MY IMGUIZMO UPDATES
											// IT WILL BE REMOVED ONCE ALL TESTS ARE DONE 
											// AND CONTROLLER API WILL BE USED INSTEAD

											// translations
											{
												ImGuizmo::OPERATION ioType;
												const auto dScalar = ImGuizmo::GetTranslationDeltaScalar(&ioType);

												if (dScalar)
												{
													switch (ioType)
													{
													case ImGuizmo::OPERATION::TRANSLATE_X:
													{
														requestMagnitudeUpdateWithScalar(0.f, dScalar, std::abs(dScalar), CVirtualGimbalEvent::VirtualEventType::MoveRight, CVirtualGimbalEvent::VirtualEventType::MoveLeft);
													} break;

													case ImGuizmo::OPERATION::TRANSLATE_Y:
													{
														requestMagnitudeUpdateWithScalar(0.f, dScalar, std::abs(dScalar), CVirtualGimbalEvent::VirtualEventType::MoveUp, CVirtualGimbalEvent::VirtualEventType::MoveDown);
													} break;

													case ImGuizmo::OPERATION::TRANSLATE_Z:
													{
														requestMagnitudeUpdateWithScalar(0.f, dScalar, std::abs(dScalar), CVirtualGimbalEvent::VirtualEventType::MoveForward, CVirtualGimbalEvent::VirtualEventType::MoveBackward);
													} break;

													default: break;
													}
												}
											}

											// TODO: ok becuase I have only one reference from imguizmo I must do it differently when 
											// I have local base && want to do rotation with respect to world instead; we almost there
												
											// rotations
											{
												ImGuizmo::OPERATION ioType;
												float dRadians = ImGuizmo::GetRotationDeltaRadians(&ioType);

												if (dRadians)
												{
													switch (ioType)
													{
													case ImGuizmo::OPERATION::ROTATE_X:
													{
														requestMagnitudeUpdateWithScalar(0.f, dRadians, std::abs(dRadians), CVirtualGimbalEvent::VirtualEventType::TiltUp, CVirtualGimbalEvent::VirtualEventType::TiltDown);
													} break;

													case ImGuizmo::OPERATION::ROTATE_Y:
													{
														requestMagnitudeUpdateWithScalar(0.f, dRadians, std::abs(dRadians), CVirtualGimbalEvent::VirtualEventType::PanRight, CVirtualGimbalEvent::VirtualEventType::PanLeft);
													} break;

													case ImGuizmo::OPERATION::ROTATE_Z:
													{
														requestMagnitudeUpdateWithScalar(0.f, dRadians, std::abs(dRadians), CVirtualGimbalEvent::VirtualEventType::RollRight, CVirtualGimbalEvent::VirtualEventType::RollLeft);
													} break;

													default:
														assert(false); break; // should never be hit
													}
												}
											}

											const auto vCount = virtualEvents.size();

											if (vCount)
											{
												// I start to think controller should be able to set sensitivity to scale magnitudes of generated events
												// in order for camera to not keep any magnitude scalars like move or rotation speed scales

												targetGimbalManipulationCamera->manipulateWithUnitMotionScales({ virtualEvents.data(), vCount }, &referenceFrame);
											}

										}
									}
									else
									{
										// again, for scene demo model full affine transformation without limits is assumed 
										m_model = float32_t3x4(hlsl::transpose(imguizmoModel.outTRS));
										boundCameraToManipulate = nullptr;
										boundPlanarCameraIxToManipulate = std::nullopt;
									}
								}

									if (ImGuizmo::IsOver() and not ImGuizmo::IsUsingAny() && not enableActiveCameraMovement)
									{
									if (targetGimbalManipulationCamera && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
									{
										const uint32_t newPlanarIx = modelIx - 1u;
										if (newPlanarIx < m_planarProjections.size())
										{
											binding.activePlanarIx = newPlanarIx;
											binding.pickDefaultProjections(m_planarProjections[binding.activePlanarIx]->getPlanarProjections());
											if (!(m_scriptedInput.enabled && m_scriptedInput.exclusive))
												activeRenderWindowIx = windowIx;
										}
									}

									ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.2f, 0.2f, 0.2f, 0.8f));
									ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
									ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);

									ImGuiIO& io = ImGui::GetIO();
									ImVec2 mousePos = io.MousePos;
									ImGui::SetNextWindowPos(ImVec2(mousePos.x + 10, mousePos.y + 10), ImGuiCond_Always);

									ImGui::Begin("InfoOverlay", nullptr,
										ImGuiWindowFlags_NoDecoration |
										ImGuiWindowFlags_AlwaysAutoResize |
										ImGuiWindowFlags_NoSavedSettings);

									std::string ident;

									if (targetGimbalManipulationCamera)
										ident = targetGimbalManipulationCamera->getIdentifier();
									else
										ident = "Geometry Creator Object";

									ImGui::Text("Identifier: %s", ident.c_str());
									ImGui::Text("Object Ix: %u", modelIx);
									if (targetGimbalManipulationCamera)
									{
										ImGui::Separator();
										ImGui::TextDisabled("RMB: switch view to this camera");
										ImGui::TextDisabled("LMB drag: manipulate gizmo");
										ImGui::TextDisabled("SPACE: toggle move mode");
									}

									ImGui::End();

									ImGui::PopStyleVar();
									ImGui::PopStyleColor(2);
									}
								}
								ImGuizmo::PopID();
							}
						}

						ImGui::End();
						ImGui::PopStyleColor(1);
						ImGui::PopStyleVar(3);
					}
					if (windowBindings.size() > 1u)
					{
						const auto& topRw = wInit.renderWindows[0];
						const float splitY = topRw.iPos.y + topRw.iSize.y;
						const float gap = std::max(0.0f, wInit.renderWindows[1].iPos.y - splitY);
						ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
						ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
						ImGui::Begin("SplitOverlay", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus);
						auto* drawList = ImGui::GetWindowDrawList();
						if (gap >= 2.0f)
							drawList->AddRectFilled(ImVec2(0.0f, splitY), ImVec2(io.DisplaySize.x, splitY + gap), ImGui::ColorConvertFloat4ToU32(ImVec4(0.05f, 0.06f, 0.08f, 0.85f)));
						else
							drawList->AddLine(ImVec2(0.0f, splitY), ImVec2(io.DisplaySize.x, splitY), ImGui::ColorConvertFloat4ToU32(ImVec4(0.80f, 0.84f, 0.92f, 0.75f)), 2.0f);
						ImGui::End();
					}
					assert(manipulationCounter <= 1u);
				}
				// render selected camera view onto full screen
				else
				{
					info.textureID = 1u + activeRenderWindowIx;

					ImGui::SetNextWindowPos(ImVec2(0, 0));
					ImGui::SetNextWindowSize(io.DisplaySize);
					ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0)); // fully transparent fake window
					ImGui::Begin("FullScreenWindow", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs);
					const ImVec2 contentRegionSize = ImGui::GetContentRegionAvail(), windowPos = ImGui::GetWindowPos(), cursorPos = ImGui::GetCursorScreenPos();
					{
						auto& binding = windowBindings[activeRenderWindowIx];
						auto& planarBound = m_planarProjections[binding.activePlanarIx];
						assert(planarBound);

						binding.aspectRatio = contentRegionSize.x / contentRegionSize.y;
						auto* planarViewCameraBound = planarBound->getCamera();

						assert(planarViewCameraBound);
						assert(binding.boundProjectionIx.has_value());

						auto& projection = planarBound->getPlanarProjections()[binding.boundProjectionIx.value()];
						applyDollyZoomProjection(planarViewCameraBound, projection);
						projection.update(binding.leftHandedProjection, binding.aspectRatio);
					}

					ImGui::Image(info, contentRegionSize);
					ImGuizmo::SetRect(cursorPos.x, cursorPos.y, contentRegionSize.x, contentRegionSize.y);

					ImGui::End();
					ImGui::PopStyleColor(1);
				}
			}

			drawScriptVisualDebugOverlay(io.DisplaySize);
			DrawControlPanel();
			UpdateBoundCameraMovement();
			UpdateCursorVisibility();

			// update camera matrices for scene rendering
			{
				for (uint32_t i = 0u; i < windowBindings.size(); ++i)
				{
					auto& binding = windowBindings[i];

					auto& planarBound = m_planarProjections[binding.activePlanarIx];
					assert(planarBound);
					auto* boundPlanarCamera = planarBound->getCamera();

					assert(binding.boundProjectionIx.has_value());
					auto& projection = planarBound->getPlanarProjections()[binding.boundProjectionIx.value()];
					applyDollyZoomProjection(boundPlanarCamera, projection);
					projection.update(binding.leftHandedProjection, binding.aspectRatio);
					binding.isOrthographicProjection = projection.getParameters().m_type == IPlanarProjection::CProjection::Orthographic;

					auto viewMatrix = getCastedMatrix<float32_t>(boundPlanarCamera->getGimbal().getViewMatrix());
					auto projectionMatrix = getCastedMatrix<float32_t>(projection.getProjectionMatrix());
					auto viewProjMatrix = mul(projectionMatrix, getMatrix3x4As4x4(viewMatrix));

					binding.viewMatrix = viewMatrix;
					binding.projectionMatrix = projectionMatrix;
					binding.viewProjMatrix = viewProjMatrix;
				}
			}



}


