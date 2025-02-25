// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.hpp"

/*
	Renders scene texture to an offline
	framebuffer which color attachment
	is then sampled into a imgui window.

	Written with Nabla, it's UI extension
	and got integrated with ImGuizmo to 
	handle scene's object translations.
*/

class UISampleApp final : public examples::SimpleWindowedApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using clock_t = std::chrono::steady_clock;

	_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_W = 1280, WIN_H = 720;

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

			m_assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(m_system));
			auto* geometry = m_assetManager->getGeometryCreator();

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

			m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
	
			for (auto i = 0u; i < MaxFramesInFlight; i++)
			{
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");
				if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
					return logFail("Couldn't create Command Buffer!");
			}
			
			//pass.scene = CScene::create<CScene::CreateResourcesDirectlyWithDevice>(smart_refctd_ptr(m_utils), smart_refctd_ptr(m_logger), gQueue, geometry);
			pass.scene = CScene::create<CScene::CreateResourcesWithAssetConverter>(smart_refctd_ptr(m_utils), smart_refctd_ptr(m_logger), gQueue, geometry);

			nbl::ext::imgui::UI::SCreationParameters params;

			params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
			params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
			params.assetManager = m_assetManager;
			params.pipelineCache = nullptr;
			params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, TexturesAmount);
			params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
			params.streamingBuffer = nullptr;
			params.subpassIx = 0u;
			params.transfer = getTransferUpQueue();
			params.utilities = m_utils;
			{
				pass.ui.manager = nbl::ext::imgui::UI::create(std::move(params));

				if (!pass.ui.manager)
					return false;

				// note that we use default layout provided by our extension, but you are free to create your own by filling nbl::ext::imgui::UI::S_CREATION_PARAMETERS::resources
				const auto* descriptorSetLayout = pass.ui.manager->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
				const auto& params = pass.ui.manager->getCreationParameters();

				IDescriptorPool::SCreateInfo descriptorPoolInfo = {};
				descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLER)] = (uint32_t)nbl::ext::imgui::UI::DefaultSamplerIx::COUNT;
				descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)] = TexturesAmount;
				descriptorPoolInfo.maxSets = 1u;
				descriptorPoolInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT;

				m_descriptorSetPool = m_device->createDescriptorPool(std::move(descriptorPoolInfo));
				assert(m_descriptorSetPool);

				m_descriptorSetPool->createDescriptorSets(1u, &descriptorSetLayout, &pass.ui.descriptorSet);
				assert(pass.ui.descriptorSet);
			}
			pass.ui.manager->registerListener([this]() -> void
				{
					ImGuiIO& io = ImGui::GetIO();

					camera.setProjectionMatrix([&]() 
					{
						static matrix4SIMD projection;

						if (isPerspective)
							if(isLH)
								projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y, zNear, zFar);
							else
								projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y, zNear, zFar);
						else
						{
							float viewHeight = viewWidth * io.DisplaySize.y / io.DisplaySize.x;

							if(isLH)
								projection = matrix4SIMD::buildProjectionMatrixOrthoLH(viewWidth, viewHeight, zNear, zFar);
							else
								projection = matrix4SIMD::buildProjectionMatrixOrthoRH(viewWidth, viewHeight, zNear, zFar);
						}

						return projection;
					}());

					ImGuizmo::SetOrthographic(false);
					ImGuizmo::BeginFrame();

					ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
					ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

					// create a window and insert the inspector
					ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
					ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
					ImGui::Begin("Editor");

					if (ImGui::RadioButton("Full view", !transformParams.useWindow))
						transformParams.useWindow = false;

					ImGui::SameLine();

					if (ImGui::RadioButton("Window", transformParams.useWindow))
						transformParams.useWindow = true;

					ImGui::Text("Camera");
					bool viewDirty = false;

					if (ImGui::RadioButton("LH", isLH))
						isLH = true;

					ImGui::SameLine();

					if (ImGui::RadioButton("RH", !isLH))
						isLH = false;

					if (ImGui::RadioButton("Perspective", isPerspective))
						isPerspective = true;

					ImGui::SameLine();

					if (ImGui::RadioButton("Orthographic", !isPerspective))
						isPerspective = false;

					ImGui::Checkbox("Enable \"view manipulate\"", &transformParams.enableViewManipulate);
					ImGui::Checkbox("Enable camera movement", &move);
					ImGui::SliderFloat("Move speed", &moveSpeed, 0.1f, 10.f);
					ImGui::SliderFloat("Rotate speed", &rotateSpeed, 0.1f, 10.f);

					// ImGui::Checkbox("Flip Gizmo's Y axis", &flipGizmoY); // let's not expose it to be changed in UI but keep the logic in case

					if (isPerspective)
						ImGui::SliderFloat("Fov", &fov, 20.f, 150.f);
					else
						ImGui::SliderFloat("Ortho width", &viewWidth, 1, 20);

					ImGui::SliderFloat("zNear", &zNear, 0.1f, 100.f);
					ImGui::SliderFloat("zFar", &zFar, 110.f, 10000.f);

					viewDirty |= ImGui::SliderFloat("Distance", &transformParams.camDistance, 1.f, 69.f);

					if (viewDirty || firstFrame)
					{
						core::vectorSIMDf cameraPosition(cosf(camYAngle)* cosf(camXAngle)* transformParams.camDistance, sinf(camXAngle)* transformParams.camDistance, sinf(camYAngle)* cosf(camXAngle)* transformParams.camDistance);
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
						core::matrix4SIMD view, projection, model;
					} imguizmoM16InOut;

					ImGuizmo::SetID(0u);

					imguizmoM16InOut.view = core::transpose(matrix4SIMD(camera.getViewMatrix()));
					imguizmoM16InOut.projection = core::transpose(camera.getProjectionMatrix());
					imguizmoM16InOut.model = core::transpose(core::matrix4SIMD(pass.scene->object.model));
					{
						if (flipGizmoY) // note we allow to flip gizmo just to match our coordinates
							imguizmoM16InOut.projection[1][1] *= -1.f; // https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/	

						transformParams.editTransformDecomposition = true;
						EditTransform(imguizmoM16InOut.view.pointer(), imguizmoM16InOut.projection.pointer(), imguizmoM16InOut.model.pointer(), transformParams);
					}

					// to Nabla + update camera & model matrices
					const auto& view = camera.getViewMatrix();
					const auto& projection = camera.getProjectionMatrix();

					// TODO: make it more nicely
					const_cast<core::matrix3x4SIMD&>(view) = core::transpose(imguizmoM16InOut.view).extractSub3x4(); // a hack, correct way would be to use inverse matrix and get position + target because now it will bring you back to last position & target when switching from gizmo move to manual move (but from manual to gizmo is ok)
					camera.setProjectionMatrix(projection); // update concatanated matrix
					{
						static nbl::core::matrix3x4SIMD modelView, normal;
						static nbl::core::matrix4SIMD modelViewProjection;

						auto& hook = pass.scene->object;
						hook.model = core::transpose(imguizmoM16InOut.model).extractSub3x4();
						{
							const auto& references = pass.scene->getResources().objects;
							const auto type = static_cast<ObjectType>(gcIndex);

							const auto& [gpu, meta] = references[type];
							hook.meta.type = type;
							hook.meta.name = meta.name;
						}

						auto& ubo = hook.viewParameters;

						modelView = nbl::core::concatenateBFollowedByA(view, hook.model);
						modelView.getSub3x3InverseTranspose(normal);
						modelViewProjection = nbl::core::concatenateBFollowedByA(camera.getConcatenatedMatrix(), hook.model);

						memcpy(ubo.MVP, modelViewProjection.pointer(), sizeof(ubo.MVP));
						memcpy(ubo.MV, modelView.pointer(), sizeof(ubo.MV));
						memcpy(ubo.NormalMat, normal.pointer(), sizeof(ubo.NormalMat));

						// object meta display
						{
							ImGui::Begin("Object");
							ImGui::Text("type: \"%s\"", hook.meta.name.data());
							ImGui::End();
						}
					}
					
					// view matrices editor
					{
						ImGui::Begin("Matrices");

						auto addMatrixTable = [&](const char* topText, const char* tableName, const int rows, const int columns, const float* pointer, const bool withSeparator = true)
						{
							ImGui::Text(topText);
							if (ImGui::BeginTable(tableName, columns))
							{
								for (int y = 0; y < rows; ++y)
								{
									ImGui::TableNextRow();
									for (int x = 0; x < columns; ++x)
									{
										ImGui::TableSetColumnIndex(x);
										ImGui::Text("%.3f", *(pointer + (y * columns) + x));
									}
								}
								ImGui::EndTable();
							}

							if (withSeparator)
								ImGui::Separator();
						};

						addMatrixTable("Model Matrix", "ModelMatrixTable", 3, 4, pass.scene->object.model.pointer());
						addMatrixTable("Camera View Matrix", "ViewMatrixTable", 3, 4, view.pointer());
						addMatrixTable("Camera View Projection Matrix", "ViewProjectionMatrixTable", 4, 4, projection.pointer(), false);

						ImGui::End();
					}

					// Nabla Imgui backend MDI buffer info
					// To be 100% accurate and not overly conservative we'd have to explicitly `cull_frees` and defragment each time,
					// so unless you do that, don't use this basic info to optimize the size of your IMGUI buffer.
					{
						auto* streaminingBuffer = pass.ui.manager->getStreamingBuffer();

						const size_t total = streaminingBuffer->get_total_size();			// total memory range size for which allocation can be requested
						const size_t freeSize = streaminingBuffer->getAddressAllocator().get_free_size();		// max total free bloock memory size we can still allocate from total memory available
						const size_t consumedMemory = total - freeSize;			// memory currently consumed by streaming buffer

						float freePercentage = 100.0f * (float)(freeSize) / (float)total;
						float allocatedPercentage = (float)(consumedMemory) / (float)total;

						ImVec2 barSize = ImVec2(400, 30);
						float windowPadding = 10.0f;
						float verticalPadding = ImGui::GetStyle().FramePadding.y;

						ImGui::SetNextWindowSize(ImVec2(barSize.x + 2 * windowPadding, 110 + verticalPadding), ImGuiCond_Always);
						ImGui::Begin("Nabla Imgui MDI Buffer Info", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar);

						ImGui::Text("Total Allocated Size: %zu bytes", total);
						ImGui::Text("In use: %zu bytes", consumedMemory);
						ImGui::Text("Buffer Usage:");

						ImGui::SetCursorPosX(windowPadding);

						if (freePercentage > 70.0f)
							ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.0f, 1.0f, 0.0f, 0.4f));  // Green
						else if (freePercentage > 30.0f)
							ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 1.0f, 0.0f, 0.4f));  // Yellow
						else
							ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 0.0f, 0.0f, 0.4f));  // Red

						ImGui::ProgressBar(allocatedPercentage, barSize, "");

						ImGui::PopStyleColor();

						ImDrawList* drawList = ImGui::GetWindowDrawList();

						ImVec2 progressBarPos = ImGui::GetItemRectMin();
						ImVec2 progressBarSize = ImGui::GetItemRectSize();

						const char* text = "%.2f%% free";
						char textBuffer[64];
						snprintf(textBuffer, sizeof(textBuffer), text, freePercentage);

						ImVec2 textSize = ImGui::CalcTextSize(textBuffer);
						ImVec2 textPos = ImVec2
						(
							progressBarPos.x + (progressBarSize.x - textSize.x) * 0.5f,
							progressBarPos.y + (progressBarSize.y - textSize.y) * 0.5f
						);

						ImVec4 bgColor = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
						drawList->AddRectFilled
						(
							ImVec2(textPos.x - 5, textPos.y - 2),
							ImVec2(textPos.x + textSize.x + 5, textPos.y + textSize.y + 2),
							ImGui::GetColorU32(bgColor)
						);

						ImGui::SetCursorScreenPos(textPos);
						ImGui::Text("%s", textBuffer);

						ImGui::Dummy(ImVec2(0.0f, verticalPadding));

						ImGui::End();
					}

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

		bool updateGUIDescriptorSet()
		{
			// texture atlas + our scene texture, note we don't create info & write pair for the font sampler because UI extension's is immutable and baked into DS layout
			static std::array<IGPUDescriptorSet::SDescriptorInfo, TexturesAmount> descriptorInfo;
			static IGPUDescriptorSet::SWriteDescriptorSet writes[TexturesAmount];

			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].desc = core::smart_refctd_ptr<nbl::video::IGPUImageView>(pass.ui.manager->getFontAtlasView());

			descriptorInfo[OfflineSceneTextureIx].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			descriptorInfo[OfflineSceneTextureIx].desc = pass.scene->getResources().attachments.color;

			for (uint32_t i = 0; i < descriptorInfo.size(); ++i)
			{
				writes[i].dstSet = pass.ui.descriptorSet.get();
				writes[i].binding = 0u;
				writes[i].arrayElement = i;
				writes[i].count = 1u;
			}
			writes[nbl::ext::imgui::UI::FontAtlasTexId].info = descriptorInfo.data() + nbl::ext::imgui::UI::FontAtlasTexId;
			writes[OfflineSceneTextureIx].info = descriptorInfo.data() + OfflineSceneTextureIx;

			return m_device->updateDescriptorSets(writes, {});
		}

		inline void workLoopBody() override
		{
			// framesInFlight: ensuring safe execution of command buffers and acquires, `framesInFlight` only affect semaphore waits, don't use this to index your resources because it can change with swapchain recreation.
			const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
			// We block for semaphores for 2 reasons here:
				// A) Resource: Can't use resource like a command buffer BEFORE previous use is finished! [MaxFramesInFlight]
				// B) Acquire: Can't have more acquires in flight than a certain threshold returned by swapchain or your surface helper class. [MaxAcquiresInFlight]
			if (m_realFrameIx >= framesInFlight)
			{
				const ISemaphore::SWaitInfo cbDonePending[] =
				{
					{
						.semaphore = m_semaphore.get(),
						.value = m_realFrameIx + 1 - framesInFlight
					}
				};
				if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

			const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

			// CPU events
			update();

			// render whole scene to offline frame buffer & submit
			pass.scene->begin();
			{
				pass.scene->update();
				pass.scene->record();
				pass.scene->end();
			}
			pass.scene->submit();

			auto* const cb = m_cmdBufs.data()[resourceIx].get();
			cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cb->beginDebugMarker("UISampleApp IMGUI Frame");

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

			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};

			IQueue::SSubmitInfo::SCommandBufferInfo commandBuffersInfo[] = {{.cmdbuf = cb }};

			// UI render pass
			{
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo = 
				{
					.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
					.colorClearValues = &clear.color,
					.depthStencilClearValues = nullptr,
					.renderArea = currentRenderArea
				};
				nbl::video::ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };

				cb->beginRenderPass(renderpassInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
				const auto uiParams = pass.ui.manager->getCreationParameters();
				auto* pipeline = pass.ui.manager->getPipeline();
				cb->bindGraphicsPipeline(pipeline);
				cb->bindDescriptorSets(EPBP_GRAPHICS, pipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &pass.ui.descriptorSet.get()); // note that we use default UI pipeline layout where uiParams.resources.textures.setIx == uiParams.resources.samplers.setIx
				
				if (!keepRunning())
					return;
				
				if (!pass.ui.manager->render(cb,waitInfo))
				{
					// TODO: need to present acquired image before bailing because its already acquired
					return;
				}
				cb->endRenderPass();
			}
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
								.commandBuffers = commandBuffersInfo,
								.signalSemaphores = rendered
							} 
						};

						const nbl::video::ISemaphore::SWaitInfo waitInfos[] = 
						{ {
							.semaphore = pass.scene->semaphore.progress.get(),
							.value = pass.scene->semaphore.finishedValue
						} };
						
						m_device->blockForSemaphores(waitInfos);

						updateGUIDescriptorSet();

						if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
							m_realFrameIx--;
					}
				}

				m_window->setCaption("[Nabla Engine] UI App Test Demo");
				m_surface->present(m_currentImageAcquire.imageIndex, rendered);
			}
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

		inline void update()
		{
			camera.setMoveSpeed(moveSpeed);
			camera.setRotateSpeed(rotateSpeed);

			static std::chrono::microseconds previousEventTimestamp{};

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

			struct
			{
				std::vector<SMouseEvent> mouse{};
				std::vector<SKeyboardEvent> keyboard{};
			} capturedEvents;

			if (move) camera.beginInputProcessing(nextPresentationTimestamp);
			{
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
				{
					if (move)
						camera.mouseProcess(events); // don't capture the events, only let camera handle them with its impl

					for (const auto& e : events) // here capture
					{
						if (e.timeStamp < previousEventTimestamp)
							continue;

						previousEventTimestamp = e.timeStamp;
						capturedEvents.mouse.emplace_back(e);

						if (e.type == nbl::ui::SMouseEvent::EET_SCROLL)
							gcIndex = std::clamp<uint16_t>(int16_t(gcIndex) + int16_t(core::sign(e.scrollEvent.verticalScroll)), int64_t(0), int64_t(OT_COUNT - (uint8_t)1u));
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

			const auto cursorPosition = m_window->getCursorControl()->getPosition();

			nbl::ext::imgui::UI::SUpdateParameters params = 
			{
				.mousePosition = nbl::hlsl::float32_t2(cursorPosition.x, cursorPosition.y) - nbl::hlsl::float32_t2(m_window->getX(), m_window->getY()),
				.displaySize = { m_window->getWidth(), m_window->getHeight() },
				.mouseEvents = { capturedEvents.mouse.data(), capturedEvents.mouse.size() },
				.keyboardEvents = { capturedEvents.keyboard.data(), capturedEvents.keyboard.size() }
			};

			pass.ui.manager->update(params);
		}

	private:
		// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
		constexpr static inline uint32_t MaxFramesInFlight = 3u;

		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;
		smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;
		smart_refctd_ptr<ISemaphore> m_semaphore;
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		uint64_t m_realFrameIx = 0;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

		smart_refctd_ptr<nbl::asset::IAssetManager> m_assetManager;
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		constexpr static inline auto TexturesAmount = 2u;

		core::smart_refctd_ptr<IDescriptorPool> m_descriptorSetPool;

		struct C_UI
		{
			nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> manager;

			struct
			{
				core::smart_refctd_ptr<video::IGPUSampler> gui, scene;
			} samplers;

			core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
		};

		struct E_APP_PASS
		{
			nbl::core::smart_refctd_ptr<CScene> scene;
			C_UI ui;
		} pass;

		Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
		video::CDumbPresentationOracle oracle;

		uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed

		TransformRequestParams transformParams;
		bool isPerspective = true, isLH = true, flipGizmoY = true, move = false;
		float fov = 60.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
		float viewWidth = 10.f;
		float camYAngle = 165.f / 180.f * 3.14159f;
		float camXAngle = 32.f / 180.f * 3.14159f;

		bool firstFrame = true;
};

NBL_MAIN_FUNC(UISampleApp)