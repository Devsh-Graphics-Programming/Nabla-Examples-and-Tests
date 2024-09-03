// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/this_example/common.hpp"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;


class ComputeShaderPathtracer final : public examples::SimpleWindowedApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using clock_t = std::chrono::steady_clock;

	enum E_LIGHT_GEOMETRY : uint8_t
	{
		ELG_SPHERE,
		ELG_TRIANGLE,
		ELG_RECTANGLE
	};

	_NBL_STATIC_INLINE_CONSTEXPR uint32_t2 WindowDimensions = { 1280, 720 };
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t FramesInFlight = 5;
	_NBL_STATIC_INLINE_CONSTEXPR clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);
	_NBL_STATIC_INLINE_CONSTEXPR E_LIGHT_GEOMETRY LightGeom = E_LIGHT_GEOMETRY::ELG_TRIANGLE;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t DefaultWorkGroupSize = 16u;
	_NBL_STATIC_INLINE std::array<std::string, 3> ShaderPaths = { "../litBySphere.comp", "../litByTriangle.comp", "../litByRectangle.comp" };

	public:
		inline ComputeShaderPathtracer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {
			const auto cameraPos = core::vectorSIMDf(0, 5, -10);
			matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(
				core::radians(fov),
				static_cast<float32_t>(WindowDimensions.x) / static_cast<float32_t>(WindowDimensions.y),
				zNear,
				zFar
			);

			camera = Camera(cameraPos, core::vectorSIMDf(0, 0, 0), proj);
		}

		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			if (!m_surface)
			{
				{
					auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
					IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
					params.width = WindowDimensions.x;
					params.height = WindowDimensions.y;
					params.x = 32;
					params.y = 32;
					params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
					params.windowCaption = "ComputeShaderPathtracer";
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
			// Init systems
			{
				m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

				if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
					return false;

				m_assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(system));
				auto* geometry = m_assetManager->getGeometryCreator();

				m_uiSemaphore = m_device->createSemaphore(m_realFrameIx);
				if (!m_uiSemaphore)
					return logFail("Failed to Create a Semaphore!");
			}

			// Create renderpass and init surface
			nbl::video::IGPURenderpass* renderpass;
			{
				ISwapchain::SCreationParams swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
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
				renderpass = scResources->getRenderpass();

				if (!renderpass)
					return logFail("Failed to create Renderpass!");

				auto gQueue = getGraphicsQueue();
				if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
					return logFail("Could not create Window & Surface or initialize the Surface!");
			}

			// Compute no of frames in flight
			{
				m_maxFramesInFlight = m_surface->getMaxFramesInFlight();
				if (FramesInFlight < m_maxFramesInFlight)
				{
					m_logger->log("Lowering frames in flight!", ILogger::ELL_WARNING);
					m_maxFramesInFlight = FramesInFlight;
				}
			}

			// Create command pool and buffers
			{
				auto gQueue = getGraphicsQueue();
				m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");

				for (auto i = 0u; i < m_maxFramesInFlight; i++)
				{
					if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
						return logFail("Couldn't create Command Buffer!");
				}
			}

			{
				using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
				{
					IGPUSampler::SParams params;
					params.AnisotropicFilter = 1u;
					params.TextureWrapU = ISampler::ETC_REPEAT;
					params.TextureWrapV = ISampler::ETC_REPEAT;
					params.TextureWrapW = ISampler::ETC_REPEAT;

					ui.samplers.gui = m_device->createSampler(params);
					ui.samplers.gui->setObjectDebugName("Nabla IMGUI UI Sampler");
				}

				{
					IGPUSampler::SParams params;
					params.MinLod = 0.f;
					params.MaxLod = 0.f;
					params.TextureWrapU = ISampler::ETC_CLAMP_TO_EDGE;
					params.TextureWrapV = ISampler::ETC_CLAMP_TO_EDGE;
					params.TextureWrapW = ISampler::ETC_CLAMP_TO_EDGE;

					ui.samplers.scene = m_device->createSampler(params);
					ui.samplers.scene->setObjectDebugName("Nabla IMGUI Scene Sampler");
				}

				std::array<core::smart_refctd_ptr<IGPUSampler>, 69u> immutableSamplers;
				for (auto& it : immutableSamplers)
					it = smart_refctd_ptr(ui.samplers.scene);

				immutableSamplers[nbl::ext::imgui::UI::NBL_FONT_ATLAS_TEX_ID] = smart_refctd_ptr(ui.samplers.gui);

				const IGPUDescriptorSetLayout::SBinding bindings[] =
				{
					{
						.binding = 0u,
						.type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
						.createFlags = core::bitflag(binding_flags_t::ECF_UPDATE_AFTER_BIND_BIT) | binding_flags_t::ECF_PARTIALLY_BOUND_BIT | binding_flags_t::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
						.count = 69u
					},
					{
						.binding = 1u,
						.type = IDescriptor::E_TYPE::ET_SAMPLER,
						.createFlags = binding_flags_t::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
						.count = 69u,
						.immutableSamplers = immutableSamplers.data()
					}
				};

				auto descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);

				ui.manager = core::make_smart_refctd_ptr<nbl::ext::imgui::UI>(smart_refctd_ptr(m_device), smart_refctd_ptr(descriptorSetLayout), (int)m_maxFramesInFlight, renderpass, nullptr, smart_refctd_ptr(m_window));

				IDescriptorPool::SCreateInfo descriptorPoolInfo = {};
				descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLER)] = 69u;
				descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)] = 69u;
				descriptorPoolInfo.maxSets = 1u;
				descriptorPoolInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT;

				m_descriptorSetPool = m_device->createDescriptorPool(std::move(descriptorPoolInfo));
				assert(m_descriptorSetPool);

				ui.descriptorSet = m_descriptorSetPool->createDescriptorSet(smart_refctd_ptr(descriptorSetLayout));
				assert(ui.descriptorSet);

			}
			ui.manager->registerListener(
				[this]() -> void {
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
							const auto type = static_cast<E_OBJECT_TYPE>(gcIndex);

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

					ImGui::End();
				}
			);

			m_winMgr->setWindowSize(m_window.get(), WindowDimensions.x, WindowDimensions.y);
			m_surface->recreateSwapchain();
			m_winMgr->show(m_window.get());
			oracle.reportBeginFrameRecord();
			camera.mapKeysToArrows();

			// Create descriptors for the pathtracer
			{
				IGPUDescriptorSetLayout::SBinding descriptorSet0Bindings[] = {
					{
						.binding = 0u,
						.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1u,
						.immutableSamplers = nullptr
					}
				};
				IGPUDescriptorSetLayout::SBinding uboBindings[] = {
					{
						.binding = 0u,
						.type = nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1u,
						.immutableSamplers = nullptr
					}
				};
				IGPUDescriptorSetLayout::SBinding descriptorSet3Bindings[] = {
					{
						.binding = 0u,
						.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1u,
						.immutableSamplers = nullptr
					},
					{
						.binding = 1u,
						.type = nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1u,
						.immutableSamplers = nullptr
					},
					{
						.binding = 2u,
						.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = 1u,
						.immutableSamplers = nullptr
					},
				};

				auto gpuDescriptorSetLayout0 = m_device->createDescriptorSetLayout({ descriptorSet0Bindings, 1 });
				auto gpuDescriptorSetLayout1 = m_device->createDescriptorSetLayout({ uboBindings, 1 });
				auto gpuDescriptorSetLayout2 = m_device->createDescriptorSetLayout({ descriptorSet3Bindings, 3 });

				if (!gpuDescriptorSetLayout0 || !gpuDescriptorSetLayout1 || !gpuDescriptorSetLayout2) {
					return logFail("Failed to create descriptor set layouts!\n");
				}

				auto createGpuResources = [&](std::string pathToShader, smart_refctd_ptr<IGPUComputePipeline> pipeline) -> bool
				{
					IAssetLoader::SAssetLoadParams lp = {};
					lp.logger = m_logger.get();
					lp.workingDirectory = ""; // virtual root
					auto assetBundle = m_assetManager->getAsset(pathToShader, lp);
					const auto assets = assetBundle.getContents();
					if (assets.empty())
					{
						return logFail("Could not load shader!");
					}

					auto source = IAsset::castDown<ICPUShader>(assets[0]);
					// The down-cast should not fail!
					assert(source);

					// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
					auto shader = m_device->createShader(source.get());
					if (!shader)
					{
						return logFail("Shader creationed failed: %s!", pathToShader);
					}

					auto gpuPipelineLayout = m_device->createPipelineLayout({}, core::smart_refctd_ptr(gpuDescriptorSetLayout0), core::smart_refctd_ptr(gpuDescriptorSetLayout1), core::smart_refctd_ptr(gpuDescriptorSetLayout2), nullptr);
					if (!gpuPipelineLayout) {
						return logFail("Failed to create pipeline layout");
					}

					IGPUComputePipeline::SCreationParams params = {};
					params.layout = gpuPipelineLayout.get();
					params.shader.shader = shader.get();
					params.shader.entryPoint = "main";
					params.shader.entries = nullptr;
					params.shader.requireFullSubgroups = true;
					params.shader.requiredSubgroupSize = static_cast<IGPUShader::SSpecInfo::SUBGROUP_SIZE>(5);
					if (!m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline)) {
						return logFail("Failed to create compute pipeline!\n");
					}

					return true;
				};

				if (!createGpuResources(ShaderPaths[LightGeom], m_pipeline)) {
					return logFail("Pipeline creation failed!");
				}
			}

			return true;
		}

		bool updateGUIDescriptorSet()
		{
			// texture atlas + our scene texture, note we don't create info & write pair for the font sampler because UI extension's is immutable and baked into DS layout
			static std::array<IGPUDescriptorSet::SDescriptorInfo, TEXTURES_AMOUNT> descriptorInfo;
			static IGPUDescriptorSet::SWriteDescriptorSet writes[TEXTURES_AMOUNT];

			descriptorInfo[nbl::ext::imgui::UI::NBL_FONT_ATLAS_TEX_ID].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			descriptorInfo[nbl::ext::imgui::UI::NBL_FONT_ATLAS_TEX_ID].desc = ui.manager->getFontAtlasView();

			descriptorInfo[CScene::NBL_OFFLINE_SCENE_TEX_ID].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

			descriptorInfo[CScene::NBL_OFFLINE_SCENE_TEX_ID].desc = pass.scene->getResources().attachments.color;

			for (uint32_t i = 0; i < descriptorInfo.size(); ++i)
			{
				writes[i].dstSet = pass.ui.descriptorSet.get();
				writes[i].binding = 0u;
				writes[i].arrayElement = i;
				writes[i].count = 1u;
			}
			writes[nbl::ext::imgui::UI::NBL_FONT_ATLAS_TEX_ID].info = descriptorInfo.data() + nbl::ext::imgui::UI::NBL_FONT_ATLAS_TEX_ID;
			writes[CScene::NBL_OFFLINE_SCENE_TEX_ID].info = descriptorInfo.data() + CScene::NBL_OFFLINE_SCENE_TEX_ID;

			return m_device->updateDescriptorSets(writes, {});
		}

		inline void workLoopBody() override
		{
			const auto resourceIx = m_realFrameIx % m_maxFramesInFlight;

			if (m_realFrameIx >= m_maxFramesInFlight)
			{
				const ISemaphore::SWaitInfo cbDonePending[] = 
				{
					{
						.semaphore = m_uiSemaphore.get(),
						.value = m_realFrameIx + 1 - m_maxFramesInFlight
					}
				};
				if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

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
			cb->beginDebugMarker("ComputeShaderPathtracer IMGUI Frame");

			auto* queue = getGraphicsQueue();

			asset::SViewport viewport;
			{
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = WindowDimensions.x;
				viewport.height = WindowDimensions.y;
			}
			cb->setViewport(0u, 1u, &viewport);

			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};

			// UI render pass
			{
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				const IGPUCommandBuffer::SRenderpassBeginInfo info = 
				{
					.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
					.colorClearValues = &clear.color,
					.depthStencilClearValues = nullptr,
					.renderArea = currentRenderArea
				};
				cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
				pass.ui.manager->render(cb, pass.ui.descriptorSet.get(), resourceIx);
				cb->endRenderPass();
			}
			cb->end();
			{
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] = 
				{ 
					{
						.semaphore = m_uiSemaphore.get(),
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

			// TODO: Use real deltaTime instead
			static float deltaTimeInSec = 0.1f;

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
							gcIndex = std::clamp<uint16_t>(int16_t(gcIndex) + int16_t(core::sign(e.scrollEvent.verticalScroll)), int64_t(0), int64_t(EOT_COUNT - (uint8_t)1u));
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

			ui.manager->update(deltaTimeInSec, { mousePosition.x , mousePosition.y }, mouseEvents, keyboardEvents);
		}

	private:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;

		// gpu resources
		smart_refctd_ptr<ISemaphore> m_uiSemaphore;
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		smart_refctd_ptr<IGPUComputePipeline> m_pipeline;
		uint64_t m_realFrameIx : 59 = 0;
		uint64_t m_maxFramesInFlight : 5;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
		NBL_CONSTEXPR_STATIC_INLINE auto TEXTURES_AMOUNT = 2u;

		core::smart_refctd_ptr<IDescriptorPool> m_descriptorSetPool;

		// system resources
		smart_refctd_ptr<nbl::asset::IAssetManager> m_assetManager;
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		struct C_UI
		{
			nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> manager;

			struct
			{
				core::smart_refctd_ptr<video::IGPUSampler> gui, scene;
			} samplers;

			core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
		} ui;

		Camera camera;
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

NBL_MAIN_FUNC(ComputeShaderPathtracer)

#if 0
smart_refctd_ptr<IGPUImageView> createHDRImageView(nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> device, asset::E_FORMAT colorFormat, uint32_t width, uint32_t height)
{
	smart_refctd_ptr<IGPUImageView> gpuImageViewColorBuffer;
	{
		IGPUImage::SCreationParams imgInfo;
		imgInfo.format = colorFormat;
		imgInfo.type = IGPUImage::ET_2D;
		imgInfo.extent.width = width;
		imgInfo.extent.height = height;
		imgInfo.extent.depth = 1u;
		imgInfo.mipLevels = 1u;
		imgInfo.arrayLayers = 1u;
		imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
		imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
		imgInfo.usage = core::bitflag(asset::IImage::EUF_STORAGE_BIT) | asset::IImage::EUF_TRANSFER_SRC_BIT;

		auto image = device->createImage(std::move(imgInfo));
		auto imageMemReqs = image->getMemoryReqs();
		imageMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		device->allocate(imageMemReqs, image.get());

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.image = std::move(image);
		imgViewInfo.format = colorFormat;
		imgViewInfo.viewType = IGPUImageView::ET_2D;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		imgViewInfo.subresourceRange.baseArrayLayer = 0u;
		imgViewInfo.subresourceRange.baseMipLevel = 0u;
		imgViewInfo.subresourceRange.layerCount = 1u;
		imgViewInfo.subresourceRange.levelCount = 1u;

		gpuImageViewColorBuffer = device->createImageView(std::move(imgViewInfo));
	}

	return gpuImageViewColorBuffer;
}

struct ShaderParameters
{
	const uint32_t MaxDepthLog2 = 4; //5
	const uint32_t MaxSamplesLog2 = 10; //18
} kShaderParameters;






int main()
{
	auto createImageView = [&](std::string pathToOpenEXRHDRIImage)
		{
#ifndef _NBL_COMPILE_WITH_OPENEXR_LOADER_
			assert(false);
#endif

			auto pathToTexture = pathToOpenEXRHDRIImage;
			IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
			auto cpuTexture = assetManager->getAsset(pathToTexture, lp);
			auto cpuTextureContents = cpuTexture.getContents();
			assert(!cpuTextureContents.empty());
			auto cpuImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(*cpuTextureContents.begin());
			cpuImage->setImageUsageFlags(IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT);

			ICPUImageView::SCreationParams viewParams;
			viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
			viewParams.image = cpuImage;
			viewParams.format = viewParams.image->getCreationParameters().format;
			viewParams.viewType = IImageView<ICPUImage>::ET_2D;
			viewParams.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = 1u;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = 1u;

			auto cpuImageView = ICPUImageView::create(std::move(viewParams));

			cpu2gpuParams.beginCommandBuffers();
			auto gpuImageView = CPU2GPU.getGPUObjectsFromAssets(&cpuImageView, &cpuImageView + 1u, cpu2gpuParams)->front();
			cpu2gpuParams.waitForCreationToComplete(false);

			return gpuImageView;
		};

	auto gpuEnvmapImageView = createImageView("../../media/envmap/envmap_0.exr");

	smart_refctd_ptr<IGPUBufferView> gpuSequenceBufferView;
	{
		const uint32_t MaxDimensions = 3u << kShaderParameters.MaxDepthLog2;
		const uint32_t MaxSamples = 1u << kShaderParameters.MaxSamplesLog2;

		auto sampleSequence = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t) * MaxDimensions * MaxSamples);

		core::OwenSampler sampler(MaxDimensions, 0xdeadbeefu);
		//core::SobolSampler sampler(MaxDimensions);

		auto out = reinterpret_cast<uint32_t*>(sampleSequence->getPointer());
		for (auto dim = 0u; dim < MaxDimensions; dim++)
			for (uint32_t i = 0; i < MaxSamples; i++)
			{
				out[i * MaxDimensions + dim] = sampler.sample(dim, i);
			}

		// TODO: Temp Fix because createFilledDeviceLocalBufferOnDedMem doesn't take in params
		// auto gpuSequenceBuffer = utilities->createFilledDeviceLocalBufferOnDedMem(graphicsQueue, sampleSequence->getSize(), sampleSequence->getPointer());
		core::smart_refctd_ptr<IGPUBuffer> gpuSequenceBuffer;
		{
			IGPUBuffer::SCreationParams params = {};
			const size_t size = sampleSequence->getSize();
			params.usage = core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT) | asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT;
			params.size = size;
			gpuSequenceBuffer = device->createBuffer(std::move(params));
			auto gpuSequenceBufferMemReqs = gpuSequenceBuffer->getMemoryReqs();
			gpuSequenceBufferMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			device->allocate(gpuSequenceBufferMemReqs, gpuSequenceBuffer.get());
			utilities->updateBufferRangeViaStagingBufferAutoSubmit(asset::SBufferRange<IGPUBuffer>{0u, size, gpuSequenceBuffer}, sampleSequence->getPointer(), graphicsQueue);
		}
		gpuSequenceBufferView = device->createBufferView(gpuSequenceBuffer.get(), asset::EF_R32G32B32_UINT);
	}

	smart_refctd_ptr<IGPUImageView> gpuScrambleImageView;
	{
		IGPUImage::SCreationParams imgParams;
		imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
		imgParams.type = IImage::ET_2D;
		imgParams.format = EF_R32G32_UINT;
		imgParams.extent = { WIN_W, WIN_H,1u };
		imgParams.mipLevels = 1u;
		imgParams.arrayLayers = 1u;
		imgParams.samples = IImage::ESCF_1_BIT;
		imgParams.usage = core::bitflag(IImage::EUF_SAMPLED_BIT) | IImage::EUF_TRANSFER_DST_BIT;
		imgParams.initialLayout = asset::IImage::EL_UNDEFINED;

		IGPUImage::SBufferCopy region = {};
		region.bufferOffset = 0u;
		region.bufferRowLength = 0u;
		region.bufferImageHeight = 0u;
		region.imageExtent = imgParams.extent;
		region.imageOffset = { 0u,0u,0u };
		region.imageSubresource.layerCount = 1u;
		region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;

		constexpr auto ScrambleStateChannels = 2u;
		const auto renderPixelCount = imgParams.extent.width * imgParams.extent.height;
		core::vector<uint32_t> random(renderPixelCount * ScrambleStateChannels);
		{
			core::RandomSampler rng(0xbadc0ffeu);
			for (auto& pixel : random)
				pixel = rng.nextSample();
		}

		// TODO: Temp Fix because createFilledDeviceLocalBufferOnDedMem doesn't take in params
		// auto buffer = utilities->createFilledDeviceLocalBufferOnDedMem(graphicsQueue, random.size()*sizeof(uint32_t), random.data());
		core::smart_refctd_ptr<IGPUBuffer> buffer;
		{
			IGPUBuffer::SCreationParams params = {};
			const size_t size = random.size() * sizeof(uint32_t);
			params.usage = core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT) | asset::IBuffer::EUF_TRANSFER_SRC_BIT;
			params.size = size;
			buffer = device->createBuffer(std::move(params));
			auto bufferMemReqs = buffer->getMemoryReqs();
			bufferMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			device->allocate(bufferMemReqs, buffer.get());
			utilities->updateBufferRangeViaStagingBufferAutoSubmit(asset::SBufferRange<IGPUBuffer>{0u, size, buffer}, random.data(), graphicsQueue);
		}

		IGPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		// TODO: Replace this IGPUBuffer -> IGPUImage to using image upload utility
		viewParams.image = utilities->createFilledDeviceLocalImageOnDedMem(std::move(imgParams), buffer.get(), 1u, &region, graphicsQueue);
		viewParams.viewType = IGPUImageView::ET_2D;
		viewParams.format = EF_R32G32_UINT;
		viewParams.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		viewParams.subresourceRange.levelCount = 1u;
		viewParams.subresourceRange.layerCount = 1u;
		gpuScrambleImageView = device->createImageView(std::move(viewParams));
	}

	// Create Out Image TODO
	constexpr uint32_t MAX_FBO_COUNT = 4u;
	smart_refctd_ptr<IGPUImageView> outHDRImageViews[MAX_FBO_COUNT] = {};
	assert(MAX_FBO_COUNT >= swapchain->getImageCount());
	for (uint32_t i = 0; i < swapchain->getImageCount(); ++i) {
		outHDRImageViews[i] = createHDRImageView(device, asset::EF_R16G16B16A16_SFLOAT, WIN_W, WIN_H);
	}

	core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSets0[FBO_COUNT] = {};
	for (uint32_t i = 0; i < FBO_COUNT; ++i)
	{
		auto& descSet = descriptorSets0[i];
		descSet = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout0));
		video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSet;
		writeDescriptorSet.dstSet = descSet.get();
		writeDescriptorSet.binding = 0;
		writeDescriptorSet.count = 1u;
		writeDescriptorSet.arrayElement = 0u;
		writeDescriptorSet.descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = outHDRImageViews[i];
			info.info.image.sampler = nullptr;
			info.info.image.imageLayout = asset::IImage::EL_GENERAL;
		}
		writeDescriptorSet.info = &info;
		device->updateDescriptorSets(1u, &writeDescriptorSet, 0u, nullptr);
	}

	struct SBasicViewParametersAligned
	{
		SBasicViewParameters uboData;
	};

	IGPUBuffer::SCreationParams gpuuboParams = {};
	gpuuboParams.usage = core::bitflag(IGPUBuffer::EUF_UNIFORM_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT;
	gpuuboParams.size = sizeof(SBasicViewParametersAligned);
	auto gpuubo = device->createBuffer(std::move(gpuuboParams));
	auto gpuuboMemReqs = gpuubo->getMemoryReqs();
	gpuuboMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
	device->allocate(gpuuboMemReqs, gpuubo.get());

	auto uboDescriptorSet1 = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout1));
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet uboWriteDescriptorSet;
		uboWriteDescriptorSet.dstSet = uboDescriptorSet1.get();
		uboWriteDescriptorSet.binding = 0;
		uboWriteDescriptorSet.count = 1u;
		uboWriteDescriptorSet.arrayElement = 0u;
		uboWriteDescriptorSet.descriptorType = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = gpuubo;
			info.info.buffer.offset = 0ull;
			info.info.buffer.size = sizeof(SBasicViewParametersAligned);
		}
		uboWriteDescriptorSet.info = &info;
		device->updateDescriptorSets(1u, &uboWriteDescriptorSet, 0u, nullptr);
	}

	ISampler::SParams samplerParams0 = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
	auto sampler0 = device->createSampler(samplerParams0);
	ISampler::SParams samplerParams1 = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
	auto sampler1 = device->createSampler(samplerParams1);

	auto descriptorSet2 = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout2));
	{
		constexpr auto kDescriptorCount = 3;
		IGPUDescriptorSet::SWriteDescriptorSet samplerWriteDescriptorSet[kDescriptorCount];
		IGPUDescriptorSet::SDescriptorInfo samplerDescriptorInfo[kDescriptorCount];
		for (auto i = 0; i < kDescriptorCount; i++)
		{
			samplerWriteDescriptorSet[i].dstSet = descriptorSet2.get();
			samplerWriteDescriptorSet[i].binding = i;
			samplerWriteDescriptorSet[i].arrayElement = 0u;
			samplerWriteDescriptorSet[i].count = 1u;
			samplerWriteDescriptorSet[i].info = samplerDescriptorInfo + i;
		}
		samplerWriteDescriptorSet[0].descriptorType = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER;
		samplerWriteDescriptorSet[1].descriptorType = nbl::asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER;
		samplerWriteDescriptorSet[2].descriptorType = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER;

		samplerDescriptorInfo[0].desc = gpuEnvmapImageView;
		{
			// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
			samplerDescriptorInfo[0].info.image.sampler = sampler0;
			samplerDescriptorInfo[0].info.image.imageLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
		}
		samplerDescriptorInfo[1].desc = gpuSequenceBufferView;
		samplerDescriptorInfo[2].desc = gpuScrambleImageView;
		{
			// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
			samplerDescriptorInfo[2].info.image.sampler = sampler1;
			samplerDescriptorInfo[2].info.image.imageLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
		}

		device->updateDescriptorSets(kDescriptorCount, samplerWriteDescriptorSet, 0u, nullptr);
	}

	constexpr uint32_t FRAME_COUNT = 500000u;

	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
	{
		imageAcquire[i] = device->createSemaphore();
		renderFinished[i] = device->createSemaphore();
	}

	CDumbPresentationOracle oracle;
	oracle.reportBeginFrameRecord();
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	// polling for events!
	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	uint32_t resourceIx = 0;
	while (windowCb->isWindowOpen())
	{
		resourceIx++;
		if (resourceIx >= FRAMES_IN_FLIGHT) {
			resourceIx = 0;
		}

		oracle.reportEndFrameRecord();
		double dt = oracle.getDeltaTimeInMicroSeconds() / 1000.0;
		auto nextPresentationTimeStamp = oracle.getNextPresentationTimeStamp();
		oracle.reportBeginFrameRecord();

		// Input 
		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		cam.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { cam.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { cam.keyboardProcess(events); }, logger.get());
		cam.endInputProcessing(nextPresentationTimeStamp);

		auto& cb = cmdbuf[resourceIx];
		auto& fence = frameComplete[resourceIx];
		if (fence)
			while (device->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT)
			{
			} else
				fence = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

			const auto viewMatrix = cam.getViewMatrix();
			const auto viewProjectionMatrix = matrix4SIMD::concatenateBFollowedByAPrecisely(
				video::ISurface::getSurfaceTransformationMatrix(swapchain->getPreTransform()),
				cam.getConcatenatedMatrix()
			);

			// safe to proceed
			cb->begin(IGPUCommandBuffer::EU_NONE);
			cb->resetQueryPool(timestampQueryPool.get(), 0u, 2u);

			// renderpass 
			uint32_t imgnum = 0u;
			swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &imgnum);
			{
				auto mv = viewMatrix;
				auto mvp = viewProjectionMatrix;
				core::matrix3x4SIMD normalMat;
				mv.getSub3x3InverseTranspose(normalMat);

				SBasicViewParametersAligned viewParams;
				memcpy(viewParams.uboData.MV, mv.pointer(), sizeof(mv));
				memcpy(viewParams.uboData.MVP, mvp.pointer(), sizeof(mvp));
				memcpy(viewParams.uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));

				asset::SBufferRange<video::IGPUBuffer> range;
				range.buffer = gpuubo;
				range.offset = 0ull;
				range.size = sizeof(viewParams);
				utilities->updateBufferRangeViaStagingBufferAutoSubmit(range, &viewParams, graphicsQueue);
			}

			// TRANSITION outHDRImageViews[imgnum] to EIL_GENERAL (because of descriptorSets0 -> ComputeShader Writes into the image)
			{
				IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[3u] = {};
				imageBarriers[0].barrier.srcAccessMask = asset::EAF_NONE;
				imageBarriers[0].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_WRITE_BIT);
				imageBarriers[0].oldLayout = asset::IImage::EL_UNDEFINED;
				imageBarriers[0].newLayout = asset::IImage::EL_GENERAL;
				imageBarriers[0].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[0].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[0].image = outHDRImageViews[imgnum]->getCreationParameters().image;
				imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				imageBarriers[0].subresourceRange.baseMipLevel = 0u;
				imageBarriers[0].subresourceRange.levelCount = 1;
				imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
				imageBarriers[0].subresourceRange.layerCount = 1;

				imageBarriers[1].barrier.srcAccessMask = asset::EAF_NONE;
				imageBarriers[1].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT);
				imageBarriers[1].oldLayout = asset::IImage::EL_UNDEFINED;
				imageBarriers[1].newLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
				imageBarriers[1].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[1].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[1].image = gpuScrambleImageView->getCreationParameters().image;
				imageBarriers[1].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				imageBarriers[1].subresourceRange.baseMipLevel = 0u;
				imageBarriers[1].subresourceRange.levelCount = 1;
				imageBarriers[1].subresourceRange.baseArrayLayer = 0u;
				imageBarriers[1].subresourceRange.layerCount = 1;

				imageBarriers[2].barrier.srcAccessMask = asset::EAF_NONE;
				imageBarriers[2].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT);
				imageBarriers[2].oldLayout = asset::IImage::EL_UNDEFINED;
				imageBarriers[2].newLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
				imageBarriers[2].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[2].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[2].image = gpuEnvmapImageView->getCreationParameters().image;
				imageBarriers[2].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				imageBarriers[2].subresourceRange.baseMipLevel = 0u;
				imageBarriers[2].subresourceRange.levelCount = gpuEnvmapImageView->getCreationParameters().subresourceRange.levelCount;
				imageBarriers[2].subresourceRange.baseArrayLayer = 0u;
				imageBarriers[2].subresourceRange.layerCount = gpuEnvmapImageView->getCreationParameters().subresourceRange.layerCount;

				cb->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 3u, imageBarriers);
			}

			// cube envmap handle
			{
				cb->writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS::EPSF_TOP_OF_PIPE_BIT, timestampQueryPool.get(), 0u);
				cb->bindComputePipeline(gpuComputePipeline.get());
				cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 0u, 1u, &descriptorSets0[imgnum].get());
				cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 1u, 1u, &uboDescriptorSet1.get());
				cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 2u, 1u, &descriptorSet2.get());
				cb->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);
				cb->writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS::EPSF_BOTTOM_OF_PIPE_BIT, timestampQueryPool.get(), 1u);
			}
			// TODO: tone mapping and stuff

			// Copy HDR Image to SwapChain
			auto srcImgViewCreationParams = outHDRImageViews[imgnum]->getCreationParameters();
			auto dstImgViewCreationParams = fbo->begin()[imgnum]->getCreationParameters().attachments[0]->getCreationParameters();

			// Getting Ready for Blit
			// TRANSITION outHDRImageViews[imgnum] to EIL_TRANSFER_SRC_OPTIMAL
			// TRANSITION `fbo[imgnum]->getCreationParameters().attachments[0]` to EIL_TRANSFER_DST_OPTIMAL
			{
				IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[2u] = {};
				imageBarriers[0].barrier.srcAccessMask = asset::EAF_NONE;
				imageBarriers[0].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
				imageBarriers[0].oldLayout = asset::IImage::EL_UNDEFINED;
				imageBarriers[0].newLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
				imageBarriers[0].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[0].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[0].image = srcImgViewCreationParams.image;
				imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				imageBarriers[0].subresourceRange.baseMipLevel = 0u;
				imageBarriers[0].subresourceRange.levelCount = 1;
				imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
				imageBarriers[0].subresourceRange.layerCount = 1;

				imageBarriers[1].barrier.srcAccessMask = asset::EAF_NONE;
				imageBarriers[1].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
				imageBarriers[1].oldLayout = asset::IImage::EL_UNDEFINED;
				imageBarriers[1].newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
				imageBarriers[1].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[1].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[1].image = dstImgViewCreationParams.image;
				imageBarriers[1].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				imageBarriers[1].subresourceRange.baseMipLevel = 0u;
				imageBarriers[1].subresourceRange.levelCount = 1;
				imageBarriers[1].subresourceRange.baseArrayLayer = 0u;
				imageBarriers[1].subresourceRange.layerCount = 1;
				cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 2u, imageBarriers);
			}

			// Blit Image
			{
				SImageBlit blit = {};
				blit.srcOffsets[0] = { 0, 0, 0 };
				blit.srcOffsets[1] = { WIN_W, WIN_H, 1 };

				blit.srcSubresource.aspectMask = srcImgViewCreationParams.subresourceRange.aspectMask;
				blit.srcSubresource.mipLevel = srcImgViewCreationParams.subresourceRange.baseMipLevel;
				blit.srcSubresource.baseArrayLayer = srcImgViewCreationParams.subresourceRange.baseArrayLayer;
				blit.srcSubresource.layerCount = srcImgViewCreationParams.subresourceRange.layerCount;
				blit.dstOffsets[0] = { 0, 0, 0 };
				blit.dstOffsets[1] = { WIN_W, WIN_H, 1 };
				blit.dstSubresource.aspectMask = dstImgViewCreationParams.subresourceRange.aspectMask;
				blit.dstSubresource.mipLevel = dstImgViewCreationParams.subresourceRange.baseMipLevel;
				blit.dstSubresource.baseArrayLayer = dstImgViewCreationParams.subresourceRange.baseArrayLayer;
				blit.dstSubresource.layerCount = dstImgViewCreationParams.subresourceRange.layerCount;

				auto srcImg = srcImgViewCreationParams.image;
				auto dstImg = dstImgViewCreationParams.image;

				cb->blitImage(srcImg.get(), asset::IImage::EL_TRANSFER_SRC_OPTIMAL, dstImg.get(), asset::IImage::EL_TRANSFER_DST_OPTIMAL, 1u, &blit, ISampler::ETF_NEAREST);
			}

			// TRANSITION `fbo[imgnum]->getCreationParameters().attachments[0]` to EIL_PRESENT
			{
				IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
				imageBarriers[0].barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
				imageBarriers[0].barrier.dstAccessMask = asset::EAF_NONE;
				imageBarriers[0].oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
				imageBarriers[0].newLayout = asset::IImage::EL_PRESENT_SRC;
				imageBarriers[0].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[0].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
				imageBarriers[0].image = dstImgViewCreationParams.image;
				imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				imageBarriers[0].subresourceRange.baseMipLevel = 0u;
				imageBarriers[0].subresourceRange.levelCount = 1;
				imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
				imageBarriers[0].subresourceRange.layerCount = 1;
				cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TOP_OF_PIPE_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);
			}

			cb->end();
			device->resetFences(1, &fence.get());
			CommonAPI::Submit(device.get(), cb.get(), graphicsQueue, imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
			CommonAPI::Present(device.get(), swapchain.get(), graphicsQueue, renderFinished[resourceIx].get(), imgnum);

			if (LOG_TIMESTAMP)
			{
				std::array<uint64_t, 4> timestamps{};
				auto queryResultFlags = core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS>(video::IQueryPool::EQRF_WAIT_BIT) | video::IQueryPool::EQRF_WITH_AVAILABILITY_BIT | video::IQueryPool::EQRF_64_BIT;
				device->getQueryPoolResults(timestampQueryPool.get(), 0u, 2u, sizeof(timestamps), timestamps.data(), sizeof(uint64_t) * 2ull, queryResultFlags);
				const float timePassed = (timestamps[2] - timestamps[0]) * device->getPhysicalDevice()->getLimits().timestampPeriodInNanoSeconds;
				logger->log("Time Passed (Seconds) = %f", system::ILogger::ELL_INFO, (timePassed * 1e-9));
				logger->log("Timestamps availablity: %d, %d", system::ILogger::ELL_INFO, timestamps[1], timestamps[3]);
			}
	}

	const auto& fboCreationParams = fbo->begin()[0]->getCreationParameters();
	auto gpuSourceImageView = fboCreationParams.attachments[0];

	device->waitIdle();

	// bool status = ext::ScreenShot::createScreenShot(device.get(), queues[decltype(initOutput)::EQT_TRANSFER_UP], renderFinished[0].get(), gpuSourceImageView.get(), assetManager.get(), "ScreenShot.png");
	// assert(status);

	return 0;
}

#endif