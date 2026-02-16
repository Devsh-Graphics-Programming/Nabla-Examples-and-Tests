// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
#include "nbl/this_example/transform.hpp"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/builtin/hlsl/surface_transform.h"
#include "nbl/this_example/common.hpp"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"
#include "nbl/builtin/hlsl/matrix_utils/transformation_matrix_utils.hlsl"
#include "app_resources/hlsl/render_common.hlsl"
#include "app_resources/hlsl/render_rwmc_common.hlsl"
#include "app_resources/hlsl/resolve_common.hlsl"
#include "app_resources/hlsl/rwmc_global_settings_common.hlsl"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;
using namespace nbl::examples;

// TODO: Add a QueryPool for timestamping once its ready
// TODO: Do buffer creation using assConv
class HLSLComputePathtracer final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
		using device_base_t = SimpleWindowedApplication;
		using asset_base_t = BuiltinResourcesApplication;
		using clock_t = std::chrono::steady_clock;

		enum E_LIGHT_GEOMETRY : uint8_t
		{
			ELG_SPHERE,
			ELG_TRIANGLE,
			ELG_RECTANGLE,
			ELG_COUNT
		};

		enum E_RENDER_MODE : uint8_t
		{
			ERM_GLSL,
			ERM_HLSL,
			// ERM_CHECKERED,
			ERM_COUNT
		};

		constexpr static inline uint32_t2 WindowDimensions = { 1280, 720 };
		constexpr static inline uint32_t MaxFramesInFlight = 5;
		constexpr static inline uint32_t MaxDescriptorCount = 256u;
		constexpr static inline uint8_t MaxUITextureCount = 1u;
		static inline std::string DefaultImagePathsFile = "envmap/envmap_0.exr";
		static inline std::string OwenSamplerFilePath = "owen_sampler_buffer.bin";
		static inline std::array<std::string, E_LIGHT_GEOMETRY::ELG_COUNT> PTGLSLShaderPaths = {
		    "app_resources/glsl/litBySphere.comp",
		    "app_resources/glsl/litByTriangle.comp",
		    "app_resources/glsl/litByRectangle.comp"
		};
		static inline std::string PTHLSLShaderPath = "app_resources/hlsl/render.comp.hlsl";
		static inline std::array<std::string, E_LIGHT_GEOMETRY::ELG_COUNT> PTHLSLShaderVariants = {
		    "SPHERE_LIGHT",
		    "TRIANGLE_LIGHT",
		    "RECTANGLE_LIGHT"
		};
		static inline std::string ResolveShaderPath = "app_resources/hlsl/resolve.comp.hlsl";
		static inline std::string PresentShaderPath = "app_resources/hlsl/present.frag.hlsl";

		const char* shaderNames[E_LIGHT_GEOMETRY::ELG_COUNT] = {
			"ELG_SPHERE",
			"ELG_TRIANGLE",
			"ELG_RECTANGLE"
		};

		const char* shaderTypes[E_RENDER_MODE::ERM_COUNT] = {
			"ERM_GLSL",
			"ERM_HLSL"
		};

	public:
		inline HLSLComputePathtracer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		inline bool isComputeOnly() const override { return false; }

		inline video::SPhysicalDeviceLimits getRequiredDeviceLimits() const override
		{
			video::SPhysicalDeviceLimits retval = device_base_t::getRequiredDeviceLimits();
			retval.storagePushConstant16 = true;
			return retval;
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

				// Remember to call the base class initialization!
				if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
					return false;
				if (!asset_base_t::onAppInitialized(std::move(system)))
					return false;

				m_semaphore = m_device->createSemaphore(m_realFrameIx);

				if (!m_semaphore)
					return logFail("Failed to create semaphore!");
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

			// Create command pool and buffers
			{
				auto gQueue = getGraphicsQueue();
				m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
				if (!m_cmdPool)
					return logFail("Couldn't create Command Pool!");

				if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data(), MaxFramesInFlight }))
					return logFail("Couldn't create Command Buffer!");
			}

			ISampler::SParams samplerParams = {
				.AnisotropicFilter = 0
			};
			auto defaultSampler = m_device->createSampler(samplerParams);

			// Create descriptors and pipeline for the pathtracer
			{
				auto convertDSLayoutCPU2GPU = [&](smart_refctd_ptr<ICPUDescriptorSetLayout> cpuLayout) {
					auto converter = CAssetConverter::create({ .device = m_device.get() });
					CAssetConverter::SInputs inputs = {};
					inputs.readCache = converter.get();
					inputs.logger = m_logger.get();
					CAssetConverter::SConvertParams params = {};
					params.utilities = m_utils.get();

					std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSetLayout>>(inputs.assets) = { &cpuLayout.get(),1 };
					// don't need to assert that we don't need to provide patches since layouts are not patchable
					//assert(true);
					auto reservation = converter->reserve(inputs);
					// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
					auto gpuLayout = reservation.getGPUObjects<ICPUDescriptorSetLayout>().front().value;
					if (!gpuLayout) {
						m_logger->log("Failed to convert %s into an IGPUDescriptorSetLayout handle", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					return gpuLayout;
					};
				auto convertDSCPU2GPU = [&](smart_refctd_ptr<ICPUDescriptorSet> cpuDS) {
					auto converter = CAssetConverter::create({ .device = m_device.get() });
					CAssetConverter::SInputs inputs = {};
					inputs.readCache = converter.get();
					inputs.logger = m_logger.get();
					CAssetConverter::SConvertParams params = {};
					params.utilities = m_utils.get();

					std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSet>>(inputs.assets) = { &cpuDS.get(), 1 };
					// don't need to assert that we don't need to provide patches since layouts are not patchable
					//assert(true);
					auto reservation = converter->reserve(inputs);
					// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
					auto gpuDS = reservation.getGPUObjects<ICPUDescriptorSet>().front().value;
					if (!gpuDS) {
						m_logger->log("Failed to convert %s into an IGPUDescriptorSet handle", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					return gpuDS;
					};

				std::array<ICPUDescriptorSetLayout::SBinding, 2> descriptorSet0Bindings = {};
				std::array<ICPUDescriptorSetLayout::SBinding, 2> descriptorSet3Bindings = {};
				std::array<IGPUDescriptorSetLayout::SBinding, 1> presentDescriptorSetBindings;

				descriptorSet0Bindings[0] = {
					.binding = 0u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};

				descriptorSet0Bindings[1] = {
					.binding = 1u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};

				descriptorSet3Bindings[0] = {
					.binding = 0u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};
				descriptorSet3Bindings[1] = {
					.binding = 2u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1u,
					.immutableSamplers = nullptr
				};

				presentDescriptorSetBindings[0] = {
					.binding = 0u,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
					.immutableSamplers = &defaultSampler
				};

				auto cpuDescriptorSetLayout0 = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(descriptorSet0Bindings);
				auto cpuDescriptorSetLayout2 = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(descriptorSet3Bindings);

				auto gpuDescriptorSetLayout0 = convertDSLayoutCPU2GPU(cpuDescriptorSetLayout0);
				auto gpuDescriptorSetLayout2 = convertDSLayoutCPU2GPU(cpuDescriptorSetLayout2);
				auto gpuPresentDescriptorSetLayout = m_device->createDescriptorSetLayout(presentDescriptorSetBindings);

				auto cpuDescriptorSet0 = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(cpuDescriptorSetLayout0));
				auto cpuDescriptorSet2 = make_smart_refctd_ptr<ICPUDescriptorSet>(std::move(cpuDescriptorSetLayout2));

				m_descriptorSet0 = convertDSCPU2GPU(cpuDescriptorSet0);
				m_descriptorSet2 = convertDSCPU2GPU(cpuDescriptorSet2);

				smart_refctd_ptr<IDescriptorPool> presentDSPool;
				{
					const video::IGPUDescriptorSetLayout* const layouts[] = { gpuPresentDescriptorSetLayout.get() };
					const uint32_t setCounts[] = { 1u };
					presentDSPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
				}
				m_presentDescriptorSet = presentDSPool->createDescriptorSet(gpuPresentDescriptorSetLayout);

				// Create Shaders
				auto loadAndCompileGLSLShader = [&](const std::string& pathToShader, bool persistentWorkGroups = false) -> smart_refctd_ptr<IShader>
				{
					IAssetLoader::SAssetLoadParams lp = {};
					lp.workingDirectory = localInputCWD;
					auto assetBundle = m_assetMgr->getAsset(pathToShader, lp);
					const auto assets = assetBundle.getContents();
					if (assets.empty())
					{
						m_logger->log("Could not load shader: ", ILogger::ELL_ERROR, pathToShader);
						std::exit(-1);
					}

					auto source = smart_refctd_ptr_static_cast<IShader>(assets[0]);
					// The down-cast should not fail!
					assert(source);

					auto compiler = make_smart_refctd_ptr<asset::CGLSLCompiler>(smart_refctd_ptr(m_system));
					CGLSLCompiler::SOptions options = {};
					options.stage = IShader::E_SHADER_STAGE::ESS_COMPUTE;	// should be compute
					options.preprocessorOptions.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
					options.spirvOptimizer = nullptr;
#ifndef _NBL_DEBUG
					ISPIRVOptimizer::E_OPTIMIZER_PASS optPasses = ISPIRVOptimizer::EOP_STRIP_DEBUG_INFO;
					auto opt = make_smart_refctd_ptr<ISPIRVOptimizer>(std::span<ISPIRVOptimizer::E_OPTIMIZER_PASS>(&optPasses, 1));
					options.spirvOptimizer = opt.get();
#endif
					options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
					options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
					options.preprocessorOptions.logger = m_logger.get();
					options.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder();

					const IShaderCompiler::SMacroDefinition persistentDefine = { "PERSISTENT_WORKGROUPS", "1" };
					if (persistentWorkGroups)
						options.preprocessorOptions.extraDefines = { &persistentDefine, &persistentDefine + 1 };

					source = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);

					// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
					auto shader = m_device->compileShader({ source.get(), nullptr, nullptr, nullptr });
					if (!shader)
					{
						m_logger->log("GLSL shader creationed failed: %s!", ILogger::ELL_ERROR, pathToShader);
						std::exit(-1);
					}

					return shader;
				};

				auto loadAndCompileHLSLShader = [&](const std::string& pathToShader, const std::string& defineMacro = "", bool persistentWorkGroups = false, bool rwmc = false) -> smart_refctd_ptr<IShader>
				{
					IAssetLoader::SAssetLoadParams lp = {};
					lp.workingDirectory = localInputCWD;
					auto assetBundle = m_assetMgr->getAsset(pathToShader, lp);
					const auto assets = assetBundle.getContents();
					if (assets.empty())
					{
						m_logger->log("Could not load shader: ", ILogger::ELL_ERROR, pathToShader);
						std::exit(-1);
					}

					auto source = smart_refctd_ptr_static_cast<IShader>(assets[0]);
					// The down-cast should not fail!
					assert(source);

					auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(m_system));
					CHLSLCompiler::SOptions options = {};
					options.stage = IShader::E_SHADER_STAGE::ESS_COMPUTE;
					options.preprocessorOptions.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
					options.spirvOptimizer = nullptr;
#ifndef _NBL_DEBUG
					ISPIRVOptimizer::E_OPTIMIZER_PASS optPasses = ISPIRVOptimizer::EOP_STRIP_DEBUG_INFO;
					auto opt = make_smart_refctd_ptr<ISPIRVOptimizer>(std::span<ISPIRVOptimizer::E_OPTIMIZER_PASS>(&optPasses, 1));
					options.spirvOptimizer = opt.get();
#endif
					options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
					options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
					options.preprocessorOptions.logger = m_logger.get();
					options.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder();
					
					core::vector<IShaderCompiler::SMacroDefinition> defines;
					defines.reserve(3);
					if (!defineMacro.empty())
						defines.push_back({ defineMacro, "" });
					if(persistentWorkGroups)
						defines.push_back({ "PERSISTENT_WORKGROUPS", "1" });
					if(rwmc)
						defines.push_back({ "RWMC_ENABLED", "" });

					options.preprocessorOptions.extraDefines = defines;

					source = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);
					
					auto shader = m_device->compileShader({ source.get(), nullptr, nullptr, nullptr });
					if (!shader)
					{
						m_logger->log("HLSL shader creationed failed: %s!", ILogger::ELL_ERROR, pathToShader);
						std::exit(-1);
					}

					return shader;
				};

				const auto deviceMinSubgroupSize = m_device->getPhysicalDevice()->getLimits().minSubgroupSize;
				auto getComputePipelineCreationParams = [deviceMinSubgroupSize](IShader* shader, IGPUPipelineLayout* pipelineLayout) -> IGPUComputePipeline::SCreationParams
				{
					IGPUComputePipeline::SCreationParams params = {};
					params.layout = pipelineLayout;
					params.shader.shader = shader;
					params.shader.entryPoint = "main";
					params.shader.entries = nullptr;
					params.cached.requireFullSubgroups = true;
					params.shader.requiredSubgroupSize = static_cast<IPipelineBase::SUBGROUP_SIZE>(5);

					return params;
				};

				// Create compute pipelines
				{
					for (int index = 0; index < E_LIGHT_GEOMETRY::ELG_COUNT; index++)
					{
						const nbl::asset::SPushConstantRange pcRange = {
							.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
							.offset = 0,
							.size = sizeof(RenderPushConstants)
						};
						auto ptPipelineLayout = m_device->createPipelineLayout(
							{ &pcRange, 1 },
							core::smart_refctd_ptr(gpuDescriptorSetLayout0),
							nullptr,
							core::smart_refctd_ptr(gpuDescriptorSetLayout2),
							nullptr
						);
						if (!ptPipelineLayout)
							return logFail("Failed to create Pathtracing pipeline layout");

						const nbl::asset::SPushConstantRange rwmcPcRange = {
							.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
							.offset = 0,
							.size = sizeof(RenderRWMCPushConstants)
						};
						auto rwmcPtPipelineLayout = m_device->createPipelineLayout(
							{ &rwmcPcRange, 1 },
							core::smart_refctd_ptr(gpuDescriptorSetLayout0),
							nullptr,
							core::smart_refctd_ptr(gpuDescriptorSetLayout2),
							nullptr
						);
						if (!rwmcPtPipelineLayout)
							return logFail("Failed to create RWMC Pathtracing pipeline layout");

						{
							auto ptShader = loadAndCompileHLSLShader(PTHLSLShaderPath, PTHLSLShaderVariants[index]);
							auto params = getComputePipelineCreationParams(ptShader.get(), ptPipelineLayout.get());
							
							if (!m_device->createComputePipelines(nullptr, { &params, 1 }, m_PTHLSLPipelines.data() + index))
								return logFail("Failed to create HLSL compute pipeline!\n");
						}
						{
							auto ptShader = loadAndCompileHLSLShader(PTHLSLShaderPath, PTHLSLShaderVariants[index], true);
							auto params = getComputePipelineCreationParams(ptShader.get(), ptPipelineLayout.get());
							
							if (!m_device->createComputePipelines(nullptr, { &params, 1 }, m_PTHLSLPersistentWGPipelines.data() + index))
								return logFail("Failed to create HLSL PersistentWG compute pipeline!\n");
						}

						// rwmc pipelines
						{
							auto ptShader = loadAndCompileHLSLShader(PTHLSLShaderPath, PTHLSLShaderVariants[index], false, true);
							auto params = getComputePipelineCreationParams(ptShader.get(), rwmcPtPipelineLayout.get());

							if (!m_device->createComputePipelines(nullptr, { &params, 1 }, m_PTHLSLPipelinesRWMC.data() + index))
								return logFail("Failed to create HLSL RWMC compute pipeline!\n");
						}
						{
							auto ptShader = loadAndCompileHLSLShader(PTHLSLShaderPath, PTHLSLShaderVariants[index], true, true);
							auto params = getComputePipelineCreationParams(ptShader.get(), rwmcPtPipelineLayout.get());

							if (!m_device->createComputePipelines(nullptr, { &params, 1 }, m_PTHLSLPersistentWGPipelinesRWMC.data() + index))
								return logFail("Failed to create HLSL RWMC PersistentWG compute pipeline!\n");
						}
					}
				}

				// Create resolve pipelines
				{
					const nbl::asset::SPushConstantRange pcRange = {
							.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
							.offset = 0,
							.size = sizeof(ResolvePushConstants)
					};

					auto pipelineLayout = m_device->createPipelineLayout(
						{ &pcRange, 1 },
						core::smart_refctd_ptr(gpuDescriptorSetLayout0)
					);

					if (!pipelineLayout) {
						return logFail("Failed to create resolve pipeline layout");
					}

					{
						auto shader = loadAndCompileHLSLShader(ResolveShaderPath);
						auto params = getComputePipelineCreationParams(shader.get(), pipelineLayout.get());

						if (!m_device->createComputePipelines(nullptr, { &params, 1 }, &m_resolvePipeline))
							return logFail("Failed to create HLSL resolve compute pipeline!\n");
					}
				}

				// Create graphics pipeline
				{
					auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
					ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
					if (!fsTriProtoPPln)
						return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

					// Load Fragment Shader
					auto fragmentShader = loadAndCompileHLSLShader(PresentShaderPath);
					if (!fragmentShader)
						return logFail("Failed to Load and Compile Fragment Shader: lumaMeterShader!");

					const IGPUPipelineBase::SShaderSpecInfo fragSpec = {
						.shader = fragmentShader.get(),
					    .entryPoint = "main"
					};

					auto presentLayout = m_device->createPipelineLayout(
						{},
						core::smart_refctd_ptr(gpuPresentDescriptorSetLayout),
						nullptr,
						nullptr,
						nullptr
					);
					m_presentPipeline = fsTriProtoPPln.createPipeline(fragSpec, presentLayout.get(), scRes->getRenderpass());
					if (!m_presentPipeline)
						return logFail("Could not create Graphics Pipeline!");

				}
			}

			// load CPUImages and convert to GPUImages
			smart_refctd_ptr<IGPUImage> envMap, scrambleMap;
			{
				auto convertImgCPU2GPU = [&](std::span<ICPUImage*> cpuImgs) {
					auto queue = getGraphicsQueue();
					auto cmdbuf = m_cmdBufs[0].get();
					cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
					std::array<IQueue::SSubmitInfo::SCommandBufferInfo, 1> commandBufferInfo = { cmdbuf };
					core::smart_refctd_ptr<ISemaphore> imgFillSemaphore = m_device->createSemaphore(0);
					imgFillSemaphore->setObjectDebugName("Image Fill Semaphore");

					auto converter = CAssetConverter::create({ .device = m_device.get() });
					// We don't want to generate mip-maps for these images, to ensure that we must override the default callbacks.
					struct SInputs final : CAssetConverter::SInputs
					{
						// we also need to override this to have concurrent sharing
						inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUImage* buffer, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
						{
							if (familyIndices.size() > 1)
								return familyIndices;
							return {};
						}

						inline uint8_t getMipLevelCount(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
						{
							return image->getCreationParameters().mipLevels;
						}
						inline uint16_t needToRecomputeMips(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
						{
							return 0b0u;
						}

						std::vector<uint32_t> familyIndices;
					} inputs = {};
					inputs.readCache = converter.get();
					inputs.logger = m_logger.get();
					{
						const core::set<uint32_t> uniqueFamilyIndices = { queue->getFamilyIndex(), queue->getFamilyIndex() };
						inputs.familyIndices = { uniqueFamilyIndices.begin(),uniqueFamilyIndices.end() };
					}
					// scratch command buffers for asset converter transfer commands
					SIntendedSubmitInfo transfer = {
						.queue = queue,
						.waitSemaphores = {},
						.prevCommandBuffers = {},
						.scratchCommandBuffers = commandBufferInfo,
						.scratchSemaphore = {
							.semaphore = imgFillSemaphore.get(),
							.value = 0,
							// because of layout transitions
							.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
						}
					};
					// as per the `SIntendedSubmitInfo` one commandbuffer must be begun
					cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
					// Normally we'd have to inherit and override the `getFinalOwnerQueueFamily` callback to ensure that the
					// compute queue becomes the owner of the buffers and images post-transfer, but in this example we use concurrent sharing
					CAssetConverter::SConvertParams params = {};
					params.transfer = &transfer;
					params.utilities = m_utils.get();

					std::get<CAssetConverter::SInputs::asset_span_t<ICPUImage>>(inputs.assets) = cpuImgs;
					// assert that we don't need to provide patches
					assert(cpuImgs[0]->getImageUsageFlags().hasFlags(ICPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT));
					auto reservation = converter->reserve(inputs);
					// the `.value` is just a funny way to make the `smart_refctd_ptr` copyable
					auto gpuImgs = reservation.getGPUObjects<ICPUImage>();
					for (auto& gpuImg : gpuImgs) {
						if (!gpuImg) {
							m_logger->log("Failed to convert %s into an IGPUImage handle", ILogger::ELL_ERROR, DefaultImagePathsFile);
							std::exit(-1);
						}
					}

					// and launch the conversions
					m_api->startCapture();
					auto result = reservation.convert(params);
					m_api->endCapture();
					if (!result.blocking() && result.copy() != IQueue::RESULT::SUCCESS) {
						m_logger->log("Failed to record or submit conversions", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					envMap = gpuImgs[0].value;
					scrambleMap = gpuImgs[1].value;
				};

				smart_refctd_ptr<ICPUImage> envMapCPU, scrambleMapCPU;
				{
					IAssetLoader::SAssetLoadParams lp;
					lp.workingDirectory = this->sharedInputCWD;
					SAssetBundle bundle = m_assetMgr->getAsset(DefaultImagePathsFile, lp);
					if (bundle.getContents().empty()) {
						m_logger->log("Couldn't load an asset.", ILogger::ELL_ERROR);
						std::exit(-1);
					}

					envMapCPU = IAsset::castDown<ICPUImage>(bundle.getContents()[0]);
					if (!envMapCPU) {
						m_logger->log("Couldn't load an asset.", ILogger::ELL_ERROR);
						std::exit(-1);
					}
				};
				{
					asset::ICPUImage::SCreationParams info;
					info.format = asset::E_FORMAT::EF_R32G32_UINT;
					info.type = asset::ICPUImage::ET_2D;
					auto extent = envMapCPU->getCreationParameters().extent;
					info.extent.width = extent.width;
					info.extent.height = extent.height;
					info.extent.depth = 1u;
					info.mipLevels = 1u;
					info.arrayLayers = 1u;
					info.samples = asset::ICPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
					info.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
					info.usage = asset::IImage::EUF_TRANSFER_SRC_BIT | asset::IImage::EUF_SAMPLED_BIT;

					scrambleMapCPU = ICPUImage::create(std::move(info));
					const uint32_t texelFormatByteSize = getTexelOrBlockBytesize(scrambleMapCPU->getCreationParameters().format);
					const uint32_t texelBufferSize = scrambleMapCPU->getImageDataSizeInBytes();
					auto texelBuffer = ICPUBuffer::create({ texelBufferSize });

					core::RandomSampler rng(0xbadc0ffeu);
					auto out = reinterpret_cast<uint32_t*>(texelBuffer->getPointer());
					for (auto index = 0u; index < texelBufferSize / 4; index++) {
						out[index] = rng.nextSample();
					}

					auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
					ICPUImage::SBufferCopy& region = regions->front();
					region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
					region.imageSubresource.mipLevel = 0u;
					region.imageSubresource.baseArrayLayer = 0u;
					region.imageSubresource.layerCount = 1u;
					region.bufferOffset = 0u;
					region.bufferRowLength = IImageAssetHandlerBase::calcPitchInBlocks(extent.width, texelFormatByteSize);
					region.bufferImageHeight = 0u;
					region.imageOffset = { 0u, 0u, 0u };
					region.imageExtent = scrambleMapCPU->getCreationParameters().extent;

					scrambleMapCPU->setBufferAndRegions(std::move(texelBuffer), regions);

					// programmatically user-created IPreHashed need to have their hash computed (loaders do it while loading)
					scrambleMapCPU->setContentHash(scrambleMapCPU->computeContentHash());
				}

				std::array<ICPUImage*, 2> cpuImgs = { envMapCPU.get(), scrambleMapCPU.get() };
				convertImgCPU2GPU(cpuImgs);
			}

			// create views for textures
			{
				auto createHDRIImage = [this](const asset::E_FORMAT colorFormat, const uint32_t width, const uint32_t height, const bool useCascadeCreationParameters = false) -> smart_refctd_ptr<IGPUImage> {
					IGPUImage::SCreationParams imgInfo;
					imgInfo.format = colorFormat;
					imgInfo.type = IGPUImage::ET_2D;
					imgInfo.extent.width = width;
					imgInfo.extent.height = height;
					imgInfo.extent.depth = 1u;
					imgInfo.mipLevels = 1u;
					imgInfo.samples = IGPUImage::ESCF_1_BIT;
					imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

					if (!useCascadeCreationParameters)
					{
						imgInfo.arrayLayers = 1u;
						imgInfo.usage = asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT | asset::IImage::EUF_SAMPLED_BIT;
					}
					else
					{
						imgInfo.arrayLayers = CascadeCount;
						imgInfo.usage = asset::IImage::EUF_STORAGE_BIT;
					}

					auto image = m_device->createImage(std::move(imgInfo));
					auto imageMemReqs = image->getMemoryReqs();
					imageMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
					m_device->allocate(imageMemReqs, image.get());

					return image;
				};
				auto createHDRIImageView = [this](smart_refctd_ptr<IGPUImage> img, const uint32_t imageArraySize = 1u, const IGPUImageView::E_TYPE imageViewType = IGPUImageView::ET_2D) -> smart_refctd_ptr<IGPUImageView>
				{
					auto format = img->getCreationParameters().format;
					IGPUImageView::SCreationParams imgViewInfo;
					imgViewInfo.image = std::move(img);
					imgViewInfo.format = format;
					imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
					imgViewInfo.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
					imgViewInfo.subresourceRange.baseArrayLayer = 0u;
					imgViewInfo.subresourceRange.baseMipLevel = 0u;
					imgViewInfo.subresourceRange.levelCount = 1u;
					imgViewInfo.viewType = imageViewType;

					imgViewInfo.subresourceRange.layerCount = imageArraySize;

					return m_device->createImageView(std::move(imgViewInfo));
				};

				auto params = envMap->getCreationParameters();
				auto extent = params.extent;

				envMap->setObjectDebugName("Env Map");
				m_envMapView = createHDRIImageView(envMap);
				m_envMapView->setObjectDebugName("Env Map View"); 

				scrambleMap->setObjectDebugName("Scramble Map");
				m_scrambleView = createHDRIImageView(scrambleMap);
				m_scrambleView->setObjectDebugName("Scramble Map View");

				auto outImg = createHDRIImage(asset::E_FORMAT::EF_R16G16B16A16_SFLOAT, WindowDimensions.x, WindowDimensions.y);
				outImg->setObjectDebugName("Output Image");
				m_outImgView = createHDRIImageView(outImg, 1, IGPUImageView::ET_2D_ARRAY);
				m_outImgView->setObjectDebugName("Output Image View");

				auto cascade = createHDRIImage(asset::E_FORMAT::EF_R16G16B16A16_SFLOAT, WindowDimensions.x, WindowDimensions.y, true);
				cascade->setObjectDebugName("Cascade");
				m_cascadeView = createHDRIImageView(cascade, CascadeCount, IGPUImageView::ET_2D_ARRAY);
				m_cascadeView->setObjectDebugName("Cascade View");

				// TODO: change cascade layout to general
			}

			// create sequence buffer view
			{
				// TODO: do this better use asset manager to get the ICPUBuffer from `.bin`
				auto createBufferFromCacheFile = [this](
					system::path filename,
					size_t bufferSize,
					void *data,
					smart_refctd_ptr<ICPUBuffer>& buffer
				) -> std::pair<smart_refctd_ptr<IFile>, bool>
				{
					ISystem::future_t<smart_refctd_ptr<nbl::system::IFile>> owenSamplerFileFuture;
					ISystem::future_t<size_t> owenSamplerFileReadFuture;
					size_t owenSamplerFileBytesRead;

					m_system->createFile(owenSamplerFileFuture, localOutputCWD / filename, IFile::ECF_READ);
					smart_refctd_ptr<IFile> owenSamplerFile;

					if (owenSamplerFileFuture.wait())
					{
						owenSamplerFileFuture.acquire().move_into(owenSamplerFile);
						if (!owenSamplerFile)
							return { nullptr, false };

						owenSamplerFile->read(owenSamplerFileReadFuture, data, 0, bufferSize);
						if (owenSamplerFileReadFuture.wait())
						{
							owenSamplerFileReadFuture.acquire().move_into(owenSamplerFileBytesRead);

							if (owenSamplerFileBytesRead < bufferSize)
							{
								buffer = asset::ICPUBuffer::create({ sizeof(uint32_t) * bufferSize });
								return { owenSamplerFile, false };
							}

							buffer = asset::ICPUBuffer::create({ { sizeof(uint32_t) * bufferSize }, data });
						}
					}

					return { owenSamplerFile, true };
				};
				auto writeBufferIntoCacheFile = [this](smart_refctd_ptr<IFile> file, size_t bufferSize, void* data)
				{
					ISystem::future_t<size_t> owenSamplerFileWriteFuture;
					size_t owenSamplerFileBytesWritten;

					file->write(owenSamplerFileWriteFuture, data, 0, bufferSize);
					if (owenSamplerFileWriteFuture.wait())
						owenSamplerFileWriteFuture.acquire().move_into(owenSamplerFileBytesWritten);
				};

				constexpr uint32_t quantizedDimensions = MaxBufferDimensions / 3u;
				constexpr size_t bufferSize = quantizedDimensions * MaxBufferSamples;
				std::array<QuantizedSequence, bufferSize> data = {};
				smart_refctd_ptr<ICPUBuffer> sampleSeq;

				auto cacheBufferResult = createBufferFromCacheFile(sharedOutputCWD/OwenSamplerFilePath, bufferSize, data.data(), sampleSeq);
				if (!cacheBufferResult.second)
				{
					core::OwenSampler sampler(MaxBufferDimensions, 0xdeadbeefu);

					ICPUBuffer::SCreationParams params = {};
					params.size = quantizedDimensions * MaxBufferSamples * sizeof(QuantizedSequence);
					sampleSeq = ICPUBuffer::create(std::move(params));

					auto out = reinterpret_cast<QuantizedSequence*>(sampleSeq->getPointer());
					for (auto dim = 0u; dim < quantizedDimensions; dim++)
					    for (uint32_t i = 0; i < MaxBufferSamples; i++)
					    {
						    auto& seq = out[i * quantizedDimensions + dim];
						    seq.setX(sampler.sample(dim * 3 + 0, i));
						    seq.setY(sampler.sample(dim * 3 + 1, i));
						    seq.setZ(sampler.sample(dim * 3 + 2, i));
					    }
					if (cacheBufferResult.first)
						writeBufferIntoCacheFile(cacheBufferResult.first, bufferSize, out);
				}

				IGPUBuffer::SCreationParams params = {};
				params.usage = asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
				params.size = bufferSize;

				// we don't want to overcomplicate the example with multi-queue
				m_utils->createFilledDeviceLocalBufferOnDedMem(
					SIntendedSubmitInfo{ .queue = getGraphicsQueue() },
					std::move(params),
					sampleSeq->getPointer()
				).move_into(m_sequenceBuffer);
			}

			// Update Descriptors
			{
				ISampler::SParams samplerParams0 = {
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
					ISampler::ETBC_FLOAT_OPAQUE_BLACK,
					ISampler::ETF_LINEAR,
					ISampler::ETF_LINEAR,
					ISampler::ESMM_LINEAR,
					0u,
					false,
					ECO_ALWAYS
				};
				auto sampler0 = m_device->createSampler(samplerParams0);
				ISampler::SParams samplerParams1 = {
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
					ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE,
					ISampler::ETBC_INT_OPAQUE_BLACK,
					ISampler::ETF_NEAREST,
					ISampler::ETF_NEAREST,
					ISampler::ESMM_NEAREST,
					0u,
					false,
					ECO_ALWAYS
				};
				auto sampler1 = m_device->createSampler(samplerParams1);

				std::array<IGPUDescriptorSet::SDescriptorInfo, 5> writeDSInfos = {};
				writeDSInfos[0].desc = m_outImgView;
				writeDSInfos[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
				writeDSInfos[1].desc = m_cascadeView;
				writeDSInfos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
				writeDSInfos[2].desc = m_envMapView;
				// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
				writeDSInfos[2].info.combinedImageSampler.sampler = sampler0;
				writeDSInfos[2].info.combinedImageSampler.imageLayout = asset::IImage::LAYOUT::READ_ONLY_OPTIMAL;
				writeDSInfos[3].desc = m_scrambleView;
				// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
				writeDSInfos[3].info.combinedImageSampler.sampler = sampler1;
				writeDSInfos[3].info.combinedImageSampler.imageLayout = asset::IImage::LAYOUT::READ_ONLY_OPTIMAL;
				writeDSInfos[4].desc = m_outImgView;
				writeDSInfos[4].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

				std::array<IGPUDescriptorSet::SWriteDescriptorSet, 5> writeDescriptorSets = {};
				writeDescriptorSets[0] = {
					.dstSet = m_descriptorSet0.get(),
					.binding = 0,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[0]
				};
				writeDescriptorSets[1] = {
					.dstSet = m_descriptorSet0.get(),
					.binding = 1,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[1]
				};
				writeDescriptorSets[2] = {
					.dstSet = m_descriptorSet2.get(),
					.binding = 0,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[2]
				};
				writeDescriptorSets[3] = {
					.dstSet = m_descriptorSet2.get(),
					.binding = 2,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[3]
				};
				writeDescriptorSets[4] = {
					.dstSet = m_presentDescriptorSet.get(),
					.binding = 0,
					.arrayElement = 0u,
					.count = 1u,
					.info = &writeDSInfos[4]
				};

				m_device->updateDescriptorSets(writeDescriptorSets, {});
			}

			// Create ui descriptors
			{
				using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
				{
					IGPUSampler::SParams params;
					params.AnisotropicFilter = 1u;
					params.TextureWrapU = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
					params.TextureWrapV = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
					params.TextureWrapW = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;

					m_ui.samplers.gui = m_device->createSampler(params);
					m_ui.samplers.gui->setObjectDebugName("Nabla IMGUI UI Sampler");
				}

				std::array<core::smart_refctd_ptr<IGPUSampler>, 69u> immutableSamplers;
				for (auto& it : immutableSamplers)
					it = smart_refctd_ptr(m_ui.samplers.scene);

				immutableSamplers[nbl::ext::imgui::UI::FontAtlasTexId] = smart_refctd_ptr(m_ui.samplers.gui);

				nbl::ext::imgui::UI::SCreationParameters params;

				params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
				params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
				params.assetManager = m_assetMgr;
				params.pipelineCache = nullptr;
				params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, MaxUITextureCount);
				params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
				params.streamingBuffer = nullptr;
				params.subpassIx = 0u;
				params.transfer = getTransferUpQueue();
				params.utilities = m_utils;
				{
					m_ui.manager = ext::imgui::UI::create(std::move(params));

					// note that we use default layout provided by our extension, but you are free to create your own by filling nbl::ext::imgui::UI::S_CREATION_PARAMETERS::resources
					const auto* descriptorSetLayout = m_ui.manager->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
					const auto& params = m_ui.manager->getCreationParameters();

					IDescriptorPool::SCreateInfo descriptorPoolInfo = {};
					descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLER)] = (uint32_t)nbl::ext::imgui::UI::DefaultSamplerIx::COUNT;
					descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)] = MaxUITextureCount;
					descriptorPoolInfo.maxSets = 1u;
					descriptorPoolInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT;

					m_guiDescriptorSetPool = m_device->createDescriptorPool(std::move(descriptorPoolInfo));
					assert(m_guiDescriptorSetPool);

					m_guiDescriptorSetPool->createDescriptorSets(1u, &descriptorSetLayout, &m_ui.descriptorSet);
					assert(m_ui.descriptorSet);
				}
			}
			m_ui.manager->registerListener(
				[this]() -> void {
					ImGuiIO& io = ImGui::GetIO();
					ImGuizmo::SetOrthographic(false);
					ImGuizmo::BeginFrame();

					m_camera.setProjectionMatrix([&]()
					{
						static matrix4SIMD projection;

						projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(fov), io.DisplaySize.x / io.DisplaySize.y, zNear, zFar);

						return projection;
					}());

					ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
					ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

					// create a window and insert the inspector
					ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
					ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);
					ImGui::Begin("Controls");

					ImGui::SameLine();

					ImGui::Text("Camera");

					ImGui::Text("Press Home to reset camera.");
					ImGui::Text("Press End to reset light.");

					ImGui::SliderFloat("Move speed", &moveSpeed, 0.1f, 10.f);
					ImGui::SliderFloat("Rotate speed", &rotateSpeed, 0.1f, 10.f);
					ImGui::SliderFloat("Fov", &fov, 20.f, 150.f);
					ImGui::SliderFloat("zNear", &zNear, 0.1f, 100.f);
					ImGui::SliderFloat("zFar", &zFar, 110.f, 10000.f);
					ImGui::Combo("Shader", &PTPipeline, shaderNames, E_LIGHT_GEOMETRY::ELG_COUNT);
					ImGui::Combo("Render Mode", &renderMode, shaderTypes, E_RENDER_MODE::ERM_COUNT);
					ImGui::SliderInt("SPP", &spp, 1, MaxBufferSamples);
					ImGui::SliderInt("Depth", &depth, 1, MaxBufferDimensions / 3);
					ImGui::Checkbox("Persistent WorkGroups", &usePersistentWorkGroups);

					ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);

					ImGui::Text("\nRWMC settings:");
					ImGui::Checkbox("Enable RWMC", &useRWMC);
					ImGui::SliderFloat("start", &rwmcStart, 1.0f, 32.0f);
					ImGui::SliderFloat("base", &rwmcBase, 1.0f, 32.0f);
					ImGui::SliderFloat("minReliableLuma", &rwmcMinReliableLuma, 0.1f, 1024.0f);
					ImGui::SliderFloat("kappa", &rwmcKappa, 0.1f, 1024.0f);

					ImGui::End();
				}
			);

			m_ui.manager->registerListener(
				[this]() -> void {
					static struct
					{
						hlsl::float32_t4x4 view, projection;
					} imguizmoM16InOut;

					ImGuizmo::SetID(0u);

					// TODO: camera will return hlsl::float32_tMxN 
					auto view = *reinterpret_cast<const float32_t3x4*>(m_camera.getViewMatrix().pointer());
					imguizmoM16InOut.view = hlsl::transpose(getMatrix3x4As4x4(view));

					// TODO: camera will return hlsl::float32_tMxN 
					imguizmoM16InOut.projection = hlsl::transpose(*reinterpret_cast<const float32_t4x4*>(m_camera.getProjectionMatrix().pointer()));
					imguizmoM16InOut.projection[1][1] *= -1.f; // https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/	

					m_transformParams.editTransformDecomposition = true;
					m_transformParams.sceneTexDescIx = 1u;

					if (ImGui::IsKeyPressed(ImGuiKey_End))
					{
						m_lightModelMatrix = hlsl::float32_t4x4(
							0.3f, 0.0f, 0.0f, 0.0f,
							0.0f, 0.3f, 0.0f, 0.0f,
							0.0f, 0.0f, 0.3f, 0.0f,
							-1.0f, 1.5f, 0.0f, 1.0f
						);
					}

					if (E_LIGHT_GEOMETRY::ELG_SPHERE == PTPipeline)
					{
						m_transformParams.allowedOp = ImGuizmo::OPERATION::TRANSLATE | ImGuizmo::OPERATION::SCALEU;
						m_transformParams.isSphere = true;
					}
					else
					{
						m_transformParams.allowedOp = ImGuizmo::OPERATION::TRANSLATE | ImGuizmo::OPERATION::ROTATE | ImGuizmo::OPERATION::SCALE;
						m_transformParams.isSphere = false;
					}
					EditTransform(&imguizmoM16InOut.view[0][0], &imguizmoM16InOut.projection[0][0], &m_lightModelMatrix[0][0], m_transformParams);

					if (E_LIGHT_GEOMETRY::ELG_SPHERE == PTPipeline)
					{
						// keep uniform scale for sphere
						float32_t uniformScale = (m_lightModelMatrix[0][0] + m_lightModelMatrix[1][1] + m_lightModelMatrix[2][2]) / 3.0f;
						m_lightModelMatrix[0][0] = uniformScale;
						m_lightModelMatrix[1][1] = uniformScale; // Doesn't affect sphere but will affect rectangle/triangle if switching shapes
						m_lightModelMatrix[2][2] = uniformScale;
					}

				}
			);

			// Set Camera
			{
				core::vectorSIMDf cameraPosition(0, 5, -10);
				matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(
					core::radians(60.0f),
					WindowDimensions.x / WindowDimensions.y,
					0.01f,
					500.0f
				);
				m_camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), proj);
			}

			m_winMgr->setWindowSize(m_window.get(), WindowDimensions.x, WindowDimensions.y);
			m_surface->recreateSwapchain();
			m_winMgr->show(m_window.get());
			m_oracle.reportBeginFrameRecord();
			m_camera.mapKeysToArrows();

			// set initial rwmc settings
			
			rwmcStart = hlsl::dot<float32_t3>(hlsl::transpose(colorspace::scRGBtoXYZ)[1], LightEminence);
			rwmcBase = 8.0f;
			rwmcMinReliableLuma = 1.0f;
			rwmcKappa = 5.0f;
			return true;
		}

		bool updateGUIDescriptorSet()
		{
			// texture atlas, note we don't create info & write pair for the font sampler because UI extension's is immutable and baked into DS layout
			static std::array<IGPUDescriptorSet::SDescriptorInfo, MaxUITextureCount> descriptorInfo;
			static IGPUDescriptorSet::SWriteDescriptorSet writes[MaxUITextureCount];

			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
			descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].desc = smart_refctd_ptr<IGPUImageView>(m_ui.manager->getFontAtlasView());

			for (uint32_t i = 0; i < descriptorInfo.size(); ++i)
			{
				writes[i].dstSet = m_ui.descriptorSet.get();
				writes[i].binding = 0u;
				writes[i].arrayElement = i;
				writes[i].count = 1u;
			}
			writes[nbl::ext::imgui::UI::FontAtlasTexId].info = descriptorInfo.data() + nbl::ext::imgui::UI::FontAtlasTexId;

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

			//m_api->startCapture();

			// CPU events
			update();

			auto queue = getGraphicsQueue();
			auto cmdbuf = m_cmdBufs[resourceIx].get();

			if (!keepRunning())
				return;

			if (renderMode != E_RENDER_MODE::ERM_HLSL)
			{
				m_logger->log("Only HLSL render mode is supported.", ILogger::ELL_ERROR);
				std::exit(-1);
			}

			cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);

			// safe to proceed
			// upload buffer data
			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->beginDebugMarker("ComputeShaderPathtracer IMGUI Frame");

			updatePathtracerPushConstants();

			// TRANSITION m_outImgView to GENERAL (because of descriptorSets0 -> ComputeShader Writes into the image)
			{
				const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers[] = {
					{
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS,
								.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
								.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
								.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
							}
						},
						.image = m_outImgView->getCreationParameters().image.get(),
						.subresourceRange = {
							.aspectMask = IImage::EAF_COLOR_BIT,
							.baseMipLevel = 0u,
							.levelCount = 1u,
							.baseArrayLayer = 0u,
							.layerCount = 1u
						},
						.oldLayout = IImage::LAYOUT::UNDEFINED,
						.newLayout = IImage::LAYOUT::GENERAL
					}
				};
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers });
			}

			// transit m_cascadeView layout to GENERAL, block until previous shader is done with reading from the cascade
			if(useRWMC)
			{
				const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> cascadeBarrier[] = {
						{
							.barrier = {
								.dep = {
									.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
									.srcAccessMask = ACCESS_FLAGS::NONE,
									.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
									.dstAccessMask = ACCESS_FLAGS::NONE
								}
							},
							.image = m_cascadeView->getCreationParameters().image.get(),
							.subresourceRange = {
								.aspectMask = IImage::EAF_COLOR_BIT,
								.baseMipLevel = 0u,
								.levelCount = 1u,
								.baseArrayLayer = 0u,
								.layerCount = CascadeCount
							},
							.oldLayout = IImage::LAYOUT::UNDEFINED,
							.newLayout = IImage::LAYOUT::GENERAL
						}
				};
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = cascadeBarrier });
			}

			{
				// TODO: shouldn't it be computed only at initialization stage and on window resize?
				const uint32_t dispatchSize = usePersistentWorkGroups ?
					m_physicalDevice->getLimits().computeOptimalPersistentWorkgroupDispatchSize(WindowDimensions.x * WindowDimensions.y, RenderWorkgroupSize) :
					1 + (WindowDimensions.x * WindowDimensions.y - 1) / RenderWorkgroupSize;

				IGPUComputePipeline* pipeline = pickPTPipeline();

				cmdbuf->bindComputePipeline(pipeline);
				cmdbuf->bindDescriptorSets(EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &m_descriptorSet0.get());
				cmdbuf->bindDescriptorSets(EPBP_COMPUTE, pipeline->getLayout(), 2u, 1u, &m_descriptorSet2.get());

				const uint32_t pushConstantsSize = useRWMC ? sizeof(RenderRWMCPushConstants) : sizeof(RenderPushConstants);
				const void* pushConstantsPtr = useRWMC ? reinterpret_cast<const void*>(&rwmcPushConstants) : reinterpret_cast<const void*>(&pc);
				cmdbuf->pushConstants(pipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, pushConstantsSize, pushConstantsPtr);

				cmdbuf->dispatch(dispatchSize, 1u, 1u);
			}

			// m_cascadeView synchronization - wait for previous compute shader to write into the cascade
			// TODO: create this and every other barrier once outside of the loop?
			if(useRWMC)
			{
				const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> cascadeBarrier[] = {
						{
							.barrier = {
								.dep = {
									.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
									.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
									.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
									.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
								}
							},
							.image = m_cascadeView->getCreationParameters().image.get(),
							.subresourceRange = {
								.aspectMask = IImage::EAF_COLOR_BIT,
								.baseMipLevel = 0u,
								.levelCount = 1u,
								.baseArrayLayer = 0u,
								.layerCount = CascadeCount
							}
						}
				};
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = cascadeBarrier });
			}

			// resolve
			if(useRWMC)
			{
				if (renderMode != E_RENDER_MODE::ERM_HLSL)
				{
					m_logger->log("RWMC is only supported with HLSL.", ILogger::ELL_ERROR);
					std::exit(-1);
				}

				// TODO: shouldn't it be computed only at initialization stage and on window resize?
				// Round up division
				const uint32_t2 dispatchSize = uint32_t2(
					(m_window->getWidth() + ResolveWorkgroupSizeX - 1) / ResolveWorkgroupSizeX,
					(m_window->getHeight() + ResolveWorkgroupSizeY - 1) / ResolveWorkgroupSizeY
				);

				IGPUComputePipeline* pipeline = m_resolvePipeline.get();

				resolvePushConstants.resolveParameters = rwmc::ResolveParameters::create(rwmcBase, spp, rwmcMinReliableLuma, rwmcKappa);

				cmdbuf->bindComputePipeline(pipeline);
				cmdbuf->bindDescriptorSets(EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &m_descriptorSet0.get());
				cmdbuf->pushConstants(pipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(ResolvePushConstants), &resolvePushConstants);

				cmdbuf->dispatch(dispatchSize.x, dispatchSize.y, 1u);
			}

			// TRANSITION m_outImgView to READ (because of descriptorSets0 -> ComputeShader Writes into the image)
			{
				const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarriers[] = {
					{
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
								.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
								.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
								.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
							}
						},
						.image = m_outImgView->getCreationParameters().image.get(),
						.subresourceRange = {
							.aspectMask = IImage::EAF_COLOR_BIT,
							.baseMipLevel = 0u,
							.levelCount = 1u,
							.baseArrayLayer = 0u,
							.layerCount = 1u
						},
						.oldLayout = IImage::LAYOUT::GENERAL,
						.newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL
					}
				};
				cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imgBarriers });
			}

			// TODO: tone mapping and stuff

			asset::SViewport viewport;
			{
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = WindowDimensions.x;
				viewport.height = WindowDimensions.y;
			}
			cmdbuf->setViewport(0u, 1u, &viewport);


			VkRect2D defaultScisors[] = { {.offset = {(int32_t)viewport.x, (int32_t)viewport.y}, .extent = {(uint32_t)viewport.width, (uint32_t)viewport.height}} };
			cmdbuf->setScissor(defaultScisors);

			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};
			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());

			// Upload m_outImg to swapchain + UI
			{
				const IGPUCommandBuffer::SRenderpassBeginInfo info =
				{
					.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
					.colorClearValues = &clearColor,
					.depthStencilClearValues = nullptr,
					.renderArea = currentRenderArea
				};
				nbl::video::ISemaphore::SWaitInfo waitInfo = { .semaphore = m_semaphore.get(), .value = m_realFrameIx + 1u };

				cmdbuf->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

				cmdbuf->bindGraphicsPipeline(m_presentPipeline.get());
				cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, m_presentPipeline->getLayout(), 0, 1u, &m_presentDescriptorSet.get());
				ext::FullScreenTriangle::recordDrawCall(cmdbuf);

				const auto uiParams = m_ui.manager->getCreationParameters();
				auto* uiPipeline = m_ui.manager->getPipeline();
				cmdbuf->bindGraphicsPipeline(uiPipeline);
				cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, uiPipeline->getLayout(), uiParams.resources.texturesInfo.setIx, 1u, &m_ui.descriptorSet.get());
				m_ui.manager->render(cmdbuf, waitInfo);

				cmdbuf->endRenderPass();
			}

			cmdbuf->end();
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
							{.cmdbuf = cmdbuf }
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

						updateGUIDescriptorSet();

						if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
							m_realFrameIx--;
					}
				}

				m_window->setCaption("[Nabla Engine] HLSL Compute Path Tracer");
				m_surface->present(m_currentImageAcquire.imageIndex, rendered);
			}
			//m_api->endCapture();
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
			m_camera.setMoveSpeed(moveSpeed);
			m_camera.setRotateSpeed(rotateSpeed);

			static std::chrono::microseconds previousEventTimestamp{};

			m_inputSystem->getDefaultMouse(&mouse);
			m_inputSystem->getDefaultKeyboard(&keyboard);

			auto updatePresentationTimestamp = [&]()
			{
				m_currentImageAcquire = m_surface->acquireNextImage();

				m_oracle.reportEndFrameRecord();
				const auto timestamp = m_oracle.getNextPresentationTimeStamp();
				m_oracle.reportBeginFrameRecord();

				return timestamp;
			};

			const auto nextPresentationTimestamp = updatePresentationTimestamp();

			struct
			{
				std::vector<SMouseEvent> mouse{};
				std::vector<SKeyboardEvent> keyboard{};
			} capturedEvents;

			m_camera.beginInputProcessing(nextPresentationTimestamp);
			{
				const auto& io = ImGui::GetIO();
				mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
					{
						if (!io.WantCaptureMouse)
							m_camera.mouseProcess(events); // don't capture the events, only let camera handle them with its impl

						for (const auto& e : events) // here capture
						{
							if (e.timeStamp < previousEventTimestamp)
								continue;

							previousEventTimestamp = e.timeStamp;
							capturedEvents.mouse.emplace_back(e);

							if (e.type == nbl::ui::SMouseEvent::EET_SCROLL)
								gcIndex = std::clamp<uint16_t>(int16_t(gcIndex) + int16_t(core::sign(e.scrollEvent.verticalScroll)), int64_t(0), int64_t(ELG_COUNT - (uint8_t)1u));
						}
					}, m_logger.get());
				
				keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
					{
						if (!io.WantCaptureKeyboard)
							m_camera.keyboardProcess(events); // don't capture the events, only let camera handle them with its impl

						for (const auto& e : events) // here capture
						{
							if (e.timeStamp < previousEventTimestamp)
								continue;

							previousEventTimestamp = e.timeStamp;
							capturedEvents.keyboard.emplace_back(e);
						}
					}, m_logger.get());
			}
			m_camera.endInputProcessing(nextPresentationTimestamp);

			const core::SRange<const nbl::ui::SMouseEvent> mouseEvents(capturedEvents.mouse.data(), capturedEvents.mouse.data() + capturedEvents.mouse.size());
			const core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents(capturedEvents.keyboard.data(), capturedEvents.keyboard.data() + capturedEvents.keyboard.size());
			const auto cursorPosition = m_window->getCursorControl()->getPosition();
			const auto mousePosition = float32_t2(cursorPosition.x, cursorPosition.y) - float32_t2(m_window->getX(), m_window->getY());

			const ext::imgui::UI::SUpdateParameters params =
			{
				.mousePosition = mousePosition,
				.displaySize = { m_window->getWidth(), m_window->getHeight() },
				.mouseEvents = mouseEvents,
				.keyboardEvents = keyboardEvents
			};

			m_ui.manager->update(params);
		}
	
	private:
		void updatePathtracerPushConstants()
		{
			// disregard surface/swapchain transformation for now
			const auto viewProjectionMatrix = m_camera.getConcatenatedMatrix();
			// TODO: rewrite the `Camera` class so it uses hlsl::float32_t4x4 instead of core::matrix4SIMD
			core::matrix4SIMD invMVP;
			viewProjectionMatrix.getInverseTransform(invMVP);
			if (useRWMC)
			{
				memcpy(&rwmcPushConstants.renderPushConstants.invMVP, invMVP.pointer(), sizeof(rwmcPushConstants.renderPushConstants.invMVP));
				rwmcPushConstants.renderPushConstants.generalPurposeLightMatrix = hlsl::float32_t3x4(transpose(m_lightModelMatrix));
				rwmcPushConstants.renderPushConstants.depth = depth;
				rwmcPushConstants.renderPushConstants.sampleCount = resolvePushConstants.sampleCount = spp;
				rwmcPushConstants.renderPushConstants.pSampleSequence = m_sequenceBuffer->getDeviceAddress();
				//rwmcPushConstants.splattingParameters.log2Start = std::log2(rwmcStart);
				//rwmcPushConstants.splattingParameters.log2Base = std::log2(rwmcBase);
				float32_t2 packLogs = float32_t2(std::log2(rwmcStart), 1.0f/std::log2(rwmcBase));
				rwmcPushConstants.splattingParameters.packedLog2 = hlsl::packHalf2x16(packLogs);
			}
			else
			{
				memcpy(&pc.invMVP, invMVP.pointer(), sizeof(pc.invMVP));
				pc.generalPurposeLightMatrix = hlsl::float32_t3x4(transpose(m_lightModelMatrix));
				pc.sampleCount = spp;
				pc.depth = depth;
				pc.pSampleSequence = m_sequenceBuffer->getDeviceAddress();
			}
		}

		IGPUComputePipeline* pickPTPipeline()
		{
			IGPUComputePipeline* pipeline;
			if (useRWMC)
			{
				if (renderMode != E_RENDER_MODE::ERM_HLSL)
				{
					m_logger->log("RWMC is only supported with HLSL.", ILogger::ELL_ERROR);
					std::exit(-1);
				}

				pipeline = usePersistentWorkGroups ? m_PTHLSLPersistentWGPipelinesRWMC[PTPipeline].get() : m_PTHLSLPipelinesRWMC[PTPipeline].get();
			}
			else
			{
				if (usePersistentWorkGroups)
					pipeline = renderMode == E_RENDER_MODE::ERM_HLSL ? m_PTHLSLPersistentWGPipelines[PTPipeline].get() : m_PTGLSLPersistentWGPipelines[PTPipeline].get();
				else
					pipeline = renderMode == E_RENDER_MODE::ERM_HLSL ? m_PTHLSLPipelines[PTPipeline].get() : m_PTGLSLPipelines[PTPipeline].get();
			}

			return pipeline;
		}

	private:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;

		// gpu resources
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		std::array<smart_refctd_ptr<IGPUComputePipeline>, E_LIGHT_GEOMETRY::ELG_COUNT> m_PTGLSLPipelines;
		std::array<smart_refctd_ptr<IGPUComputePipeline>, E_LIGHT_GEOMETRY::ELG_COUNT> m_PTHLSLPipelines;
		std::array<smart_refctd_ptr<IGPUComputePipeline>, E_LIGHT_GEOMETRY::ELG_COUNT> m_PTGLSLPersistentWGPipelines;
		std::array<smart_refctd_ptr<IGPUComputePipeline>, E_LIGHT_GEOMETRY::ELG_COUNT> m_PTHLSLPersistentWGPipelines;
		std::array<smart_refctd_ptr<IGPUComputePipeline>, E_LIGHT_GEOMETRY::ELG_COUNT> m_PTHLSLPipelinesRWMC;
		std::array<smart_refctd_ptr<IGPUComputePipeline>, E_LIGHT_GEOMETRY::ELG_COUNT> m_PTHLSLPersistentWGPipelinesRWMC;
		smart_refctd_ptr<IGPUComputePipeline> m_resolvePipeline;
		smart_refctd_ptr<IGPUGraphicsPipeline> m_presentPipeline;
		uint64_t m_realFrameIx = 0;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
		ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
		smart_refctd_ptr<IGPUDescriptorSet> m_descriptorSet0, m_descriptorSet2, m_presentDescriptorSet;

		core::smart_refctd_ptr<IDescriptorPool> m_guiDescriptorSetPool;

		// system resources
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
        InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		// pathtracer resources
		smart_refctd_ptr<IGPUImageView> m_envMapView, m_scrambleView;
		smart_refctd_ptr<IGPUBuffer> m_sequenceBuffer;
		smart_refctd_ptr<IGPUImageView> m_outImgView;
		smart_refctd_ptr<IGPUImageView> m_cascadeView;

		// sync
		smart_refctd_ptr<ISemaphore> m_semaphore;

		// image upload resources
		smart_refctd_ptr<ISemaphore> m_scratchSemaphore;
		SIntendedSubmitInfo m_intendedSubmit;

		struct C_UI
		{
			nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> manager;

			struct
			{
				core::smart_refctd_ptr<video::IGPUSampler> gui, scene;
			} samplers;

			core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
		} m_ui;

		Camera m_camera;

		video::CDumbPresentationOracle m_oracle;

		uint16_t gcIndex = {}; // note: this is dirty however since I assume only single object in scene I can leave it now, when this example is upgraded to support multiple objects this needs to be changed

		float fov = 60.f, zNear = 0.1f, zFar = 10000.f, moveSpeed = 1.f, rotateSpeed = 1.f;
		float viewWidth = 10.f;
		float camYAngle = 165.f / 180.f * 3.14159f;
		float camXAngle = 32.f / 180.f * 3.14159f;
		int PTPipeline = E_LIGHT_GEOMETRY::ELG_SPHERE;
		int renderMode = E_RENDER_MODE::ERM_HLSL;
		int spp = 32;
		int depth = 3;
		float rwmcMinReliableLuma;
		float rwmcKappa;
		float rwmcStart;
		float rwmcBase;
		bool usePersistentWorkGroups = false;
		bool useRWMC = false;
		RenderRWMCPushConstants rwmcPushConstants;
		RenderPushConstants pc;
		ResolvePushConstants resolvePushConstants;

		hlsl::float32_t4x4 m_lightModelMatrix = {
			0.3f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.3f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.3f, 0.0f,
			-1.0f, 1.5f, 0.0f, 1.0f,
		};
		TransformRequestParams m_transformParams;

		bool m_firstFrame = true;
		IGPUCommandBuffer::SClearColorValue clearColor = { .float32 = {0.f,0.f,0.f,1.f} };
};

NBL_MAIN_FUNC(HLSLComputePathtracer)
