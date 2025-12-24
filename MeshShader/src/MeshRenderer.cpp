#include "MeshRenderer.hpp"



namespace nbl::examples {



	#define EXPOSE_NABLA_NAMESPACES \
		using namespace nbl::core; \
		using namespace nbl::system; \
		using namespace nbl::asset; \
		using namespace nbl::video

	EXPOSE_NABLA_NAMESPACES;

	std::array<const core::smart_refctd_ptr<nbl::asset::IShader>, 2> MeshDebugRenderer::CreateTestShader(asset::IAssetManager* assMan, video::IGPURenderpass* renderpass, const uint32_t subpassIX) {
		auto device = const_cast<ILogicalDevice*>(renderpass->getOriginDevice());
		auto logger = device->getLogger();
		auto loadCompileAndCreateShader = [&](const std::string& relPath, hlsl::ShaderStage stage, std::span<const asset::IShaderCompiler::SMacroDefinition> extraDefines) -> smart_refctd_ptr<IShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = logger;
				lp.workingDirectory = ""; // virtual root
				auto assetBundle = assMan->getAsset(relPath, lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty()) {
					printf("asset was empty - %s\n", relPath.c_str());
					return nullptr;
				}

				// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
				auto sourceRaw = IAsset::castDown<IShader>(assets[0]);
				if (!sourceRaw) {
					printf("source raw was nullptr - %s\n", relPath.c_str());
					return nullptr;
				}

				nbl::video::ILogicalDevice::SShaderCreationParameters creationParams{
					.source = sourceRaw.get(),
					.optimizer = nullptr,
					.readCache = nullptr,
					.writeCache = nullptr,
					.extraDefines = extraDefines,
					.stage = stage
				};

				auto ret = device->compileShader(creationParams);
				if (ret.get() == nullptr) {
					printf("failed to compile shader - %s\n", relPath.c_str());
				}
				//m_assetMgr->removeAssetFromCache(assetBundle);
				//return nullptr;
				//i dont think that ^ was working
				return ret;
			};
		constexpr uint32_t WorkgroupSize = 64;
		//const uint32_t ObjectCount = 7;
		//const uint32_t InstanceCount = 8; //this is going to be based off limits. 64 is PROBABLY safe on all hardware, but cant guarantee
		const std::string WorkgroupSizeAsStr = std::to_string(WorkgroupSize);
		//const std::string ObjectCountAsStr = std::to_string(ObjectCount);
		//const std::string InstanceCountAsStr = std::to_string(InstanceCount);

		const IShaderCompiler::SMacroDefinition WorkgroupSizeDefine = { "WORKGROUP_SIZE",WorkgroupSizeAsStr };
		//const IShaderCompiler::SMacroDefinition ObjectCountDefine = { "OBJECT_COUNT", ObjectCountAsStr };
		//const IShaderCompiler::SMacroDefinition InstanceCountDefine = { "INSTANCE_COUNT", InstanceCountAsStr };

		const IShaderCompiler::SMacroDefinition meshArray[] = { WorkgroupSizeDefine };// , ObjectCountDefine, InstanceCountDefine};
		return {
			//loadCompileAndCreateShader("app_resources/geom.task.hlsl", IShader::E_SHADER_STAGE::ESS_TASK, { meshArray }),
			loadCompileAndCreateShader("app_resources/geom.mesh.hlsl", IShader::E_SHADER_STAGE::ESS_MESH, { meshArray }),
			loadCompileAndCreateShader("app_resources/geom.frag.hlsl", IShader::E_SHADER_STAGE::ESS_FRAGMENT, {})
		};
	}

	core::smart_refctd_ptr<MeshDebugRenderer> MeshDebugRenderer::create(asset::IAssetManager* assMan, video::IGPURenderpass* renderpass, const uint32_t subpassIX)
	{
		EXPOSE_NABLA_NAMESPACES;

		if (!renderpass)
			return nullptr;
		auto device = const_cast<ILogicalDevice*>(renderpass->getOriginDevice());
		auto logger = device->getLogger();

		if (!assMan)
			return nullptr;

		SInitParams init;

		smart_refctd_ptr<IGPUDescriptorSetLayout> meshLayout;

		// create descriptor set
		{
			//creating meshdatabuffer descriptor set
			{
				using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
				const IGPUDescriptorSetLayout::SBinding bindings[] =
				{ //meshletdataobject
					{
						.binding = 0,
						.type = IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
						.createFlags = binding_flags_t::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_MESH,
						.count = 1
					}
				};
				meshLayout = device->createDescriptorSetLayout(bindings);
				if (!meshLayout)
				{
					logger->log("Could not create mesh descriptor set layout!", ILogger::ELL_ERROR);
					return nullptr;
				}
			}

			// create Descriptor Set
			std::vector< IGPUDescriptorSetLayout const*> dsls{ meshLayout.get() };

			auto pool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, dsls);		}

		// create pipeline layout
		const SPushConstantRange ranges[] = { {
			.stageFlags = IShader::E_SHADER_STAGE::ESS_TASK | IShader::E_SHADER_STAGE::ESS_MESH | hlsl::ShaderStage::ESS_FRAGMENT,
			.offset = 0,
			.size = sizeof(SInstance::SPushConstants),
		} };

		//because of the move semantics, the descriptor set we just created is no longer valid. instead, we need to go and rebuild a smart pointer to that descriptor set.
		init.pipe_layout = device->createPipelineLayout(ranges, smart_refctd_ptr<const IGPUDescriptorSetLayout>(meshLayout));
		auto shaderRet = CreateTestShader(assMan, renderpass, subpassIX);
		// create pipelines
		{
			//this needs to be fixed, the mesh and frag use different files
			IGPUMeshPipeline::SCreationParams params{
				.layout = init.pipe_layout.get(),
				//.taskShader = {.shader = shaderRet[0].get(), .entryPoint = "main"},
				.meshShader = {.shader = shaderRet[0].get(), .entryPoint = "main" },
				.fragmentShader = {.shader = shaderRet[1].get(), .entryPoint = "main" }
			};
			// no vertex input, or assembly
			auto& rasterization = params.cached.rasterization;
			auto& blend = params.cached.blend;
			rasterization.faceCullingMode = EFCM_NONE;
			params.cached.subpassIx = subpassIX;
			params.renderpass = renderpass;

			if (!device->createMeshPipelines(nullptr, { &params, 1 }, &init.pipeline))
			{
				logger->log("Could not create Mesh Pipeline!", ILogger::ELL_ERROR);
				return nullptr;
			}
		}

		auto ret = smart_refctd_ptr<MeshDebugRenderer>(new MeshDebugRenderer(std::move(init)), dont_grab);
		ret->mesh_layout = meshLayout;

		return ret;
	}


	void MeshDebugRenderer::clearGeometries(const video::ISemaphore::SWaitInfo& info) {
		//im currently assuming every object gets loaded correctly. definitely incorrect
		for (uint8_t i = 0; i < m_geoms.MaxObjectCount; i++) {
			removeGeometry(i, info);
		}
	}

	void MeshDebugRenderer::removeGeometry(const uint32_t ix, const video::ISemaphore::SWaitInfo& info)
	{
		EXPOSE_NABLA_NAMESPACES;


	}

	void MeshDebugRenderer::render(video::IGPUCommandBuffer* cmdbuf, nbl::hlsl::float32_t4x4 const& mvp) const
	{
		EXPOSE_NABLA_NAMESPACES;

		cmdbuf->beginDebugMarker("MeshDebugRenderer::render");

		const auto* layout = m_params.pipe_layout.get();
		std::array descriptors = { m_params.meshDescriptor.get()};
		cmdbuf->bindDescriptorSets(E_PIPELINE_BIND_POINT::EPBP_GRAPHICS, layout, 0, descriptors.size(), descriptors.data());

		cmdbuf->bindMeshPipeline(m_params.pipeline.get());
		SInstance::SPushConstants pc{
			.viewProj = mvp,
			.vertCount = 36
		};
		cmdbuf->pushConstants(layout, hlsl::ShaderStage::ESS_TASK | hlsl::ShaderStage::ESS_MESH | hlsl::ShaderStage::ESS_FRAGMENT, 0, sizeof(pc), &pc);

		cmdbuf->drawMeshTasks(1, 1, 1);
		
		cmdbuf->endDebugMarker();
	}
}//namespace nbl::examples