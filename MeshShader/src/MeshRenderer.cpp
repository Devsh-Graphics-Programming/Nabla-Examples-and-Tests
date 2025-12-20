#include "MeshRenderer.hpp"

namespace nbl::examples {
	#define EXPOSE_NABLA_NAMESPACES \
		using namespace nbl::core; \
		using namespace nbl::system; \
		using namespace nbl::asset; \
		using namespace nbl::video

	EXPOSE_NABLA_NAMESPACES;

	constexpr static inline auto DefaultPolygonGeometryPatch = []()->video::CAssetConverter::patch_t<asset::ICPUPolygonGeometry> {
		// we want to use the vertex data through UTBs
		using usage_f = video::IGPUBuffer::E_USAGE_FLAGS;
		video::CAssetConverter::patch_t<asset::ICPUPolygonGeometry> patch = {};
		patch.positionBufferUsages = usage_f::EUF_UNIFORM_TEXEL_BUFFER_BIT;
		patch.indexBufferUsages = usage_f::EUF_INDEX_BUFFER_BIT;
		patch.otherBufferUsages = usage_f::EUF_UNIFORM_TEXEL_BUFFER_BIT;
		return patch;
	}();

	MeshDebugRenderer::SViewParams::SViewParams(const hlsl::float32_t3x4& _view, const hlsl::float32_t4x4& _viewProj)
	{
		view = _view;
		viewProj = _viewProj;
		using namespace nbl::hlsl;
		normal = transpose(inverse(float32_t3x3(view)));
	}

	hlsl::float32_t4x4 MeshDebugRenderer::SViewParams::computeForInstance(hlsl::float32_t3x4 world) const
	{
		using namespace nbl::hlsl;
		return float32_t4x4(math::linalg::promoted_mul(float64_t4x4(viewProj), float64_t3x4(world)));
	}


	std::array<const core::smart_refctd_ptr<nbl::asset::IShader>, 3> MeshDebugRenderer::CreateTestShader(asset::IAssetManager* assMan, video::IGPURenderpass* renderpass, const uint32_t subpassIX) {
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
		const uint32_t ObjectCount = 7;
		const uint32_t InstanceCount = WorkgroupSize; //this is going to be based off limits. 64 is PROBABLY safe on all hardware, but cant guarantee
		const std::string WorkgroupSizeAsStr = std::to_string(WorkgroupSize);
		const std::string ObjectCountAsStr = std::to_string(ObjectCount);
		const std::string InstanceCountAsStr = std::to_string(InstanceCount);

		const IShaderCompiler::SMacroDefinition WorkgroupSizeDefine = { "WORKGROUP_SIZE",WorkgroupSizeAsStr };
		const IShaderCompiler::SMacroDefinition ObjectCountDefine = { "OBJECT_COUNT", ObjectCountAsStr };
		const IShaderCompiler::SMacroDefinition InstanceCountDefine = { "INSTANCE_COUNT", InstanceCountAsStr };

		const IShaderCompiler::SMacroDefinition meshArray[] = { WorkgroupSizeDefine, ObjectCountDefine, InstanceCountDefine };
		return {
			loadCompileAndCreateShader("app_resources/geom.task.hlsl", IShader::E_SHADER_STAGE::ESS_TASK, { meshArray }),
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

		// create descriptor set
		{
			// create Descriptor Set Layout
			smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
			{
				using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
				const IGPUDescriptorSetLayout::SBinding bindings[] =
				{
					{ //vertices
						.binding = VertexAttrubUTBDescBinding,
						.type = IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER,
						// need this trifecta of flags for `SubAllocatedDescriptorSet` to accept the binding as suballocatable
						.createFlags = binding_flags_t::ECF_UPDATE_AFTER_BIND_BIT | binding_flags_t::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT | binding_flags_t::ECF_PARTIALLY_BOUND_BIT,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_TASK | IShader::E_SHADER_STAGE::ESS_MESH | IShader::E_SHADER_STAGE::ESS_FRAGMENT,
						.count = MissingView
					},
					//{//indices, none of these objects use indices so I'll skip over this

					//},
					{ //meshletdataobject
						.binding = 1,
						.type = IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
						.createFlags = binding_flags_t::ECF_UPDATE_AFTER_BIND_BIT,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_TASK | IShader::E_SHADER_STAGE::ESS_MESH | IShader::E_SHADER_STAGE::ESS_FRAGMENT,
						.count = 1
					}
				};
				dsLayout = device->createDescriptorSetLayout(bindings);
				if (!dsLayout)
				{
					logger->log("Could not create descriptor set layout!", ILogger::ELL_ERROR);
					return nullptr;
				}
			}

			// create Descriptor Set
			auto pool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, { &dsLayout.get(),1 });
			auto ds = pool->createDescriptorSet(std::move(dsLayout));
			if (!ds)
			{
				logger->log("Could not descriptor set!", ILogger::ELL_ERROR);
				return nullptr;
			}
			init.subAllocDS = make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(ds));
		}

		// create pipeline layout
		const SPushConstantRange ranges[] = { {
			.stageFlags = IShader::E_SHADER_STAGE::ESS_TASK | IShader::E_SHADER_STAGE::ESS_MESH | hlsl::ShaderStage::ESS_FRAGMENT,
			.offset = 0,
			.size = sizeof(SInstance::SPushConstants),
		} };
		init.layout = device->createPipelineLayout(ranges, smart_refctd_ptr<const IGPUDescriptorSetLayout>(init.subAllocDS->getDescriptorSet()->getLayout()));
		auto shaderRet = CreateTestShader(assMan, renderpass, subpassIX);
		// create pipelines
		{
			//this needs to be fixed, the mesh and frag use different files
			IGPUMeshPipeline::SCreationParams params{
				.layout = init.layout.get(),
				.taskShader = {.shader = shaderRet[0].get(), .entryPoint = "main"},
				.meshShader = {.shader = shaderRet[1].get(), .entryPoint = "main" },
				.fragmentShader = {.shader = shaderRet[2].get(), .entryPoint = "main" }
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

		return smart_refctd_ptr<MeshDebugRenderer>(new MeshDebugRenderer(std::move(init)), dont_grab);
	}

	bool MeshDebugRenderer::addGeometries(const std::span<const video::IGPUPolygonGeometry* const> geometries)
	{
		EXPOSE_NABLA_NAMESPACES;
		if (geometries.empty())
			return false;
		auto device = const_cast<ILogicalDevice*>(m_params.layout->getOriginDevice());

		core::vector<IGPUDescriptorSet::SWriteDescriptorSet> writes;
		core::vector<IGPUDescriptorSet::SDescriptorInfo> infos;
		bool anyFailed = false;
		auto allocateUTB = [&](const IGeometry<const IGPUBuffer>::SDataView& view)->decltype(SubAllocatedDescriptorSet::invalid_value)
			{
				if (!view)
					return MissingView;
				auto index = SubAllocatedDescriptorSet::invalid_value;
				if (m_params.subAllocDS->multi_allocate(VertexAttrubUTBDescBinding, 1, &index) != 0)
				{
					anyFailed = true;
					return MissingView;
				}
				const auto infosOffset = infos.size();
				infos.emplace_back().desc = device->createBufferView(view.src, view.composed.format);
				writes.emplace_back() = {
					.dstSet = m_params.subAllocDS->getDescriptorSet(),
					.binding = VertexAttrubUTBDescBinding,
					.arrayElement = index,
					.count = 1,
					.info = reinterpret_cast<const IGPUDescriptorSet::SDescriptorInfo*>(infosOffset)
				};
				return index;
			};

		auto resetGeoms = core::makeRAIIExiter(
			[&]()->void {
				for (auto& write : writes) {
					immediateDealloc(write.arrayElement);
				}
			}
		);

		//the order doesnt really matter as long as the data is respective
		uint8_t meshIndex = 0;
		for (const auto geom : geometries)
		{
			// could also check device origin on all buffers
			if (!geom->valid())
				return false;

			
			auto& out = m_geoms.meshData[meshIndex];
			meshIndex++;
			out.vertCount = geom->getVertexReferenceCount();
			out.positionView = allocateUTB(geom->getPositionView());
			out.normalView = allocateUTB(geom->getNormalView());

			if(geom->getIndexingCallback()->knownTopology() == E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_FAN){
				out.objectType &= 2;
			}
			const auto& view = geom->getIndexView();
			if (view) {
				view.getElementCount();
				assert(out.vertCount == view.getElementCount() && "not currently setup to support index buffer");
			}
		}

		if (anyFailed)
			device->getLogger()->log("Failed to allocate a UTB for some geometries, probably ran out of space in Descriptor Set!", system::ILogger::ELL_ERROR);

		// no geometry
		if (infos.empty())
			return false;

		// unbase our pointers
		for (auto& write : writes)
			write.info = infos.data() + reinterpret_cast<const size_t&>(write.info);
		if (!device->updateDescriptorSets(writes, {}))
			return false;

		// retain
		writes.clear();
		return true;
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

		core::vector<SubAllocatedDescriptorSet::value_type> deferredFree;
		deferredFree.reserve(3);
		auto deallocate = [&](SubAllocatedDescriptorSet::value_type index)->void
			{
				if (index >= MissingView)
					return;
				if (info.semaphore)
					deferredFree.push_back(index);
				else
					immediateDealloc(index);
			};
		auto geo = m_geoms.meshData[ix];
		deallocate(geo.positionView);
		deallocate(geo.normalView);

		if (deferredFree.empty())
			return;
		m_params.subAllocDS->multi_deallocate(VertexAttrubUTBDescBinding, deferredFree.size(), deferredFree.data(), info);
	}

	void MeshDebugRenderer::render(video::IGPUCommandBuffer* cmdbuf, const SViewParams& viewParams) const
	{
		EXPOSE_NABLA_NAMESPACES;

		cmdbuf->beginDebugMarker("MeshDebugRenderer::render");

		const auto* layout = m_params.layout.get();
		const auto ds = m_params.subAllocDS->getDescriptorSet();
		cmdbuf->bindDescriptorSets(E_PIPELINE_BIND_POINT::EPBP_GRAPHICS, layout, 0, 1, &ds);

		for (const auto& instance : m_instances) {
			cmdbuf->bindMeshPipeline(m_params.pipeline.get());
			const auto pc = instance.computePushConstants(viewParams);
			cmdbuf->pushConstants(layout, hlsl::ShaderStage::ESS_TASK | hlsl::ShaderStage::ESS_MESH | hlsl::ShaderStage::ESS_FRAGMENT, 0, sizeof(pc), &pc);
			//if (m_geoms->indexBuffer)
			//{
				//cmdbuf->bindIndexBuffer(geo->indexBuffer,geo->indexType);
				//cmdbuf->drawIndexed(geo->elementCount,1,0,0,0);
				//cmdbuf->bindDescriptorSets(geo->indexBuffer);
			//}
			//else {
				//cmdbuf->bindDescriptorSets(geo->indexBuffer);
				//cmdbuf->draw(geo->elementCount, 1, 0, 0);
			//}
			cmdbuf->drawMeshTasks(1, 1, 1);
		}
		cmdbuf->endDebugMarker();
	}
}//namespace nbl::examples