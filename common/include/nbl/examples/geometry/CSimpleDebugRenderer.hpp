#ifndef _NBL_EXAMPLES_C_SIMPLE_DEBUG_RENDERER_H_INCLUDED_
#define _NBL_EXAMPLES_C_SIMPLE_DEBUG_RENDERER_H_INCLUDED_


#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/examples/geometry/SPushConstants.hlsl"

// TODO: Arek bring back
//#include "nbl/examples/geometry/spirv/builtin/CArchive.h"
//#include "nbl/examples/geometry/spirv/builtin/builtinResources.h"


namespace nbl::examples
{

class CSimpleDebugRenderer final : public core::IReferenceCounted
{
#define EXPOSE_NABLA_NAMESPACES \
			using namespace nbl::core; \
			using namespace nbl::system; \
			using namespace nbl::asset; \
			using namespace nbl::video

	public:
		//
		constexpr static inline uint16_t VertexAttrubUTBDescBinding = 0;
		//
		struct SViewParams
		{
			inline SViewParams(const hlsl::float32_t3x4& _view, const hlsl::float32_t4x4& _viewProj)
			{
				view = _view;
				viewProj = _viewProj;
				using namespace nbl::hlsl;
				normal = transpose(inverse(float32_t3x3(view)));
			}

			inline auto computeForInstance(hlsl::float32_t3x4 world) const
			{
				using namespace nbl::hlsl;
				hlsl::examples::geometry_creator_scene::SInstanceMatrices retval = {
					.worldViewProj = float32_t4x4(math::linalg::promoted_mul(float64_t4x4(viewProj),float64_t3x4(world)))
				};
				const auto sub3x3 = mul(float64_t3x3(viewProj),float64_t3x3(world));
				retval.normal = float32_t3x3(transpose(inverse(sub3x3)));
				return retval;
			}

			hlsl::float32_t3x4 view;
			hlsl::float32_t4x4 viewProj;
			hlsl::float32_t3x3 normal;
		};
		//
		struct SPackedGeometry
		{
			core::smart_refctd_ptr<const video::IGPUGraphicsPipeline> pipeline = {};
			asset::SBufferBinding<const video::IGPUBuffer> indexBuffer = {};
			uint32_t elementCount = 0;
			// indices into the descriptor set
			uint8_t positionView = 0;
			uint8_t normalView = 0;
			uint8_t uvView = 0;
			asset::E_INDEX_TYPE indexType = asset::EIT_UNKNOWN;
		};
		//
		struct SInstance
		{
			using SPushConstants = hlsl::examples::geometry_creator_scene::SPushConstants;
			inline SPushConstants computePushConstants(const SViewParams& viewParams) const
			{
				using namespace hlsl;
				return {
					.matrices = viewParams.computeForInstance(world),
					.positionView = packedGeo->positionView,
					.normalView = packedGeo->normalView,
					.uvView = packedGeo->uvView
				};
			}

			hlsl::float32_t3x4 world;
			const SPackedGeometry* packedGeo;
		};

		//
		constexpr static inline auto DefaultPolygonGeometryPatch = []()->video::CAssetConverter::patch_t<asset::ICPUPolygonGeometry>
		{
			// we want to use the vertex data through UTBs
			using usage_f = video::IGPUBuffer::E_USAGE_FLAGS;
			video::CAssetConverter::patch_t<asset::ICPUPolygonGeometry> patch = {};
			patch.positionBufferUsages = usage_f::EUF_UNIFORM_TEXEL_BUFFER_BIT;
			patch.indexBufferUsages = usage_f::EUF_INDEX_BUFFER_BIT;
			patch.otherBufferUsages = usage_f::EUF_UNIFORM_TEXEL_BUFFER_BIT;
			return patch;
		}();

		//
		static inline core::smart_refctd_ptr<CSimpleDebugRenderer> create(asset::IAssetManager* assMan, video::IGPURenderpass* renderpass, const uint32_t subpassIX)
		{
			EXPOSE_NABLA_NAMESPACES;

			if (!renderpass)
				return nullptr;
			auto device = const_cast<ILogicalDevice*>(renderpass->getOriginDevice());
			auto logger = device->getLogger();

			if (!assMan)
				return nullptr;

			// load shader
			smart_refctd_ptr<IShader> shader;
			{
				constexpr std::string_view key = "nbl/examples/shaders/geometry/unified.hlsl.spv";
				const auto bundle = assMan->getAsset(key.data(), {});

				const auto contents = bundle.getContents();
				if (contents.empty() || bundle.getAssetType()!=IAsset::ET_SHADER)
					return nullptr;
				shader = IAsset::castDown<IShader>(contents[0]);
				
				if (!shader)
					return nullptr;
			}

			SInitParams init;

			// create descriptor set
			{
				// create Descriptor Set Layout
				smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
				{
					using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
					const IGPUDescriptorSetLayout::SBinding bindings[] =
					{
						{
							.binding = VertexAttrubUTBDescBinding,
							.type = IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER,
							// need this trifecta of flags for `SubAllocatedDescriptorSet` to accept the binding as suballocatable
							.createFlags = binding_flags_t::ECF_UPDATE_AFTER_BIND_BIT|binding_flags_t::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT |binding_flags_t::ECF_PARTIALLY_BOUND_BIT,
							.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX|IShader::E_SHADER_STAGE::ESS_FRAGMENT,
							.count = SInstance::SPushConstants::DescriptorCount
						}
					};
					dsLayout = device->createDescriptorSetLayout(bindings);
					if (!dsLayout)
					{
						logger->log("Could not create descriptor set layout!",ILogger::ELL_ERROR);
						return nullptr;
					}
				}

				// create Descriptor Set
				auto pool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT,{&dsLayout.get(),1});
				auto ds = pool->createDescriptorSet(std::move(dsLayout));
				if (!ds)
				{
					logger->log("Could not descriptor set!",ILogger::ELL_ERROR);
					return nullptr;
				}
				init.subAllocDS = make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(ds));
			}

			// create pipeline layout
			const SPushConstantRange ranges[] = {{
				.stageFlags = hlsl::ShaderStage::ESS_VERTEX,
				.offset = 0,
				.size = sizeof(SInstance::SPushConstants),
			}};
			init.layout = device->createPipelineLayout(ranges,smart_refctd_ptr<const IGPUDescriptorSetLayout>(init.subAllocDS->getDescriptorSet()->getLayout()));

			// create pipelines
			using pipeline_e = SInitParams::PipelineType;
			{
				IGPUGraphicsPipeline::SCreationParams params[pipeline_e::Count] = {};
				params[pipeline_e::BasicTriangleList].vertexShader = {.shader=shader.get(),.entryPoint="BasicVS"};
				params[pipeline_e::BasicTriangleList].fragmentShader = {.shader=shader.get(),.entryPoint="BasicFS"};
				params[pipeline_e::BasicTriangleFan].vertexShader = {.shader=shader.get(),.entryPoint="BasicVS"};
				params[pipeline_e::BasicTriangleFan].fragmentShader = {.shader=shader.get(),.entryPoint="BasicFS"};
				params[pipeline_e::Cone].vertexShader = {.shader=shader.get(),.entryPoint="ConeVS"};
				params[pipeline_e::Cone].fragmentShader = {.shader=shader.get(),.entryPoint="ConeFS"};
				for (auto i=0; i<pipeline_e::Count; i++)
				{
					params[i].layout = init.layout.get();
					// no vertex input
					auto& primitiveAssembly = params[i].cached.primitiveAssembly;
					auto& rasterization = params[i].cached.rasterization;
					auto& blend = params[i].cached.blend;
					const auto type = static_cast<pipeline_e>(i);
					switch (type)
					{
						case pipeline_e::BasicTriangleFan:
							primitiveAssembly.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_FAN;
							break;
						default:
							primitiveAssembly.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST;
							break;
					}
					primitiveAssembly.primitiveRestartEnable = false;
					primitiveAssembly.tessPatchVertCount = 3;
					rasterization.faceCullingMode = EFCM_NONE;
					params[i].cached.subpassIx = subpassIX;
					params[i].renderpass = renderpass;
				}
				if (!device->createGraphicsPipelines(nullptr,params,init.pipelines))
				{
					logger->log("Could not create Graphics Pipelines!",ILogger::ELL_ERROR);
					return nullptr;
				}
			}

			return smart_refctd_ptr<CSimpleDebugRenderer>(new CSimpleDebugRenderer(std::move(init)),dont_grab);
		}

		//
		static inline core::smart_refctd_ptr<CSimpleDebugRenderer> create(asset::IAssetManager* assMan, video::IGPURenderpass* renderpass, const uint32_t subpassIX, const std::span<const video::IGPUPolygonGeometry* const> geometries)
		{
			auto retval = create(assMan,renderpass,subpassIX);
			if (retval)
				retval->addGeometries(geometries);
			return retval;
		}

		//
		struct SInitParams
		{
			enum PipelineType : uint8_t
			{
				BasicTriangleList,
				BasicTriangleFan,
				Cone, // special case
				Count
			};

			core::smart_refctd_ptr<video::SubAllocatedDescriptorSet> subAllocDS;
			core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
			core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipelines[PipelineType::Count];
		};
		inline const SInitParams& getInitParams() const {return m_params;}

		//
		inline bool addGeometries(const std::span<const video::IGPUPolygonGeometry* const> geometries)
		{
			EXPOSE_NABLA_NAMESPACES;
			if (geometries.empty())
				return false;
			auto device = const_cast<ILogicalDevice*>(m_params.layout->getOriginDevice());

			core::vector<IGPUDescriptorSet::SWriteDescriptorSet> writes;
			core::vector<IGPUDescriptorSet::SDescriptorInfo> infos;
			auto allocateUTB = [&](const IGeometry<const IGPUBuffer>::SDataView& view)->uint8_t
			{
				if (!view)
					return SInstance::SPushConstants::DescriptorCount;
				auto index = SubAllocatedDescriptorSet::invalid_value;
				if (m_params.subAllocDS->multi_allocate(VertexAttrubUTBDescBinding,1,&index)!=0)
					return SInstance::SPushConstants::DescriptorCount;
				const auto retval = infos.size();
				infos.emplace_back().desc = device->createBufferView(view.src,view.composed.format);
				writes.emplace_back() = {
					.dstSet = m_params.subAllocDS->getDescriptorSet(),
					.binding = VertexAttrubUTBDescBinding,
					.arrayElement = index,
					.count = 1,
					.info = reinterpret_cast<const IGPUDescriptorSet::SDescriptorInfo*>(retval)
				};
				return retval;
			};

			auto sizeToSet = m_geoms.size();
			auto resetGeoms = core::makeRAIIExiter([&]()->void
				{
					for (auto& write : writes)
						immediateDealloc(write.arrayElement);
					m_geoms.resize(sizeToSet);
				}
			);
			for (const auto geom : geometries)
			{
				// could also check device origin on all buffers
				if (!geom->valid())
					return false;
				auto& out = m_geoms.emplace_back();
				using pipeline_e = SInitParams::PipelineType;
				switch (geom->getIndexingCallback()->knownTopology())
				{
					case E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_FAN:
						out.pipeline = m_params.pipelines[pipeline_e::BasicTriangleFan];
						break;
					default:
						out.pipeline = m_params.pipelines[pipeline_e::BasicTriangleList];
						break;
				}
				if (const auto& view=geom->getIndexView(); view)
				{
					out.indexBuffer.offset = view.src.offset;
					out.indexBuffer.buffer = view.src.buffer;
					switch (view.composed.format)
					{
						case E_FORMAT::EF_R16_UINT:
							out.indexType = EIT_16BIT;
							break;
						case E_FORMAT::EF_R32_UINT:
							out.indexType = EIT_32BIT;
							break;
						default:
							return false;
					}
				}
				out.elementCount = geom->getVertexReferenceCount();
				out.positionView = allocateUTB(geom->getPositionView());
				out.normalView = allocateUTB(geom->getNormalView());
				// the first view is usually the UV
				if (const auto& auxViews = geom->getAuxAttributeViews(); !auxViews.empty())
					out.uvView = allocateUTB(auxViews.front());
			}

			// no geometry
			if (infos.empty())
				return false;

			// unbase our pointers
			for (auto& write : writes)
				write.info = infos.data()+reinterpret_cast<const size_t&>(write.info);
			if (!device->updateDescriptorSets(writes,{}))
				return false;

			// retain
			writes.clear();
			sizeToSet = m_geoms.size();
			return true;
		}

		//
		inline void removeGeometry(const uint32_t ix, const video::ISemaphore::SWaitInfo& info)
		{
			EXPOSE_NABLA_NAMESPACES;
			if (ix>=m_geoms.size())
				return;

			core::vector<SubAllocatedDescriptorSet::value_type> deferredFree;
			deferredFree.reserve(3);
			auto deallocate = [&](SubAllocatedDescriptorSet::value_type index)->void
			{
				if (info.semaphore)
					deferredFree.push_back(index);
				else
					immediateDealloc(index);
			};
			auto geo = m_geoms.begin() + ix;
			deallocate(geo->positionView);
			deallocate(geo->normalView);
			deallocate(geo->uvView);
			m_geoms.erase(geo);

			if (deferredFree.empty())
				return;

			core::vector<IGPUDescriptorSet::SDropDescriptorSet> nullify(deferredFree.size());
			const_cast<ILogicalDevice*>(m_params.layout->getOriginDevice())->nullifyDescriptors(nullify);
		}

		//
		inline void clearGeometries(const video::ISemaphore::SWaitInfo& info)
		{
			// back to front to avoid O(n^2) resize
			while (!m_geoms.empty())
				removeGeometry(m_geoms.size()-1,info);
		}

		//
		inline const auto& getGeometries() const {return m_geoms;}
		inline auto& getGeometry(const uint32_t ix) {return m_geoms[ix];}

		//
		inline void render(video::IGPUCommandBuffer* cmdbuf, const SViewParams& viewParams) const
		{
			EXPOSE_NABLA_NAMESPACES;

			cmdbuf->beginDebugMarker("CSimpleDebugRenderer::render");

			const auto* layout = m_params.layout.get();
			const auto ds = m_params.subAllocDS->getDescriptorSet();
			cmdbuf->bindDescriptorSets(E_PIPELINE_BIND_POINT::EPBP_GRAPHICS,layout,0,1,&ds);

			for (const auto& instance : m_instances)
			{
				const auto* geo = instance.packedGeo;
				cmdbuf->bindGraphicsPipeline(geo->pipeline.get());
				const auto pc = instance.computePushConstants(viewParams);
				cmdbuf->pushConstants(layout,hlsl::ShaderStage::ESS_VERTEX,0,sizeof(pc),&pc);
				if (geo->indexBuffer)
				{
					cmdbuf->bindIndexBuffer(geo->indexBuffer,geo->indexType);
					cmdbuf->drawIndexed(geo->elementCount,1,0,0,0);
				}
				else
					cmdbuf->draw(geo->elementCount,1,0,0);
			}
			cmdbuf->endDebugMarker();
		}

		core::vector<SInstance> m_instances;

	protected:
		inline CSimpleDebugRenderer(SInitParams&& _params) : m_params(std::move(_params)) {}
		inline ~CSimpleDebugRenderer()
		{
			// clean shutdown, can also make SubAllocatedDescriptorSet resillient against that, and issue `device->waitIdle` if not everything is freed
			const_cast<video::ILogicalDevice*>(m_params.layout->getOriginDevice())->waitIdle();
			clearGeometries({});
		}

		inline void immediateDealloc(video::SubAllocatedDescriptorSet::value_type index)
		{
			video::IGPUDescriptorSet::SDropDescriptorSet dummy[1];
			m_params.subAllocDS->multi_deallocate(dummy,VertexAttrubUTBDescBinding,1,&index);
		}

		SInitParams m_params;
		core::vector<SPackedGeometry> m_geoms;
#undef EXPOSE_NABLA_NAMESPACES
};

}
#endif