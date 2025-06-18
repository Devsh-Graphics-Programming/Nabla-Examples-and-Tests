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
		constexpr static inline auto DescriptorCount = 255;
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
		static inline core::smart_refctd_ptr<CSimpleDebugRenderer> create(asset::IAssetManager* assMan, video::IGPURenderpass* renderpass, const uint32_t subpassIX, const CGeometryCreatorScene* scene)
		{
			EXPOSE_NABLA_NAMESPACES;

			if (!!renderpass)
				return nullptr;
			auto device = const_cast<ILogicalDevice*>(renderpass->getOriginDevice());
			auto logger = device->getLogger();

			if (!assMan || !scene)
				return nullptr;
			const auto namedGeoms = scene->getGeometries();
			if (namedGeoms.empty())
				return nullptr;

			// load shader
			smart_refctd_ptr<IShader> shader;
			{
				const auto bundle = assMan->getAsset("nbl/examples/geometry/shaders/unified.hlsl",{});
				//const auto bundle = assMan->getAsset("nbl/examples/geometry/shaders/unified.spv",{});
				const auto contents = bundle.getContents();
				if (bundle.getAssetType()!=IAsset::ET_SHADER || contents.empty())
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
					const IGPUDescriptorSetLayout::SBinding bindings[] =
					{
						{
							.binding = 0,
							.type = IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER,
							// some geometries may not have particular attributes
							.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_PARTIALLY_BOUND_BIT,
							.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX|IShader::E_SHADER_STAGE::ESS_FRAGMENT,
							.count = DescriptorCount
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
				init.ds = pool->createDescriptorSet(std::move(dsLayout));
				if (!init.ds)
				{
					logger->log("Could not descriptor set!",ILogger::ELL_ERROR);
					return nullptr;
				}
			}

			// create pipeline layout
			const SPushConstantRange ranges[] = {{
				.stageFlags = hlsl::ShaderStage::ESS_VERTEX,
				.offset = 0,
				.size = sizeof(SInstance::SPushConstants),
			}};
			init.layout = device->createPipelineLayout(ranges,smart_refctd_ptr<const IGPUDescriptorSetLayout>(init.ds->getLayout()));

			// create pipelines
			enum PipelineType : uint8_t
			{
				BasicTriangleList,
				BasicTriangleFan,
				Cone,
				Count
			};
			smart_refctd_ptr<IGPUGraphicsPipeline> pipelines[PipelineType::Count] = {};
			{
				IGPUGraphicsPipeline::SCreationParams params[PipelineType::Count] = {};
				params[PipelineType::BasicTriangleList].vertexShader = {.shader=shader.get(),.entryPoint="BasicTriangleListVS"};
				params[PipelineType::BasicTriangleList].fragmentShader = {.shader=shader.get(),.entryPoint="BasicFS"};
				params[PipelineType::BasicTriangleFan].vertexShader = {.shader=shader.get(),.entryPoint="BasicTriangleFanVS"};
				params[PipelineType::BasicTriangleFan].fragmentShader = {.shader=shader.get(),.entryPoint="BasicFS"};
				params[PipelineType::Cone].vertexShader = {.shader=shader.get(),.entryPoint="ConeVS"};
				params[PipelineType::Cone].fragmentShader = {.shader=shader.get(),.entryPoint="ConeFS"};
				for (auto i=0; i< PipelineType::Count; i++)
				{
					params[i].layout = init.layout.get();
					// no vertex input
					auto& primitiveAssembly = params[i].cached.primitiveAssembly;
					auto& rasterization = params[i].cached.rasterization;
					auto& blend = params[i].cached.blend;
					const auto type = static_cast<PipelineType>(i);
					switch (type)
					{
						case PipelineType::BasicTriangleFan:
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
				if (!device->createGraphicsPipelines(nullptr,params,pipelines))
				{
					logger->log("Could not create Graphics Pipelines!",ILogger::ELL_ERROR);
					return nullptr;
				}
			}

			// write geometries' attributes to descriptor set
			{
				core::vector<IGPUDescriptorSet::SDescriptorInfo> infos;
				auto allocateUTB = [device,&infos](const IGeometry<const IGPUBuffer>::SDataView& view)->uint8_t
				{
					if (!view)
						return DescriptorCount;
					const auto retval = infos.size();
					infos.emplace_back().desc = device->createBufferView(view.src, view.composed.format);
					return retval;
				};

				for (const auto& entry : namedGeoms)
				{
					const auto* geom = entry.geom.get();
					// could also check device origin on all buffers
					if (!geom->valid())
						continue;
					auto& out = init.geoms.emplace_back();
					switch (geom->getIndexingCallback()->knownTopology())
					{
						case E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_FAN:
							out.pipeline = pipelines[PipelineType::BasicTriangleFan];
							break;
						default:
							out.pipeline = pipelines[PipelineType::BasicTriangleList];
							break;
					}
					// special case
					if (entry.name=="Cone")
						out.pipeline = pipelines[PipelineType::Cone];
					if (const auto& view=geom->getIndexView(); view)
					{
						out.indexBuffer.offset = view.src.offset;
						out.indexBuffer.buffer = view.src.buffer;
					}
					out.elementCount = geom->getVertexReferenceCount();
					out.positionView = allocateUTB(geom->getPositionView());
					out.normalView = allocateUTB(geom->getNormalView());
					// the first view is usually the UV
					if (const auto& auxViews = geom->getAuxAttributeViews(); !auxViews.empty())
						out.uvView = allocateUTB(auxViews.front());
				}

				if (infos.empty())
					return nullptr;
				const IGPUDescriptorSet::SWriteDescriptorSet write = {
					.dstSet = init.ds.get(),
					.binding = 0,
					.arrayElement = 0,
					.count = static_cast<uint32_t>(infos.size()),
					.info = infos.data()
				};
				if (!device->updateDescriptorSets({&write,1},{}))
					return nullptr;
			}

			return smart_refctd_ptr<CSimpleDebugRenderer>(new CSimpleDebugRenderer(std::move(init)),dont_grab);
		}

		//
		struct SInitParams
		{
			core::smart_refctd_ptr<video::IGPUDescriptorSet> ds;
			core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
			core::vector<SPackedGeometry> geoms;
		};
		inline const SInitParams& getInitParams() const {return m_params;}

		//
		inline void render(video::IGPUCommandBuffer* cmdbuf, const SViewParams& viewParams) const
		{
			EXPOSE_NABLA_NAMESPACES;

			cmdbuf->beginDebugMarker("CSimpleDebugRenderer::render");

			const auto* layout = m_params.layout.get();
			cmdbuf->bindDescriptorSets(E_PIPELINE_BIND_POINT::EPBP_GRAPHICS,layout,0,1,&m_params.ds.get());

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

		SInitParams m_params;
#undef EXPOSE_NABLA_NAMESPACES
};

}
#endif