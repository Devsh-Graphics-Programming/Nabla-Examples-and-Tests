#ifndef _NBL_EXAMPLES_C_GEOMETRY_CREATOR_SCENE_H_INCLUDED_
#define _NBL_EXAMPLES_C_GEOMETRY_CREATOR_SCENE_H_INCLUDED_


#include <nabla.h>
#include "nbl/asset/utils/CGeometryCreator.h"


namespace nbl::examples
{

#define EXPOSE_NABLA_NAMESPACES using namespace nbl::core; \
using namespace nbl::system; \
using namespace nbl::asset; \
using namespace nbl::video

class CGeometryCreatorScene : public core::IReferenceCounted
{
	public:
		//
		enum ObjectType : uint8_t
		{
			OT_CUBE,
			OT_SPHERE,
			OT_CYLINDER,
			OT_RECTANGLE,
			OT_DISK,
			OT_ARROW,
			OT_CONE,
			OT_ICOSPHERE,

			OT_COUNT,
			OT_UNKNOWN = OT_COUNT
		};

		//
		struct SCreateParams
		{
			video::IQueue* transferQueue;
			video::IUtilities* utilities;
			system::ILogger* logger;
			std::span<const uint32_t> addtionalBufferOwnershipFamilies = {};
		};
		static inline core::smart_refctd_ptr<CGeometryCreatorScene> create(SCreateParams&& params, const video::CAssetConverter::patch_t<asset::ICPUPolygonGeometry>& geometryPatch)
		{
			EXPOSE_NABLA_NAMESPACES;
			auto* logger = params.logger;
			assert(logger);
			if (!params.transferQueue)
			{
				logger->log("Pass a non-null `IQueue* transferQueue`!",ILogger::ELL_ERROR);
				return nullptr;
			}
			if (!params.utilities)
			{
				logger->log("Pass a non-null `IUtilities* utilities`!",ILogger::ELL_ERROR);
				return nullptr;
			}


			core::vector<SNamedGeometry> namedGeometries;
			core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
			// create out geometries
			{
				auto addGeometry = [&namedGeometries,&geometries](const std::string_view name, smart_refctd_ptr<const ICPUPolygonGeometry>&& geom)->void
				{
					namedGeometries.emplace_back().name = name;
					geometries.push_back(std::move(geom));
				};

				auto creator = core::make_smart_refctd_ptr<CGeometryCreator>();
				/* TODO: others
				ReferenceObjectCpu {.meta = {.type = OT_CUBE, .name = "Cube Mesh" }, .shadersType = GP_BASIC, .data = gc->createCubeMesh(nbl::core::vector3df(1.f, 1.f, 1.f)) },
				ReferenceObjectCpu {.meta = {.type = OT_SPHERE, .name = "Sphere Mesh" }, .shadersType = GP_BASIC, .data = gc->createSphereMesh(2, 16, 16) },
				ReferenceObjectCpu {.meta = {.type = OT_CYLINDER, .name = "Cylinder Mesh" }, .shadersType = GP_BASIC, .data = gc->createCylinderMesh(2, 2, 20) },
				ReferenceObjectCpu {.meta = {.type = OT_RECTANGLE, .name = "Rectangle Mesh" }, .shadersType = GP_BASIC, .data = gc->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3)) },
				ReferenceObjectCpu {.meta = {.type = OT_DISK, .name = "Disk Mesh" }, .shadersType = GP_BASIC, .data = gc->createDiskMesh(2, 30) },
				ReferenceObjectCpu {.meta = {.type = OT_ARROW, .name = "Arrow Mesh" }, .shadersType = GP_BASIC, .data = gc->createArrowMesh() },
				ReferenceObjectCpu {.meta = {.type = OT_CONE, .name = "Cone Mesh" }, .shadersType = GP_CONE, .data = gc->createConeMesh(2, 3, 10) },
				ReferenceObjectCpu {.meta = {.type = OT_ICOSPHERE, .name = "Icoshpere Mesh" }, .shadersType = GP_ICO, .data = gc->createIcoSphere(1, 3, true) }
				*/
				addGeometry("Cube",creator->createCube({1.f,1.f,1.f}));
				addGeometry("Rectangle",creator->createRectangle({1.5f,3.f}));
				addGeometry("Disk",creator->createDisk(2.f,30));
			}

			// convert the geometries
			{
				auto device = params.utilities->getLogicalDevice();
				smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({.device=device});


				const auto transferFamily = params.transferQueue->getFamilyIndex();

				struct SInputs : CAssetConverter::SInputs
				{
					virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUBuffer* buffer, const CAssetConverter::patch_t<asset::ICPUBuffer>& patch) const
					{
						return sharedBufferOwnership;
					}

					core::vector<uint32_t> sharedBufferOwnership;
				} inputs = {};
				core::vector<CAssetConverter::patch_t<ICPUPolygonGeometry>> patches(geometries.size(),geometryPatch);
				{
					inputs.logger = logger;
					std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = {&geometries.front().get(),geometries.size()};
					std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = patches;
					// set up shared ownership so we don't have to 
					core::unordered_set<uint32_t> families;
					families.insert(transferFamily);
					families.insert(params.addtionalBufferOwnershipFamilies.begin(),params.addtionalBufferOwnershipFamilies.end());
					if (families.size()>1)
					for (const auto fam : families)
						inputs.sharedBufferOwnership.push_back(fam);
				}
				
				// reserve
				auto reservation = converter->reserve(inputs);
				if (!reservation)
				{
					logger->log("Failed to reserve GPU objects for CPU->GPU conversion!",ILogger::ELL_ERROR);
					return nullptr;
				}

				// convert
				{
					auto semaphore = device->createSemaphore(0u);

					constexpr auto MultiBuffering = 2;
					std::array<smart_refctd_ptr<IGPUCommandBuffer>,MultiBuffering> commandBuffers = {};
					{
						auto pool = device->createCommandPool(transferFamily,IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT|IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
						pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,commandBuffers,smart_refctd_ptr<ILogger>(logger));
					}
					commandBuffers.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

					std::array<IQueue::SSubmitInfo::SCommandBufferInfo,MultiBuffering> commandBufferSubmits;
					for (auto i=0; i<MultiBuffering; i++)
						commandBufferSubmits[i].cmdbuf = commandBuffers[i].get();

					SIntendedSubmitInfo transfer = {};
					transfer.queue = params.transferQueue;
					transfer.scratchCommandBuffers = commandBufferSubmits;
					transfer.scratchSemaphore = {
						.semaphore = semaphore.get(),
						.value = 0u,
						.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
					};

					CAssetConverter::SConvertParams cpar = {};
					cpar.utilities = params.utilities;
					cpar.transfer = &transfer;

					// basically it records all data uploads and submits them right away
					auto future = reservation.convert(cpar);
					if (future.copy()!=IQueue::RESULT::SUCCESS)
					{
						logger->log("Failed to await submission feature!", ILogger::ELL_ERROR);
						return nullptr;
					}
				}

				// assign outputs
				{
					auto inIt = reservation.getGPUObjects<ICPUPolygonGeometry>().data();
					for (auto outIt=namedGeometries.begin(); outIt!=namedGeometries.end(); inIt++)
					{
						if (inIt->value)
							(outIt++)->geom = inIt->value;
						else
						{
							logger->log("Failed to convert ICPUPolygonGeometry %s to GPU!",ILogger::ELL_ERROR,outIt->name.data());
							outIt = namedGeometries.erase(outIt);
						}
					}
				}
			}

			return smart_refctd_ptr<CGeometryCreatorScene>(new CGeometryCreatorScene(std::move(namedGeometries)),dont_grab);
		}

		//
		struct SNamedGeometry
		{
			std::string_view name = {};
			core::smart_refctd_ptr<video::IGPUPolygonGeometry> geom;
		};
		std::span<const SNamedGeometry> getGeometries() const {return m_geometries;}

	protected:
		inline CGeometryCreatorScene(core::vector<SNamedGeometry>&& _geometries) : m_geometries(std::move(_geometries)) {}

		core::vector<SNamedGeometry> m_geometries;
};

}
//!


#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/examples/geometry/SPushConstants.hlsl"

// TODO: Arek bring back
//#include "nbl/examples/geometry/spirv/builtin/CArchive.h"
//#include "nbl/examples/geometry/spirv/builtin/builtinResources.h"


namespace nbl::examples
{

class CSimpleDebugRenderer final : public core::IReferenceCounted
{
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
		static inline core::smart_refctd_ptr<CSimpleDebugRenderer> create(video::IGPURenderpass* renderpass, const uint32_t subpassIX, const CGeometryCreatorScene* scene)
		{
			EXPOSE_NABLA_NAMESPACES;

			if (!renderpass)
				return nullptr;
			auto device = const_cast<ILogicalDevice*>(renderpass->getOriginDevice());
			auto logger = device->getLogger();

			if (!scene)
				return nullptr;
			const auto namedGeoms = scene->getGeometries();
			if (namedGeoms.empty())
				return nullptr;

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

			//
			const SPushConstantRange ranges[] = {{
				.stageFlags = hlsl::ShaderStage::ESS_VERTEX,
				.offset = 0,
				.size = sizeof(SInstance::SPushConstants),
			}};
			init.layout = device->createPipelineLayout(ranges,smart_refctd_ptr<const IGPUDescriptorSetLayout>(init.ds->getLayout()));

			// TODO: Load Shaders and Create Pipelines
			{
				//
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
};

#undef EXPOSE_NABLA_NAMESPACES
#if 0
class ResourceBuilder
{
private:

	bool createShaders()
	{
		EXPOSE_NABLA_NAMESPACES();

		auto createShader = [&]<StringLiteral virtualPath>(IShader::E_SHADER_STAGE stage, smart_refctd_ptr<typename Types::shader_t>& outShader) -> smart_refctd_ptr<typename Types::shader_t>
		{
			// TODO: use SPIRV loader & our ::system ns to get those cpu shaders, do not create myself (shit I forgot it exists)

			const SBuiltinFile& in = ::geometry::creator::spirv::builtin::get_resource<virtualPath>();
			const auto buffer = ICPUBuffer::create({ { in.size }, (void*)in.contents, core::getNullMemoryResource() }, adopt_memory);
			auto shader = make_smart_refctd_ptr<ICPUShader>(smart_refctd_ptr(buffer), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, ""); // must create cpu instance regardless underlying type

			if constexpr (withAssetConverter)
			{
				buffer->setContentHash(buffer->computeContentHash());
				outShader = std::move(shader);
			}

			return outShader;
		};

		typename ResourcesBundleScratch::Shaders& basic = scratch.shaders[GeometriesCpu::GP_BASIC];
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.vertex.spv") > (IShader::E_SHADER_STAGE::ESS_VERTEX, basic.vertex);
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (IShader::E_SHADER_STAGE::ESS_FRAGMENT, basic.fragment);

		typename ResourcesBundleScratch::Shaders& cone = scratch.shaders[GeometriesCpu::GP_CONE];
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.cone.vertex.spv") > (IShader::E_SHADER_STAGE::ESS_VERTEX, cone.vertex);
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (IShader::E_SHADER_STAGE::ESS_FRAGMENT, cone.fragment); // note we reuse fragment from basic!

		typename ResourcesBundleScratch::Shaders& ico = scratch.shaders[GeometriesCpu::GP_ICO];
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.ico.vertex.spv") > (IShader::E_SHADER_STAGE::ESS_VERTEX, ico.vertex);
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (IShader::E_SHADER_STAGE::ESS_FRAGMENT, ico.fragment); // note we reuse fragment from basic!
			
		for (const auto& it : scratch.shaders)
		{
			if (!it.vertex || !it.fragment)
			{
				logger->log("Could not create shaders!", ILogger::ELL_ERROR);
				return false;
			}
		}

		return true;
	}

	bool createGeometries()
	{
		EXPOSE_NABLA_NAMESPACES();

		for (uint32_t i = 0; i < geometries.objects.size(); ++i)
		{
			const auto& inGeometry = geometries.objects[i];
			auto& [obj, meta] = scratch.objects[i];

			bool status = true;

			meta.name = inGeometry.meta.name;
			meta.type = inGeometry.meta.type;

			struct
			{
				SBlendParams blend;
				SRasterizationParams rasterization;
				typename Types::graphics_pipeline_t::SCreationParams pipeline;
			} params;
				
			{
				params.blend.logicOp = ELO_NO_OP;

				auto& b = params.blend.blendParams[0];
				b.srcColorFactor = EBF_SRC_ALPHA;
				b.dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;
				b.colorBlendOp = EBO_ADD;
				b.srcAlphaFactor = EBF_SRC_ALPHA;
				b.dstAlphaFactor = EBF_SRC_ALPHA;
				b.alphaBlendOp = EBO_ADD;
				b.colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);
			}

			params.rasterization.faceCullingMode = EFCM_NONE;
			{
				const typename Types::shader_t::SSpecInfo info [] =
				{
					{.entryPoint = "VSMain", .shader = scratch.shaders[inGeometry.shadersType].vertex.get() },
					{.entryPoint = "PSMain", .shader = scratch.shaders[inGeometry.shadersType].fragment.get() }
				};

				params.pipeline.layout = scratch.pipelineLayout.get();
				params.pipeline.shaders = info;
				params.pipeline.renderpass = scratch.renderpass.get();
				params.pipeline.cached = { .vertexInput = inGeometry.data.inputParams, .primitiveAssembly = inGeometry.data.assemblyParams, .rasterization = params.rasterization, .blend = params.blend, .subpassIx = 0u };

				obj.indexCount = inGeometry.data.indexCount;
				obj.indexType = inGeometry.data.indexType;

				// TODO: cache pipeline & try lookup for existing one first maybe

				// similar issue like with shaders again, in this case gpu contructor allows for extra cache parameters + there is no constructor you can use to fire make_smart_refctd_ptr yourself for cpu
				if constexpr (withAssetConverter)
					obj.pipeline = ICPUGraphicsPipeline::create(params.pipeline);
				else
				{
					const std::array<const IGPUGraphicsPipeline::SCreationParams,1> info = { { params.pipeline } };
					utilities->getLogicalDevice()->createGraphicsPipelines(nullptr, info, &obj.pipeline);
				}

				if (!obj.pipeline)
				{
					logger->log("Could not create graphics pipeline for [%s] object!", ILogger::ELL_ERROR, meta.name.data());
					status = false;
				}

				// object buffers
				auto createVIBuffers = [&]() -> bool
				{
					using ibuffer_t = ::nbl::asset::IBuffer; // seems to be ambigous, both asset & core namespaces has IBuffer

					// note: similar issue like with shaders, this time with cpu-gpu constructors differing in arguments
					auto vBuffer = smart_refctd_ptr(inGeometry.data.bindings[0].buffer); // no offset
					constexpr static auto VERTEX_USAGE = bitflag(ibuffer_t::EUF_VERTEX_BUFFER_BIT) | ibuffer_t::EUF_TRANSFER_DST_BIT | ibuffer_t::EUF_INLINE_UPDATE_VIA_CMDBUF;
					obj.bindings.vertex.offset = 0u;
						
					auto iBuffer = smart_refctd_ptr(inGeometry.data.indexBuffer.buffer); // no offset
					constexpr static auto INDEX_USAGE = bitflag(ibuffer_t::EUF_INDEX_BUFFER_BIT) | ibuffer_t::EUF_VERTEX_BUFFER_BIT | ibuffer_t::EUF_TRANSFER_DST_BIT | ibuffer_t::EUF_INLINE_UPDATE_VIA_CMDBUF;
					obj.bindings.index.offset = 0u;

					if constexpr (withAssetConverter)
					{
						if (!vBuffer)
							return false;

						vBuffer->addUsageFlags(VERTEX_USAGE);
						vBuffer->setContentHash(vBuffer->computeContentHash());
						obj.bindings.vertex = { .offset = 0u, .buffer = vBuffer };

						if (inGeometry.data.indexType != EIT_UNKNOWN)
							if (iBuffer)
							{
								iBuffer->addUsageFlags(INDEX_USAGE);
								iBuffer->setContentHash(iBuffer->computeContentHash());
							}
							else
								return false;

						obj.bindings.index = { .offset = 0u, .buffer = iBuffer };
					}
					else
					{
						auto vertexBuffer = utilities->getLogicalDevice()->createBuffer(IGPUBuffer::SCreationParams({ .size = vBuffer->getSize(), .usage = VERTEX_USAGE }));
						auto indexBuffer = iBuffer ? utilities->getLogicalDevice()->createBuffer(IGPUBuffer::SCreationParams({ .size = iBuffer->getSize(), .usage = INDEX_USAGE })) : nullptr;

						if (!vertexBuffer)
							return false;

						if (inGeometry.data.indexType != EIT_UNKNOWN)
							if (!indexBuffer)
								return false;

						const auto mask = utilities->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
						for (auto it : { vertexBuffer , indexBuffer })
						{
							if (it)
							{
								auto reqs = it->getMemoryReqs();
								reqs.memoryTypeBits &= mask;

								utilities->getLogicalDevice()->allocate(reqs, it.get());
							}
						}

						// record transfer uploads
						obj.bindings.vertex = { .offset = 0u, .buffer = std::move(vertexBuffer) };
						{
							const SBufferRange<IGPUBuffer> range = { .offset = obj.bindings.vertex.offset, .size = obj.bindings.vertex.buffer->getSize(), .buffer = obj.bindings.vertex.buffer };
							if (!commandBuffer->updateBuffer(range, vBuffer->getPointer()))
							{
								logger->log("Could not record vertex buffer transfer upload for [%s] object!", ILogger::ELL_ERROR, meta.name.data());
								status = false;
							}
						}
						obj.bindings.index = { .offset = 0u, .buffer = std::move(indexBuffer) };
						{
							if (iBuffer)
							{
								const SBufferRange<IGPUBuffer> range = { .offset = obj.bindings.index.offset, .size = obj.bindings.index.buffer->getSize(), .buffer = obj.bindings.index.buffer };

								if (!commandBuffer->updateBuffer(range, iBuffer->getPointer()))
								{
									logger->log("Could not record index buffer transfer upload for [%s] object!", ILogger::ELL_ERROR, meta.name.data());
									status = false;
								}
							}
						}
					}
						
					return true;
				};

				if (!createVIBuffers())
				{
					logger->log("Could not create buffers for [%s] object!", ILogger::ELL_ERROR, meta.name.data());
					status = false;
				}

				if (!status)
				{
					logger->log("[%s] object will not be created!", ILogger::ELL_ERROR, meta.name.data());

					obj.bindings.vertex = {};
					obj.bindings.index = {};
					obj.indexCount = 0u;
					obj.indexType = E_INDEX_TYPE::EIT_UNKNOWN;
					obj.pipeline = nullptr;

					continue;
				}
			}
		}

		return true;
	}


	struct GeometriesCpu
	{
		enum GeometryShader
		{
			GP_BASIC = 0,
			GP_CONE,
			GP_ICO,

			GP_COUNT
		};


	};

		struct Shaders
		{
			nbl::core::smart_refctd_ptr<typename Types::shader_t> vertex = nullptr, fragment = nullptr;
		};

		std::array<Shaders, GeometriesCpu::GP_COUNT> shaders;
};
#endif

}
#endif