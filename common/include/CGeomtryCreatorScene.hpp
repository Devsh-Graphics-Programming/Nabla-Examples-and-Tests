#ifndef _NBL_GEOMETRY_CREATOR_SCENE_H_INCLUDED_
#define _NBL_GEOMETRY_CREATOR_SCENE_H_INCLUDED_

#include <nabla.h>

#include "nbl/asset/utils/CGeometryCreator.h"
#include "SBasicViewParameters.hlsl"
#include "geometry/creator/spirv/builtin/CArchive.h"
#include "geometry/creator/spirv/builtin/builtinResources.h"

namespace nbl::scene::geometrycreator
{
#define EXPOSE_NABLA_NAMESPACES() using namespace nbl; \
using namespace core; \
using namespace asset; \
using namespace video; \
using namespace scene; \
using namespace system

struct Traits
{
	static constexpr auto DefaultFramebufferW = 1280u, DefaultFramebufferH = 720u;
	static constexpr auto ColorFboAttachmentFormat = nbl::asset::EF_R8G8B8A8_SRGB, DepthFboAttachmentFormat = nbl::asset::EF_D32_SFLOAT;
	static constexpr auto Samples = nbl::video::IGPUImage::ESCF_1_BIT;
	static constexpr nbl::video::IGPUCommandBuffer::SClearColorValue clearColor = { .float32 = {0.f,0.f,0.f,1.f} };
	static constexpr nbl::video::IGPUCommandBuffer::SClearDepthStencilValue clearDepth = { .depth = 0.f };
};

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
	OT_UNKNOWN = std::numeric_limits<uint8_t>::max()
};

struct ObjectMeta
{
	ObjectType type = OT_UNKNOWN;
	std::string_view name = "Unknown";
};

struct ResourcesBundle : public virtual nbl::core::IReferenceCounted
{
	struct ReferenceObject
	{
		struct Bindings
		{
			nbl::asset::SBufferBinding<nbl::video::IGPUBuffer> vertex, index;
		};

		nbl::core::smart_refctd_ptr<nbl::video::IGPUGraphicsPipeline> pipeline = nullptr;

		Bindings bindings;
		nbl::asset::E_INDEX_TYPE indexType = nbl::asset::E_INDEX_TYPE::EIT_UNKNOWN;
		uint32_t indexCount = {};
	};

	using ReferenceDrawHook = std::pair<ReferenceObject, ObjectMeta>;

	std::array<ReferenceDrawHook, OT_COUNT> objects;
	nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> dsLayout;

	static inline nbl::core::smart_refctd_ptr<ResourcesBundle> create(nbl::video::ILogicalDevice* const device, nbl::system::ILogger* const logger, nbl::video::CThreadSafeQueueAdapter* transferCapableQueue, const nbl::asset::IGeometryCreator* gc)
	{
		EXPOSE_NABLA_NAMESPACES();

		if (!device)
			return nullptr;

		if (!logger)
			return nullptr;

		if (!transferCapableQueue)
			return nullptr;

		auto cPool = device->createCommandPool(transferCapableQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

		if (!cPool)
		{
			logger->log("Couldn't create command pool!", ILogger::ELL_ERROR);
			return nullptr;
		}

		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmd;

		if (!cPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmd , 1 }))
		{
			logger->log("Couldn't create command buffer!", ILogger::ELL_ERROR);
			return nullptr;
		}

		if (!cmd)
			return nullptr;

		cmd->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cmd->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cmd->beginDebugMarker("GC Scene resources upload buffer");

		//! descriptor set layout
		
		IGPUDescriptorSetLayout::SBinding bindings[] =
		{
			{
				.binding = 0u,
				.type = IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT,
				.count = 1u,
			}
		};

		auto dsLayout = device->createDescriptorSetLayout(bindings);

		if (!dsLayout)
		{
			logger->log("Could not descriptor set layout!", ILogger::ELL_ERROR);
			return nullptr;
		}

		//! pipeline layout
		
		auto pipelineLayout = device->createPipelineLayout({}, nullptr, smart_refctd_ptr(dsLayout), nullptr, nullptr);

		if (!pipelineLayout)
		{
			logger->log("Could not create pipeline layout!", ILogger::ELL_ERROR);
			return nullptr;
		}

		//! renderpass
		
		static constexpr IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] =
		{
			{
				{
					{
						.format = Traits::ColorFboAttachmentFormat,
						.samples = Traits::Samples,
						.mayAlias = false
					},
					/* .loadOp = */ IGPURenderpass::LOAD_OP::CLEAR,
					/* .storeOp = */ IGPURenderpass::STORE_OP::STORE,
					/* .initialLayout = */ IGPUImage::LAYOUT::UNDEFINED,
					/* .finalLayout = */ IGPUImage::LAYOUT::READ_ONLY_OPTIMAL
				}
			},
			IGPURenderpass::SCreationParams::ColorAttachmentsEnd
		};

		static constexpr IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] =
		{
			{
				{
					{
						.format = Traits::DepthFboAttachmentFormat,
						.samples = Traits::Samples,
						.mayAlias = false
					},
					/* .loadOp = */ {IGPURenderpass::LOAD_OP::CLEAR},
					/* .storeOp = */ {IGPURenderpass::STORE_OP::STORE},
					/* .initialLayout = */ {IGPUImage::LAYOUT::UNDEFINED},
					/* .finalLayout = */ {IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}
				}
			},
			IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd
		};

		IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] =
		{
			{},
			IGPURenderpass::SCreationParams::SubpassesEnd
		};

		subpasses[0].depthStencilAttachment.render = { .attachmentIndex = 0u,.layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL };
		subpasses[0].colorAttachments[0] = { .render = {.attachmentIndex = 0u, .layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL } };

		static constexpr IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] =
		{
			// wipe-transition of Color to ATTACHMENT_OPTIMAL
			{
				.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.dstSubpass = 0,
				.memoryBarrier =
				{
				// 
				.srcStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
				// only write ops, reads can't be made available
				.srcAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT,
				// destination needs to wait as early as possible
				.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
				// because of depth test needing a read and a write
				.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_READ_BIT | ACCESS_FLAGS::COLOR_ATTACHMENT_READ_BIT | ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
			}
			// leave view offsets and flags default
			},
			// color from ATTACHMENT_OPTIMAL to PRESENT_SRC
			{
				.srcSubpass = 0,
				.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.memoryBarrier =
				{
				// last place where the depth can get modified
				.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
				// only write ops, reads can't be made available
				.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
				// 
				.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
				//
				.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT
				// 
				}
			// leave view offsets and flags default
			},
			IGPURenderpass::SCreationParams::DependenciesEnd
		};

		IGPURenderpass::SCreationParams params = {};
		params.colorAttachments = colorAttachments;
		params.depthStencilAttachments = depthAttachments;
		params.subpasses = subpasses;
		params.dependencies = dependencies;

		auto renderpass = device->createRenderpass(params);

		if (!renderpass)
		{
			logger->log("Could not create render pass!", ILogger::ELL_ERROR);
			return nullptr;
		}

		//! shaders
		
		auto createShader = [&]<StringLiteral virtualPath>(IShader::E_SHADER_STAGE stage, smart_refctd_ptr<IGPUShader>& outShader) -> smart_refctd_ptr<IGPUShader>
		{
			const SBuiltinFile& in = ::geometry::creator::spirv::builtin::get_resource<virtualPath>();
			const auto buffer = ICPUBuffer::create({ { in.size }, (void*)in.contents, core::getNullMemoryResource() }, adopt_memory);
			auto shader = make_smart_refctd_ptr<ICPUShader>(smart_refctd_ptr(buffer), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, "");

			outShader = device->createShader(shader.get());

			return outShader;
		};

		struct GeometriesCpu
		{
			enum GeometryShader
			{
				GP_BASIC = 0,
				GP_CONE,
				GP_ICO,

				GP_COUNT
			};

			struct ReferenceObjectCpu
			{
				ObjectMeta meta;
				GeometryShader shadersType;
				nbl::asset::CGeometryCreator::return_type data;
			};

			GeometriesCpu(const nbl::asset::IGeometryCreator* _gc)
				: gc(_gc),
				objects
				({
					ReferenceObjectCpu {.meta = {.type = OT_CUBE, .name = "Cube Mesh" }, .shadersType = GP_BASIC, .data = gc->createCubeMesh(nbl::core::vector3df(1.f, 1.f, 1.f)) },
					ReferenceObjectCpu {.meta = {.type = OT_SPHERE, .name = "Sphere Mesh" }, .shadersType = GP_BASIC, .data = gc->createSphereMesh(2, 16, 16) },
					ReferenceObjectCpu {.meta = {.type = OT_CYLINDER, .name = "Cylinder Mesh" }, .shadersType = GP_BASIC, .data = gc->createCylinderMesh(2, 2, 20) },
					ReferenceObjectCpu {.meta = {.type = OT_RECTANGLE, .name = "Rectangle Mesh" }, .shadersType = GP_BASIC, .data = gc->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3)) },
					ReferenceObjectCpu {.meta = {.type = OT_DISK, .name = "Disk Mesh" }, .shadersType = GP_BASIC, .data = gc->createDiskMesh(2, 30) },
					ReferenceObjectCpu {.meta = {.type = OT_ARROW, .name = "Arrow Mesh" }, .shadersType = GP_BASIC, .data = gc->createArrowMesh() },
					ReferenceObjectCpu {.meta = {.type = OT_CONE, .name = "Cone Mesh" }, .shadersType = GP_CONE, .data = gc->createConeMesh(2, 3, 10) },
					ReferenceObjectCpu {.meta = {.type = OT_ICOSPHERE, .name = "Icoshpere Mesh" }, .shadersType = GP_ICO, .data = gc->createIcoSphere(1, 3, true) }
					})
			{
				gc = nullptr; // one shot
			}

		private:
			const nbl::asset::IGeometryCreator* gc;

		public:
			const std::array<ReferenceObjectCpu, OT_COUNT> objects;
		};

		struct Shaders
		{
			nbl::core::smart_refctd_ptr<IGPUShader> vertex = nullptr, fragment = nullptr;
		};

		GeometriesCpu geometries(gc);
		std::array<Shaders, GeometriesCpu::GP_COUNT> shaders;

		auto& basic = shaders[GeometriesCpu::GP_BASIC];
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.vertex.spv") > (IShader::E_SHADER_STAGE::ESS_VERTEX, basic.vertex);
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (IShader::E_SHADER_STAGE::ESS_FRAGMENT, basic.fragment);

		auto& cone = shaders[GeometriesCpu::GP_CONE];
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.cone.vertex.spv") > (IShader::E_SHADER_STAGE::ESS_VERTEX, cone.vertex);
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (IShader::E_SHADER_STAGE::ESS_FRAGMENT, cone.fragment); // note we reuse fragment from basic!

		auto& ico = shaders[GeometriesCpu::GP_ICO];
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.ico.vertex.spv") > (IShader::E_SHADER_STAGE::ESS_VERTEX, ico.vertex);
		createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (IShader::E_SHADER_STAGE::ESS_FRAGMENT, ico.fragment); // note we reuse fragment from basic!

		for (const auto& it : shaders)
		{
			if (!it.vertex || !it.fragment)
			{
				logger->log("Could not create a shader!", ILogger::ELL_ERROR);
				return nullptr;
			}
		}

		// geometries

		auto output = make_smart_refctd_ptr<ResourcesBundle>();
		output->renderpass = smart_refctd_ptr(renderpass);
		output->dsLayout = smart_refctd_ptr(dsLayout);

		for (uint32_t i = 0; i < geometries.objects.size(); ++i)
		{
			const auto& inGeometry = geometries.objects[i];
			auto& [obj, meta] = output->objects[i];

			meta.name = inGeometry.meta.name;
			meta.type = inGeometry.meta.type;

			struct
			{
				SBlendParams blend;
				SRasterizationParams rasterization;
				IGPUGraphicsPipeline::SCreationParams pipeline;
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
				const IGPUShader::SSpecInfo sInfo [] =
				{
					{.entryPoint = "VSMain", .shader = shaders[inGeometry.shadersType].vertex.get() },
					{.entryPoint = "PSMain", .shader = shaders[inGeometry.shadersType].fragment.get() }
				};

				params.pipeline.layout = pipelineLayout.get();
				params.pipeline.shaders = sInfo;
				params.pipeline.renderpass = renderpass.get();
				params.pipeline.cached = { .vertexInput = inGeometry.data.inputParams, .primitiveAssembly = inGeometry.data.assemblyParams, .rasterization = params.rasterization, .blend = params.blend, .subpassIx = 0u };

				obj.indexCount = inGeometry.data.indexCount;
				obj.indexType = inGeometry.data.indexType;

				const std::array<const IGPUGraphicsPipeline::SCreationParams, 1> pInfo = { { params.pipeline } };
				device->createGraphicsPipelines(nullptr, pInfo, &obj.pipeline);

				if (!obj.pipeline)
				{
					logger->log("Could not create graphics pipeline for [%s] object!", ILogger::ELL_ERROR, meta.name.data());
					return nullptr;
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

					auto vertexBuffer = device->createBuffer(IGPUBuffer::SCreationParams({ .size = vBuffer->getSize(), .usage = VERTEX_USAGE }));
					auto indexBuffer = iBuffer ? device->createBuffer(IGPUBuffer::SCreationParams({ .size = iBuffer->getSize(), .usage = INDEX_USAGE })) : nullptr;

					if (!vertexBuffer)
						return false;

					if (inGeometry.data.indexType != EIT_UNKNOWN)
						if (!indexBuffer)
							return false;

					const auto mask = device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
					for (auto it : { vertexBuffer , indexBuffer })
					{
						if (it)
						{
							auto reqs = it->getMemoryReqs();
							reqs.memoryTypeBits &= mask;

							device->allocate(reqs, it.get());
						}
					}

					// record transfer uploads
					obj.bindings.vertex = { .offset = 0u, .buffer = std::move(vertexBuffer) };
					{
						const SBufferRange<IGPUBuffer> range = { .offset = obj.bindings.vertex.offset, .size = obj.bindings.vertex.buffer->getSize(), .buffer = obj.bindings.vertex.buffer };
						if (!cmd->updateBuffer(range, vBuffer->getPointer()))
						{
							logger->log("Could not record vertex buffer transfer upload for [%s] object!", ILogger::ELL_ERROR, meta.name.data());
							return false;
						}
					}
					obj.bindings.index = { .offset = 0u, .buffer = std::move(indexBuffer) };
					{
						if (iBuffer)
						{
							const SBufferRange<IGPUBuffer> range = { .offset = obj.bindings.index.offset, .size = obj.bindings.index.buffer->getSize(), .buffer = obj.bindings.index.buffer };

							if (!cmd->updateBuffer(range, iBuffer->getPointer()))
							{
								logger->log("Could not record index buffer transfer upload for [%s] object!", ILogger::ELL_ERROR, meta.name.data());
								return false;
							}
						}
					}
						
					return true;
				};

				if (!createVIBuffers())
				{
					logger->log("Could not create buffers for [%s] object!", ILogger::ELL_ERROR, meta.name.data());
					return nullptr;
				}
			}
		}

		cmd->end();
		
		// submit
		{
			std::array<IQueue::SSubmitInfo::SCommandBufferInfo, 1u> commandBuffers = {};
			{
				commandBuffers.front().cmdbuf = cmd.get();
			}

			auto completed = device->createSemaphore(0u);

			std::array<IQueue::SSubmitInfo::SSemaphoreInfo, 1u> signals;
			{
				auto& signal = signals.front();
				signal.value = 1;
				signal.stageMask = bitflag(PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS);
				signal.semaphore = completed.get();
			}

			const IQueue::SSubmitInfo infos[] =
			{
				{
					.waitSemaphores = {},
					.commandBuffers = commandBuffers,
					.signalSemaphores = signals
				}
			};

			if (transferCapableQueue->submit(infos) != IQueue::RESULT::SUCCESS)
			{
				logger->log("Failed to submit transfer upload operations!", ILogger::ELL_ERROR);
				return nullptr;
			}

			const ISemaphore::SWaitInfo info[] =
			{ {
				.semaphore = completed.get(),
				.value = 1
			} };

			device->blockForSemaphores(info);
		}

		return output;
	}
};

struct ObjectInstance
{
	nbl::asset::SBasicViewParameters viewParameters;
	ObjectMeta meta;
};

class CScene final : public nbl::core::IReferenceCounted
{
public:
	ObjectInstance object; // optional TODO: MDI, allow for multiple objects on the scene -> read (*) bellow at private class members

	struct
	{
		static constexpr uint32_t startedValue = 0, finishedValue = 0x45;
		nbl::core::smart_refctd_ptr<nbl::video::ISemaphore> progress;
	} semaphore;

	static inline nbl::core::smart_refctd_ptr<CScene> create(nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> device, nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger, nbl::video::CThreadSafeQueueAdapter* const transferCapableQueue, const nbl::core::smart_refctd_ptr<const ResourcesBundle> resources, const uint32_t framebufferW = Traits::DefaultFramebufferW, const uint32_t framebufferH = Traits::DefaultFramebufferH)
	{
		EXPOSE_NABLA_NAMESPACES();

		if (!device)
			return nullptr;

		if (!logger)
			return nullptr;

		if (!transferCapableQueue)
			return nullptr;

		if (!resources)
			return nullptr;

		// cmd

		auto cPool = device->createCommandPool(transferCapableQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

		if (!cPool)
		{
			logger->log("Couldn't create command pool!", ILogger::ELL_ERROR);
			return nullptr;
		}

		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmd;

		if (!cPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmd , 1 }))
		{
			logger->log("Couldn't create command buffer!", ILogger::ELL_ERROR);
			return nullptr;
		}

		if (!cmd)
			return nullptr;

		// UBO with basic view parameters
		
		using ibuffer_t = ::nbl::asset::IBuffer;
		constexpr static auto UboUsage = bitflag(ibuffer_t::EUF_UNIFORM_BUFFER_BIT) | ibuffer_t::EUF_TRANSFER_DST_BIT | ibuffer_t::EUF_INLINE_UPDATE_VIA_CMDBUF;

		const auto mask = device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
		auto uboBuffer = device->createBuffer(IGPUBuffer::SCreationParams({ .size = sizeof(SBasicViewParameters), .usage = UboUsage }));

		if (!uboBuffer)
			logger->log("Could not create UBO!", ILogger::ELL_ERROR);

		for (auto it : { uboBuffer })
		{
			IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = it->getMemoryReqs();
			reqs.memoryTypeBits &= mask;

			device->allocate(reqs, it.get());
		}

		nbl::asset::SBufferBinding<nbl::video::IGPUBuffer> ubo = { .offset = 0u, .buffer = std::move(uboBuffer) };

		// descriptor set for the resource
		
		const IGPUDescriptorSetLayout* const layouts[] = { resources->dsLayout.get() };
		const uint32_t setCounts[] = { 1u };

		// note descriptor set has back smart pointer to its pool, so we dont need to keep it explicitly
		auto dPool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);

		if (!dPool)
		{
			logger->log("Could not create Descriptor Pool!", ILogger::ELL_ERROR);
			return nullptr;
		}

		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> ds;
		dPool->createDescriptorSets(layouts, &ds);

		if (!ds)
		{
			logger->log("Could not create Descriptor Set!", ILogger::ELL_ERROR);
			return nullptr;
		}

		// write the descriptor set
		{
			// descriptor write ubo
			IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = ds.get();
			write.binding = 0;
			write.arrayElement = 0u;
			write.count = 1u;

			IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = smart_refctd_ptr(ubo.buffer);
				info.info.buffer.offset = ubo.offset;
				info.info.buffer.size = ubo.buffer->getSize();
			}

			write.info = &info;

			if (!device->updateDescriptorSets(1u, &write, 0u, nullptr))
			{
				logger->log("Could not write descriptor set!", ILogger::ELL_ERROR);
				return nullptr;
			}
		}

		// color & depth attachments
		
		auto createImageView = [&]<E_FORMAT format>(smart_refctd_ptr<IGPUImageView>&outView) -> smart_refctd_ptr<IGPUImageView>
		{
			constexpr bool IS_DEPTH = isDepthOrStencilFormat<format>();
			constexpr auto USAGE = [](const bool isDepth)
				{
					bitflag<IGPUImage::E_USAGE_FLAGS> usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT;

					if (!isDepth)
						usage |= IGPUImage::EUF_SAMPLED_BIT;

					return usage;
				}(IS_DEPTH);
			constexpr auto ASPECT = IS_DEPTH ? IImage::E_ASPECT_FLAGS::EAF_DEPTH_BIT : IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			constexpr std::string_view DEBUG_NAME = IS_DEPTH ? "GC Scene Depth Attachment Image" : "GC Scene Color Attachment Image";
			{
				smart_refctd_ptr<IGPUImage> image;
				{
					auto params = IGPUImage::SCreationParams(
						{
							.type = IGPUImage::ET_2D,
							.samples = Traits::Samples,
							.format = format,
							.extent = { framebufferW, framebufferH, 1u },
							.mipLevels = 1u,
							.arrayLayers = 1u,
							.usage = USAGE
						});

					image = device->createImage(std::move(params));
				}

				if (!image)
				{
					logger->log("Could not create image!", ILogger::ELL_ERROR);
				}

				image->setObjectDebugName(DEBUG_NAME.data());

				if (!device->allocate(image->getMemoryReqs(), image.get()).isValid())
				{
					logger->log("Could not allocate memory for an image!", ILogger::ELL_ERROR);
					return nullptr;
				}

				auto params = IGPUImageView::SCreationParams
				({
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = USAGE,
					.image = std::move(image),
					.viewType = IGPUImageView::ET_2D,
					.format = format,
					.subresourceRange = {.aspectMask = ASPECT, .baseMipLevel = 0u, .levelCount = 1u, .baseArrayLayer = 0u, .layerCount = 1u }
					});

				outView = device->createImageView(std::move(params));

				if (!outView)
				{
					logger->log("Could not create image view!", ILogger::ELL_ERROR);
					return nullptr;
				}

				return smart_refctd_ptr(outView);
			}
		};

		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> color, depth;
		const bool allocated = createImageView.template operator() < Traits::ColorFboAttachmentFormat > (color) && createImageView.template operator() < Traits::DepthFboAttachmentFormat > (depth);

		if (!allocated)
		{
			logger->log("Could not allocate frame buffer's attachments!", ILogger::ELL_ERROR);
			return nullptr;
		}

		//! frame buffer
		
		const auto extent = color->getCreationParameters().image->getCreationParameters().extent;

		IGPUFramebuffer::SCreationParams params =
		{
			{
				.renderpass = smart_refctd_ptr<IGPURenderpass>((IGPURenderpass*)resources->renderpass.get()), // NOTE: those creation params are to be corrected & this should take immutable renderpass (smart pointer OK but take const for type)
				.depthStencilAttachments = &depth.get(),
				.colorAttachments = &color.get(),
				.width = extent.width,
				.height = extent.height,
				.layers = 1u
			}
		};

		auto frameBuffer = device->createFramebuffer(std::move(params));

		if (!frameBuffer)
		{
			logger->log("Could not create frame buffer!", ILogger::ELL_ERROR);
			return nullptr;
		}

		auto output = new CScene(smart_refctd_ptr(device), smart_refctd_ptr(logger), smart_refctd_ptr(cmd), smart_refctd_ptr(frameBuffer), smart_refctd_ptr(ds), ubo, smart_refctd_ptr(color), smart_refctd_ptr(depth), smart_refctd_ptr(resources));
		return smart_refctd_ptr<CScene>(output);
	}

	~CScene() {}

	inline void begin()
	{
		EXPOSE_NABLA_NAMESPACES();

		m_commandBuffer->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		m_commandBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		m_commandBuffer->beginDebugMarker("UISampleApp Offline Scene Frame");

		semaphore.progress = m_device->createSemaphore(semaphore.startedValue);
	}

	inline bool record()
	{
		EXPOSE_NABLA_NAMESPACES();
		bool valid = true;

		const struct 
		{
			const uint32_t width, height;
		} fbo = { .width = m_frameBuffer->getCreationParameters().width, .height = m_frameBuffer->getCreationParameters().height };

		SViewport viewport;
		{
			viewport.minDepth = 1.f;
			viewport.maxDepth = 0.f;
			viewport.x = 0u;
			viewport.y = 0u;
			viewport.width = fbo.width;
			viewport.height = fbo.height;
		}

		valid &= m_commandBuffer->setViewport(0u, 1u, &viewport);
		
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = { fbo.width, fbo.height };

		valid &= m_commandBuffer->setScissor(0u, 1u, &scissor);

		const VkRect2D renderArea =
		{
			.offset = { 0,0 },
			.extent = { fbo.width, fbo.height }
		};

		const IGPUCommandBuffer::SRenderpassBeginInfo info =
		{
			.framebuffer = m_frameBuffer.get(),
			.colorClearValues = &Traits::clearColor,
			.depthStencilClearValues = &Traits::clearDepth,
			.renderArea = renderArea
		};

		valid &= m_commandBuffer->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
		valid &= draw(m_commandBuffer.get());
		valid &= m_commandBuffer->endRenderPass();

		return valid;
	}

	inline void end()
	{
		m_commandBuffer->end();
	}

	inline bool submit(nbl::video::CThreadSafeQueueAdapter* queue)
	{
		EXPOSE_NABLA_NAMESPACES();

		if (!queue)
			return false;

		const IQueue::SSubmitInfo::SCommandBufferInfo buffers[] =
		{
			{ .cmdbuf = m_commandBuffer.get() }
		};

		const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = semaphore.progress.get(),.value = semaphore.finishedValue,.stageMask = PIPELINE_STAGE_FLAGS::FRAMEBUFFER_SPACE_BITS} };

		const IQueue::SSubmitInfo infos[] =
		{
			{
				.waitSemaphores = {},
				.commandBuffers = buffers,
				.signalSemaphores = signals
			}
		};

		return queue->submit(infos) == IQueue::RESULT::SUCCESS;
	}

	inline bool update(nbl::video::IGPUCommandBuffer* cmdbuf = nullptr)
	{
		EXPOSE_NABLA_NAMESPACES();

		SBufferRange<IGPUBuffer> range;
		range.buffer = smart_refctd_ptr(m_ubo.buffer);
		range.size = m_ubo.buffer->getSize();

		if(cmdbuf)
			return cmdbuf->updateBuffer(range, &object.viewParameters);

		return m_commandBuffer->updateBuffer(range, &object.viewParameters);
	}

	inline auto getColorAttachment() { return nbl::core::smart_refctd_ptr(m_color); }

private:
	inline bool draw(nbl::video::IGPUCommandBuffer* cmdbuf)
	{
		EXPOSE_NABLA_NAMESPACES();
		bool valid = true;

		const auto& [hook, meta] = m_resources->objects[object.meta.type];
		const auto* rawPipeline = hook.pipeline.get();

		SBufferBinding<const IGPUBuffer> vertex = hook.bindings.vertex, index = hook.bindings.index;

		valid &= cmdbuf->bindGraphicsPipeline(rawPipeline);
		valid &= cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, rawPipeline->getLayout(), 1, 1, &m_ds.get());
		valid &= cmdbuf->bindVertexBuffers(0, 1, &vertex);

		if (index.buffer && hook.indexType != EIT_UNKNOWN)
		{
			valid &= cmdbuf->bindIndexBuffer(index, hook.indexType);
			valid &= cmdbuf->drawIndexed(hook.indexCount, 1, 0, 0, 0);
		}
		else
			valid &= cmdbuf->draw(hook.indexCount, 1, 0, 0);

		return valid;
	}

	CScene(nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> device, nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger, nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffer, nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer> frameBuffer, nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> ds, nbl::asset::SBufferBinding<nbl::video::IGPUBuffer> ubo, nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> color, nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> depth, const nbl::core::smart_refctd_ptr<const ResourcesBundle> resources)
		: m_device(nbl::core::smart_refctd_ptr(device)), m_logger(nbl::core::smart_refctd_ptr(logger)), m_commandBuffer(nbl::core::smart_refctd_ptr(commandBuffer)), m_frameBuffer(nbl::core::smart_refctd_ptr(frameBuffer)), m_ds(nbl::core::smart_refctd_ptr(ds)), m_ubo(ubo), m_color(nbl::core::smart_refctd_ptr(color)), m_depth(nbl::core::smart_refctd_ptr(depth)), m_resources(nbl::core::smart_refctd_ptr(resources)) {}

	nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> m_device;
	nbl::core::smart_refctd_ptr<nbl::system::ILogger> m_logger;

	//! (*) I still make an assumption we have only one object on the scene,
	//! I'm not going to make it ext-like and go with MDI + streaming buffer now, 
	//! but I want it to be easy to spam multiple instances of this class to have many 
	//! frame buffers we can render too given resource geometry buffers with Traits constraints
	//! 
	//! optional TODO: make it ImGUI-ext-like -> renderpass as creation input, ST buffer, MDI, ds outside

	nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_commandBuffer = nullptr;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer> m_frameBuffer = nullptr;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_ds = nullptr;
	nbl::asset::SBufferBinding<nbl::video::IGPUBuffer> m_ubo = {};
	nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_color = nullptr, m_depth = nullptr;

	const nbl::core::smart_refctd_ptr<const ResourcesBundle> m_resources;
};

} // nbl::scene::geometrycreator

#endif // _NBL_GEOMETRY_CREATOR_SCENE_H_INCLUDED_