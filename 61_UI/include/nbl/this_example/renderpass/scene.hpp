#ifndef __NBL_THIS_EXAMPLE_SCENE_H_INCLUDED__
#define __NBL_THIS_EXAMPLE_SCENE_H_INCLUDED__

#include <nabla.h>

#include "nbl/asset/utils/CGeometryCreator.h"
#include "SBasicViewParameters.hlsl"
#include "CAssetConverter.h"
#include "geometry/creator/spirv/builtin/CArchive.h"
#include "geometry/creator/spirv/builtin/builtinResources.h"

enum E_OBJECT_TYPE : uint8_t
{
	EOT_CUBE,
	EOT_SPHERE,
	EOT_CYLINDER,
	EOT_RECTANGLE,
	EOT_DISK,
	EOT_ARROW,
	EOT_CONE,
	EOT_ICOSPHERE,

	EOT_COUNT,
	EOT_UNKNOWN = std::numeric_limits<uint8_t>::max()
};

struct OBJECT_META
{
	E_OBJECT_TYPE type = EOT_UNKNOWN;
	std::string_view name = "Unknown";
};

NBL_CONSTEXPR_STATIC_INLINE struct CLEAR_VALUES
{
	nbl::video::IGPUCommandBuffer::SClearColorValue color = { .float32 = {0.f,0.f,0.f,1.f} };
	nbl::video::IGPUCommandBuffer::SClearDepthStencilValue depth = { .depth = 0.f };
} clear;

template<typename T, typename... Types>
concept _implIsResourceTypeC = (std::same_as<T, Types> || ...);

template<typename T, typename Types>
concept RESOURCE_TYPE_CONCEPT = _implIsResourceTypeC<T, typename Types::DESCRIPTOR_SET_LAYOUT, typename Types::PIPELINE_LAYOUT, typename Types::RENDERPASS, typename Types::IMAGE_VIEW, typename Types::IMAGE, typename Types::BUFFER, typename Types::SHADER, typename Types::GRAPHICS_PIPELINE>;

#define TYPES_IMPL_BOILERPLATE(WITH_CONVERTER) struct TYPES \
{ \
	using DESCRIPTOR_SET_LAYOUT = std::conditional_t<WITH_CONVERTER, nbl::asset::ICPUDescriptorSetLayout, nbl::video::IGPUDescriptorSetLayout>; \
	using PIPELINE_LAYOUT = std::conditional_t<WITH_CONVERTER, nbl::asset::ICPUPipelineLayout, nbl::video::IGPUPipelineLayout>; \
	using RENDERPASS = std::conditional_t<WITH_CONVERTER, nbl::asset::ICPURenderpass, nbl::video::IGPURenderpass>; \
	using IMAGE_VIEW = std::conditional_t<WITH_CONVERTER, nbl::asset::ICPUImageView, nbl::video::IGPUImageView>; \
	using IMAGE = std::conditional_t<WITH_CONVERTER, nbl::asset::ICPUImage, nbl::video::IGPUImage>; \
	using BUFFER = std::conditional_t<WITH_CONVERTER, nbl::asset::ICPUBuffer, nbl::video::IGPUBuffer>; \
	using SHADER = std::conditional_t<WITH_CONVERTER, nbl::asset::ICPUShader, nbl::video::IGPUShader>; \
	using GRAPHICS_PIPELINE = std::conditional_t<WITH_CONVERTER, nbl::asset::ICPUGraphicsPipeline, nbl::video::IGPUGraphicsPipeline>; \
}

template<bool withAssetConverter>
struct RESOURCES_BUNDLE_BASE
{
	TYPES_IMPL_BOILERPLATE(withAssetConverter);

	struct BUFFER_AND_USAGE
	{
		nbl::asset::SBufferBinding<typename TYPES::BUFFER> binding;
		nbl::core::bitflag<nbl::asset::IBuffer::E_USAGE_FLAGS> usage = nbl::asset::IBuffer::EUF_NONE;
	};

	struct REFERENCE_OBJECT
	{
		struct BUFFERS
		{
			BUFFER_AND_USAGE vertex, index;
		};

		nbl::core::smart_refctd_ptr<typename TYPES::GRAPHICS_PIPELINE> pipeline = nullptr;

		BUFFERS buffers;
		nbl::asset::E_INDEX_TYPE indexType = nbl::asset::E_INDEX_TYPE::EIT_UNKNOWN;
		uint32_t indexCount = {};
	};

	using REFERENCE_DRAW_HOOK = std::pair<REFERENCE_OBJECT, OBJECT_META>;

	nbl::core::smart_refctd_ptr<typename TYPES::RENDERPASS> renderpass;
	std::array<REFERENCE_DRAW_HOOK, EOT_COUNT> objects;
	BUFFER_AND_USAGE ubo;

	struct
	{
		nbl::core::smart_refctd_ptr<typename TYPES::IMAGE_VIEW> color, depth;
	} attachments;
};

struct RESOURCES_BUNDLE : public RESOURCES_BUNDLE_BASE<false>
{
	using BASE_T = RESOURCES_BUNDLE_BASE<false>;

	nbl::core::smart_refctd_ptr<nbl::video::IDescriptorPool> descriptorPool;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> descriptorSet;
};

#define EXPOSE_NABLA_NAMESPACES() using namespace nbl; \
using namespace core; \
using namespace asset; \
using namespace video; \
using namespace scene; \
using namespace system

template<bool withAssetConverter>
class RESOURCES_BUILDER
{
public:
	TYPES_IMPL_BOILERPLATE(withAssetConverter);

	RESOURCES_BUILDER(nbl::video::ILogicalDevice* const _device, nbl::system::ILogger* const _logger, const nbl::asset::IGeometryCreator* const _geometryCreator)
		: device(_device), logger(_logger), geometryCreator(_geometryCreator)
	{
		assert(device);
		assert(logger);
		assert(geometryCreator);
	}

	inline bool build()
	{
		EXPOSE_NABLA_NAMESPACES();

		// TODO: we could make those params templated with default values like below
		_NBL_STATIC_INLINE_CONSTEXPR auto FRAMEBUFFER_W = 1280u, FRAMEBUFFER_H = 720u;
		_NBL_STATIC_INLINE_CONSTEXPR auto COLOR_FBO_ATTACHMENT_FORMAT = EF_R8G8B8A8_SRGB, DEPTH_FBO_ATTACHMENT_FORMAT = EF_D16_UNORM;
		_NBL_STATIC_INLINE_CONSTEXPR auto SAMPLES = IGPUImage::ESCF_1_BIT;

		// descriptor set layout
		{
			typename TYPES::DESCRIPTOR_SET_LAYOUT::SBinding bindings [] =
			{
				{
					.binding = 0u,
					.type = IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
					.createFlags = TYPES::DESCRIPTOR_SET_LAYOUT::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT,
					.count = 1u,
				}
			};

			scratch.descriptorSetLayout = create<typename TYPES::DESCRIPTOR_SET_LAYOUT>(bindings);

			if (!scratch.descriptorSetLayout)
			{
				logger->log("Could not descriptor set layout!", ILogger::ELL_ERROR);
				return false;
			}
		}

		// pipeline layout
		{
			const std::span<const SPushConstantRange> range = {};

			scratch.pipelineLayout = create<typename TYPES::PIPELINE_LAYOUT>(range, nullptr, smart_refctd_ptr(scratch.descriptorSetLayout), nullptr, nullptr);

			if (!scratch.pipelineLayout)
			{
				logger->log("Could not create pipeline layout!", ILogger::ELL_ERROR);
				return false;
			}
		}
		
		// renderpass
		{
			_NBL_STATIC_INLINE_CONSTEXPR TYPES::RENDERPASS::SCreationParams::SColorAttachmentDescription colorAttachments[] =
			{
				{
					{
						{
							.format = COLOR_FBO_ATTACHMENT_FORMAT,
							.samples = SAMPLES,
							.mayAlias = false
						},
						/* .loadOp = */ TYPES::RENDERPASS::LOAD_OP::CLEAR,
						/* .storeOp = */ TYPES::RENDERPASS::STORE_OP::STORE,
						/* .initialLayout = */ TYPES::IMAGE::LAYOUT::UNDEFINED,
						/* .finalLayout = */ TYPES::IMAGE::LAYOUT::READ_ONLY_OPTIMAL
					}
				},
				TYPES::RENDERPASS::SCreationParams::ColorAttachmentsEnd
			};

			_NBL_STATIC_INLINE_CONSTEXPR TYPES::RENDERPASS::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] =
			{
				{
					{
						{
							.format = DEPTH_FBO_ATTACHMENT_FORMAT,
							.samples = SAMPLES,
							.mayAlias = false
						},
						/* .loadOp = */ {TYPES::RENDERPASS::LOAD_OP::CLEAR},
						/* .storeOp = */ {TYPES::RENDERPASS::STORE_OP::STORE},
						/* .initialLayout = */ {TYPES::IMAGE::LAYOUT::UNDEFINED},
						/* .finalLayout = */ {TYPES::IMAGE::LAYOUT::ATTACHMENT_OPTIMAL}
					}
				},
				TYPES::RENDERPASS::SCreationParams::DepthStencilAttachmentsEnd
			};

			typename TYPES::RENDERPASS::SCreationParams::SSubpassDescription subpasses[] =
			{
				{},
				TYPES::RENDERPASS::SCreationParams::SubpassesEnd
			};

			subpasses[0].depthStencilAttachment.render = { .attachmentIndex = 0u,.layout = TYPES::IMAGE::LAYOUT::ATTACHMENT_OPTIMAL };
			subpasses[0].colorAttachments[0] = { .render = {.attachmentIndex = 0u, .layout = TYPES::IMAGE::LAYOUT::ATTACHMENT_OPTIMAL } };

			_NBL_STATIC_INLINE_CONSTEXPR TYPES::RENDERPASS::SCreationParams::SSubpassDependency dependencies[] =
			{
				// wipe-transition of Color to ATTACHMENT_OPTIMAL
				{
					.srcSubpass = TYPES::RENDERPASS::SCreationParams::SSubpassDependency::External,
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
					.dstSubpass = TYPES::RENDERPASS::SCreationParams::SSubpassDependency::External,
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
				TYPES::RENDERPASS::SCreationParams::DependenciesEnd
			};

			typename TYPES::RENDERPASS::SCreationParams params = {};
			params.colorAttachments = colorAttachments;
			params.depthStencilAttachments = depthAttachments;
			params.subpasses = subpasses;
			params.dependencies = dependencies;

			if constexpr (withAssetConverter)
				scratch.renderpass = ICPURenderpass::create(params);
			else
				scratch.renderpass = create<typename TYPES::RENDERPASS>(params);

			if (!scratch.renderpass)
			{
				logger->log("Could not create render pass!", ILogger::ELL_ERROR);
				return false;
			}
		}

		// frame buffer's attachments
		{
			auto createImageView = [&]<E_FORMAT format>(smart_refctd_ptr<typename TYPES::IMAGE_VIEW>& outView) -> smart_refctd_ptr<typename TYPES::IMAGE_VIEW>
			{
				constexpr bool IS_DEPTH = isDepthOrStencilFormat<format>();
				constexpr auto USAGE = [](const bool isDepth)
				{
					bitflag<TYPES::IMAGE::E_USAGE_FLAGS> usage = TYPES::IMAGE::EUF_RENDER_ATTACHMENT_BIT;

					if (!isDepth)
						usage |= TYPES::IMAGE::EUF_SAMPLED_BIT;

					return usage;
				}(IS_DEPTH);
				constexpr auto ASPECT = IS_DEPTH ? IImage::E_ASPECT_FLAGS::EAF_DEPTH_BIT : IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				constexpr std::string_view DEBUG_NAME = IS_DEPTH ? "UI Scene Depth Attachment Image" : "UI Scene Color Attachment Image";
				{
					smart_refctd_ptr<typename TYPES::IMAGE> image;
					{
						auto params = typename TYPES::IMAGE::SCreationParams(
						{
							.type = TYPES::IMAGE::ET_2D,
							.samples = SAMPLES,
							.format = format,
							.extent = { FRAMEBUFFER_W, FRAMEBUFFER_H, 1u },
							.mipLevels = 1u,
							.arrayLayers = 1u,
							.usage = USAGE
						});

						if constexpr (withAssetConverter)
							image = ICPUImage::create(params);
						else
							image = create<typename TYPES::IMAGE>(std::move(params));
					}

					if (!image)
					{
						logger->log("Could not create image!", ILogger::ELL_ERROR);
						return nullptr;
					}

					if constexpr (!withAssetConverter) // valid only for gpu instance
					{
						image->setObjectDebugName(DEBUG_NAME.data());

						if (!device->allocate(image->getMemoryReqs(), image.get()).isValid())
						{
							logger->log("Could not allocate memory for an image!", ILogger::ELL_ERROR);
							return nullptr;
						}
					}

					auto params = typename TYPES::IMAGE_VIEW::SCreationParams(
					{
						.flags = TYPES::IMAGE_VIEW::ECF_NONE,
						.subUsages = USAGE,
						.image = std::move(image),
						.viewType = TYPES::IMAGE_VIEW::ET_2D,
						.format = format,
						.subresourceRange = { ASPECT, 0u, 1u, 0u, 1u }
					});

					outView = create<typename TYPES::IMAGE_VIEW>(std::move(params));

					if (!outView)
					{
						logger->log("Could not create image view!", ILogger::ELL_ERROR);
						return nullptr;
					}

					return smart_refctd_ptr(outView);
				}
			};

			const bool allocated = createImageView.template operator() < COLOR_FBO_ATTACHMENT_FORMAT > (scratch.attachments.color) && createImageView.template operator() < DEPTH_FBO_ATTACHMENT_FORMAT > (scratch.attachments.depth);

			if (!allocated)
			{
				logger->log("Could not allocate frame buffer's attachments!", ILogger::ELL_ERROR);
				return false;
			}
		}

		// shaders
		{
			auto createShader = [&]<nbl::core::StringLiteral virtualPath>(IShader::E_SHADER_STAGE stage, smart_refctd_ptr<typename TYPES::SHADER>& outShader) -> smart_refctd_ptr<typename TYPES::SHADER>
			{
				const nbl::system::SBuiltinFile& in = ::geometry::creator::spirv::builtin::get_resource<virtualPath>();
				const auto buffer = make_smart_refctd_ptr<CCustomAllocatorCPUBuffer<null_allocator<uint8_t>, true> >(in.size, (void*)in.contents, adopt_memory);
				auto shader = nbl::core::make_smart_refctd_ptr<ICPUShader>(smart_refctd_ptr(buffer), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, ""); // must create cpu instance regardless underlying type

				if constexpr (withAssetConverter)
					outShader = std::move(shader);
				else
					outShader = create<typename TYPES::SHADER>(shader.get()); // note: dependency between cpu object instance & gpu object creation, not sure if its our API design failure or maybe I'm just thinking too much

				return outShader;
			};

			// TODO: return value validation

			typename RESOURCES_BUNDLE_SCRATCH::SHADERS& basic = scratch.shaders[GEOMETRIES_CPU::EGP_BASIC];
			createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.vertex.spv") > (IShader::E_SHADER_STAGE::ESS_VERTEX, basic.vertex);
			createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (IShader::E_SHADER_STAGE::ESS_FRAGMENT, basic.fragment);

			typename RESOURCES_BUNDLE_SCRATCH::SHADERS& cone = scratch.shaders[GEOMETRIES_CPU::EGP_CONE];
			createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.cone.vertex.spv") > (IShader::E_SHADER_STAGE::ESS_VERTEX, cone.vertex);
			createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (IShader::E_SHADER_STAGE::ESS_FRAGMENT, cone.fragment); // note we reuse fragment from basic!

			typename RESOURCES_BUNDLE_SCRATCH::SHADERS& ico = scratch.shaders[GEOMETRIES_CPU::EGP_ICO];
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
		}

		// geometries
		{
			auto geometries = GEOMETRIES_CPU(geometryCreator);

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
					typename TYPES::GRAPHICS_PIPELINE::SCreationParams pipeline;
				} params;
				
				{
					params.blend.logicOp = ELO_NO_OP;

					auto& b = params.blend.blendParams[0];
					b.srcColorFactor = EBF_SRC_ALPHA;//VK_BLEND_FACTOR_SRC_ALPHA;
					b.dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
					b.colorBlendOp = EBO_ADD;//VK_BLEND_OP_ADD;
					b.srcAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
					b.dstAlphaFactor = EBF_ZERO;//VK_BLEND_FACTOR_ZERO;
					b.alphaBlendOp = EBO_ADD;//VK_BLEND_OP_ADD;
					b.colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);//VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
				}

				params.rasterization.faceCullingMode = EFCM_NONE;
				{
					const typename TYPES::SHADER::SSpecInfo info [] =
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
						const std::span<const IGPUGraphicsPipeline::SCreationParams> info = { { params.pipeline } };
						create<typename TYPES::GRAPHICS_PIPELINE>(nullptr, info, &obj.pipeline);
					}

					if (!obj.pipeline)
					{
						logger->log("Could not create graphics pipeline for [%s] object!", ILogger::ELL_ERROR, meta.name.data());
						status = false;
					}

					// object buffers
					auto createVIBuffers = [&]() -> bool
					{
						using IBUFFER = nbl::asset::IBuffer; // seems to be ambigous, both asset & core namespaces has IBuffer

						// note: similar issue like with shaders, this time with cpu-gpu constructors differing in arguments
						auto vBuffer = smart_refctd_ptr(inGeometry.data.bindings[0].buffer); // no offset
						obj.buffers.vertex.usage = bitflag(IBUFFER::EUF_VERTEX_BUFFER_BIT) | IBUFFER::EUF_TRANSFER_DST_BIT | IBUFFER::EUF_INLINE_UPDATE_VIA_CMDBUF;
						obj.buffers.vertex.binding.offset = 0u;
						
						auto iBuffer = smart_refctd_ptr(inGeometry.data.indexBuffer.buffer); // no offset
						obj.buffers.index.usage = bitflag(IBUFFER::EUF_INDEX_BUFFER_BIT) | IBUFFER::EUF_VERTEX_BUFFER_BIT | IBUFFER::EUF_TRANSFER_DST_BIT | IBUFFER::EUF_INLINE_UPDATE_VIA_CMDBUF;
						obj.buffers.index.binding.offset = 0u;

						if constexpr (withAssetConverter)
						{
							obj.buffers.vertex.binding = { .offset = 0u, .buffer = vBuffer };
							obj.buffers.index.binding = { .offset = 0u, .buffer = iBuffer };
						}
						else
						{
							auto vertexBuffer = create<typename TYPES::BUFFER>(typename TYPES::BUFFER::SCreationParams({ .size = vBuffer->getSize(), .usage = obj.buffers.vertex.usage }));
							auto indexBuffer = iBuffer ? create<typename TYPES::BUFFER>(typename TYPES::BUFFER::SCreationParams({ .size = iBuffer->getSize(), .usage = obj.buffers.index.usage })) : nullptr;

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

							auto fillGPUBuffer = [&logger = logger](smart_refctd_ptr<ICPUBuffer> cBuffer, smart_refctd_ptr<IGPUBuffer> gBuffer)
							{
								auto binding = gBuffer->getBoundMemory();

								if (!binding.memory->map({ 0ull, binding.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
								{
									logger->log("Could not map device memory", ILogger::ELL_ERROR);
									return false;
								}

								if (!binding.memory->isCurrentlyMapped())
								{
									logger->log("Buffer memory is not mapped!", system::ILogger::ELL_ERROR);
									return false;
								}

								auto* mPointer = binding.memory->getMappedPointer();
								memcpy(mPointer, cBuffer->getPointer(), gBuffer->getSize());
								binding.memory->unmap();

								return true;
							};

							if (!fillGPUBuffer(vBuffer, vertexBuffer))
								return false;

							if (indexBuffer)
								if (!fillGPUBuffer(iBuffer, indexBuffer))
									return false;

							obj.buffers.vertex.binding = { .offset = 0u, .buffer = std::move(vertexBuffer) };
							obj.buffers.index.binding = { .offset = 0u, .buffer = std::move(indexBuffer) };
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

						obj.buffers.vertex = {};
						obj.buffers.index = {};
						obj.indexCount = 0u;
						obj.indexType = E_INDEX_TYPE::EIT_UNKNOWN;
						obj.pipeline = nullptr;

						continue;
					}
				}
			}
		}

		// view parameters ubo buffer
		{
			using IBUFFER = nbl::asset::IBuffer; // seems to be ambigous, both asset & core namespaces has IBuffer

			// note: similar issue like with shaders, this time with cpu-gpu constructors differing in arguments
			scratch.ubo.usage = bitflag(IBUFFER::EUF_UNIFORM_BUFFER_BIT) | IBUFFER::EUF_TRANSFER_DST_BIT | IBUFFER::EUF_INLINE_UPDATE_VIA_CMDBUF;

			if constexpr (withAssetConverter)
			{
				auto uboBuffer = make_smart_refctd_ptr<ICPUBuffer>(sizeof(SBasicViewParameters));
				scratch.ubo.binding = { .offset = 0u, .buffer = std::move(uboBuffer) };
			}
			else
			{
				const auto mask = device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

				auto uboBuffer = create<typename TYPES::BUFFER>(typename TYPES::BUFFER::SCreationParams({ .size = sizeof(SBasicViewParameters), .usage = scratch.ubo.usage }));

				for (auto it : { uboBuffer })
				{
					video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = it->getMemoryReqs();
					reqs.memoryTypeBits &= mask;

					device->allocate(reqs, it.get());
				}

				scratch.ubo.binding = { .offset = 0u, .buffer = std::move(uboBuffer) };
			}
		}

		return true;
	}

	inline bool finalize(RESOURCES_BUNDLE& output)
	{
		EXPOSE_NABLA_NAMESPACES();

		if constexpr (withAssetConverter)
		{
			// asset converter - scratch at this point has ready to convert cpu resources
			smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = device,.optimizer = {} });
			CAssetConverter::SInputs inputs = {};
			inputs.logger = logger;

			// TODO: gather ALL resources as inputs for converter, currently testing if converter can convert all of my pipelines

			// gather cpu assets 
			std::array<ICPUGraphicsPipeline*, std::tuple_size<decltype(scratch.objects)>::value> hooks;
			for (uint32_t i = 0u; i < hooks.size(); ++i)
			{
				auto& [reference, meta] = scratch.objects[static_cast<E_OBJECT_TYPE>(i)];
				hooks[i] = reference.pipeline.get();
			}

			// assign to inputs
			{
				std::get<CAssetConverter::SInputs::asset_span_t<ICPUGraphicsPipeline>>(inputs.assets) = hooks;
			}

			// reserve and create the GPU object handles
			CAssetConverter::SResults reservation = converter->reserve(inputs);
			{
				// retrieve the reserved handles
				const auto pipelines = reservation.getGPUObjects<ICPUGraphicsPipeline>();

				for (const auto& gpu : pipelines)
				{
					// anything that fails to be reserved is a nullptr in the span of GPU Objects
					auto pipeline = gpu.value;

					if (!pipeline)
					{
						logger->log("Failed to convert a CPU pipeline to GPU pipeline!", nbl::system::ILogger::ELL_ERROR);
						return false;
					}
				}
			}

			// actual convert recording covering basically all data uploads, but remember for gpu objects to be created you also have to submit the conversion later!
			CAssetConverter::SConvertParams params = {};
			if (!reservation.convert(params))
			{
				logger->log("Failed to record assets conversion!", nbl::system::ILogger::ELL_ERROR); // TODO: to check, so it submits something or just records? 
				return false;
			}
			
			// `autoSubmit` actually returns a pair of ISempahore::future_t one for compute and one for xfer
			// you can store them and delay blocking for conversion to be complete

			if (!params.autoSubmit())
			{
				logger->log("Failed to submit & await conversions!", nbl::system::ILogger::ELL_ERROR);
				return false;
			}
		}
		else
			static_cast<RESOURCES_BUNDLE::BASE_T&>(output) = static_cast<RESOURCES_BUNDLE::BASE_T&>(scratch); // scratch has all ready to use gpu resources with allocated memory, just give the output ownership

		// gpu resources are created at this point, let's create left gpu objects
		
		// descriptor set
		{
			auto* descriptorSetLayout = output.objects.front().first.pipeline->getLayout()->getDescriptorSetLayout(1u); // let's just take any, the layout is shared across all possible pipelines

			const nbl::video::IGPUDescriptorSetLayout* const layouts[] = { nullptr, descriptorSetLayout };
			const uint32_t setCounts[] = { 0u, 1u };

			output.descriptorPool = device->createDescriptorPoolForDSLayouts(nbl::video::IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);

			if (!output.descriptorPool)
			{
				logger->log("Could not create Descriptor Pool!", nbl::system::ILogger::ELL_ERROR);
				return false;
			}

			// I think this could also be created with converter?
			// TODO: once asset converter descriptor set conversion works update accordingly to work with the builder interface
			output.descriptorPool->createDescriptorSets({{descriptorSetLayout}}, &output.descriptorSet);

			if (!output.descriptorSet)
			{
				logger->log("Could not create Descriptor Set!", nbl::system::ILogger::ELL_ERROR);
				return false;
			}
		}

		// write the descriptor set
		{
			// descriptor write ubo
			nbl::video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = output.descriptorSet.get();
			write.binding = 0;
			write.arrayElement = 0u;
			write.count = 1u;

			nbl::video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = nbl::core::smart_refctd_ptr(output.ubo.binding.buffer);
				info.info.buffer.offset = output.ubo.binding.offset;
				info.info.buffer.size = output.ubo.binding.buffer->getSize();
			}

			write.info = &info;

			if(!device->updateDescriptorSets(1u, &write, 0u, nullptr))
			{
				logger->log("Could not write descriptor set!", nbl::system::ILogger::ELL_ERROR);
				return false;
			}
		}

		return true;
	}

private:
	struct GEOMETRIES_CPU
	{
		enum E_GEOMETRY_SHADER
		{
			EGP_BASIC = 0,
			EGP_CONE,
			EGP_ICO,

			EGP_COUNT
		};

		struct REFERENCE_OBJECT_CPU
		{
			OBJECT_META meta;
			E_GEOMETRY_SHADER shadersType;
			nbl::asset::CGeometryCreator::return_type data;
		};

		GEOMETRIES_CPU(const nbl::asset::IGeometryCreator* _gc)
			: gc(_gc),
			objects
			({
				REFERENCE_OBJECT_CPU {.meta = {.type = EOT_CUBE, .name = "Cube Mesh" }, .shadersType = EGP_BASIC, .data = gc->createCubeMesh(nbl::core::vector3df(1.f, 1.f, 1.f)) },
				REFERENCE_OBJECT_CPU {.meta = {.type = EOT_SPHERE, .name = "Sphere Mesh" }, .shadersType = EGP_BASIC, .data = gc->createSphereMesh(2, 16, 16) },
				REFERENCE_OBJECT_CPU {.meta = {.type = EOT_CYLINDER, .name = "Cylinder Mesh" }, .shadersType = EGP_BASIC, .data = gc->createCylinderMesh(2, 2, 20) },
				REFERENCE_OBJECT_CPU {.meta = {.type = EOT_RECTANGLE, .name = "Rectangle Mesh" }, .shadersType = EGP_BASIC, .data = gc->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3)) },
				REFERENCE_OBJECT_CPU {.meta = {.type = EOT_DISK, .name = "Disk Mesh" }, .shadersType = EGP_BASIC, .data = gc->createDiskMesh(2, 30) },
				REFERENCE_OBJECT_CPU {.meta = {.type = EOT_ARROW, .name = "Arrow Mesh" }, .shadersType = EGP_BASIC, .data = gc->createArrowMesh() },
				REFERENCE_OBJECT_CPU {.meta = {.type = EOT_CONE, .name = "Cone Mesh" }, .shadersType = EGP_CONE, .data = gc->createConeMesh(2, 3, 10) },
				REFERENCE_OBJECT_CPU {.meta = {.type = EOT_ICOSPHERE, .name = "Icoshpere Mesh" }, .shadersType = EGP_ICO, .data = gc->createIcoSphere(1, 3, true) }
			})
		{
			gc = nullptr; // one shot
		}

	private:
		const nbl::asset::IGeometryCreator* gc;

	public:
		const std::array<REFERENCE_OBJECT_CPU, EOT_COUNT> objects;
	};

	template<typename T, typename... Args>
	inline nbl::core::smart_refctd_ptr<T> create(Args&&... args) requires RESOURCE_TYPE_CONCEPT<T, TYPES>
	{
		if constexpr (withAssetConverter)
			return nbl::core::make_smart_refctd_ptr<T>(std::forward<Args>(args)...); // TODO: cases where our api requires to call ::create(...) instead directly calling "make smart pointer" could be here handled instead of in .build method
		else
			if constexpr (std::same_as<T, typename TYPES::DESCRIPTOR_SET_LAYOUT>)
				return device->createDescriptorSetLayout(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::PIPELINE_LAYOUT>)
				return device->createPipelineLayout(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::RENDERPASS>)
				return device->createRenderpass(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::IMAGE_VIEW>)
				return device->createImageView(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::IMAGE>)
				return device->createImage(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::BUFFER>)
				return device->createBuffer(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::SHADER>)
				return device->createShader(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::GRAPHICS_PIPELINE>)
			{
				bool status = device->createGraphicsPipelines(std::forward<Args>(args)...);
				return nullptr; // I assume caller with use output from forwarded args, another inconsistency in our api imho
			}
			else
				return nullptr; // TODO: should static assert
	}

	using RESOURCES_BUNDLE_BASE_T = RESOURCES_BUNDLE_BASE<withAssetConverter>;

	struct RESOURCES_BUNDLE_SCRATCH : public RESOURCES_BUNDLE_BASE_T
	{
		using TYPES = RESOURCES_BUNDLE_BASE_T::TYPES;

		RESOURCES_BUNDLE_SCRATCH()
			: RESOURCES_BUNDLE_BASE_T() {}

		struct SHADERS
		{
			nbl::core::smart_refctd_ptr<typename TYPES::SHADER> vertex = nullptr, fragment = nullptr;
		};

		nbl::core::smart_refctd_ptr<typename TYPES::DESCRIPTOR_SET_LAYOUT> descriptorSetLayout;
		nbl::core::smart_refctd_ptr<typename TYPES::PIPELINE_LAYOUT> pipelineLayout;
		std::array<SHADERS, GEOMETRIES_CPU::EGP_COUNT> shaders; //! note, shaders differ from common interface creation rules and cpu-gpu constructors are different, gpu requires cpu shader to be constructed first anyway (so no interface shadered params!) 
	};

	RESOURCES_BUNDLE_SCRATCH scratch;

	nbl::video::ILogicalDevice* const device;
	nbl::system::ILogger* const logger;
	const nbl::asset::IGeometryCreator* const geometryCreator;
};

#undef TYPES_IMPL_BOILERPLATE

/*
	Rendering to offline framebuffer which we don't present, color 
	scene attachment texture we use for second UI renderpass 
	sampling it & rendering into desired GUI area.

	The scene can be created from simple geometry
	using our Geomtry Creator class.
*/

class CScene final : public nbl::core::IReferenceCounted
{
public:

	_NBL_STATIC_INLINE_CONSTEXPR auto NBL_OFFLINE_SCENE_TEX_ID = 1u;

	struct OBJECT_DRAW_HOOK_CPU
	{
		nbl::core::matrix3x4SIMD model;
		nbl::asset::SBasicViewParameters viewParameters;
		OBJECT_META meta;
	};

	OBJECT_DRAW_HOOK_CPU object; // TODO: this could be a vector (to not complicate the example I leave it single object), we would need a better system for drawing then to make only 1 max 2 indirect draw calls (indexed and not indexed objects)

	struct
	{
		const uint32_t startedValue = 0, finishedValue = 0x45;
		nbl::core::smart_refctd_ptr<nbl::video::ISemaphore> progress;
	} semaphore;

	struct CREATE_RESOURCES_DIRECTLY_WITH_DEVICE { using BUILDER = ::RESOURCES_BUILDER<false>; };
	struct CREATE_RESOURCES_WITH_ASSET_CONVERTER { using BUILDER = ::RESOURCES_BUILDER<true>; };

	~CScene() {}

	template<typename CREATE_WITH, typename... Args>
	static auto create(Args&&... args) -> decltype(auto)
	{
		/*
			user should call the constructor's args without last argument explicitly, this is a trick to make constructor templated, 
			eg.create(smart_refctd_ptr(device), smart_refctd_ptr(logger), queuePointer, geometryPointer)
		*/

		auto* scene = new CScene(std::forward<Args>(args)..., CREATE_WITH {});
		nbl::core::smart_refctd_ptr<CScene> smart(scene, nbl::core::dont_grab);

		return smart;
	}

	inline void begin()
	{
		m_commandBuffer->reset(nbl::video::IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		m_commandBuffer->begin(nbl::video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		m_commandBuffer->beginDebugMarker("UISampleApp Offline Scene Frame");

		semaphore.progress = m_device->createSemaphore(semaphore.startedValue);
	}

	inline void record()
	{
		const struct 
		{
			const uint32_t width, height;
		} fbo = { .width = m_frameBuffer->getCreationParameters().width, .height = m_frameBuffer->getCreationParameters().height };

		nbl::asset::SViewport viewport;
		{
			viewport.minDepth = 1.f;
			viewport.maxDepth = 0.f;
			viewport.x = 0u;
			viewport.y = 0u;
			viewport.width = fbo.width;
			viewport.height = fbo.height;
		}

		m_commandBuffer->setViewport(0u, 1u, &viewport);
		
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = { fbo.width, fbo.height };
		m_commandBuffer->setScissor(0u, 1u, &scissor);

		const VkRect2D renderArea =
		{
			.offset = { 0,0 },
			.extent = { fbo.width, fbo.height }
		};

		const nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo info =
		{
			.framebuffer = m_frameBuffer.get(),
			.colorClearValues = &clear.color,
			.depthStencilClearValues = &clear.depth,
			.renderArea = renderArea
		};

		m_commandBuffer->beginRenderPass(info, nbl::video::IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

		const auto& [hook, meta] = resources.objects[object.meta.type];
		auto* rawPipeline = hook.pipeline.get();

		nbl::asset::SBufferBinding<const nbl::video::IGPUBuffer> vertex = hook.buffers.vertex.binding, index = hook.buffers.index.binding;

		m_commandBuffer->bindGraphicsPipeline(rawPipeline);
		m_commandBuffer->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS, rawPipeline->getLayout(), 1, 1, &resources.descriptorSet.get());
		m_commandBuffer->bindVertexBuffers(0, 1, &vertex);

		if (index.buffer && hook.indexType != nbl::asset::EIT_UNKNOWN)
		{
			m_commandBuffer->bindIndexBuffer(index, hook.indexType);
			m_commandBuffer->drawIndexed(hook.indexCount, 1, 0, 0, 0);
		}
		else
			m_commandBuffer->draw(hook.indexCount, 1, 0, 0);

		m_commandBuffer->endRenderPass();
	}

	inline void end()
	{
		m_commandBuffer->end();
	}

	inline bool submit()
	{
		const nbl::video::IQueue::SSubmitInfo::SCommandBufferInfo buffers[] =
		{
			{ .cmdbuf = m_commandBuffer.get() }
		};

		const nbl::video::IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = semaphore.progress.get(),.value = semaphore.finishedValue,.stageMask = nbl::asset::PIPELINE_STAGE_FLAGS::FRAMEBUFFER_SPACE_BITS} };

		const nbl::video::IQueue::SSubmitInfo infos[] =
		{
			{
				.waitSemaphores = {},
				.commandBuffers = buffers,
				.signalSemaphores = signals
			}
		};

		return queue->submit(infos) == nbl::video::IQueue::RESULT::SUCCESS;
	}

	// note: must be updated outside render pass
	inline void update()
	{
		nbl::asset::SBufferRange<nbl::video::IGPUBuffer> range;
		range.buffer = nbl::core::smart_refctd_ptr(resources.ubo.binding.buffer);
		range.size = resources.ubo.binding.buffer->getSize();

		m_commandBuffer->updateBuffer(range, &object.viewParameters);
	}

	inline decltype(auto) getResources()
	{
		return (resources); // note: do not remove "()" - it makes the return type lvalue reference instead of copy 
	}

private:
	template<typename CREATE_WITH = CREATE_RESOURCES_DIRECTLY_WITH_DEVICE> // TODO: enforce constraints, only those 2 above are valid
	CScene(nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> _device, nbl::core::smart_refctd_ptr<nbl::system::ILogger> _logger, nbl::video::CThreadSafeQueueAdapter* _graphicsQueue, const nbl::asset::IGeometryCreator* _geometryCreator, CREATE_WITH createWith = {})
		: m_device(nbl::core::smart_refctd_ptr(_device)), m_logger(nbl::core::smart_refctd_ptr(_logger)), queue(_graphicsQueue)
	{
		using BUILDER = typename CREATE_WITH::BUILDER;

		bool status = createCommandBuffer();
		BUILDER builder(m_device.get(), m_logger.get(), _geometryCreator);

		// gpu resources
		if (builder.build())
		{
			if (!builder.finalize(resources))
				m_logger->log("Could not finalize to gpu objects!", nbl::system::ILogger::ELL_ERROR);
		}
		else
			m_logger->log("Could not build resource objects!", nbl::system::ILogger::ELL_ERROR);

		// frame buffer
		{
			const auto extent = resources.attachments.color->getCreationParameters().image->getCreationParameters().extent;

			nbl::video::IGPUFramebuffer::SCreationParams params =
			{
				{
					.renderpass = nbl::core::smart_refctd_ptr(resources.renderpass),
					.depthStencilAttachments = &resources.attachments.depth.get(),
					.colorAttachments = &resources.attachments.color.get(),
					.width = extent.width,
					.height = extent.height,
					.layers = 1u
				}
			};

			m_frameBuffer = m_device->createFramebuffer(std::move(params));

			if (!m_frameBuffer)
			{
				m_logger->log("Could not create frame buffer!", nbl::system::ILogger::ELL_ERROR);
				return;
			}
		}
	}

	bool createCommandBuffer()
	{
		m_commandPool = m_device->createCommandPool(queue->getFamilyIndex(), nbl::video::IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

		if (!m_commandPool)
		{
			m_logger->log("Couldn't create Command Pool!", nbl::system::ILogger::ELL_ERROR);
			return false;
		}

		if (!m_commandPool->createCommandBuffers(nbl::video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &m_commandBuffer , 1 }))
		{
			m_logger->log("Couldn't create Command Buffer!", nbl::system::ILogger::ELL_ERROR);
			return false;
		}

		return true;
	}

	nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> m_device;
	nbl::core::smart_refctd_ptr<nbl::system::ILogger> m_logger;

	nbl::video::CThreadSafeQueueAdapter* queue;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> m_commandPool; // TODO: decide if we should reuse main app's pool to allocate the cmd
	nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_commandBuffer;

	nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer> m_frameBuffer;

	RESOURCES_BUNDLE resources;
};

#endif // __NBL_THIS_EXAMPLE_SCENE_H_INCLUDED__