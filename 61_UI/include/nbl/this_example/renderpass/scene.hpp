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
	EOT_UNKNOWN = ~0
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

	struct REFERENCE_OBJECT
	{
		nbl::core::smart_refctd_ptr<typename TYPES::GRAPHICS_PIPELINE> pipeline = nullptr;
		nbl::core::smart_refctd_ptr<typename TYPES::BUFFER> vertexBuffer = nullptr, indexBuffer = nullptr;
		nbl::asset::E_INDEX_TYPE indexType = nbl::asset::E_INDEX_TYPE::EIT_UNKNOWN;
		uint32_t indexCount = {};
	};

	using REFERENCE_DRAW_HOOK = std::pair<REFERENCE_OBJECT, OBJECT_META>;

	nbl::core::smart_refctd_ptr<typename TYPES::RENDERPASS> renderpass;
	std::array<REFERENCE_DRAW_HOOK, EOT_COUNT> objects;

	struct
	{
		nbl::core::smart_refctd_ptr<typename TYPES::IMAGE_VIEW> color, depth;
	} attachments;
};

using RESOURCES_BUNDLE = RESOURCES_BUNDLE_BASE<false>;

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
		using namespace nbl;
		using namespace core;
		using namespace asset;
		using namespace video;
		using namespace scene;
		using namespace system;

		// TODO: we could make those params templated with default values like below
		_NBL_STATIC_INLINE_CONSTEXPR auto FRAMEBUFFER_W = 1280u, FRAMEBUFFER_H = 720u;
		_NBL_STATIC_INLINE_CONSTEXPR auto COLOR_FBO_ATTACHMENT_FORMAT = EF_R8G8B8A8_SRGB, DEPTH_FBO_ATTACHMENT_FORMAT = EF_D16_UNORM;
		_NBL_STATIC_INLINE_CONSTEXPR auto SAMPLES = IGPUImage::ESCF_1_BIT;

		// TODO: build all

		// descriptor set layout
		smart_refctd_ptr<typename TYPES::DESCRIPTOR_SET_LAYOUT> descriptorSetLayout;
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

			descriptorSetLayout = create<typename TYPES::DESCRIPTOR_SET_LAYOUT>(bindings);

			if (!descriptorSetLayout)
			{
				logger->log("Could not descriptor set layout!", ILogger::ELL_ERROR);
				return false;
			}
		}

		// pipeline layout
		smart_refctd_ptr<typename TYPES::PIPELINE_LAYOUT> pipelineLayout;
		{
			const std::span<const SPushConstantRange> range = {};

			pipelineLayout = create<typename TYPES::PIPELINE_LAYOUT>(range, nullptr, smart_refctd_ptr(descriptorSetLayout));

			if (!pipelineLayout)
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
				auto shader = nbl::core::make_smart_refctd_ptr<nbl::asset::ICPUShader>(smart_refctd_ptr(buffer), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, ""); // must create cpu instance regardless underlying type

				if constexpr (withAssetConverter)
					outShader = std::move(shader);
				else
					outShader = create<typename TYPES::SHADER>(shader.get()); // note: dependency between cpu object instance & gpu object creation, not sure if its our API design failure or maybe I'm just thinking too much

				return outShader;
			};

			typename RESOURCES_BUNDLE_SCRATCH::SHADERS& basic = scratch.shaders[GEOMETRIES_CPU::EGP_BASIC], cone = scratch.shaders[GEOMETRIES_CPU::EGP_CONE], ico = scratch.shaders[GEOMETRIES_CPU::EGP_ICO];

			// TODO: return value validation
			createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.vertex.spv") > (IShader::E_SHADER_STAGE::ESS_VERTEX, basic.vertex);
			createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (IShader::E_SHADER_STAGE::ESS_FRAGMENT, basic.fragment);

			createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.cone.vertex.spv") > (IShader::E_SHADER_STAGE::ESS_VERTEX, cone.vertex);
			createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (IShader::E_SHADER_STAGE::ESS_FRAGMENT, cone.fragment); // note we reuse fragment from basic!

			createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.ico.vertex.spv") > (IShader::E_SHADER_STAGE::ESS_VERTEX, ico.vertex);
			createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (IShader::E_SHADER_STAGE::ESS_FRAGMENT, ico.fragment); // note we reuse fragment from basic!
		}

		// geometries
		{
			auto geometries = GEOMETRIES_CPU(geometryCreator);

			// TODO: move creation here
		}

		return true;
	}

	inline bool finalize(RESOURCES_BUNDLE& output)
	{
		if constexpr (withAssetConverter)
		{
			// TODO: asset converter for input info, scratch at this point has ready to convert cpu resources
		}
		else
			output = scratch; // scratch has all ready to use gpu resources, just assign to info

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
			return nbl::core::make_smart_refctd_ptr<T>(std::forward<Args>(args)...);
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
			else if constexpr (std::same_as<T, typename TYPES::SHADER>)
				return device->createShader(std::forward<Args>(args)...);
			else
				return nullptr;
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

	nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_colorAttachment, m_depthAttachment;

	struct
	{
		const uint32_t startedValue = 0, finishedValue = 0x45;
		nbl::core::smart_refctd_ptr<nbl::video::ISemaphore> progress;
	} semaphore;

	CScene(nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> _device, nbl::core::smart_refctd_ptr<nbl::system::ILogger> _logger, nbl::video::CThreadSafeQueueAdapter* _graphicsQueue, const nbl::asset::IGeometryCreator* _geometryCreator)
		: m_device(nbl::core::smart_refctd_ptr(_device)), m_logger(nbl::core::smart_refctd_ptr(_logger)), queue(_graphicsQueue)
	{
		_NBL_STATIC_INLINE_CONSTEXPR bool BUILD_WITH_CONVERTER = false; // tmp
		using BUILDER = ::RESOURCES_BUILDER<BUILD_WITH_CONVERTER>;

		bool status = createCommandBuffer();
		BUILDER builder (m_device.get(), m_logger.get(), _geometryCreator);

		if (builder.build())
		{
			if (!builder.finalize(resources))
				m_logger->log("Could not finalize to gpu objects!", nbl::system::ILogger::ELL_ERROR);
		}
		else
			m_logger->log("Could not build resource objects!", nbl::system::ILogger::ELL_ERROR);

#if 0
		// gpu resources created, let's create descriptor set
		{
			const nbl::video::IGPUDescriptorSetLayout* const layouts[] = { nullptr, descriptorSetLayout.get() };
			const uint32_t setCounts[] = { 0u, 1u };

			m_descriptorPool = m_device->createDescriptorPoolForDSLayouts(nbl::video::IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);

			if (!m_descriptorPool)
			{
				m_logger->log("Could not create Descriptor Pool!", nbl::system::ILogger::ELL_ERROR);
				return nullptr;
			}
		}

		m_gpuDescriptorSet = m_descriptorPool->createDescriptorSet(descriptorSetLayout);

		if (!m_gpuDescriptorSet)
		{
			m_logger->log("Could not create Descriptor Set!", nbl::system::ILogger::ELL_ERROR);
			return nullptr;
		}
#endif
		
		{
			// descriptor write ubo
			nbl::video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = m_gpuDescriptorSet.get();
			write.binding = 0;
			write.arrayElement = 0u;
			write.count = 1u;

			nbl::video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = nbl::core::smart_refctd_ptr(m_ubo);
				info.info.buffer.offset = 0ull;
				info.info.buffer.size = m_ubo->getSize();
			}

			write.info = &info;
			m_device->updateDescriptorSets(1u, &write, 0u, nullptr);
		}
	}
	~CScene() {}

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

		m_commandBuffer->bindGraphicsPipeline(rawPipeline);
		m_commandBuffer->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS, rawPipeline->getLayout(), 1, 1, &m_gpuDescriptorSet.get());

		const nbl::asset::SBufferBinding<const nbl::video::IGPUBuffer> vertices = { .offset = 0, .buffer = hook.vertexBuffer }, indices = { .offset = 0, .buffer = hook.indexBuffer };

		m_commandBuffer->bindVertexBuffers(0, 1, &vertices);

		if (indices.buffer && hook.indexType != nbl::asset::EIT_UNKNOWN)
		{
			m_commandBuffer->bindIndexBuffer(indices, hook.indexType);
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

	// note, must be updated outside render pass
	inline void update()
	{
		nbl::asset::SBufferRange<nbl::video::IGPUBuffer> range;
		range.buffer = nbl::core::smart_refctd_ptr(m_ubo);
		range.size = m_ubo->getSize();

		m_commandBuffer->updateBuffer(range, &object.viewParameters);
	}

	inline decltype(auto) getReferenceObjects()
	{
		return (resources.objects); // note: do not remove "()" - it makes the return type lvalue reference instead of copy 
	}

private:
	#if 0
	// we will make this call templated - first instance will use our gpu creation as it was with a few improvements second will asset converter to create gpu resources
	template<bool withAssetConverter>
	void createGPUResources(const GEOMETRIES_CPU& geometries)
	{

		// gpu geometries' pipelines & buffers
		for (const auto& inGeometry : geometries.objects)
		{
			auto& outData = referenceObjects.emplace_back();
			auto& [gpu, meta] = outData;

			bool status = true;

			meta.name = inGeometry.meta.name;
			meta.type = inGeometry.meta.type;

			struct
			{
				nbl::asset::SBlendParams blend;
				nbl::asset::SRasterizationParams rasterization;
				nbl::video::IGPUGraphicsPipeline::SCreationParams pipeline;
			} params;

			{
				params.blend.logicOp = nbl::asset::ELO_NO_OP;

				auto& param = params.blend.blendParams[0];
				param.srcColorFactor = nbl::asset::EBF_SRC_ALPHA;//VK_BLEND_FACTOR_SRC_ALPHA;
				param.dstColorFactor = nbl::asset::EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
				param.colorBlendOp = nbl::asset::EBO_ADD;//VK_BLEND_OP_ADD;
				param.srcAlphaFactor = nbl::asset::EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
				param.dstAlphaFactor = nbl::asset::EBF_ZERO;//VK_BLEND_FACTOR_ZERO;
				param.alphaBlendOp = nbl::asset::EBO_ADD;//VK_BLEND_OP_ADD;
				param.colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);//VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			}

			params.rasterization.faceCullingMode = nbl::asset::EFCM_NONE;
			{
				const nbl::video::IGPUShader::SSpecInfo info [] =
				{
					{.entryPoint = "VSMain", .shader = shaders[inGeometry.shadersType].vertex.get() },
					{.entryPoint = "PSMain", .shader = shaders[inGeometry.shadersType].fragment.get() }
				};

				params.pipeline.layout = pipelineLayout;
				params.pipeline.shaders = info;
				params.pipeline.renderpass = m_renderpass.get();
				params.pipeline.cached = { .vertexInput = inGeometry.data.inputParams, .primitiveAssembly = inGeometry.data.assemblyParams, .rasterization = params.rasterization, .blend = params.blend, .subpassIx = 0u };

				gpu.indexCount = inGeometry.data.indexCount;
				gpu.indexType = inGeometry.data.indexType;

				// TODO: cache pipeline & try lookup for existing one first

				if (!m_device->createGraphicsPipelines(nullptr, { { params.pipeline } }, &gpu.pipeline))
				{
					m_logger->log("Could not create GPU Graphics Pipeline for [%s] object!", nbl::system::ILogger::ELL_ERROR, meta.name.data());
					status = false;
				}

				if (!createVIBuffers(inGeometry, outData))
				{
					m_logger->log("Could not create GPU buffers for [%s] object!", nbl::system::ILogger::ELL_ERROR, meta.name.data());
					status = false;
				}

				if (!status)
				{
					m_logger->log("[%s] object will not be created!", nbl::system::ILogger::ELL_ERROR, meta.name.data());

					gpu.valid = false;
					gpu.vertexBuffer = nullptr;
					gpu.indexBuffer = nullptr;
					gpu.indexCount = 0u;
					gpu.indexType = nbl::asset::E_INDEX::EIT_UNKNOWN;
					gpu.pipeline = nullptr;

					continue;
				}
			}
		}

		// gpu view params ubo buffer
		{
			const auto mask = m_device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

			m_ubo = m_device->createBuffer({ {.size = sizeof(nbl::asset::SBasicViewParameters), .usage = nbl::core::bitflag(nbl::asset::IBuffer::EUF_UNIFORM_BUFFER_BIT) | nbl::asset::IBuffer::EUF_TRANSFER_DST_BIT | nbl::asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF} });

			for (auto it : { m_ubo })
			{
				nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = it->getMemoryReqs();
				reqs.memoryTypeBits &= mask;

				m_device->allocate(reqs, it.get());
			}
		}

		//////////////////////////////
		// TODO
		// builder.finalize 
		// (build's finalize calls asset converter or if we build without the converter we are already finalized)
		// after the call we have all gpu instances

		// frame buffer
		{
			// note there is no such a thing like cpu frame buffer, we assume we have finalized resources at this point and have gpu instances
			nbl::video::IGPUFramebuffer::SCreationParams params =
			{
				{
					.renderpass = nbl::core::smart_refctd_ptr(m_renderpass),
					.depthStencilAttachments = &m_depthAttachment.get(),
					.colorAttachments = &m_colorAttachment.get(),
					.width = FRAMEBUFFER_W,
					.height = FRAMEBUFFER_H,
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

	bool createVIBuffers(const GEOMETRIES_CPU::REFERENCE_OBJECT_CPU& inGeometry, REFERENCE_DRAW_HOOK_GPU& outData)
	{
		const auto mask = m_device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

		auto vBuffer = nbl::core::smart_refctd_ptr(inGeometry.data.bindings[0].buffer); // no offset
		auto iBuffer = nbl::core::smart_refctd_ptr(inGeometry.data.indexBuffer.buffer); // no offset

		outData.first.vertexBuffer = m_device->createBuffer({ {.size = vBuffer->getSize(), .usage = nbl::core::bitflag(nbl::asset::IBuffer::EUF_VERTEX_BUFFER_BIT) | nbl::asset::IBuffer::EUF_TRANSFER_DST_BIT | nbl::asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF} });
		outData.first.indexBuffer = iBuffer ? m_device->createBuffer({ {.size = iBuffer->getSize(), .usage = nbl::core::bitflag(nbl::asset::IBuffer::EUF_INDEX_BUFFER_BIT) | nbl::asset::IBuffer::EUF_VERTEX_BUFFER_BIT | nbl::asset::IBuffer::EUF_TRANSFER_DST_BIT | nbl::asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF} }) : nullptr;

		if (!outData.first.vertexBuffer)
			return false;

		if (inGeometry.data.indexType != nbl::asset::EIT_UNKNOWN)
			if (!outData.first.indexBuffer)
				return false;

		for (auto it : { outData.first.vertexBuffer , outData.first.indexBuffer })
		{
			if (it)
			{
				auto reqs = it->getMemoryReqs();
				reqs.memoryTypeBits &= mask;

				m_device->allocate(reqs, it.get());
			}
		}

		{
			auto fillGPUBuffer = [&m_logger = m_logger](nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer> cBuffer, nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> gBuffer)
			{
				auto binding = gBuffer->getBoundMemory();

				if (!binding.memory->map({ 0ull, binding.memory->getAllocationSize() }, nbl::video::IDeviceMemoryAllocation::EMCAF_READ))
				{
					m_logger->log("Could not map device memory", nbl::system::ILogger::ELL_ERROR);
					return false;
				}

				if (!binding.memory->isCurrentlyMapped())
				{
					m_logger->log("Buffer memory is not mapped!", nbl::system::ILogger::ELL_ERROR);
					return false;
				}

				auto* mPointer = binding.memory->getMappedPointer();
				memcpy(mPointer, cBuffer->getPointer(), gBuffer->getSize());
				binding.memory->unmap();

				return true;
			};

			if (!fillGPUBuffer(vBuffer, outData.first.vertexBuffer))
				return false;

			if (outData.first.indexBuffer)
				if (!fillGPUBuffer(iBuffer, outData.first.indexBuffer))
					return false;
		}

		return true;
	}
#endif

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

	nbl::core::smart_refctd_ptr<nbl::video::IDescriptorPool> m_descriptorPool;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_gpuDescriptorSet;

	nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer> m_frameBuffer;

	RESOURCES_BUNDLE resources;

	// TODO: TO REMOVE, those will be/are in resources
	nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> m_renderpass;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_ubo;
};

#endif // __NBL_THIS_EXAMPLE_SCENE_H_INCLUDED__