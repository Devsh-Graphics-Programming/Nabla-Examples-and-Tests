#ifndef _NBL_GEOMETRY_CREATOR_SCENE_H_INCLUDED_
#define _NBL_GEOMETRY_CREATOR_SCENE_H_INCLUDED_

#include <nabla.h>

#include "nbl/asset/utils/CGeometryCreator.h"
#include "SBasicViewParameters.hlsl"
#include "geometry/creator/spirv/builtin/CArchive.h"
#include "geometry/creator/spirv/builtin/builtinResources.h"

namespace nbl::scene::geometrycreator
{

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

constexpr static inline struct ClearValues
{
	nbl::video::IGPUCommandBuffer::SClearColorValue color = { .float32 = {0.f,0.f,0.f,1.f} };
	nbl::video::IGPUCommandBuffer::SClearDepthStencilValue depth = { .depth = 0.f };
} clear;

#define TYPES_IMPL_BOILERPLATE(WithConverter) struct Types \
{ \
	using descriptor_set_layout_t = std::conditional_t<WithConverter, nbl::asset::ICPUDescriptorSetLayout, nbl::video::IGPUDescriptorSetLayout>; \
	using pipeline_layout_t = std::conditional_t<WithConverter, nbl::asset::ICPUPipelineLayout, nbl::video::IGPUPipelineLayout>; \
	using renderpass_t = std::conditional_t<WithConverter, nbl::asset::ICPURenderpass, nbl::video::IGPURenderpass>; \
	using image_view_t = std::conditional_t<WithConverter, nbl::asset::ICPUImageView, nbl::video::IGPUImageView>; \
	using image_t = std::conditional_t<WithConverter, nbl::asset::ICPUImage, nbl::video::IGPUImage>; \
	using buffer_t = std::conditional_t<WithConverter, nbl::asset::ICPUBuffer, nbl::video::IGPUBuffer>; \
	using shader_t = std::conditional_t<WithConverter, nbl::asset::ICPUShader, nbl::video::IGPUShader>; \
	using graphics_pipeline_t = std::conditional_t<WithConverter, nbl::asset::ICPUGraphicsPipeline, nbl::video::IGPUGraphicsPipeline>; \
	using descriptor_set = std::conditional_t<WithConverter, nbl::asset::ICPUDescriptorSet, nbl::video::IGPUDescriptorSet>; \
}

template<bool withAssetConverter>
struct ResourcesBundleBase
{
	TYPES_IMPL_BOILERPLATE(withAssetConverter);

	struct ReferenceObject
	{
		struct Bindings
		{
			nbl::asset::SBufferBinding<typename Types::buffer_t> vertex, index;
		};

		nbl::core::smart_refctd_ptr<typename Types::graphics_pipeline_t> pipeline = nullptr;

		Bindings bindings;
		nbl::asset::E_INDEX_TYPE indexType = nbl::asset::E_INDEX_TYPE::EIT_UNKNOWN;
		uint32_t indexCount = {};
	};

	using ReferenceDrawHook = std::pair<ReferenceObject, ObjectMeta>;

	nbl::core::smart_refctd_ptr<typename Types::renderpass_t> renderpass;
	std::array<ReferenceDrawHook, OT_COUNT> objects;
	nbl::asset::SBufferBinding<typename Types::buffer_t> ubo;

	struct
	{
		nbl::core::smart_refctd_ptr<typename Types::image_view_t> color, depth;
	} attachments;

	nbl::core::smart_refctd_ptr<typename Types::descriptor_set> descriptorSet;
};

struct ResourcesBundle : public ResourcesBundleBase<false>
{
	using base_t = ResourcesBundleBase<false>;
};

#define EXPOSE_NABLA_NAMESPACES() using namespace nbl; \
using namespace core; \
using namespace asset; \
using namespace video; \
using namespace scene; \
using namespace system

template<bool withAssetConverter>
class ResourceBuilder
{
public:
	TYPES_IMPL_BOILERPLATE(withAssetConverter);

	using this_t = ResourceBuilder<withAssetConverter>;

	ResourceBuilder(nbl::video::IUtilities* const _utilities, nbl::video::IGPUCommandBuffer* const _commandBuffer, nbl::system::ILogger* const _logger, const nbl::asset::IGeometryCreator* const _geometryCreator)
		: utilities(_utilities), commandBuffer(_commandBuffer), logger(_logger), geometries(_geometryCreator)
	{
		assert(utilities);
		assert(logger);
	}

	/*
		if (withAssetConverter) then
			-> .build cpu objects
		else
			-> .build gpu objects & record any resource update upload transfers into command buffer
	*/

	inline bool build()
	{
		EXPOSE_NABLA_NAMESPACES();

		if constexpr (!withAssetConverter)
		{
			commandBuffer->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			commandBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			commandBuffer->beginDebugMarker("Resources builder's buffers upload [manual]");
		}

		using functor_t = std::function<bool(void)>;

		auto work = std::to_array
		({
			functor_t(std::bind(&this_t::createDescriptorSetLayout, this)),
			functor_t(std::bind(&this_t::createPipelineLayout, this)),
			functor_t(std::bind(&this_t::createRenderpass, this)),
			functor_t(std::bind(&this_t::createFramebufferAttachments, this)),
			functor_t(std::bind(&this_t::createShaders, this)),
			functor_t(std::bind(&this_t::createGeometries, this)),
			functor_t(std::bind(&this_t::createViewParametersUboBuffer, this)),
			functor_t(std::bind(&this_t::createDescriptorSet, this))
		});

		for (auto& task : work)
			if (!task())
				return false;

		if constexpr (!withAssetConverter)
			commandBuffer->end();

		return true;
	}

	/*
		if (withAssetConverter) then
			-> .convert cpu objects to gpu & update gpu buffers
		else
			-> update gpu buffers
	*/

	inline bool finalize(ResourcesBundle& output, nbl::video::CThreadSafeQueueAdapter* transferCapableQueue)
	{
		EXPOSE_NABLA_NAMESPACES();

		// TODO: use multiple command buffers
		std::array<IQueue::SSubmitInfo::SCommandBufferInfo,1u> commandBuffers = {};
		{
			commandBuffers.front().cmdbuf = commandBuffer;
		}

		if constexpr (withAssetConverter)
		{
			// note that asset converter records basic transfer uploads itself, we only begin the recording with ONE_TIME_SUBMIT_BIT
			commandBuffer->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			commandBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			commandBuffer->beginDebugMarker("Resources builder's buffers upload [asset converter]");

			// asset converter - scratch at this point has ready to convert cpu resources
			smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = utilities->getLogicalDevice(),.optimizer = {} });
			CAssetConverter::SInputs inputs = {};
			inputs.logger = logger;

			struct ProxyCpuHooks
			{
				using object_size_t = std::tuple_size<decltype(scratch.objects)>;

				std::array<ICPURenderpass*, 1u> renderpass;
				std::array<ICPUGraphicsPipeline*, object_size_t::value> pipelines;
				std::array<ICPUBuffer*, object_size_t::value * 2u + 1u > buffers;
				std::array<ICPUImageView*, 2u> attachments;
				std::array<ICPUDescriptorSet*, 1u> descriptorSet;
			} hooks;

			enum AttachmentIx
			{
				AI_COLOR = 0u,
				AI_DEPTH = 1u,

				AI_COUNT
			};
			
			// gather CPU assets into span memory views
			{ 
				hooks.renderpass.front() = scratch.renderpass.get();
				for (uint32_t i = 0u; i < hooks.pipelines.size(); ++i)
				{
					auto& [reference, meta] = scratch.objects[static_cast<ObjectType>(i)];
					hooks.pipelines[i] = reference.pipeline.get();

					// [[ [vertex, index] [vertex, index] [vertex, index] ... [ubo] ]]
					hooks.buffers[2u * i + 0u] = reference.bindings.vertex.buffer.get();
					hooks.buffers[2u * i + 1u] = reference.bindings.index.buffer.get();
				}
				hooks.buffers.back() = scratch.ubo.buffer.get();
				hooks.attachments[AI_COLOR] = scratch.attachments.color.get();
				hooks.attachments[AI_DEPTH] = scratch.attachments.depth.get();
				hooks.descriptorSet.front() = scratch.descriptorSet.get();
			}

			// assign the CPU hooks to converter's inputs
			{
				std::get<CAssetConverter::SInputs::asset_span_t<ICPURenderpass>>(inputs.assets) = hooks.renderpass;
				std::get<CAssetConverter::SInputs::asset_span_t<ICPUGraphicsPipeline>>(inputs.assets) = hooks.pipelines;
				std::get<CAssetConverter::SInputs::asset_span_t<ICPUBuffer>>(inputs.assets) = hooks.buffers;
				// std::get<CAssetConverter::SInputs::asset_span_t<ICPUImageView>>(inputs.assets) = hooks.attachments; // NOTE: THIS IS NOT IMPLEMENTED YET IN CONVERTER!
				std::get<CAssetConverter::SInputs::asset_span_t<ICPUDescriptorSet>>(inputs.assets) = hooks.descriptorSet;
			}

			// reserve and create the GPU object handles
			auto reservation = converter->reserve(inputs);
			{
				auto prepass = [&]<typename asset_type_t>(const auto& references) -> bool
				{
					// retrieve the reserved handles
					auto objects = reservation.getGPUObjects<asset_type_t>();

					uint32_t counter = {};
					for (auto& object : objects)
					{
						// anything that fails to be reserved is a nullptr in the span of GPU Objects
						auto gpu = object.value;
						auto* reference = references[counter];

						if (reference)
						{
							// validate
							if (!gpu) // throw errors only if corresponding cpu hook was VALID (eg. we may have nullptr for some index buffers in the span for converter but it's OK, I'm too lazy to filter them before passing to the converter inputs and don't want to deal with dynamic alloc)
							{
								logger->log("Failed to convert a CPU object to GPU!", ILogger::ELL_ERROR);
								return false;
							}
						}
						
						++counter;
					}

					return true;
				};
				
				prepass.template operator() < ICPURenderpass > (hooks.renderpass);
				prepass.template operator() < ICPUGraphicsPipeline > (hooks.pipelines);
				prepass.template operator() < ICPUBuffer > (hooks.buffers);
				// validate.template operator() < ICPUImageView > (hooks.attachments);
				prepass.template operator() < ICPUDescriptorSet > (hooks.descriptorSet);
			}

			auto semaphore = utilities->getLogicalDevice()->createSemaphore(0u);

			// TODO: compute submit as well for the images' mipmaps
			SIntendedSubmitInfo transfer = {};
			transfer.queue = transferCapableQueue;
			transfer.scratchCommandBuffers = commandBuffers;
			transfer.scratchSemaphore = {
				.semaphore = semaphore.get(),
				.value = 0u,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};
			// issue the convert call
			{
				CAssetConverter::SConvertParams params = {};
				params.utilities = utilities;
				params.transfer = &transfer;

				// basically it records all data uploads and submits them right away
				auto future = reservation.convert(params);
				if (future.copy()!=IQueue::RESULT::SUCCESS)
				{
					logger->log("Failed to await submission feature!", ILogger::ELL_ERROR);
					return false;
				}

				// assign gpu objects to output
				auto& base = static_cast<ResourcesBundle::base_t&>(output);
				{
					auto&& [renderpass, pipelines, buffers, descriptorSet] = std::make_tuple(reservation.getGPUObjects<ICPURenderpass>().front().value, reservation.getGPUObjects<ICPUGraphicsPipeline>(), reservation.getGPUObjects<ICPUBuffer>(), reservation.getGPUObjects<ICPUDescriptorSet>().front().value);
					{
						base.renderpass = renderpass;
						for (uint32_t i = 0u; i < pipelines.size(); ++i)
						{
							const auto type = static_cast<ObjectType>(i);
							const auto& [rcpu, rmeta] = scratch.objects[type];
							auto& [gpu, meta] = base.objects[type];

							gpu.pipeline = pipelines[i].value;
							// [[ [vertex, index] [vertex, index] [vertex, index] ... [ubo] ]]
							gpu.bindings.vertex = {.offset = 0u, .buffer = buffers[2u * i + 0u].value};
							gpu.bindings.index = {.offset = 0u, .buffer = buffers[2u * i + 1u].value};

							gpu.indexCount = rcpu.indexCount;
							gpu.indexType = rcpu.indexType;
							meta.name = rmeta.name;
							meta.type = rmeta.type;
						}
						base.ubo = {.offset = 0u, .buffer = buffers.back().value};
						base.descriptorSet = descriptorSet;
					
						/*
							// base.attachments.color = attachments[AI_COLOR].value;
							// base.attachments.depth = attachments[AI_DEPTH].value;

							note conversion of image views is not yet supported by the asset converter 
							- it's complicated, we have to kinda temporary ignore DRY a bit here to not break the design which is correct

							TEMPORARY: we patch attachments by allocating them ourselves here given cpu instances & parameters
							TODO: remove following code once asset converter works with image views & update stuff
						*/

						for (uint32_t i = 0u; i < AI_COUNT; ++i)
						{
							const auto* reference = hooks.attachments[i];
							auto& out = (i == AI_COLOR ? base.attachments.color : base.attachments.depth);

							const auto& viewParams = reference->getCreationParameters();
							const auto& imageParams = viewParams.image->getCreationParameters();

							auto image = utilities->getLogicalDevice()->createImage
							(
								IGPUImage::SCreationParams
								({
									.type = imageParams.type,
									.samples = imageParams.samples,
									.format = imageParams.format,
									.extent = imageParams.extent,
									.mipLevels = imageParams.mipLevels,
									.arrayLayers = imageParams.arrayLayers,
									.usage = imageParams.usage
								})
							);

							if (!image)
							{
								logger->log("Could not create image!", ILogger::ELL_ERROR);
								return false;
							}

							bool IS_DEPTH = isDepthOrStencilFormat(imageParams.format);
							std::string_view DEBUG_NAME = IS_DEPTH ? "UI Scene Depth Attachment Image" : "UI Scene Color Attachment Image";
							image->setObjectDebugName(DEBUG_NAME.data());

							if (!utilities->getLogicalDevice()->allocate(image->getMemoryReqs(), image.get()).isValid())
							{
								logger->log("Could not allocate memory for an image!", ILogger::ELL_ERROR);
								return false;
							}
						
							out = utilities->getLogicalDevice()->createImageView
							(
								IGPUImageView::SCreationParams
								({
									.flags = viewParams.flags,
									.subUsages = viewParams.subUsages,
									.image = std::move(image),
									.viewType = viewParams.viewType,
									.format = viewParams.format,
									.subresourceRange = viewParams.subresourceRange
								})
							);

							if (!out)
							{
								logger->log("Could not create image view!", ILogger::ELL_ERROR);
								return false;
							}
						}

						logger->log("Image View attachments has been allocated by hand after asset converter successful submit becasuse it doesn't support converting them yet!", ILogger::ELL_WARNING);
					}
				}
			}
		}
		else
		{
			auto completed = utilities->getLogicalDevice()->createSemaphore(0u);

			std::array<IQueue::SSubmitInfo::SSemaphoreInfo, 1u> signals;
			{
				auto& signal = signals.front();
				signal.value = 1;
				signal.stageMask = bitflag(PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS);
				signal.semaphore = completed.get();
			}

			const IQueue::SSubmitInfo infos [] =
			{
				{
					.waitSemaphores = {},
					.commandBuffers = commandBuffers, // note that here our command buffer is already recorded!
					.signalSemaphores = signals
				}
			};

			if (transferCapableQueue->submit(infos) != IQueue::RESULT::SUCCESS)
			{
				logger->log("Failed to submit transfer upload operations!", ILogger::ELL_ERROR);
				return false;
			}

			const ISemaphore::SWaitInfo info [] =
			{ {
				.semaphore = completed.get(),
				.value = 1
			} };

			utilities->getLogicalDevice()->blockForSemaphores(info);

			static_cast<ResourcesBundle::base_t&>(output) = static_cast<ResourcesBundle::base_t&>(scratch); // scratch has all ready to use allocated gpu resources with uploaded memory so now just assign resources to base output
		}

		// write the descriptor set
		{
			// descriptor write ubo
			IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = output.descriptorSet.get();
			write.binding = 0;
			write.arrayElement = 0u;
			write.count = 1u;

			IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = smart_refctd_ptr(output.ubo.buffer);
				info.info.buffer.offset = output.ubo.offset;
				info.info.buffer.size = output.ubo.buffer->getSize();
			}

			write.info = &info;

			if(!utilities->getLogicalDevice()->updateDescriptorSets(1u, &write, 0u, nullptr))
			{
				logger->log("Could not write descriptor set!", ILogger::ELL_ERROR);
				return false;
			}
		}

		return true;
	}

private:
	bool createDescriptorSetLayout()
	{
		EXPOSE_NABLA_NAMESPACES();

		typename Types::descriptor_set_layout_t::SBinding bindings[] =
		{
			{
				.binding = 0u,
				.type = IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
				.createFlags = Types::descriptor_set_layout_t::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT,
				.count = 1u,
			}
		};

		if constexpr (withAssetConverter)
			scratch.descriptorSetLayout = make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindings);
		else
			scratch.descriptorSetLayout = utilities->getLogicalDevice()->createDescriptorSetLayout(bindings);

		if (!scratch.descriptorSetLayout)
		{
			logger->log("Could not descriptor set layout!", ILogger::ELL_ERROR);
			return false;
		}

		return true;
	}

	bool createDescriptorSet()
	{
		EXPOSE_NABLA_NAMESPACES();

		if constexpr (withAssetConverter)
			scratch.descriptorSet = make_smart_refctd_ptr<ICPUDescriptorSet>(smart_refctd_ptr(scratch.descriptorSetLayout));
		else
		{
			const IGPUDescriptorSetLayout* const layouts[] = { scratch.descriptorSetLayout.get()};
			const uint32_t setCounts[] = { 1u };

			// note descriptor set has back smart pointer to its pool, so we dont need to keep it explicitly
			auto pool = utilities->getLogicalDevice()->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);

			if (!pool)
			{
				logger->log("Could not create Descriptor Pool!", ILogger::ELL_ERROR);
				return false;
			}

			pool->createDescriptorSets(layouts, &scratch.descriptorSet);
		}

		if (!scratch.descriptorSet)
		{
			logger->log("Could not create Descriptor Set!", ILogger::ELL_ERROR);
			return false;
		}

		return true;
	}

	bool createPipelineLayout()
	{
		EXPOSE_NABLA_NAMESPACES();

		const std::span<const SPushConstantRange> range = {};

		if constexpr (withAssetConverter)
			scratch.pipelineLayout = make_smart_refctd_ptr<ICPUPipelineLayout>(range, nullptr, smart_refctd_ptr(scratch.descriptorSetLayout), nullptr, nullptr);
		else
			scratch.pipelineLayout = utilities->getLogicalDevice()->createPipelineLayout(range, nullptr, smart_refctd_ptr(scratch.descriptorSetLayout), nullptr, nullptr);

		if (!scratch.pipelineLayout)
		{
			logger->log("Could not create pipeline layout!", ILogger::ELL_ERROR);
			return false;
		}

		return true;
	}

	bool createRenderpass()
	{
		EXPOSE_NABLA_NAMESPACES();

		static constexpr Types::renderpass_t::SCreationParams::SColorAttachmentDescription colorAttachments[] =
		{
			{
				{
					{
						.format = ColorFboAttachmentFormat,
						.samples = Samples,
						.mayAlias = false
					},
					/* .loadOp = */ Types::renderpass_t::LOAD_OP::CLEAR,
					/* .storeOp = */ Types::renderpass_t::STORE_OP::STORE,
					/* .initialLayout = */ Types::image_t::LAYOUT::UNDEFINED,
					/* .finalLayout = */ Types::image_t::LAYOUT::READ_ONLY_OPTIMAL
				}
			},
			Types::renderpass_t::SCreationParams::ColorAttachmentsEnd
		};

		static constexpr Types::renderpass_t::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] =
		{
			{
				{
					{
						.format = DepthFboAttachmentFormat,
						.samples = Samples,
						.mayAlias = false
					},
					/* .loadOp = */ {Types::renderpass_t::LOAD_OP::CLEAR},
					/* .storeOp = */ {Types::renderpass_t::STORE_OP::STORE},
					/* .initialLayout = */ {Types::image_t::LAYOUT::UNDEFINED},
					/* .finalLayout = */ {Types::image_t::LAYOUT::ATTACHMENT_OPTIMAL}
				}
			},
			Types::renderpass_t::SCreationParams::DepthStencilAttachmentsEnd
		};

		typename Types::renderpass_t::SCreationParams::SSubpassDescription subpasses[] =
		{
			{},
			Types::renderpass_t::SCreationParams::SubpassesEnd
		};

		subpasses[0].depthStencilAttachment.render = { .attachmentIndex = 0u,.layout = Types::image_t::LAYOUT::ATTACHMENT_OPTIMAL };
		subpasses[0].colorAttachments[0] = { .render = {.attachmentIndex = 0u, .layout = Types::image_t::LAYOUT::ATTACHMENT_OPTIMAL } };

		static constexpr Types::renderpass_t::SCreationParams::SSubpassDependency dependencies[] =
		{
			// wipe-transition of Color to ATTACHMENT_OPTIMAL
			{
				.srcSubpass = Types::renderpass_t::SCreationParams::SSubpassDependency::External,
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
				.dstSubpass = Types::renderpass_t::SCreationParams::SSubpassDependency::External,
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
			Types::renderpass_t::SCreationParams::DependenciesEnd
		};

		typename Types::renderpass_t::SCreationParams params = {};
		params.colorAttachments = colorAttachments;
		params.depthStencilAttachments = depthAttachments;
		params.subpasses = subpasses;
		params.dependencies = dependencies;

		if constexpr (withAssetConverter)
			scratch.renderpass = ICPURenderpass::create(params);
		else
			scratch.renderpass = utilities->getLogicalDevice()->createRenderpass(params);

		if (!scratch.renderpass)
		{
			logger->log("Could not create render pass!", ILogger::ELL_ERROR);
			return false;
		}

		return true;
	}

	bool createFramebufferAttachments()
	{
		EXPOSE_NABLA_NAMESPACES();

		auto createImageView = [&]<E_FORMAT format>(smart_refctd_ptr<typename Types::image_view_t>& outView) -> smart_refctd_ptr<typename Types::image_view_t>
		{
			constexpr bool IS_DEPTH = isDepthOrStencilFormat<format>();
			constexpr auto USAGE = [](const bool isDepth)
			{
				bitflag<Types::image_t::E_USAGE_FLAGS> usage = Types::image_t::EUF_RENDER_ATTACHMENT_BIT;

				if (!isDepth)
					usage |= Types::image_t::EUF_SAMPLED_BIT;

				return usage;
			}(IS_DEPTH);
			constexpr auto ASPECT = IS_DEPTH ? IImage::E_ASPECT_FLAGS::EAF_DEPTH_BIT : IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			constexpr std::string_view DEBUG_NAME = IS_DEPTH ? "UI Scene Depth Attachment Image" : "UI Scene Color Attachment Image";
			{
				smart_refctd_ptr<typename Types::image_t> image;
				{
					auto params = typename Types::image_t::SCreationParams(
					{
						.type = Types::image_t::ET_2D,
						.samples = Samples,
						.format = format,
						.extent = { FramebufferW, FramebufferH, 1u },
						.mipLevels = 1u,
						.arrayLayers = 1u,
						.usage = USAGE
					});

					if constexpr (withAssetConverter)
						image = ICPUImage::create(params);
					else
						image = utilities->getLogicalDevice()->createImage(std::move(params));
				}

				if (!image)
				{
					logger->log("Could not create image!", ILogger::ELL_ERROR);
					return nullptr;
				}

				if constexpr (withAssetConverter)
				{
					auto dummyBuffer = ICPUBuffer::create({ FramebufferW * FramebufferH * getTexelOrBlockBytesize<format>() });
					dummyBuffer->setContentHash(dummyBuffer->computeContentHash());

					auto regions = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
					auto& region = regions->front();

					region.imageSubresource = { .aspectMask = ASPECT, .mipLevel = 0u, .baseArrayLayer = 0u, .layerCount = 0u };
					region.bufferOffset = 0u;
					region.bufferRowLength = IImageAssetHandlerBase::calcPitchInBlocks(FramebufferW, getTexelOrBlockBytesize<format>());
					region.bufferImageHeight = 0u;
					region.imageOffset = { 0u, 0u, 0u };
					region.imageExtent = { FramebufferW, FramebufferH, 1u };

					if (!image->setBufferAndRegions(std::move(dummyBuffer), regions))
					{
						logger->log("Could not set image's regions!", ILogger::ELL_ERROR);
						return nullptr;
					}
					image->setContentHash(image->computeContentHash());
				}
				else
				{
					image->setObjectDebugName(DEBUG_NAME.data());

					if (!utilities->getLogicalDevice()->allocate(image->getMemoryReqs(), image.get()).isValid())
					{
						logger->log("Could not allocate memory for an image!", ILogger::ELL_ERROR);
						return nullptr;
					}
				}

				auto params = typename Types::image_view_t::SCreationParams
				({
					.flags = Types::image_view_t::ECF_NONE,
					.subUsages = USAGE,
					.image = std::move(image),
					.viewType = Types::image_view_t::ET_2D,
					.format = format,
					.subresourceRange = { .aspectMask = ASPECT, .baseMipLevel = 0u, .levelCount = 1u, .baseArrayLayer = 0u, .layerCount = 1u }
				});

				if constexpr (withAssetConverter)
					outView = make_smart_refctd_ptr<ICPUImageView>(std::move(params));
				else
					outView = utilities->getLogicalDevice()->createImageView(std::move(params));
 
				if (!outView)
				{
					logger->log("Could not create image view!", ILogger::ELL_ERROR);
					return nullptr;
				}

				return smart_refctd_ptr(outView);
			}
		};

		const bool allocated = createImageView.template operator() < ColorFboAttachmentFormat > (scratch.attachments.color) && createImageView.template operator() < DepthFboAttachmentFormat > (scratch.attachments.depth);

		if (!allocated)
		{
			logger->log("Could not allocate frame buffer's attachments!", ILogger::ELL_ERROR);
			return false;
		}

		return true;
	}

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
			else
				outShader = utilities->getLogicalDevice()->createShader(shader.get());

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

	bool createViewParametersUboBuffer()
	{
		EXPOSE_NABLA_NAMESPACES();

		using ibuffer_t = ::nbl::asset::IBuffer; // seems to be ambigous, both asset & core namespaces has IBuffer
		constexpr static auto UboUsage = bitflag(ibuffer_t::EUF_UNIFORM_BUFFER_BIT) | ibuffer_t::EUF_TRANSFER_DST_BIT | ibuffer_t::EUF_INLINE_UPDATE_VIA_CMDBUF;

		if constexpr (withAssetConverter)
		{
			auto uboBuffer = ICPUBuffer::create({ sizeof(SBasicViewParameters) });
			uboBuffer->addUsageFlags(UboUsage);
			uboBuffer->setContentHash(uboBuffer->computeContentHash());
			scratch.ubo = { .offset = 0u, .buffer = std::move(uboBuffer) };
		}
		else
		{
			const auto mask = utilities->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
			auto uboBuffer = utilities->getLogicalDevice()->createBuffer(IGPUBuffer::SCreationParams({ .size = sizeof(SBasicViewParameters), .usage = UboUsage }));

			if (!uboBuffer)
				return false;

			for (auto it : { uboBuffer })
			{
				IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = it->getMemoryReqs();
				reqs.memoryTypeBits &= mask;

				utilities->getLogicalDevice()->allocate(reqs, it.get());
			}

			scratch.ubo = { .offset = 0u, .buffer = std::move(uboBuffer) };
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

	using resources_bundle_base_t = ResourcesBundleBase<withAssetConverter>;

	struct ResourcesBundleScratch : public resources_bundle_base_t
	{
		using Types = resources_bundle_base_t::Types;

		ResourcesBundleScratch()
			: resources_bundle_base_t() {}

		struct Shaders
		{
			nbl::core::smart_refctd_ptr<typename Types::shader_t> vertex = nullptr, fragment = nullptr;
		};

		nbl::core::smart_refctd_ptr<typename Types::descriptor_set_layout_t> descriptorSetLayout;
		nbl::core::smart_refctd_ptr<typename Types::pipeline_layout_t> pipelineLayout;
		std::array<Shaders, GeometriesCpu::GP_COUNT> shaders;
	};

	// TODO: we could make those params templated with default values like below
	static constexpr auto FramebufferW = 1280u, FramebufferH = 720u;
	static constexpr auto ColorFboAttachmentFormat = nbl::asset::EF_R8G8B8A8_SRGB, DepthFboAttachmentFormat = nbl::asset::EF_D16_UNORM;
	static constexpr auto Samples = nbl::video::IGPUImage::ESCF_1_BIT;

	ResourcesBundleScratch scratch;

	nbl::video::IUtilities* const utilities;
	nbl::video::IGPUCommandBuffer* const commandBuffer;
	nbl::system::ILogger* const logger;
	GeometriesCpu geometries;
};

#undef TYPES_IMPL_BOILERPLATE

struct ObjectDrawHookCpu
{
	nbl::core::matrix3x4SIMD model;
	nbl::asset::SBasicViewParameters viewParameters;
	ObjectMeta meta;
};

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
	ObjectDrawHookCpu object; // TODO: this could be a vector (to not complicate the example I leave it single object), we would need a better system for drawing then to make only 1 max 2 indirect draw calls (indexed and not indexed objects)

	struct
	{
		const uint32_t startedValue = 0, finishedValue = 0x45;
		nbl::core::smart_refctd_ptr<nbl::video::ISemaphore> progress;
	} semaphore;

	struct CreateResourcesDirectlyWithDevice { using Builder = ResourceBuilder<false>; };
	struct CreateResourcesWithAssetConverter { using Builder = ResourceBuilder<true>; };

	~CScene() {}

	static inline nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> createCommandBuffer(nbl::video::ILogicalDevice* const device, nbl::system::ILogger* const logger, const uint32_t familyIx)
	{
		EXPOSE_NABLA_NAMESPACES();
		auto pool = device->createCommandPool(familyIx, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

		if (!pool)
		{
			logger->log("Couldn't create Command Pool!", ILogger::ELL_ERROR);
			return nullptr;
		}

		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmd;

		if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmd , 1 }))
		{
			logger->log("Couldn't create Command Buffer!", ILogger::ELL_ERROR);
			return nullptr;
		}

		return cmd;
	}

	template<typename CreateWith, typename... Args>
	static auto create(Args&&... args) -> decltype(auto)
	{
		EXPOSE_NABLA_NAMESPACES();

		/*
			user should call the constructor's args without last argument explicitly, this is a trick to make constructor templated, 
			eg.create(smart_refctd_ptr(device), smart_refctd_ptr(logger), queuePointer, geometryPointer)
		*/

		auto* scene = new CScene(std::forward<Args>(args)..., CreateWith {});
		smart_refctd_ptr<CScene> smart(scene, dont_grab);

		return smart;
	}

	inline void begin()
	{
		EXPOSE_NABLA_NAMESPACES();

		m_commandBuffer->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		m_commandBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		m_commandBuffer->beginDebugMarker("UISampleApp Offline Scene Frame");

		semaphore.progress = m_utilities->getLogicalDevice()->createSemaphore(semaphore.startedValue);
	}

	inline void record()
	{
		EXPOSE_NABLA_NAMESPACES();

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

		const IGPUCommandBuffer::SRenderpassBeginInfo info =
		{
			.framebuffer = m_frameBuffer.get(),
			.colorClearValues = &clear.color,
			.depthStencilClearValues = &clear.depth,
			.renderArea = renderArea
		};

		m_commandBuffer->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

		const auto& [hook, meta] = resources.objects[object.meta.type];
		auto* rawPipeline = hook.pipeline.get();

		SBufferBinding<const IGPUBuffer> vertex = hook.bindings.vertex, index = hook.bindings.index;

		m_commandBuffer->bindGraphicsPipeline(rawPipeline);
		m_commandBuffer->bindDescriptorSets(EPBP_GRAPHICS, rawPipeline->getLayout(), 1, 1, &resources.descriptorSet.get());
		m_commandBuffer->bindVertexBuffers(0, 1, &vertex);

		if (index.buffer && hook.indexType != EIT_UNKNOWN)
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
		EXPOSE_NABLA_NAMESPACES();

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

	// note: must be updated outside render pass
	inline void update()
	{
		EXPOSE_NABLA_NAMESPACES();

		SBufferRange<IGPUBuffer> range;
		range.buffer = smart_refctd_ptr(resources.ubo.buffer);
		range.size = resources.ubo.buffer->getSize();

		m_commandBuffer->updateBuffer(range, &object.viewParameters);
	}

	inline decltype(auto) getResources()
	{
		return (resources); // note: do not remove "()" - it makes the return type lvalue reference instead of copy 
	}

private:
	template<typename CreateWith = CreateResourcesDirectlyWithDevice> // TODO: enforce constraints, only those 2 above are valid
	CScene(nbl::core::smart_refctd_ptr<nbl::video::IUtilities> _utilities, nbl::core::smart_refctd_ptr<nbl::system::ILogger> _logger, nbl::video::CThreadSafeQueueAdapter* _graphicsQueue, const nbl::asset::IGeometryCreator* _geometryCreator, CreateWith createWith = {})
		: m_utilities(nbl::core::smart_refctd_ptr(_utilities)), m_logger(nbl::core::smart_refctd_ptr(_logger)), queue(_graphicsQueue)
	{
		EXPOSE_NABLA_NAMESPACES();
		using Builder = typename CreateWith::Builder;

		m_commandBuffer = createCommandBuffer(m_utilities->getLogicalDevice(), m_utilities->getLogger(), queue->getFamilyIndex());
		Builder builder(m_utilities.get(), m_commandBuffer.get(), m_logger.get(), _geometryCreator);

		// gpu resources
		if (builder.build())
		{
			if (!builder.finalize(resources, queue))
				m_logger->log("Could not finalize resource objects to gpu objects!", ILogger::ELL_ERROR);
		}
		else
			m_logger->log("Could not build resource objects!", ILogger::ELL_ERROR);

		// frame buffer
		{
			const auto extent = resources.attachments.color->getCreationParameters().image->getCreationParameters().extent;

			IGPUFramebuffer::SCreationParams params =
			{
				{
					.renderpass = smart_refctd_ptr(resources.renderpass),
					.depthStencilAttachments = &resources.attachments.depth.get(),
					.colorAttachments = &resources.attachments.color.get(),
					.width = extent.width,
					.height = extent.height,
					.layers = 1u
				}
			};

			m_frameBuffer = m_utilities->getLogicalDevice()->createFramebuffer(std::move(params));

			if (!m_frameBuffer)
			{
				m_logger->log("Could not create frame buffer!", ILogger::ELL_ERROR);
				return;
			}
		}
	}

	nbl::core::smart_refctd_ptr<nbl::video::IUtilities> m_utilities;
	nbl::core::smart_refctd_ptr<nbl::system::ILogger> m_logger;

	nbl::video::CThreadSafeQueueAdapter* queue;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_commandBuffer;

	nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer> m_frameBuffer;

	ResourcesBundle resources;
};

} // nbl::scene::geometrycreator

#endif // _NBL_GEOMETRY_CREATOR_SCENE_H_INCLUDED_