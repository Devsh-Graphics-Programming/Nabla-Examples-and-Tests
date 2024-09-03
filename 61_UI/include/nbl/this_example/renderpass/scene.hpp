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

	struct REFERENCE_OBJECT
	{
		struct BINDINGS
		{
			nbl::asset::SBufferBinding<typename TYPES::BUFFER> vertex, index;
		};

		nbl::core::smart_refctd_ptr<typename TYPES::GRAPHICS_PIPELINE> pipeline = nullptr;

		BINDINGS bindings;
		nbl::asset::E_INDEX_TYPE indexType = nbl::asset::E_INDEX_TYPE::EIT_UNKNOWN;
		uint32_t indexCount = {};
	};

	using REFERENCE_DRAW_HOOK = std::pair<REFERENCE_OBJECT, OBJECT_META>;

	nbl::core::smart_refctd_ptr<typename TYPES::RENDERPASS> renderpass;
	std::array<REFERENCE_DRAW_HOOK, EOT_COUNT> objects;
	nbl::asset::SBufferBinding<typename TYPES::BUFFER> ubo;

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

	using THIS_T = RESOURCES_BUILDER<withAssetConverter>;

	RESOURCES_BUILDER(nbl::video::IUtilities* const _utilities, nbl::video::IGPUCommandBuffer* const _commandBuffer, nbl::system::ILogger* const _logger, const nbl::asset::IGeometryCreator* const _geometryCreator)
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
			commandBuffer->reset(nbl::video::IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			commandBuffer->begin(nbl::video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			commandBuffer->beginDebugMarker("Resources builder's buffers upload [manual]");
		}

		using FUNCTOR_T = std::function<bool(void)>;

		auto work = std::to_array
		({
			FUNCTOR_T(std::bind(&THIS_T::createDescriptorSetLayout, this)),
			FUNCTOR_T(std::bind(&THIS_T::createPipelineLayout, this)),
			FUNCTOR_T(std::bind(&THIS_T::createRenderpass, this)),
			FUNCTOR_T(std::bind(&THIS_T::createFramebufferAttachments, this)),
			FUNCTOR_T(std::bind(&THIS_T::createShaders, this)),
			FUNCTOR_T(std::bind(&THIS_T::createGeometries, this)),
			FUNCTOR_T(std::bind(&THIS_T::createViewParametersUboBuffer, this))
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

	inline bool finalize(RESOURCES_BUNDLE& output, nbl::video::CThreadSafeQueueAdapter* transferCapableQueue)
	{
		EXPOSE_NABLA_NAMESPACES();

		auto completed = utilities->getLogicalDevice()->createSemaphore(0u);

		std::array<IQueue::SSubmitInfo::SSemaphoreInfo, 1u> signals;
		{
			auto& signal = signals.front();
			signal.value = 0x45;
			signal.stageMask = bitflag(PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS);
			signal.semaphore = completed.get();
		}

		std::array<IQueue::SSubmitInfo::SCommandBufferInfo, 1u> commandBuffers = {};
		{
			commandBuffers.front().cmdbuf = commandBuffer;
		}

		if constexpr (withAssetConverter)
		{
			// note that asset converter records basic transfer uploads itself, we only begin the recording with ONE_TIME_SUBMIT_BIT
			commandBuffer->reset(nbl::video::IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			commandBuffer->begin(nbl::video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			commandBuffer->beginDebugMarker("Resources builder's buffers upload [asset converter]");

			// asset converter - scratch at this point has ready to convert cpu resources
			smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = utilities->getLogicalDevice(),.optimizer = {} });
			CAssetConverter::SInputs inputs = {};
			inputs.logger = logger;

			struct PROXY_CPU_HOOKS
			{
				using OBJECTS_SIZE = std::tuple_size<decltype(scratch.objects)>;

				std::array<ICPURenderpass*, 1u> renderpass;
				std::array<ICPUGraphicsPipeline*, OBJECTS_SIZE::value> pipelines;
				std::array<ICPUBuffer*, OBJECTS_SIZE::value * 2u + 1u > buffers;
				std::array<ICPUImageView*, 2u> attachments;
			} hooks;

			enum E_ATTACHMENT_ID
			{
				EAI_COLOR = 0u,
				EAI_DEPTH = 1u,

				EAI_COUNT
			};
			
			// gather CPU assets into span memory views
			{ 
				hooks.renderpass.front() = scratch.renderpass.get();
				for (uint32_t i = 0u; i < hooks.pipelines.size(); ++i)
				{
					auto& [reference, meta] = scratch.objects[static_cast<E_OBJECT_TYPE>(i)];
					hooks.pipelines[i] = reference.pipeline.get();

					// [[ [vertex, index] [vertex, index] [vertex, index] ... [ubo] ]]
					hooks.buffers[2u * i + 0u] = reference.bindings.vertex.buffer.get();
					hooks.buffers[2u * i + 1u] = reference.bindings.index.buffer.get();
				}
				hooks.buffers.back() = scratch.ubo.buffer.get();
				hooks.attachments[EAI_COLOR] = scratch.attachments.color.get();
				hooks.attachments[EAI_DEPTH] = scratch.attachments.depth.get();
			}

			// assign the CPU hooks to converter's inputs
			{
				std::get<CAssetConverter::SInputs::asset_span_t<ICPURenderpass>>(inputs.assets) = hooks.renderpass;
				std::get<CAssetConverter::SInputs::asset_span_t<ICPUGraphicsPipeline>>(inputs.assets) = hooks.pipelines;
				std::get<CAssetConverter::SInputs::asset_span_t<ICPUBuffer>>(inputs.assets) = hooks.buffers;
				// std::get<CAssetConverter::SInputs::asset_span_t<ICPUImageView>>(inputs.assets) = hooks.attachments; // NOTE: THIS IS NOT IMPLEMENTED YET IN CONVERTER!
			}

			// reserve and create the GPU object handles
			auto reservation = converter->reserve(inputs);
			{
				auto prepass = [&]<typename ASSET_TYPE>(const auto& references) -> bool
				{
					// retrieve the reserved handles
					auto objects = reservation.getGPUObjects<ASSET_TYPE>();

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
								logger->log("Failed to convert a CPU object to GPU!", nbl::system::ILogger::ELL_ERROR);
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
			}

			auto semaphore = utilities->getLogicalDevice()->createSemaphore(0u);

			CAssetConverter::SConvertParams params = {};
			params.utilities = utilities;
			params.transfer.queue = transferCapableQueue;
			params.transfer.scratchSemaphore.semaphore = semaphore.get();
			params.transfer.scratchSemaphore.value = 0u; // the initial signal value is incremented by one for each submit so let's start with 0u, we will have only one submit so our scratch signal value will be 1u
			params.transfer.scratchSemaphore.stageMask = bitflag(PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS);
			params.transfer.commandBuffers = commandBuffers;

			// basically it records all data uploads, but remember for gpu objects to be finalized you also have to submit the conversion afterwards to the queue!
			auto result = reservation.convert(params);

			if (!result)
			{
				logger->log("Failed to record assets conversion!", nbl::system::ILogger::ELL_ERROR);
				return false;
			}

			// submit the work to queue, note we will have 2 semaphores under the hood, scratch + our signal
			auto future = result.submit(signals);

			// and note that this operator actually blocks for semaphores!
			if (!future)
			{
				logger->log("Failed to await submission feature!", nbl::system::ILogger::ELL_ERROR);
				return false;
			}

			// assign base gpu objects to output
			auto& base = static_cast<RESOURCES_BUNDLE::BASE_T&>(output);
			{
				auto&& [renderpass, pipelines, buffers] = std::make_tuple(reservation.getGPUObjects<ICPURenderpass>().front().value, reservation.getGPUObjects<ICPUGraphicsPipeline>(), reservation.getGPUObjects<ICPUBuffer>());
				{
					base.renderpass = renderpass;
					for (uint32_t i = 0u; i < pipelines.size(); ++i)
					{
						const auto type = static_cast<E_OBJECT_TYPE>(i);
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
					
					/*
						// base.attachments.color = attachments[EAI_COLOR].value;
						// base.attachments.depth = attachments[EAI_DEPTH].value;

						note conversion of image views is not yet supported by the asset converter 
						- it's complicated, we have to kinda temporary ignore DRY a bit here to not break the design which is correct

						TEMPORARY: we patch attachments by allocating them ourselves here given cpu instances & parameters
						TODO: remove following code once asset converter works with image views & update stuff
					*/

					for (uint32_t i = 0u; i < EAI_COUNT; ++i)
					{
						const auto* reference = hooks.attachments[i];
						auto& out = (i == EAI_COLOR ? base.attachments.color : base.attachments.depth);

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
		else
		{
			const nbl::video::IQueue::SSubmitInfo infos [] =
			{
				{
					.waitSemaphores = {},
					.commandBuffers = commandBuffers, // note that here our command buffer is already recorded!
					.signalSemaphores = signals
				}
			};

			if (transferCapableQueue->submit(infos) != nbl::video::IQueue::RESULT::SUCCESS)
			{
				logger->log("Failed to submit transfer upload operations!", nbl::system::ILogger::ELL_ERROR);
				return false;
			}

			const nbl::video::ISemaphore::SWaitInfo info [] =
			{ {
				.semaphore = completed.get(),
				.value = 0x45
			} };

			utilities->getLogicalDevice()->blockForSemaphores(info);

			static_cast<RESOURCES_BUNDLE::BASE_T&>(output) = static_cast<RESOURCES_BUNDLE::BASE_T&>(scratch); // scratch has all ready to use allocated gpu resources with uploaded memory so now just assign resources to base output
		}

		// base gpu resources are created at this point and stored into output, let's create left gpu objects
		
		// descriptor set
		{
			auto* descriptorSetLayout = output.objects.front().first.pipeline->getLayout()->getDescriptorSetLayout(1u); // let's just take any, the layout is shared across all possible pipelines

			const nbl::video::IGPUDescriptorSetLayout* const layouts[] = { nullptr, descriptorSetLayout };
			const uint32_t setCounts[] = { 0u, 1u };

			output.descriptorPool = utilities->getLogicalDevice()->createDescriptorPoolForDSLayouts(nbl::video::IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);

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
				info.desc = nbl::core::smart_refctd_ptr(output.ubo.buffer);
				info.info.buffer.offset = output.ubo.offset;
				info.info.buffer.size = output.ubo.buffer->getSize();
			}

			write.info = &info;

			if(!utilities->getLogicalDevice()->updateDescriptorSets(1u, &write, 0u, nullptr))
			{
				logger->log("Could not write descriptor set!", nbl::system::ILogger::ELL_ERROR);
				return false;
			}
		}

		return true;
	}

private:
	bool createDescriptorSetLayout()
	{
		EXPOSE_NABLA_NAMESPACES();

		typename TYPES::DESCRIPTOR_SET_LAYOUT::SBinding bindings[] =
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

		return true;
	}

	bool createPipelineLayout()
	{
		EXPOSE_NABLA_NAMESPACES();

		const std::span<const SPushConstantRange> range = {};

		scratch.pipelineLayout = create<typename TYPES::PIPELINE_LAYOUT>(range, nullptr, smart_refctd_ptr(scratch.descriptorSetLayout), nullptr, nullptr);

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

		return true;
	}

	bool createFramebufferAttachments()
	{
		EXPOSE_NABLA_NAMESPACES();

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

				if constexpr (withAssetConverter)
				{
					auto dummyBuffer = make_smart_refctd_ptr<ICPUBuffer>(FRAMEBUFFER_W * FRAMEBUFFER_H * getTexelOrBlockBytesize<format>());
					dummyBuffer->setContentHash(dummyBuffer->computeContentHash());

					auto regions = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
					auto& region = regions->front();

					region.imageSubresource = { .aspectMask = ASPECT, .mipLevel = 0u, .baseArrayLayer = 0u, .layerCount = 0u };
					region.bufferOffset = 0u;
					region.bufferRowLength = IImageAssetHandlerBase::calcPitchInBlocks(FRAMEBUFFER_W, getTexelOrBlockBytesize<format>());
					region.bufferImageHeight = 0u;
					region.imageOffset = { 0u, 0u, 0u };
					region.imageExtent = { FRAMEBUFFER_W, FRAMEBUFFER_H, 1u };

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

				auto params = typename TYPES::IMAGE_VIEW::SCreationParams(
				{
					.flags = TYPES::IMAGE_VIEW::ECF_NONE,
					.subUsages = USAGE,
					.image = std::move(image),
					.viewType = TYPES::IMAGE_VIEW::ET_2D,
					.format = format,
					.subresourceRange = { .aspectMask = ASPECT, .baseMipLevel = 0u, .levelCount = 1u, .baseArrayLayer = 0u, .layerCount = 1u }
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

		return true;
	}

	bool createShaders()
	{
		EXPOSE_NABLA_NAMESPACES();

		auto createShader = [&]<nbl::core::StringLiteral virtualPath>(IShader::E_SHADER_STAGE stage, smart_refctd_ptr<typename TYPES::SHADER>& outShader) -> smart_refctd_ptr<typename TYPES::SHADER>
		{
			// TODO: use SPIRV loader & our ::system ns to get those cpu shaders, do not create myself (shit I forgot it exists)

			const nbl::system::SBuiltinFile& in = ::geometry::creator::spirv::builtin::get_resource<virtualPath>();
			const auto buffer = make_smart_refctd_ptr<CCustomAllocatorCPUBuffer<null_allocator<uint8_t>, true> >(in.size, (void*)in.contents, adopt_memory);
			auto shader = nbl::core::make_smart_refctd_ptr<ICPUShader>(smart_refctd_ptr(buffer), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, ""); // must create cpu instance regardless underlying type

			if constexpr (withAssetConverter)
			{
				buffer->setContentHash(buffer->computeContentHash());
				outShader = std::move(shader);
			}
			else
				outShader = create<typename TYPES::SHADER>(shader.get()); // note: dependency between cpu object instance & gpu object creation, not sure if its our API design failure or maybe I'm just thinking too much

			return outShader;
		};

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
					constexpr static auto VERTEX_USAGE = bitflag(IBUFFER::EUF_VERTEX_BUFFER_BIT) | IBUFFER::EUF_TRANSFER_DST_BIT | IBUFFER::EUF_INLINE_UPDATE_VIA_CMDBUF;
					obj.bindings.vertex.offset = 0u;
						
					auto iBuffer = smart_refctd_ptr(inGeometry.data.indexBuffer.buffer); // no offset
					constexpr static auto INDEX_USAGE = bitflag(IBUFFER::EUF_INDEX_BUFFER_BIT) | IBUFFER::EUF_VERTEX_BUFFER_BIT | IBUFFER::EUF_TRANSFER_DST_BIT | IBUFFER::EUF_INLINE_UPDATE_VIA_CMDBUF;
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
						auto vertexBuffer = create<typename TYPES::BUFFER>(typename TYPES::BUFFER::SCreationParams({ .size = vBuffer->getSize(), .usage = VERTEX_USAGE }));
						auto indexBuffer = iBuffer ? create<typename TYPES::BUFFER>(typename TYPES::BUFFER::SCreationParams({ .size = iBuffer->getSize(), .usage = INDEX_USAGE })) : nullptr;

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

		// note: similar issue like with shaders, this time with cpu-gpu constructors differing in arguments
		using IBUFFER = nbl::asset::IBuffer; // seems to be ambigous, both asset & core namespaces has IBuffer
		constexpr static auto UBO_USAGE = bitflag(IBUFFER::EUF_UNIFORM_BUFFER_BIT) | IBUFFER::EUF_TRANSFER_DST_BIT | IBUFFER::EUF_INLINE_UPDATE_VIA_CMDBUF;

		if constexpr (withAssetConverter)
		{
			auto uboBuffer = make_smart_refctd_ptr<ICPUBuffer>(sizeof(SBasicViewParameters));
			uboBuffer->addUsageFlags(UBO_USAGE);
			uboBuffer->setContentHash(uboBuffer->computeContentHash());
			scratch.ubo = { .offset = 0u, .buffer = std::move(uboBuffer) };
		}
		else
		{
			const auto mask = utilities->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

			auto uboBuffer = create<typename TYPES::BUFFER>(typename TYPES::BUFFER::SCreationParams({ .size = sizeof(SBasicViewParameters), .usage = UBO_USAGE }));

			if (!uboBuffer)
				return false;

			for (auto it : { uboBuffer })
			{
				video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = it->getMemoryReqs();
				reqs.memoryTypeBits &= mask;

				utilities->getLogicalDevice()->allocate(reqs, it.get());
			}

			scratch.ubo = { .offset = 0u, .buffer = std::move(uboBuffer) };
		}

		return true;
	}

	template<typename T, typename... Args>
	inline nbl::core::smart_refctd_ptr<T> create(Args&&... args) requires RESOURCE_TYPE_CONCEPT<T, TYPES>
	{
		if constexpr (withAssetConverter)
			return nbl::core::make_smart_refctd_ptr<T>(std::forward<Args>(args)...); // TODO: cases where our api requires to call ::create(...) instead directly calling "make smart pointer" could be here handled instead of in .build method
		else
			if constexpr (std::same_as<T, typename TYPES::DESCRIPTOR_SET_LAYOUT>)
				return utilities->getLogicalDevice()->createDescriptorSetLayout(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::PIPELINE_LAYOUT>)
				return utilities->getLogicalDevice()->createPipelineLayout(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::RENDERPASS>)
				return utilities->getLogicalDevice()->createRenderpass(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::IMAGE_VIEW>)
				return utilities->getLogicalDevice()->createImageView(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::IMAGE>)
				return utilities->getLogicalDevice()->createImage(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::BUFFER>)
				return utilities->getLogicalDevice()->createBuffer(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::SHADER>)
				return utilities->getLogicalDevice()->createShader(std::forward<Args>(args)...);
			else if constexpr (std::same_as<T, typename TYPES::GRAPHICS_PIPELINE>)
			{
				bool status = utilities->getLogicalDevice()->createGraphicsPipelines(std::forward<Args>(args)...);
				return nullptr; // I assume caller with use output from forwarded args, another inconsistency in our api imho
			}
			else
				return nullptr; // TODO: should static assert
	}

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

	// TODO: we could make those params templated with default values like below
	_NBL_STATIC_INLINE_CONSTEXPR auto FRAMEBUFFER_W = 1280u, FRAMEBUFFER_H = 720u;
	_NBL_STATIC_INLINE_CONSTEXPR auto COLOR_FBO_ATTACHMENT_FORMAT = nbl::asset::EF_R8G8B8A8_SRGB, DEPTH_FBO_ATTACHMENT_FORMAT = nbl::asset::EF_D16_UNORM;
	_NBL_STATIC_INLINE_CONSTEXPR auto SAMPLES = nbl::video::IGPUImage::ESCF_1_BIT;

	RESOURCES_BUNDLE_SCRATCH scratch;

	nbl::video::IUtilities* const utilities;
	nbl::video::IGPUCommandBuffer* const commandBuffer;
	nbl::system::ILogger* const logger;
	GEOMETRIES_CPU geometries;
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

		semaphore.progress = m_utilities->getLogicalDevice()->createSemaphore(semaphore.startedValue);
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

		nbl::asset::SBufferBinding<const nbl::video::IGPUBuffer> vertex = hook.bindings.vertex, index = hook.bindings.index;

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
		range.buffer = nbl::core::smart_refctd_ptr(resources.ubo.buffer);
		range.size = resources.ubo.buffer->getSize();

		m_commandBuffer->updateBuffer(range, &object.viewParameters);
	}

	inline decltype(auto) getResources()
	{
		return (resources); // note: do not remove "()" - it makes the return type lvalue reference instead of copy 
	}

private:
	template<typename CREATE_WITH = CREATE_RESOURCES_DIRECTLY_WITH_DEVICE> // TODO: enforce constraints, only those 2 above are valid
	CScene(nbl::core::smart_refctd_ptr<nbl::video::IUtilities> _utilities, nbl::core::smart_refctd_ptr<nbl::system::ILogger> _logger, nbl::video::CThreadSafeQueueAdapter* _graphicsQueue, const nbl::asset::IGeometryCreator* _geometryCreator, CREATE_WITH createWith = {})
		: m_utilities(nbl::core::smart_refctd_ptr(_utilities)), m_logger(nbl::core::smart_refctd_ptr(_logger)), queue(_graphicsQueue)
	{
		using BUILDER = typename CREATE_WITH::BUILDER;

		bool status = createCommandBuffer();
		BUILDER builder(m_utilities.get(), m_commandBuffer.get(), m_logger.get(), _geometryCreator);

		// gpu resources
		if (builder.build())
		{
			if (!builder.finalize(resources, queue))
				m_logger->log("Could not finalize resource objects to gpu objects!", nbl::system::ILogger::ELL_ERROR);
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

			m_frameBuffer = m_utilities->getLogicalDevice()->createFramebuffer(std::move(params));

			if (!m_frameBuffer)
			{
				m_logger->log("Could not create frame buffer!", nbl::system::ILogger::ELL_ERROR);
				return;
			}
		}
	}

	bool createCommandBuffer()
	{
		m_commandPool = m_utilities->getLogicalDevice()->createCommandPool(queue->getFamilyIndex(), nbl::video::IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

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

	nbl::core::smart_refctd_ptr<nbl::video::IUtilities> m_utilities;
	nbl::core::smart_refctd_ptr<nbl::system::ILogger> m_logger;

	nbl::video::CThreadSafeQueueAdapter* queue;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> m_commandPool; // TODO: decide if we should reuse main app's pool to allocate the cmd
	nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_commandBuffer;

	nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer> m_frameBuffer;

	RESOURCES_BUNDLE resources;
};

#endif // __NBL_THIS_EXAMPLE_SCENE_H_INCLUDED__