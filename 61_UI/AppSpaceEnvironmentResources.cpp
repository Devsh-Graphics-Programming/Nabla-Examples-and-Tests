#include "app/App.hpp"

#include "app/AppResourceUtilities.hpp"

struct SSpaceEnvironmentTextureSpec final
{
	E_FORMAT format = EF_R16G16B16A16_SFLOAT;
	asset::VkExtent3D extent = {};
	uint32_t mipLevels = 1u;
	uint32_t arrayLayers = 1u;
	std::array<IImage::SBufferCopy, 1u> regions = {};
};

inline SSpaceEnvironmentTextureSpec buildSpaceEnvironmentTextureSpec(const nbl::system::SSpaceEnvBlobHeader& envBlobHeader)
{
	SSpaceEnvironmentTextureSpec textureSpec = {};
	textureSpec.format = EF_R16G16B16A16_SFLOAT;
	textureSpec.extent = { envBlobHeader.width, envBlobHeader.height, 1u };
	textureSpec.regions = {{
		{
			.bufferOffset = 0ull,
			.bufferRowLength = 0u,
			.bufferImageHeight = 0u,
			.imageSubresource = {
				.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
				.mipLevel = 0u,
				.baseArrayLayer = 0u,
				.layerCount = textureSpec.arrayLayers
			},
			.imageOffset = { 0, 0, 0 },
			.imageExtent = textureSpec.extent
		}
	}};
	return textureSpec;
}

bool App::initializeSpaceEnvironmentResources()
{
	nbl::system::SSpaceEnvBlobHeader envBlobHeader = {};
	std::vector<uint8_t> envBlobPayload;
	nbl::system::loadPreferredSpaceEnvBlob(getCameraAppResourceContext(), envBlobHeader, envBlobPayload);
	if (envBlobPayload.empty())
		return logFail("Failed to load space environment blob from available assets.");

	const auto textureSpec = buildSpaceEnvironmentTextureSpec(envBlobHeader);

	const auto createSpaceEnvironmentImage = [&]() -> bool
	{
		IGPUImage::SCreationParams imageParams = {};
		imageParams.type = IGPUImage::ET_2D;
		imageParams.samples = IGPUImage::ESCF_1_BIT;
		imageParams.format = textureSpec.format;
		imageParams.extent = textureSpec.extent;
		imageParams.mipLevels = textureSpec.mipLevels;
		imageParams.arrayLayers = textureSpec.arrayLayers;
		imageParams.flags = IGPUImage::ECF_NONE;
		imageParams.usage = IGPUImage::EUF_SAMPLED_BIT | IGPUImage::EUF_TRANSFER_DST_BIT;
		m_spaceEnvironment.image = m_device->createImage(std::move(imageParams));
		if (!m_spaceEnvironment.image)
			return false;

		m_spaceEnvironment.image->setObjectDebugName("61_UI Space Environment");
		auto memReqs = m_spaceEnvironment.image->getMemoryReqs();
		memReqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
		return m_device->allocate(memReqs, m_spaceEnvironment.image.get()).isValid();
	};

	const auto uploadSpaceEnvironmentImage = [&]() -> bool
	{
		auto uploadResult = m_utils->autoSubmit(
			SIntendedSubmitInfo{ .queue = getGraphicsQueue() },
			[&](SIntendedSubmitInfo& submitInfo) -> bool
			{
				auto* recordingInfo = submitInfo.getCommandBufferForRecording();
				if (!recordingInfo)
					return false;

				auto* cmdbuf = recordingInfo->cmdbuf;
				using image_barrier_t = IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t;
				const image_barrier_t preBarrier[] = {{
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
							.srcAccessMask = ACCESS_FLAGS::NONE,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
							.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
						}
					},
					.image = m_spaceEnvironment.image.get(),
					.subresourceRange = {
						.aspectMask = IGPUImage::EAF_COLOR_BIT,
						.baseMipLevel = 0u,
						.levelCount = textureSpec.mipLevels,
						.baseArrayLayer = 0u,
						.layerCount = textureSpec.arrayLayers
					},
					.oldLayout = IGPUImage::LAYOUT::UNDEFINED,
					.newLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL
				}};
				const IGPUCommandBuffer::SPipelineBarrierDependencyInfo preDep = { .imgBarriers = preBarrier };
				bool success = cmdbuf->pipelineBarrier(asset::EDF_NONE, preDep);
				success = success && m_utils->updateImageViaStagingBuffer(
					submitInfo,
					envBlobPayload.data(),
					textureSpec.format,
					m_spaceEnvironment.image.get(),
					IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
					std::span<const IImage::SBufferCopy>(textureSpec.regions));

				recordingInfo = submitInfo.getCommandBufferForRecording();
				if (!recordingInfo)
					return false;

				cmdbuf = recordingInfo->cmdbuf;
				const image_barrier_t postBarrier[] = {{
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
							.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
							.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT
						}
					},
					.image = m_spaceEnvironment.image.get(),
					.subresourceRange = {
						.aspectMask = IGPUImage::EAF_COLOR_BIT,
						.baseMipLevel = 0u,
						.levelCount = textureSpec.mipLevels,
						.baseArrayLayer = 0u,
						.layerCount = textureSpec.arrayLayers
					},
					.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
					.newLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL
				}};
				const IGPUCommandBuffer::SPipelineBarrierDependencyInfo postDep = { .imgBarriers = postBarrier };
				return success && cmdbuf->pipelineBarrier(asset::EDF_NONE, postDep);
			});
		return uploadResult.copy() == IQueue::RESULT::SUCCESS;
	};

	const auto createSpaceEnvironmentImageViewAndSampler = [&]() -> bool
	{
		IGPUImageView::SCreationParams viewParams = {};
		viewParams.subUsages = IGPUImage::EUF_SAMPLED_BIT;
		viewParams.image = core::smart_refctd_ptr(m_spaceEnvironment.image);
		viewParams.viewType = IGPUImageView::ET_2D;
		viewParams.format = textureSpec.format;
		viewParams.subresourceRange.aspectMask = IGPUImage::EAF_COLOR_BIT;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = textureSpec.mipLevels;
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = textureSpec.arrayLayers;
		m_spaceEnvironment.imageView = m_device->createImageView(std::move(viewParams));
		if (!m_spaceEnvironment.imageView)
			return false;

		IGPUSampler::SParams samplerParams = {};
		samplerParams.MinFilter = ISampler::ETF_LINEAR;
		samplerParams.MaxFilter = ISampler::ETF_LINEAR;
		samplerParams.MipmapMode = ISampler::ESMM_LINEAR;
		samplerParams.TextureWrapU = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
		samplerParams.TextureWrapV = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE;
		samplerParams.TextureWrapW = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE;
		samplerParams.AnisotropicFilter = 0u;
		samplerParams.CompareEnable = false;
		samplerParams.CompareFunc = ISampler::ECO_ALWAYS;
		m_spaceEnvironment.sampler = m_device->createSampler(samplerParams);
		return static_cast<bool>(m_spaceEnvironment.sampler);
	};

	const auto createSpaceEnvironmentPipelineAndDescriptors = [&]() -> bool
	{
		const IGPUDescriptorSetLayout::SBinding bindings[] = {{
			.binding = 0u,
			.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
			.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
			.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
			.count = 1u,
			.immutableSamplers = &m_spaceEnvironment.sampler
		}};
		m_spaceEnvironment.descriptorSetLayout = m_device->createDescriptorSetLayout(std::span{ bindings });
		if (!m_spaceEnvironment.descriptorSetLayout)
			return false;

		const asset::SPushConstantRange pushConstantRange = {
			.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
			.offset = 0u,
			.size = sizeof(SpaceEnvPushConstants)
		};
		auto pipelineLayout = m_device->createPipelineLayout(
			{ &pushConstantRange, 1u },
			core::smart_refctd_ptr(m_spaceEnvironment.descriptorSetLayout),
			nullptr,
			nullptr,
			nullptr);
		if (!pipelineLayout)
			return false;

		const auto spaceFragKey = nbl::this_example::builtin::build::get_spirv_key<"sky_env_fragment">(m_device.get());
		auto fragmentShader = nbl::system::loadPrecompiledShaderFromAppResources(*m_assetMgr, m_logger.get(), spaceFragKey);
		if (!fragmentShader)
			return false;

		nbl::ext::FullScreenTriangle::ProtoPipeline fsTriProto(m_assetMgr.get(), m_device.get(), m_logger.get());
		if (!fsTriProto)
			return false;

		const IGPUPipelineBase::SShaderSpecInfo fragmentSpec = {
			.shader = fragmentShader.get(),
			.entryPoint = "main"
		};
		m_spaceEnvironment.pipeline = fsTriProto.createPipeline(fragmentSpec, pipelineLayout.get(), m_debugScene.renderpass.get());
		if (!m_spaceEnvironment.pipeline)
			return false;

		uint32_t setCount = 1u;
		const IGPUDescriptorSetLayout* setLayouts[] = { m_spaceEnvironment.descriptorSetLayout.get() };
		m_spaceEnvironment.descriptorPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, setLayouts, &setCount);
		if (!m_spaceEnvironment.descriptorPool)
			return false;

		m_spaceEnvironment.descriptorSet = m_spaceEnvironment.descriptorPool->createDescriptorSet(core::smart_refctd_ptr(m_spaceEnvironment.descriptorSetLayout));
		if (!m_spaceEnvironment.descriptorSet)
			return false;

		IGPUDescriptorSet::SDescriptorInfo info = {};
		info.desc = m_spaceEnvironment.imageView;
		info.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

		IGPUDescriptorSet::SWriteDescriptorSet write = {};
		write.dstSet = m_spaceEnvironment.descriptorSet.get();
		write.binding = 0u;
		write.arrayElement = 0u;
		write.count = 1u;
		write.info = &info;
		return m_device->updateDescriptorSets({ &write, 1u }, {});
	};

	if (!createSpaceEnvironmentImage())
		return logFail("Failed to create space environment image.");
	if (!uploadSpaceEnvironmentImage())
		return logFail("Failed to upload space environment map.");
	if (!createSpaceEnvironmentImageViewAndSampler())
		return logFail("Failed to create space environment image view or sampler.");
	if (!createSpaceEnvironmentPipelineAndDescriptors())
		return logFail("Failed to initialize space environment pipeline resources.");

	return true;
}
