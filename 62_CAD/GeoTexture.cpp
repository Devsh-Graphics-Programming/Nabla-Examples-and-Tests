#include "GeoTexture.h"

bool GeoTextureRenderer::initialize(
		IGPUShader* vertexShader,
		IGPUShader* fragmentShader,
		IGPURenderpass* compatibleRenderPass,
		const smart_refctd_ptr<IGPUBuffer>& globalsBuffer)
{
	video::IGPUDescriptorSetLayout::SBinding bindingsSet0[] = {
		{
			.binding = 0u,
			.type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
			.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
			.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_VERTEX | asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
			.count = 1u,
		},
	};
	m_descriptorSetLayout0 = m_device->createDescriptorSetLayout(bindingsSet0);
	if (!m_descriptorSetLayout0)
		return logFail("Failed to Create Descriptor Layout 0");
			
	video::IGPUDescriptorSetLayout::SBinding bindingsSet1[] = {
		{
			.binding = 0u,
			.type = asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
			.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
			.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
			.count = 1u,
		},
		{
			.binding = 1u,
			.type = asset::IDescriptor::E_TYPE::ET_SAMPLER,
			.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
			.stageFlags = asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
			.count = 1u,
		},
	};
	m_descriptorSetLayout1 = m_device->createDescriptorSetLayout(bindingsSet1);
	if (!m_descriptorSetLayout1)
		return logFail("Failed to Create Descriptor Layout 1");

	const video::IGPUDescriptorSetLayout* const layouts[2u] = { m_descriptorSetLayout0.get(), m_descriptorSetLayout1.get() };

	{
		const uint32_t setCounts[2u] = { 1u, MaxGeoTextures};
		m_descriptorPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, layouts, setCounts);
		if (!m_descriptorPool)
			return logFail("Failed to Create Descriptor Pool");
	}
	

	asset::SPushConstantRange pushConstantRanges[1u] =
	{
		{.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX, .offset = 0ull, .size = sizeof(GeoTextureOBB)},
	};
	m_pipelineLayout = m_device->createPipelineLayout(pushConstantRanges, core::smart_refctd_ptr(m_descriptorSetLayout0), core::smart_refctd_ptr(m_descriptorSetLayout1), nullptr, nullptr);
	
	// Set 0 Create and Bind
	m_descriptorSet0 = m_descriptorPool->createDescriptorSet(smart_refctd_ptr(m_descriptorSetLayout0));
	constexpr uint32_t DescriptorCountSet0 = 1u;
	IGPUDescriptorSet::SDescriptorInfo descriptorInfosSet0[DescriptorCountSet0] = {};
	
	descriptorInfosSet0[0u].info.buffer.offset = 0u;
	descriptorInfosSet0[0u].info.buffer.size = globalsBuffer->getCreationParams().size;
	descriptorInfosSet0[0u].desc = globalsBuffer;

	constexpr uint32_t DescriptorUpdatesCount = DescriptorCountSet0;
	video::IGPUDescriptorSet::SWriteDescriptorSet descriptorUpdates[DescriptorUpdatesCount] = {};

	descriptorUpdates[0u].dstSet = m_descriptorSet0.get();
	descriptorUpdates[0u].binding = 0u;
	descriptorUpdates[0u].arrayElement = 0u;
	descriptorUpdates[0u].count = 1u;
	descriptorUpdates[0u].info = &descriptorInfosSet0[0u];
	m_device->updateDescriptorSets(DescriptorUpdatesCount, descriptorUpdates, 0u, nullptr);

	// Shared Blend Params between pipelines
	//TODO: Where does GeoTexture rendering fit into pipelines, separate renderpass? separate submit? under blending? over blending?
	SBlendParams blendParams = {};
	blendParams.blendParams[0u].srcColorFactor = asset::EBF_SRC_ALPHA;
	blendParams.blendParams[0u].dstColorFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
	blendParams.blendParams[0u].colorBlendOp = asset::EBO_ADD;
	blendParams.blendParams[0u].srcAlphaFactor = asset::EBF_ONE;
	blendParams.blendParams[0u].dstAlphaFactor = asset::EBF_ZERO;
	blendParams.blendParams[0u].alphaBlendOp = asset::EBO_ADD;
	blendParams.blendParams[0u].colorWriteMask = (1u << 4u) - 1u;

	// Create Main Graphics Pipelines 
	{
		IGPUShader::SSpecInfo specInfo[2] = {
			{.shader=vertexShader },
			{.shader=fragmentShader },
		};

		IGPUGraphicsPipeline::SCreationParams params[1] = {};
		params[0].layout = m_pipelineLayout.get();
		params[0].shaders = specInfo;
		params[0].cached = {
			.vertexInput = {},
			.primitiveAssembly = {
				.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST,
			},
			.rasterization = {
				.polygonMode = EPM_FILL,
				.faceCullingMode = EFCM_NONE,
				.depthWriteEnable = false,
			},
			.blend = blendParams,
		};
		params[0].renderpass = compatibleRenderPass;
			
		if (!m_device->createGraphicsPipelines(nullptr,params,&m_graphicsPipeline))
			return logFail("Graphics Pipeline Creation Failed.");
	}

}
