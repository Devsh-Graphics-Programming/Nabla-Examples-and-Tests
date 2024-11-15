#pragma once

using namespace nbl::hlsl;
#include "shaders/geotexture/common.hlsl"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

class GeoTexture : public nbl::core::IReferenceCounted
{
	GeoTextureOBB obbInfo = {};
	smart_refctd_ptr<IGPUDescriptorSet>		descriptorSet; // or index allocated in main geo texture renderer 
	smart_refctd_ptr<IGPUImageView>			texture;
};

class GeoTextureRenderer
{
public:
	static constexpr const char* VertexShaderRelativePath = "../shaders/geotexture/vertex_shader.hlsl";
	static constexpr const char* FragmentShaderRelativePath = "../shaders/geotexture/fragment_shader.hlsl";

	GeoTextureRenderer(smart_refctd_ptr<video::ILogicalDevice>&& device, smart_refctd_ptr<system::ILogger>&& logger)
		: m_device(device)
		, m_logger(logger)
	{}

	bool initialize(
		IGPUShader* vertexShader,
		IGPUShader* fragmentShader,
		IGPURenderpass* compatibleRenderPass,
		const smart_refctd_ptr<IGPUBuffer>& globalsBuffer);

	void createGeoTexture(const nbl::system::path& geoTexturePath); // + OBB Info (center, rotation, aspect ratio from image?)

	void bindPipeline(video::IGPUCommandBuffer* commandBuffer);
	
	void drawGeoTexture(const GeoTexture* geoTexture, video::IGPUCommandBuffer* commandBuffer);

private:
	
	// made it return false so we can save some lines writing `if (failCond) {logFail(); return false;}`
	template<typename... Args>
	inline bool logFail(const char* msg, Args&&... args)
	{
		m_logger->log(msg,system::ILogger::ELL_ERROR,std::forward<Args>(args)...);
		return false;
	}

private:
	smart_refctd_ptr<ILogicalDevice>  m_device;
	smart_refctd_ptr<system::ILogger> m_logger;

	smart_refctd_ptr<IGPUPipelineLayout>			m_pipelineLayout;
	smart_refctd_ptr<IGPUGraphicsPipeline>			m_graphicsPipeline;
	smart_refctd_ptr<IGPUSampler>					m_sampler;
	smart_refctd_ptr<IDescriptorPool>				m_descriptorPool;
	smart_refctd_ptr<IGPUDescriptorSetLayout>		m_descriptorSetLayout0; // globals
	smart_refctd_ptr<IGPUDescriptorSet>				m_descriptorSet0;
	smart_refctd_ptr<IGPUDescriptorSetLayout>		m_descriptorSetLayout1; // contains geo texture
};
