#pragma once

#include <nabla.h>

using namespace nbl;
using namespace nbl::hlsl;
using namespace core;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

class GeoTexture : public nbl::core::IReferenceCounted
{
	struct Info
	{
		// OBB Information
	} info;
	smart_refctd_ptr<IGPUDescriptorSet>		descriptorSet;
	smart_refctd_ptr<IGPUImageView>			texture;
};

class GeoTextureRenderer : public nbl::core::IReferenceCounted
{
public:
	GeoTextureRenderer(core::smart_refctd_ptr<video::ILogicalDevice> device) 
		: m_device(device)
	{}

	void initialize();

	void drawGeoTexture(const smart_refctd_ptr<GeoTexture>& geoTexture);

private:
	smart_refctd_ptr<ILogicalDevice>				m_device;
	smart_refctd_ptr<IGPUPipelineLayout>			pipelineLayout;
	smart_refctd_ptr<IGPUGraphicsPipeline>			graphicsPipeline;
	smart_refctd_ptr<IGPUSampler>					sampler;
	smart_refctd_ptr<IGPUDescriptorSetLayout>		descriptorSetLayout;
	
	smart_refctd_ptr<IGPUDescriptorSet>		descriptorSet;
	smart_refctd_ptr<IGPUImageView>			texture;
};
