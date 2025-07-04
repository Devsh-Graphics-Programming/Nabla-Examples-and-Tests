// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO move this into nabla

#include "CDrawAABB.h"

using namespace nbl;
using namespace hlsl;

namespace nbl::ext::drawdebug
{

core::smart_refctd_ptr<DrawAABB> DrawAABB::create(SCreationParameters&& params)
{
    return core::smart_refctd_ptr<DrawAABB>(new DrawAABB(std::move(params)));
}

DrawAABB::DrawAABB(SCreationParameters&& _params)
    : m_creationParams(_params)
{
}

DrawAABB::~DrawAABB()
{
}

core::smart_refctd_ptr<video::IGPUPipelineLayout> DrawAABB::createDefaultPipelineLayout(video::ILogicalDevice* device, const asset::SPushConstantRange& pcRange)
{
	return device->createPipelineLayout({ &pcRange , 1 }, nullptr, nullptr, nullptr, nullptr);
}

bool DrawAABB::createDefaultPipeline(core::smart_refctd_ptr<video::IGPUGraphicsPipeline>* pipeline, video::ILogicalDevice* device, video::IGPUPipelineLayout* layout, video::IGPURenderpass* renderpass, video::IGPUGraphicsPipeline::SShaderSpecInfo& vertex, video::IGPUGraphicsPipeline::SShaderSpecInfo& fragment)
{
	video::IGPUGraphicsPipeline::SCreationParams params[1] = {};
	params[0].layout = layout;
	params[0].vertexShader = vertex;
	params[0].fragmentShader = fragment;
	params[0].cached = {
		.primitiveAssembly = {
			.primitiveType = asset::E_PRIMITIVE_TOPOLOGY::EPT_LINE_LIST,
		}
	};
	params[0].renderpass = renderpass;

	return device->createGraphicsPipelines(nullptr, params, pipeline);
}

bool DrawAABB::renderSingle(video::IGPUCommandBuffer* commandBuffer)
{
	commandBuffer->setLineWidth(1.f);
	commandBuffer->draw(24, 1, 0, 0);

	return true;
}

std::array<float32_t3, 24> DrawAABB::getVerticesFromAABB(const core::aabbox3d<float>& aabb)
{
	const auto& pMin = aabb.MinEdge;
	const auto& pMax = aabb.MaxEdge;

	std::array<float32_t3, 24> vertices;
	vertices[0] = float32_t3(pMin.X, pMin.Y, pMin.Z);
	vertices[1] = float32_t3(pMax.X, pMin.Y, pMin.Z);
	vertices[2] = float32_t3(pMin.X, pMin.Y, pMin.Z);
	vertices[3] = float32_t3(pMin.X, pMin.Y, pMax.Z);

	vertices[4] = float32_t3(pMax.X, pMin.Y, pMax.Z);
	vertices[5] = float32_t3(pMax.X, pMin.Y, pMin.Z);
	vertices[6] = float32_t3(pMax.X, pMin.Y, pMax.Z);
	vertices[7] = float32_t3(pMin.X, pMin.Y, pMax.Z);

	vertices[8] = float32_t3(pMin.X, pMax.Y, pMin.Z);
	vertices[9] = float32_t3(pMax.X, pMax.Y, pMin.Z);
	vertices[10] = float32_t3(pMin.X, pMax.Y, pMin.Z);
	vertices[11] = float32_t3(pMin.X, pMax.Y, pMax.Z);

	vertices[12] = float32_t3(pMax.X, pMax.Y, pMax.Z);
	vertices[13] = float32_t3(pMax.X, pMax.Y, pMin.Z);
	vertices[14] = float32_t3(pMax.X, pMax.Y, pMax.Z);
	vertices[15] = float32_t3(pMin.X, pMax.Y, pMax.Z);

	vertices[16] = float32_t3(pMin.X, pMin.Y, pMin.Z);
	vertices[17] = float32_t3(pMin.X, pMax.Y, pMin.Z);
	vertices[18] = float32_t3(pMax.X, pMin.Y, pMin.Z);
	vertices[19] = float32_t3(pMax.X, pMax.Y, pMin.Z);

	vertices[20] = float32_t3(pMin.X, pMin.Y, pMax.Z);
	vertices[21] = float32_t3(pMin.X, pMax.Y, pMax.Z);
	vertices[22] = float32_t3(pMax.X, pMin.Y, pMax.Z);
	vertices[23] = float32_t3(pMax.X, pMax.Y, pMax.Z);

	return vertices;
}

void DrawAABB::addAABB(const core::aabbox3d<float>& aabb, const hlsl::float32_t3& color)
{
	InstanceData instance;
	instance.color = color;

	core::matrix3x4SIMD instanceTransform;
	instanceTransform.setTranslation(core::vectorSIMDf(aabb.MinEdge.X, aabb.MinEdge.Y, aabb.MinEdge.Z, 0));
	const auto diagonal = aabb.MaxEdge - aabb.MinEdge;
	instanceTransform.setScale(core::vectorSIMDf(diagonal.X, diagonal.Y, diagonal.Z));
	memcpy(instance.transform, instanceTransform.pointer(), sizeof(core::matrix3x4SIMD));

	m_instances.push_back(instance);
}

}
