// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "renderer/resolve/CBasicRWMCResolver.h"

namespace nbl::this_example
{
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::hlsl;
using namespace nbl::ui;
using namespace nbl::video;

//
smart_refctd_ptr<CBasicRWMCResolver> CBasicRWMCResolver::create(SCreationParams&& _params)
{
	auto logger = _params.renderer->getLogger();
	if (!_params)
	{
		logger.log("`CBasicRWMCResolver::SCreationParams` are invalid!",ILogger::ELL_ERROR);
		return nullptr;
	}
	CBasicRWMCResolver::SConstructorParams params = {std::move(_params)};

	auto* const device = _params.renderer->getDevice();
	{
		const SPushConstantRange pcRange[] = {
			{.stageFlags=ShaderStage::ESS_COMPUTE,.offset=0,.size=sizeof(SResolveConstants)}
		};
		if (!(params.layout=device->createPipelineLayout(pcRange,_params.renderer->getConstructionParams().sensorDSLayout)))
		{
			logger.log("`CBasicRWMCResolver::create` failed to create Pipeline Layout!",ILogger::ELL_ERROR);
			return nullptr;
		}
	}

	// TODO: create all the pipelines!

	return smart_refctd_ptr<CBasicRWMCResolver>(new CBasicRWMCResolver(std::move(params)),dont_grab);
}

bool CBasicRWMCResolver::changeSession_impl()
{
	return true;
}

bool CBasicRWMCResolver::resolve(video::IGPUCommandBuffer* cb, video::IGPUBuffer* scratch)
{
	if (!cb)
		return false;

	switch (m_activeSession->getConstructionParams().mode)
	{
		case CSession::RenderMode::Previs: [[fallthrough]];
		case CSession::RenderMode::Debug:
			return true; // do nothing
		case CSession::RenderMode::Beauty:
			break;
		default:
			return false;
	}

	const auto* const layout = m_construction.layout.get();
	{
		constexpr auto raytracingStages = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT;
		constexpr auto firstResolveStage = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
		// TODO: pipeline barrier from raytracing pipeline to first resolve pass
	}

	// compute passes

	return false; // TODO: uimplemented yet
}

}