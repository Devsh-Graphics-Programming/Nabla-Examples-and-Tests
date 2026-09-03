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
	if (!m_activeSession || !cb)
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

	bool success = true;
	if (success)
	{
		constexpr auto raytracingStages = PIPELINE_STAGE_FLAGS::RAY_TRACING_SHADER_BIT;
		constexpr auto firstResolveStage = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
	
		using image_barrier_t = IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t;
		core::vector<image_barrier_t> barr;
		{
			constexpr image_barrier_t base = {
				.barrier = {
					.dep = {
						.srcStageMask = raytracingStages,
						.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
						.dstStageMask = firstResolveStage,
						.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
					}
				},
				.subresourceRange = {},
				.newLayout = IGPUImage::LAYOUT::GENERAL
			};
			barr.reserve(4);

			auto enqueueBarrier = [&barr,base](const CSession::SImageWithViews& img)->void
			{
				auto& out = barr.emplace_back(base);
				out.image = img.image.get();
				out.subresourceRange = {
					.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
					.levelCount = 1,
					.layerCount = out.image->getCreationParameters().arrayLayers
				};
			};
			const auto& immutables = m_activeSession->getActiveResources().immutables;
			enqueueBarrier(immutables.rwmcCascades);
			enqueueBarrier(immutables.albedo);
			enqueueBarrier(immutables.normal);
			enqueueBarrier(immutables.motion);
			enqueueBarrier(immutables.mask);
			// this one is slightly different, we barrier against ourselves, and we'll also be writing to it
			enqueueBarrier(immutables.beauty);
			barr.back().barrier.dep.srcStageMask = firstResolveStage;
			barr.back().barrier.dep.dstAccessMask |= ACCESS_FLAGS::SHADER_WRITE_BITS;

		}
		success = cb->pipelineBarrier(asset::EDF_NONE,{.imgBarriers=barr});
	}

	const auto* const layout = m_construction.layout.get();
	// TODO: uimplemented yet

	// compute passes

	return success;
}

}