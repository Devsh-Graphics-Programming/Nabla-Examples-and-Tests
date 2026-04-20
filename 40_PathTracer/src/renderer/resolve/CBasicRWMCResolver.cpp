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

	{
		auto shader = CRenderer::loadPrecompiledShader<"resolve_rwmc">(_params.assMan,device,logger);
		if (!shader)
		{
			logger.log("`CBasicRWMCResolver::create` failed to load RWMC resolve shader!",ILogger::ELL_ERROR);
			return nullptr;
		}

		IGPUComputePipeline::SCreationParams pipelineParams = {};
		pipelineParams.layout = params.layout.get();
		pipelineParams.shader.entryPoint = "resolve";
		pipelineParams.shader.shader = shader.get();
		if (!device->createComputePipelines(nullptr,{&pipelineParams,1},&params.rwmcResolve))
		{
			logger.log("`CBasicRWMCResolver::create` failed to create RWMC resolve pipeline!",ILogger::ELL_ERROR);
			return nullptr;
		}
	}

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
			barr.reserve(2);

			auto enqueueBarrier = [&barr,base](const CSession::SImageWithViews& img, const ACCESS_FLAGS dstAccess)->void
			{
				auto& out = barr.emplace_back(base);
				out.image = img.image.get();
				out.subresourceRange = {
					.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
					.levelCount = 1,
					.layerCount = out.image->getCreationParameters().arrayLayers
				};
				out.barrier.dep.dstAccessMask = dstAccess;
			};
			const auto& immutables = m_activeSession->getActiveResources().immutables;
			enqueueBarrier(immutables.rwmcCascades,ACCESS_FLAGS::SHADER_READ_BITS);
			enqueueBarrier(immutables.beauty,ACCESS_FLAGS::SHADER_WRITE_BITS);
			barr.back().barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::NONE;
			barr.back().barrier.dep.srcAccessMask = ACCESS_FLAGS::NONE;

		}
		success = cb->pipelineBarrier(asset::EDF_NONE,{.imgBarriers=barr});
	}

	const auto* const pipeline = m_construction.rwmcResolve.get();
	const auto& sessionParams = m_activeSession->getConstructionParams();
	const auto renderSize = sessionParams.uniforms.renderSize;
	const uint32_t layerCount = sessionParams.type==CSession::sensor_type_e::Env ? 6u:1u;
	const auto* const ds = m_activeSession->getActiveResources().immutables.ds.get();
	success = success && cb->bindComputePipeline(pipeline);
	success = success && cb->bindDescriptorSets(EPBP_COMPUTE,pipeline->getLayout(),0u,1u,&ds);
	success = success && cb->pushConstants(pipeline->getLayout(),ShaderStage::ESS_COMPUTE,0u,sizeof(sessionParams.initResolveConstants),&sessionParams.initResolveConstants);
	success = success && cb->dispatch(
		(uint32_t(renderSize.x)+ResolveWorkgroupSizeX-1u)/ResolveWorkgroupSizeX,
		(uint32_t(renderSize.y)+ResolveWorkgroupSizeY-1u)/ResolveWorkgroupSizeY,
		layerCount
	);

	return success;
}

}
