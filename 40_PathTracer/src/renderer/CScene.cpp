// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/builtin/hlsl/limits.hlsl"

#include "renderer/CRenderer.h"

namespace nbl::this_example
{
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::hlsl;
using namespace nbl::video;

//
smart_refctd_ptr<CSession> CScene::createSession(const CSession::SCreationParams& _params)
{
	if (!_params)
		return nullptr;

	const auto& constants = _params.sensor->constants;
	const auto& dynDefaults = _params.sensor->dynamicDefaults;
	const auto& mutDefaults = _params.sensor->mutableDefaults;
	const auto& raygen = mutDefaults.raygen;

	CSession::SConstructionParams params = {std::move(_params)};
	params.scene = smart_refctd_ptr<const CScene>(this);
	params.cropOffsets = {mutDefaults.cropOffsetX,mutDefaults.cropOffsetY};
	params.cropResolution = {mutDefaults.cropWidth,mutDefaults.cropHeight};
	params.type = raygen.getType();
	params.outputFilePath = dynDefaults.outputFilePath;
	params.postProcess = dynDefaults.postProc;
	
	const uint16_t2 renderSize(constants.width,constants.height);
	assert(all(params.cropOffsets<renderSize));
	assert(all(params.cropOffsets+params.cropResolution<=renderSize));
	assert(params.type!=CSession::sensor_type_e::Env || params.cropResolution==renderSize);

	// fill uniforms
	{
		const uint16_t maxPathDepth = hlsl::clamp<uint16_t>(mutDefaults.maxPathDepth,1,m_construction.renderer->getConstructionParams().getSequenceMaxPathDepth());
		const uint16_t russianRouletteDepth = hlsl::clamp<uint16_t>(mutDefaults.russianRouletteDepth,1,maxPathDepth);
		params.uniforms = {
			.rcpPixelSize = promote<float32_t2>(1.f)/float32_t2(renderSize),
			.splatting = hlsl::rwmc::SPackedSplattingParameters::create(mutDefaults.cascadeLuminanceBase,mutDefaults.cascadeLuminanceStart,constants.cascadeCount),
			.renderSize = renderSize,
			.lastPathDepth = static_cast<uint16_t>(maxPathDepth-1),
			.lastNoRussianRouletteDepth = static_cast<uint16_t>(russianRouletteDepth-1),
			.lastCascadeIndex = static_cast<uint16_t>(constants.cascadeCount-1),
			.hideEnvironment = mutDefaults.hideEnvironment
		};
	}

	//
	params.initDynamics = {
		.invView = mutDefaults.absoluteTransform,
		.ndcToRay = float32_t2x3(mutDefaults.raygen),
		.nearClip = mutDefaults.nearClip,
		.tMax = mutDefaults.farClip,
		.minSPP = core::min(dynDefaults.samplesNeeded,16), // for later enhancement
		.maxSPP = dynDefaults.samplesNeeded,
		.orthoCam = mutDefaults.raygen.getType()==decltype(mutDefaults.raygen)::Type::Ortho
	};

	//
	{
		hlsl::rwmc::SResolveParameters::SCreateParams resolveParams = {};
		resolveParams.minReliableLuma = dynDefaults.Emin;
		resolveParams.kappa = dynDefaults.kappa>0.f ? dynDefaults.kappa:1.f;
		resolveParams.start = mutDefaults.cascadeLuminanceStart;
		resolveParams.base = mutDefaults.cascadeLuminanceBase>0.f ? mutDefaults.cascadeLuminanceBase:1.f;
		resolveParams.sampleCount = core::max<uint32_t>(dynDefaults.samplesNeeded,1u);
		params.initResolveConstants = {
			.resolveParameters = hlsl::rwmc::SResolveParameters::create(resolveParams),
			.cascadeCount = constants.cascadeCount
		};
	}

	return smart_refctd_ptr<CSession>(new CSession(std::move(params)),dont_grab);
}

}
