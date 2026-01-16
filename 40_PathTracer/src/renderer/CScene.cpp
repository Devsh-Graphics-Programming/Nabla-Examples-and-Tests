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
smart_refctd_ptr<CSession> CScene::createSession(const sensor_t& sensor)
{
	const auto& constants = sensor.constants;
	const auto& dynDefaults = sensor.dynamicDefaults;
	const auto& mutDefaults = sensor.mutableDefaults;
	const auto& raygen = mutDefaults.raygen;

	CSession::SConstructionParams params = {
		.scene = smart_refctd_ptr<const CScene>(this),
		.type = raygen.getType()
	};

	// fill uniforms
	{
		const uint16_t2 renderSize(constants.width,constants.height);
		const uint16_t maxPathDepth = hlsl::clamp<uint16_t>(mutDefaults.maxPathDepth,1,0x1u<<SSensorUniforms::MaxPathDepthLog2);
		const uint16_t russianRouletteDepth = hlsl::clamp<uint16_t>(mutDefaults.russianRouletteDepth,1,maxPathDepth);
		params.uniforms = {
			.rcpPixelSize = promote<float32_t2>(1.f)/float32_t2(renderSize),
			.splatting = {}, // TODO
			.renderSize = renderSize,
			.lastCascadeIndex = static_cast<uint16_t>(constants.cascadeCount-1),
			.hideEnvironment = mutDefaults.hideEnvironment,
			.lastPathDepth = static_cast<uint16_t>(maxPathDepth-1),
			.lastNoRussianRouletteDepth = static_cast<uint16_t>(russianRouletteDepth-1)
		};
	}

	//
	params.initDynamics = {
		.ndcToRay = {}, // TODO
		.tMax = mutDefaults.farClip,
		.minSPP = core::min(dynDefaults.samplesNeeded,16), // for later enhancement
		.maxSPP = dynDefaults.samplesNeeded
	};

	//
	{
		const auto reciprocalKappa = 1.f/dynDefaults.kappa;
		params.initResolveConstants = {
			.rwmc = {
				.initialEmin = dynDefaults.Emin,
				.reciprocalBase = 1.f/mutDefaults.cascadeLuminanceBase,
				.reciprocalKappa = reciprocalKappa,
				.colorReliabilityFactor = hlsl::mix(mutDefaults.cascadeLuminanceBase,1.f,reciprocalKappa)
			},
			.cascadeCount = constants.cascadeCount
		};
	}

	return smart_refctd_ptr<CSession>(new CSession(std::move(params)),dont_grab);
}

}