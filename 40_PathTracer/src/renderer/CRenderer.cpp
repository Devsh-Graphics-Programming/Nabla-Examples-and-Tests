// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "renderer/CRenderer.h"
#include "renderer/SAASequence.h"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include "nbl/this_example/builtin/build/spirv/keys.hpp"

namespace nbl::this_example
{
using namespace nbl::asset;
using namespace nbl::video;

//
core::smart_refctd_ptr<CRenderer> CRenderer::create(SCreationParams&& _params)
{
	if (!_params)
		return nullptr;
	SConstructorParams params = {std::move(_params)};

	//
	ILogicalDevice* device = params.utilities->getLogicalDevice();


	// create the layouts
	{
		// one descriptor layout to rule them all
		{
			// bindless textures
			// bindless storage images
			// bindless buffer views
			// bindless buffer storage views
		}

		// but many push constant ranges
		// and first descriptor set layout for 1 UBO to put image indices and BDA (fast swap at will)
	}

	// create the pipelines
	{
		// TODO
	}

	// the renderpass: custom dependencies, but everything else fixed from outside (format, and number of subpasses)
	{
//		params.presentRenderpass = device->createRenderpass();
	}

	// present pipelines
	{
		// TODO
	}

	return core::smart_refctd_ptr<CRenderer>(new CRenderer(std::move(params)),core::dont_grab);
}


core::smart_refctd_ptr<CScene> CRenderer::createScene(CScene::SCreationParams&& _params)
{
	if (!_params)
		return nullptr;
	auto converter = core::smart_refctd_ptr<CAssetConverter>(_params.converter);

	CScene::SConstructorParams params = {std::move(_params)};
	
	// new cache if none provided
	if (!converter)
		converter = CAssetConverter::create({.device=getDevice(),.optimizer={}});

	// build the BLAS and TLAS
	{
	}

	// fill out the render classes but don't init yet

	return core::smart_refctd_ptr<CScene>(new CScene(std::move(params)),core::dont_grab);
}

}