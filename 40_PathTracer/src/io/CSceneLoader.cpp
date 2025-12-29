// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "io/CSceneLoader.h"

#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"
#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"

namespace nbl::this_example
{
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ext::MitsubaLoader;

//
smart_refctd_ptr<CSceneLoader> CSceneLoader::create(SCreationParams&& _params)
{
	if (!_params)
		return nullptr;
	SConstructorParams params = {std::move(_params)};

	// add the loaders
	{
		auto* const assMan = params.assMan.get();
		auto* const system = assMan->getSystem();

		bool success = true;
		success = success && assMan->addAssetLoader(make_smart_refctd_ptr<CMitsubaLoader>(smart_refctd_ptr<ISystem>(system)))!=0xdeadbeefu;
		// some of our test scenes won't load without the `.serialized` support
		success = success && assMan->addAssetLoader(make_smart_refctd_ptr<CSerializedLoader>()) != 0xdeadbeefu;

		if (!success)
		{
			params.logger.log("Could not add Mitsuba Asset Loaders", ILogger::ELL_ERROR);
			return nullptr;
		}
	}

	return core::smart_refctd_ptr<CSceneLoader>(new CSceneLoader(std::move(params)),core::dont_grab);
}

auto CSceneLoader::load(SLoadParams&& _params) -> SLoadResult
{
	IAssetLoader::SAssetLoadParams params = {};
	params.workingDirectory = _params.workingDirectory;
	params.logger = m_params.logger.get().get();
	const auto relPath = _params.relPath.lexically_normal().string();
	auto asset = m_params.assMan->getAsset(relPath,params);
	const auto type = asset.getAssetType();
	if (asset.getContents().empty() || type!=IAsset::E_TYPE::ET_SCENE)
	{
		m_params.logger.log(
			"Failed to Load Mitsuba scene from \"%s\" with working directory \"%s\" type is %d",
			ILogger::ELL_ERROR,relPath.c_str(),_params.workingDirectory.lexically_normal().string().c_str(),type // TODO: specialize `system::impl::to_string_helper` for IAsset::E_TYPE
		);
		return {};
	}
	m_params.logger.log("Loaded %s",ILogger::ELL_INFO,relPath.c_str());
	
	const auto* const untypedMeta = asset.getMetadata();
	if (!untypedMeta || strcmpi(untypedMeta->getLoaderName(),CMitsubaMetadata::LoaderName)!=0)
	{
		params.logger.log("Loaded an ICPUScene but without `CMistubaMetadata`",ILogger::ELL_ERROR);
		return {};
	}
	const auto* const meta = static_cast<const CMitsubaMetadata*>(untypedMeta);

	//
	core::vector<SLoadResult::SSensor> sensors;
	auto& _sensors = meta->m_global.m_sensors;
	if (_sensors.empty())
	{
		params.logger.log("The `CMistubaMetadata` contains no sensors",ILogger::ELL_ERROR);
		return {};
	}
	else
	{
		sensors.resize(_sensors.size());
		//for () // TODO: load the stuff
	}

	// TODO: any CPU-side touch-ups we need to do, like Material IR options

	return {
		.scene = IAsset::castDown<const ICPUScene>(asset.getContents()[0]),
		.sensors = std::move(sensors)
	};
}

}