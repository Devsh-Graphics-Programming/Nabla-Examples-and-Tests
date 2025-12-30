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

	auto* const assMan = m_params.assMan.get();
	// handle archive stuff
	const auto relPath = _params.relPath.lexically_normal();
	auto* const system = assMan->getSystem();
	core::stack<IFileArchive*> archiveStack;
	for (auto it=relPath.begin(); it!=relPath.end();)
	{
		const auto ext = (it++)->extension().string();
		if (strcmpi(ext.c_str(),".zip")==0)
		{
			// some N4950 defect makes it impossible
			//const auto archPath = system::path(relPath.begin(),it);
			const auto archPath = std::accumulate(relPath.begin(),it,system::path(),[](const system::path& lhs, const system::path& rhs)->system::path
				{
					return lhs/rhs;
				}
			);
			auto archive = system->openFileArchive(archPath);
			archiveStack.push(archive.get());
			system->mount(std::move(archive));
		}
	}

	const auto relPathStr = relPath.string();
	auto asset = assMan->getAsset(relPathStr,params);
	if (asset.getContents().empty())
	{
		m_params.logger.log(
			"Failed to Load Mitsuba scene from \"%s\" with working directory \"%s\"",
			ILogger::ELL_ERROR,relPathStr.c_str(),_params.workingDirectory.lexically_normal().string().c_str()
		);
		return {};
	}
	m_params.logger.log("Loaded %s",ILogger::ELL_INFO,relPathStr.c_str());

	// now unmount the archives
	for (; !archiveStack.empty(); archiveStack.pop())
		system->unmount(archiveStack.top());
	
	const auto type = asset.getAssetType();
	if (type!=IAsset::E_TYPE::ET_SCENE)
	{
		m_params.logger.log("But did not load an `ICPUScene` type is %S",ILogger::ELL_ERROR,system::to_string(type));
		return {};
	}
	
	const auto* const untypedMeta = asset.getMetadata();
	if (!untypedMeta || strcmpi(untypedMeta->getLoaderName(),CMitsubaMetadata::LoaderName)!=0)
	{
		m_params.logger.log("Loaded an ICPUScene but without `CMistubaMetadata`",ILogger::ELL_ERROR);
		return {};
	}
	const auto* const meta = static_cast<const CMitsubaMetadata*>(untypedMeta);

	//
	core::vector<SLoadResult::SSensor> sensors;
	auto& _sensors = meta->m_global.m_sensors;
	if (_sensors.empty())
	{
		m_params.logger.log("The `CMistubaMetadata` contains no sensors",ILogger::ELL_ERROR);
		return {};
	}
	else
	{
		sensors.resize(_sensors.size());
		m_params.logger.log("Total number of Sensors = %d",ILogger::ELL_INFO,sensors.size());
		//for () // TODO: load the stuff
	}

	// TODO: any CPU-side touch-ups we need to do, like Material IR options

	
	// empty out the cache from individual images and meshes taht are not used by the scene
	assMan->clearAllAssetCache();
	// return
	return {
		.scene = IAsset::castDown<const ICPUScene>(asset.getContents()[0]),
		.sensors = std::move(sensors)
	};
}

}