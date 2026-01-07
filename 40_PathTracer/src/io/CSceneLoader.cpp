// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#define _NBL_THIS_EXAMPLE_C_SCENE_LOADER_CPP_
#include "io/CSceneLoader.h"

#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"
#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"

#include "nlohmann/json.hpp"


//
namespace nbl::system::impl
{
template<>
struct to_string_helper<nbl::this_example::CSceneLoader::SLoadResult::SSensor>
{
	public:
		static inline std::string __call(const nbl::this_example::CSceneLoader::SLoadResult::SSensor& value)
		{
			nlohmann::json j;
			j["valid"] = bool(value);

			auto& constants = j["constants"];
			{
				auto& mutableDefaults = j["mutableDefaults"];
				const auto& _mutableDefaults = value.mutableDefaults;
				{
					auto& clipPlanes = mutableDefaults["clipPlanes"];
					for (uint8_t i=0,count=_mutableDefaults.getClipPlaneCount(); i<count; i++)
						clipPlanes.emplace_back(system::to_string(_mutableDefaults.clipPlanes[i]));
				}
			}
			{
				auto& mutableDefaults = j["mutableDefaults"];
				const auto& _mutableDefaults = value.mutableDefaults;
				mutableDefaults["absoluteTransform"] = system::to_string(_mutableDefaults.absoluteTransform);
			}
			{
				auto& dynamicDefaults = j["dynamicDefaults"];
				const auto& _dynamicDefaults = value.dynamicDefaults;
				dynamicDefaults["outputFilePath"] = _dynamicDefaults.outputFilePath;
//				dynamicDefaults[""] = _dynamicDefaults.;
				dynamicDefaults["up"] = system::to_string(_dynamicDefaults.up);
				dynamicDefaults["rotateSpeed"] = _dynamicDefaults.rotateSpeed;
				dynamicDefaults["zoomSpeed"] = _dynamicDefaults.zoomable.speed;
				dynamicDefaults["moveSpeed"] = _dynamicDefaults.moveSpeed;
				dynamicDefaults["samplesNeeded"] = _dynamicDefaults.samplesNeeded;
				dynamicDefaults["kappa"] = _dynamicDefaults.kappa;
				dynamicDefaults["Emin"] = _dynamicDefaults.Emin;
			}

			return j.dump(4);
		}
};
}

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
	auto logger = params.logger = m_params.logger.get().get();

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
		logger.log(
			"Failed to Load Mitsuba scene from \"%s\" with working directory \"%s\"",
			ILogger::ELL_ERROR,relPathStr.c_str(),_params.workingDirectory.lexically_normal().string().c_str()
		);
		return {};
	}
	logger.log("Loaded %s",ILogger::ELL_INFO,relPathStr.c_str());

	// now unmount the archives
	for (; !archiveStack.empty(); archiveStack.pop())
		system->unmount(archiveStack.top());
	
	const auto type = asset.getAssetType();
	if (type!=IAsset::E_TYPE::ET_SCENE)
	{
		logger.log("But did not load an `ICPUScene` type is %S",ILogger::ELL_ERROR,system::to_string(type));
		return {};
	}
	
	const auto* const untypedMeta = asset.getMetadata();
	if (!untypedMeta || strcmpi(untypedMeta->getLoaderName(),CMitsubaMetadata::LoaderName)!=0)
	{
		logger.log("Loaded an ICPUScene but without `CMistubaMetadata`",ILogger::ELL_ERROR);
		return {};
	}
	const auto* const meta = static_cast<const CMitsubaMetadata*>(untypedMeta);

	// TODO: compute/get this from minumum extent of scene
	float sceneSize = 50.f;

	//
	core::vector<SLoadResult::SSensor> sensors;
	auto& _sensors = meta->m_global.m_sensors;
	if (_sensors.empty())
	{
		logger.log("The `CMistubaMetadata` contains no sensors",ILogger::ELL_ERROR);
		return {};
	}
	else
	{
		sensors.resize(_sensors.size());
		logger.log("Total number of Sensors = %d",ILogger::ELL_INFO,sensors.size());
		const bool shouldHaveSensorIdxInFileName = sensors.size()>1;
		const auto mainFileName = relPath.filename();
		for (auto i=0; i<sensors.size(); i++)
		{
			using namespace nbl::hlsl;
			using mts_sensor_t = ext::MitsubaLoader::CElementSensor;
			const auto& _sensor = _sensors[i];
			const char* id = _sensor.id.c_str();
			auto absoluteTransform = float32_t3x4(_sensor.transform.matrix);
			const bool isSpherical = _sensor.type==mts_sensor_t::SPHERICAL;
			const auto& base = _sensor.base;
			const auto& film = _sensor.film;
			auto& constants = sensors[i].constants;
			{
				constants.bloomFilePath = std::filesystem::path(film.denoiserBloomFilePath);
				//
				if (isSpherical)
				{
					constants.width = film.cropWidth;
					constants.height = film.cropHeight;
				}
				else
				{
					constants.width = film.width;
					constants.height = film.height;
				}
				{
					const float det = determinant(float32_t3x3(absoluteTransform));
					if (hlsl::abs(det)<hlsl::numeric_limits<float32_t>::min)
					{
						logger.log("Sensor %s (%d-th in XML) has non invertible singular transformation!",ILogger::ELL_ERROR,id,i);
						constants = {};
						continue;
					}
					constants.rightHandedCamera = det>0.f;
				}
				constants.cascadeCount = hlsl::max(film.cascadeCount,1);
#if 0 // move to raygen compute
				const float aspectRatio = float(constants.width) / float(constants.height);
				auto convertFromXFoV = [=](float xfov) -> float
				{
					float aspectX = tan(radians(xfov)*0.5f);
					return degrees(atan(aspectX/aspectRatio)*2.f);
				};
#endif
			}
			{
				auto& mutableDefaults = sensors[i].mutableDefaults;
				mutableDefaults.absoluteTransform = absoluteTransform;
				auto outClipPlane = mutableDefaults.clipPlanes.begin();
				for (auto i=0; i<mutableDefaults.MaxClipPlanes; i++)
				{
					const auto& plane = base.clipPlanes[i];
					const auto rhs = promote<float32_t3>(0.f);
					if (any(glsl::notEqual<float32_t3>(plane,rhs)))
						*(outClipPlane++) = plane;
				}
				// ignore crops for spherical cameras
				if (!isSpherical)
				{
					mutableDefaults.cropWidth = film.cropWidth;
					mutableDefaults.cropHeight = film.cropHeight;
					mutableDefaults.cropOffsetX = film.cropOffsetX;
					mutableDefaults.cropOffsetY = film.cropOffsetY;
				}
				//
				mutableDefaults.nearClip = base.nearClip;
				mutableDefaults.farClip = base.farClip;
				//
				mutableDefaults.cascadeLuminanceBase = film.cascadeLuminanceBase;
				mutableDefaults.cascadeLuminanceStart = film.cascadeLuminanceStart;
			}
			{
				using dyn_t = SLoadResult::SSensor::SDynamic;
				dyn_t& dynamicDefaults = sensors[i].dynamicDefaults;
				// output file settings
				{
					std::filesystem::path outputFilePath = film.outputFilePath;
					// handle missing output path
					if (outputFilePath.empty())
					{
						const auto extensionStr = fileExtensionFromFormat(film.fileFormat);
						core::string filename = "Render_" + mainFileName.stem().string();
						if(shouldHaveSensorIdxInFileName)
							filename +=  "_Sensor_" + system::to_string(i) + extensionStr.data();
						else
							filename += extensionStr;
						logger.log("Sensor %s (%d-th in XML) has no output path, deduced to \"%s\"",ILogger::ELL_WARNING,id,i,filename.c_str());
						outputFilePath = filename;
					}
					std::string_view extension = "";
					bool invalid = false;
					if (auto ext=outputFilePath.extension().string(); ext.size()>2)
					{
						extension = {ext.begin()+1,ext.end()};
						using format_e = ext::MitsubaLoader::CElementFilm::FileFormat;
						switch (film.fileFormat)
						{
							case format_e::PNG:
								invalid = strcmpi(extension.data(),"png")!=0;
								break;
							case format_e::OPENEXR:
								invalid = strcmpi(extension.data(),"exr")!=0;
								break;
							case format_e::JPEG:
								invalid = strcmpi(extension.data(),"jpg")!=0 && strcmpi(extension.data(),"jpe")!=0 && strcmpi(extension.data(),"jpeg")!=0 &&
									strcmpi(extension.data(),"jif")!=0 && strcmpi(extension.data(),"jfif")!=0 && strcmpi(extension.data(),"jfi")!=0;
								break;
							default:
								break;
						}
					}
					if (invalid)
					{
						logger.log("Sensor %s (%d-th in XML) has invalid <film> format %d or extension \"%s\"",ILogger::ELL_ERROR,id,i,system::to_string(film.fileFormat),extension.data());
						dynamicDefaults = {};
						continue;
					}
					dynamicDefaults.outputFilePath = std::move(outputFilePath);
#if 0 // not part of the loader, do somewhere else
					//
					if (outputFilePath.is_relative())
					{
						logger.log("Film output path is relative: \"%s\"",ILogger::ELL_INFO,outputFilePath.c_str());
						// output relative to output dir
						// or the XML if so wished (walk backward and determine which directories are read only)
					}
#endif
				}
				// post process
				{
					dynamicDefaults.postProc.bloomScale = film.denoiserBloomScale;
					dynamicDefaults.postProc.bloomIntensity = film.denoiserBloomIntensity;
					dynamicDefaults.postProc.tonemapperArgs = std::string(film.denoiserTonemapperArgs);
				}
				// rotate
				{
					dynamicDefaults.rotateSpeed = hlsl::isnan(base.rotateSpeed) ? base.rotateSpeed:dyn_t::DefaultRotateSpeed;
				}
				// move speed
				{
					if (hlsl::isnan(base.moveSpeed))
					{
						dynamicDefaults.moveSpeed = sceneSize*(dyn_t::DefaultMoveSpeed/dyn_t::DefaultSceneSize);
						logger.log("Sensor %s (%d-th in XML) Move Spped is NaN, deducing default from Scene Bounds",ILogger::ELL_WARNING,id,i);
					}
					else
						dynamicDefaults.moveSpeed = base.moveSpeed;
					logger.log("Sensor %s (%d-th in XML) move speed is %f",ILogger::ELL_INFO,id,i,dynamicDefaults.moveSpeed);
				}
				// ignore zoom for spherical cameras
				if (!isSpherical)
				{
#if 0 // TODO
					// Deduce Move and Zoom Speeds if it is nan
					{
						float linearStepZoomSpeed = base.zoomSpeed;
						if (hlsl::isnan(linearStepZoomSpeed))
							linearStepZoomSpeed = sceneSize * (dyn_t::DefaultZoomSpeed / dyn_t::DefaultSceneSize);
					}
					dynamicDefaults.zoomable.speed = ;
#endif
				}
				else if (!hlsl::isnan(base.zoomSpeed))
					logger.log("Sensor %s (%d-th in XML) is SPHERICAL, zoom speed gets ignored!",ILogger::ELL_WARNING,id,i);
				dynamicDefaults.samplesNeeded = _sensor.sampler.sampleCount;
				dynamicDefaults.kappa = constants.cascadeCount<2 ? 0.f:film.rfilter.kappa;
				dynamicDefaults.Emin = film.rfilter.Emin;
				if (film.envmapRegularizationFactor>0.f)
					logger.log("Sensor %s (%d-th in XML) `envmapRegularizationFactor=%f` is deprecated and ignored, we do MIS now",ILogger::ELL_WARNING,id,i,film.envmapRegularizationFactor);
			}
		}
		// log
		for (auto i=0; i<sensors.size(); i++)
		{
			const char* id = _sensors[i].id.c_str();
			logger.log("Sensor %d id=\"%s\" = %s",ILogger::ELL_INFO,sensors.size(),id,system::to_string(sensors[i]).c_str());
		}
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