// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include <chrono>
#include <filesystem>
#include <fstream>

#include "../common/QToQuitEventReceiver.h"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"
#include "CommandLineHandler.hpp"

#include "CSceneNodeAnimatorCameraModifiedMaya.h"
#include "Renderer.h"

using namespace nbl;
using namespace core;

class RaytracerExampleEventReceiver : public nbl::IEventReceiver
{
	public:
		RaytracerExampleEventReceiver() : running(true), renderingBeauty(true)
		{
			resetKeys();
		}

		bool OnEvent(const nbl::SEvent& event)
		{
			if (event.EventType == nbl::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
			{
				switch (event.KeyInput.Key)
				{
					case ResetKey:
						resetViewKeyPressed = true;
						break;
					case NextKey:
						nextKeyPressed = true;
						break;
					case PreviousKey:
						previousKeyPressed = true;
						break;
					case ScreenshotKey:
						screenshotKeyPressed = true;
						break;
					case LogProgressKey:
						logProgressKeyPressed = true;
						break;
					case SkipKey:
						skipKeyPressed = true;
						break;
					case BeautyKey:
						renderingBeauty = !renderingBeauty;
						break;
					case ReloadKey:
						reloadKeyPressed = true;
						break;
					case OverloadCameraKey:
						overloadCameraKeyPressed = true;
						break;
					case QuitKey:
						running = false;
						return true;
					default:
						break;
				}
			}

			return false;
		}
		
		inline bool keepOpen() const { return running; }

		inline bool isSkipKeyPressed() const { return skipKeyPressed; }
		
		inline bool isResetViewPressed() const { return resetViewKeyPressed; }
		
		inline bool isNextPressed() const { return nextKeyPressed; }

		inline bool isPreviousPressed() const { return previousKeyPressed; }

		inline bool isScreenshotKeyPressed() const { return screenshotKeyPressed; }

		inline bool isLogProgressKeyPressed() const { return logProgressKeyPressed; }

		inline bool isRenderingBeauty() const { return renderingBeauty; }

		inline bool isReloadKeyPressed() const { return reloadKeyPressed; }

		inline bool isOverloadCameraKeyPressed() const { return overloadCameraKeyPressed; }

		inline void resetKeys()
		{
			skipKeyPressed = false;
			resetViewKeyPressed = false;
			nextKeyPressed = false;
			previousKeyPressed = false;
			screenshotKeyPressed = false;
			logProgressKeyPressed = false;
			reloadKeyPressed = false;
			overloadCameraKeyPressed = false;
		}

	private:
		static constexpr nbl::EKEY_CODE QuitKey = nbl::KEY_KEY_Q;
		static constexpr nbl::EKEY_CODE SkipKey = nbl::KEY_END;
		static constexpr nbl::EKEY_CODE ResetKey = nbl::KEY_HOME;
		static constexpr nbl::EKEY_CODE NextKey = nbl::KEY_PRIOR; // PAGE_UP
		static constexpr nbl::EKEY_CODE PreviousKey = nbl::KEY_NEXT; // PAGE_DOWN
		static constexpr nbl::EKEY_CODE ScreenshotKey = nbl::KEY_KEY_P;
		static constexpr nbl::EKEY_CODE LogProgressKey = nbl::KEY_KEY_L;
		static constexpr nbl::EKEY_CODE BeautyKey = nbl::KEY_KEY_B;
		static constexpr nbl::EKEY_CODE ReloadKey = nbl::KEY_F5;
		static constexpr nbl::EKEY_CODE OverloadCameraKey = nbl::KEY_KEY_C;

		bool running;
		bool renderingBeauty;

		bool skipKeyPressed;
		bool resetViewKeyPressed;
		bool nextKeyPressed;
		bool previousKeyPressed;
		bool screenshotKeyPressed;
		bool logProgressKeyPressed;
		bool reloadKeyPressed;
		bool overloadCameraKeyPressed;
};


int main(int argc, char** argv)
{
	std::vector<std::string> arguments;
	if (argc>1)
	{
		for (auto i = 1ul; i < argc; ++i)
			arguments.emplace_back(argv[i]);
	}

	{
		CommandLineHandler cmdHandler = CommandLineHandler(arguments);

		applicationState.processSensorsBehaviour = cmdHandler.getProcessSensorsBehaviour();
		applicationState.startSensorID = cmdHandler.getSensorID();
		applicationState.isInteractiveMode = (applicationState.processSensorsBehaviour == ProcessSensorsBehaviour::PSB_INTERACTIVE_AT_SENSOR);
		applicationState.isDenoiseDeferred = cmdHandler.getDeferredDenoiseFlag();

		auto sceneDir = cmdHandler.getSceneDirectory();

		std::string filePath = (sceneDir.size() >= 1) ? sceneDir[0] : ""; // zip or xml
		std::string extraPath = (sceneDir.size() >= 2) ? sceneDir[1] : "";; // xml in zip
		if (core::hasFileExtension(io::path(filePath.c_str()), "zip", "ZIP"))
		{
			applicationState.zipPath = filePath;
			applicationState.xmlPath = extraPath;
		}
		else
		{
			applicationState.xmlPath = filePath;
		}
		// After this, there could be 4 cases:
		// 1. zipPath filled, xmlPath filled -> load xml from zip
		// 2. zipPath empty, xmlPath filled -> directly load xml
		// 3. zipPath filled, xmlPath empty -> load chosen (or default to first) xml from zip
		// 4. zipPath empty, xmlPath empty -> try to restore state after asking for the user to choose files
	}

	bool takeScreenShots = true;

// DEVICE CREATION EMITTED

	//
	asset::SAssetBundle meshes = {};
	core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata> globalMeta;
	{

// LOADER ADDITION EMITTED

		if (applicationState.zipPath.empty() && applicationState.xmlPath.empty() && !applicationIsReloaded)
		{
			pfd::message("Choose file to load", "Choose mitsuba XML file to load or ZIP containing an XML. If you cancel or choosen file fails to load, previous state of the application will be restored, if available.", pfd::choice::ok);
			pfd::open_file file("Choose XML or ZIP file", "../../media/mitsuba", { "All Supported Formats", "*.xml *.zip", "ZIP files (.zip)", "*.zip", "XML files (.xml)", "*.xml" });
			if (!file.result().empty())
			{
				if (core::hasFileExtension(io::path(file.result()[0].c_str()), "zip", "ZIP"))
					applicationState.zipPath = file.result()[0];
				else
					applicationState.xmlPath = file.result()[0];
			}
		}

		auto loadScene = [&device, &am, &fs](const std::string& _zipPath, std::string& _xmlPath, std::string& _mainFileName) -> asset::SAssetBundle
		{
			asset::SAssetBundle result = {};

// ADD ARCHIVE AND VALIDATION EMITTED


				auto flist = arch->getFileList();
				if (!flist)
					return {};

				auto files = flist->getFiles();
				for (auto it = files.begin(); it != files.end(); )
				{
					if (core::hasFileExtension(it->FullName, "xml", "XML"))
						it++;
					else
						it = files.erase(it);
				}
				if (files.size() == 0u)
				{
					printf("[ERROR]: No XML files found in the ZIP archive!\n");
					return result;
				}

				if (_xmlPath.empty())
				{
					uint32_t chosen = 0xffffffffu;

					// Only ask for choosing a file when there are multiple of them.
					if (files.size() > 1)
					{
						printf("Choose File (0-%llu):\n", files.size() - 1ull);
						for (auto i = 0u; i < files.size(); i++)
							printf("%u: %s\n", i, files[i].FullName.c_str());

						// std::cin with timeout
						{
							std::atomic<bool> started = false;
							std::thread cin_thread([&chosen, &started]()
							{
								started = true;
								std::cin >> chosen;
							});
							cin_thread.detach();
							const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10u);
							while (!started || chosen == 0xffffffffu && std::chrono::steady_clock::now() < end) {}
						}
					}
					else
					{
						printf("[INFO]: The only available XML in the ZIP is selected.\n");
					}

					if (chosen >= files.size())
						chosen = 0u;

					_xmlPath = std::string(files[chosen].FullName.c_str());
				}
				else
				{
					// Verify that the passed XML path is in the ZIP archive.
					bool found = false;
					for (auto it = files.begin(); it != files.end(); it++)
					{
						// In `PersistentState` we save the full XML name `_xmlPath` could also be that.
						if ((_xmlPath == std::string(it->Name.c_str())) || (_xmlPath == std::string(it->FullName.c_str())))
						{
							_xmlPath = std::string(it->FullName.c_str());
							found = true;
							break;
						}
					}

					if (!found)
					{
						printf("[ERROR]: Cannot find requested XML file (%s) in the ZIP (%s)\n", _xmlPath.c_str(), _zipPath.c_str());
						return result;
					}
				}

				_mainFileName += std::string("_") + std::filesystem::path(_xmlPath.c_str()).filename().replace_extension().string();


			printf("[INFO]: Loading XML file: %s\n", _xmlPath.c_str());

			asset::CQuantNormalCache* qnc = am->getMeshManipulator()->getQuantNormalCache();

			//! read cache results -- speeds up mesh generation
			qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");
			//! load the mitsuba scene
			result = am->getAsset(_xmlPath, {});
			//! cache results -- speeds up mesh generation on second run
			qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");
			
			return result;
		};

		meshes = loadScene(applicationState.zipPath, applicationState.xmlPath, mainFileName);

// APPLICATION RESTORE OMITTED

	}
	

	struct SensorData
	{
// ...

		void resetInteractiveCamera()
		{
			core::vectorSIMDf cameraTarget = staticCamera->getTarget();
			core::vector3df cameraTargetVec3f(cameraTarget.x, cameraTarget.y, cameraTarget.z); // I have to do this because of inconsistencies in using vectorSIMDf and vector3df in code most places.

			interactiveCamera->setPosition(staticCamera->getPosition());
			interactiveCamera->setTarget(cameraTargetVec3f);
			interactiveCamera->setUpVector(staticCamera->getUpVector());
			interactiveCamera->setLeftHanded(staticCamera->getLeftHanded());
			interactiveCamera->setProjectionMatrix(staticCamera->getProjectionMatrix());
		
			core::vectorSIMDf cameraPos; cameraPos.set(staticCamera->getPosition());
			auto modifiedMayaAnim = getInteractiveCameraAnimator();
			modifiedMayaAnim->setZoomAndRotationBasedOnTargetAndPosition(cameraPos, cameraTarget);
		}
	};


// ...

	auto extractAndAddToSensorData = [&](const ext::MitsubaLoader::CElementSensor& sensor, uint32_t idx) -> bool
	{
		SensorData mainSensorData = {};


// ...



		if (mainSensorData.type == ext::MitsubaLoader::CElementSensor::Type::SPHERICAL)
		{
#ifdef 0 // camera setup cubemap
			nbl::core::vectorSIMDf camViews[6] =
			{
				nbl::core::vectorSIMDf(-1, 0, 0, 0), // -X
				nbl::core::vectorSIMDf(+1, 0, 0, 0), // +X
				nbl::core::vectorSIMDf(0, -1, 0, 0), // -Y
				nbl::core::vectorSIMDf(0, +1, 0, 0), // +Y
				nbl::core::vectorSIMDf(0, 0, -1, 0), // -Z
				nbl::core::vectorSIMDf(0, 0, +1, 0), // +Z
			};

			const nbl::core::vectorSIMDf upVectors[6] =
			{
				nbl::core::vectorSIMDf(0, +1, 0, 0), // +Y
				nbl::core::vectorSIMDf(0, +1, 0, 0), // +Y
				nbl::core::vectorSIMDf(0, 0, -1, 0), // -Z
				nbl::core::vectorSIMDf(0, 0, +1, 0), // +Z
				nbl::core::vectorSIMDf(0, +1, 0, 0), // +Y
				nbl::core::vectorSIMDf(0, +1, 0, 0), // +Y
			};

			CubemapRender cubemapRender = {};
			cubemapRender.sensorIdx = sensors.size();
			cubemapRenders.push_back(cubemapRender);

			for(uint32_t i = 0; i < 6; ++i)
			{
				// FIXME: suffix added after extension
				cubemapFaceSensorData.outputFilePath.replace_extension();
				constexpr const char* suffixes[6] =
				{
					"_x-.exr",
					"_x+.exr",
					"_y-.exr",
					"_y+.exr",
					"_z-.exr",
					"_z+.exr",
				};
				cubemapFaceSensorData.outputFilePath += suffixes[i];

				staticCamera->setTarget((mainCamPos + camViews[i]).getAsVector3df());
				staticCamera->setUpVector(upVectors[i]);

				const float w = float(cubemapFaceSensorData.width)/float(cubemapFaceSensorData.cropWidth);
				const float h = float(cubemapFaceSensorData.height)/float(cubemapFaceSensorData.cropHeight);
				
				const auto fov = 45 degree nondiag;
				const auto aspectRatio = 1.f;
				if (mainSensorData.rightHandedCamera)
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(fov, aspectRatio, nearClip, farClip));
				else
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(fov, aspectRatio, nearClip, farClip));
			}
#endif
		}
		else
		{
// camera setup non spherical	
		}

		return true;
	};

	auto driver = device->getVideoDriver();

	core::smart_refctd_ptr<Renderer> renderer = core::make_smart_refctd_ptr<Renderer>(driver,device->getAssetManager(),smgr,applicationState.isDenoiseDeferred);
	renderer->initSceneResources(meshes,"LowDiscrepancySequenceCache.bin");

// free memory
meshes = {};
device->getAssetManager()->clearAllGPUObjects();


// Deduce Move and Zoom Speeds if it is nan
	auto sceneBoundsExtent = renderer->getSceneBound().getExtent();
	auto sceneDiagonal = sceneBoundsExtent.getLength(); 

	for(uint32_t s = 0u; s < sensors.size(); ++s)
	{
		auto& sensorData = sensors[s];
		
		float linearStepZoomSpeed = sensorData.stepZoomSpeed;
		if(core::isnan<float>(sensorData.stepZoomSpeed))
		{
			linearStepZoomSpeed = sceneDiagonal * (DefaultZoomSpeed / DefaultSceneDiagonal);
		}

		// Set Zoom Multiplier
		{
			float logarithmicZoomSpeed = std::pow(sceneDiagonal, linearStepZoomSpeed / sceneDiagonal);
			sensorData.stepZoomSpeed =  logarithmicZoomSpeed;
			sensorData.getInteractiveCameraAnimator()->setStepZoomMultiplier(logarithmicZoomSpeed);
			printf("[INFO] Sensor[%d] Camera Step Zoom Speed deduced from scene bounds = %f [Linear], %f [Logarithmic] \n", s, linearStepZoomSpeed, logarithmicZoomSpeed);
		}

		
	}

	core::SRange<SensorData> nonInteractiveSensors = { nullptr, nullptr };
	if (!applicationState.isInteractiveMode)
	{
		assert(applicationState.startSensorID < globalMeta->m_global.m_sensors.size() && "startSensorID should've been a valid value by now.");

		uint32_t onePastLastSensorID = applicationState.startSensorID;
		if ((applicationState.processSensorsBehaviour == ProcessSensorsBehaviour::PSB_RENDER_ALL_THEN_INTERACTIVE) || (applicationState.processSensorsBehaviour == ProcessSensorsBehaviour::PSB_RENDER_ALL_THEN_TERMINATE))
			onePastLastSensorID = sensors.size();
		else if (applicationState.processSensorsBehaviour == ProcessSensorsBehaviour::PSB_RENDER_SENSOR_THEN_INTERACTIVE)
			onePastLastSensorID = applicationState.startSensorID + 1;
		nonInteractiveSensors = { sensors.data() + applicationState.startSensorID, sensors.data() + onePastLastSensorID };
	}
	assert(nonInteractiveSensors.size() <= sensors.size());

// ...

	// Render To file
	int32_t prevWidth = 0;
	int32_t prevHeight = 0;
	int32_t prevCascadeCount = 0;
	float prevRegFactor = 0.0f;
	if (!nonInteractiveSensors.empty())
	{
		for (const auto& sensor : nonInteractiveSensors)
		{
			if(!receiver.keepOpen())
				break;

			const uint32_t s = &sensor - sensors.data();

			// Save application's current persistent state to disk.
			{
				applicationState.isBeauty = receiver.isRenderingBeauty();
				applicationState.isInteractiveMode = false;
				applicationState.startSensorID = s;

				if (!applicationState.writeToDisk())
					printf("[ERROR]: Cannot write application state to disk\n");
			}
		
			printf("[INFO] Rendering %s - Sensor(%d) to file.\n", applicationState.xmlPath.c_str(), s);

			bool needsReinit = prevWidth!=sensor.width || prevHeight!=sensor.height || prevCascadeCount!=sensor.cascadeCount || prevRegFactor!=sensor.envmapRegFactor;
			prevWidth = sensor.width;
			prevHeight = sensor.height;
			prevCascadeCount = sensor.cascadeCount;
			prevRegFactor = sensor.envmapRegFactor;
		
			renderer->resetSampleAndFrameCounters(); // so that renderer->getTotalSamplesPerPixelComputed is 0 at the very beginning
			if(needsReinit) 
			{
				renderer->deinitScreenSizedResources();
				renderer->initScreenSizedResources(sensor.width,sensor.height,sensor.envmapRegFactor,sensor.cascadeCount,sensor.cascadeLuminanceBase,sensor.cascadeLuminanceStart,sensor.Emin,sensor.clipPlanes);
			}
		
			smgr->setActiveCamera(sensor.staticCamera);

			const uint32_t samplesPerPixelPerDispatch = renderer->getSamplesPerPixelPerDispatch();
			const uint32_t maxNeededIterations = (sensor.samplesNeeded + samplesPerPixelPerDispatch - 1) / samplesPerPixelPerDispatch;
		
			uint32_t itr = 0u;
			bool takenEnoughSamples = false;
			bool renderFailed = false;
			auto lastTimeLoggedProgress = std::chrono::steady_clock::now();
			while(!takenEnoughSamples && (device->run() && !receiver.isSkipKeyPressed() && receiver.keepOpen()))
			{
				if(itr >= maxNeededIterations)
					std::cout << "[ERROR] Samples taken (" << renderer->getTotalSamplesPerPixelComputed() << ") must've exceeded samples needed for Sensor (" << sensor.samplesNeeded << ") by now; something is wrong." << std::endl;

				// Handle Inputs
				{
					if(receiver.isLogProgressKeyPressed() || (std::chrono::steady_clock::now()-lastTimeLoggedProgress)>std::chrono::seconds(3))
					{
						int progress = float(renderer->getTotalSamplesPerPixelComputed())/float(sensor.samplesNeeded) * 100;
						printf("[INFO] Rendering in progress - %d%% Progress = %u/%u SamplesPerPixel. \n", progress, renderer->getTotalSamplesPerPixelComputed(), sensor.samplesNeeded);
						lastTimeLoggedProgress = std::chrono::steady_clock::now();
					}
					if (receiver.isReloadKeyPressed())
					{
						printf("[INFO]: Reloading..\n");
						reloadApplication();
					}
					receiver.resetKeys();
				}

				driver->beginScene(false, false);

				if(!renderer->render(device->getTimer(),sensor.kappa,sensor.Emin,!sensor.envmap))
				{
					renderFailed = true;
					driver->endScene();
					break;
				}

				auto oldVP = driver->getViewPort();
				driver->blitRenderTargets(renderer->getColorBuffer(),nullptr,false,false,{},{},true);
				driver->setViewPort(oldVP);

				driver->endScene();
			
				if(renderer->getTotalSamplesPerPixelComputed() >= sensor.samplesNeeded)
					takenEnoughSamples = true;
			
				itr++;
			}

			auto screenshotFilePath = sensor.outputFilePath;
		
			if(renderFailed)
			{
				std::cout << "[ERROR] Render Failed." << std::endl;
			}
			else
			{

				const bool shouldDenoise = sensor.type != ext::MitsubaLoader::CElementSensor::Type::SPHERICAL;
				renderer->takeAndSaveScreenShot(screenshotFilePath, shouldDenoise, sensor.denoiserInfo);
				int progress = float(renderer->getTotalSamplesPerPixelComputed())/float(sensor.samplesNeeded) * 100;
				printf("[INFO] Rendered Successfully - %d%% Progress = %u/%u SamplesPerPixel - FileName = %s. \n", progress, renderer->getTotalSamplesPerPixelComputed(), sensor.samplesNeeded, screenshotFilePath.filename().string().c_str());
				auto filename_wo_ext = screenshotFilePath;
				filename_wo_ext.replace_extension();
				auto stream = std::make_shared<simplejson::Stream>();
				stream->begin_json_object();
				stream->emit_json_key_value("output_tonemap", filename_wo_ext.string() +".exr");
				stream->emit_json_key_value("output_albedo", filename_wo_ext.string() + "_albedo.exr");
				stream->emit_json_key_value("output_normal", filename_wo_ext.string() + "_normal.exr");
				if(shouldDenoise)
					stream->emit_json_key_value("output_denoised", filename_wo_ext.string() + "_denoised.exr");
				stream->end_json_object();
				std::cout << "\n[JSON] " << stream->str() << "\n[ENDJSON]" << std::endl;
				
			}

			receiver.resetKeys();
		}

		// Denoise Cubemaps that weren't denoised seperately
		for(uint32_t i = 0; i < cubemapRenders.size(); ++i)
		{
			uint32_t beginIdx = cubemapRenders[i].getSensorsBeginIdx();
			assert(beginIdx + 6 <= sensors.size());

			std::filesystem::path filePaths[6] = {};

			for(uint32_t f = beginIdx; f < beginIdx + 6; ++f)
			{
				const auto & sensor = sensors[f];
				filePaths[f] = sensor.outputFilePath;
			}

			std::string mergedFileName = "Merge_CubeMap_" + mainFileName;
			renderer->denoiseCubemapFaces(
				filePaths,
				mergedFileName,
				sensors[beginIdx].cropOffsetX, sensors[beginIdx].cropOffsetY, sensors[beginIdx].cropWidth, sensors[beginIdx].cropHeight,
				sensors[beginIdx].denoiserInfo);
		}
	}

	// Interactive
	if((applicationState.processSensorsBehaviour != ProcessSensorsBehaviour::PSB_RENDER_ALL_THEN_TERMINATE) && receiver.keepOpen())
	{
		int activeSensor = -1;

		auto setActiveSensor = [&](int index) 
		{
			if(index >= 0 && index < sensors.size())
			{
				bool needsReinit;
				if (activeSensor != -1)
				{
					const auto& activeSensorData = sensors[activeSensor];
					const auto& nextSensorData = sensors[index];
					needsReinit = activeSensorData.width!=nextSensorData.width ||
						activeSensorData.height!=nextSensorData.height ||
						activeSensorData.cascadeCount!=nextSensorData.cascadeCount ||
						activeSensorData.envmapRegFactor!=nextSensorData.envmapRegFactor;
				}
				else
					needsReinit = true;
				activeSensor = index;

				renderer->resetSampleAndFrameCounters();
				if(needsReinit)
				{
					renderer->deinitScreenSizedResources();
					const auto& sensorData = sensors[activeSensor];
					renderer->initScreenSizedResources(sensorData.width,sensorData.height,sensorData.envmapRegFactor,sensorData.cascadeCount,sensorData.cascadeLuminanceBase,sensorData.cascadeLuminanceStart,sensorData.Emin,sensorData.clipPlanes);
				}

				smgr->setActiveCamera(sensors[activeSensor].interactiveCamera);
				std::cout << "Active Sensor = " << activeSensor << std::endl;
			}
		};

		setActiveSensor(applicationState.startSensorID);

		bool writeLastRunState = false;
		uint64_t lastFPSTime = 0;
		auto start = std::chrono::steady_clock::now();
		bool renderFailed = false;
		while (device->run() && receiver.keepOpen())
		{
			// Handle Inputs
			{
				if(receiver.isResetViewPressed())
				{
					sensors[activeSensor].resetInteractiveCamera();
					std::cout << "Interactive Camera Position and Target has been Reset." << std::endl;
				}
				else if(receiver.isOverloadCameraKeyPressed())
				{
					pfd::open_file file("Choose XML file to overload camera with (only first sensor overrides)", "../../media/mitsuba", { "XML files (.xml)", "*.xml" });
					if (!file.result().empty())
					{
						const auto filePath = file.result()[0];
						using namespace nbl::asset;
						smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata> mitsubaMetadata;
						{
							static const IAssetLoader::SAssetLoadParams mitsubaLoaderParams = { 0, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES, nullptr, IAssetLoader::ELPF_LOAD_METADATA_ONLY };
							auto meshes_bundle = device->getAssetManager()->getAsset(filePath.data(),mitsubaLoaderParams);
							if (!meshes_bundle.getContents().empty())
								mitsubaMetadata = smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata>(static_cast<const ext::MitsubaLoader::CMitsubaMetadata*>(meshes_bundle.getMetadata()));
						}
						if (!mitsubaMetadata || mitsubaMetadata->m_global.m_sensors.empty())
							os::Printer::log("ERROR (" + std::to_string(__LINE__) + " line): The xml file is invalid/cannot be loaded! File path: " + filePath, ELL_ERROR);
						else
						{
							const uint32_t originalSensorCount = sensors.size();
							uint32_t idx = originalSensorCount;
							for (const auto& sensor : mitsubaMetadata->m_global.m_sensors)
								extractAndAddToSensorData(sensor,idx++);
							setActiveSensor(originalSensorCount);
						}
						writeLastRunState = true;
					}
				}
				else if(receiver.isNextPressed())
				{
					setActiveSensor(activeSensor + 1);
					writeLastRunState = true;
				}
				else if(receiver.isPreviousPressed())
				{
					setActiveSensor(activeSensor - 1);
					writeLastRunState = true;
				}
				if(receiver.isScreenshotKeyPressed())
				{
					const std::string screenShotFilesPrefix = "ScreenShot";
					const char seperator = '_';
					int maxFileNumber = -1;
					for (const auto & entry : std::filesystem::directory_iterator(std::filesystem::current_path()))
					{
						const auto entryPathStr = entry.path().filename().string();
						const auto firstSeperatorLoc = entryPathStr.find_first_of(seperator) ;
						const auto lastSeperatorLoc = entryPathStr.find_last_of(seperator);
						const auto firstDotLoc = entryPathStr.find_first_of('.');

						const auto firstSection = entryPathStr.substr(0u, firstSeperatorLoc);
						const bool isScreenShot = (firstSection == screenShotFilesPrefix);
						if(isScreenShot)
						{
							const auto middleSection = entryPathStr.substr(firstSeperatorLoc + 1, lastSeperatorLoc - (firstSeperatorLoc + 1));
							const auto numberString = entryPathStr.substr(lastSeperatorLoc + 1, firstDotLoc - (lastSeperatorLoc + 1));

							if(middleSection == mainFileName) 
							{
								const auto number = std::stoi(numberString);
								if(number > maxFileNumber)
								{
									maxFileNumber = number;
								}
							}
						}
					}
					std::string fileNameWoExt = screenShotFilesPrefix + seperator + mainFileName + seperator + std::to_string(maxFileNumber + 1);
					renderer->takeAndSaveScreenShot(std::filesystem::path(fileNameWoExt), true, sensors[activeSensor].denoiserInfo);
				}
				if(receiver.isLogProgressKeyPressed())
				{
					printf("[INFO] Rendering in progress - %d Total SamplesPerPixel Computed.\n", renderer->getTotalSamplesPerPixelComputed());
				}
			}

			driver->beginScene(false, false);
			if(!renderer->render(
					device->getTimer(),
					activeSensor!=-1 ? sensors[activeSensor].kappa:0.f,
					activeSensor!=-1 ? sensors[activeSensor].Emin:0.f,
					true,receiver.isRenderingBeauty()
			))
			{
				renderFailed = true;
				driver->endScene();
				break;
			}

			if (writeLastRunState)
			{
				applicationState.isBeauty = receiver.isRenderingBeauty();
				applicationState.isInteractiveMode = true;
				applicationState.startSensorID = activeSensor;
				applicationState.isInteractiveViewMatrixLH = !sensors[activeSensor].rightHandedCamera;
				applicationState.interactiveCameraViewMatrix = sensors[activeSensor].interactiveCamera->getViewMatrix();

				applicationState.writeToDisk();

				writeLastRunState = false;
			}

			// Post-frame input handling
			{
				if (receiver.isReloadKeyPressed())
				{
					applicationState.isBeauty = receiver.isRenderingBeauty();
					applicationState.isInteractiveMode = true;
					applicationState.startSensorID = activeSensor;
					applicationState.isInteractiveViewMatrixLH = !sensors[activeSensor].rightHandedCamera;
					applicationState.interactiveCameraViewMatrix = sensors[activeSensor].interactiveCamera->getViewMatrix();

					applicationState.writeToDisk();

					reloadApplication();
				}
				receiver.resetKeys();
			}

			auto oldVP = driver->getViewPort();
			driver->blitRenderTargets(renderer->getColorBuffer(),nullptr,false,false,{},{},true);
			driver->setViewPort(oldVP);

			driver->endScene();

			// display frames per second in window title
			uint64_t time = device->getTimer()->getRealTime();
			if (time - lastFPSTime > 1000)
			{
				std::wostringstream str;
				auto samples = renderer->getTotalSamplesComputed();
				auto rays = renderer->getTotalRaysCast();
				const double microsecondsElapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
				str << L"Nabla Path Tracer: " << applicationState.zipPath.c_str() << "\\" << applicationState.xmlPath.c_str()
					<< "   MegaSamples: " << samples/1000000ull
					<< "   MSample/s: " << double(samples)/microsecondsElapsed
					<< "   MRay/s: " << double(rays)/microsecondsElapsed;

				device->setWindowCaption(str.str());
				lastFPSTime = time;
			}
		}
		
		if(renderFailed)
		{
			std::cout << "[ERROR] Render Failed." << std::endl;
		}
		else
		{
			auto extensionStr = getFileExtensionFromFormat(sensors[activeSensor].fileFormat);
			renderer->takeAndSaveScreenShot(std::filesystem::path("LastView_" + mainFileName + "_Sensor_" + std::to_string(activeSensor) + extensionStr), true, sensors[activeSensor].denoiserInfo);
		}

		renderer->deinitScreenSizedResources();
	}

	renderer->deinitSceneResources();
	renderer = nullptr;

	// will leak thread because there's no cross platform input!
	std::exit(0);
	return 0;
}

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }