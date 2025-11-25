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
#include "SimpleJson.h"

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

struct PersistentState
{
	bool isBeauty;
	bool isInteractiveMode;
	bool isInteractiveViewMatrixLH;
	bool isDenoiseDeferred;
	uint32_t startSensorID;
	std::string zipPath;
	std::string xmlPath;
	ProcessSensorsBehaviour processSensorsBehaviour;
	// It is important to initialize it to all 0s because we use the condition of determinant 0 as an invalid condition for the view matrix.
	core::matrix3x4SIMD interactiveCameraViewMatrix = core::matrix3x4SIMD(core::vectorSIMDf(), core::vectorSIMDf(), core::vectorSIMDf());

	bool readFromDisk()
	{
		bool readSuccess = false;
		std::ifstream readFile("lastRun.cache", std::ios::in | std::ios::binary | std::ios::ate);
		if (readFile.is_open())
		{
			auto readSize = readFile.tellg();
			if (readSize != std::istream::pos_type(-1))
			{
				std::unique_ptr<uint8_t[]> readBuffer = std::make_unique<uint8_t[]>(readSize);
				readFile.seekg(0, std::ios::beg);
				readFile.read(reinterpret_cast<char*>(readBuffer.get()), readSize);
				if (readFile.rdstate() == std::ios_base::goodbit)
				{
					uint64_t offset = 0;

					memcpy(&isBeauty, readBuffer.get() + offset, sizeof(bool));
					offset += sizeof(bool);

					memcpy(&isInteractiveMode, readBuffer.get() + offset, sizeof(bool));
					offset += sizeof(bool);

					memcpy(&isInteractiveViewMatrixLH, readBuffer.get() + offset, sizeof(bool));
					offset += sizeof(bool);

					memcpy(&startSensorID, readBuffer.get() + offset, sizeof(uint32_t));
					offset += sizeof(uint32_t);

					memcpy(&processSensorsBehaviour, readBuffer.get() + offset, sizeof(ProcessSensorsBehaviour));
					offset += sizeof(ProcessSensorsBehaviour);

					memcpy(&interactiveCameraViewMatrix, readBuffer.get() + offset, sizeof(core::matrix3x4SIMD));
					offset += sizeof(core::matrix3x4SIMD);

					const char* path = reinterpret_cast<const char*>(readBuffer.get() + offset);
					zipPath = std::string(path);
					offset += zipPath.length() + 1;

					path = reinterpret_cast<const char*>(readBuffer.get() + offset);
					xmlPath = std::string(path);
					offset += xmlPath.length() + 1;

					readSuccess = (offset == static_cast<uint64_t>(readSize));
				}
			}

			readFile.close();
		}

		return readSuccess;
	}

	bool writeToDisk() const
	{
		bool writeSuccess = false;
		std::ofstream outFile("lastRun.cache", std::ios::out | std::ios::binary);
		if (outFile.is_open())
		{
			const size_t writeSize = getSerializedMemorySize();

			std::unique_ptr<uint8_t[]> writeBuffer = std::make_unique<uint8_t[]>(writeSize);

			uint64_t offset = 0;

			memcpy(writeBuffer.get() + offset, &isBeauty, sizeof(bool));
			offset += sizeof(bool);

			memcpy(writeBuffer.get() + offset, &isInteractiveMode, sizeof(bool));
			offset += sizeof(bool);

			memcpy(writeBuffer.get() + offset, &isInteractiveViewMatrixLH, sizeof(bool));
			offset += sizeof(bool);

			memcpy(writeBuffer.get() + offset, &startSensorID, sizeof(uint32_t));
			offset += sizeof(uint32_t);

			memcpy(writeBuffer.get() + offset, &processSensorsBehaviour, sizeof(ProcessSensorsBehaviour));
			offset += sizeof(ProcessSensorsBehaviour);

			memcpy(writeBuffer.get() + offset, &interactiveCameraViewMatrix, sizeof(core::matrix3x4SIMD));
			offset += sizeof(core::matrix3x4SIMD);

			memcpy(writeBuffer.get() + offset, zipPath.c_str(), zipPath.length() + 1);
			offset += zipPath.length() + 1;

			memcpy(writeBuffer.get() + offset, xmlPath.c_str(), xmlPath.length() + 1);
			offset += xmlPath.length() + 1;

			assert(offset == static_cast<uint32_t>(writeSize));

			outFile.write(reinterpret_cast<char*>(writeBuffer.get()), writeSize);
			if (outFile.rdstate() == std::ios_base::goodbit)
				writeSuccess = true;

			outFile.close();
		}

		if (!writeSuccess)
			printf("[ERROR]: Failed to write the persistent state cache.\n");

		return writeSuccess;
	}

private:
	inline size_t getSerializedMemorySize() const
	{
		const size_t result =
			sizeof(bool)					+ // isBeauty
			sizeof(bool)					+ // isInteractiveMode
			sizeof(bool)                    + // isInteractiveViewMatrixLH
			sizeof(uint32_t)				+ // startSensorID
			sizeof(ProcessSensorsBehaviour) + // processSensorsBehaviour
			sizeof(core::matrix3x4SIMD)		+ // interactiveCameraViewMatrix
			(zipPath.length() + 1)			+
			(xmlPath.length() + 1)			;

		return result;
	}
};

int main(int argc, char** argv)
{
	std::vector<std::string> arguments;
	if (argc>1)
	{
		for (auto i = 1ul; i < argc; ++i)
			arguments.emplace_back(argv[i]);
	}
	std::cout << std::endl;
	std::cout << "-- Build URL:" << std::endl;
	std::cout << NBL_BUILD_URL << std::endl;
	std::cout << std::endl;
	std::cout << "-- Build log:" << std::endl;
	std::cout << NBL_GIT_LOG << std::endl;
	std::cout  << std::endl;

	bool applicationIsReloaded = false;
	PersistentState applicationState;
	{
		CommandLineHandler cmdHandler = CommandLineHandler(arguments);

		applicationState.processSensorsBehaviour = cmdHandler.getProcessSensorsBehaviour();
		applicationState.startSensorID = cmdHandler.getSensorID();
		applicationState.isInteractiveMode = (applicationState.processSensorsBehaviour == ProcessSensorsBehaviour::PSB_INTERACTIVE_AT_SENSOR);
		applicationState.isDenoiseDeferred = cmdHandler.getDeferredDenoiseFlag();

		auto sceneDir = cmdHandler.getSceneDirectory();
		if ((sceneDir.size() == 1) && (sceneDir[0] == "")) // special condition for reloading the application
			applicationIsReloaded = true;

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
	std::string mainFileName; // std::filesystem::path(filePath).filename().string();

	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	params.WindowSize = dimension2d<uint32_t>(1920, 1080);
	auto device = createDeviceEx(params);
	if (!device)
		return 1; // could not create selected driver.

	//
	asset::SAssetBundle meshes = {};
	core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata> globalMeta;
	{
		io::IFileSystem* fs = device->getFileSystem();
		asset::IAssetManager* am = device->getAssetManager();

		auto serializedLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CSerializedLoader>(am);
		auto mitsubaLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CMitsubaLoader>(am, fs);
		serializedLoader->initialize();
		mitsubaLoader->initialize();
		am->addAssetLoader(std::move(serializedLoader));
		am->addAssetLoader(std::move(mitsubaLoader));

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
			if (_zipPath.empty() && _xmlPath.empty())
				return result;

			_mainFileName = "";
			if (!_zipPath.empty())
			{
				_mainFileName = std::filesystem::path(_zipPath).filename().string();
				_mainFileName = _mainFileName.substr(0u, _mainFileName.find_first_of('.'));

				io::IFileArchive* arch = nullptr;
				device->getFileSystem()->addFileArchive(_zipPath.c_str(), io::EFAT_ZIP, "", &arch);
				if (!arch)
					return result;

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
			}
			else if (!_xmlPath.empty())
			{
				_mainFileName = std::filesystem::path(_xmlPath).filename().string();
				_mainFileName = _mainFileName.substr(0u, _mainFileName.find_first_of('.'));
			}

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
		if (meshes.getContents().empty() || applicationIsReloaded)
		{
			if (meshes.getContents().empty() && !applicationState.xmlPath.empty())
				printf("[ERROR]: Failed to load asset at: %s\n", applicationState.xmlPath.c_str());

			// Restore state to get new values for zipPath and xmlPath and try loading again
			printf("[INFO]: Trying to restore the application to its previous state.\n");

			bool restoreSuccess = false;
			if (applicationState.readFromDisk())
			{
				meshes = loadScene(applicationState.zipPath, applicationState.xmlPath, mainFileName);
				if (!meshes.getContents().empty())
					restoreSuccess = true;
			}

			if (!restoreSuccess)
			{
				pfd::message("ERROR", "Cannot restore application to its previous state.", pfd::choice::ok);
				return 2;
			}
		}

		globalMeta = core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata>(meshes.getMetadata()->selfCast<const ext::MitsubaLoader::CMitsubaMetadata>());
		if (!globalMeta)
		{
			std::cout << "[ERROR] Couldn't get global Meta";
			return 3;
		}

		std::cout << "Total number of Sensors = " << globalMeta->m_global.m_sensors.size() << std::endl;

		if (globalMeta->m_global.m_sensors.empty())
		{
			std::cout << "[ERROR] No Sensors found." << std::endl;
			assert(false);
			return 5; // return code?
		}

		if (applicationState.startSensorID >= globalMeta->m_global.m_sensors.size())
		{
			applicationState.startSensorID = 0;
			printf("[WARNING]: A valid sensor ID was not found. Selecting the sensor: %u\n", applicationState.startSensorID);
		}

		// empty out the cache from individual images and meshes taht are not used by the scene
		am->clearAllAssetCache();
	}
	
	constexpr float DefaultRotateSpeed = 300.0f;
	constexpr float DefaultZoomSpeed = 1.0f;
	constexpr float DefaultMoveSpeed = 100.0f;
	constexpr float DefaultSceneDiagonal = 50.0f; // reference for default zoom and move speed;

	struct SensorData
	{
		int32_t width = 0u;
		int32_t height = 0u;
		int32_t cropWidth = 0u;
		int32_t cropHeight = 0u;
		int32_t cropOffsetX = 0u;
		int32_t cropOffsetY = 0u;
		bool rightHandedCamera = true;
		uint32_t samplesNeeded = 0u;
		float moveSpeed = core::nan<float>();
		float stepZoomSpeed = core::nan<float>();
		float rotateSpeed = core::nan<float>();
		scene::ICameraSceneNode * staticCamera;
		scene::ICameraSceneNode * interactiveCamera;
		std::filesystem::path outputFilePath;
		ext::MitsubaLoader::CElementSensor::Type type;
		ext::MitsubaLoader::CElementFilm::FileFormat fileFormat;
		Renderer::DenoiserArgs denoiserInfo = {};
		int32_t cascadeCount = 1;
		float cascadeLuminanceBase = core::nan<float>();
		float cascadeLuminanceStart = core::nan<float>();
		float kappa = 0.f;
		float Emin = 0.05f;
		bool envmap = false;
		float envmapRegFactor = 0.0f;
		core::vector<core::vectorSIMDf> clipPlanes;

		scene::CSceneNodeAnimatorCameraModifiedMaya* getInteractiveCameraAnimator()
		{
			return reinterpret_cast<scene::CSceneNodeAnimatorCameraModifiedMaya*>(interactiveCamera->getAnimators()[0]);
		}

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
	
	struct CubemapRender
	{
		uint32_t sensorIdx = 0u;
		uint32_t getSensorsBeginIdx() const { return sensorIdx; }
		uint32_t getSensorsEndIdx() const { return sensorIdx + 5; }
	};

	auto smgr = device->getSceneManager();
	
	// When outputFilePath isn't set in Film Element in Mitsuba, use this to find the extension string.
	auto getFileExtensionFromFormat= [](ext::MitsubaLoader::CElementFilm::FileFormat format) -> std::string
	{
		std::string ret = "";
		using FileFormat = ext::MitsubaLoader::CElementFilm::FileFormat;
		switch (format)
		{
			case FileFormat::PNG:
				ret = ".png";
				break;
			case FileFormat::OPENEXR:
				ret = ".exr";
				break;
			case FileFormat::JPEG:
				ret = ".jpg";
				break;
			default: // TODO?
				break;
		}
		return ret;
	};

	auto isFileExtensionCompatibleWithFormat = [](std::string extension, ext::MitsubaLoader::CElementFilm::FileFormat format) -> bool
	{
		if(extension.empty())
			return false;

		if(extension[0] == '.')
			extension = extension.substr(1, extension.size());

		// TODO: get the supported extensions from loaders(?)
		using FileFormat = ext::MitsubaLoader::CElementFilm::FileFormat;
		switch (format)
		{
			case FileFormat::PNG:
				return extension == "png";
			case FileFormat::OPENEXR:
				return extension == "exr";
			case FileFormat::JPEG:
				return extension == "jpg" || extension == "jpeg" || extension == "jpe" || extension == "jif" || extension == "jfif" || extension == "jfi";
			default:
				return false;
		}
	};
	
	const bool shouldHaveSensorIdxInFileName = globalMeta->m_global.m_sensors.size() > 1;
	std::vector<SensorData> sensors;
	std::vector<CubemapRender> cubemapRenders;

	auto extractAndAddToSensorData = [&](const ext::MitsubaLoader::CElementSensor& sensor, uint32_t idx) -> bool
	{
		SensorData mainSensorData = {};

		const auto& film = sensor.film;
		mainSensorData.denoiserInfo.bloomFilePath = std::filesystem::path(film.denoiserBloomFilePath);
		mainSensorData.denoiserInfo.bloomScale = film.denoiserBloomScale;
		mainSensorData.denoiserInfo.bloomIntensity = film.denoiserBloomIntensity;
		mainSensorData.denoiserInfo.tonemapperArgs = std::string(film.denoiserTonemapperArgs);
		mainSensorData.fileFormat = film.fileFormat;
		mainSensorData.cascadeCount = film.cascadeCount;
		mainSensorData.cascadeLuminanceBase = film.cascadeLuminanceBase;
		mainSensorData.cascadeLuminanceStart = film.cascadeLuminanceStart;
		mainSensorData.kappa = mainSensorData.cascadeCount<2 ? 0.f:film.rfilter.kappa;
		mainSensorData.Emin = film.rfilter.Emin;
		mainSensorData.envmapRegFactor = core::clamp(film.envmapRegularizationFactor, 0.0f, 0.8f);
		mainSensorData.outputFilePath = std::filesystem::path(film.outputFilePath);
		// handle missing output path
		if (mainSensorData.outputFilePath.empty())
		{
			auto extensionStr = getFileExtensionFromFormat(mainSensorData.fileFormat);
			if(shouldHaveSensorIdxInFileName)
				mainSensorData.outputFilePath = std::filesystem::path("Render_" + mainFileName + "_Sensor_" + std::to_string(idx) + extensionStr);
			else
				mainSensorData.outputFilePath = std::filesystem::path("Render_" + mainFileName + extensionStr);
		}
		if(!isFileExtensionCompatibleWithFormat(mainSensorData.outputFilePath.extension().string(), mainSensorData.fileFormat))
			std::cout << "[ERROR] film.outputFilePath's extension is not compatible with film.fileFormat" << std::endl;

		mainSensorData.samplesNeeded = sensor.sampler.sampleCount;
		std::cout << "\t SamplesPerPixelNeeded = " << mainSensorData.samplesNeeded << std::endl;

		const ext::MitsubaLoader::CElementSensor::PerspectivePinhole* persp = nullptr;
		const ext::MitsubaLoader::CElementSensor::Orthographic* ortho = nullptr;
		const ext::MitsubaLoader::CElementSensor::CameraBase* cameraBase = nullptr;
		switch (sensor.type)
		{
			case ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE:
				persp = &sensor.perspective;
				cameraBase = persp;
				std::cout << "\t Type = PERSPECTIVE" << std::endl;
				break;
			case ext::MitsubaLoader::CElementSensor::Type::THINLENS:
				persp = &sensor.thinlens;
				cameraBase = persp;
				std::cout << "\t Type = THINLENS" << std::endl;
				break;
			case ext::MitsubaLoader::CElementSensor::Type::ORTHOGRAPHIC:
				ortho = &sensor.orthographic;
				cameraBase = ortho;
				std::cout << "\t Type = ORTHOGRAPHIC" << std::endl;
				break;
			case ext::MitsubaLoader::CElementSensor::Type::TELECENTRIC:
				ortho = &sensor.telecentric;
				cameraBase = ortho;
				std::cout << "\t Type = TELECENTRIC" << std::endl;
				break;
			case ext::MitsubaLoader::CElementSensor::Type::SPHERICAL:
				cameraBase = &sensor.spherical;
				std::cout << "\t Type = SPHERICAL" << std::endl;
				break;
			default:
				std::cout << "\tSensor Type is not valid" << std::endl;
				return false;
		}
		mainSensorData.type = sensor.type;
		
		for (auto i=0; i<sensor.MaxClipPlanes; i++)
		{
			const auto& plane = cameraBase->clipPlanes[i];
			if ((plane!=core::vectorSIMDf()).any())
			{
				mainSensorData.clipPlanes.push_back(plane);
				printf("Found Clip Plane %f,%f,%f,%f\n",plane[0],plane[1],plane[2],plane[3]);
			}
		}

		mainSensorData.rotateSpeed = cameraBase->rotateSpeed;
		mainSensorData.stepZoomSpeed = cameraBase->zoomSpeed;
		mainSensorData.moveSpeed = cameraBase->moveSpeed;
		
		if(core::isnan<float>(mainSensorData.rotateSpeed))
		{
			mainSensorData.rotateSpeed = DefaultRotateSpeed;
			std::cout << "\t Camera Rotate Speed = " << mainSensorData.rotateSpeed << " = [Default Value]" << std::endl;
		}
		else
			std::cout << "\t Camera Rotate Speed = " << mainSensorData.rotateSpeed << std::endl;

		if(core::isnan<float>(mainSensorData.stepZoomSpeed))
			std::cout << "\t Camera Step Zoom Speed [Linear] = " << "[Value will be deduced from Scene Bounds] " << std::endl;
		else
			std::cout << "\t Camera Step Zoom Speed [Linear] = " << mainSensorData.stepZoomSpeed << std::endl;
		
		if(core::isnan<float>(mainSensorData.moveSpeed))
			std::cout << "\t Camera Move Speed = " << "[Value will be deduced from Scene Bounds] " << std::endl;
		else
			std::cout << "\t Camera Move Speed = " << mainSensorData.moveSpeed << std::endl;
		
		float defaultZoomSpeedMultiplier = std::pow(DefaultSceneDiagonal, DefaultZoomSpeed / DefaultSceneDiagonal);
		mainSensorData.interactiveCamera = smgr->addCameraSceneNodeModifiedMaya(nullptr, -1.0f * mainSensorData.rotateSpeed, 50.0f, mainSensorData.moveSpeed, -1, 2.0f, defaultZoomSpeedMultiplier, false, true);
		
		nbl::core::vectorSIMDf mainCamPos;
		nbl::core::vectorSIMDf mainCamUp;
		nbl::core::vectorSIMDf mainCamView;
		// need to extract individual components from matrix to camera
		{
			auto relativeTransform = sensor.transform.matrix.extractSub3x4();
			if (applicationState.isInteractiveMode && (idx == applicationState.startSensorID) && (core::abs(applicationState.interactiveCameraViewMatrix.getPseudoDeterminant().x) > 1e-6f))
			{
				if (!applicationState.interactiveCameraViewMatrix.getInverse(relativeTransform))
					printf("[ERROR]: Previously saved interactive camera's view matrix is not invertible.\n");

				if (applicationState.isInteractiveViewMatrixLH)
				{
					// invert signs in the first col only
					relativeTransform.rows[0].x *= -1.f;
					relativeTransform.rows[1].x *= -1.f;
					relativeTransform.rows[2].x *= -1.f;
				}
				else
				{
					// invert signs both in the first and third cols
					relativeTransform.rows[0].x *= -1.f;
					relativeTransform.rows[1].x *= -1.f;
					relativeTransform.rows[2].x *= -1.f;

					relativeTransform.rows[0].z *= -1.f;
					relativeTransform.rows[1].z *= -1.f;
					relativeTransform.rows[2].z *= -1.f;
				}
			}
			
			if (relativeTransform.getPseudoDeterminant().x < 0.f)
				mainSensorData.rightHandedCamera = false;
			else
				mainSensorData.rightHandedCamera = true;
			
			std::cout << "\t IsRightHanded=" << ((mainSensorData.rightHandedCamera) ? "TRUE" : "FALSE") << std::endl;

			mainCamPos = relativeTransform.getTranslation();
			
			std::cout << "\t Camera Position = <" << mainCamPos.x << "," << mainCamPos.y << "," << mainCamPos.z << ">" << std::endl;

			auto tpose = core::transpose(core::matrix4SIMD(relativeTransform));
			mainCamUp = tpose.rows[1];
			mainCamView = tpose.rows[2];

			std::cout << "\t Camera Reconstructed UpVector = <" << mainCamUp.x << "," << mainCamUp.y << "," << mainCamUp.z << ">" << std::endl;
			std::cout << "\t Camera Reconstructed Forward = <" << mainCamView.x << "," << mainCamView.y << "," << mainCamView.z << ">" << std::endl;
		}
		
		float realFoVDegrees;
		auto width = film.cropWidth;
		auto height = film.cropHeight;

		float aspectRatio = float(width) / float(height);
		auto convertFromXFoV = [=](float fov) -> float
		{
			float aspectX = tan(core::radians(fov)*0.5f);
			return core::degrees(atan(aspectX/aspectRatio)*2.f);
		};

		float nearClip = cameraBase->nearClip;
		float farClip = cameraBase->farClip;
		if(farClip > nearClip * 10'000.0f)
			std::cout << "[WARN] Depth Range is too big: nearClip = " << nearClip << ", farClip = " << farClip << std::endl;

		if (mainSensorData.type == ext::MitsubaLoader::CElementSensor::Type::SPHERICAL)
		{
			mainSensorData.width = film.width;
			mainSensorData.height = film.height;
			mainSensorData.cropWidth = film.cropWidth;
			mainSensorData.cropHeight = film.cropHeight;
			mainSensorData.cropOffsetX = film.cropOffsetX;
			mainSensorData.cropOffsetY = film.cropOffsetY;

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
				SensorData cubemapFaceSensorData = mainSensorData;
				cubemapFaceSensorData.envmap = true;

				if (mainSensorData.cropWidth != mainSensorData.cropHeight)
				{
					std::cout << "[ERROR] Cannot generate cubemap faces where film.cropWidth and film.cropHeight are not equal. (Aspect Ratio must be 1)" << std::endl;
					assert(false);
				}

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

				cubemapFaceSensorData.staticCamera = smgr->addCameraSceneNode(nullptr); 
				auto& staticCamera = cubemapFaceSensorData.staticCamera;
				
				const auto& camView = camViews[i];
				const auto& upVector = upVectors[i];

				staticCamera->setPosition(mainCamPos.getAsVector3df());
				staticCamera->setTarget((mainCamPos + camView).getAsVector3df());
				staticCamera->setUpVector(upVector);

				const float w = float(cubemapFaceSensorData.width)/float(cubemapFaceSensorData.cropWidth);
				const float h = float(cubemapFaceSensorData.height)/float(cubemapFaceSensorData.cropHeight);
				
				const auto fov = atanf(h)*2.f;
				const auto aspectRatio = h/w;
				if (mainSensorData.rightHandedCamera)
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(fov, aspectRatio, nearClip, farClip));
				else
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(fov, aspectRatio, nearClip, farClip));
				
				cubemapFaceSensorData.interactiveCamera = smgr->addCameraSceneNodeModifiedMaya(nullptr, -1.0f * mainSensorData.rotateSpeed, 50.0f, mainSensorData.moveSpeed, -1, 2.0f, defaultZoomSpeedMultiplier, false, true);
				cubemapFaceSensorData.resetInteractiveCamera();
				sensors.push_back(cubemapFaceSensorData);
			}
		}
		else
		{
			mainSensorData.width = film.cropWidth;
			mainSensorData.height = film.cropHeight;
			
			if(film.cropOffsetX != 0 || film.cropOffsetY != 0)
			{
				std::cout << "[WARN] CropOffsets are non-zero. cropping is not supported for non cubemap renders." << std::endl;
			}

			mainSensorData.staticCamera = smgr->addCameraSceneNode(nullptr); 
			auto& staticCamera = mainSensorData.staticCamera;

			staticCamera->setPosition(mainCamPos.getAsVector3df());
			
			{
				auto target = mainCamView+mainCamPos;
				std::cout << "\t Camera Target = <" << target.x << "," << target.y << "," << target.z << ">" << std::endl;
				staticCamera->setTarget(target.getAsVector3df());
			}

			{
				auto declaredUp = cameraBase->up;
				auto reconstructedRight = core::cross(declaredUp,mainCamView);
				auto actualRight = core::cross(mainCamUp,mainCamView);
				// special formulation avoiding multiple sqrt and inversesqrt to preserve precision
				const float dp = core::dot(reconstructedRight,actualRight).x/core::sqrt((core::dot(reconstructedRight,reconstructedRight)*core::dot(actualRight,actualRight)).x);
				const float pb = core::dot(declaredUp,mainCamView).x/core::sqrt((core::dot(declaredUp,declaredUp)*core::dot(mainCamView,mainCamView)).x);
				std::cout << "\t Camera Reconstructed UpVector match score = "<< dp << std::endl;
				if (dp>0.97f && dp<1.03f && abs(pb)<0.9996f)
					staticCamera->setUpVector(declaredUp);
				else
					staticCamera->setUpVector(mainCamUp);
			}

			//
			if (ortho)
			{
				const auto scale = sensor.transform.matrix.extractSub3x4().getScale();
				const float volumeX = 2.f*scale.x;
				const float volumeY = (2.f/aspectRatio)*scale.y;
				if (mainSensorData.rightHandedCamera)
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixOrthoRH(volumeX, volumeY, nearClip, farClip));
				else
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixOrthoLH(volumeX, volumeY, nearClip, farClip));
			}
			else if (persp)
			{
				switch (persp->fovAxis)
				{
					case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::X:
						realFoVDegrees = convertFromXFoV(persp->fov);
						break;
					case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::Y:
						realFoVDegrees = persp->fov;
						break;
					case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::DIAGONAL:
						{
							float aspectDiag = tan(core::radians(persp->fov)*0.5f);
							float aspectY = aspectDiag/core::sqrt(1.f+aspectRatio*aspectRatio);
							realFoVDegrees = core::degrees(atan(aspectY)*2.f);
						}
						break;
					case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::SMALLER:
						if (width < height)
							realFoVDegrees = convertFromXFoV(persp->fov);
						else
							realFoVDegrees = persp->fov;
						break;
					case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::LARGER:
						if (width < height)
							realFoVDegrees = persp->fov;
						else
							realFoVDegrees = convertFromXFoV(persp->fov);
						break;
					default:
						realFoVDegrees = NAN;
						assert(false);
						break;
				}
				core::matrix4SIMD projMat;
				projMat.setTranslation(core::vectorSIMDf(persp->shiftX,-persp->shiftY,0.f,1.f));
				if (mainSensorData.rightHandedCamera)
					projMat = core::concatenateBFollowedByA(projMat,core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(realFoVDegrees), aspectRatio, nearClip, farClip));
				else
					projMat = core::concatenateBFollowedByA(projMat,core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(realFoVDegrees), aspectRatio, nearClip, farClip));
				staticCamera->setProjectionMatrix(projMat);
			}
			else
			{
				assert(false);
			}

			mainSensorData.resetInteractiveCamera();
			sensors.push_back(mainSensorData);
		}

		return true;
	};

	// Always add all the sensors because the interactive mode wants all the sensors.
	for(uint32_t s = 0u; s < globalMeta->m_global.m_sensors.size(); ++s)
	{
		std::cout << "Sensors[" << s << "] = " << std::endl;
		const auto& sensor = globalMeta->m_global.m_sensors[s];
		extractAndAddToSensorData(sensor, s);
	}

	auto driver = device->getVideoDriver();

	core::smart_refctd_ptr<Renderer> renderer = core::make_smart_refctd_ptr<Renderer>(driver,device->getAssetManager(),smgr,applicationState.isDenoiseDeferred);
	renderer->initSceneResources(meshes,"LowDiscrepancySequenceCache.bin");
	// free memory
	meshes = {};
	device->getAssetManager()->clearAllGPUObjects();
	
	RaytracerExampleEventReceiver receiver;
	device->setEventReceiver(&receiver);

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

		if(core::isnan<float>(sensorData.moveSpeed))
		{
			float newMoveSpeed = DefaultMoveSpeed * (sceneDiagonal / DefaultSceneDiagonal);
			sensorData.moveSpeed = newMoveSpeed;
			sensorData.getInteractiveCameraAnimator()->setMoveSpeed(newMoveSpeed);
			printf("[INFO] Sensor[%d] Camera Move Speed deduced from scene bounds = %f\n", s, newMoveSpeed);
		}
		
		assert(!core::isnan<float>(sensorData.getInteractiveCameraAnimator()->getRotateSpeed()));
		//assert(!core::isnan<float>(sensorData.getInteractiveCameraAnimator()->getStepZoomSpeed()));
		assert(!core::isnan<float>(sensorData.getInteractiveCameraAnimator()->getMoveSpeed()));
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

	auto reloadApplication = [argv]()
	{
		printf("[INFO]: Reloading..\n");

		// Set up the special reload condition.
		const char* cmdLineParams = "-SCENE=";
		HINSTANCE result = ShellExecuteA(NULL, "open", argv[0], cmdLineParams, NULL, SW_SHOWNORMAL);
		if ((uint64_t)result <= 32)
			printf("[ERROR]: Failed to reload.\n");
		else
			exit(0);
	};

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