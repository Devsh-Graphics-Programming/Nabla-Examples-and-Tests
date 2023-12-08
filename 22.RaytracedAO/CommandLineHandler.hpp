// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _RAYTRACER_COMMAND_LINE_HANDLER_
#define _RAYTRACER_COMMAND_LINE_HANDLER_

#include <iostream>
#include <cstdio>
#include <chrono>
#include "nabla.h"
#include "nbl/core/core.h"
#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"

constexpr std::string_view helpMessage = R"(

Parameters:
-SCENE=sceneMitsubaXMLPathOrZipAndXML

Description and usage: 

-SCENE:
	some/path extra/path which will make it skip the file choose dialog

-PROCESS_SENSORS ID:
	It will control the behaviour of sensors in the app as detailed above.
	If the option is not passed, then it defaults to RenderAllThenInteractive.
	If the ID is not passed, then it defaults to 0.
	
Example Usages :
	raytracedao.exe -SCENE=../../media/kitchen.zip scene.xml
	raytracedao.exe -SCENE=../../media/kitchen.zip scene.xml -PROCESS_SENSORS RenderAllThenInteractive
	raytracedao.exe -SCENE="../../media/my good kitchen.zip" scene.xml -PROCESS_SENSORS RenderAllThenTerminate 0
	raytracedao.exe -SCENE="../../media/my good kitchen.zip scene.xml" -PROCESS_SENSORS RenderSensorThenInteractive 1
	raytracedao.exe -SCENE="../../media/extraced folder/scene.xml" -PROCESS_SENSORS InteractiveAtSensor 2
)";
 

constexpr std::string_view SCENE_VAR_NAME						= "SCENE";
constexpr std::string_view SCREENSHOT_OUTPUT_FOLDER_VAR_NAME	= "SCREENSHOT_OUTPUT_FOLDER";
constexpr std::string_view PROCESS_SENSORS_VAR_NAME				= "PROCESS_SENSORS";
constexpr std::string_view DEFER_DENOISE_VAR_NAME               = "DEFER_DENOISE";

constexpr uint32_t MaxRayTracerCommandLineArgs = 8;

enum RaytracerExampleArguments
{
	REA_SCENE,
	REA_PROCESS_SENSORS,
	REA_DEFER_DENOISE,
	REA_COUNT,
};

enum class ProcessSensorsBehaviour
{
	PSB_RENDER_ALL_THEN_INTERACTIVE,
	PSB_RENDER_ALL_THEN_TERMINATE,
	PSB_RENDER_SENSOR_THEN_INTERACTIVE,
	PSB_INTERACTIVE_AT_SENSOR,
	PSB_COUNT
};

using variablesType = std::unordered_map<RaytracerExampleArguments, std::optional<std::vector<std::string>>>;

class CommandLineHandler
{
	public:
		CommandLineHandler(const std::vector<std::string>& argv);

		inline auto& getSceneDirectory() const
		{
			return sceneDirectory;
		}

		inline auto& getProcessSensorsBehaviour() const
		{
			return processSensorsBehaviour;
		}

		inline auto& getSensorID() const
		{
			return sensorID;
		}

		inline bool getDeferredDenoiseFlag() const
		{
			return isDenoiseDeferred;
		}

	private:
		void initializeMatchingMap()
		{
			rawVariables[REA_SCENE];
			rawVariables[REA_PROCESS_SENSORS];
			rawVariables[REA_DEFER_DENOISE];
		}

		RaytracerExampleArguments getMatchedVariableMapID(const std::string& variableName)
		{
			if (variableName == SCENE_VAR_NAME)
				return REA_SCENE;
			else if (variableName == PROCESS_SENSORS_VAR_NAME)
				return REA_PROCESS_SENSORS;
			else if (variableName == DEFER_DENOISE_VAR_NAME)
				return REA_DEFER_DENOISE;
			else
				return REA_COUNT;
		}

		bool validateParameters();

		void performFinalAssignmentStepForUsefulVariables()
		{
			if(rawVariables[REA_SCENE].has_value())
				sceneDirectory = rawVariables[REA_SCENE].value();
			if (rawVariables[REA_PROCESS_SENSORS].has_value())
			{
				const auto& values = rawVariables[REA_PROCESS_SENSORS].value();
				for (uint32_t i = 0; i < values.size(); ++i)
				{
					if (i == 0)
					{
						const char* behaviour = values[0].c_str();
						if (strcmp(behaviour, "RenderAllThenInteractive") == 0)
						{
							processSensorsBehaviour = ProcessSensorsBehaviour::PSB_RENDER_ALL_THEN_INTERACTIVE;
						}
						else if (strcmp(behaviour, "RenderAllThenTerminate") == 0)
						{
							processSensorsBehaviour = ProcessSensorsBehaviour::PSB_RENDER_ALL_THEN_TERMINATE;
						}
						else if (strcmp(behaviour, "RenderSensorThenInteractive") == 0)
						{
							processSensorsBehaviour = ProcessSensorsBehaviour::PSB_RENDER_SENSOR_THEN_INTERACTIVE;
						}
						else if (strcmp(behaviour, "InteractiveAtSensor") == 0)
						{
							processSensorsBehaviour = ProcessSensorsBehaviour::PSB_INTERACTIVE_AT_SENSOR;
						}
						else
						{
							printf("[ERROR]: Invalid option for '%s'. Using RenderAllThenInteractive.\n", PROCESS_SENSORS_VAR_NAME.data());
						}
					}
					else if (i == 1)
					{
                        sensorID = std::stoi(values[1]);
					}
				}
			}
			
			isDenoiseDeferred = rawVariables[REA_DEFER_DENOISE].has_value();
		}

		variablesType rawVariables;

		// Loaded from CMD
		std::vector<std::string> sceneDirectory; // [0] zip [1] optional xml in zip
		std::string outputScreenshotsFolderPath;
		bool isDenoiseDeferred;
		struct
		{
			ProcessSensorsBehaviour processSensorsBehaviour = ProcessSensorsBehaviour::PSB_RENDER_ALL_THEN_INTERACTIVE;
			uint32_t sensorID = 0;
		};

};

#endif // _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_