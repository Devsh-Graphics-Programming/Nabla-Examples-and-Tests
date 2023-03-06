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
	
Example Usages :
	raytracedao.exe -SCENE=../../media/kitchen.zip scene.xml
	raytracedao.exe -SCENE="../../media/my good kitchen.zip" scene.xml
	raytracedao.exe -SCENE="../../media/my good kitchen.zip scene.xml"
	raytracedao.exe -SCENE="../../media/extraced folder/scene.xml"
)";
 

constexpr std::string_view SCENE_VAR_NAME						= "SCENE";
constexpr std::string_view SCREENSHOT_OUTPUT_FOLDER_VAR_NAME	= "SCREENSHOT_OUTPUT_FOLDER";
constexpr std::string_view PROCESS_SENSORS_VAR_NAME				= "PROCESS_SENSORS";

constexpr uint32_t MaxRayTracerCommandLineArgs = 8;

enum RaytracerExampleArguments
{
	REA_SCENE,
	REA_PROCESS_SENSORS,
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

	private:
		void initializeMatchingMap()
		{
			rawVariables[REA_SCENE];
			rawVariables[REA_PROCESS_SENSORS];
		}

		RaytracerExampleArguments getMatchedVariableMapID(const std::string& variableName)
		{
			if (variableName == SCENE_VAR_NAME)
				return REA_SCENE;
			else if (variableName == PROCESS_SENSORS_VAR_NAME)
				return REA_PROCESS_SENSORS;
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
				processSensorsBehaviour = ProcessSensorsBehaviour::PSB_RENDER_ALL_THEN_INTERACTIVE; // default, when no behaviour option is specified

				const auto& values = rawVariables[REA_PROCESS_SENSORS].value();
				if (!values.empty())
				{
					const char* behaviour = values[0].c_str();
					if (strcmp(behaviour, "RenderAllThenInteractive") == 0)
					{
						// Do nothing
					}
					else if (strcmp(behaviour, "RenderAllThenTerminate") == 0)
					{
						processSensorsBehaviour = ProcessSensorsBehaviour::PSB_RENDER_ALL_THEN_TERMINATE;
						if (values.size() > 1)
							printf("[WARNING]: You passed a sensor ID to the 'RenderAllThenTerminate' option, but it will be ignored.\n");
					}
					else if (strcmp(behaviour, "RenderSensorThenInteractive") == 0)
					{
						processSensorsBehaviour = ProcessSensorsBehaviour::PSB_RENDER_SENSOR_THEN_INTERACTIVE;
						sensorID = std::stoi(values[1]);
					}
					else if (strcmp(behaviour, "InteractiveAtSensor") == 0)
					{
						processSensorsBehaviour = ProcessSensorsBehaviour::PSB_INTERACTIVE_AT_SENSOR;
						sensorID = std::stoi(values[1]);
					}
					else
					{
						printf("[ERROR]: Invalid option for '%s'. Using 'RenderAllThenInteractive'.\n", PROCESS_SENSORS_VAR_NAME.data());
					}
				}
			}
		}

		variablesType rawVariables;

		// Loaded from CMD
		std::vector<std::string> sceneDirectory; // [0] zip [1] optional xml in zip
		std::string outputScreenshotsFolderPath;
		struct
		{
			ProcessSensorsBehaviour processSensorsBehaviour = ProcessSensorsBehaviour::PSB_COUNT;
			std::optional<uint32_t> sensorID;
		};

};

#endif // _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_