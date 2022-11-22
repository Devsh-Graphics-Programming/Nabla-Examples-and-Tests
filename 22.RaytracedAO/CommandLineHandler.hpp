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
-TERMINATE

Description and usage: 

-SCENE:
	some/path extra/path which will make it skip the file choose dialog

-TERMINATE:
	which will make the app stop when the required amount of samples has been renderered (its in the Mitsuba Scene metadata) and obviously take screenshot when quitting
	
Example Usages :
	raytracedao.exe -SCENE=../../media/kitchen.zip scene.xml -TERMINATE
	raytracedao.exe -SCENE="../../media/my good kitchen.zip" scene.xml -TERMINATE
	raytracedao.exe -SCENE="../../media/my good kitchen.zip scene.xml" -TERMINATE
	raytracedao.exe -SCENE="../../media/extraced folder/scene.xml" -TERMINATE
)";
 

constexpr std::string_view SCENE_VAR_NAME						= "SCENE";
constexpr std::string_view SCREENSHOT_OUTPUT_FOLDER_VAR_NAME	= "SCREENSHOT_OUTPUT_FOLDER";
constexpr std::string_view TERMINATE_VAR_NAME					= "TERMINATE";
constexpr std::string_view PROCESS_SENSORS_VAR_NAME				= "PROCESS_SENSORS";

constexpr uint32_t MaxRayTracerCommandLineArgs = 8;

enum RaytracerExampleArguments
{
	REA_SCENE,
	REA_TERMINATE,
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

		auto& getSceneDirectory() const
		{
			return sceneDirectory;
		}

		auto& getTerminate() const
		{
			return terminate;
		}

		inline ProcessSensorsBehaviour getProcessSensorBehaviour() const
		{
			const ProcessSensorsBehaviour behaviour = processSensorsBehaviourAndID.first;
			assert(behaviour != ProcessSensorsBehaviour::PSB_COUNT);
			return behaviour;
		}

		inline uint32_t getProcessSensorID() const
		{
			const uint32_t sensorID = processSensorsBehaviourAndID.second;
			assert(sensorID != ~0u);
			return sensorID;
		}

	private:

		void initializeMatchingMap()
		{
			rawVariables[REA_SCENE];
			rawVariables[REA_TERMINATE];
			rawVariables[REA_PROCESS_SENSORS];
		}

		RaytracerExampleArguments getMatchedVariableMapID(const std::string& variableName)
		{
			if (variableName == SCENE_VAR_NAME)
				return REA_SCENE;
			else if (variableName == TERMINATE_VAR_NAME)
				return REA_TERMINATE;
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
			if(rawVariables[REA_TERMINATE].has_value())
				terminate = true;
			if (rawVariables[REA_PROCESS_SENSORS].has_value())
			{
				const auto& values = rawVariables[REA_PROCESS_SENSORS].value();
				uint32_t sensorID = ~0u;
				if (values.empty())
				{
					processSensorsBehaviourAndID = { ProcessSensorsBehaviour::PSB_RENDER_ALL_THEN_INTERACTIVE, sensorID };
				}
				else
				{
					const char* behaviour = rawVariables[REA_PROCESS_SENSORS].value()[0].c_str();

					if (rawVariables[REA_PROCESS_SENSORS].value().size() > 1)
						sensorID = std::stoi(rawVariables[REA_PROCESS_SENSORS].value()[1]);

					if (strcmp(behaviour, "RenderAllThenInteractive") == 0)
						processSensorsBehaviourAndID = { ProcessSensorsBehaviour::PSB_RENDER_ALL_THEN_INTERACTIVE, sensorID };
					else if (strcmp(behaviour, "RenderAllThenTerminate") == 0)
						processSensorsBehaviourAndID = { ProcessSensorsBehaviour::PSB_RENDER_ALL_THEN_INTERACTIVE, sensorID };
					else if (strcmp(behaviour, "RenderSensorThenInteractive") == 0)
						processSensorsBehaviourAndID = { ProcessSensorsBehaviour::PSB_RENDER_SENSOR_THEN_INTERACTIVE, sensorID };
					else if (strcmp(behaviour, "InteractiveAtSensor") == 0)
						processSensorsBehaviourAndID = { ProcessSensorsBehaviour::PSB_INTERACTIVE_AT_SENSOR, sensorID };
				}
			}
		}

		variablesType rawVariables;

		// Loaded from CMD
		std::vector<std::string> sceneDirectory; // [0] zip [1] optional xml in zip
		std::string outputScreenshotsFolderPath;
		bool terminate = false;
		std::pair<ProcessSensorsBehaviour, uint32_t> processSensorsBehaviourAndID = {ProcessSensorsBehaviour::PSB_COUNT, ~0u};

};

#endif // _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_