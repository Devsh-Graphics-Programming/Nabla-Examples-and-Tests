// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "app/PathTracingApp.h"
#include "app/PathTracerAppTypes.h"

#include "argparse/argparse.hpp"
#include "nbl/examples/common/BuiltinResourcesApplication.hpp"
#include "nbl/examples/examples.hpp"
#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "io/RuntimeConfig.h"
#include "io/PathTracerReport.h"
#include "renderer/CRenderer.h"
#include "renderer/resolve/CBasicRWMCResolver.h"
#include "renderer/present/CWindowPresenter.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifndef NBL_THIS_EXAMPLE_BUILD_CONFIG
#define NBL_THIS_EXAMPLE_BUILD_CONFIG "unknown"
#endif

using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::application_templates;
using namespace nbl::examples;
using namespace nbl::this_example;
using nlohmann_json = nlohmann::json;

class PathTracingApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
	using device_base_t = SimpleWindowedApplication;
	using asset_base_t = BuiltinResourcesApplication;

	static inline void jsonizeGitInfo(nlohmann_json& target, const nbl::gtml::GitInfo& info)
	{
		target["isPopulated"] = info.isPopulated;
		if (info.hasUncommittedChanges.has_value())
			target["hasUncommittedChanges"] = info.hasUncommittedChanges.value();
		else
			target["hasUncommittedChanges"] = "UNKNOWN, BUILT WITHOUT DIRTY-CHANGES CAPTURE";

		target["commitAuthorName"] = info.commitAuthorName;
		target["commitAuthorEmail"] = info.commitAuthorEmail;
		target["commitHash"] = info.commitHash;
		target["commitShortHash"] = info.commitShortHash;
		target["commitDate"] = info.commitDate;
		target["commitSubject"] = info.commitSubject;
		target["commitBody"] = info.commitBody;
		target["describe"] = info.describe;
		target["branchName"] = info.branchName;
		target["latestTag"] = info.latestTag;
		target["latestTagName"] = info.latestTagName;
	}

	std::string makeBuildInfoJson() const
	{
		nlohmann_json j;

		auto& modules = j["modules"];
		jsonizeGitInfo(modules["nabla"],nbl::gtml::nabla_git_info);
		jsonizeGitInfo(modules["dxc"],nbl::gtml::dxc_git_info);
		return j.dump(4);
	}

	static std::string environmentValue(const char* name)
	{
		if (const char* value = std::getenv(name); value && value[0])
			return value;
		return {};
	}

	std::string makeMachineInfoJson() const
	{
		nlohmann_json machine;
#ifdef _WIN32
		machine["os"] = "Windows";
		char computerName[MAX_COMPUTERNAME_LENGTH+1] = {};
		DWORD computerNameSize = static_cast<DWORD>(sizeof(computerName)/sizeof(computerName[0]));
		if (GetComputerNameA(computerName,&computerNameSize))
			machine["hostname"] = computerName;

		MEMORYSTATUSEX memory = {};
		memory.dwLength = sizeof(memory);
		if (GlobalMemoryStatusEx(&memory))
		{
			machine["ram"] = {
				{"totalBytes",memory.ullTotalPhys},
				{"availableBytes",memory.ullAvailPhys}
			};
		}
#else
		machine["os"] = "unknown";
		machine["hostname"] = environmentValue("HOSTNAME");
#endif
		machine["cpu"] = {
			{"description",environmentValue("PROCESSOR_IDENTIFIER")},
			{"logicalThreads",std::thread::hardware_concurrency()}
		};

		if (m_device && m_device->getPhysicalDevice())
		{
			const auto* const physicalDevice = m_device->getPhysicalDevice();
			const auto& properties = physicalDevice->getProperties();
			const auto& memoryProperties = physicalDevice->getMemoryProperties();
			nlohmann_json gpu = {
				{"name",properties.deviceName},
				{"vendorId",properties.vendorID},
				{"deviceId",properties.deviceID},
				{"driverVersion",properties.driverVersion},
				{"apiVersion",{
					{"major",properties.apiVersion.major},
					{"minor",properties.apiVersion.minor},
					{"patch",properties.apiVersion.patch}
				}}
			};

			uint64_t deviceLocalBytes = 0u;
			auto& heaps = gpu["memoryHeaps"];
			heaps = nlohmann_json::array();
			using heap_flags_e = IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS;
			for (uint32_t i=0u; i<memoryProperties.memoryHeapCount; ++i)
			{
				const auto& heap = memoryProperties.memoryHeaps[i];
				const bool deviceLocal = heap.flags.hasFlags(heap_flags_e::EMHF_DEVICE_LOCAL_BIT);
				if (deviceLocal)
					deviceLocalBytes += heap.size;
				heaps.push_back({
					{"sizeBytes",heap.size},
					{"deviceLocal",deviceLocal}
				});
			}
			gpu["deviceLocalBytes"] = deviceLocalBytes;
			machine["gpu"] = std::move(gpu);
		}

		return machine.dump(2);
	}

	void printGitInfos() const
	{
		m_logger->log("Build Info:\n%s",ILogger::ELL_INFO,m_buildInfoJson.c_str());
	}

	path resolvePathAgainstCurrentWorkingDirectory(const path& candidate) const
	{
		if (candidate.empty() || candidate.is_absolute())
			return candidate;
		return (std::filesystem::current_path()/candidate).lexically_normal();
	}

	void normalizeSceneArchiveFallback(SceneJob& job) const
	{
		if (!hasZipExtension(job.scenePath) || std::filesystem::exists(job.scenePath))
			return;

		path unpackedDirectory = job.scenePath;
		unpackedDirectory.replace_extension();
		const path entry = job.sceneEntry.value_or(path("scene.xml"));
		const path unpackedScene = (unpackedDirectory/entry).lexically_normal();
		if (!std::filesystem::exists(unpackedScene))
			return;

		job.scenePath = unpackedScene;
		job.sceneEntry = std::nullopt;
	}

	bool readOptionDouble(const std::vector<std::string>& tokens, size_t& index, std::string_view option, double& outValue) const
	{
		const auto& token = tokens[index];
		std::string rawValue;
		const std::string optionWithEquals = std::string(option)+"=";
		if (token.rfind(optionWithEquals,0u)==0u)
			rawValue = token.substr(optionWithEquals.size());
		else
		{
			if (index+1u>=tokens.size())
			{
				std::fprintf(stderr,"Scene list option %s requires a value.\n",std::string(option).c_str());
				return false;
			}
			rawValue = tokens[++index];
		}

		try
		{
			outValue = std::stod(rawValue);
		}
		catch (const std::exception&)
		{
			std::fprintf(stderr,"Scene list option %s has invalid value: %s\n",std::string(option).c_str(),rawValue.c_str());
			return false;
		}
		return true;
	}

	bool parseSceneListJob(const std::vector<std::string>& tokens, const uint32_t lineNumber, SceneJob& job) const
	{
		job.sensor = m_args.sensor;
		job.deferPostProcess = m_args.deferPostProcess;
		job.compare = m_args.compare;

		std::optional<std::string> sceneValue;
		for (size_t i=0u; i<tokens.size(); ++i)
		{
			const auto& token = tokens[i];
			using mode_t = PathTracerReport::SCompareSettings::EAllowedErrorPixelMode;

			if (token=="--abs")
			{
				job.compare.allowedErrorPixelMode = mode_t::AbsoluteCount;
				continue;
			}
			if (token=="--rel")
			{
				job.compare.allowedErrorPixelMode = mode_t::RelativeToResolution;
				continue;
			}
			if (token=="--errcount" || token.rfind("--errcount=",0u)==0u)
			{
				double value = 0.0;
				if (!readOptionDouble(tokens,i,"--errcount",value))
					return false;
				if (job.compare.allowedErrorPixelMode==mode_t::AbsoluteCount)
					job.compare.allowedErrorPixelCount = static_cast<uint64_t>(std::ceil(std::max(0.0,value)));
				else
					job.compare.allowedErrorPixelRatio = value;
				continue;
			}
			if (token=="--errpixel" || token.rfind("--errpixel=",0u)==0u)
			{
				if (!readOptionDouble(tokens,i,"--errpixel",job.compare.errorThreshold))
					return false;
				continue;
			}
			if (token=="--errssim" || token.rfind("--errssim=",0u)==0u)
			{
				if (!readOptionDouble(tokens,i,"--errssim",job.compare.ssimErrorThreshold))
					return false;
				continue;
			}
			if (token=="--epsilon" || token.rfind("--epsilon=",0u)==0u)
			{
				if (!readOptionDouble(tokens,i,"--epsilon",job.compare.epsilon))
					return false;
				continue;
			}
			if (token=="-DEFER_DENOISE" || token=="--defer-denoise")
			{
				job.deferPostProcess = true;
				continue;
			}
			if (token=="-TERMINATE")
				continue;

			if (isOptionToken(token))
			{
				std::fprintf(stderr,"Unsupported scene list option on line %u: %s\n",lineNumber,token.c_str());
				return false;
			}

			if (!sceneValue.has_value())
			{
				sceneValue = token;
				continue;
			}
			if (hasZipExtension(path(sceneValue.value())) && !job.sceneEntry.has_value())
			{
				job.sceneEntry = path(token);
				continue;
			}

			std::fprintf(stderr,"Unexpected scene list token on line %u: %s\n",lineNumber,token.c_str());
			return false;
		}

		if (!sceneValue.has_value())
		{
			std::fprintf(stderr,"Scene list line %u does not contain a scene path.\n",lineNumber);
			return false;
		}

		std::optional<std::string> archiveEntry;
		auto sceneText = sceneValue.value();
		splitSceneArchiveEntry(sceneText,archiveEntry);
		job.scenePath = resolvePathAgainstCurrentWorkingDirectory(path(stripWrappingQuotes(sceneText)));
		if (archiveEntry.has_value())
			job.sceneEntry = path(stripWrappingQuotes(archiveEntry.value()));
		normalizeSceneArchiveFallback(job);
		return true;
	}

	bool loadSceneList(path sceneListPath)
	{
		sceneListPath = resolvePathAgainstCurrentWorkingDirectory(sceneListPath);
		std::ifstream file(sceneListPath);
		if (!file)
		{
			std::fprintf(stderr,"Failed to open scene list: %s\n",sceneListPath.string().c_str());
			return false;
		}

		std::string line;
		uint32_t lineNumber = 0u;
		while (std::getline(file,line))
		{
			++lineNumber;
			const auto stripped = trim(line);
			if (stripped.empty() || stripped.front()==';')
				continue;

			SceneJob job;
			if (!parseSceneListJob(tokenizeSceneListLine(stripped),lineNumber,job))
				return false;
			m_sceneJobs.push_back(std::move(job));
		}

		if (m_sceneJobs.empty())
		{
			std::fprintf(stderr,"Scene list does not contain any runnable scenes: %s\n",sceneListPath.string().c_str());
			return false;
		}
		m_args.sceneListPath = sceneListPath;
		return true;
	}

	bool parseCommandLine()
	{
		auto normalizedArguments = normalizeCompatibleArguments(argv);
		if (!applyRuntimeConfigDefaults(normalizedArguments))
			return false;
		if (normalizedArguments.empty())
		{
			std::fprintf(stderr,"Failed to parse arguments: no arguments are available.\n");
			return false;
		}

		argparse::ArgumentParser parser("40_pathtracer","1.0");
		parser.add_description("Path tracer CLI with ditt-compatible Mitsuba scene loading and sensor batch processing.");
		parser.add_argument("--scene")
			.help("Path to a Mitsuba XML file or a ZIP archive.");
		parser.add_argument("--scene-list")
			.help("Text file with old CI scene lines to render into a single report.");
		parser.add_argument("--scene-entry")
			.help("Optional XML path inside a ZIP archive.");
		parser.add_argument("--mode","--process-sensors")
			.help("Sensor workflow: RenderAllThenInteractive, RenderAllThenTerminate, RenderSensorThenInteractive or InteractiveAtSensor.")
			.default_value(std::string("RenderAllThenInteractive"));
		parser.add_argument("--sensor")
			.help("Sensor index for workflows that start from a chosen sensor.")
			.scan<'u',uint32_t>()
			.default_value(0u);
		parser.add_argument("--headless")
			.help("Disable swapchain creation and run without a presentation window.")
			.default_value(false)
			.implicit_value(true);
		parser.add_argument("--defer-denoise")
			.help("Queue internal postprocess finalization until shutdown.")
			.default_value(false)
			.implicit_value(true);
		parser.add_argument("--output-dir")
			.help("Prefix directory for relative film output paths.");
		parser.add_argument("--report-dir")
			.help("Directory for the self-contained HTML report bundle.");
		parser.add_argument("--reference-dir")
			.help("Optional directory with reference EXR files for native image comparison.");
		parser.add_argument("--compare-error-threshold")
			.help("Relative per-channel error threshold for native image comparison.")
			.scan<'g',double>()
			.default_value(0.05);
		parser.add_argument("--compare-epsilon")
			.help("Absolute epsilon used before relative image comparison.")
			.scan<'g',double>()
			.default_value(0.00001);
		parser.add_argument("--compare-allowed-error-ratio")
			.help("Allowed ratio of pixels that may exceed the native comparison threshold.")
			.scan<'g',double>()
			.default_value(0.0001);
		parser.add_argument("--compare-allowed-error-count")
			.help("Absolute allowed count of pixels that may exceed the native comparison threshold.")
			.scan<'g',double>();
		parser.add_argument("--compare-ssim-threshold")
			.help("Maximum allowed SSIM difference for denoised output comparison.")
			.scan<'g',double>()
			.default_value(0.001);

		try
		{
			parser.parse_args(normalizedArguments);
		}
		catch (const std::exception& e)
		{
			std::fprintf(stderr,"Failed to parse arguments: %s\n", e.what());
			return false;
		}

		const auto workflowName = parser.get<std::string>("--process-sensors");
		const auto workflow = parseSensorWorkflow(workflowName);
		if (!workflow.has_value())
		{
			std::fprintf(stderr,"Unsupported sensor workflow: %s\n", workflowName.c_str());
			return false;
		}
		m_args.workflow = workflow.value();
		m_args.sensor = parser.get<uint32_t>("--sensor");
		m_args.headless = parser.get<bool>("--headless");
		m_args.deferPostProcess = parser.get<bool>("--defer-denoise");

		if (const auto outputDir = parser.present("--output-dir"); outputDir.has_value())
			m_args.outputDir = resolvePathAgainstCurrentWorkingDirectory(stripWrappingQuotes(outputDir.value()));
		if (const auto reportDir = parser.present("--report-dir"); reportDir.has_value())
			m_args.reportDir = resolvePathAgainstCurrentWorkingDirectory(stripWrappingQuotes(reportDir.value()));
		if (const auto referenceDir = parser.present("--reference-dir"); referenceDir.has_value())
			m_args.referenceDir = resolvePathAgainstCurrentWorkingDirectory(stripWrappingQuotes(referenceDir.value()));
		m_args.compare.errorThreshold = parser.get<double>("--compare-error-threshold");
		m_args.compare.epsilon = parser.get<double>("--compare-epsilon");
		m_args.compare.allowedErrorPixelRatio = parser.get<double>("--compare-allowed-error-ratio");
		if (const auto allowedErrorCount = parser.present<double>("--compare-allowed-error-count"); allowedErrorCount.has_value())
		{
			m_args.compare.allowedErrorPixelMode = PathTracerReport::SCompareSettings::EAllowedErrorPixelMode::AbsoluteCount;
			m_args.compare.allowedErrorPixelCount = static_cast<uint64_t>(std::ceil(std::max(0.0,allowedErrorCount.value())));
		}
		m_args.compare.ssimErrorThreshold = parser.get<double>("--compare-ssim-threshold");

		const auto sceneValue = parser.present("--scene");
		const auto sceneListValue = parser.present("--scene-list");
		if (sceneValue.has_value() && sceneListValue.has_value())
		{
			std::fprintf(stderr,"Use either --scene or --scene-list, not both.\n");
			return false;
		}
		if (!sceneValue.has_value() && !sceneListValue.has_value())
		{
			std::fprintf(stderr,"Scene path is required. Use --scene, --scene-list or -SCENE=...\n");
			return false;
		}

		if (sceneListValue.has_value())
			return loadSceneList(path(stripWrappingQuotes(sceneListValue.value())));

		SceneJob job;
		job.scenePath = resolvePathAgainstCurrentWorkingDirectory(stripWrappingQuotes(sceneValue.value()));
		if (const auto sceneEntry = parser.present("--scene-entry"); sceneEntry.has_value())
			job.sceneEntry = path(stripWrappingQuotes(sceneEntry.value()));
		job.sensor = m_args.sensor;
		job.deferPostProcess = m_args.deferPostProcess;
		job.compare = m_args.compare;
		normalizeSceneArchiveFallback(job);
		m_sceneJobs.push_back(std::move(job));

		return true;
	}

	path composeSceneLoadPath() const
	{
		if (!hasZipExtension(m_args.scenePath))
			return m_args.scenePath;

		const path entry = m_args.sceneEntry.value_or(path("scene.xml"));
		return (m_args.scenePath/entry).lexically_normal();
	}

	path getSceneWorkingDirectory() const
	{
		if (m_args.scenePath.has_parent_path())
			return m_args.scenePath.parent_path();
		return std::filesystem::current_path();
	}

	bool failSceneJob(const char* message)
	{
		m_lastSceneLoadError = message ? message:"Scene failed before render session was queued.";
		return logFail(m_lastSceneLoadError.c_str());
	}

	std::string sceneNameForJob(const SceneJob& job) const
	{
		return sceneReferenceNameFromPath(job.scenePath,job.sceneEntry);
	}

	std::vector<std::string> referenceNamesForOutput(const std::string& outputName, const std::string& sourceSceneName, const std::string& originalOutputName) const
	{
		std::vector<std::string> names;
		addUniqueName(names,outputName);
		addUniqueName(names,sourceSceneName);
		addUniqueName(names,originalOutputName);
		return names;
	}

	std::vector<std::string> referenceSessionNamesForJob(const SceneJob& job) const
	{
		const auto sourceSceneName = sceneNameForJob(job);
		std::vector<std::string> names;
		if (!m_args.referenceDir.empty())
		{
			std::error_code ec;
			for (const auto& entry : std::filesystem::directory_iterator(m_args.referenceDir,ec))
			{
				if (!entry.is_directory(ec))
					continue;
				const auto referenceName = entry.path().filename().string();
				if (referenceName==sourceSceneName || referenceName.rfind(sourceSceneName+"_",0u)==0u)
					addUniqueName(names,referenceName);
			}
		}
		if (names.empty())
			names.push_back(sourceSceneName);
		std::sort(names.begin(),names.end());
		return names;
	}

	ExportOutputs buildExpectedOutputsForJob(const SceneJob& job, const size_t jobIndex, const std::string& outputName) const
	{
		const auto sourceSceneName = sceneNameForJob(job);
		auto outputFile = (std::string("Render_")+outputName+".exr");
		path outputPath = m_args.outputDir.empty() ? (localOutputCWD/outputFile):(m_args.outputDir/outputFile);
		outputPath = outputPath.lexically_normal();

		const auto artifactName = makeArtifactName(jobIndex,outputName);
		outputPath = outputPath.parent_path()/artifactName/outputPath.filename();

		ExportOutputs outputs;
		outputs.artifactName = artifactName;
		outputs.displayName = outputName;
		outputs.referenceNames = referenceNamesForOutput(outputName,sourceSceneName,outputName);
		outputs.tonemap = outputPath;
		outputs.rwmcCascades = appendBeforeExtension(outputPath,"_rwmc_cascades");
		outputs.albedo = appendBeforeExtension(outputPath,"_albedo");
		outputs.normal = appendBeforeExtension(outputPath,"_normal");
		outputs.denoised = appendBeforeExtension(outputPath,"_denoised");
		return outputs;
	}

	void recordFailedSceneJob(const SceneJob& job)
	{
		if (!m_report)
			return;

		for (const auto& outputName : referenceSessionNamesForJob(job))
		{
			const auto outputs = buildExpectedOutputsForJob(job,m_nextSceneJob,outputName);
			m_report->addSession(PathTracerReport::SSession{
				.sceneName = outputs.artifactName,
				.displayName = outputName,
				.referenceNames = outputs.referenceNames,
				.scenePath = job.scenePath,
				.sensorIndex = job.sensor,
				.status = "failed",
				.details = m_lastSceneLoadError.empty() ? "Scene failed before render session was queued.":m_lastSceneLoadError,
				.compare = job.compare,
				.images = {
					{"tonemap","Tonemap",outputs.tonemap,true},
					{"rwmc_cascades","RWMC Cascades",outputs.rwmcCascades,false},
					{"albedo","Albedo",outputs.albedo,true},
					{"normal","Normal",outputs.normal,true},
					{"denoised","Denoised",outputs.denoised,true}
				}
			});
		}
	}

	bool shouldExitAfterQueue() const
	{
		return m_args.headless || m_args.workflow==SensorWorkflow::RenderAllThenTerminate;
	}

	std::string makePortableCommandLineArgument(std::string argument) const
	{
		if (argument.empty())
			return argument;

		const auto equals = argument.find('=');
		if (equals!=std::string::npos && equals+1u<argument.size())
			return argument.substr(0u,equals+1u)+makePortableCommandLineArgument(argument.substr(equals+1u));

		const auto stripped = stripWrappingQuotes(argument);
		path argumentPath = stripped;
		if (!argumentPath.is_absolute())
			return argument;

		std::error_code ec;
		auto relative = std::filesystem::relative(argumentPath,std::filesystem::current_path(),ec);
		if (!ec && !relative.empty())
			return relative.generic_string();
		return argumentPath.filename().generic_string();
	}

	std::string makeCommandLine() const
	{
		std::string commandLine;
		for (size_t i=0u; i<argv.size(); ++i)
		{
			std::string argument = i==0u ? path(argv[i]).filename().string():makePortableCommandLineArgument(argv[i]);
			if (!commandLine.empty())
				commandLine += ' ';
			if (argument.find_first_of(" \t\"")!=std::string::npos)
				commandLine += '"' + argument + '"';
			else
				commandLine += argument;
		}
		return commandLine;
	}

	path getReportDirectory() const
	{
		if (!m_args.reportDir.empty())
			return m_args.reportDir;
		if (!m_args.outputDir.empty())
			return m_args.outputDir;
		return localOutputCWD;
	}

	void createReport()
	{
		m_report = std::make_unique<PathTracerReport>(PathTracerReport::SCreationParams{
			.reportDir = getReportDirectory(),
			.referenceDir = m_args.referenceDir,
			.workingDirectory = std::filesystem::current_path(),
			.lowDiscrepancySequenceCachePath = sharedOutputCWD/nbl::examples::CCachedOwenScrambledSequence::SCreationParams::DefaultFilename,
			.commandLine = makeCommandLine(),
			.buildConfig = NBL_THIS_EXAMPLE_BUILD_CONFIG,
			.buildInfoJson = m_buildInfoJson,
			.machineInfoJson = makeMachineInfoJson(),
			.compare = m_args.compare,
			.assetManager = m_assetMgr.get(),
			.logger = m_logger.get()
		});
	}

public:
	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!parseCommandLine())
			return false;

		if (!m_args.headless)
			m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

		if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;

		if (m_args.headless)
		{
			if (!BasicMultiQueueApplication::onAppInitialized(smart_refctd_ptr(system)))
				return false;
		}
		else if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;

		m_buildInfoJson = makeBuildInfoJson();
		printGitInfos();
		createReport();

		if (!m_args.headless)
		{
			m_presenter = CWindowPresenter::create({
				{
					.assMan = m_assetMgr,
					.logger = smart_refctd_ptr(m_logger)
				},
				{
					.winMgr = m_winMgr
				},
				m_api,
				make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem),smart_refctd_ptr(m_logger)),
				"Path Tracer"
			});
			if (!m_presenter)
				return logFail("Failed to create CWindowPresenter");
		}

		m_renderer = CRenderer::create({
			{
				.graphicsQueue = getGraphicsQueue(),
				.computeQueue = getComputeQueue(),
				.uploadQueue = getTransferUpQueue(),
				.utilities = smart_refctd_ptr(m_utils)
			},
			m_assetMgr.get(),
			(sharedOutputCWD/nbl::examples::CCachedOwenScrambledSequence::SCreationParams::DefaultFilename).string()
		});
		if (!m_renderer)
			return logFail("Failed to create CRenderer");

		if (!m_args.headless && !m_presenter->init(m_renderer.get()))
			return logFail("Failed to initialize CWindowPresenter");

		m_resolver = CBasicRWMCResolver::create({
			{},
			m_renderer.get(),
			m_assetMgr.get()
		});
		if (!m_resolver)
			return logFail("Failed to create CBasicRWMCResolver");

		m_sceneLoader = CSceneLoader::create({
			{
				.assMan = smart_refctd_ptr(m_assetMgr),
				.logger = smart_refctd_ptr(m_logger)
			}
		});
		if (!m_sceneLoader)
			return logFail("Failed to create CSceneLoader");

		if (!loadNextSceneJob())
			m_exitRequested = true;
		return true;
	}

	bool loadNextSceneJob()
	{
		while (m_nextSceneJob<m_sceneJobs.size())
		{
			while (!m_sessionQueue.empty())
				m_sessionQueue.pop();

			if (m_resolver && m_resolver->getActiveSession())
			{
				if (m_device && m_device->waitIdle()!=IQueue::RESULT::SUCCESS)
					return logFail("Failed to idle the device before loading the next scene.");
				m_resolver->clearSession();
			}
			m_scene = {};

			const auto job = m_sceneJobs[m_nextSceneJob++];
			m_activeSceneJob = job;
			m_activeSceneJobIndex = m_nextSceneJob;
			m_args.scenePath = job.scenePath;
			m_args.sceneEntry = job.sceneEntry;
			m_args.sensor = job.sensor;
			m_args.deferPostProcess = job.deferPostProcess;
			m_args.compare = job.compare;
			m_lastFinalizedSession = nullptr;
			m_lastSceneLoadError.clear();

			m_logger->log("Starting scene job %u/%u",ILogger::ELL_INFO,static_cast<uint32_t>(m_nextSceneJob),static_cast<uint32_t>(m_sceneJobs.size()));
			if (initializeSceneAndQueueSessions())
				return true;
			recordFailedSceneJob(job);
		}

		return false;
	}

	bool initializeSceneAndQueueSessions()
	{
		const auto loadPath = composeSceneLoadPath();
		m_logger->log("Loading scene from \"%s\"",ILogger::ELL_INFO,loadPath.string().c_str());

		m_scene = m_renderer->createScene({
			.load = m_sceneLoader->load({
				.relPath = loadPath,
				.workingDirectory = getSceneWorkingDirectory()
			}),
			.converter = nullptr
		});
		if (!m_scene)
			return failSceneJob("Could not create scene");

		const auto sensors = m_scene->getSensors();
		if (sensors.empty())
			return failSceneJob("Loaded scene does not expose any sensors.");

		uint32_t sensorIndex = m_args.sensor;
		if (sensorIndex>=sensors.size())
		{
			m_logger->log("Requested sensor %u is out of range, defaulting to sensor 0.",ILogger::ELL_WARNING,sensorIndex);
			sensorIndex = 0u;
		}
		m_args.sensor = sensorIndex;

		const auto enqueueSensor = [&](const uint32_t index)->bool
		{
			auto session = m_scene->createSession({
				{.mode = CSession::RenderMode::Beauty},
				&sensors[index]
			});
			if (!session)
				return false;
			m_sessionQueue.push(std::move(session));
			return true;
		};

		switch (m_args.workflow)
		{
			case SensorWorkflow::RenderAllThenInteractive:
			case SensorWorkflow::RenderAllThenTerminate:
				for (uint32_t index = sensorIndex; index < sensors.size(); ++index)
				{
					if (!enqueueSensor(index))
						return failSceneJob("Failed to queue render sessions");
				}
				break;
			case SensorWorkflow::RenderSensorThenInteractive:
			case SensorWorkflow::InteractiveAtSensor:
				if (!enqueueSensor(sensorIndex))
					return failSceneJob("Failed to queue render sessions");
				break;
		}

		return true;
	}

	bool activateNextQueuedSession()
	{
		if (m_sessionQueue.empty())
			return false;

		auto session = std::move(m_sessionQueue.front());
		m_sessionQueue.pop();
		auto* const rawSession = session.get();
		const auto initialized = m_utils->autoSubmit<SIntendedSubmitInfo>({.queue=getGraphicsQueue()},[rawSession](SIntendedSubmitInfo& info)->bool
			{
				return rawSession->init(info);
			}
		).copy<IQueue::RESULT>();
		if (initialized!=IQueue::RESULT::SUCCESS)
		{
			m_logger->log("Failed to initialize render session.",ILogger::ELL_ERROR);
			return false;
		}
		if (!m_resolver->changeSession(std::move(session)))
		{
			m_logger->log("Failed to initialize render session.",ILogger::ELL_ERROR);
			return false;
		}
		m_lastFinalizedSession = nullptr;
		return true;
	}

	std::string makeDisplayPath(const path& filePath) const
	{
		std::error_code ec;
		const auto relativePath = std::filesystem::relative(filePath,std::filesystem::current_path(),ec);
		if (!ec && !relativePath.empty())
			return makeGenericPathString(relativePath);
		return makeGenericPathString(filePath);
	}

	ExportOutputs buildOutputPaths(const CSession& session) const
	{
		auto outputFile = session.getConstructionParams().outputFilePath;
		if (outputFile.empty())
			outputFile = "Render.exr";
		if (outputFile.is_relative())
		{
			if (!m_args.outputDir.empty())
				outputFile = m_args.outputDir/outputFile;
			else
				outputFile = localOutputCWD/outputFile;
		}
		outputFile = outputFile.lexically_normal();
		outputFile.replace_extension(".exr");
		const auto originalOutputName = sceneNameFromOutputPath(outputFile);
		const auto sourceSceneName = sceneNameForJob(m_activeSceneJob);
		replaceDeducedSceneOutputName(outputFile,sourceSceneName);
		const auto outputName = sceneNameFromOutputPath(outputFile);
		const auto artifactName = makeArtifactName(m_activeSceneJobIndex,outputName);
		outputFile = outputFile.parent_path()/artifactName/outputFile.filename();

		ExportOutputs outputs;
		outputs.artifactName = artifactName;
		outputs.displayName = outputName;
		outputs.referenceNames = referenceNamesForOutput(outputName,sourceSceneName,originalOutputName);
		outputs.tonemap = outputFile;
		outputs.rwmcCascades = appendBeforeExtension(outputFile,"_rwmc_cascades");
		outputs.albedo = appendBeforeExtension(outputFile,"_albedo");
		outputs.normal = appendBeforeExtension(outputFile,"_normal");
		outputs.denoised = appendBeforeExtension(outputFile,"_denoised");

		outputs.tonemapDisplay = makeDisplayPath(outputs.tonemap);
		outputs.rwmcCascadesDisplay = makeDisplayPath(outputs.rwmcCascades);
		outputs.albedoDisplay = makeDisplayPath(outputs.albedo);
		outputs.normalDisplay = makeDisplayPath(outputs.normal);
		outputs.denoisedDisplay = makeDisplayPath(outputs.denoised);
		return outputs;
	}

	bool ensureParentDirectoryExists(const path& filePath) const
	{
		const auto parent = filePath.parent_path();
		if (parent.empty())
			return true;
		std::error_code ec;
		std::filesystem::create_directories(parent,ec);
		return !ec;
	}

	bool writeImageView(const ICPUImageView* view, const path& destination) const
	{
		if (!view)
			return false;
		if (!ensureParentDirectoryExists(destination))
			return false;

		IAssetWriter::SAssetWriteParams writeParams(const_cast<ICPUImageView*>(view),EWF_NONE,0.f,0u,nullptr,nullptr,logger_opt_ptr(m_logger.get()));
		if (!m_assetMgr->writeAsset(destination.string(),writeParams))
		{
			m_logger->log("Failed to write \"%s\"",ILogger::ELL_ERROR,destination.string().c_str());
			return false;
		}
		return true;
	}

	smart_refctd_ptr<ICPUImageView> makeImageView(smart_refctd_ptr<ICPUImage>&& image) const
	{
		if (!image)
			return nullptr;

		const auto& params = image->getCreationParameters();
		ICPUImageView::SCreationParams viewParams = {};
		viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = std::move(image);
		viewParams.format = params.format;
		switch (params.type)
		{
			case ICPUImage::ET_1D:
				viewParams.viewType = params.arrayLayers>1u ? ICPUImageView::ET_1D_ARRAY:ICPUImageView::ET_1D;
				break;
			case ICPUImage::ET_2D:
				viewParams.viewType = params.arrayLayers>1u ? ICPUImageView::ET_2D_ARRAY:ICPUImageView::ET_2D;
				break;
			default:
				viewParams.viewType = ICPUImageView::ET_3D;
				break;
		}
		viewParams.subresourceRange.aspectMask = IImage::EAF_COLOR_BIT;
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = params.arrayLayers;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = params.mipLevels;
		return ICPUImageView::create(std::move(viewParams));
	}

	bool exportImageView(const IGPUImageView* view, const path& exrDestination) const
	{
		if (!view)
			return false;

		auto cpuImageView = nbl::ext::ScreenShot::createScreenShot(
			m_device.get(),
			getGraphicsQueue()->getUnderlyingQueue(),
			nullptr,
			view,
			ACCESS_FLAGS::SHADER_WRITE_BITS,
			IImage::LAYOUT::GENERAL
		);
		if (!cpuImageView)
		{
			m_logger->log("Failed to read back image for export",ILogger::ELL_ERROR);
			return false;
		}
		return writeImageView(cpuImageView.get(),exrDestination);
	}

	bool copyOutputFile(const path& source, const path& destination) const
	{
		if (!ensureParentDirectoryExists(destination))
			return false;

		std::error_code ec;
		std::filesystem::copy_file(source,destination,std::filesystem::copy_options::overwrite_existing,ec);
		if (ec)
		{
			m_logger->log("Failed to copy \"%s\" to \"%s\": %s",ILogger::ELL_ERROR,source.string().c_str(),destination.string().c_str(),ec.message().c_str());
			return false;
		}
		return true;
	}

	bool exportCompletedSession(const CSession& session, const ExportOutputs& outputs)
	{
		const auto& immutables = session.getActiveResources().immutables;

		const auto* const tonemapView = immutables.beauty.getView(E_FORMAT::EF_R16G16B16A16_SFLOAT);
		if (!tonemapView)
		{
			m_logger->log("Missing image view for export format %s",ILogger::ELL_ERROR,to_string(E_FORMAT::EF_R16G16B16A16_SFLOAT).c_str());
			return false;
		}
		if (!exportImageView(tonemapView,outputs.tonemap))
			return false;

		const auto* const rwmcCascadesView = immutables.rwmcCascades.getView(E_FORMAT::EF_R16G16B16A16_SFLOAT);
		if (!rwmcCascadesView)
		{
			m_logger->log("Missing image view for export format %s",ILogger::ELL_ERROR,to_string(E_FORMAT::EF_R16G16B16A16_SFLOAT).c_str());
			return false;
		}
		if (!exportImageView(rwmcCascadesView,outputs.rwmcCascades))
			return false;

		const auto* const albedoView = immutables.albedo.getView(E_FORMAT::EF_R16G16B16A16_SFLOAT);
		if (!albedoView)
		{
			m_logger->log("Missing image view for export format %s",ILogger::ELL_ERROR,to_string(E_FORMAT::EF_R16G16B16A16_SFLOAT).c_str());
			return false;
		}
		if (!exportImageView(albedoView,outputs.albedo))
			return false;

		const auto* const normalView = immutables.normal.getView(E_FORMAT::EF_R16G16B16A16_SFLOAT);
		if (!normalView)
		{
			m_logger->log("Missing image view for export format %s",ILogger::ELL_ERROR,to_string(E_FORMAT::EF_R16G16B16A16_SFLOAT).c_str());
			return false;
		}
		if (!exportImageView(normalView,outputs.normal))
			return false;

		return true;
	}

	bool waitForRenderedSubmit(const IQueue::SSubmitInfo::SSemaphoreInfo& rendered) const
	{
		if (!rendered.semaphore)
			return true;
		const ISemaphore::SWaitInfo waitInfo = {
			.semaphore = rendered.semaphore,
			.value = rendered.value
		};
		return m_device->blockForSemaphores({&waitInfo,&waitInfo+1})==ISemaphore::WAIT_RESULT::SUCCESS;
	}

	bool runPostProcessJob(const PostProcessJob& job)
	{
		return copyOutputFile(job.outputs.tonemap,job.outputs.denoised);
	}

	void emitOutputJson(const ExportOutputs& outputs) const
	{
		nlohmann_json payload = {
			{"output_tonemap",outputs.tonemapDisplay},
			{"output_rwmc_cascades",outputs.rwmcCascadesDisplay},
			{"output_albedo",outputs.albedoDisplay},
			{"output_normal",outputs.normalDisplay},
			{"output_denoised",outputs.denoisedDisplay}
		};
		std::cout << "[JSON] " << payload.dump() << "\n[ENDJSON]\n";
	}

	bool addReportSession(const ExportOutputs& outputs)
	{
		if (!m_report)
			return true;

		PathTracerReport::SSession reportSession;
		reportSession.sceneName = outputs.artifactName;
		reportSession.displayName = outputs.displayName.empty() ? outputs.artifactName:outputs.displayName;
		reportSession.referenceNames = outputs.referenceNames;
		reportSession.scenePath = m_args.scenePath;
		reportSession.sensorIndex = m_args.sensor;
		reportSession.compare = m_args.compare;
		reportSession.images = {
			{"tonemap","Tonemap",outputs.tonemap,true},
			{"rwmc_cascades","RWMC Cascades",outputs.rwmcCascades,false},
			{"albedo","Albedo",outputs.albedo,true},
			{"normal","Normal",outputs.normal,true},
			{"denoised","Denoised",outputs.denoised,true}
		};
		return m_report->addSession(std::move(reportSession));
	}

	bool finalizeCompletedSession(CSession* session, const IQueue::SSubmitInfo::SSemaphoreInfo& rendered)
	{
		if (!session || session==m_lastFinalizedSession)
			return true;
		if (!waitForRenderedSubmit(rendered))
			return false;

		const auto outputs = buildOutputPaths(*session);
		if (!exportCompletedSession(*session,outputs))
			return false;

		const PostProcessJob job = {
			.outputs = outputs,
			.postProcess = session->getConstructionParams().postProcess
		};
		if (m_args.deferPostProcess)
			m_deferredPostProcessJobs.push_back(job);
		else if (!runPostProcessJob(job))
			return false;

		emitOutputJson(outputs);
		if (!addReportSession(outputs))
			return false;
		m_lastFinalizedSession = session;
		return true;
	}

	bool flushDeferredPostProcessJobs()
	{
		for (const auto& job : m_deferredPostProcessJobs)
		{
			if (!runPostProcessJob(job))
				return false;
		}
		m_deferredPostProcessJobs.clear();
		return true;
	}

	SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	{
		auto retval = device_base_t::getRequiredDeviceFeatures();
		if (m_args.headless)
			retval.swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;
		return retval.unionWith(CRenderer::RequiredDeviceFeatures());
	}

	SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
	{
		auto retval = device_base_t::getPreferredDeviceFeatures();
		if (m_args.headless)
			retval.swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;
		return retval.unionWith(CRenderer::PreferredDeviceFeatures());
	}

	IAPIConnection::SFeatures getAPIFeaturesToEnable() override
	{
		auto retval = device_base_t::getAPIFeaturesToEnable();
		if (m_args.headless)
		{
			retval.swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;
			retval.validations = false;
			retval.synchronizationValidation = false;
			retval.debugUtils = false;
		}
		return retval;
	}

	SPhysicalDeviceLimits getRequiredDeviceLimits() const override
	{
		auto retval = device_base_t::getRequiredDeviceLimits();
		retval.rayTracingInvocationReorder = true;
		retval.rayTracingPositionFetch = true;
		retval.shaderStorageImageReadWithoutFormat = true;
		return retval;
	}

	void filterDevices(nbl::core::set<IPhysicalDevice*>& physicalDevices) const override
	{
		device_base_t::filterDevices(physicalDevices);
		if (m_args.headless)
			return;
		std::erase_if(physicalDevices,[&](const IPhysicalDevice* device)->bool
			{
				const auto& props = device->getMemoryProperties();
				uint64_t largestVRAMHeap = 0;
				using heap_flags_e = IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS;
				for (uint32_t h=0; h<props.memoryHeapCount; h++)
				if (const auto& heap=props.memoryHeaps[h]; heap.flags.hasFlags(heap_flags_e::EMHF_DEVICE_LOCAL_BIT))
					largestVRAMHeap = nbl::hlsl::max(largestVRAMHeap,heap.size);
				const auto typeBits = device->getDirectVRAMAccessMemoryTypeBits();
				for (uint32_t t=0; t<props.memoryTypeCount; t++)
				if (((typeBits>>t)&0x1u) && props.memoryHeaps[props.memoryTypes[t].heapIndex].size==largestVRAMHeap)
					return false;
				m_logger->log("Filtering out Device %p (%s) due to lack of ReBAR",ILogger::ELL_WARNING,device,device->getProperties().deviceName);
				return true;
			}
		);
	}

	nbl::core::vector<SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		if (m_args.headless)
			return {};

		if (m_presenter)
			return {{static_cast<const CWindowPresenter*>(m_presenter.get())->getSurface()}};
		return {};
	}

	void workLoopBody() override
	{
		CSession* session = m_resolver ? m_resolver->getActiveSession() : nullptr;
		bool sameSession = true;
		while (!session || session->getProgress()>=1.f)
		{
			if (!session)
			{
				if (!activateNextQueuedSession())
				{
					if (!m_args.headless)
						handleInputs();
					return;
				}
			}
			else if (m_sessionQueue.empty())
			{
				if (m_args.headless && loadNextSceneJob())
				{
					session = nullptr;
					sameSession = false;
					continue;
				}
				if (!m_args.headless)
					handleInputs();
				return;
			}
			else if (!activateNextQueuedSession())
			{
				m_exitRequested = true;
				return;
			}

			session = m_resolver->getActiveSession();
			sameSession = false;
		}

		if (sameSession)
			session->update(session->getActiveResources().prevSensorState);

		m_api->startCapture();
		auto deferredSubmit = m_renderer->render(session);
		if (!deferredSubmit)
		{
			m_exitRequested = true;
			m_api->endCapture();
			return;
		}

		IGPUCommandBuffer* const cb = deferredSubmit;
		const auto maxSPP = session->getConstructionParams().initDynamics.maxSPP;
		const auto samplesAfterDispatch = session->getSamplesDispatched()+CRenderer::BeautySamplesPerDispatch;
		const bool shouldResolve = !m_args.headless || maxSPP<=samplesAfterDispatch;
		if (shouldResolve)
		{
			if (!m_resolver->resolve(cb,nullptr))
			{
				m_exitRequested = true;
				m_api->endCapture();
				return;
			}
			deferredSubmit.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
		}
		auto rendered = deferredSubmit({});
		m_api->endCapture();

		if (!rendered.semaphore)
		{
			m_exitRequested = true;
			return;
		}

		if (session->getProgress()>=1.f)
		{
			if (!finalizeCompletedSession(session,rendered))
			{
				m_exitRequested = true;
				return;
			}
		}

		if (m_args.headless)
			return;

		handleInputs();
		if (!keepRunning())
			return;

		m_presenter->acquire(session);
		auto* const cbPresent = m_presenter->beginRenderpass();
		(void)cbPresent;
		m_presenter->endRenderpassAndPresent(rendered);
	}

	void handleInputs()
	{
		if (m_args.headless)
			return;

		m_inputSystem->getDefaultMouse(&m_mouse);
		m_inputSystem->getDefaultKeyboard(&m_keyboard);

		struct
		{
			std::vector<SMouseEvent> mouse{};
			std::vector<SKeyboardEvent> keyboard{};
		} capturedEvents;

		static std::chrono::microseconds previousEventTimestamp{};
		m_mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
			{
				for (const auto& e : events)
				{
					if (e.timeStamp < previousEventTimestamp)
						continue;

					previousEventTimestamp = e.timeStamp;
					capturedEvents.mouse.emplace_back(e);
				}
			}, m_logger.get()
		);
		m_keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
			{
				for (const auto& e : events)
				{
					if (e.timeStamp < previousEventTimestamp)
						continue;

					previousEventTimestamp = e.timeStamp;
					capturedEvents.keyboard.emplace_back(e);
				}
			}, m_logger.get()
		);
	}

	bool keepRunning() override
	{
		if (m_exitRequested)
			return false;

		if (shouldExitAfterQueue())
		{
			const auto* const currentSession = m_resolver ? m_resolver->getActiveSession() : nullptr;
			if (m_sessionQueue.empty() && (!currentSession || currentSession==m_lastFinalizedSession) && m_nextSceneJob>=m_sceneJobs.size())
				return false;
			return true;
		}

		return m_presenter && !m_presenter->irrecoverable();
	}

	bool onAppTerminated() override
	{
		const bool postProcessOk = flushDeferredPostProcessJobs();
		const bool reportOk = m_report ? m_report->write() : true;
		const bool reportPassed = m_report ? !m_report->hasFailures() : true;
		if (!postProcessOk || !reportOk || !reportPassed)
			return false;
		return device_base_t::onAppTerminated();
	}

public:
	inline PathTracingApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

private:
	AppArguments m_args = {};
	bool m_exitRequested = false;
	std::vector<SceneJob> m_sceneJobs = {};
	size_t m_nextSceneJob = 0u;
	SceneJob m_activeSceneJob = {};
	size_t m_activeSceneJobIndex = 0u;
	const CSession* m_lastFinalizedSession = nullptr;
	std::string m_lastSceneLoadError;
	std::deque<PostProcessJob> m_deferredPostProcessJobs = {};
	std::string m_buildInfoJson = {};
	std::unique_ptr<PathTracerReport> m_report = {};

	smart_refctd_ptr<InputSystem> m_inputSystem;
	InputSystem::ChannelReader<IMouseEventChannel> m_mouse;
	InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;

	smart_refctd_ptr<CWindowPresenter> m_presenter;
	smart_refctd_ptr<CRenderer> m_renderer;
	smart_refctd_ptr<CBasicRWMCResolver> m_resolver;
	smart_refctd_ptr<CSceneLoader> m_sceneLoader;
	smart_refctd_ptr<CScene> m_scene;
	nbl::core::queue<smart_refctd_ptr<CSession>> m_sessionQueue;
};

namespace nbl::this_example
{

int runPathTracingApp(int argc, char** argv)
{
	return ::PathTracingApp::main<::PathTracingApp>(argc,argv);
}

}
