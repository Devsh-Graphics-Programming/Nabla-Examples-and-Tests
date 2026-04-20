// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "argparse/argparse.hpp"
#include "nbl/examples/common/BuiltinResourcesApplication.hpp"
#include "nbl/examples/examples.hpp"
#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "io/RuntimeConfig.h"
#include "renderer/CRenderer.h"
#include "renderer/resolve/CBasicRWMCResolver.h"
#include "renderer/present/CWindowPresenter.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <deque>
#include <execution>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string_view>
#include <system_error>
#include <vector>


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

namespace
{

enum class SensorWorkflow : uint8_t
{
	RenderAllThenInteractive,
	RenderAllThenTerminate,
	RenderSensorThenInteractive,
	InteractiveAtSensor
};

struct AppArguments
{
	path scenePath = {};
	std::optional<path> sceneEntry = std::nullopt;
	SensorWorkflow workflow = SensorWorkflow::RenderAllThenInteractive;
	uint32_t sensor = 0u;
	bool headless = false;
	bool deferPostProcess = false;
	path outputDir = {};
};

struct ExportOutputs
{
	path tonemap = {};
	path tonemapPng = {};
	path rwmcCascades = {};
	path rwmcCascadesPng = {};
	path albedo = {};
	path albedoPng = {};
	path normal = {};
	path normalPng = {};
	path denoised = {};
	path denoisedPng = {};

	std::string tonemapDisplay = {};
	std::string tonemapPngDisplay = {};
	std::string rwmcCascadesDisplay = {};
	std::string rwmcCascadesPngDisplay = {};
	std::string albedoDisplay = {};
	std::string albedoPngDisplay = {};
	std::string normalDisplay = {};
	std::string normalPngDisplay = {};
	std::string denoisedDisplay = {};
	std::string denoisedPngDisplay = {};
};

struct PostProcessJob
{
	ExportOutputs outputs = {};
	CSceneLoader::SLoadResult::SSensor::SDynamic::SPostProcess postProcess = {};
};

std::string trim(std::string_view input)
{
	size_t begin = 0u;
	while (begin<input.size() && std::isspace(static_cast<unsigned char>(input[begin])))
		++begin;

	size_t end = input.size();
	while (end>begin && std::isspace(static_cast<unsigned char>(input[end-1u])))
		--end;

	return std::string(input.substr(begin,end-begin));
}

std::string stripWrappingQuotes(std::string_view input)
{
	auto stripped = trim(input);
	if (stripped.size()>=2u)
	{
		const char first = stripped.front();
		const char last = stripped.back();
		if ((first=='\"' && last=='\"') || (first=='\'' && last=='\''))
			stripped = stripped.substr(1u,stripped.size()-2u);
	}
	return stripped;
}

std::string toLowerCopy(std::string_view input)
{
	std::string lowered(input);
	for (char& c : lowered)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return lowered;
}

bool isOptionToken(const std::string& token)
{
	return !token.empty() && token.front()=='-';
}

bool hasZipExtension(const path& filePath)
{
	return toLowerCopy(filePath.extension().string())==".zip";
}

std::optional<SensorWorkflow> parseSensorWorkflow(std::string_view rawValue)
{
	const auto lowered = toLowerCopy(rawValue);
	if (lowered=="renderalltheninteractive" || lowered=="render-all-then-interactive" || lowered=="all-interactive")
		return SensorWorkflow::RenderAllThenInteractive;
	if (lowered=="renderallthenterminate" || lowered=="render-all-then-terminate" || lowered=="all-terminate")
		return SensorWorkflow::RenderAllThenTerminate;
	if (lowered=="rendersensortheninteractive" || lowered=="render-sensor-then-interactive" || lowered=="sensor-interactive")
		return SensorWorkflow::RenderSensorThenInteractive;
	if (lowered=="interactiveatsensor" || lowered=="interactive-at-sensor" || lowered=="interactive")
		return SensorWorkflow::InteractiveAtSensor;
	return std::nullopt;
}

path appendBeforeExtension(path input, std::string_view suffix)
{
	auto stem = input.stem().string();
	stem += suffix;
	input.replace_filename(stem + input.extension().string());
	return input;
}

path replaceExtension(path input, const path& extension)
{
	input.replace_extension(extension);
	return input;
}

std::string makeGenericPathString(const path& input)
{
	return input.generic_string();
}

std::vector<std::string> normalizeCompatibleArguments(const nbl::core::vector<std::string>& rawArguments)
{
	if (rawArguments.empty())
		return {};

	std::vector<std::string> normalized;
	normalized.reserve(rawArguments.size()*2u);
	normalized.push_back(rawArguments.front());

	for (size_t i = 1u; i < rawArguments.size(); ++i)
	{
		const auto& token = rawArguments[i];

		if (token.rfind("-SCENE=",0u)==0u)
		{
			auto sceneValue = stripWrappingQuotes(token.substr(7u));
			std::optional<std::string> sceneEntry = std::nullopt;
			if (const auto lowered = toLowerCopy(sceneValue); lowered.find(".zip")!=std::string::npos)
			{
				const auto zipPos = lowered.find(".zip");
				const size_t entryPos = sceneValue.find_first_not_of(" \t",zipPos+4u);
				if (entryPos!=std::string::npos)
				{
					sceneEntry = trim(sceneValue.substr(entryPos));
					sceneValue = trim(sceneValue.substr(0u,zipPos+4u));
				}
				else if (i+1u<rawArguments.size() && !isOptionToken(rawArguments[i+1u]))
				{
					sceneEntry = stripWrappingQuotes(rawArguments[++i]);
				}
			}

			normalized.push_back("--scene");
			normalized.push_back(sceneValue);
			if (sceneEntry.has_value())
			{
				normalized.push_back("--scene-entry");
				normalized.push_back(sceneEntry.value());
			}
			continue;
		}

		if (token=="-PROCESS_SENSORS" || token.rfind("-PROCESS_SENSORS=",0u)==0u)
		{
			std::string workflow;
			if (token=="-PROCESS_SENSORS")
			{
				if (i+1u>=rawArguments.size())
					continue;
				workflow = stripWrappingQuotes(rawArguments[++i]);
			}
			else
			{
				workflow = stripWrappingQuotes(token.substr(std::char_traits<char>::length("-PROCESS_SENSORS=")));
			}

			normalized.push_back("--process-sensors");
			normalized.push_back(workflow);
			if (i+1u<rawArguments.size() && !isOptionToken(rawArguments[i+1u]))
			{
				normalized.push_back("--sensor");
				normalized.push_back(stripWrappingQuotes(rawArguments[++i]));
			}
			continue;
		}

		if (token=="-DEFER_DENOISE")
		{
			normalized.push_back("--defer-denoise");
			continue;
		}

		if (token=="-TERMINATE")
		{
			normalized.push_back("--process-sensors");
			normalized.push_back("RenderAllThenTerminate");
			continue;
		}

		normalized.push_back(token);
	}

	return normalized;
}

}

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

	void printGitInfos() const
	{
		nlohmann_json j;

		auto& modules = j["modules"];
		jsonizeGitInfo(modules["nabla"],nbl::gtml::nabla_git_info);
		jsonizeGitInfo(modules["dxc"],nbl::gtml::dxc_git_info);

		m_logger->log("Build Info:\n%s",ILogger::ELL_INFO,j.dump(4).c_str());
	}

	path resolvePathAgainstCurrentWorkingDirectory(const path& candidate) const
	{
		if (candidate.empty() || candidate.is_absolute())
			return candidate;
		return (std::filesystem::current_path()/candidate).lexically_normal();
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

		try
		{
			parser.parse_args(normalizedArguments);
		}
		catch (const std::exception& e)
		{
			std::fprintf(stderr,"Failed to parse arguments: %s\n", e.what());
			return false;
		}

		const auto sceneValue = parser.present("--scene");
		if (!sceneValue.has_value())
		{
			std::fprintf(stderr,"Scene path is required. Use --scene or -SCENE=...\n");
			return false;
		}

		m_args.scenePath = resolvePathAgainstCurrentWorkingDirectory(stripWrappingQuotes(sceneValue.value()));
		if (const auto sceneEntry = parser.present("--scene-entry"); sceneEntry.has_value())
			m_args.sceneEntry = path(stripWrappingQuotes(sceneEntry.value()));

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

	bool shouldExitAfterQueue() const
	{
		return m_args.headless || m_args.workflow==SensorWorkflow::RenderAllThenTerminate;
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

		printGitInfos();

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

		return initializeSceneAndQueueSessions();
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
			return logFail("Could not create scene");

		const auto sensors = m_scene->getSensors();
		if (sensors.empty())
			return logFail("Loaded scene does not expose any sensors.");

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
						return logFail("Failed to queue render sessions");
				}
				break;
			case SensorWorkflow::RenderSensorThenInteractive:
			case SensorWorkflow::InteractiveAtSensor:
				if (!enqueueSensor(sensorIndex))
					return logFail("Failed to queue render sessions");
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

		ExportOutputs outputs;
		outputs.tonemap = outputFile;
		outputs.tonemapPng = replaceExtension(outputFile,".png");
		outputs.rwmcCascades = appendBeforeExtension(outputFile,"_rwmc_cascades");
		outputs.rwmcCascadesPng = replaceExtension(outputs.rwmcCascades,".png");
		outputs.albedo = appendBeforeExtension(outputFile,"_albedo");
		outputs.albedoPng = replaceExtension(outputs.albedo,".png");
		outputs.normal = appendBeforeExtension(outputFile,"_normal");
		outputs.normalPng = replaceExtension(outputs.normal,".png");
		outputs.denoised = appendBeforeExtension(outputFile,"_denoised");
		outputs.denoisedPng = replaceExtension(outputs.denoised,".png");

		outputs.tonemapDisplay = makeDisplayPath(outputs.tonemap);
		outputs.tonemapPngDisplay = makeDisplayPath(outputs.tonemapPng);
		outputs.rwmcCascadesDisplay = makeDisplayPath(outputs.rwmcCascades);
		outputs.rwmcCascadesPngDisplay = makeDisplayPath(outputs.rwmcCascadesPng);
		outputs.albedoDisplay = makeDisplayPath(outputs.albedo);
		outputs.albedoPngDisplay = makeDisplayPath(outputs.albedoPng);
		outputs.normalDisplay = makeDisplayPath(outputs.normal);
		outputs.normalPngDisplay = makeDisplayPath(outputs.normalPng);
		outputs.denoisedDisplay = makeDisplayPath(outputs.denoised);
		outputs.denoisedPngDisplay = makeDisplayPath(outputs.denoisedPng);
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

	smart_refctd_ptr<ICPUImage> createPngPreviewImage(const ICPUImageView* sourceView) const
	{
		if (!sourceView)
			return nullptr;

		constexpr auto OutputFormat = E_FORMAT::EF_R8G8B8A8_SRGB;
		const auto& sourceViewParams = sourceView->getCreationParameters();
		auto sourceImage = sourceViewParams.image;
		if (!sourceImage)
			return nullptr;

		const auto& sourceSubresource = sourceViewParams.subresourceRange;
		const auto sourceMipLevel = sourceSubresource.baseMipLevel;
		const auto sourceBaseLayer = sourceSubresource.baseArrayLayer;
		const auto sourceExtent = sourceImage->getMipSize(sourceMipLevel);
		const uint32_t width = sourceExtent.x;
		const uint32_t height = sourceExtent.y;
		if (width==0u || height==0u)
			return nullptr;

		IImage::SCreationParams imageParams = {};
		imageParams.type = IImage::ET_2D;
		imageParams.format = OutputFormat;
		imageParams.extent = {width,height,1u};
		imageParams.mipLevels = 1u;
		imageParams.arrayLayers = 1u;
		imageParams.samples = IImage::ESCF_1_BIT;
		imageParams.usage = IImage::EUF_SAMPLED_BIT;
		auto outputImage = ICPUImage::create(std::move(imageParams));
		if (!outputImage)
			return nullptr;

		auto regions = make_refctd_dynamic_array<smart_refctd_dynamic_array<IImage::SBufferCopy>>(1u);
		auto& region = regions->front();
		region.bufferOffset = 0u;
		region.bufferRowLength = width;
		region.bufferImageHeight = height;
		region.imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
		region.imageSubresource.mipLevel = 0u;
		region.imageSubresource.baseArrayLayer = 0u;
		region.imageSubresource.layerCount = 1u;
		region.imageExtent = {width,height,1u};
		region.imageOffset = {0u,0u,0u};

		const auto outputPixelSize = getTexelOrBlockBytesize(OutputFormat);
		const uint64_t outputSize = uint64_t(width)*uint64_t(height)*outputPixelSize;
		auto outputBuffer = ICPUBuffer::create({outputSize});
		if (!outputBuffer)
			return nullptr;
		outputImage->setBufferAndRegions(std::move(outputBuffer),std::move(regions));
		using convert_filter_t = CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,DefaultSwizzle,IdentityDither,void,true>;
		convert_filter_t::state_type state = {};
		static_cast<DefaultSwizzle&>(state).swizzle = {
			ICPUImageView::SComponentMapping::ES_R,
			ICPUImageView::SComponentMapping::ES_G,
			ICPUImageView::SComponentMapping::ES_B,
			ICPUImageView::SComponentMapping::ES_ONE
		};
		state.inImage = sourceImage.get();
		state.outImage = outputImage.get();
		state.inOffset = {0,0,0};
		state.inBaseLayer = sourceBaseLayer;
		state.outOffset = {0,0,0};
		state.outBaseLayer = 0u;
		state.extent = {width,height,1u};
		state.layerCount = 1u;
		state.inMipLevel = sourceMipLevel;
		state.outMipLevel = 0u;
		if (!convert_filter_t::execute(nbl::core::execution::seq,&state))
		{
			m_logger->log("Failed to convert readback image to PNG preview format",ILogger::ELL_ERROR);
			return nullptr;
		}
		return outputImage;
	}

	bool exportImageView(const IGPUImageView* view, const path& exrDestination, const path& pngDestination) const
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
		if (!writeImageView(cpuImageView.get(),exrDestination))
			return false;

		auto pngImage = createPngPreviewImage(cpuImageView.get());
		auto pngImageView = makeImageView(std::move(pngImage));
		return writeImageView(pngImageView.get(),pngDestination);
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
		if (!exportImageView(tonemapView,outputs.tonemap,outputs.tonemapPng))
			return false;

		const auto* const rwmcCascadesView = immutables.rwmcCascades.getView(E_FORMAT::EF_R16G16B16A16_SFLOAT);
		if (!rwmcCascadesView)
		{
			m_logger->log("Missing image view for export format %s",ILogger::ELL_ERROR,to_string(E_FORMAT::EF_R16G16B16A16_SFLOAT).c_str());
			return false;
		}
		if (!exportImageView(rwmcCascadesView,outputs.rwmcCascades,outputs.rwmcCascadesPng))
			return false;

		const auto* const albedoView = immutables.albedo.getView(E_FORMAT::EF_R16G16B16A16_SFLOAT);
		if (!albedoView)
		{
			m_logger->log("Missing image view for export format %s",ILogger::ELL_ERROR,to_string(E_FORMAT::EF_R16G16B16A16_SFLOAT).c_str());
			return false;
		}
		if (!exportImageView(albedoView,outputs.albedo,outputs.albedoPng))
			return false;

		const auto* const normalView = immutables.normal.getView(E_FORMAT::EF_R16G16B16A16_SFLOAT);
		if (!normalView)
		{
			m_logger->log("Missing image view for export format %s",ILogger::ELL_ERROR,to_string(E_FORMAT::EF_R16G16B16A16_SFLOAT).c_str());
			return false;
		}
		if (!exportImageView(normalView,outputs.normal,outputs.normalPng))
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
		return copyOutputFile(job.outputs.tonemap,job.outputs.denoised) &&
			copyOutputFile(job.outputs.tonemapPng,job.outputs.denoisedPng);
	}

	void emitOutputJson(const ExportOutputs& outputs) const
	{
		nlohmann_json payload = {
			{"output_tonemap",outputs.tonemapDisplay},
			{"output_tonemap_png",outputs.tonemapPngDisplay},
			{"output_rwmc_cascades",outputs.rwmcCascadesDisplay},
			{"output_rwmc_cascades_png",outputs.rwmcCascadesPngDisplay},
			{"output_albedo",outputs.albedoDisplay},
			{"output_albedo_png",outputs.albedoPngDisplay},
			{"output_normal",outputs.normalDisplay},
			{"output_normal_png",outputs.normalPngDisplay},
			{"output_denoised",outputs.denoisedDisplay},
			{"output_denoised_png",outputs.denoisedPngDisplay}
		};
		std::cout << "[JSON] " << payload.dump() << "\n[ENDJSON]\n";
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
			retval.swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;
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
			if (m_sessionQueue.empty() && (!currentSession || currentSession==m_lastFinalizedSession))
				return false;
			return true;
		}

		return m_presenter && !m_presenter->irrecoverable();
	}

	bool onAppTerminated() override
	{
		if (!flushDeferredPostProcessJobs())
			return false;
		return device_base_t::onAppTerminated();
	}

public:
	inline PathTracingApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

private:
	AppArguments m_args = {};
	bool m_exitRequested = false;
	const CSession* m_lastFinalizedSession = nullptr;
	std::deque<PostProcessJob> m_deferredPostProcessJobs = {};

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

NBL_MAIN_FUNC(PathTracingApp)
