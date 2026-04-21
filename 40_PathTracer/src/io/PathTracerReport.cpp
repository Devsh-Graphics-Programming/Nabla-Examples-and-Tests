// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "io/PathTracerReport.h"

#include "nbl/asset/asset.h"
#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"
#include "nbl/core/hash/blake.h"
#include "nbl/system/ILogger.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <execution>
#include <optional>
#include <sstream>
#include <string_view>
#include <system_error>
#include <vector>

namespace nbl::this_example
{
namespace
{
using namespace nbl::asset;
using namespace nbl::core;
using nlohmann_json = nlohmann::json;

std::string makeGenericPathString(const system::path& input)
{
	return input.generic_string();
}

std::string nowString()
{
	const auto now = std::chrono::system_clock::now();
	const auto time = std::chrono::system_clock::to_time_t(now);
	std::tm localTm = {};
	std::tm utcTm = {};
#ifdef _WIN32
	localtime_s(&localTm,&time);
	gmtime_s(&utcTm,&time);
#else
	localtime_r(&time,&localTm);
	gmtime_r(&time,&utcTm);
#endif
	auto localEpochTm = localTm;
	auto utcEpochTm = utcTm;
	utcEpochTm.tm_isdst = localEpochTm.tm_isdst;
	const auto localEpoch = std::mktime(&localEpochTm);
	const auto utcEpoch = std::mktime(&utcEpochTm);
	const auto offsetSeconds = static_cast<long>(std::difftime(localEpoch,utcEpoch));
	const auto absOffsetSeconds = offsetSeconds<0 ? -offsetSeconds:offsetSeconds;
	const auto offsetHours = absOffsetSeconds/3600;
	const auto offsetMinutes = (absOffsetSeconds%3600)/60;
	std::ostringstream out;
	out << std::put_time(&localTm,"%Y-%m-%d %H:%M:%S")
		<< ' ' << (offsetSeconds<0 ? '-':'+')
		<< std::setw(2) << std::setfill('0') << offsetHours
		<< std::setw(2) << std::setfill('0') << offsetMinutes;
	return out.str();
}

std::string statusColor(const std::string& status)
{
	if (status=="passed")
		return "green";
	if (status=="failed")
		return "red";
	if (status=="missing-reference" || status=="missing-render")
		return "orange";
	if (status=="error")
		return "red";
	return "gray";
}

bool ensureParentDirectoryExists(const system::path& filePath)
{
	const auto parent = filePath.parent_path();
	if (parent.empty())
		return true;
	std::error_code ec;
	std::filesystem::create_directories(parent,ec);
	return !ec;
}

system::path relativePath(const system::path& path, const system::path& base)
{
	std::error_code ec;
	auto rel = std::filesystem::relative(path,base,ec);
	if (!ec && !rel.empty())
		return rel;
	return path;
}

bool startsWithParentTraversal(const system::path& path)
{
	auto it = path.begin();
	return it!=path.end() && it->string()=="..";
}

system::path portablePath(const system::path& input, const system::path& reportDir, const system::path& workingDirectory)
{
	if (input.empty())
		return {};
	if (input.is_relative())
		return input.lexically_normal();

	const auto normalized = input.lexically_normal();
	const auto reportRelative = relativePath(normalized,reportDir);
	if (!reportRelative.empty() && !reportRelative.is_absolute() && !startsWithParentTraversal(reportRelative))
		return reportRelative;

	const auto workingRelative = relativePath(normalized,workingDirectory);
	if (!workingRelative.empty() && !workingRelative.is_absolute())
		return workingRelative;

	return normalized.filename();
}

std::string portablePathString(const system::path& input, const system::path& reportDir, const system::path& workingDirectory)
{
	return portablePath(input,reportDir,workingDirectory).generic_string();
}

void replaceAll(std::string& text, const std::string& from, const std::string& to)
{
	if (from.empty())
		return;

	size_t pos = 0u;
	while ((pos=text.find(from,pos))!=std::string::npos)
	{
		text.replace(pos,from.size(),to);
		pos += to.size();
	}
}

std::string portableCommandLineString(std::string commandLine, const system::path& reportDir, const system::path& workingDirectory)
{
	const auto portableReportDir = portablePathString(reportDir,reportDir,workingDirectory);
	const auto portableWorkingDirectory = portablePathString(workingDirectory,reportDir,workingDirectory);

	replaceAll(commandLine,reportDir.string(),portableReportDir);
	replaceAll(commandLine,reportDir.generic_string(),portableReportDir);
	replaceAll(commandLine,workingDirectory.string(),portableWorkingDirectory);
	replaceAll(commandLine,workingDirectory.generic_string(),portableWorkingDirectory);
	return commandLine;
}

std::string toLowerCopy(std::string_view input)
{
	std::string lowered(input);
	for (char& c : lowered)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return lowered;
}

bool isDenoisedOutput(const std::string& identifier)
{
	return toLowerCopy(identifier)=="denoised";
}

bool copyFile(const system::path& source, const system::path& destination)
{
	if (!ensureParentDirectoryExists(destination))
		return false;
	std::error_code ec;
	std::filesystem::copy_file(source,destination,std::filesystem::copy_options::overwrite_existing,ec);
	return !ec;
}

std::string hexString(const uint8_t* data, const size_t size)
{
	std::ostringstream out;
	out << std::hex << std::setfill('0');
	for (size_t i=0u; i<size; ++i)
		out << std::setw(2) << uint32_t(data[i]);
	return out.str();
}

std::optional<std::string> blake3FileHash(const system::path& path)
{
	std::ifstream file(path,std::ios::binary);
	if (!file)
		return std::nullopt;

	core::blake3_hasher hasher;
	std::vector<char> buffer(1u<<20);
	while (file)
	{
		file.read(buffer.data(),buffer.size());
		const auto readBytes = file.gcount();
		if (readBytes>0)
			hasher.update(buffer.data(),static_cast<size_t>(readBytes));
	}

	const auto hash = static_cast<core::blake3_hash_t>(hasher);
	return hexString(hash.data,sizeof(hash.data));
}

std::vector<std::string> referenceNamesWithFallbacks(const std::string& sceneName, const std::vector<std::string>& referenceNames)
{
	std::vector<std::string> names;
	for (const auto& name : referenceNames)
	{
		if (!name.empty() && std::find(names.begin(),names.end(),name)==names.end())
			names.push_back(name);
	}
	if (!sceneName.empty() && std::find(names.begin(),names.end(),sceneName)==names.end())
		names.push_back(sceneName);
	return names;
}

bool endsWith(std::string_view value, std::string_view suffix)
{
	return value.size()>=suffix.size() && value.substr(value.size()-suffix.size())==suffix;
}

void appendUniquePath(std::vector<system::path>& paths, system::path path)
{
	if (!path.empty() && std::find(paths.begin(),paths.end(),path)==paths.end())
		paths.push_back(std::move(path));
}

std::vector<system::path> referenceFilenameFallbacks(const std::string& imageIdentifier, const system::path& renderPath)
{
	std::vector<system::path> filenames;
	const auto filename = renderPath.filename();
	appendUniquePath(filenames,filename);

	const auto filenameString = filename.string();
	if (!filenameString.empty() && filenameString.rfind("Render_",0u)!=0u)
		appendUniquePath(filenames,system::path("Render_"+filenameString));

	if (isDenoisedOutput(imageIdentifier))
	{
		const auto stem = filename.stem().string();
		const auto extension = filename.extension().string();
		const std::string suffix = "_denoised";
		if (endsWith(stem,suffix))
		{
			const auto tonemapStem = stem.substr(0u,stem.size()-suffix.size());
			const auto tonemapFilename = tonemapStem+extension;
			appendUniquePath(filenames,system::path(tonemapFilename));
			if (tonemapFilename.rfind("Render_",0u)!=0u)
				appendUniquePath(filenames,system::path("Render_"+tonemapFilename));
		}
	}

	return filenames;
}

system::path firstExistingReferencePath(const system::path& referenceDir, const std::string& sceneName, const std::vector<std::string>& referenceNames, const std::string& imageIdentifier, const system::path& renderPath)
{
	if (referenceDir.empty())
		return {};

	std::vector<system::path> candidates;
	const auto filenames = referenceFilenameFallbacks(imageIdentifier,renderPath);
	for (const auto& filename : filenames)
	{
		for (const auto& name : referenceNamesWithFallbacks(sceneName,referenceNames))
			appendUniquePath(candidates,referenceDir/name/filename);
		appendUniquePath(candidates,referenceDir/filename);
	}

	for (const auto& candidate : candidates)
	{
		if (std::filesystem::exists(candidate))
			return candidate;
	}
	return candidates.empty() ? system::path{}:candidates.front();
}

smart_refctd_ptr<ICPUImageView> makeImageView(smart_refctd_ptr<ICPUImage>&& image)
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

smart_refctd_ptr<ICPUImage> convertImageView(const ICPUImageView* sourceView, const E_FORMAT outputFormat)
{
	if (!sourceView)
		return nullptr;

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
	imageParams.format = outputFormat;
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

	const auto outputPixelSize = getTexelOrBlockBytesize(outputFormat);
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
		return nullptr;
	return outputImage;
}

bool writeImageView(IAssetManager* assetManager, system::ILogger* logger, const ICPUImageView* view, const system::path& destination)
{
	if (!assetManager || !view || !ensureParentDirectoryExists(destination))
		return false;

	IAssetWriter::SAssetWriteParams writeParams(const_cast<ICPUImageView*>(view),EWF_NONE,0.f,0u,nullptr,nullptr,system::logger_opt_ptr(logger));
	if (!assetManager->writeAsset(destination.string(),writeParams))
	{
		if (logger)
			logger->log("Failed to write \"%s\"",system::ILogger::ELL_ERROR,destination.string().c_str());
		return false;
	}
	return true;
}

smart_refctd_ptr<ICPUImage> createFloatDiffImage(const ICPUImage* renderImage, const ICPUImage* referenceImage, uint64_t& outErrorPixels, double& outMaxAbsError, const PathTracerReport::SCompareSettings& settings)
{
	if (!renderImage || !referenceImage)
		return nullptr;

	const auto& renderParams = renderImage->getCreationParameters();
	const auto& referenceParams = referenceImage->getCreationParameters();
	if (renderParams.extent.width!=referenceParams.extent.width || renderParams.extent.height!=referenceParams.extent.height)
		return nullptr;

	const auto width = renderParams.extent.width;
	const auto height = renderParams.extent.height;
	const auto pixelCount = uint64_t(width)*uint64_t(height);

	IImage::SCreationParams imageParams = {};
	imageParams.type = IImage::ET_2D;
	imageParams.format = E_FORMAT::EF_R32G32B32A32_SFLOAT;
	imageParams.extent = {width,height,1u};
	imageParams.mipLevels = 1u;
	imageParams.arrayLayers = 1u;
	imageParams.samples = IImage::ESCF_1_BIT;
	imageParams.usage = IImage::EUF_SAMPLED_BIT;
	auto diffImage = ICPUImage::create(std::move(imageParams));
	if (!diffImage)
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

	const uint64_t outputSize = pixelCount*4u*sizeof(float);
	auto outputBuffer = ICPUBuffer::create({outputSize});
	if (!outputBuffer)
		return nullptr;
	diffImage->setBufferAndRegions(std::move(outputBuffer),std::move(regions));

	const auto* render = reinterpret_cast<const float*>(renderImage->getBuffer()->getPointer());
	const auto* reference = reinterpret_cast<const float*>(referenceImage->getBuffer()->getPointer());
	auto* diff = reinterpret_cast<float*>(diffImage->getBuffer()->getPointer());

	outErrorPixels = 0u;
	outMaxAbsError = 0.0;
	for (uint64_t pixel=0u; pixel<pixelCount; ++pixel)
	{
		bool pixelFailed = false;
		for (uint32_t channel=0u; channel<3u; ++channel)
		{
			const auto index = pixel*4u+channel;
			const double renderValue = render[index];
			const double referenceValue = reference[index];
			const double absError = std::abs(renderValue-referenceValue);
			const double minValue = std::min(std::abs(renderValue),std::abs(referenceValue));
			const double maxValue = std::max(std::abs(renderValue),std::abs(referenceValue));
			const bool channelFailed = minValue>settings.epsilon ? (absError/minValue)>settings.errorThreshold : maxValue>settings.epsilon;
			pixelFailed = pixelFailed || channelFailed;
			outMaxAbsError = std::max(outMaxAbsError,absError);
			diff[index] = static_cast<float>(absError);
		}
		diff[pixel*4u+3u] = 1.f;
		if (pixelFailed)
			++outErrorPixels;
	}
	return diffImage;
}

double computeSsimDifference(const ICPUImage* renderImage, const ICPUImage* referenceImage)
{
	if (!renderImage || !referenceImage)
		return 1.0;

	const auto& renderParams = renderImage->getCreationParameters();
	const auto& referenceParams = referenceImage->getCreationParameters();
	if (renderParams.extent.width!=referenceParams.extent.width || renderParams.extent.height!=referenceParams.extent.height)
		return 1.0;

	const auto pixelCount = uint64_t(renderParams.extent.width)*uint64_t(renderParams.extent.height);
	if (pixelCount==0u)
		return 0.0;

	const auto* render = reinterpret_cast<const float*>(renderImage->getBuffer()->getPointer());
	const auto* reference = reinterpret_cast<const float*>(referenceImage->getBuffer()->getPointer());

	constexpr double C1 = 0.01*0.01;
	constexpr double C2 = 0.03*0.03;
	double ssimSum = 0.0;
	for (uint32_t channel=0u; channel<3u; ++channel)
	{
		double sumX = 0.0;
		double sumY = 0.0;
		double sumXX = 0.0;
		double sumYY = 0.0;
		double sumXY = 0.0;
		for (uint64_t pixel=0u; pixel<pixelCount; ++pixel)
		{
			const auto index = pixel*4u+channel;
			const double x = render[index];
			const double y = reference[index];
			sumX += x;
			sumY += y;
			sumXX += x*x;
			sumYY += y*y;
			sumXY += x*y;
		}

		const double invCount = 1.0/double(pixelCount);
		const double meanX = sumX*invCount;
		const double meanY = sumY*invCount;
		const double varianceX = std::max(0.0,sumXX*invCount-meanX*meanX);
		const double varianceY = std::max(0.0,sumYY*invCount-meanY*meanY);
		const double covariance = sumXY*invCount-meanX*meanY;
		const double numerator = (2.0*meanX*meanY+C1)*(2.0*covariance+C2);
		const double denominator = (meanX*meanX+meanY*meanY+C1)*(varianceX+varianceY+C2);
		const double ssim = denominator!=0.0 ? std::clamp(numerator/denominator,0.0,1.0):1.0;
		ssimSum += ssim;
	}

	return 1.0-ssimSum/3.0;
}

smart_refctd_ptr<ICPUImageView> loadImageView(IAssetManager* assetManager, system::ILogger* logger, const system::path& path)
{
	if (!assetManager)
		return nullptr;

	IAssetLoader::SAssetLoadParams params;
	params.logger = logger;
	params.workingDirectory = path.parent_path();
	const auto key = path.filename().string();
	auto bundle = assetManager->getAsset(key,params);
	if (bundle.getContents().empty())
		return nullptr;

	if (auto view = IAsset::castDown<ICPUImageView>(bundle.getContents()[0]))
		return view;
	if (auto image = IAsset::castDown<ICPUImage>(bundle.getContents()[0]))
		return makeImageView(std::move(image));
	return nullptr;
}

struct SImageResult
{
	std::string identifier;
	std::string title;
	bool requiresReference = true;
	std::string status = "not-checked";
	std::string details = "No reference directory configured";
	system::path renderExr;
	system::path referenceExr;
	system::path diffExr;
	uint64_t errorPixels = 0u;
	uint64_t allowedErrorPixels = 0u;
	uint64_t totalPixels = 0u;
	double maxAbsError = 0.0;
	bool hasSsimDifference = false;
	double ssimDifference = 0.0;
};

struct SSessionResult
{
	std::string sceneName;
	std::string displayName;
	std::vector<std::string> referenceNames;
	system::path scenePath;
	uint32_t sensorIndex = 0u;
	PathTracerReport::SCompareSettings compare;
	std::string status = "not-checked";
	std::string details;
	std::vector<SImageResult> images;
};

std::string allowedErrorPixelModeName(const PathTracerReport::SCompareSettings::EAllowedErrorPixelMode mode)
{
	using mode_t = PathTracerReport::SCompareSettings::EAllowedErrorPixelMode;
	return mode==mode_t::AbsoluteCount ? "absolute-count":"relative-to-resolution";
}

}

struct PathTracerReport::Impl
{
	explicit Impl(PathTracerReport::SCreationParams&& params) : params(std::move(params))
	{
		if (!this->params.lowDiscrepancySequenceCachePath.empty())
			initialLdsCacheHash = blake3FileHash(this->params.lowDiscrepancySequenceCachePath);
	}

	bool compareImage(SSessionResult& session, SImageResult& image)
	{
		if (params.referenceDir.empty())
			return true;

		const auto renderExists = std::filesystem::exists(image.renderExr);
		const auto externalReferenceExr = firstExistingReferencePath(params.referenceDir,session.sceneName,session.referenceNames,image.identifier,image.renderExr);
		if (!std::filesystem::exists(externalReferenceExr))
		{
			if (image.requiresReference)
			{
				image.status = "missing-reference";
				image.details = renderExists ? "Reference file is missing":"Render and reference files are missing";
				hasFailures = true;
			}
			else
			{
				image.status = renderExists ? "not-checked":"missing-render";
				image.details = renderExists ? "No reference is expected for this diagnostic output":"Render file is missing and no reference is expected for this diagnostic output";
				hasFailures = hasFailures || !renderExists;
			}
			return true;
		}

		const auto referenceDir = params.reportDir/"references"/session.sceneName;
		image.referenceExr = referenceDir/externalReferenceExr.filename();
		if (!copyFile(externalReferenceExr,image.referenceExr))
		{
			image.status = "error";
			image.details = "Could not copy reference image into report";
			hasErrors = true;
			return false;
		}

		if (!renderExists)
		{
			image.status = "missing-render";
			image.details = "Render file is missing. Reference was copied into the report";
			hasFailures = true;
			return true;
		}

		auto renderView = loadImageView(params.assetManager,params.logger,image.renderExr);
		auto referenceView = loadImageView(params.assetManager,params.logger,externalReferenceExr);
		auto renderFloat = convertImageView(renderView.get(),E_FORMAT::EF_R32G32B32A32_SFLOAT);
		auto referenceFloat = convertImageView(referenceView.get(),E_FORMAT::EF_R32G32B32A32_SFLOAT);
		if (!renderFloat || !referenceFloat)
		{
			image.status = "error";
			image.details = "Could not load or convert image pair";
			hasErrors = true;
			return false;
		}

		const auto& renderParams = renderFloat->getCreationParameters();
		const auto& referenceParams = referenceFloat->getCreationParameters();
		if (renderParams.extent.width!=referenceParams.extent.width || renderParams.extent.height!=referenceParams.extent.height)
		{
			image.status = "failed";
			image.details = "Image dimensions differ";
			hasErrors = true;
			return true;
		}

		image.totalPixels = uint64_t(renderParams.extent.width)*uint64_t(renderParams.extent.height);
		const auto& compare = session.compare;
		using mode_t = PathTracerReport::SCompareSettings::EAllowedErrorPixelMode;
		image.allowedErrorPixels = compare.allowedErrorPixelMode==mode_t::AbsoluteCount ?
			compare.allowedErrorPixelCount:static_cast<uint64_t>(std::ceil(double(image.totalPixels)*compare.allowedErrorPixelRatio));

		uint64_t errorPixels = 0u;
		double maxAbsError = 0.0;
		auto diffImage = createFloatDiffImage(renderFloat.get(),referenceFloat.get(),errorPixels,maxAbsError,compare);
		if (!diffImage)
		{
			image.status = "error";
			image.details = "Could not create difference image";
			hasErrors = true;
			return false;
		}

		image.errorPixels = errorPixels;
		image.maxAbsError = maxAbsError;
		if (isDenoisedOutput(image.identifier))
		{
			image.hasSsimDifference = true;
			image.ssimDifference = computeSsimDifference(renderFloat.get(),referenceFloat.get());
			image.status = image.ssimDifference<=compare.ssimErrorThreshold ? "passed":"failed";
		}
		else
			image.status = errorPixels<=image.allowedErrorPixels ? "passed":"failed";
		hasFailures = hasFailures || image.status=="failed";

		const auto sceneDir = system::path("diff_images")/session.sceneName;
		const auto diffBaseName = image.renderExr.stem().string()+"_diff";
		image.diffExr = params.reportDir/sceneDir/(diffBaseName+".exr");
		auto diffView = makeImageView(smart_refctd_ptr<ICPUImage>(diffImage));
		if (!writeImageView(params.assetManager,params.logger,diffView.get(),image.diffExr))
		{
			image.status = "error";
			image.details = "Could not write difference image";
			hasErrors = true;
			return false;
		}

		std::ostringstream details;
		if (image.hasSsimDifference)
			details << "Difference (SSIM): " << std::fixed << std::setprecision(4) << image.ssimDifference;
		else
		{
			const auto errorPercent = image.totalPixels ? (100.0*double(image.errorPixels)/double(image.totalPixels)):0.0;
			details << "Errors: " << image.errorPixels << " (" << std::fixed << std::setprecision(3) << errorPercent << "%) / " << image.allowedErrorPixels;
			if (compare.allowedErrorPixelMode==mode_t::RelativeToResolution)
				details << " (" << std::fixed << std::setprecision(3) << 100.0*compare.allowedErrorPixelRatio << "%)";
		}
		image.details = details.str();
		return true;
	}

	bool addSession(PathTracerReport::SSession&& input)
	{
		SSessionResult result;
		result.sceneName = input.sceneName.empty() ? "scene":input.sceneName;
		result.displayName = input.displayName.empty() ? result.sceneName:input.displayName;
		result.referenceNames = std::move(input.referenceNames);
		result.scenePath = std::move(input.scenePath);
		result.sensorIndex = input.sensorIndex;
		result.compare = input.compare;
		result.status = input.status.empty() ? "not-checked":std::move(input.status);
		result.details = std::move(input.details);
		result.images.reserve(input.images.size());

		for (auto& artifact : input.images)
		{
			SImageResult image;
			image.identifier = std::move(artifact.identifier);
			image.title = std::move(artifact.title);
			image.requiresReference = artifact.requiresReference;
			image.renderExr = std::move(artifact.exrPath);
			result.images.push_back(std::move(image));
		}

		sessions.push_back(std::move(result));
		return true;
	}

	void compareSessions()
	{
		if (compared || params.referenceDir.empty())
		{
			compared = true;
			return;
		}

		for (auto& session : sessions)
		{
			if (session.status=="failed" || session.status=="error")
			{
				hasFailures = true;
				if (session.images.empty())
					continue;
			}
			else
				session.status = "passed";
			for (auto& image : session.images)
			{
				compareImage(session,image);
				if (image.status=="failed" || image.status=="error" || image.status=="missing-render")
				{
					session.status = "failed";
					hasFailures = true;
				}
				else if (image.status=="missing-reference" && session.status!="failed")
					session.status = "missing-reference";
			}
		}
		compared = true;
	}

	nlohmann_json makeLowDiscrepancySequenceCacheJson() const
	{
		nlohmann_json cache;
		if (params.lowDiscrepancySequenceCachePath.empty())
		{
			cache["status"] = "not-configured";
			return cache;
		}

		cache["path"] = portablePathString(params.lowDiscrepancySequenceCachePath,params.reportDir,params.workingDirectory);
		if (initialLdsCacheHash.has_value())
			cache["initialHash"] = initialLdsCacheHash.value();

		const auto finalHash = blake3FileHash(params.lowDiscrepancySequenceCachePath);
		if (!finalHash.has_value())
		{
			cache["status"] = "missing";
			return cache;
		}

		cache["hash"] = finalHash.value();
		std::error_code ec;
		const auto size = std::filesystem::file_size(params.lowDiscrepancySequenceCachePath,ec);
		if (!ec)
			cache["sizeBytes"] = size;

		if (!initialLdsCacheHash.has_value())
			cache["status"] = "created";
		else if (initialLdsCacheHash.value()==finalHash.value())
			cache["status"] = "did-not-change";
		else
			cache["status"] = "changed";
		return cache;
	}

	nlohmann_json makeSummaryJson() const
	{
		nlohmann_json summary;
		summary["identifier"] = "40_PathTracer";
		summary["datetime"] = generatedAt;
		summary["buildConfig"] = params.buildConfig.empty() ? "unknown" : params.buildConfig;
		summary["workingDirectory"] = portablePathString(params.workingDirectory,params.reportDir,params.workingDirectory);
		summary["reportDir"] = portablePathString(params.reportDir,params.reportDir,params.workingDirectory);
		summary["referenceDir"] = params.referenceDir.empty() ? "" : portablePathString(params.referenceDir,params.reportDir,params.workingDirectory);
		summary["commandLine"] = portableCommandLineString(params.commandLine,params.reportDir,params.workingDirectory);
		summary["compare"] = {
			{"errorThreshold",params.compare.errorThreshold},
			{"epsilon",params.compare.epsilon},
			{"allowedErrorPixelMode",allowedErrorPixelModeName(params.compare.allowedErrorPixelMode)},
			{"allowedErrorPixelRatio",params.compare.allowedErrorPixelRatio},
			{"allowedErrorPixelCount",params.compare.allowedErrorPixelCount},
			{"ssimErrorThreshold",params.compare.ssimErrorThreshold}
		};
		summary["postprocess"] = {
			{"denoiser",{
				{"mode","no-op-copy"},
				{"message","Denoiser integration is pending. output_denoised is currently a no-op copy of output_tonemap so the report and CI pipeline can be validated before real denoising lands."}
			}}
		};
		summary["lowDiscrepancySequenceCache"] = makeLowDiscrepancySequenceCacheJson();

		try
		{
			summary["build"] = nlohmann_json::parse(params.buildInfoJson);
		}
		catch (...)
		{
			summary["build"] = params.buildInfoJson;
		}
		if (!params.machineInfoJson.empty())
		{
			try
			{
				summary["machine"] = nlohmann_json::parse(params.machineInfoJson);
			}
			catch (...)
			{
				summary["machine"] = params.machineInfoJson;
			}
		}

		auto& results = summary["results"];
		results = nlohmann_json::array();
		uint32_t index = 1u;
		uint32_t failureCount = 0u;
		for (const auto& session : sessions)
		{
			nlohmann_json sessionJson;
			sessionJson["index"] = index++;
			sessionJson["artifact_name"] = session.sceneName;
			sessionJson["scene_name"] = session.sceneName;
			sessionJson["display_name"] = session.displayName;
			sessionJson["reference_names"] = session.referenceNames;
			sessionJson["scene_path"] = portablePathString(session.scenePath,params.reportDir,params.workingDirectory);
			sessionJson["sensor"] = session.sensorIndex;
			sessionJson["status"] = session.status;
			sessionJson["status_color"] = statusColor(session.status);
			sessionJson["details"] = session.details;
			sessionJson["compare"] = {
				{"errorThreshold",session.compare.errorThreshold},
				{"epsilon",session.compare.epsilon},
				{"allowedErrorPixelMode",allowedErrorPixelModeName(session.compare.allowedErrorPixelMode)},
				{"allowedErrorPixelRatio",session.compare.allowedErrorPixelRatio},
				{"allowedErrorPixelCount",session.compare.allowedErrorPixelCount},
				{"ssimErrorThreshold",session.compare.ssimErrorThreshold}
			};
			auto& images = sessionJson["array"];
			images = nlohmann_json::array();
			if ((session.status=="failed" || session.status=="error") && session.images.empty())
				++failureCount;
			for (const auto& image : session.images)
			{
				if (image.status=="failed" || image.status=="error" || image.status=="missing-reference" || image.status=="missing-render")
					++failureCount;
				nlohmann_json imageJson;
				imageJson["identifier"] = image.identifier;
				imageJson["title"] = image.title;
				imageJson["filename"] = image.renderExr.filename().string();
				imageJson["status"] = image.status;
				imageJson["status_color"] = statusColor(image.status);
				imageJson["details"] = image.details;
				if (!image.renderExr.empty() && std::filesystem::exists(image.renderExr))
					imageJson["render"] = makeGenericPathString(relativePath(image.renderExr,params.reportDir));
				else
					imageJson["expected_render"] = makeGenericPathString(relativePath(image.renderExr,params.reportDir));
				if (!image.referenceExr.empty())
					imageJson["reference"] = makeGenericPathString(relativePath(image.referenceExr,params.reportDir));
				if (!image.diffExr.empty())
					imageJson["difference"] = makeGenericPathString(relativePath(image.diffExr,params.reportDir));
				const bool hasComparisonMetric = (image.status=="passed" || image.status=="failed") && image.totalPixels>0u;
				if (hasComparisonMetric)
				{
					imageJson["error_pixels"] = image.errorPixels;
					imageJson["allowed_error_pixels"] = image.allowedErrorPixels;
					imageJson["total_pixels"] = image.totalPixels;
					imageJson["max_abs_error"] = image.maxAbsError;
					if (image.hasSsimDifference)
					{
						imageJson["metric"] = "ssim";
						imageJson["ssim_difference"] = image.ssimDifference;
						imageJson["ssim_error_threshold"] = params.compare.ssimErrorThreshold;
					}
					else
						imageJson["metric"] = "pixel-error";
				}
				images.push_back(std::move(imageJson));
			}
			results.push_back(std::move(sessionJson));
		}
		summary["num_of_tests"] = sessions.size();
		summary["failure_count"] = failureCount;
		summary["pass_status"] = hasFailures || hasErrors || failureCount>0u ? "failed":"passed";
		if (params.referenceDir.empty())
			summary["pass_status"] = "not-checked";
		return summary;
	}

	bool writeTextFile(const system::path& path, const std::string& contents) const
	{
		if (!ensureParentDirectoryExists(path))
			return false;
		std::ofstream file(path,std::ios::binary);
		if (!file)
			return false;
		file << contents;
		return true;
	}

	bool write()
	{
		std::error_code ec;
		std::filesystem::create_directories(params.reportDir,ec);
		if (ec)
			return false;

		compareSessions();

		const auto summary = makeSummaryJson().dump(2);
		if (!writeTextFile(params.reportDir/"summary.json",summary))
			return false;

		if (params.logger)
			params.logger->log("Path tracer report summary written to \"%s\"",system::ILogger::ELL_INFO,(params.reportDir/"summary.json").string().c_str());
		return true;
	}

	PathTracerReport::SCreationParams params;
	std::vector<SSessionResult> sessions;
	std::string generatedAt = nowString();
	std::optional<std::string> initialLdsCacheHash;
	bool hasFailures = false;
	bool hasErrors = false;
	bool compared = false;
};

PathTracerReport::PathTracerReport(SCreationParams&& params) : m_impl(std::make_unique<Impl>(std::move(params))) {}
PathTracerReport::~PathTracerReport() = default;
PathTracerReport::PathTracerReport(PathTracerReport&&) noexcept = default;
PathTracerReport& PathTracerReport::operator=(PathTracerReport&&) noexcept = default;

bool PathTracerReport::addSession(SSession&& session)
{
	return m_impl->addSession(std::move(session));
}

bool PathTracerReport::write()
{
	return m_impl->write();
}

const system::path& PathTracerReport::getReportDirectory() const
{
	return m_impl->params.reportDir;
}

bool PathTracerReport::hasFailures() const
{
	return m_impl->hasFailures || m_impl->hasErrors;
}

}
