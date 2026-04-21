// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_PATH_TRACER_REPORT_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_PATH_TRACER_REPORT_H_INCLUDED_

#include "nbl/system/path.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nbl::asset
{
class IAssetManager;
}

namespace nbl::system
{
class ILogger;
}

namespace nbl::this_example
{

class PathTracerReport final
{
	public:
		struct SCompareSettings
		{
			enum class EAllowedErrorPixelMode : uint8_t
			{
				RelativeToResolution,
				AbsoluteCount
			};

			double errorThreshold = 0.05;
			double epsilon = 0.00001;
			EAllowedErrorPixelMode allowedErrorPixelMode = EAllowedErrorPixelMode::RelativeToResolution;
			double allowedErrorPixelRatio = 0.0001;
			uint64_t allowedErrorPixelCount = 0u;
			double ssimErrorThreshold = 0.001;
		};

		struct SImageArtifact
		{
			std::string identifier;
			std::string title;
			system::path exrPath;
			bool requiresReference = true;
		};

		struct SSession
		{
			std::string sceneName;
			std::string displayName;
			std::vector<std::string> referenceNames;
			system::path scenePath;
			uint32_t sensorIndex = 0u;
			std::string status;
			std::string details;
			SCompareSettings compare;
			std::vector<SImageArtifact> images;
		};

		struct SCreationParams
		{
			system::path reportDir;
			system::path referenceDir;
			system::path workingDirectory;
			system::path lowDiscrepancySequenceCachePath;
			std::string commandLine;
			std::string buildConfig;
			std::string buildInfoJson;
			std::string machineInfoJson;
			SCompareSettings compare;
			asset::IAssetManager* assetManager = nullptr;
			system::ILogger* logger = nullptr;
		};

		explicit PathTracerReport(SCreationParams&& params);
		~PathTracerReport();

		PathTracerReport(const PathTracerReport&) = delete;
		PathTracerReport& operator=(const PathTracerReport&) = delete;
		PathTracerReport(PathTracerReport&&) noexcept;
		PathTracerReport& operator=(PathTracerReport&&) noexcept;

		bool addSession(SSession&& session);
		bool write();

		const system::path& getReportDirectory() const;
		bool hasFailures() const;

	private:
		struct Impl;
		std::unique_ptr<Impl> m_impl;
};

}

#endif
