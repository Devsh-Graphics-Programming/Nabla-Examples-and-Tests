// Copyright (C) 2018-2024 - DevSH GrapMonoAssetManagerAndBuiltinResourceApplicationhics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include <future>

#include "nlohmann/json.hpp"
#include "argparse/argparse.hpp"

using json = nlohmann::json;

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

class JpegLoaderApp final : public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using clock_t = std::chrono::steady_clock;
	using clock_resolution_t = std::chrono::milliseconds;
	using base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
public:
	using base_t::base_t;

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		argparse::ArgumentParser program("Color Space");

		program.add_argument<std::string>("--directory")
			.required()
			.help("Path to a directory where all JPEG files are stored (not recursive)");

		program.add_argument<std::string>("--output")
			.default_value("output.json")
			.help("Path to the file where the benchmark result will be stored");

		try
		{
			program.parse_args({ argv.data(), argv.data() + argv.size() });
		}
		catch (const std::exception& err)
		{
			std::cerr << err.what() << std::endl << program; // NOTE: std::cerr because logger isn't initialized yet
			return false;
		}

		if (!base_t::onAppInitialized(std::move(system)))
			return false;

		options.directory = program.get<std::string>("--directory");
		options.outputFile = program.get<std::string>("--output");

		auto start = clock_t::now();

		{ // TODO: Make this multi-threaded
			constexpr auto cachingFlags = static_cast<IAssetLoader::E_CACHING_FLAGS>(IAssetLoader::ECF_DONT_CACHE_REFERENCES & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
			const IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags, IAssetLoader::ELPF_NONE, m_logger.get());

			for (auto& item : std::filesystem::directory_iterator(options.directory))
			{

				auto& path = item.path();
				auto extension = path.extension();

				if (path.has_extension() && extension == ".jpg") 
				{
					m_logger->log("Loading %S", ILogger::ELL_INFO, path.c_str());
					m_assetMgr->getAsset(path.generic_string(), loadParams); // discard the loaded image
				}
			}
		}

		auto stop = clock_t::now();
		auto passed = std::chrono::duration_cast<clock_resolution_t>(stop - start).count();

		m_logger->log("Process took %llu ms", ILogger::ELL_INFO, passed);

		return true;
	}

	inline bool keepRunning() override
	{
		return false;
	}

	inline void workLoopBody() override
	{

	}
private:
	struct NBL_APP_OPTIONS
	{
		std::string directory;
		std::string outputFile;
	} options;
};

NBL_MAIN_FUNC(JpegLoaderApp)