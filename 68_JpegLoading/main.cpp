// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include "nlohmann/json.hpp"
#include "argparse/argparse.hpp"

using json = nlohmann::json;

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

class JpegLoadingSampleApp final : public application_templates::MonoAssetManagerAndBuiltinResourceApplication 
{
public:
	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		argparse::ArgumentParser program("JPEG Loader");

		program.add_argument("--directory")
			.default_value(false)
			.implicit_value(true)
			.help("Path to the directory with JPEGs (not recursive)");

		program.add_argument("--output")
			.default_value(false)
			.implicit_value(true)
			.help("Path to the json file with benchmark results");

		try 
		{
			program.parse_args({ argv.data(), argv.data() + argv.size() });
		}
		catch (const std::exception& err)
		{
			std::cerr << err.what() << std::endl << program;
			return 1;
		}
 	}
};

NBL_MAIN_FUNC(JpegLoadingSampleApp);