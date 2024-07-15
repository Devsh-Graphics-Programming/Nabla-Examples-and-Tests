// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nabla.h>
#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/asset/filters/CFlattenRegionsStreamHashImageFilter.h"
#include "nlohmann/json.hpp"
#include "argparse/argparse.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

using json = nlohmann::json;

class StreamHashApp final : public application_templates::MonoDeviceApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
public:
	StreamHashApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		argparse::ArgumentParser program("x256 Hash Filter");

		program.add_argument("--verbose")
			.default_value(false)
			.implicit_value(true)
			.help("Print detailed logs.");

		program.add_argument("--test")
			.default_value(true)
			.implicit_value(true)
			.help("Perform tests & compare current data with references.");

		program.add_argument("--update-references")
			.default_value(false)
			.implicit_value(true)
			.help("Update JSON references with current data.");

		try
		{
			program.parse_args({ argv.data(), argv.data() + argv.size() });
		}
		catch (const std::exception& err)
		{
			std::cerr << err.what() << std::endl << program;
			return 1;
		}

		const bool verbose = program.get<bool>("--verbose");
		const bool test = program.get<bool>("--test");
		const bool updateReferences = program.get<bool>("--update-references");

		if (!device_base_t::MonoSystemMonoLoggerApplication::onAppInitialized(smart_refctd_ptr(system)))
			return false;

		auto assetManager = make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(m_system));

		IAssetLoader::SAssetLoadParams params = {};
		params.logger = m_logger.get();
		params.workingDirectory = sharedInputCWD;
		
		auto bundle = assetManager->getAsset((sharedInputCWD / "GLI/earth-array.dds").string(), params);
		const auto assets = bundle.getContents();
		if (assets.empty())
		{
			logFail("Could not load the image!");
			return false;
		}

		const auto* imageView = IAsset::castDown<const ICPUImageView>(assets[0].get());

		using filter_t = CFlattenRegionsStreamHashImageFilter;
		using state_t = filter_t::CState;

		bool status = true;
		filter_t filter;
		{
			state_t state;
			state.inImage = imageView->getCreationParameters().image.get();
			state.scratchMemory = filter.allocateScratchMemory(state.inImage);

			auto executeFilter = [&]<typename ExecutionPolicy>(ExecutionPolicy&& policy, bool& status)
			{
				using policy_t = std::remove_cvref<decltype(policy)>::type;

				auto policyToString = []() -> std::string 
				{
					if constexpr (std::is_same_v<policy_t, std::execution::sequenced_policy>)
						return "std::execution::sequenced_policy";
					else if constexpr (std::is_same_v<policy_t, std::execution::parallel_policy>)
						return "std::execution::parallel_policy";
					else if constexpr (std::is_same_v<policy_t, std::execution::parallel_unsequenced_policy>)
						return "std::execution::parallel_unsequenced_policy";
					else
						return "unknown";
				};

				const auto policyName = policyToString();

				const auto start = std::chrono::high_resolution_clock::now();

				m_logger->log("Executing hash filter with " + policyName + " policy!", ILogger::ELL_PERFORMANCE);

				if (!filter.execute(policy, &state))
				{
					logFail(("Failed to create hash for input image with " + policyName + " execution policy!").c_str());
					status = false;
					return;
				}

				const auto end = std::chrono::high_resolution_clock::now();
				const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

				m_logger->log("Took: " + std::to_string(duration.count()) + " ms", ILogger::ELL_PERFORMANCE);

				json outJson;
				outJson["image"] = json::array();
				for (const auto& it : state.outHash)
					outJson["image"].push_back(it);

				outJson["policy"] = policyName;

				const auto& parameters = state.inImage->getCreationParameters();
				outJson["mipLevels"] = json::array();
				for (auto miplevel = 0u; miplevel < parameters.mipLevels; ++miplevel)
				{
					json mipLevelJson;
					mipLevelJson["layers"] = json::array();

					for (auto layer = 0u; layer < parameters.arrayLayers; ++layer)
					{
						const auto* hash = reinterpret_cast<state_t::hash_t*>(state.scratchMemory.heap->getPointer()) + (miplevel * parameters.arrayLayers) + layer;

						json layerJson;
						layerJson["hash"] = json::array();
						for (const auto& it : *hash)
							layerJson["hash"].push_back(it);

						mipLevelJson["layers"].push_back(layerJson);
					}

					outJson["mipLevels"].push_back(mipLevelJson);
				}

				const std::string prettyJson = outJson.dump(4);

				if(verbose)
					m_logger->log(prettyJson, ILogger::ELL_INFO);
				
				auto cwd = localOutputCWD;

				auto fileName = policyName;
				std::replace(fileName.begin(), fileName.end(), ':', '_');
				fileName += ".json";

				const auto filePath = (cwd / fileName).make_preferred();

				system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
				m_system->createFile(future, filePath, system::IFileBase::ECF_WRITE);
				if (auto file = future.acquire(); file && bool(*file))
				{
					system::IFile::success_t succ;
					(*file)->write(succ, prettyJson.data(), 0, prettyJson.size());
					succ.getBytesProcessed(true);
				}
				else
					logFail(("Failed to write " + fileName + " data to \"" + filePath.string() + "\"").c_str());

				m_logger->log("Saved " + fileName + " to \"" + filePath.string() + "\"!", ILogger::ELL_INFO);

				const std::string referenceJsonPath = std::filesystem::absolute((cwd / ("../test/references/" + fileName)).make_preferred()).string();

				if (test)
				{
					std::ifstream referenceFile(referenceJsonPath);

					json referenceJson;
					if (referenceFile.is_open())
					{
						referenceFile >> referenceJson;

						m_logger->log("Comparing " + policyName + "'s reference, performing test..", ILogger::ELL_WARNING);
						const bool passed = outJson == referenceJson;

						if (passed)
							m_logger->log("Passed!", ILogger::ELL_WARNING);
						else
						{
							logFail("Failed!");
							status = false;
						}
					}
					else
					{
						logFail(("Could not open " + policyName + "'s reference file, skipping requested test. If the reference doesn't exist make sure to create one with --update-references flag!").c_str());
						status = false;
					}
				}

				if (updateReferences)
				{
					std::error_code errorCode;
					std::filesystem::copy(filePath, referenceJsonPath, std::filesystem::copy_options::overwrite_existing, errorCode);
					if (errorCode)
					{
						logFail(("Failed to update " + policyName + "'s reference!").c_str());
						status = false;
					}
					else
						m_logger->log("Updated " + policyName + "'s reference, saved to \"" + referenceJsonPath + "\"!", ILogger::ELL_INFO);
				}
			};

			executeFilter(std::execution::seq, status);
			executeFilter(std::execution::par, status); // looks we we have OOB writes and access violations
		}

		return status;
	}

	void workLoopBody() override {}
	bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(StreamHashApp)
