// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nabla.h>
#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/asset/filters/CFlattenRegionsStreamHashImageFilter.h"
#include "nlohmann/json.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

class StreamHashApp final : public application_templates::MonoDeviceApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
public:
	StreamHashApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
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
			assert(false);
		}

		const auto* imageView = IAsset::castDown<const ICPUImageView>(assets[0].get());

		using filter_t = CFlattenRegionsStreamHashImageFilter;
		using state_t = filter_t::CState;
		using json = nlohmann::json;

		filter_t filter;
		{
			state_t state;
			state.inImage = imageView->getCreationParameters().image.get();
			state.scratchMemory = filter.allocateScratchMemory(state.inImage);

			auto executeFilter = [&]<typename ExecutionPolicy>(ExecutionPolicy&& policy)
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

				auto start = std::chrono::high_resolution_clock::now();

				if (!filter.execute(policy, &state))
				{
					logFail("Failed to create hash for input image!");
					assert(false);
				}

				auto end = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
				m_logger->log("Filter execution time: " + std::to_string(duration.count()) + " ms", ILogger::ELL_PERFORMANCE);
				m_logger->log("Policy: " + policyName, ILogger::ELL_PERFORMANCE);

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

				std::string prettyJson = outJson.dump(4);
				m_logger->log(prettyJson, ILogger::ELL_DEBUG);
				
				auto cwd = localOutputCWD;

				auto fileName = policyName;
				std::replace(fileName.begin(), fileName.end(), ':', '_');

				fileName += ".json";

				auto filePath = (cwd / fileName).make_preferred();

				system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
				m_system->createFile(future, filePath, system::IFileBase::ECF_WRITE);
				if (auto file = future.acquire(); file && bool(*file))
				{
					system::IFile::success_t succ;
					(*file)->write(succ, prettyJson.data(), 0, prettyJson.size());
					succ.getBytesProcessed(true);
				}
			};

			executeFilter(std::execution::seq);
			// executeFilter(std::execution::par); // looks we we have OOB writes and access violations
		}

		return true;
	}

	void workLoopBody() override {}
	bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(StreamHashApp)
