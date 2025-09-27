// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
//#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"
#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;


// Testing our Mitsuba Loader
class MitsubaLoaderTest final : public BuiltinResourcesApplication
{
		using base_t = BuiltinResourcesApplication;

		bool failedTest(const core::string& relPath)
		{
			IAssetLoader::SAssetLoadParams params = {};
			params.logger = m_logger.get();
			auto asset = m_assetMgr->getAsset(relPath,params);
			if (asset.getContents().empty())
			{
				m_logger->log("Failed To Load %s",ILogger::ELL_ERROR);
				return true;
			}
			// so we don't run out of RAM during testing
			m_assetMgr->clearAllAssetCache();
			return false;
		}

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		MitsubaLoaderTest(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

//			m_assetMgr->addAssetLoader(make_smart_refctd_ptr<ext::MitsubaLoader::CMitsubaLoader>());

			// first batch
			if (failedTest("shapetest.xml")) return false;
			if (failedTest("daily_pt.xml")) return false;
			if (failedTest("brdf_eval_test.xml")) return false;
			if (failedTest("brdf_eval_test_as.xml")) return false;
			if (failedTest("brdf_eval_test_diffuse.xml")) return false;
			if (failedTest("brdf_eval_test_lambert.xml")) return false;

			// some of our test scenes won't load without the `.serialized` support
			m_assetMgr->addAssetLoader(make_smart_refctd_ptr<ext::MitsubaLoader::CSerializedLoader>());

			return true;
		}

		// One-shot App
		bool keepRunning() override { return false; }

		// One-shot App
		void workLoopBody() override {}

		// Cleanup
		bool onAppTerminated() override
		{
			return base_t::onAppTerminated();
		}
};


NBL_MAIN_FUNC(MitsubaLoaderTest)