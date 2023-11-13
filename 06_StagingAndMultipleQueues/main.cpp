// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/BasicMultiQueueApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


#include "app_resources/common.hlsl"


// This time we let the new base class score and pick queue families, as well as initialize `nbl::video::IUtilities` for us
class StagingAndMultipleQueuesApp final : public examples::BasicMultiQueueApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
		using video_base_t = examples::BasicMultiQueueApplication;
		using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;
	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		StagingAndMultipleQueuesApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

		// what will we do this time? Compute AABBs of animated points?
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(std::move(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			return true;
		}

		//
		void workLoopBody() override {}

		//
		bool keepRunning() override {return false;}

};


NBL_MAIN_FUNC(StagingAndMultipleQueuesApp)