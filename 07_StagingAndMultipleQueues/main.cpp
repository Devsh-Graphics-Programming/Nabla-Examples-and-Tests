// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "nbl/application_templates/BasicMultiQueueApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


#include "app_resources/common.hlsl"


// This time we let the new base class score and pick queue families, as well as initialize `nbl::video::IUtilities` for us
class StagingAndMultipleQueuesApp final : public application_templates::BasicMultiQueueApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = application_templates::BasicMultiQueueApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		StagingAndMultipleQueuesApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

		// This time we will load images and compute their histograms and output them as CSV
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(std::move(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			// TODO: for later
			// - fire up 2 aux threads (one to upload images and transfer ownership [if necessary], another to acquire ownership of histogram buffers and write out CSVs)
			// - main thread grabs `IGPUImage` from a queue and acquires ownership [if necessary] performs all setup to launch a dispatch hands off a histogram buffer

			return true;
		}

		//
		void workLoopBody() override {}

		//
		bool keepRunning() override {return false;}

		//
		bool onAppTerminated() override
		{
			return device_base_t::onAppTerminated();
		}

	protected:
		// Override will become irrelevant in the vulkan_1_3 branch
		SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
		{
			auto retval = device_base_t::getRequiredDeviceFeatures();
			retval.shaderStorageImageWriteWithoutFormat = true;
			return retval;
		}

		// Ideally don't want to have to 
		SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
		{
			auto retval = device_base_t::getPreferredDeviceFeatures();
			retval.shaderStorageImageReadWithoutFormat = true;
			return retval;
		}

};


NBL_MAIN_FUNC(StagingAndMultipleQueuesApp)