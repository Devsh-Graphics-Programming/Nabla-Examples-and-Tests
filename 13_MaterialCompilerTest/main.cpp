// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/examples/examples.hpp"

//! Temporary
#include "nbl/asset/material_compiler3/IR.h"


using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;


// Testing our material compiler
class MaterialCompilerTest final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
		using device_base_t = application_templates::MonoDeviceApplication;
		using asset_base_t = BuiltinResourcesApplication;

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		MaterialCompilerTest(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			//

			return true;
		}

		// One-shot App
		bool keepRunning() override { return false; }

		// One-shot App
		void workLoopBody() override{}

		// Cleanup
		bool onAppTerminated() override
		{
			return device_base_t::onAppTerminated();
		}
};


NBL_MAIN_FUNC(MaterialCompilerTest)