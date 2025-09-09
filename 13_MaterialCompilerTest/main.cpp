// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/examples/examples.hpp"

//! Temporary
#include "nbl/asset/material_compiler3/CFrontendIR.h"


using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::asset::material_compiler3;
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

			auto forest = CFrontendIR::create();

			auto logger = m_logger.get();
// TODO: use std::source_info
#define ASSERT_VALUE(WHAT,VALUE,MSG) if (WHAT!=VALUE) return logFail("%s:%d test doesn't match expected value. %s",__FILE__,__LINE__,MSG)

			// simple white furnace testing materials
			{
				// transmission
				{
					auto layerH = forest->_new<CFrontendIR::CLayer>();
					auto* layer = forest->deref(layerH);
					layer->debugInfo = forest->_new<CNodePool::CDebugInfo>("MyWeirdInvisibleMaterial");
					layer->btdf = forest->_new<CFrontendIR::CDeltaTransmission>();
					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
				}
				// delta reflection
				// cook torrance GGX
				// cook torrance GGX with Fresnel
			}

			// diffuse
			// conductor (smooth and rough)
			// thindielectric
			// dielectric
			// diffuse transmitter

			// rough plastic

			// coated diffuse transmitter leaf
			// with subsurface beer scattering

			smart_refctd_ptr<IFile> file;
			{
				m_system->deleteFile(localOutputCWD/"frontend.dot");
				ISystem::future_t<smart_refctd_ptr<IFile>> future;
				m_system->createFile(future,localOutputCWD/"frontend.dot",IFileBase::E_CREATE_FLAGS::ECF_WRITE);
				if (!future.wait())
					return logFail("Failed to Open file for writing");
				future.acquire().move_into(file);
			}
			if (file)
			{
				auto visualization = forest->printDotGraph();
				// file write does not take an internal copy of pointer given, need to keep source alive till end
				IFile::success_t succ;
				file->write(succ,visualization.c_str(),0,visualization.size());
				succ.getBytesProcessed();
			}

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