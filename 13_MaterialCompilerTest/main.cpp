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

			using spectral_var_t = CFrontendIR::CSpectralVariable;
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

				// creating a node and changing our mind
				{
					auto image = ICPUImage::create({
						.type = IImage::E_TYPE::ET_2D,
						.samples = IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
						.format = EF_R16_SFLOAT,
						.extent = {32,32,1},
						.mipLevels = 1,
						.arrayLayers = 1
					});
					auto view = ICPUImageView::create({.image=image,.viewType=ICPUImageView::ET_2D,.format=EF_R16_SFLOAT});

					spectral_var_t::SCreationParams<1> params = {};
					params.knots.params[0].scale = 4.5f;
					params.knots.params[0].view = view;

					ASSERT_VALUE(view->getReferenceCount(),2,"initial reference count");

					auto handle = forest->_new<spectral_var_t>(std::move(params));
					ASSERT_VALUE(view->getReferenceCount(),2,"transferred reference count");

					// cleaning it up right away should run the destructor immediately and drop the image view refcount
					forest->_delete(handle);
					ASSERT_VALUE(view->getReferenceCount(),1,"after deletion reference count");
				}

				// delta reflection
				{
					const auto layerH = forest->_new<CFrontendIR::CLayer>();
					auto* layer = forest->deref(layerH);
					layer->debugInfo = forest->_new<CNodePool::CDebugInfo>("PerfectMirror");
					
					{
						const auto ctH = forest->_new<CFrontendIR::CCookTorrance>();
						auto* ct = forest->deref(ctH);
						ct->debugInfo = forest->_new<CNodePool::CDebugInfo>("Smooth NDF");
						ASSERT_VALUE(ct->ndParams.getRougness()[0].scale,0.f,"Initial NDF Params must be Smooth");
						ASSERT_VALUE(ct->ndParams.getRougness()[1].scale,0.f,"Initial NDF Params must be Smooth");
						ASSERT_VALUE(ct->ndParams.getDerivMap()[0].scale,0.f,"Initial NDF Params must be Flat");
						ASSERT_VALUE(ct->ndParams.getDerivMap()[1].scale,0.f,"Initial NDF Params must be Flat");
						layer->brdfTop = ctH;
					}

					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
				}

				// diffuse

				// twosided diffuse

				// diffuse transmissive
			}

			// emitter without IES profile

			// emitter with IES profile
			
			// anisotropic cook torrance GGX with Conductor Fresnel
			{
				const auto layerH = forest->_new<CFrontendIR::CLayer>();
				auto* layer = forest->deref(layerH);
				layer->debugInfo = forest->_new<CNodePool::CDebugInfo>("Anisotropic Aluminium");
					
				const auto mulH = forest->_new<CFrontendIR::CMul>();
				auto* mul = forest->deref(mulH);
				// BxDF always goes in left hand side of Mul
				{
					const auto ctH = forest->_new<CFrontendIR::CCookTorrance>();
					auto* ct = forest->deref(ctH);
					ct->debugInfo = forest->_new<CNodePool::CDebugInfo>("First Anisotropic GGX");
					ct->ndParams.getRougness()[0].scale = 0.2f;
					ct->ndParams.getRougness()[1].scale = 0.01f;
					mul->lhs = ctH;
				}
				// other multipliers in not-left subtrees
				{
					const auto frH = forest->_new<CFrontendIR::CFresnel>();
					auto* fr = forest->deref(frH);
					fr->debugInfo = forest->_new<CNodePool::CDebugInfo>("Aluminium Fresnel");
					{
						spectral_var_t::SCreationParams<3> params = {};
						params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
						params.knots.params[0].scale = 1.3404f;
						params.knots.params[1].scale = 0.95151f;
						params.knots.params[2].scale = 0.68603f;
						fr->orientedRealEta = forest->_new<spectral_var_t>(std::move(params));
					}
					{
						spectral_var_t::SCreationParams<3> params = {};
						params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
						params.knots.params[0].scale = 7.3509f;
						params.knots.params[1].scale = 6.4542f;
						params.knots.params[2].scale = 5.6351f;
						fr->orientedImagEta = forest->_new<spectral_var_t>(std::move(params));
					}
					mul->rhs = frH;
				}
				layer->brdfTop = mulH;

				// test that our bad subtree checks by swapping lhs with rhs
				std::swap(mul->lhs,mul->rhs);
				ASSERT_VALUE(forest->addMaterial(layerH,logger),false,"Contributor not in left subtree check failed");

				// should work now
				std::swap(mul->lhs,mul->rhs);
				ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Contributor in left subtree check failed");
			}

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