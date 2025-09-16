// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/examples/examples.hpp"

//! Temporary, for faster iteration outside of PCH
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

			// dummy image views
			smart_refctd_ptr<ICPUImageView> monochromeImageView, rgbImageView;
			{
				constexpr auto format = EF_R16_SFLOAT;
				auto image = ICPUImage::create({
					.type = IImage::E_TYPE::ET_2D,
					.samples = IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
					.format = format,
					.extent = {32,32,1},
					.mipLevels = 1,
					.arrayLayers = 1
				});
				monochromeImageView = ICPUImageView::create({.image=std::move(image),.viewType=ICPUImageView::ET_2D,.format=format});
			}
			{
				constexpr auto format = EF_R8G8B8A8_SRGB;
				auto image = ICPUImage::create({
					.type = IImage::E_TYPE::ET_2D,
					.samples = IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
					.format = format,
					.extent = {1024,1024,1},
					.mipLevels = 11,
					.arrayLayers = 72 // fur teh lulz
				});
				rgbImageView = ICPUImageView::create({.image=std::move(image),.viewType=ICPUImageView::ET_2D_ARRAY,.format=format});
			}

// TODO: use std::source_info
#define ASSERT_VALUE(WHAT,VALUE,MSG) if (WHAT!=VALUE) return logFail("%s:%d test doesn't match expected value. %s",__FILE__,__LINE__,MSG)

			using spectral_var_t = CFrontendIR::CSpectralVariable;
			// simple white furnace testing materials
			{
				// transmission
				{
					const auto layerH = forest->_new<CFrontendIR::CLayer>();
					auto* layer = forest->deref(layerH);
					layer->debugInfo = forest->_new<CNodePool::CDebugInfo>("MyWeirdInvisibleMaterial");
					layer->btdf = forest->_new<CFrontendIR::CDeltaTransmission>();
					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
				}

				// creating a node and changing our mind
				{

					spectral_var_t::SCreationParams<1> params = {};
					params.knots.params[0].scale = 4.5f;
					params.knots.params[0].view = monochromeImageView;

					ASSERT_VALUE(monochromeImageView->getReferenceCount(),2,"initial reference count");

					const auto handle = forest->_new<spectral_var_t>(std::move(params));
					ASSERT_VALUE(monochromeImageView->getReferenceCount(),2,"transferred reference count");

					// cleaning it up right away should run the destructor immediately and drop the image view refcount
					forest->_delete(handle);
					ASSERT_VALUE(monochromeImageView->getReferenceCount(),1,"after deletion reference count");
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

				// two-sided diffuse
				{
					const auto layerH = forest->_new<CFrontendIR::CLayer>();
					auto* layer = forest->deref(layerH);
					layer->debugInfo = forest->_new<CNodePool::CDebugInfo>("Twosided Diffuse");
					const auto orenNayarH = forest->_new<CFrontendIR::COrenNayar>();
					auto* orenNayar = forest->deref(orenNayarH);
					orenNayar->debugInfo = forest->_new<CNodePool::CDebugInfo>("Actually Lambertian");
					// TODO: add a derivative map for testing the printing and compilation
					layer->brdfTop = orenNayarH;
					layer->brdfBottom = orenNayarH;
					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
				}

				// diffuse isotropic rough transmissive
				{
					const auto layerH = forest->_new<CFrontendIR::CLayer>();
					auto* layer = forest->deref(layerH);
					layer->debugInfo = forest->_new<CNodePool::CDebugInfo>("Rough Diffuse Transmitter");
					// The material compiler can't handle the BRDF vs. BTDF normalization and energy conservation for you.
					// Given a BRDF expression we simply can't tell if the missing energy was supposed
					// to be transferred to the BTDF or absorbed by the BRDF itself.
					// Hence the BTDF expression must contain the BRDF coating term.
					const auto mulH = forest->_new<CFrontendIR::CMul>();
					auto* mul = forest->deref(mulH);
					// regular BRDF will normalize to 100% over a hemisphere, if we allow a BTDF term we must split it half/half
					{
						spectral_var_t::SCreationParams<1> params = {};
						params.knots.params[0].scale = 0.5f;
						mul->rhs = forest->_new<spectral_var_t>(std::move(params));
					}
					// create the BxDF as we'd do for a single BRDF or BTDF
					{
						const auto orenNayarH = forest->_new<CFrontendIR::COrenNayar>();
						auto* orenNayar = forest->deref(orenNayarH);
						orenNayar->debugInfo = forest->_new<CNodePool::CDebugInfo>("BxDF Normalized For Whole Sphere");
						auto roughness = orenNayar->ndParams.getRougness();
						roughness[1].scale = roughness[0].scale = 0.8f;
						mul->lhs = orenNayarH;
					}
					// TODO: add a derivative map for testing the printing and compilation
					layer->brdfTop = mulH;
					layer->btdf = mulH;
					layer->brdfBottom = mulH;
					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
				}
			}

			// emitter without IES profile
			{
				const auto layerH = forest->_new<CFrontendIR::CLayer>();
				auto* layer = forest->deref(layerH);
				layer->debugInfo = forest->_new<CNodePool::CDebugInfo>("Twosided Constant Emitter");
				{
					const auto mulH = forest->_new<CFrontendIR::CMul>();
					auto* mul = forest->deref(mulH);
					{
						const auto emitterH = forest->_new<CFrontendIR::CEmitter>();
						// no profile, unit emission
						mul->lhs = emitterH;
					}
					// we multiply the unit emitter by the value we actually want
					{
						spectral_var_t::SCreationParams<3> params = {};
						params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
						params.knots.params[0].scale = 3.f;
						params.knots.params[1].scale = 7.f;
						params.knots.params[2].scale = 15.f;
						mul->rhs = forest->_new<spectral_var_t>(std::move(params));
					}
					layer->brdfTop = mulH;
					layer->brdfBottom = mulH;
				}
				ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
			}

			// emitter with IES profile
			{
				const auto layerH = forest->_new<CFrontendIR::CLayer>();
				auto* layer = forest->deref(layerH);
				layer->debugInfo = forest->_new<CNodePool::CDebugInfo>("IES Profile Emitter");
				{
					const auto mulH = forest->_new<CFrontendIR::CMul>();
					auto* mul = forest->deref(mulH);
					{
						const auto emitterH = forest->_new<CFrontendIR::CEmitter>();
						auto* emitter = forest->deref(emitterH);
						// you should use this to normalize the profile to unit emission over the hemisphere
						// so the light gets picked "fairly"
						emitter->profile.scale = 0.01f;
						emitter->profile.viewChannel = 0;
						emitter->profile.view = monochromeImageView;
						// these are defaults but going to set them
						emitter->profile.sampler.TextureWrapU = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
						emitter->profile.sampler.TextureWrapV = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
						// TODO: set transform after merging the OBB PR
						//emitter->profileTransform = ;
						mul->lhs = emitterH;
					}
					// we multiply the unit emitter by the emission color value we actually want
					{
						spectral_var_t::SCreationParams<3> params = {};
						params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
						params.knots.params[0].scale = 60.f;
						params.knots.params[1].scale = 90.f;
						params.knots.params[2].scale = 45.f;
						mul->rhs = forest->_new<spectral_var_t>(std::move(params));
					}
					layer->brdfTop = mulH;
				}
				ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
			}

			// onesided emitter with spatially varying emission from the backside
			{
				const auto layerH = forest->_new<CFrontendIR::CLayer>();
				auto* layer = forest->deref(layerH);
				layer->debugInfo = forest->_new<CNodePool::CDebugInfo>("Spatially Varying Emitter");
				{
					const auto mulH = forest->_new<CFrontendIR::CMul>();
					auto* mul = forest->deref(mulH);
					{
						const auto emitterH = forest->_new<CFrontendIR::CEmitter>();
						// no profile, unit emission
						mul->lhs = emitterH;
					}
					// we multiply the unit emitter by the value we actually want
					{
						spectral_var_t::SCreationParams<3> params = {};
						params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
						for (auto c=0; c<3; c++)
						{
							params.knots.params[c].scale = 4.9f;
							params.knots.params[c].viewChannel = c;
							params.knots.params[c].view = rgbImageView;
							params.knots.params[c].sampler.TextureWrapU = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_BORDER;
							params.knots.params[c].sampler.TextureWrapV = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_BORDER;
							params.knots.params[c].sampler.BorderColor = ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_BLACK;
						}
						mul->rhs = forest->_new<spectral_var_t>(std::move(params));
					}
					layer->brdfBottom = mulH;
				}
				ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
			}

			// spatially varying emission but with a profile (think classroom projector)
			{
				const auto layerH = forest->_new<CFrontendIR::CLayer>();
				auto* layer = forest->deref(layerH);
				layer->debugInfo = forest->_new<CNodePool::CDebugInfo>("Spatially Varying Emitter with IES profile e.g. Digital Projector");
				{
					const auto mulH = forest->_new<CFrontendIR::CMul>();
					auto* mul = forest->deref(mulH);
					{
						const auto emitterH = forest->_new<CFrontendIR::CEmitter>();
						auto* emitter = forest->deref(emitterH);
						emitter->profile.scale = 67.f;
						emitter->profile.viewChannel = 0;
						emitter->profile.view = monochromeImageView;
						// lets try some other samplers
						emitter->profile.sampler.TextureWrapU = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE;
						emitter->profile.sampler.TextureWrapV = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE;
						// try with default transform
						mul->lhs = emitterH;
					}
					// we multiply the unit emitter by the value we actually want
					{
						spectral_var_t::SCreationParams<3> params = {};
						params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
						for (auto c=0; c<3; c++)
						{
							params.knots.params[c].scale = 900.f; // super bright cause its probably small
							params.knots.params[c].viewChannel = c;
							params.knots.params[c].view = rgbImageView;
							params.knots.params[c].sampler.TextureWrapU = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_BORDER;
							params.knots.params[c].sampler.TextureWrapV = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_BORDER;
							params.knots.params[c].sampler.BorderColor = ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_BLACK;
						}
						mul->rhs = forest->_new<spectral_var_t>(std::move(params));
					}
					layer->brdfTop = mulH;
				}
				ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
			}
			
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