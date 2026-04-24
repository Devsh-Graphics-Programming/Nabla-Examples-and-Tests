// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/examples/examples.hpp"



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

			auto forest = CFrontendIR::create({.composed={.blockSizeKBLog2=4}});
			auto& forestPool = forest->getObjectPool();

			auto logger = m_logger.get();

			{
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
	#define ASSERT_VALUE(WHAT,VALUE,MSG) if (WHAT!=VALUE) \
		return logFail("%s:%d test doesn't match expected value. %s",__FILE__,__LINE__,MSG); \
	else if (!VALUE) \
	if constexpr (std::is_same_v<decltype(VALUE),bool>) \
		m_logger->log("Disregard the error above, its expected.",system::ILogger::ELL_INFO)


				using spectral_var_t = CFrontendIR::CSpectralVariable;
				// simple white furnace testing materials
				{
					// transmission
					{
						const auto layerH = forestPool.emplace<CFrontendIR::CLayer>();
						auto* layer = forestPool.deref(layerH);
						layer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("MyWeirdInvisibleMaterial");
						layer->btdf = forestPool.emplace<CFrontendIR::CDeltaTransmission>();
						ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
					}

					// creating a node and changing our mind
					{

						spectral_var_t::SCreationParams<1> params = {};
						params.knots.params[0].scale = 4.5f;
						params.knots.params[0].view = monochromeImageView;

						ASSERT_VALUE(monochromeImageView->getReferenceCount(),2,"initial reference count");

						const auto handle = forestPool.emplace<spectral_var_t>(std::move(params));
						ASSERT_VALUE(monochromeImageView->getReferenceCount(),2,"transferred reference count");

						// cleaning it up right away should run the destructor immediately and drop the image view refcount
						forestPool._delete(handle);
						ASSERT_VALUE(monochromeImageView->getReferenceCount(),1,"after deletion reference count");
					}

					// delta reflection
					{
						const auto layerH = forestPool.emplace<CFrontendIR::CLayer>();
						auto* layer = forestPool.deref(layerH);
						layer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("PerfectMirror");
					
						{
							const auto ctH = forestPool.emplace<CFrontendIR::CCookTorrance>();
							auto* ct = forestPool.deref(ctH);
							ct->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Smooth NDF");
							ASSERT_VALUE(ct->ndParams.getRougness()[0].scale,0.f,"Initial NDF Params must be Smooth");
							ASSERT_VALUE(ct->ndParams.getRougness()[1].scale,0.f,"Initial NDF Params must be Smooth");
							ASSERT_VALUE(ct->ndParams.getDerivMap()[0].scale,0.f,"Initial NDF Params must be Flat");
							ASSERT_VALUE(ct->ndParams.getDerivMap()[1].scale,0.f,"Initial NDF Params must be Flat");
							layer->brdfTop = ctH;
						}

						// test layer cycle detection
						layer->coated = layerH;
						ASSERT_VALUE(forest->addMaterial(layerH,logger),false,"Layer Cycle Detection");
						layer->coated = {};

						ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
					}

					// two-sided diffuse
					{
						const auto layerH = forestPool.emplace<CFrontendIR::CLayer>();
						auto* layer = forestPool.deref(layerH);
						layer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Twosided Diffuse");
						const auto orenNayarH = forestPool.emplace<CFrontendIR::COrenNayar>();
						auto* orenNayar = forestPool.deref(orenNayarH);
						orenNayar->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Actually Lambertian");
						// TODO: add a derivative map for testing the printing and compilation
						layer->brdfTop = orenNayarH;
						layer->brdfBottom = orenNayarH;
						ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
					}

					// diffuse isotropic rough transmissive
					{
						const auto layerH = forestPool.emplace<CFrontendIR::CLayer>();
						auto* layer = forestPool.deref(layerH);
						layer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Rough Diffuse Transmitter");
						// The material compiler can't handle the BRDF vs. BTDF normalization and energy conservation for you.
						// Given a BRDF expression we simply can't tell if the missing energy was supposed
						// to be transferred to the BTDF or absorbed by the BRDF itself.
						// Hence the BTDF expression must contain the BRDF coating term (how much energy is "taken" by the BRDF).
						const auto mulH = forestPool.emplace<CFrontendIR::CMul>();
						layer->brdfTop = mulH;
						layer->btdf = mulH;
						layer->brdfBottom = mulH;

						auto* mul = forestPool.deref(mulH);
						// regular BRDF will normalize to 100% over a hemisphere, if we allow a BTDF term we must split it half/half
						{
							spectral_var_t::SCreationParams<1> params = {};
							params.knots.params[0].scale = 0.5f;
							mul->rhs = forestPool.emplace<spectral_var_t>(std::move(params));
						}

						// test expression cycle detection
						mul->lhs = mulH;
						ASSERT_VALUE(forest->addMaterial(layerH,logger),false,"Expression Cycle Detection");

						// create the BxDF as we'd do for a single BRDF or BTDF
						{
							const auto orenNayarH = forestPool.emplace<CFrontendIR::COrenNayar>();
							auto* orenNayar = forestPool.deref(orenNayarH);
							orenNayar->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("BxDF Normalized For Whole Sphere");
							auto roughness = orenNayar->ndParams.getRougness();
							roughness[1].scale = roughness[0].scale = 0.8f;
							mul->lhs = orenNayarH;
						}
						// TODO: add a derivative map for testing the printing and compilation
						ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
					}
				}

				// emitter without IES profile
				{
					const auto layerH = forestPool.emplace<CFrontendIR::CLayer>();
					auto* layer = forestPool.deref(layerH);
					layer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Twosided Constant Emitter");
					{
						const auto mulH = forestPool.emplace<CFrontendIR::CMul>();
						auto* mul = forestPool.deref(mulH);
						{
							const auto emitterH = forestPool.emplace<CFrontendIR::CEmitter>();
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
							mul->rhs = forestPool.emplace<spectral_var_t>(std::move(params));
						}
						layer->brdfTop = mulH;
						layer->brdfBottom = mulH;
					}
					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
				}

				// emitter with IES profile
				{
					const auto layerH = forestPool.emplace<CFrontendIR::CLayer>();
					auto* layer = forestPool.deref(layerH);
					layer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("IES Profile Emitter");
					{
						const auto mulH = forestPool.emplace<CFrontendIR::CMul>();
						auto* mul = forestPool.deref(mulH);
						{
							const auto emitterH = forestPool.emplace<CFrontendIR::CEmitter>();
							auto* emitter = forestPool.deref(emitterH);
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
							mul->rhs = forestPool.emplace<spectral_var_t>(std::move(params));
						}
						layer->brdfTop = mulH;
					}
					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
				}

				// onesided emitter with spatially varying emission from the backside
				{
					const auto layerH = forestPool.emplace<CFrontendIR::CLayer>();
					auto* layer = forestPool.deref(layerH);
					layer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Spatially Varying Emitter");
					{
						const auto mulH = forestPool.emplace<CFrontendIR::CMul>();
						auto* mul = forestPool.deref(mulH);
						{
							const auto emitterH = forestPool.emplace<CFrontendIR::CEmitter>();
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
							mul->rhs = forestPool.emplace<spectral_var_t>(std::move(params));
						}
						layer->brdfBottom = mulH;
					}
					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
				}

				// spatially varying emission but with a profile (think classroom projector)
				{
					const auto layerH = forestPool.emplace<CFrontendIR::CLayer>();
					auto* layer = forestPool.deref(layerH);
					layer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Spatially Varying Emitter with IES profile e.g. Digital Projector");
					{
						const auto mulH = forestPool.emplace<CFrontendIR::CMul>();
						auto* mul = forestPool.deref(mulH);
						{
							const auto emitterH = forestPool.emplace<CFrontendIR::CEmitter>();
							auto* emitter = forestPool.deref(emitterH);
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
							mul->rhs = forestPool.emplace<spectral_var_t>(std::move(params));
						}
						layer->brdfTop = mulH;
					}
					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Add Material");
				}
			
				// anisotropic cook torrance GGX with Conductor Fresnel
				{
					const auto layerH = forestPool.emplace<CFrontendIR::CLayer>();
					auto* layer = forestPool.deref(layerH);
					layer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Anisotropic Aluminium");
					
					const auto mulH = forestPool.emplace<CFrontendIR::CMul>();
					auto* mul = forestPool.deref(mulH);
					// BxDF always goes in left hand side of Mul
					{
						const auto ctH = forestPool.emplace<CFrontendIR::CCookTorrance>();
						auto* ct = forestPool.deref(ctH);
						ct->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("First Anisotropic GGX");
						ct->ndParams.getRougness()[0].scale = 0.2f;
						ct->ndParams.getRougness()[1].scale = 0.01f;
						mul->lhs = ctH;
					}
					// other multipliers in not-left subtrees
					mul->rhs = forest->createNamedFresnel("Al");
					layer->brdfTop = mulH;

					// test that our bad subtree checks by swapping lhs with rhs
					std::swap(mul->lhs,mul->rhs);
					ASSERT_VALUE(forest->addMaterial(layerH,logger),false,"Contributor not in left subtree check failed");

					// should work now
					std::swap(mul->lhs,mul->rhs);
					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Contributor in left subtree check failed");
				}

				// dielectric
				const auto dielectricH = forestPool.emplace<CFrontendIR::CMul>();
				{
					auto* mul = forestPool.deref(dielectricH);
					// do fresnel first
					const auto fresnelH = forest->createNamedFresnel("ThF4");
					auto* fresnel = forestPool.deref(fresnelH);
					mul->rhs = fresnelH;
					// BxDF always goes in left hand side of Mul
					{
						const auto ctH = forestPool.emplace<CFrontendIR::CCookTorrance>();
						auto* ct = forestPool.deref(ctH);
						ct->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("First Isotropic GGX");
						ct->ndParams.getRougness()[0].scale = ct->ndParams.getRougness()[1].scale = 0.05f;
						// ignored for BRDFs, needed for BTDFs
						ct->orientedRealEta = fresnel->orientedRealEta;
						mul->lhs = ctH;
					}
				}
				{
					const auto layerH = forestPool.emplace<CFrontendIR::CLayer>();
					auto* layer = forestPool.deref(layerH);
					layer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Glass");
					
					// use same BxDF for all parts of a layer
					layer->brdfTop = dielectricH;
					layer->btdf = dielectricH;
					layer->brdfBottom = dielectricH;

					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"Dielectric");
				}

				// correlated thindielectric (exit through a microfacet with identical normal on the other side - no refraction possible) 
				{
					const auto layerH = forestPool.emplace<CFrontendIR::CLayer>();
					auto* layer = forestPool.deref(layerH);
					layer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Correlated Single Pane");
					
					// do fresnel first for all to have the same one
					const auto fresnelH = forest->createNamedFresnel("ThF4");
					const auto* fresnel = forestPool.deref(fresnelH);

					const auto ctH = forestPool.emplace<CFrontendIR::CCookTorrance>();
					{
						auto* ct = forestPool.deref(ctH);
						ct->ndParams.getRougness()[0].scale = ct->ndParams.getRougness()[1].scale = 0.1f;
						// ignored for BRDFs, needed for BTDFs
						ct->orientedRealEta = fresnel->orientedRealEta;
						layer->brdfTop = forest->createMul(ctH,fresnelH);
					}
					// the BTDF
					{
						const auto thinInfiniteScatterH = forestPool.emplace<CFrontendIR::CThinInfiniteScatterCorrection>();
						{
							auto* thinInfiniteScatter = forestPool.deref(thinInfiniteScatterH);
							thinInfiniteScatter->reflectanceTop = fresnelH;
							thinInfiniteScatter->reflectanceBottom = fresnelH;
							// without extinction
						}
						layer->btdf = forest->createMul(forestPool.emplace<CFrontendIR::CDeltaTransmission>(),thinInfiniteScatterH);
					}
					// the interface on top as Air->ThF4, we need interface on bottom to be ThF4->Air so reciprocate te Eta
					layer->brdfBottom = forest->createMul(forest->reciprocate(ctH)._const_cast(),fresnelH);
				
					{
						auto* imagEta = forestPool.deref(fresnel->orientedImagEta);
						imagEta->getParam(0)->scale = std::numeric_limits<float>::min();
						imagEta->getParam(1)->scale = -std::numeric_limits<float>::max();
						imagEta->getParam(2)->scale = 0.5f;
						ASSERT_VALUE(forest->addMaterial(layerH,logger),false,"Imaginary Fresnel disallowed");
						for (uint8_t i=0; i<3; i++)
							imagEta->getParam(i)->scale = 0.f;
					}

					ASSERT_VALUE(forest->addMaterial(layerH,logger),true,"ThinDielectric");
				}

				// complex materials with coatings with IOR 1.5
				{
					// make the nodes everyone shares
					const auto roughDiffuseH = forestPool.emplace<CFrontendIR::CMul>();
					{
						auto* mul = forestPool.deref(roughDiffuseH);
						{
							const auto orenNayarH = forestPool.emplace<CFrontendIR::COrenNayar>();
							auto* orenNayar = forestPool.deref(orenNayarH);
							orenNayar->ndParams.getRougness()[0].scale = orenNayar->ndParams.getRougness()[1].scale = 0.2f;
							mul->lhs = orenNayarH;
						}
						{
							spectral_var_t::SCreationParams<3> params = {};
							params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
							params.knots.params[0].scale = 0.9f;
							params.knots.params[1].scale = 0.6f;
							params.knots.params[2].scale = 0.01f;
							const auto albedoH = forestPool.emplace<CFrontendIR::CSpectralVariable>(std::move(params));
							forestPool.deref(albedoH)->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Albedo");
							mul->rhs = albedoH;
						}
					}
					const auto fresnelH = forestPool.emplace<CFrontendIR::CFresnel>();
					{
						auto* fresnel = forestPool.deref(fresnelH);
						spectral_var_t::SCreationParams<1> params = {};
						params.knots.params[0].scale = 1.5f;
						fresnel->orientedRealEta = forestPool.emplace<CFrontendIR::CSpectralVariable>(std::move(params));
					}
					// the delta layering should optimize out nicely due to the sampling property
					const auto transH = forest->createMul(forestPool.emplace<CFrontendIR::CDeltaTransmission>(),fresnelH);
					// can't attach a copy of the top layer because we'll have a cycle, also the BRDF needs to be on the other side
					const auto bottomH = forestPool.emplace<CFrontendIR::CLayer>();
					{
						auto* bottomLayer = forestPool.deref(bottomH);
						bottomLayer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Rough Coating Copy");
						// no brdf on the top of last layer, kill multiscattering
						bottomLayer->btdf = transH;
						// need the interface to be Air on the bottom, not Glass
						bottomLayer->brdfBottom = forest->reciprocate(dielectricH)._const_cast();
					}

					// twosided rough plastic
					{
						const auto rootH = forestPool.emplace<CFrontendIR::CLayer>();
						auto* topLayer = forestPool.deref(rootH);
						topLayer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Twosided Rough Plastic");

						topLayer->brdfTop = dielectricH;
						topLayer->btdf = transH;
						// no brdf on the bottom of first layer, kill multiscattering

						const auto diffuseH = forestPool.emplace<CFrontendIR::CLayer>();
						topLayer->coated = diffuseH;
						{
							auto* midLayer = forestPool.deref(diffuseH);
							midLayer->brdfTop = roughDiffuseH;
							// no transmission in the mid-layer, so the backend needs to decompose into separate front/back materials
							midLayer->brdfBottom = roughDiffuseH;
							midLayer->coated = bottomH;
						}
					
						ASSERT_VALUE(forest->addMaterial(rootH,logger),true,"Twosided Rough Plastic");
					}

					// Diffuse transmitter normalized to whole sphere
					const auto roughDiffTransH = forestPool.emplace<CFrontendIR::CMul>();
					{
						// normalize the Oren Nayar over the full sphere
						auto* mul = forestPool.deref(roughDiffTransH);
						mul->lhs = roughDiffuseH;
						{
							spectral_var_t::SCreationParams<1> params = {};
							params.knots.params[0].scale = 0.5f;
							mul->rhs = forestPool.emplace<CFrontendIR::CSpectralVariable>(std::move(params));
						}
					}

					// coated diffuse transmitter
					{
						const auto rootH = forestPool.emplace<CFrontendIR::CLayer>();
						auto* topLayer = forestPool.deref(rootH);
						topLayer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Coated Diffuse Transmitter");

						topLayer->brdfTop = dielectricH;
						topLayer->btdf = transH;
						// no brdf on the bottom of first layer, kill multiscattering

						const auto midH = forestPool.emplace<CFrontendIR::CLayer>();
						auto* midLayer = forestPool.deref(midH);
						{
							midLayer->brdfTop = roughDiffTransH;
							midLayer->btdf = roughDiffTransH;
							midLayer->brdfBottom = roughDiffTransH;

							// we could even have a BSDF with a different Roughness on the bottom layer!
							midLayer->coated = bottomH;
						}
						topLayer->coated = midH;
					
						ASSERT_VALUE(forest->addMaterial(rootH,logger),true,"Coated Diffuse Transmitter");
					}

					// same thing but with subsurface beer absorption
					{
						const auto rootH = forestPool.emplace<CFrontendIR::CLayer>();
						auto* topLayer = forestPool.deref(rootH);
						topLayer->debugInfo = forestPool.emplace<CNodePool::CDebugInfo>("Coated Diffuse Extinction Transmitter");
// TODO: triple check this example Material
						// we have a choice of where to stick the Beer Absorption:
						// - on the BTDF of the outside layer, means that it will be applied to the transmission so twice according to VdotN and LdotN
						// (but delta transmission makes special weight nodes behave in a special and only once because `L=-V` is forced in a single scattering)
						// - inner layer BRDF or BTDF but thats intractable for most compiler backends because the `L` and `V` in the internal layers are not trivially known
						//	 unless the previous layers are delta distributions (in which case we can equivalently hoist beer to the previous layer). 
						const auto beerH = forestPool.emplace<CFrontendIR::CBeer>();
						auto* beer = forestPool.deref(beerH);
						{
							spectral_var_t::SCreationParams<3> params = {};
							params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
							params.knots.params[0].scale = 0.3f;
							params.knots.params[1].scale = 0.9f;
							params.knots.params[2].scale = 0.7f;
							beer->perpTransmittance = forestPool.emplace<spectral_var_t>(std::move(params));
						}
						{
							spectral_var_t::SCreationParams<1> params = {};
							params.knots.params[0].scale = 1.f;
							beer->thickness = forestPool.emplace<spectral_var_t>(std::move(params));
						}

						topLayer->brdfTop = dielectricH;
						// simplest/recommended
						{
							const auto transAbsorbH = forestPool.emplace<CFrontendIR::CMul>();
							auto* transAbsorb = forestPool.deref(transAbsorbH);
							transAbsorb->lhs = transH;
							{
								transAbsorb->rhs = beerH;
							}
							topLayer->btdf = transAbsorbH;
						}

						const auto midH = forestPool.emplace<CFrontendIR::CLayer>();
						auto* midLayer = forestPool.deref(midH);
						{
							midLayer->brdfTop = roughDiffTransH;
							midLayer->btdf = roughDiffTransH;
							// making extra work for our canonicalizer
							{
								midLayer->brdfBottom = forest->createMul(roughDiffTransH,beerH);
							}

							// we could even have a BSDF with a different Fresnel and Roughness on the bottom layer!
							midLayer->coated = bottomH;
						}
						topLayer->coated = midH;
					
						ASSERT_VALUE(forest->addMaterial(rootH,logger),true,"Coated Diffuse Extinction Transmitter");
					}
				}

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
					CFrontendIR::SDotPrinter printer = {forest.get(),forest->getMaterials()};
					auto visualization = printer();
					// file write does not take an internal copy of pointer given, need to keep source alive till end
					IFile::success_t succ;
					file->write(succ,visualization.c_str(),0,visualization.size());
					succ.getBytesProcessed();
				}
			}

			// Frontend AST -> IR compilation
			{
				auto ir = CTrueIR::create({.composed={.blockSizeKBLog2=4}});
				auto& irPool = forest->getObjectPool();

				ASSERT_VALUE(ir->addMaterials(forest.get()),true,"Material IR adding");
			}

			// Reference Backend Codegen
			{
			}

			// Compilation from HLSL to SPIR-V just to make sure it works
			{
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