#include <nabla.h>

#include "nbl/examples/examples.hpp"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;
using namespace nbl::hlsl;
using namespace nbl::examples;

// sampling headers (HLSL/C++ compatible)
#include "nbl/builtin/hlsl/sampling/concentric_mapping.hlsl"
#include "nbl/builtin/hlsl/sampling/linear.hlsl"
#include "nbl/builtin/hlsl/sampling/bilinear.hlsl"
#include "nbl/builtin/hlsl/sampling/uniform_spheres.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#include "nbl/builtin/hlsl/sampling/box_muller_transform.hlsl"
#include "nbl/builtin/hlsl/sampling/spherical_triangle.hlsl"
#include "nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl"
#include "nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl"

// concepts header — include AFTER sampler headers, and only in the test
#include "nbl/builtin/hlsl/sampling/concepts.hlsl"

class HLSLSamplingTests final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t = BuiltinResourcesApplication;

public:
	HLSLSamplingTests(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;

		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		// test compile with dxc
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = "app_resources";
			auto key = nbl::this_example::builtin::build::get_spirv_key<"shader">(m_device.get());
			auto bundle = m_assetMgr->getAsset(key.c_str(), lp);

			const auto assets = bundle.getContents();
			if (assets.empty())
			{
				m_logger->log("Could not load shader!", ILogger::ELL_ERROR);
				return false;
			}

			auto shader = IAsset::castDown<IShader>(assets[0]);
			if (!shader)
			{
				m_logger->log("compile shader test failed!", ILogger::ELL_ERROR);
				return false;
			}

			m_logger->log("Shader compilation test passed.", ILogger::ELL_INFO);
		}

		// ================================================================
		// Compile-time concept verification via static_assert
		// ================================================================

		// --- BasicSampler (level 1) --- generate(domain_type) -> codomain_type
		static_assert(sampling::concepts::BasicSampler<sampling::Linear<float>>);
		static_assert(sampling::concepts::BasicSampler<sampling::UniformHemisphere<float>>);
		static_assert(sampling::concepts::BasicSampler<sampling::UniformSphere<float>>);
		static_assert(sampling::concepts::BasicSampler<sampling::ProjectedHemisphere<float>>);
		static_assert(sampling::concepts::BasicSampler<sampling::ProjectedSphere<float>>);

		// TODO: remaining samplers need generate(domain_type)->codomain_type overload to satisfy BasicSampler:
		//   Bilinear					- generate takes (rcpPdf, u), needs single-arg overload
		//   SphericalTriangle			- generate takes (rcpPdf, u), needs single-arg overload
		//   SphericalRectangle			- generate takes (extents, uv, S), needs single-arg overload
		//   BoxMullerTransform			- uses operator() instead of generate
		//   ProjectedSphericalTriangle - generate takes (rcpPdf, normal, isBSDF, u)
		//   Concentric_mapping			- is a free function, needs to be wrapped in a struct with concept type aliases

		// TODO: higher-level concepts require method refactoring:
		// --- BackwardDensitySampler (level 3) --- needs generate(domain_type)->sample_type, forwardPdf, backwardPdf
		// static_assert(sampling::concepts::BackwardDensitySampler<sampling::SphericalRectangle<float>>);
		// static_assert(sampling::concepts::BackwardDensitySampler<sampling::SphericalTriangle<float>>);
		// static_assert(sampling::concepts::BackwardDensitySampler<sampling::BoxMullerTransform<float>>);
		// static_assert(sampling::concepts::BackwardDensitySampler<sampling::ProjectedSphericalTriangle<float>>);
		// --- BijectiveSampler (level 4) --- needs + invertGenerate(codomain_type)->inverse_sample_type
		// static_assert(sampling::concepts::BijectiveSampler<sampling::Linear<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::Bilinear<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::UniformHemisphere<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::UniformSphere<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::ProjectedHemisphere<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::ProjectedSphere<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::SphericalTriangle<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::ConcentricMapping<float>>);



		m_logger->log("All sampling concept tests passed.", ILogger::ELL_INFO);
		return true;
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }

	bool onAppTerminated() override
	{
		return device_base_t::onAppTerminated();
	}
};

NBL_MAIN_FUNC(HLSLSamplingTests)
