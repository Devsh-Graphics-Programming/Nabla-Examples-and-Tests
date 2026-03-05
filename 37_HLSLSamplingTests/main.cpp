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

// ITester-based testers
#include "CLinearTester.h"
#include "CUniformHemisphereTester.h"
#include "CUniformSphereTester.h"
#include "CProjectedHemisphereTester.h"
#include "CProjectedSphereTester.h"
#include "CSphericalTriangleJacobianTester.h"

#include "CSamplerBenchmark.h"

constexpr bool DoBenchmark = true;

class HLSLSamplingTests final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t = BuiltinResourcesApplication;

	// Helper to create pipeline setup data
	template<typename Tester>
	auto createSetupData(const std::string& shaderKey) -> typename Tester::PipelineSetupData
	{
		typename Tester::PipelineSetupData data;
		data.device = m_device;
		data.api = m_api;
		data.assetMgr = m_assetMgr;
		data.logger = m_logger;
		data.physicalDevice = m_physicalDevice;
		data.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
		data.shaderKey = shaderKey;
		return data;
	}

	CSamplerBenchmark::SetupData createBenchmarkSetupData(const std::string& shaderKey, uint32_t dispatchGroupCount, uint32_t samplesPerDispatch, size_t inputBufferBytes, size_t outputBufferBytes)
	{
		CSamplerBenchmark::SetupData data;
		data.device = m_device;
		data.api = m_api;
		data.assetMgr = m_assetMgr;
		data.logger = m_logger;
		data.physicalDevice = m_physicalDevice;
		data.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
		data.shaderKey = shaderKey;
		data.dispatchGroupCount = dispatchGroupCount;
		data.samplesPerDispatch = samplesPerDispatch;
		data.inputBufferBytes = inputBufferBytes;
		data.outputBufferBytes = outputBufferBytes;
		return data;
	}

		public:
	HLSLSamplingTests(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) : system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

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
		// --- InvertibleSampler (level 3) --- needs generate(domain_type)->sample_type, forwardPdf, backwardPdf
		//static_assert(sampling::concepts::InvertibleSampler<sampling::SphericalRectangle<float>>);
		//static_assert(sampling::concepts::InvertibleSampler<sampling::SphericalTriangle<float>>);
		// static_assert(sampling::concepts::InvertibleSampler<sampling::BoxMullerTransform<float>>);
		// static_assert(sampling::concepts::InvertibleSampler<sampling::ProjectedSphericalTriangle<float>>);
		// --- BijectiveSampler (level 4) --- needs + invertGenerate(codomain_type)->inverse_sample_type
		// static_assert(sampling::concepts::BijectiveSampler<sampling::Linear<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::Bilinear<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::UniformHemisphere<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::UniformSphere<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::ProjectedHemisphere<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::ProjectedSphere<float>>);
		//static_assert(sampling::concepts::BijectiveSampler<sampling::SphericalTriangle<float>>);
		// static_assert(sampling::concepts::BijectiveSampler<sampling::ConcentricMapping<float>>);

		m_logger->log("All sampling concept tests passed.", ILogger::ELL_INFO);

		// ================================================================
		// Runtime CPU/GPU comparison tests using ITester harness
		// ================================================================
		bool pass = true;
		const uint32_t workgroupSize = 64;
		const uint32_t testBatchCount = 64; // 64 * workgroupSize = 4096 tests per sampler


		// --- BasicSampler tests ---
		{
			m_logger->log("Running Linear sampler tests...", ILogger::ELL_INFO);
			auto data = createSetupData<CLinearTester>(nbl::this_example::builtin::build::get_spirv_key<"linear_test">(m_device.get()));
			CLinearTester tester(testBatchCount, workgroupSize);
			tester.setupPipeline(data);
			pass &= tester.performTestsAndVerifyResults("LinearTestLog.txt");
		}
		{
			m_logger->log("Running UniformHemisphere sampler tests...", ILogger::ELL_INFO);
			auto data = createSetupData<CUniformHemisphereTester>(nbl::this_example::builtin::build::get_spirv_key<"uniform_hemisphere_test">(m_device.get()));
			CUniformHemisphereTester tester(testBatchCount, workgroupSize);
			tester.setupPipeline(data);
			pass &= tester.performTestsAndVerifyResults("UniformHemisphereTestLog.txt");
		}
		{
			m_logger->log("Running UniformSphere sampler tests...", ILogger::ELL_INFO);
			auto data = createSetupData<CUniformSphereTester>(nbl::this_example::builtin::build::get_spirv_key<"uniform_sphere_test">(m_device.get()));
			CUniformSphereTester tester(testBatchCount, workgroupSize);
			tester.setupPipeline(data);
			pass &= tester.performTestsAndVerifyResults("UniformSphereTestLog.txt");
		}
		{
			m_logger->log("Running ProjectedHemisphere sampler tests...", ILogger::ELL_INFO);
			auto data = createSetupData<CProjectedHemisphereTester>(nbl::this_example::builtin::build::get_spirv_key<"projected_hemisphere_test">(m_device.get()));
			CProjectedHemisphereTester tester(testBatchCount, workgroupSize);
			tester.setupPipeline(data);
			pass &= tester.performTestsAndVerifyResults("ProjectedHemisphereTestLog.txt");
		}
		{
			m_logger->log("Running ProjectedSphere sampler tests...", ILogger::ELL_INFO);
			auto data = createSetupData<CProjectedSphereTester>(nbl::this_example::builtin::build::get_spirv_key<"projected_sphere_test">(m_device.get()));
			CProjectedSphereTester tester(testBatchCount, workgroupSize);
			tester.setupPipeline(data);
			pass &= tester.performTestsAndVerifyResults("ProjectedSphereTestLog.txt");
		}

		// --- Jacobian tests for bijective samplers ---
		{
			m_logger->log("Running SphericalTriangle Jacobian tests...", ILogger::ELL_INFO);
			auto data = createSetupData<CSphericalTriangleJacobianTester>(nbl::this_example::builtin::build::get_spirv_key<"spherical_triangle_jacobian">(m_device.get()));
			CSphericalTriangleJacobianTester tester(testBatchCount, workgroupSize);
			tester.setupPipeline(data);
			pass &= tester.performTestsAndVerifyResults("SphericalTriangleJacobianTestLog.txt");
		}

		if (pass)
			m_logger->log("All sampling tests PASSED.", ILogger::ELL_INFO);
		else
			m_logger->log("Some sampling tests FAILED. Check log files for details.", ILogger::ELL_ERROR);

		// ======================================================================
		// GPU throughput benchmarks (1000 warmup + 20000 timed dispatches each)
		// ======================================================================
		if constexpr (DoBenchmark)
		{
			m_logger->log("=== GPU Sampler Benchmarks ===", ILogger::ELL_PERFORMANCE);
			constexpr uint32_t totalItems = testBatchCount * workgroupSize;
			constexpr uint32_t benchIters = 4096; // internal to shader, set in CMakeLists.txt
			constexpr uint32_t benchSamplesPerDispatch = totalItems * benchIters;

			{
				CSamplerBenchmark bench;
				bench.setup(createBenchmarkSetupData(
					nbl::this_example::builtin::build::get_spirv_key<"linear_bench">(m_device.get()),
					testBatchCount, benchSamplesPerDispatch,
					sizeof(LinearInputValues) * totalItems,
					sizeof(LinearTestResults) * totalItems));
				bench.run("Linear");
			}
			{
				CSamplerBenchmark bench;
				bench.setup(createBenchmarkSetupData(
					nbl::this_example::builtin::build::get_spirv_key<"uniform_hemisphere_bench">(m_device.get()),
					testBatchCount, benchSamplesPerDispatch,
					sizeof(UniformHemisphereInputValues) * totalItems,
					sizeof(UniformHemisphereTestResults) * totalItems));
				bench.run("UniformHemisphere");
			}
			{
				CSamplerBenchmark bench;
				bench.setup(createBenchmarkSetupData(
					nbl::this_example::builtin::build::get_spirv_key<"uniform_sphere_bench">(m_device.get()),
					testBatchCount, benchSamplesPerDispatch,
					sizeof(UniformSphereInputValues) * totalItems,
					sizeof(UniformSphereTestResults) * totalItems));
				bench.run("UniformSphere");
			}
			{
				CSamplerBenchmark bench;
				bench.setup(createBenchmarkSetupData(
					nbl::this_example::builtin::build::get_spirv_key<"projected_hemisphere_bench">(m_device.get()),
					testBatchCount, benchSamplesPerDispatch,
					sizeof(ProjectedHemisphereInputValues) * totalItems,
					sizeof(ProjectedHemisphereTestResults) * totalItems));
				bench.run("ProjectedHemisphere");
			}
			{
				CSamplerBenchmark bench;
				bench.setup(createBenchmarkSetupData(
					nbl::this_example::builtin::build::get_spirv_key<"projected_sphere_bench">(m_device.get()),
					testBatchCount, benchSamplesPerDispatch,
					sizeof(ProjectedSphereInputValues) * totalItems,
					sizeof(ProjectedSphereTestResults) * totalItems));
				bench.run("ProjectedSphere");
			}
			{
				CSamplerBenchmark bench;
				bench.setup(createBenchmarkSetupData(
					nbl::this_example::builtin::build::get_spirv_key<"spherical_triangle_bench">(m_device.get()),
					testBatchCount, benchSamplesPerDispatch,
					sizeof(SphericalTriangleJacobianInputValues) * totalItems,
					sizeof(SphericalTriangleJacobianTestResults) * totalItems));
				bench.run("SphericalTriangle");
			}
		}

		return pass;
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }

	bool onAppTerminated() override
	{
		return device_base_t::onAppTerminated();
	}
};

NBL_MAIN_FUNC(HLSLSamplingTests)
