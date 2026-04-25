// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include <nabla.h>

#include "nbl/examples/examples.hpp"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "benchmarks/CSobolBenchmark.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::video;
using namespace nbl::examples;


class SobolBenchmarkApp final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t  = BuiltinResourcesApplication;

public:
	SobolBenchmarkApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
	{
		auto retval                   = device_base_t::getPreferredDeviceFeatures();
		retval.pipelineExecutableInfo = true;
		return retval;
	}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		// Smaller dispatch than 37's light-sampler bench since each thread does Depth*Triplets*Components
		// matrix-muls per outer iteration (so the inner work is much heavier per "sample").
		constexpr uint32_t testBatchCount         = 256;
		constexpr uint32_t benchWorkgroupSize     = WORKGROUP_SIZE;
		constexpr uint32_t totalThreadsPerDispatch = testBatchCount * benchWorkgroupSize;
		constexpr uint32_t iterationsPerThread    = BENCH_ITERS;
		constexpr uint32_t benchSamplesPerDispatch = totalThreadsPerDispatch * iterationsPerThread;

		// Bench shader doesn't read input; we still allocate a single uint32 to satisfy
		// the 2-binding descriptor layout the harness expects.
		constexpr size_t benchInputBytes  = sizeof(uint32_t);
		constexpr size_t benchOutputBytes = sizeof(uint32_t) * totalThreadsPerDispatch;

		struct BenchEntry
		{
			CSobolBenchmark bench;
			std::string     name;
		};
		std::vector<BenchEntry> benchmarks;

		auto addBench = [&](const char* name, const std::string& shaderKey)
		{
			auto& entry = benchmarks.emplace_back();
			entry.name  = name;

			CSobolBenchmark::SetupData data;
			data.device             = m_device;
			data.api                = m_api;
			data.assetMgr           = m_assetMgr;
			data.logger             = m_logger;
			data.physicalDevice     = m_physicalDevice;
			data.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
			data.shaderKey          = shaderKey;
			data.dispatchGroupCount = testBatchCount;
			data.samplesPerDispatch = benchSamplesPerDispatch;
			data.inputBufferBytes   = benchInputBytes;
			data.outputBufferBytes  = benchOutputBytes;
			entry.bench.setup(data);
		};

		addBench("Sobol RowMajor", nbl::this_example::builtin::build::get_spirv_key<"sobol_bench_row_major">(m_device.get()));
		addBench("Sobol ColMajor", nbl::this_example::builtin::build::get_spirv_key<"sobol_bench_col_major">(m_device.get()));

		for (auto& entry : benchmarks)
			entry.bench.logPipelineReport(entry.name);

		constexpr uint32_t warmupDispatches = 100;
		constexpr uint32_t benchDispatches  = 500;
		m_logger->log("=== GPU Sobol Benchmarks (%u dispatches, %u threads/dispatch, %u outer iters/thread, inner work = DEPTH * 6 matrix-muls) ===",
			ILogger::ELL_PERFORMANCE, benchDispatches, totalThreadsPerDispatch, iterationsPerThread);
		m_logger->log("            %-28s | %12s | %12s | %12s",
			ILogger::ELL_PERFORMANCE, "Variant", "ps/sample", "GSamples/s", "ms total");
		for (auto& entry : benchmarks)
			entry.bench.run(entry.name, warmupDispatches, benchDispatches);

		return true;
	}

	void onAppTerminated_impl() override
	{
		m_device->waitIdle();
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(SobolBenchmarkApp)
