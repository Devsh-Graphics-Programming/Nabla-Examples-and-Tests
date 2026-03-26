#ifndef __NBL_THIS_EXAMPLE_PATH_TRACER_PIPELINE_STATE_HPP_INCLUDED__
#define __NBL_THIS_EXAMPLE_PATH_TRACER_PIPELINE_STATE_HPP_INCLUDED__

#include "nbl/this_example/common.hpp"
#include "nbl/this_example/render_variant_enums.hlsl"

#include "nbl/asset/utils/ISPIRVEntryPointTrimmer.h"

#include <array>
#include <chrono>
#include <deque>
#include <future>
#include <mutex>
#include <optional>

namespace nbl::this_example
{
using pipeline_future_t = std::future<core::smart_refctd_ptr<video::IGPUComputePipeline>>;
using shader_array_t = std::array<core::smart_refctd_ptr<asset::IShader>, E_LIGHT_GEOMETRY::ELG_COUNT>;
using pipeline_method_array_t = std::array<core::smart_refctd_ptr<video::IGPUComputePipeline>, EPM_COUNT>;
using pipeline_future_method_array_t = std::array<pipeline_future_t, EPM_COUNT>;
using pipeline_array_t = std::array<pipeline_method_array_t, E_LIGHT_GEOMETRY::ELG_COUNT>;
using pipeline_future_array_t = std::array<pipeline_future_method_array_t, E_LIGHT_GEOMETRY::ELG_COUNT>;

template<size_t BinaryToggleCount>
struct SRenderPipelineStorage
{
	std::array<std::array<shader_array_t, BinaryToggleCount>, BinaryToggleCount> shaders = {};
	std::array<std::array<pipeline_array_t, BinaryToggleCount>, BinaryToggleCount> pipelines = {};
	std::array<std::array<pipeline_future_array_t, BinaryToggleCount>, BinaryToggleCount> pendingPipelines = {};

	static constexpr size_t boolToIndex(const bool value)
	{
		return static_cast<size_t>(value);
	}

	shader_array_t& getShaders(const bool persistentWorkGroups, const bool rwmc)
	{
		return shaders[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
	}

	const shader_array_t& getShaders(const bool persistentWorkGroups, const bool rwmc) const
	{
		return shaders[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
	}

	pipeline_array_t& getPipelines(const bool persistentWorkGroups, const bool rwmc)
	{
		return pipelines[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
	}

	const pipeline_array_t& getPipelines(const bool persistentWorkGroups, const bool rwmc) const
	{
		return pipelines[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
	}

	pipeline_future_array_t& getPendingPipelines(const bool persistentWorkGroups, const bool rwmc)
	{
		return pendingPipelines[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
	}

	const pipeline_future_array_t& getPendingPipelines(const bool persistentWorkGroups, const bool rwmc) const
	{
		return pendingPipelines[boolToIndex(rwmc)][boolToIndex(persistentWorkGroups)];
	}
};

struct SResolvePipelineState
{
	core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
	core::smart_refctd_ptr<asset::IShader> shader;
	core::smart_refctd_ptr<video::IGPUComputePipeline> pipeline;
	pipeline_future_t pendingPipeline;
};

struct SWarmupJob
{
	enum class E_TYPE : uint8_t
	{
		Render,
		Resolve
	};

	E_TYPE type = E_TYPE::Render;
	E_LIGHT_GEOMETRY geometry = ELG_SPHERE;
	bool persistentWorkGroups = false;
	bool rwmc = false;
	E_POLYGON_METHOD polygonMethod = EPM_PROJECTED_SOLID_ANGLE;
};

struct SPipelineCacheState
{
	struct STrimmedShaderCache
	{
		core::smart_refctd_ptr<asset::ISPIRVEntryPointTrimmer> trimmer;
		system::path rootDir;
		system::path validationDir;
		size_t loadedFromDiskCount = 0ull;
		size_t generatedCount = 0ull;
		size_t savedToDiskCount = 0ull;
		size_t loadedBytes = 0ull;
		size_t savedBytes = 0ull;
		core::unordered_map<std::string, core::smart_refctd_ptr<asset::IShader>> runtimeShaders;
		std::mutex mutex;
	} trimmedShaders;

	struct SWarmupState
	{
		bool started = false;
		bool loggedComplete = false;
		std::chrono::steady_clock::time_point beganAt = std::chrono::steady_clock::now();
		size_t budget = 1ull;
		size_t queuedJobs = 0ull;
		size_t launchedJobs = 0ull;
		size_t skippedJobs = 0ull;
		std::deque<SWarmupJob> queue;
	} warmup;

	core::smart_refctd_ptr<video::IGPUPipelineCache> object;
	system::path blobPath;
	bool dirty = false;
	bool loadedFromDisk = false;
	bool clearedOnStartup = false;
	size_t loadedBytes = 0ull;
	size_t savedBytes = 0ull;
	size_t newlyReadyPipelinesSinceLastSave = 0ull;
	bool checkpointedAfterFirstSubmit = false;
	std::chrono::steady_clock::time_point lastSaveAt = std::chrono::steady_clock::now();
};

struct SStartupLogState
{
	bool hasPathtraceOutput = false;
	bool loggedFirstFrameLoop = false;
	bool loggedFirstRenderDispatch = false;
	bool loggedFirstRenderSubmit = false;
};
}

#endif
