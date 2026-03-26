#ifndef _NBL_EXAMPLES_PATH_TRACING_CACHED_PIPELINE_STATE_HPP_INCLUDED_
#define _NBL_EXAMPLES_PATH_TRACING_CACHED_PIPELINE_STATE_HPP_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "nbl/asset/utils/ISPIRVEntryPointTrimmer.h"

#include <array>
#include <chrono>
#include <deque>
#include <future>
#include <mutex>

namespace nbl::examples::path_tracing
{
using pipeline_future_t = std::future<core::smart_refctd_ptr<video::IGPUComputePipeline>>;

template<size_t GeometryCount, size_t MethodCount, size_t BinaryToggleCount = 2ull>
struct SRenderPipelineStorage
{
	using shader_array_t = std::array<core::smart_refctd_ptr<asset::IShader>, GeometryCount>;
	using pipeline_method_array_t = std::array<core::smart_refctd_ptr<video::IGPUComputePipeline>, MethodCount>;
	using pipeline_future_method_array_t = std::array<pipeline_future_t, MethodCount>;
	using pipeline_array_t = std::array<pipeline_method_array_t, GeometryCount>;
	using pipeline_future_array_t = std::array<pipeline_future_method_array_t, GeometryCount>;

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

template<typename GeometryType, typename MethodType>
struct SWarmupJob
{
	enum class E_TYPE : uint8_t
	{
		Render,
		Resolve
	};

	E_TYPE type = E_TYPE::Render;
	GeometryType geometry = {};
	bool persistentWorkGroups = false;
	bool rwmc = false;
	MethodType method = {};
};

template<typename WarmupJobType>
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
		std::deque<WarmupJobType> queue;
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

template<typename PipelineFuture, typename PipelinePtr>
inline bool pollPendingPipeline(PipelineFuture& future, PipelinePtr& pipeline)
{
	if (!future.valid() || pipeline)
		return false;
	if (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
		return false;
	pipeline = future.get();
	return static_cast<bool>(pipeline);
}

template<typename PipelineFuture, typename PipelinePtr>
inline bool waitForPendingPipeline(PipelineFuture& future, PipelinePtr& pipeline)
{
	if (!future.valid() || pipeline)
		return false;
	future.wait();
	pipeline = future.get();
	return static_cast<bool>(pipeline);
}

template<size_t GeometryCount, size_t MethodCount, size_t BinaryToggleCount>
inline size_t getRunningPipelineBuildCount(
	const SRenderPipelineStorage<GeometryCount, MethodCount, BinaryToggleCount>& renderStorage,
	const SResolvePipelineState& resolveState)
{
	size_t count = 0ull;
	for (const auto rwmc : { false, true })
	{
		for (const auto persistentWorkGroups : { false, true })
		{
			const auto& futures = renderStorage.getPendingPipelines(persistentWorkGroups, rwmc);
			const auto& pipelines = renderStorage.getPipelines(persistentWorkGroups, rwmc);
			for (size_t geometry = 0ull; geometry < GeometryCount; ++geometry)
			{
				for (size_t method = 0ull; method < MethodCount; ++method)
				{
					if (futures[geometry][method].valid() && !pipelines[geometry][method])
						++count;
				}
			}
		}
	}
	if (resolveState.pendingPipeline.valid() && !resolveState.pipeline)
		++count;
	return count;
}

template<size_t GeometryCount, size_t MethodCount, size_t BinaryToggleCount>
inline size_t getReadyRenderPipelineCount(const SRenderPipelineStorage<GeometryCount, MethodCount, BinaryToggleCount>& renderStorage)
{
	size_t count = 0ull;
	for (const auto rwmc : { false, true })
	{
		for (const auto persistentWorkGroups : { false, true })
		{
			const auto& pipelines = renderStorage.getPipelines(persistentWorkGroups, rwmc);
			for (const auto& perGeometry : pipelines)
			{
				for (const auto& pipeline : perGeometry)
				{
					if (pipeline)
						++count;
				}
			}
		}
	}
	return count;
}

template<size_t GeometryCount, size_t MethodCount, size_t BinaryToggleCount>
inline void pollPendingPipelines(
	SRenderPipelineStorage<GeometryCount, MethodCount, BinaryToggleCount>& renderStorage,
	SResolvePipelineState& resolveState,
	bool& dirty,
	size_t& newlyReadyPipelinesSinceLastSave)
{
	for (const auto rwmc : { false, true })
	{
		for (const auto persistentWorkGroups : { false, true })
		{
			auto& pendingPipelines = renderStorage.getPendingPipelines(persistentWorkGroups, rwmc);
			auto& pipelines = renderStorage.getPipelines(persistentWorkGroups, rwmc);
			for (size_t geometry = 0ull; geometry < GeometryCount; ++geometry)
			{
				for (size_t method = 0ull; method < MethodCount; ++method)
				{
					if (pollPendingPipeline(pendingPipelines[geometry][method], pipelines[geometry][method]))
					{
						dirty = true;
						++newlyReadyPipelinesSinceLastSave;
					}
				}
			}
		}
	}

	if (pollPendingPipeline(resolveState.pendingPipeline, resolveState.pipeline))
	{
		dirty = true;
		++newlyReadyPipelinesSinceLastSave;
	}
}

template<size_t GeometryCount, size_t MethodCount, size_t BinaryToggleCount>
inline void waitForPendingPipelines(
	SRenderPipelineStorage<GeometryCount, MethodCount, BinaryToggleCount>& renderStorage,
	SResolvePipelineState& resolveState,
	bool& dirty,
	size_t& newlyReadyPipelinesSinceLastSave)
{
	for (const auto rwmc : { false, true })
	{
		for (const auto persistentWorkGroups : { false, true })
		{
			auto& pendingPipelines = renderStorage.getPendingPipelines(persistentWorkGroups, rwmc);
			auto& pipelines = renderStorage.getPipelines(persistentWorkGroups, rwmc);
			for (size_t geometry = 0ull; geometry < GeometryCount; ++geometry)
			{
				for (size_t method = 0ull; method < MethodCount; ++method)
				{
					if (waitForPendingPipeline(pendingPipelines[geometry][method], pipelines[geometry][method]))
					{
						dirty = true;
						++newlyReadyPipelinesSinceLastSave;
					}
				}
			}
		}
	}

	if (waitForPendingPipeline(resolveState.pendingPipeline, resolveState.pipeline))
	{
		dirty = true;
		++newlyReadyPipelinesSinceLastSave;
	}
}
}

#endif
