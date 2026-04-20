#ifndef _NBL_EXAMPLES_COMMON_CACHED_PIPELINE_STATE_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_CACHED_PIPELINE_STATE_HPP_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "nbl/asset/utils/ISPIRVEntryPointTrimmer.h"

#include <array>
#include <chrono>
#include <deque>
#include <future>
#include <mutex>

namespace nbl::examples::common
{
using pipeline_future_t = std::future<core::smart_refctd_ptr<video::IGPUComputePipeline>>;

// TODO: WHY IS THIS EXAMPLE 31 SPECIFIC THING IN COMMON HEADERS !? !? !? 
template<size_t GeometryCount, size_t MethodCount>
struct SRenderPipelineStorage
{
	using shader_array_t = std::array<core::smart_refctd_ptr<asset::IShader>, GeometryCount>;
	using pipeline_method_array_t = std::array<core::smart_refctd_ptr<video::IGPUComputePipeline>, MethodCount>;
	using pipeline_future_method_array_t = std::array<pipeline_future_t, MethodCount>;
	using pipeline_array_t = std::array<pipeline_method_array_t, GeometryCount>;
	using pipeline_future_array_t = std::array<pipeline_future_method_array_t, GeometryCount>;

	// TODO: Binary Toggle Count is Semantically awful, use an enum + special count value
	constexpr static inline size_t BinaryToggleCount = 2;
	std::array<shader_array_t, BinaryToggleCount> shaders = {};
	std::array<pipeline_array_t, BinaryToggleCount> pipelines = {};
	std::array<pipeline_future_array_t, BinaryToggleCount> pendingPipelines = {};

	static constexpr size_t boolToIndex(const bool value)
	{
		return static_cast<size_t>(value);
	}

	shader_array_t& getShaders(const bool rwmc)
	{
		return shaders[boolToIndex(rwmc)];
	}

	const shader_array_t& getShaders(const bool rwmc) const
	{
		return shaders[boolToIndex(rwmc)];
	}

	pipeline_array_t& getPipelines(const bool rwmc)
	{
		return pipelines[boolToIndex(rwmc)];
	}

	const pipeline_array_t& getPipelines(const bool rwmc) const
	{
		return pipelines[boolToIndex(rwmc)];
	}

	pipeline_future_array_t& getPendingPipelines(const bool rwmc)
	{
		return pendingPipelines[boolToIndex(rwmc)];
	}

	const pipeline_future_array_t& getPendingPipelines(const bool rwmc) const
	{
		return pendingPipelines[boolToIndex(rwmc)];
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
	// TODO: WHY IS THIS EX 31 SPECIFIC THING IN COMMON?
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

// TODO: THIS IS ALL EX31 SPECIFIC, why is it here!?
template<size_t GeometryCount, size_t MethodCount>
inline size_t getRunningPipelineBuildCount(
	const SRenderPipelineStorage<GeometryCount, MethodCount>& renderStorage,
	const SResolvePipelineState& resolveState)
{
	size_t count = 0ull;
	for (const auto rwmc : { false, true })
	{
		const auto& futures = renderStorage.getPendingPipelines(rwmc);
		const auto& pipelines = renderStorage.getPipelines(rwmc);
		for (size_t geometry = 0ull; geometry < GeometryCount; ++geometry)
		{
			for (size_t method = 0ull; method < MethodCount; ++method)
			{
				if (futures[geometry][method].valid() && !pipelines[geometry][method])
					++count;
			}
		}
	}
	if (resolveState.pendingPipeline.valid() && !resolveState.pipeline)
		++count;
	return count;
}

template<size_t GeometryCount, size_t MethodCount>
inline size_t getReadyRenderPipelineCount(const SRenderPipelineStorage<GeometryCount, MethodCount>& renderStorage)
{
	size_t count = 0ull;
	for (const auto rwmc : { false, true })
	{
		const auto& pipelines = renderStorage.getPipelines(rwmc);
		for (const auto& perGeometry : pipelines)
		{
			for (const auto& pipeline : perGeometry)
			{
				if (pipeline)
					++count;
			}
		}
	}
	return count;
}

template<size_t GeometryCount, size_t MethodCount>
inline void pollPendingPipelines(
	SRenderPipelineStorage<GeometryCount, MethodCount>& renderStorage,
	SResolvePipelineState& resolveState,
	bool& dirty,
	size_t& newlyReadyPipelinesSinceLastSave)
{
	for (const auto rwmc : { false, true })
	{
		auto& pendingPipelines = renderStorage.getPendingPipelines(rwmc);
		auto& pipelines = renderStorage.getPipelines(rwmc);
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

	if (pollPendingPipeline(resolveState.pendingPipeline, resolveState.pipeline))
	{
		dirty = true;
		++newlyReadyPipelinesSinceLastSave;
	}
}

template<size_t GeometryCount, size_t MethodCount>
inline void waitForPendingPipelines(
	SRenderPipelineStorage<GeometryCount, MethodCount>& renderStorage,
	SResolvePipelineState& resolveState,
	bool& dirty,
	size_t& newlyReadyPipelinesSinceLastSave)
{
	for (const auto rwmc : { false, true })
	{
		auto& pendingPipelines = renderStorage.getPendingPipelines(rwmc);
		auto& pipelines = renderStorage.getPipelines(rwmc);
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

	if (waitForPendingPipeline(resolveState.pendingPipeline, resolveState.pipeline))
	{
		dirty = true;
		++newlyReadyPipelinesSinceLastSave;
	}
}
}

#endif
