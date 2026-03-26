#ifndef __NBL_THIS_EXAMPLE_PATH_TRACER_PIPELINE_STATE_HPP_INCLUDED__
#define __NBL_THIS_EXAMPLE_PATH_TRACER_PIPELINE_STATE_HPP_INCLUDED__

#include "nbl/examples/path_tracing/CachedPipelineState.hpp"
#include "nbl/this_example/render_variant_enums.hlsl"

namespace nbl::this_example
{
using pipeline_future_t = examples::path_tracing::pipeline_future_t;
using path_tracing_storage_traits_t = examples::path_tracing::SRenderPipelineStorage<ELG_COUNT, EPM_COUNT, 2ull>;
using shader_array_t = typename path_tracing_storage_traits_t::shader_array_t;
using pipeline_method_array_t = typename path_tracing_storage_traits_t::pipeline_method_array_t;
using pipeline_future_method_array_t = typename path_tracing_storage_traits_t::pipeline_future_method_array_t;
using pipeline_array_t = typename path_tracing_storage_traits_t::pipeline_array_t;
using pipeline_future_array_t = typename path_tracing_storage_traits_t::pipeline_future_array_t;

template<size_t BinaryToggleCount>
using SRenderPipelineStorage = examples::path_tracing::SRenderPipelineStorage<ELG_COUNT, EPM_COUNT, BinaryToggleCount>;

using SResolvePipelineState = examples::path_tracing::SResolvePipelineState;
using SWarmupJob = examples::path_tracing::SWarmupJob<E_LIGHT_GEOMETRY, E_POLYGON_METHOD>;
using SPipelineCacheState = examples::path_tracing::SPipelineCacheState<SWarmupJob>;
using SStartupLogState = examples::path_tracing::SStartupLogState;
}

#endif
