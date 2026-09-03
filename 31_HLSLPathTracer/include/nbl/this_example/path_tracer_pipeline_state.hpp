#ifndef _NBL_THIS_EXAMPLE_PATH_TRACER_PIPELINE_STATE_HPP_INCLUDED_
#define _NBL_THIS_EXAMPLE_PATH_TRACER_PIPELINE_STATE_HPP_INCLUDED_


#include "nbl/examples/common/CachedPipelineState.hpp"
#include "nbl/this_example/render_variant_enums.hlsl"


namespace nbl::this_example
{
using pipeline_future_t = examples::common::pipeline_future_t;
using cached_pipeline_storage_traits_t = examples::common::SRenderPipelineStorage<ELG_COUNT, EPM_COUNT>;
using shader_array_t = typename cached_pipeline_storage_traits_t::shader_array_t;
using pipeline_method_array_t = typename cached_pipeline_storage_traits_t::pipeline_method_array_t;
using pipeline_future_method_array_t = typename cached_pipeline_storage_traits_t::pipeline_future_method_array_t;
using pipeline_array_t = typename cached_pipeline_storage_traits_t::pipeline_array_t;
using pipeline_future_array_t = typename cached_pipeline_storage_traits_t::pipeline_future_array_t;

using SRenderPipelineStorage = examples::common::SRenderPipelineStorage<ELG_COUNT, EPM_COUNT>;

using SResolvePipelineState = examples::common::SResolvePipelineState;
using SWarmupJob = examples::common::SWarmupJob<E_LIGHT_GEOMETRY, E_POLYGON_METHOD>;
using SPipelineCacheState = examples::common::SPipelineCacheState<SWarmupJob>;
using SStartupLogState = examples::common::SStartupLogState;
}

#endif
