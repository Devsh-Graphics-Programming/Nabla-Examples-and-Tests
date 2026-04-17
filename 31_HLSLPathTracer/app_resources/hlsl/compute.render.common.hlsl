#ifndef PATH_TRACER_USE_RWMC
#error PATH_TRACER_USE_RWMC must be defined before including compute.render.common.hlsl
#endif

#ifndef PATH_TRACER_ENABLE_LINEAR
#define PATH_TRACER_ENABLE_LINEAR 1
#endif

#ifndef PATH_TRACER_ENABLE_PERSISTENT
#define PATH_TRACER_ENABLE_PERSISTENT 1
#endif

#if !PATH_TRACER_ENABLE_LINEAR && !PATH_TRACER_ENABLE_PERSISTENT
#error At least one path tracer entrypoint mode must be enabled
#endif

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
//
#include "nbl/builtin/hlsl/random/pcg.hlsl"
#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"
//
#include "nbl/builtin/hlsl/morton.hlsl"
//
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"
//
#include "nbl/builtin/hlsl/path_tracing/basic_ray_gen.hlsl"
#include "nbl/builtin/hlsl/path_tracing/unidirectional.hlsl"

#include "nbl/examples/common/KeyedQuantizedSequence.hlsl"

#include "render_common.hlsl"

#if PATH_TRACER_USE_RWMC
#include "nbl/builtin/hlsl/rwmc/CascadeAccumulator.hlsl"
#include "render_rwmc_common.hlsl"
#else
#include "nbl/builtin/hlsl/path_tracing/default_accumulator.hlsl"
#endif

#if PATH_TRACER_USE_RWMC
[[vk::push_constant]] RenderRWMCPushConstants pc;
#else
[[vk::push_constant]] RenderPushConstants pc;
#endif

[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2D<float3> envMap;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState envSampler;

[[vk::combinedImageSampler]] [[vk::binding(1, 0)]] Texture2D<uint2> scramblebuf;
[[vk::combinedImageSampler]] [[vk::binding(1, 0)]] SamplerState scrambleSampler;

[[vk::image_format("rgba16f")]] [[vk::binding(2, 0)]] RWTexture2DArray<float32_t4> outImage;

#if PATH_TRACER_USE_RWMC
[[vk::image_format("rgba16f")]] [[vk::binding(3, 0)]] RWTexture2DArray<float32_t4> cascade;
#endif

[[vk::combinedImageSampler]] [[vk::binding(4, 0)]] Texture2D<float3> normalMap;
[[vk::combinedImageSampler]] [[vk::binding(4, 0)]] SamplerState normalSampler;

#include "example_common.hlsl"
#include "intersector.hlsl"
#include "material_system.hlsl"
#include "next_event_estimator.hlsl"

using namespace nbl;
using namespace hlsl;
