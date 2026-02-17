#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/random/pcg.hlsl"
#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"
#ifdef PERSISTENT_WORKGROUPS
#include "nbl/builtin/hlsl/morton.hlsl"
#endif

#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"

#include "nbl/builtin/hlsl/path_tracing/basic_ray_gen.hlsl"
#include "nbl/builtin/hlsl/path_tracing/unidirectional.hlsl"

// add these defines (one at a time) using -D argument to dxc
// #define SPHERE_LIGHT
// #define TRIANGLE_LIGHT
// #define RECTANGLE_LIGHT

#include "render_common.hlsl"
#include "resolve_common.hlsl"

#ifdef RWMC_ENABLED
#include <nbl/builtin/hlsl/rwmc/CascadeAccumulator.hlsl>
#include <render_rwmc_common.hlsl>
#endif

#ifdef RWMC_ENABLED
[[vk::push_constant]] RenderRWMCPushConstants pc;
#else
[[vk::push_constant]] RenderPushConstants pc;
#endif

[[vk::combinedImageSampler]] [[vk::binding(0, 2)]] Texture2D<float3> envMap;      // unused
[[vk::combinedImageSampler]] [[vk::binding(0, 2)]] SamplerState envSampler;

[[vk::combinedImageSampler]] [[vk::binding(2, 2)]] Texture2D<uint2> scramblebuf;
[[vk::combinedImageSampler]] [[vk::binding(2, 2)]] SamplerState scrambleSampler;

[[vk::image_format("rgba16f")]] [[vk::binding(0)]] RWTexture2DArray<float32_t4> outImage;
[[vk::image_format("rgba16f")]] [[vk::binding(1)]] RWTexture2DArray<float32_t4> cascade;

#include "example_common.hlsl"
#include "scene.hlsl"
#include "rand_gen.hlsl"
#include "intersector.hlsl"
#include "material_system.hlsl"
#include "next_event_estimator.hlsl"

using namespace nbl;
using namespace hlsl;

#ifdef SPHERE_LIGHT
NBL_CONSTEXPR ProceduralShapeType LIGHT_TYPE = PST_SPHERE;
#endif
#ifdef TRIANGLE_LIGHT
NBL_CONSTEXPR ProceduralShapeType LIGHT_TYPE = PST_TRIANGLE;
#endif
#ifdef RECTANGLE_LIGHT
NBL_CONSTEXPR ProceduralShapeType LIGHT_TYPE = PST_RECTANGLE;
#endif

NBL_CONSTEXPR PTPolygonMethod POLYGON_METHOD = PPM_SOLID_ANGLE;

int32_t2 getCoordinates()
{
    uint32_t width, height, imageArraySize;
    outImage.GetDimensions(width, height, imageArraySize);
    return int32_t2(glsl::gl_GlobalInvocationID().x % width, glsl::gl_GlobalInvocationID().x / width);
}

float32_t2 getTexCoords()
{
    uint32_t width, height, imageArraySize;
    outImage.GetDimensions(width, height, imageArraySize);
    int32_t2 iCoords = getCoordinates();
    return float32_t2(float(iCoords.x) / width, 1.0 - float(iCoords.y) / height);
}

using spectral_t = vector<float, 3>;
using ray_dir_info_t = bxdf::ray_dir_info::SBasic<float>;
using iso_interaction = bxdf::surface_interactions::SIsotropic<ray_dir_info_t, spectral_t>;
using aniso_interaction = bxdf::surface_interactions::SAnisotropic<iso_interaction>;
using sample_t = bxdf::SLightSample<ray_dir_info_t>;
using iso_cache = bxdf::SIsotropicMicrofacetCache<float>;
using aniso_cache = bxdf::SAnisotropicMicrofacetCache<iso_cache>;
using quotient_pdf_t = sampling::quotient_and_pdf<float32_t3, float>;

using iso_config_t = bxdf::SConfiguration<sample_t, iso_interaction, spectral_t>;
using iso_microfacet_config_t = bxdf::SMicrofacetConfiguration<sample_t, iso_interaction, iso_cache, spectral_t>;

using diffuse_bxdf_type = bxdf::reflection::SOrenNayar<iso_config_t>;
using conductor_bxdf_type = bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>;
using dielectric_bxdf_type = bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>;
using iri_conductor_bxdf_type = bxdf::reflection::SIridescent<iso_microfacet_config_t>;
using iri_dielectric_bxdf_type = bxdf::transmission::SIridescent<iso_microfacet_config_t>;

using ray_type = Ray<float>;
using light_type = Light<spectral_t>;
using bxdfnode_type = BxDFNode<spectral_t>;
using scene_type = Scene<LIGHT_TYPE>;
using randgen_type = RandGen::UniformND<Xoroshiro64Star,3>;
using raygen_type = path_tracing::BasicRayGenerator<ray_type>;
using intersector_type = Intersector<ray_type, scene_type>;
using material_system_type = MaterialSystem<bxdfnode_type, diffuse_bxdf_type, conductor_bxdf_type, dielectric_bxdf_type, iri_conductor_bxdf_type, iri_dielectric_bxdf_type, scene_type>;
using nee_type = NextEventEstimator<scene_type, light_type, ray_type, sample_t, aniso_interaction, IM_PROCEDURAL, LIGHT_TYPE, POLYGON_METHOD>;

#ifdef RWMC_ENABLED
using accumulator_type = rwmc::CascadeAccumulator<float32_t3, CascadeCount>;
#else
#include "nbl/builtin/hlsl/path_tracing/default_accumulator.hlsl"
using accumulator_type = path_tracing::DefaultAccumulator<float32_t3>;
#endif

using pathtracer_type = path_tracing::Unidirectional<randgen_type, raygen_type, intersector_type, material_system_type, nee_type, accumulator_type, scene_type>;

#ifdef SPHERE_LIGHT
static const Shape<float, PST_SPHERE> spheres[scene_type::SCENE_LIGHT_COUNT] = {
    Shape<float, PST_SPHERE>::create(float3(-1.5, 1.5, 0.0), 0.3, bxdfnode_type::INVALID_ID, 0u)
};
#endif

#ifdef TRIANGLE_LIGHT
static const Shape<float, PST_TRIANGLE> triangles[scene_type::SCENE_LIGHT_COUNT] = {
    Shape<float, PST_TRIANGLE>::create(float3(-1.8,0.35,0.3) * 10.0, float3(-1.2,0.35,0.0) * 10.0, float3(-1.5,0.8,-0.3) * 10.0, bxdfnode_type::INVALID_ID, 0u)
};
#endif

#ifdef RECTANGLE_LIGHT
static const Shape<float, PST_RECTANGLE> rectangles[scene_type::SCENE_LIGHT_COUNT] = {
    Shape<float, PST_RECTANGLE>::create(float3(-3.8,0.35,1.3), normalize(float3(2,0,-1))*7.0, normalize(float3(2,-5,4))*0.1, bxdfnode_type::INVALID_ID, 0u)
};
#endif

static const light_type lights[scene_type::SCENE_LIGHT_COUNT] = {
    light_type::create(LightEminence,
#ifdef SPHERE_LIGHT
        scene_type::SCENE_SPHERE_COUNT,
#else
        0u,
#endif
        IM_PROCEDURAL, LIGHT_TYPE)
};

static const bxdfnode_type bxdfs[scene_type::SCENE_BXDF_COUNT] = {
    bxdfnode_type::create(MaterialType::DIFFUSE, false, float2(0,0), spectral_t(0.8,0.8,0.8)),
    bxdfnode_type::create(MaterialType::DIFFUSE, false, float2(0,0), spectral_t(0.8,0.4,0.4)),
    bxdfnode_type::create(MaterialType::DIFFUSE, false, float2(0,0), spectral_t(0.4,0.8,0.4)),
    bxdfnode_type::create(MaterialType::CONDUCTOR, false, float2(0,0), spectral_t(1.02,1.02,1.3), spectral_t(1.0,1.0,2.0)),
    bxdfnode_type::create(MaterialType::CONDUCTOR, false, float2(0,0), spectral_t(1.02,1.3,1.02), spectral_t(1.0,2.0,1.0)),
    bxdfnode_type::create(MaterialType::CONDUCTOR, false, float2(0.15,0.15), spectral_t(1.02,1.3,1.02), spectral_t(1.0,2.0,1.0)),
    bxdfnode_type::create(MaterialType::DIELECTRIC, false, float2(0.0625,0.0625), spectral_t(1,1,1), spectral_t(1.4,1.45,1.5)),
    bxdfnode_type::create(MaterialType::IRIDESCENT_CONDUCTOR, false, 0.0, 505.0, spectral_t(1.39,1.39,1.39), spectral_t(1.2,1.2,1.2), spectral_t(0.5,0.5,0.5)),
    bxdfnode_type::create(MaterialType::IRIDESCENT_DIELECTRIC, false, 0.0, 400.0, spectral_t(1.7,1.7,1.7), spectral_t(1.0,1.0,1.0), spectral_t(0,0,0))
};

RenderPushConstants retireveRenderPushConstants()
{
#ifdef RWMC_ENABLED
    return pc.renderPushConstants;
#else
    return pc;
#endif
}

[numthreads(RenderWorkgroupSize, 1, 1)]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
    const RenderPushConstants renderPushConstants = retireveRenderPushConstants();

    uint32_t width, height, imageArraySize;
    outImage.GetDimensions(width, height, imageArraySize);
#ifdef PERSISTENT_WORKGROUPS
    const uint32_t NumWorkgroupsX = width / RenderWorkgroupSizeSqrt;
    const uint32_t NumWorkgroupsY = height / RenderWorkgroupSizeSqrt;
    [loop]
    for (uint32_t wgBase = glsl::gl_WorkGroupID().x; wgBase < NumWorkgroupsX*NumWorkgroupsY; wgBase += glsl::gl_NumWorkGroups().x)
    {
        const int32_t2 wgCoords = int32_t2(wgBase % NumWorkgroupsX, wgBase / NumWorkgroupsX);
        morton::code<true, 32, 2> mc;
        mc.value = glsl::gl_LocalInvocationIndex().x;
        const int32_t2 localCoords = _static_cast<int32_t2>(mc);
        const int32_t2 coords = wgCoords * int32_t2(RenderWorkgroupSizeSqrt,RenderWorkgroupSizeSqrt) + localCoords;
#else
    const int32_t2 coords = getCoordinates();
#endif
    float32_t2 texCoord = float32_t2(coords) / float32_t2(width, height);
    texCoord.y = 1.0 - texCoord.y;

    if (any(coords < int32_t2(0,0)) || any(coords >= int32_t2(width, height))) {
#ifdef PERSISTENT_WORKGROUPS
        continue;
#else
        return;
#endif
    }

    if (((renderPushConstants.depth - 1) >> MaxDepthLog2) > 0 || ((renderPushConstants.sampleCount - 1) >> MaxSamplesLog2) > 0)
    {
        float32_t4 pixelCol = float32_t4(1.0,0.0,0.0,1.0);
        outImage[uint3(coords.x, coords.y, 0)] = pixelCol;
#ifdef PERSISTENT_WORKGROUPS
        continue;
#else
        return;
#endif
    }

    // set up scene
    scene_type scene;
#ifdef SPHERE_LIGHT
    scene.light_spheres[0] = spheres[0];
#endif
#ifdef TRIANGLE_LIGHT
    scene.light_triangles[0] = triangles[0];
#endif
#ifdef RECTANGLE_LIGHT
    scene.light_rectangles[0] = rectangles[0];
#endif

    // set up path tracer
    pathtracer_type pathtracer;
    pathtracer.randGen = randgen_type::create(scramblebuf[coords].rg);

    uint2 scrambleDim;
    scramblebuf.GetDimensions(scrambleDim.x, scrambleDim.y);
    float32_t2 pixOffsetParam = (float2)1.0 / float2(scrambleDim);

    float32_t4 NDC = float4(texCoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    float32_t3 camPos;
    {
        float4 tmp = mul(renderPushConstants.invMVP, NDC);
        camPos = tmp.xyz / tmp.w;
        NDC.z = 1.0;
    }
    
    scene.updateLight(renderPushConstants.generalPurposeLightMatrix);
    pathtracer.rayGen.pixOffsetParam = pixOffsetParam; 
    pathtracer.rayGen.camPos = camPos;
    pathtracer.rayGen.NDC = NDC;
    pathtracer.rayGen.invMVP = renderPushConstants.invMVP;
    pathtracer.nee.lights = lights;
    pathtracer.nee.lightCount = scene_type::SCENE_LIGHT_COUNT;
    pathtracer.materialSystem.bxdfs = bxdfs;
    pathtracer.materialSystem.bxdfCount = scene_type::SCENE_BXDF_COUNT;
    pathtracer.pSampleBuffer = renderPushConstants.pSampleSequence;

#ifdef RWMC_ENABLED
    accumulator_type accumulator = accumulator_type::create(pc.getSplattingParams(CascadeCount));
#else
    accumulator_type accumulator = accumulator_type::create();
#endif
    // path tracing loop
    for(int i = 0; i < renderPushConstants.sampleCount; ++i)
        pathtracer.sampleMeasure(i, renderPushConstants.depth, scene, accumulator);

#ifdef RWMC_ENABLED
    for (uint32_t i = 0; i < CascadeCount; ++i)
        cascade[uint3(coords.x, coords.y, i)] = float32_t4(accumulator.accumulation.data[i], 1.0f);
#else
    outImage[uint3(coords.x, coords.y, 0)] = float32_t4(accumulator.accumulation, 1.0);
#endif

#ifdef PERSISTENT_WORKGROUPS
    }
#endif
}