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

[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2D<float3> envMap;      // unused
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState envSampler;

[[vk::combinedImageSampler]] [[vk::binding(1, 0)]] Texture2D<uint2> scramblebuf;
[[vk::combinedImageSampler]] [[vk::binding(1, 0)]] SamplerState scrambleSampler;

[[vk::image_format("rgba16f")]] [[vk::binding(2, 0)]] RWTexture2DArray<float32_t4> outImage;
[[vk::image_format("rgba16f")]] [[vk::binding(3, 0)]] RWTexture2DArray<float32_t4> cascade;

#include "example_common.hlsl"
#include "rand_gen.hlsl"
#include "intersector.hlsl"
#include "material_system.hlsl"
#include "next_event_estimator.hlsl"

using namespace nbl;
using namespace hlsl;

#ifdef SPHERE_LIGHT
#include "scene_sphere_light.hlsl"
#endif
#ifdef TRIANGLE_LIGHT
#include "scene_triangle_light.hlsl"
#endif
#ifdef RECTANGLE_LIGHT
#include "scene_rectangle_light.hlsl"
#endif

NBL_CONSTEXPR NEEPolygonMethod POLYGON_METHOD = PPM_SOLID_ANGLE;

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
using iso_interaction = PTIsotropicInteraction<ray_dir_info_t, spectral_t>;
using aniso_interaction = PTAnisotropicInteraction<iso_interaction>;
using sample_t = bxdf::SLightSample<ray_dir_info_t>;
using iso_cache = bxdf::SIsotropicMicrofacetCache<float>;
using aniso_cache = bxdf::SAnisotropicMicrofacetCache<iso_cache>;
using quotient_pdf_t = sampling::quotient_and_pdf<float32_t3, float>;

using iso_config_t = PTIsoConfiguration<sample_t, iso_interaction, spectral_t>;
using iso_microfacet_config_t = PTIsoMicrofacetConfiguration<sample_t, iso_interaction, iso_cache, spectral_t>;

using diffuse_bxdf_type = bxdf::reflection::SOrenNayar<iso_config_t>;
using conductor_bxdf_type = bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>;
using dielectric_bxdf_type = bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>;
using iri_conductor_bxdf_type = bxdf::reflection::SIridescent<iso_microfacet_config_t>;
using iri_dielectric_bxdf_type = bxdf::transmission::SIridescent<iso_microfacet_config_t>;

using payload_type = Payload<float>;
using ray_type = Ray<payload_type,POLYGON_METHOD>;
using randgen_type = RandomUniformND<Xoroshiro64Star,3>;
using raygen_type = path_tracing::BasicRayGenerator<ray_type>;
using intersector_type = Intersector<ray_type, scene_type, aniso_interaction>;
using material_system_type = MaterialSystem<bxdfnode_type, diffuse_bxdf_type, conductor_bxdf_type, dielectric_bxdf_type, iri_conductor_bxdf_type, iri_dielectric_bxdf_type, scene_type>;
using nee_type = NextEventEstimator<scene_type, light_type, ray_type, sample_t, aniso_interaction, LIGHT_TYPE, POLYGON_METHOD>;

#ifdef RWMC_ENABLED
using accumulator_type = rwmc::CascadeAccumulator<rwmc::DefaultCascades<float32_t3, CascadeCount> >;
#else
#include "nbl/builtin/hlsl/path_tracing/default_accumulator.hlsl"
using accumulator_type = path_tracing::DefaultAccumulator<float32_t3>;
#endif

using pathtracer_type = path_tracing::Unidirectional<randgen_type, ray_type, intersector_type, material_system_type, nee_type, accumulator_type, scene_type>;

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

    // set up path tracer
    pathtracer_type pathtracer;

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
    
    scene_type scene;
    scene.updateLight(renderPushConstants.generalPurposeLightMatrix);

    raygen_type rayGen;
    rayGen.pixOffsetParam = pixOffsetParam;
    rayGen.camPos = camPos;
    rayGen.NDC = NDC;
    rayGen.invMVP = renderPushConstants.invMVP;

    pathtracer.scene = scene;
    pathtracer.randGen = randgen_type::create(scramblebuf[coords].rg, renderPushConstants.pSampleSequence);
    pathtracer.nee.lights = lights;
    pathtracer.nee.lightCount = scene_type::SCENE_LIGHT_COUNT;
    pathtracer.materialSystem.bxdfs = bxdfs;
    pathtracer.materialSystem.bxdfCount = scene_type::SCENE_BXDF_COUNT;
    pathtracer.bxdfPdfThreshold = 0.0001;
    pathtracer.lumaContributionThreshold = hlsl::dot(colorspace::scRGBtoXYZ[1], colorspace::eotf::sRGB(hlsl::promote<spectral_t>(1.0 / 255.0)));
    pathtracer.spectralTypeToLumaCoeffs = colorspace::scRGBtoXYZ[1];

#ifdef RWMC_ENABLED
    accumulator_type accumulator = accumulator_type::create(pc.splattingParameters);
#else
    accumulator_type accumulator = accumulator_type::create();
#endif
    // path tracing loop
    for(int i = 0; i < renderPushConstants.sampleCount; ++i)
    {
        float32_t3 uvw = pathtracer.randGen(0u, i);
        ray_type ray = rayGen.generate(uvw);
        ray.initPayload();
        pathtracer.sampleMeasure(ray, i, renderPushConstants.depth, accumulator);
    }

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