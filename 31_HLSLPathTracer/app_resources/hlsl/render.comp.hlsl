#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/random/pcg.hlsl"
#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"

#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"

#include "pathtracer.hlsl"

// add these defines (one at a time) using -D argument to dxc
// #define SPHERE_LIGHT
// #define TRIANGLE_LIGHT
// #define RECTANGLE_LIGHT

#ifdef SPHERE_LIGHT
#define SPHERE_COUNT 9
#define LIGHT_TYPE ext::PST_SPHERE
#else
#define SPHERE_COUNT 8
#endif

using namespace nbl::hlsl;

NBL_CONSTEXPR uint32_t WorkgroupSize = 32;
NBL_CONSTEXPR uint32_t MAX_DEPTH_LOG2 = 4;
NBL_CONSTEXPR uint32_t MAX_SAMPLES_LOG2 = 10;

struct SPushConstants
{
    float32_t4x4 invMVP;
    int sampleCount;
    int depth;
};

[[vk::push_constant]] SPushConstants pc;

[[vk::combinedImageSampler]][[vk::binding(0, 2)]] Texture2D<float3> envMap;      // unused
[[vk::combinedImageSampler]][[vk::binding(0, 2)]] SamplerState envSampler;

[[vk::binding(1, 2)]] Buffer sampleSequence;

[[vk::combinedImageSampler]][[vk::binding(2, 2)]] Texture2D<uint2> scramblebuf; // unused
[[vk::combinedImageSampler]][[vk::binding(2, 2)]] SamplerState scrambleSampler;

[[vk::image_format("rgba16f")]][[vk::binding(0, 0)]] RWTexture2D<float32_t4> outImage;

int32_t2 getCoordinates()
{
    return int32_t2(glsl::gl_GlobalInvocationID().xy);
}

float32_t2 getTexCoords()
{
    uint32_t width, height;
    outImage.GetDimensions(width, height);
    int32_t2 iCoords = getCoordinates();
    return float32_t2(float(iCoords.x) / width, 1.0 - float(iCoords.y) / height);
}

using ray_dir_info_t = bxdf::ray_dir_info::SBasic<float>;
using iso_interaction = bxdf::surface_interactions::SIsotropic<ray_dir_info_t>;
using aniso_interaction = bxdf::surface_interactions::SAnisotropic<ray_dir_info_t>;
using sample_t = bxdf::SLightSample<ray_dir_info_t>;
using iso_cache = bxdf::SIsotropicMicrofacetCache<float>;
using aniso_cache = bxdf::SAnisotropicMicrofacetCache<float>;
using quotient_pdf_t = bxdf::quotient_and_pdf<float32_t3, float>;
using spectral_t = vector<float, 3>;
using params_t = bxdf::SBxDFParams<float>;
using create_params_t = bxdf::SBxDFCreationParams<float, spectral_t>;

using diffuse_bxdf_type = bxdf::reflection::SOrenNayarBxDF<sample_t, iso_interaction, aniso_interaction, spectral_t>;
using conductor_bxdf_type = bxdf::reflection::SGGXBxDF<sample_t, iso_cache, aniso_cache, spectral_t>;
using dielectric_bxdf_type = bxdf::transmission::SGGXDielectricBxDF<sample_t, iso_cache, aniso_cache, spectral_t>;

using ray_type = ext::Ray<float>;
using light_type = ext::Light<spectral_t>;
using bxdfnode_type = ext::BxDFNode<spectral_t>;
using randgen_type = ext::RandGen::Uniform3D<Xoroshiro64Star>;
using raygen_type = ext::RayGen::Basic<ray_type>;
using intersector_type = ext::Intersector::Comprehensive<ray_type, light_type, bxdfnode_type>;
using material_system_type = ext::MaterialSystem::System<diffuse_bxdf_type, conductor_bxdf_type, dielectric_bxdf_type>;
using nee_type = ext::NextEventEstimator::Estimator<light_type, ray_type, sample_t, aniso_interaction>;
using pathtracer_type = ext::PathTracer::Unidirectional<randgen_type, raygen_type, intersector_type, material_system_type, nee_type>;

static const ext::Shape<ext::PST_SPHERE> spheres[SPHERE_COUNT] = {
    ext::Shape<ext::PST_SPHERE>::create(float3(0.0, -100.5, -1.0), 100.0, 0u, light_type::INVALID_ID),
    ext::Shape<ext::PST_SPHERE>::create(float3(2.0, 0.0, -1.0), 0.5, 1u, light_type::INVALID_ID),
    ext::Shape<ext::PST_SPHERE>::create(float3(0.0, 0.0, -1.0), 0.5, 2u, light_type::INVALID_ID),
    ext::Shape<ext::PST_SPHERE>::create(float3(-2.0, 0.0, -1.0), 0.5, 3u, light_type::INVALID_ID),
    ext::Shape<ext::PST_SPHERE>::create(float3(2.0, 0.0, 1.0), 0.5, 4u, light_type::INVALID_ID),
    ext::Shape<ext::PST_SPHERE>::create(float3(0.0, 0.0, 1.0), 0.5, 4u, light_type::INVALID_ID),
    ext::Shape<ext::PST_SPHERE>::create(float3(-2.0, 0.0, 1.0), 0.5, 5u, light_type::INVALID_ID),
    ext::Shape<ext::PST_SPHERE>::create(float3(0.5, 1.0, 0.5), 0.5, 6u, light_type::INVALID_ID)
#ifdef SPHERE_LIGHT
    ,ext::Shape<ext::PST_SPHERE>::create(float3(-1.5, 1.5, 0.0), 0.3, bxdfnode_type::INVALID_ID, 0u)
#endif
};

#ifdef TRIANGLE_LIGHT
#define LIGHT_TYPE ext::PST_TRIANGLE
#define TRIANGLE_COUNT 1
static const ext::Shape<ext::PST_TRIANGLE> triangles[TRIANGLE_COUNT] = {
    ext::Shape<ext::PST_TRIANGLE>::create(float3(-1.8,0.35,0.3) * 10.0, float3(-1.2,0.35,0.0) * 10.0, float3(-1.5,0.8,-0.3) * 10.0, bxdfnode_type::INVALID_ID, 0u)
};
#endif

#ifdef RECTANGLE_LIGHT
#define LIGHT_TYPE ext::PST_RECTANGLE
#define RECTANGLE_COUNT 1
static const ext::Shape<ext::PST_RECTANGLE> rectangles[RECTANGLE_COUNT] = {
    ext::Shape<ext::PST_RECTANGLE>::create(float3(-3.8,0.35,1.3), normalize(float3(2,0,-1))*7.0, normalize(float3(2,-5,4))*0.1, bxdfnode_type::INVALID_ID, 0u)
};
#endif

#define LIGHT_COUNT 1
static const light_type lights[LIGHT_COUNT] = {
    light_type::create(spectral_t(30.0,25.0,15.0), ext::ObjectID::create(8u, ext::NextEventEstimator::Event::Mode::PROCEDURAL, LIGHT_TYPE))
};

#define BXDF_COUNT 7
static const bxdfnode_type bxdfs[BXDF_COUNT] = {
    bxdfnode_type::create(ext::MaterialSystem::Material::Type::DIFFUSE, false, float2(0,0), spectral_t(1,1,1), spectral_t(1.25,1.25,1.25)),
    bxdfnode_type::create(ext::MaterialSystem::Material::Type::DIFFUSE, false, float2(0,0), spectral_t(1,1,1), spectral_t(1.25,2.5,2.5)),
    bxdfnode_type::create(ext::MaterialSystem::Material::Type::DIFFUSE, false, float2(0,0), spectral_t(1,1,1), spectral_t(2.5,1.25,2.5)),
    bxdfnode_type::create(ext::MaterialSystem::Material::Type::CONDUCTOR, false, float2(0,0), spectral_t(1,1,1), spectral_t(0.98,0.98,0.77)),
    bxdfnode_type::create(ext::MaterialSystem::Material::Type::CONDUCTOR, false, float2(0,0), spectral_t(1,1,1), spectral_t(0.98,0.77,0.98)),
    bxdfnode_type::create(ext::MaterialSystem::Material::Type::CONDUCTOR, false, float2(0.15,0.15), spectral_t(1,1,1), spectral_t(0.98,0.77,0.98)),
    bxdfnode_type::create(ext::MaterialSystem::Material::Type::DIELECTRIC, false, float2(0.0625,0.0625), spectral_t(1,1,1), spectral_t(0.71,0.69,0.67))
};

[numthreads(WorkgroupSize, WorkgroupSize, 1)]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
    uint32_t width, height;
    outImage.GetDimensions(width, height);
    const int32_t2 coords = getCoordinates();
    float32_t2 texCoord = float32_t2(coords) / float32_t2(width, height);
    texCoord.y = 1.0 - texCoord.y;

    if (false == (all((int32_t2)0 < coords)) && all(int32_t2(width, height) < coords)) {
        return;
    }

    if (((pc.depth - 1) >> MAX_DEPTH_LOG2) > 0 || ((pc.sampleCount - 1) >> MAX_SAMPLES_LOG2) > 0)
    {
        float32_t4 pixelCol = float32_t4(1.0,0.0,0.0,1.0);
        outImage[coords] = pixelCol;
        return;
    }

    int flatIdx = glsl::gl_GlobalInvocationID().y * glsl::gl_NumWorkGroups().x * WorkgroupSize + glsl::gl_GlobalInvocationID().x;
    PCG32x2 pcg = PCG32x2::construct(flatIdx);  // replaces scramblebuf?

    // set up path tracer
    ext::PathTracer::PathTracerCreationParams<create_params_t, float> ptCreateParams;
    ptCreateParams.rngState = pcg();

    uint2 scrambleDim;
    scramblebuf.GetDimensions(scrambleDim.x, scrambleDim.y);
    ptCreateParams.pixOffsetParam = (float2)1.0 / float2(scrambleDim);

    float4 NDC = float4(texCoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    {
        float4 tmp = mul(pc.invMVP, NDC);
        ptCreateParams.camPos = tmp.xyz / tmp.w;
        NDC.z = 1.0;
    }

    ptCreateParams.NDC = NDC;
    ptCreateParams.invMVP = pc.invMVP;

    ptCreateParams.diffuseParams = bxdfs[0].params;
    ptCreateParams.conductorParams = bxdfs[3].params;
    ptCreateParams.dielectricParams = bxdfs[6].params;

    pathtracer_type pathtracer = pathtracer_type::create(ptCreateParams, sampleSequence);

    // set up scene (can do as global var?)
    ext::Scene<light_type, bxdfnode_type> scene;
    scene.sphereCount = SPHERE_COUNT;
    for (uint32_t i = 0; i < SPHERE_COUNT; i++)
        scene.spheres[i] = spheres[i];
#ifdef TRIANGLE_LIGHT
    scene.triangleCount = TRIANGLE_COUNT;
    for (uint32_t i = 0; i < TRIANGLE_COUNT; i++)
        scene.triangles[i] = triangles[i];
#else
    scene.triangleCount = 0;
#endif
#ifdef RECTANGLE_LIGHT
    scene.rectangleCount = RECTANGLE_COUNT;
    for (uint32_t i = 0; i < RECTANGLE_COUNT; i++)
        scene.rectangles[i] = rectangles[i];
#else
    scene.rectangleCount = 0;
#endif
    scene.lightCount = LIGHT_COUNT;
    for (uint32_t i = 0; i < LIGHT_COUNT; i++)
        scene.lights[i] = lights[i];
    scene.bxdfCount = BXDF_COUNT;
    for (uint32_t i = 0; i < BXDF_COUNT; i++)
        scene.bxdfs[i] = bxdfs[i];

    float32_t3 color = pathtracer.getMeasure(pc.sampleCount, pc.depth, scene);
    float32_t4 pixCol = float32_t4(color, 1.0);
    outImage[coords] = pixCol;
}
