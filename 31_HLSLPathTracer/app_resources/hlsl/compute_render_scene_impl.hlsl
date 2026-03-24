#ifndef PATH_TRACER_USE_RWMC
#error PATH_TRACER_USE_RWMC must be defined before including compute_render_scene_impl.hlsl
#endif

namespace pathtracer_render_variant
{
using namespace nbl;
using namespace hlsl;

using ray_dir_info_t = bxdf::ray_dir_info::SBasic<float>;
using iso_interaction = PTIsotropicInteraction<ray_dir_info_t, spectral_t>;
using aniso_interaction = PTAnisotropicInteraction<iso_interaction>;
using sample_t = bxdf::SLightSample<ray_dir_info_t>;
using iso_cache = bxdf::SIsotropicMicrofacetCache<float>;
using aniso_cache = bxdf::SAnisotropicMicrofacetCache<iso_cache>;

using iso_config_t = PTIsoConfiguration<sample_t, iso_interaction, spectral_t>;
using iso_microfacet_config_t = PTIsoMicrofacetConfiguration<sample_t, iso_interaction, iso_cache, spectral_t>;

using diffuse_bxdf_type = bxdf::reflection::SOrenNayar<iso_config_t>;
using conductor_bxdf_type = bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>;
using dielectric_bxdf_type = bxdf::transmission::SGGXDielectricIsotropic<iso_microfacet_config_t>;
using iri_conductor_bxdf_type = bxdf::reflection::SIridescent<iso_microfacet_config_t>;
using iri_dielectric_bxdf_type = bxdf::transmission::SIridescent<iso_microfacet_config_t>;

using payload_type = Payload<float>;
using ray_type = Ray<payload_type, PPM_APPROX_PROJECTED_SOLID_ANGLE>;
using randgen_type = RandomUniformND<Xoroshiro64Star, 3>;
using raygen_type = path_tracing::BasicRayGenerator<ray_type>;
using intersector_type = Intersector<ray_type, scene_type, aniso_interaction>;
using material_system_type = MaterialSystem<bxdfnode_type, diffuse_bxdf_type, conductor_bxdf_type, dielectric_bxdf_type, iri_conductor_bxdf_type, iri_dielectric_bxdf_type, scene_type>;
using nee_type = NextEventEstimator<scene_type, light_type, ray_type, sample_t, aniso_interaction, LIGHT_TYPE>;

#if PATH_TRACER_USE_RWMC
using accumulator_type = rwmc::CascadeAccumulator<rwmc::DefaultCascades<float32_t3, CascadeCount> >;
#else
using accumulator_type = path_tracing::DefaultAccumulator<float32_t3>;
#endif

using pathtracer_type = path_tracing::Unidirectional<randgen_type, ray_type, intersector_type, material_system_type, nee_type, accumulator_type, scene_type>;

RenderPushConstants getRenderPushConstants()
{
#if PATH_TRACER_USE_RWMC
	return ::pc.renderPushConstants;
#else
	return ::pc;
#endif
}

void tracePixel(int32_t2 coords, NEEPolygonMethod polygonMethod)
{
	const RenderPushConstants renderPushConstants = getRenderPushConstants();

	uint32_t width, height, imageArraySize;
	::outImage.GetDimensions(width, height, imageArraySize);
	if (any(coords < int32_t2(0, 0)) || any(coords >= int32_t2(width, height)))
		return;

	float32_t2 texCoord = float32_t2(coords) / float32_t2(width, height);
	texCoord.y = 1.0 - texCoord.y;

	if (((renderPushConstants.depth - 1) >> MaxDepthLog2) > 0 || ((renderPushConstants.sampleCount - 1) >> MaxSamplesLog2) > 0)
	{
		::outImage[uint3(coords.x, coords.y, 0)] = float32_t4(1.0, 0.0, 0.0, 1.0);
		return;
	}

	pathtracer_type pathtracer;

	uint2 scrambleDim;
	::scramblebuf.GetDimensions(scrambleDim.x, scrambleDim.y);
	const float32_t2 pixOffsetParam = float32_t2(1.0, 1.0) / float32_t2(scrambleDim);

	float32_t4 NDC = float32_t4(texCoord * float32_t2(2.0, -2.0) + float32_t2(-1.0, 1.0), 0.0, 1.0);
	float32_t3 camPos;
	{
		float32_t4 tmp = mul(renderPushConstants.invMVP, NDC);
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
	pathtracer.randGen = randgen_type::create(::scramblebuf[coords].rg, renderPushConstants.pSampleSequence);
	pathtracer.nee.lights = lights;
	pathtracer.nee.polygonMethod = polygonMethod;
	pathtracer.materialSystem.bxdfs = bxdfs;
	pathtracer.bxdfPdfThreshold = 0.0001;
	pathtracer.lumaContributionThreshold = hlsl::dot(colorspace::scRGBtoXYZ[1], colorspace::eotf::sRGB(hlsl::promote<spectral_t>(1.0 / 255.0)));
	pathtracer.spectralTypeToLumaCoeffs = colorspace::scRGBtoXYZ[1];

#if PATH_TRACER_USE_RWMC
	accumulator_type accumulator = accumulator_type::create(::pc.splattingParameters);
#else
	accumulator_type accumulator = accumulator_type::create();
#endif

	for (int i = 0; i < renderPushConstants.sampleCount; ++i)
	{
		const float32_t3 uvw = pathtracer.randGen(0u, i);
		ray_type ray = rayGen.generate(uvw);
		ray.initPayload();
		pathtracer.sampleMeasure(ray, i, renderPushConstants.depth, accumulator);
	}

#if PATH_TRACER_USE_RWMC
	for (uint32_t i = 0; i < CascadeCount; ++i)
		::cascade[uint3(coords.x, coords.y, i)] = float32_t4(accumulator.accumulation.data[i], 1.0f);
#else
	::outImage[uint3(coords.x, coords.y, 0)] = float32_t4(accumulator.accumulation, 1.0);
#endif
}

#if PATH_TRACER_ENABLE_LINEAR
void runLinear(uint32_t3 threadID, NEEPolygonMethod polygonMethod)
{
	uint32_t width, height, imageArraySize;
	::outImage.GetDimensions(width, height, imageArraySize);
	tracePixel(int32_t2(threadID.x % width, threadID.x / width), polygonMethod);
}
#endif

#if PATH_TRACER_ENABLE_PERSISTENT
void runPersistent(NEEPolygonMethod polygonMethod)
{
	uint32_t width, height, imageArraySize;
	::outImage.GetDimensions(width, height, imageArraySize);
	const uint32_t numWorkgroupsX = width / RenderWorkgroupSizeSqrt;
	const uint32_t numWorkgroupsY = height / RenderWorkgroupSizeSqrt;

	[loop]
	for (uint32_t wgBase = glsl::gl_WorkGroupID().x; wgBase < numWorkgroupsX * numWorkgroupsY; wgBase += glsl::gl_NumWorkGroups().x)
	{
		const int32_t2 wgCoords = int32_t2(wgBase % numWorkgroupsX, wgBase / numWorkgroupsX);
		morton::code<true, 32, 2> mc;
		mc.value = glsl::gl_LocalInvocationIndex().x;
		const int32_t2 localCoords = _static_cast<int32_t2>(mc);
		tracePixel(wgCoords * int32_t2(RenderWorkgroupSizeSqrt, RenderWorkgroupSizeSqrt) + localCoords, polygonMethod);
	}
}
#endif
}
#undef PATH_TRACER_USE_RWMC
