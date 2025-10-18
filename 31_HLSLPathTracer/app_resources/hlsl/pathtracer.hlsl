#ifndef _NBL_HLSL_EXT_PATHTRACER_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACER_INCLUDED_

#include <nbl/builtin/hlsl/colorspace/EOTF.hlsl>
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>

#include "rand_gen.hlsl"
#include "ray_gen.hlsl"
#include "intersector.hlsl"
#include "material_system.hlsl"
#include "next_event_estimator.hlsl"
#include "scene.hlsl"

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace PathTracer
{

template<typename BxDFCreation, typename Scalar>
struct PathTracerCreationParams
{
    // rng gen
    uint32_t2 rngState;

    // ray gen
    vector<Scalar, 2> pixOffsetParam;
    vector<Scalar, 3> camPos;
    vector<Scalar, 4> NDC;
    matrix<Scalar, 4, 4> invMVP;

    // mat
    BxDFCreation diffuseParams;
    BxDFCreation conductorParams;
    BxDFCreation dielectricParams;
};

template<typename OutputTypeVec NBL_PRIMARY_REQUIRES(concepts::FloatingPointVector<OutputTypeVec>)
struct DefaultAccumulator
{
    struct DefaultAccumulatorInitializationSettings {};

    using output_storage_type = OutputTypeVec;
    using initialization_data = DefaultAccumulatorInitializationSettings;
    output_storage_type accumulation;

    void initialize(in initialization_data initializationData)
    {
        accumulation = (output_storage_type)0.0f;
    }

    void addSample(uint32_t sampleIndex, float32_t3 sample)
    {
        using ScalarType = typename vector_traits<OutputTypeVec>::scalar_type;
        ScalarType rcpSampleSize = 1.0 / (sampleIndex + 1);
        accumulation += (sample - accumulation) * rcpSampleSize;
    }
};

template<class RandGen, class RayGen, class Intersector, class MaterialSystem, /* class PathGuider, */ class NextEventEstimator, class Accumulator>
struct Unidirectional
{
    using this_t = Unidirectional<RandGen, RayGen, Intersector, MaterialSystem, NextEventEstimator, Accumulator>;
    using randgen_type = RandGen;
    using raygen_type = RayGen;
    using intersector_type = Intersector;
    using material_system_type = MaterialSystem;
    using nee_type = NextEventEstimator;

    using scalar_type = typename MaterialSystem::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using measure_type = typename MaterialSystem::measure_type;
    using output_storage_type = typename Accumulator::output_storage_type;
    using sample_type = typename NextEventEstimator::sample_type;
    using ray_dir_info_type = typename sample_type::ray_dir_info_type;
    using ray_type = typename RayGen::ray_type;
    using light_type = Light<measure_type>;
    using bxdfnode_type = BxDFNode<measure_type>;
    using anisotropic_type = typename MaterialSystem::anisotropic_type;
    using isotropic_type = typename anisotropic_type::isotropic_type;
    using anisocache_type = typename MaterialSystem::anisocache_type;
    using isocache_type = typename anisocache_type::isocache_type;
    using quotient_pdf_type = typename NextEventEstimator::quotient_pdf_type;
    using params_type = typename MaterialSystem::params_t;
    using create_params_type = typename MaterialSystem::create_params_t;
    using scene_type = Scene<light_type, bxdfnode_type>;

    using diffuse_op_type = typename MaterialSystem::diffuse_op_type;
    using conductor_op_type = typename MaterialSystem::conductor_op_type;
    using dielectric_op_type = typename MaterialSystem::dielectric_op_type;

    static this_t create(NBL_CONST_REF_ARG(PathTracerCreationParams<create_params_type, scalar_type>) params)
    {
        this_t retval;
        retval.randGen = randgen_type::create(params.rngState);
        retval.rayGen = raygen_type::create(params.pixOffsetParam, params.camPos, params.NDC, params.invMVP);
        retval.materialSystem = material_system_type::create(params.diffuseParams, params.conductorParams, params.dielectricParams);
        return retval;
    }

    vector3_type rand3d(uint32_t protoDimension, uint32_t _sample, uint32_t i)
    {
        uint32_t address = glsl::bitfieldInsert<uint32_t>(protoDimension, _sample, MAX_DEPTH_LOG2, MAX_SAMPLES_LOG2);
	    uint32_t3 seqVal = sampleSequence[address + i].xyz;
	    seqVal ^= randGen();
        return vector3_type(seqVal) * bit_cast<scalar_type>(0x2f800004u);
    }

    scalar_type getLuma(NBL_CONST_REF_ARG(vector3_type) col)
    {
        return hlsl::dot<vector3_type>(hlsl::transpose(colorspace::scRGBtoXYZ)[1], col);
    }

    // TODO: probably will only work with procedural shapes, do the other ones
    bool closestHitProgram(uint32_t depth, uint32_t _sample, NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(scene_type) scene)
    {
        const ObjectID objectID = ray.objectID;
        const vector3_type intersection = ray.origin + ray.direction * ray.intersectionT;

        uint32_t bsdfLightIDs;
        anisotropic_type interaction;
        isotropic_type iso_interaction;
        uint32_t mode = objectID.mode;
        switch (mode)
        {
            // TODO
            case IM_RAY_QUERY:
            case IM_RAY_TRACING:
                break;
            case IM_PROCEDURAL:
            {
                bsdfLightIDs = scene.getBsdfLightIDs(objectID);
                vector3_type N = scene.getNormal(objectID, intersection);
                N = nbl::hlsl::normalize(N);
                ray_dir_info_type V;
                V.direction = -ray.direction;
                isotropic_type iso_interaction = isotropic_type::create(V, N);
                interaction = anisotropic_type::create(iso_interaction);
            }
            break;
            default:
                break;
        }

        vector3_type throughput = ray.payload.throughput;

        // emissive
        const uint32_t lightID = glsl::bitfieldExtract(bsdfLightIDs, 16, 16);
        if (lightID != light_type::INVALID_ID)
        {
            float _pdf;
            ray.payload.accumulation += nee.deferredEvalAndPdf(_pdf, scene, lightID, ray) * throughput / (1.0 + _pdf * _pdf * ray.payload.otherTechniqueHeuristic);
        }

        const uint32_t bsdfID = glsl::bitfieldExtract(bsdfLightIDs, 0, 16);
        if (bsdfID == bxdfnode_type::INVALID_ID)
            return false;

        bxdfnode_type bxdf = scene.bxdfs[bsdfID];

        // TODO: ifdef kill diffuse specular paths

        const bool isBSDF = (bxdf.materialType == ext::MaterialSystem::MaterialType::DIFFUSE) ? bxdf_traits<diffuse_op_type>::type == BT_BSDF :
                            (bxdf.materialType == ext::MaterialSystem::MaterialType::CONDUCTOR) ? bxdf_traits<conductor_op_type>::type == BT_BSDF :
                            bxdf_traits<dielectric_op_type>::type == BT_BSDF;

        vector3_type eps0 = rand3d(depth, _sample, 0u);
        vector3_type eps1 = rand3d(depth, _sample, 1u);

        // thresholds
        const scalar_type bxdfPdfThreshold = 0.0001;
        const scalar_type lumaContributionThreshold = getLuma(colorspace::eotf::sRGB<vector3_type>((vector3_type)1.0 / 255.0)); // OETF smallest perceptible value
        const vector3_type throughputCIE_Y = hlsl::transpose(colorspace::sRGBtoXYZ)[1] * throughput;    // TODO: this only works if spectral_type is dim 3
        const measure_type eta = bxdf.params.ior0 / bxdf.params.ior1;   // assume it's real, not imaginary?
        const scalar_type monochromeEta = hlsl::dot<vector3_type>(throughputCIE_Y, eta) / (throughputCIE_Y.r + throughputCIE_Y.g + throughputCIE_Y.b);  // TODO: imaginary eta?

        // sample lights
        const scalar_type neeProbability = 1.0; // BSDFNode_getNEEProb(bsdf);
        scalar_type rcpChoiceProb;
        if (!math::partitionRandVariable(neeProbability, eps0.z, rcpChoiceProb) && depth < 2u)
        {
            uint32_t randLightID = uint32_t(float32_t(randGen().x) / numeric_limits<uint32_t>::max) * scene.lightCount;
            quotient_pdf_type neeContrib_pdf;
            scalar_type t;
            sample_type nee_sample = nee.generate_and_quotient_and_pdf(
                neeContrib_pdf, t,
                scene, randLightID, intersection, interaction,
                isBSDF, eps0, depth
            );

            // We don't allow non watertight transmitters in this renderer
            bool validPath = nee_sample.NdotL > numeric_limits<scalar_type>::min;
            // but if we allowed non-watertight transmitters (single water surface), it would make sense just to apply this line by itself
            anisocache_type _cache;
            validPath = validPath && anisocache_type::template compute<ray_dir_info_type, ray_dir_info_type>(_cache, interaction, nee_sample, monochromeEta);
            bxdf.params.eta = monochromeEta;

            if (neeContrib_pdf.pdf < numeric_limits<scalar_type>::max)
            {
                if (nbl::hlsl::any(isnan(nee_sample.L.direction)))
                    ray.payload.accumulation += vector3_type(1000.f, 0.f, 0.f);
                else if (nbl::hlsl::all((vector3_type)69.f == nee_sample.L.direction))
                    ray.payload.accumulation += vector3_type(0.f, 1000.f, 0.f);
                else if (validPath)
                {
                    bxdf::BxDFClampMode _clamp;
                    _clamp = (bxdf.materialType == ext::MaterialSystem::MaterialType::DIELECTRIC) ? bxdf::BxDFClampMode::BCM_ABS : bxdf::BxDFClampMode::BCM_MAX;
                    // example only uses isotropic bxdfs
                    params_type params = params_type::template create<sample_type, isotropic_type, isocache_type>(nee_sample, interaction.isotropic, _cache.iso_cache, _clamp);

                    quotient_pdf_type bsdf_quotient_pdf = materialSystem.quotient_and_pdf(bxdf.materialType, bxdf.params, params);
                    neeContrib_pdf.quotient *= bxdf.albedo * throughput * bsdf_quotient_pdf.quotient;
                    const scalar_type otherGenOverChoice = bsdf_quotient_pdf.pdf * rcpChoiceProb;
                    const scalar_type otherGenOverLightAndChoice = otherGenOverChoice / bsdf_quotient_pdf.pdf;
                    neeContrib_pdf.quotient *= otherGenOverChoice / (1.f + otherGenOverLightAndChoice * otherGenOverLightAndChoice);   // balance heuristic

                    // TODO: ifdef NEE only
                    // neeContrib_pdf.quotient *= otherGenOverChoice;

                    ray_type nee_ray;
                    nee_ray.origin = intersection + nee_sample.L.direction * t * Tolerance<scalar_type>::getStart(depth);
                    nee_ray.direction = nee_sample.L.direction;
                    nee_ray.intersectionT = t;
                    if (bsdf_quotient_pdf.pdf < numeric_limits<scalar_type>::max && getLuma(neeContrib_pdf.quotient) > lumaContributionThreshold && intersector_type::traceRay(nee_ray, scene).id == -1)
                        ray.payload.accumulation += neeContrib_pdf.quotient;
                }
            }
        }

        // return false;   // NEE only

        // sample BSDF
        scalar_type bxdfPdf;
        vector3_type bxdfSample;
        {
            anisocache_type _cache;
            sample_type bsdf_sample = materialSystem.generate(bxdf.materialType, bxdf.params, interaction, eps1, _cache);

            bxdf::BxDFClampMode _clamp;
            _clamp = (bxdf.materialType == ext::MaterialSystem::MaterialType::DIELECTRIC) ? bxdf::BxDFClampMode::BCM_ABS : bxdf::BxDFClampMode::BCM_MAX;
            // example only uses isotropic bxdfs
            params_type params = params_type::template create<sample_type, isotropic_type, isocache_type>(bsdf_sample, interaction.isotropic, _cache.iso_cache, _clamp);

            // the value of the bsdf divided by the probability of the sample being generated
            quotient_pdf_type bsdf_quotient_pdf = materialSystem.quotient_and_pdf(bxdf.materialType, bxdf.params, params);
            throughput *= bxdf.albedo * bsdf_quotient_pdf.quotient;
            bxdfPdf = bsdf_quotient_pdf.pdf;
            bxdfSample = bsdf_sample.L.direction;
        }

        // additional threshold
        const float lumaThroughputThreshold = lumaContributionThreshold;
        if (bxdfPdf > bxdfPdfThreshold && getLuma(throughput) > lumaThroughputThreshold)
        {
            ray.payload.throughput = throughput;
            scalar_type otherTechniqueHeuristic = neeProbability / bxdfPdf; // numerically stable, don't touch
            ray.payload.otherTechniqueHeuristic = otherTechniqueHeuristic * otherTechniqueHeuristic;

            // trace new ray
            ray.origin = intersection + bxdfSample * (1.0/*kSceneSize*/) * Tolerance<scalar_type>::getStart(depth);
            ray.direction = bxdfSample;
            if ((PTPolygonMethod)nee_type::PolygonMethod == PPM_APPROX_PROJECTED_SOLID_ANGLE)
            {
                ray.normalAtOrigin = interaction.isotropic.N;
                ray.wasBSDFAtOrigin = isBSDF;
            }
            return true;
        }

        return false;
    }

    void missProgram(NBL_REF_ARG(ray_type) ray)
    {
        vector3_type finalContribution = ray.payload.throughput;
        // #ifdef USE_ENVMAP
        //     vec2 uv = SampleSphericalMap(_immutable.direction);
        //     finalContribution *= textureLod(envMap, uv, 0.0).rgb;
        // #else
        const vector3_type kConstantEnvLightRadiance = vector3_type(0.15, 0.21, 0.3);   // TODO: match spectral_type
        finalContribution *= kConstantEnvLightRadiance;
        ray.payload.accumulation += finalContribution;
        // #endif
    }

    // Li
    output_storage_type getMeasure(uint32_t numSamples, uint32_t depth, NBL_CONST_REF_ARG(scene_type) scene, NBL_REF_ARG(typename Accumulator::initialization_data) accumulatorInitData)
    {
        Accumulator accumulator;
        accumulator.initialize(accumulatorInitData);
        //scalar_type meanLumaSq = 0.0;
        for (uint32_t i = 0; i < numSamples; i++)
        {
            vector3_type uvw = rand3d(0u, i, randGen.rng());    // TODO: take from scramblebuf?
            ray_type ray = rayGen.generate(uvw);

            // bounces
            bool hit = true;
            bool rayAlive = true;
            for (int d = 1; (d <= depth) && hit && rayAlive; d += 2)
            {
                ray.intersectionT = numeric_limits<scalar_type>::max;
                ray.objectID = intersector_type::traceRay(ray, scene);

                hit = ray.objectID.id != -1;
                if (hit)
                    rayAlive = closestHitProgram(1, i, ray, scene);
            }
            if (!hit)
                missProgram(ray);

            accumulator.addSample(i, ray.payload.accumulation);

            // TODO: visualize high variance

            // TODO: russian roulette early exit?
        }

        return accumulator.accumulation;
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t MAX_DEPTH_LOG2 = 4u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t MAX_SAMPLES_LOG2 = 10u;

    randgen_type randGen;
    raygen_type rayGen;
    material_system_type materialSystem;
    nee_type nee;
};

}
}
}
}

#endif
