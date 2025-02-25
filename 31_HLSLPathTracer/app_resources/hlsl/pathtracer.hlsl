#ifndef _NBL_HLSL_EXT_PATHTRACER_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACER_INCLUDED_

#include <nbl/builtin/hlsl/colorspace/EOTF.hlsl>
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl>

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

template<class RandGen, class RayGen, class Intersector, class MaterialSystem, /* class PathGuider, */ class NextEventEstimator>
struct Unidirectional
{
    using this_t = Unidirectional<RandGen, RayGen, Intersector, MaterialSystem, NextEventEstimator>;
    using randgen_type = RandGen;
    using raygen_type = RayGen;
    using intersector_type = Intersector;
    using material_system_type = MaterialSystem;
    using nee_type = NextEventEstimator;

    using scalar_type = typename MaterialSystem::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using measure_type = typename MaterialSystem::measure_type;
    using sample_type = typename NextEventEstimator::sample_type;
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

    // static this_t create(RandGen randGen,
    //                     RayGen rayGen,
    //                     Intersector intersector,
    //                     MaterialSystem materialSystem,
    //                     /* PathGuider pathGuider, */
    //                     NextEventEstimator nee)
    // {}

    static this_t create(NBL_CONST_REF_ARG(PathTracerCreationParams<create_params_type, scalar_type>) params, Buffer sampleSequence)
    {
        this_t retval;
        retval.randGen = randgen_type::create(params.rngState);
        retval.rayGen = raygen_type::create(params.pixOffsetParam, params.camPos, params.NDC, params.invMVP);
        retval.materialSystem = material_system_type::create(params.diffuseParams, params.conductorParams, params.dielectricParams);
        retval.sampleSequence = sampleSequence;
        return retval;
    }

    vector3_type rand3d(uint32_t protoDimension, uint32_t _sample, uint32_t i)
    {
        uint32_t address = glsl::bitfieldInsert<uint32_t>(protoDimension, _sample, MAX_DEPTH_LOG2, MAX_SAMPLES_LOG2);
	    uint32_t3 seqVal = sampleSequence[address + i].xyz;
	    seqVal ^= randGen();
        return vector3_type(seqVal) * asfloat(0x2f800004u);
    }

    scalar_type getLuma(NBL_CONST_REF_ARG(vector3_type) col)
    {
        return nbl::hlsl::dot(nbl::hlsl::transpose(colorspace::scRGBtoXYZ)[1], col);
    }

    // TODO: probably will only work with procedural shapes, do the other ones
    bool closestHitProgram(uint32_t depth, uint32_t _sample, NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(scene_type) scene)
    {
        const ObjectID objectID = ray.objectID;
        const vector3_type intersection = ray.origin + ray.direction * ray.intersectionT;

        uint32_t bsdfLightIDs;
        anisotropic_type interaction;
        isotropic_type iso_interaction;
        ext::Intersector::IntersectData::Mode mode = (ext::Intersector::IntersectData::Mode)objectID.mode;
        switch (mode)
        {
            // TODO
            case ext::Intersector::IntersectData::Mode::RAY_QUERY:
            case ext::Intersector::IntersectData::Mode::RAY_TRACING:
                break;
            case ext::Intersector::IntersectData::Mode::PROCEDURAL:
            {
                bsdfLightIDs = scene.getBsdfLightIDs(objectID);
                vector3_type N = scene.getNormal(objectID, intersection);
                N = nbl::hlsl::normalize(N);
                typename isotropic_type::ray_dir_info_type V;
                V.direction = nbl::hlsl::normalize(-ray.direction);
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
            float pdf;
            ray.payload.accumulation += nee.deferredEvalAndPdf(pdf, scene.lights[lightID], ray, scene.toNextEvent(lightID)) * throughput / (1.0 + pdf * pdf * ray.payload.otherTechniqueHeuristic);
        }

        const uint32_t bsdfID = glsl::bitfieldExtract(bsdfLightIDs, 0, 16);
        if (bsdfID == bxdfnode_type::INVALID_ID)
            return false;

        bxdfnode_type bxdf = scene.bxdfs[bsdfID];

        // TODO: ifdef kill diffuse specular paths

        const bool isBSDF = (bxdf.materialType == ext::MaterialSystem::Material::Type::DIFFUSE) ? bxdf_traits<diffuse_op_type>::type == BT_BSDF :
                            (bxdf.materialType == ext::MaterialSystem::Material::Type::CONDUCTOR) ? bxdf_traits<conductor_op_type>::type == BT_BSDF :
                            bxdf_traits<dielectric_op_type>::type == BT_BSDF;

        vector3_type eps0 = rand3d(depth, _sample, 0u);
        vector3_type eps1 = rand3d(depth, _sample, 1u);

        // thresholds
        const scalar_type bxdfPdfThreshold = 0.0001;
        const scalar_type lumaContributionThreshold = getLuma(colorspace::eotf::sRGB<vector3_type>((vector3_type)1.0 / 255.0)); // OETF smallest perceptible value
        const vector3_type throughputCIE_Y = nbl::hlsl::transpose(colorspace::sRGBtoXYZ)[1] * throughput;   // TODO: this only works if spectral_type is dim 3
        const measure_type eta = bxdf.params.ior0 / bxdf.params.ior1;   // assume it's real, not imaginary?
        const scalar_type monochromeEta = nbl::hlsl::dot(throughputCIE_Y, eta) / (throughputCIE_Y.r + throughputCIE_Y.g + throughputCIE_Y.b);  // TODO: imaginary eta?

        // sample lights
        const scalar_type neeProbability = 1.0; // BSDFNode_getNEEProb(bsdf);
        scalar_type rcpChoiceProb;
        if (!math::partitionRandVariable(neeProbability, eps0.z, rcpChoiceProb) && depth < 2u)
        {
            quotient_pdf_type neeContrib_pdf;
            scalar_type t;
            sample_type nee_sample = nee.generate_and_quotient_and_pdf(
                neeContrib_pdf, t,
                scene.lights[lightID], intersection, interaction,
                isBSDF, eps0, depth, scene.toNextEvent(lightID)
            );

            // We don't allow non watertight transmitters in this renderer
            bool validPath = nee_sample.NdotL > numeric_limits<scalar_type>::min;
            // but if we allowed non-watertight transmitters (single water surface), it would make sense just to apply this line by itself
            anisocache_type _cache;
            validPath = validPath && anisocache_type::compute(_cache, interaction, nee_sample, monochromeEta);
            bxdf.params.A = nbl::hlsl::max(bxdf.params.A, vector<scalar_type, 2>(0,0));
            bxdf.params.eta = monochromeEta;

            if (neeContrib_pdf.pdf < numeric_limits<scalar_type>::max)
            {
                if (nbl::hlsl::any(isnan(nee_sample.L.direction)))
                    ray.payload.accumulation += vector3_type(1000.f, 0.f, 0.f);
                else if (nbl::hlsl::all((vector3_type)69.f == nee_sample.L.direction))
                    ray.payload.accumulation += vector3_type(0.f, 1000.f, 0.f);
                else if (validPath)
                {
                    ext::MaterialSystem::Material material;
                    material.type = bxdf.materialType;
                    params_type params;

                    // TODO: does not yet account for smooth dielectric
                    if (!isBSDF && bxdf.materialType == ext::MaterialSystem::Material::DIFFUSE)
                    {
                        params = params_type::template create<sample_type, isotropic_type>(nee_sample, iso_interaction, bxdf::BCM_MAX);
                    }
                    else if (!isBSDF && bxdf.materialType != ext::MaterialSystem::Material::DIFFUSE)
                    {
                        if (bxdf.params.is_aniso)
                            params = params_type::template create<sample_type, anisotropic_type, anisocache_type>(nee_sample, interaction, _cache, bxdf::BCM_MAX);
                        else
                        {
                            isocache_type isocache = (isocache_type)_cache;
                            params = params_type::template create<sample_type, isotropic_type, isocache_type>(nee_sample, iso_interaction, isocache, bxdf::BCM_MAX);
                        }
                    }
                    else if (isBSDF && bxdf.materialType == ext::MaterialSystem::Material::DIFFUSE)
                    {
                        params = params_type::template create<sample_type, isotropic_type>(nee_sample, iso_interaction, bxdf::BCM_ABS);
                    }
                    else if (isBSDF && bxdf.materialType != ext::MaterialSystem::Material::DIFFUSE)
                    {
                        if (bxdf.params.is_aniso)
                            params = params_type::template create<sample_type, anisotropic_type, anisocache_type>(nee_sample, interaction, _cache, bxdf::BCM_ABS);
                        else
                        {
                            isocache_type isocache = (isocache_type)_cache;
                            params = params_type::template create<sample_type, isotropic_type, isocache_type>(nee_sample, iso_interaction, isocache, bxdf::BCM_ABS);
                        }
                    }

                    quotient_pdf_type bsdf_quotient_pdf = materialSystem.quotient_and_pdf(material, bxdf.params, params) * throughput;
                    neeContrib_pdf.quotient *= bsdf_quotient_pdf.quotient;
                    const scalar_type otherGenOverChoice = bsdf_quotient_pdf.pdf * rcpChoiceProb;
                    const scalar_type otherGenOverLightAndChoice = otherGenOverChoice / bsdf_quotient_pdf.pdf;
                    neeContrib_pdf.quotient *= otherGenOverChoice/(1.f + otherGenOverLightAndChoice * otherGenOverLightAndChoice);   // balance heuristic

                    // TODO: ifdef NEE only

                    ray_type nee_ray;
                    nee_ray.origin = intersection + nee_sample.L.direction * t * Tolerance<scalar_type>::getStart(depth);
                    nee_ray.direction = nee_sample.L.direction;
                    nee_ray.intersectionT = t;
                    if (bsdf_quotient_pdf.pdf < numeric_limits<scalar_type>::max && getLuma(neeContrib_pdf.quotient) > lumaContributionThreshold && intersector_type::traceRay(nee_ray, scene).id == -1)
                        ray._payload.accumulation += neeContrib_pdf.quotient;
                }
            }
        }

        // sample BSDF
        scalar_type bxdfPdf;
        vector3_type bxdfSample;
        {
            ext::MaterialSystem::Material material;
            material.type = bxdf.materialType;

            anisocache_type _cache;
            sample_type bsdf_sample = materialSystem.generate(material, bxdf.params, interaction, eps1, _cache);

            // TODO: does not yet account for smooth dielectric
            params_type params;            
            if (!isBSDF && bxdf.materialType == ext::MaterialSystem::Material::DIFFUSE)
            {
                params = params_type::template create<sample_type, isotropic_type>(bsdf_sample, iso_interaction, bxdf::BCM_MAX);
            }
            else if (!isBSDF && bxdf.materialType != ext::MaterialSystem::Material::DIFFUSE)
            {
                if (bxdf.params.is_aniso)
                    params = params_type::template create<sample_type, anisotropic_type, anisocache_type>(bsdf_sample, interaction, _cache, bxdf::BCM_MAX);
                else
                {
                    isocache_type isocache = (isocache_type)_cache;
                    params = params_type::template create<sample_type, isotropic_type, isocache_type>(bsdf_sample, iso_interaction, isocache, bxdf::BCM_MAX);
                }
            }
            else if (isBSDF && bxdf.materialType == ext::MaterialSystem::Material::DIFFUSE)
            {
                params = params_type::template create<sample_type, isotropic_type>(bsdf_sample, iso_interaction, bxdf::BCM_ABS);
            }
            else if (isBSDF && bxdf.materialType != ext::MaterialSystem::Material::DIFFUSE)
            {
                if (bxdf.params.is_aniso)
                    params = params_type::template create<sample_type, anisotropic_type, anisocache_type>(bsdf_sample, interaction, _cache, bxdf::BCM_ABS);
                else
                {
                    isocache_type isocache = (isocache_type)_cache;
                    params = params_type::template create<sample_type, isotropic_type, isocache_type>(bsdf_sample, iso_interaction, isocache, bxdf::BCM_ABS);
                }
            }

            // the value of the bsdf divided by the probability of the sample being generated
            throughput *= materialSystem.quotient_and_pdf(material, bxdf.params, params);
            bxdfSample = bsdf_sample.L.direction;
        }

        // additional threshold
        const float lumaThroughputThreshold = lumaContributionThreshold;
        if (bxdfPdf > bxdfPdfThreshold && getLuma(throughput) > lumaThroughputThreshold)
        {
            ray.payload.throughput = throughput;
            ray.payload.otherTechniqueHeuristic = neeProbability / bxdfPdf; // numerically stable, don't touch
            ray.payload.otherTechniqueHeuristic *= ray.payload.otherTechniqueHeuristic;
                    
            // trace new ray
            ray.origin = intersection + bxdfSample * (1.0/*kSceneSize*/) * Tolerance<scalar_type>::getStart(depth);
            ray.direction = bxdfSample;
            // #if POLYGON_METHOD==2
            // ray._immutable.normalAtOrigin = interaction.isotropic.N;
            // ray._immutable.wasBSDFAtOrigin = isBSDF;
            // #endif
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
    measure_type getMeasure(uint32_t numSamples, uint32_t depth, NBL_CONST_REF_ARG(scene_type) scene)
    {
        measure_type Li = (measure_type)0.0;
        scalar_type meanLumaSq = 0.0;
        for (uint32_t i = 0; i < numSamples; i++)
        {
            vector3_type uvw = rand3d(0u, i, randGen.rng());    // TODO: take from scramblebuf?
            ray_type ray = rayGen.generate(uvw);

            // bounces
            bool hit = true;
            bool rayAlive = true;
            for (int d = 1; d <= depth && hit && rayAlive; d += 2)
            {
                ray.intersectionT = numeric_limits<scalar_type>::max;
                ray.objectID = intersector_type::traceRay(ray, scene);

                hit = ray.objectID.id != -1;
                if (hit)
                    rayAlive = closestHitProgram(d, i, ray, scene);
            }
            if (!hit)
                missProgram(ray);

            measure_type accumulation = ray.payload.accumulation;
            scalar_type rcpSampleSize = 1.0 / (i + 1);
            Li += (accumulation - Li) * rcpSampleSize;

            // TODO: visualize high variance

            // TODO: russian roulette early exit?
        }

        return Li;
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t MAX_DEPTH_LOG2 = 4u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t MAX_SAMPLES_LOG2 = 10u;

    randgen_type randGen;
    raygen_type rayGen;
    material_system_type materialSystem;
    nee_type nee;

    Buffer sampleSequence;
};

}
}
}
}

#endif
