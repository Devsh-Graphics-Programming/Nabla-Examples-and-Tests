#ifndef _NBL_HLSL_EXT_PATHTRACER_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACER_INCLUDED_

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
    using ray_type = typename RayGen::ray_type;

    // static this_t create(RandGen randGen,
    //                     RayGen rayGen,
    //                     Intersector intersector,
    //                     MaterialSystem materialSystem,
    //                     /* PathGuider pathGuider, */
    //                     NextEventEstimator nee)
    // {}

    static this_t create(NBL_CONST_REF_ARG(PathTracerCreationParams) params)
    {
        this_t retval;
        retval.randGen = randgen_type::create(params.rngState);
        retval.rayGen = raygen_type::create(params.pixOffsetParam, params.camPos, params.NDC, params.invMVP);
        retval.materialSystem = material_system_type::create(diffuseParams, conductorParams, dielectricParams);
        return retval;
    }

    // TODO: get working, what is sampleSequence stuff
    vector3_type rand3d(uint32_t protoDimension, uint32_t _sample)
    {
        uint32_t address = spirv::bitfieldInsert(protoDimension, _sample, MAX_DEPTH_LOG2, MAX_SAMPLES_LOG2);
	    unit32_t3 seqVal = texelFetch(sampleSequence, int(address) + i).xyz;
	    seqVal ^= unit32_t3(randGen(), randGen(), randGen());
        return vector3_type(seqVal) * asfloat(0x2f800004u);
    }

    bool closestHitProgram(unit32_t depth, uint32_t _sample, NBL_REF_ARG(ray_type) ray)
    {
        const uint32_t objectID = ray.objectID;
        const vector3_type intersection = ray.origin + ray.direction * ray.intersectionT;

        uint32_t bsdfLightIDs;
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
    measure_type getMeasure(uint32_t numSamples, uint32_t depth, NBL_CONST_REF_ARG(Scene) scene)
    {
        // loop through bounces, do closest hit
        // return ray.payload.accumulation --> color

        // TODO: not hardcode this, pass value from somewhere?, where to get objects?
        Intersector::IntersectData data;

        measure_type Li = (measure_type)0.0;
        scalar_type meanLumaSq = 0.0;
        for (uint32_t i = 0; i < numSamples; i++)
        {
            vector3_type uvw = rand3d(0u, i);
            ray_type ray = rayGen.generate(uvw);

            // bounces
            bool hit = true;
            bool rayAlive = true;
            for (int d = 1; d <= depth && hit && rayAlive; d += 2)
            {
                ray.intersectionT = numeric_limits<scalar_type>::max;
                ray.objectID = -1;  // start with no intersect
                
                // prodedural shapes
                if (scene.sphereCount > 0)
                {
                    data = Intersector::IntersectData::encode(Intersector::IntersectData::Mode::PROCEDURAL, PST_SPHERE, scene);
                    ray.objectID = intersector.traceRay(ray, data);
                }

                if (scene.triangleCount > 0)
                {
                    data = Intersector::IntersectData::encode(Intersector::IntersectData::Mode::PROCEDURAL, PST_TRIANGLE, scene);
                    ray.objectID = intersector.traceRay(ray, data);
                }

                if (scene.rectangleCount > 0)
                {
                    data = Intersector::IntersectData::encode(Intersector::IntersectData::Mode::PROCEDURAL, PST_RECTANGLE, scene);
                    ray.objectID = intersector.traceRay(ray, data);
                }

                // TODO: trace AS

                hit = ray.objectID != -1;
                if (hit)
                    rayAlive = closestHitProgram(d, i, ray);
            }
            if (!hit)
                missProgram(ray);

            spectral_type accumulation = ray.payload.accumulation;
            scalar_type rcpSampleSize = 1.0 / (i + 1);
            Li += (accumulation - Li) * rcpSampleSize;

            // TODO: visualize high variance
        }

        return Li;
    }

    randgen_type randGen;
    raygen_type rayGen;
    intersector_type intersector;
    material_system_type materialSystem;
    nee_type nee;
};

}
}
}
}

#endif