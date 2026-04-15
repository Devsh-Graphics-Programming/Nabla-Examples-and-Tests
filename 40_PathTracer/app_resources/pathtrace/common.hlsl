#include "renderer/shaders/pathtrace/common.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"

#include "nbl/examples/common/KeyedQuantizedSequence.hlsl"

// TODO: move to material compiler
#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection/oren_nayar.hlsl"
namespace nbl
{
namespace hlsl
{
namespace material_compiler3
{
namespace backends
{
namespace default_upt // unidirectional path tracing
{
NBL_CONSTEXPR_STATIC_INLINE float32_t3 LumaConversionCoeffs = hlsl::transpose(hlsl::colorspace::scRGBtoXYZ)[1];

template<class SpectralType NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<SpectralType>)
struct SIsotropicInteraction
{
    using this_t = SIsotropicInteraction<SpectralType>;
    using spectral_type = SpectralType;
    // TODO: experiment with float16
    using ray_dir_info_type = bxdf::ray_dir_info::SBasic<float32_t>;
    using scalar_type = typename ray_dir_info_type::scalar_type;
    using vector3_type = typename ray_dir_info_type::vector3_type;


    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) normalizedV, const vector3_type normalizedN)
    {
        this_t retval;
        retval.V = normalizedV;
        retval.N = normalizedN;
        retval.NdotV = hlsl::dot<vector3_type>(retval.N,retval.V.getDirection());
        retval.NdotV2 = retval.NdotV * retval.NdotV;
        retval.luminosityContributionHint = spectral_type(LumaConversionCoeffs);

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) normalizedV, const vector3_type normalizedN, const spectral_type throughputSoFar)
    {
        this_t retval = create(normalizedV,normalizedN);
        retval.luminosityContributionHint *= throughputSoFar;
        retval.luminosityContributionHint /= math::lpNorm<spectral_type,1>(retval.luminosityContributionHint);
        return retval;
    }

    ray_dir_info_type getV() NBL_CONST_MEMBER_FUNC { return V; }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return N; }
    scalar_type getNdotV(bxdf::BxDFClampMode _clamp = bxdf::BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC
    {
        return bxdf::conditionalAbsOrMax<scalar_type>(NdotV,_clamp);
    }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return NdotV2; }

    bxdf::PathOrigin getPathOrigin() NBL_CONST_MEMBER_FUNC { return bxdf::PathOrigin::PO_SENSOR; }
    spectral_type getLuminosityContributionHint() NBL_CONST_MEMBER_FUNC { return luminosityContributionHint; }

    ray_dir_info_type V;
    vector3_type N;
    scalar_type NdotV;
    spectral_type luminosityContributionHint;
    // TODO: experiment, precompute vs not
    scalar_type NdotV2;
};


template<class IsotropicInteraction NBL_PRIMARY_REQUIRES(bxdf::surface_interactions::Isotropic<IsotropicInteraction>)
struct SAnisotropicInteraction
{
    using this_t = SAnisotropicInteraction<IsotropicInteraction>;
    using isotropic_interaction_type = IsotropicInteraction;
    using ray_dir_info_type = typename isotropic_interaction_type::ray_dir_info_type;
    using scalar_type = typename ray_dir_info_type::scalar_type;
    using vector3_type = typename ray_dir_info_type::vector3_type;
    using matrix3x3_type = matrix<scalar_type,3,3>;
    using spectral_type = typename isotropic_interaction_type::spectral_type;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic, const vector3_type normalizedT, const vector3_type normalizedB)
    {
        this_t retval;
        retval.isotropic = isotropic;

        retval.T = normalizedT;
        retval.B = normalizedB;

        retval.TdotV = hlsl::dot<vector3_type>(retval.isotropic.getV().getDirection(), retval.T);
        retval.BdotV = hlsl::dot<vector3_type>(retval.isotropic.getV().getDirection(), retval.B);

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic, const vector3_type normalizedT)
    {
        return create(isotropic, normalizedT, cross(isotropic.getN(), normalizedT));
    }
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic)
    {
        vector3_type T, B;
        math::frisvad<vector3_type>(isotropic.getN(), T, B);
        return create(isotropic, nbl::hlsl::normalize<vector3_type>(T), nbl::hlsl::normalize<vector3_type>(B));
    }

    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) normalizedV, const vector3_type normalizedN)
    {
        isotropic_interaction_type isotropic = isotropic_interaction_type::create(normalizedV, normalizedN);
        return create(isotropic);
    }

    ray_dir_info_type getV() NBL_CONST_MEMBER_FUNC { return isotropic.getV(); }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return isotropic.getN(); }
    scalar_type getNdotV(bxdf::BxDFClampMode _clamp = bxdf::BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC { return isotropic.getNdotV(_clamp); }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return isotropic.getNdotV2(); }
    bxdf::PathOrigin getPathOrigin() NBL_CONST_MEMBER_FUNC { return isotropic.getPathOrigin(); }
    spectral_type getLuminosityContributionHint() NBL_CONST_MEMBER_FUNC { return isotropic.getLuminosityContributionHint(); }
    bool isMaterialBSDF() NBL_CONST_MEMBER_FUNC { return isotropic.isMaterialBSDF(); }
    isotropic_interaction_type getIsotropic() NBL_CONST_MEMBER_FUNC { return isotropic; }

    vector3_type getT() NBL_CONST_MEMBER_FUNC { return T; }
    vector3_type getB() NBL_CONST_MEMBER_FUNC { return B; }
    scalar_type getTdotV() NBL_CONST_MEMBER_FUNC { return TdotV; }
    scalar_type getTdotV2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getTdotV(); return t*t; }
    scalar_type getBdotV() NBL_CONST_MEMBER_FUNC { return BdotV; }
    scalar_type getBdotV2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getBdotV(); return t*t; }

    vector3_type getTangentSpaceV() NBL_CONST_MEMBER_FUNC { return vector3_type(TdotV, BdotV, isotropic.getNdotV()); }
    matrix3x3_type getToTangentSpace() NBL_CONST_MEMBER_FUNC { return matrix3x3_type(T, B, isotropic.getN()); }
    matrix3x3_type getFromTangentSpace() NBL_CONST_MEMBER_FUNC { return nbl::hlsl::transpose<matrix3x3_type>(matrix3x3_type(T, B, isotropic.getN())); }

    isotropic_interaction_type isotropic;
    vector3_type T;
    vector3_type B;
    scalar_type TdotV;
    scalar_type BdotV;
};

struct BxDFConfig
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = false;

    // TODO: experiment with float16_t
    using spectral_type = float32_t3;
    using isotropic_interaction_type = SIsotropicInteraction<spectral_type>;
    
    using scalar_type = typename isotropic_interaction_type::scalar_type;
    using vector2_type = vector<scalar_type,2>;
    using vector3_type = typename isotropic_interaction_type::vector3_type;
    using anisotropic_interaction_type = SAnisotropicInteraction<isotropic_interaction_type>;
    // TODO: experiment with spectral_type's scalar type
    using monochrome_type = vector<scalar_type,1>;

    using ray_dir_info_type = typename isotropic_interaction_type::ray_dir_info_type;
    using sample_type = bxdf::SLightSample<ray_dir_info_type>;

    // TODO: change to conform to PR 1001 later
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type,scalar_type>;
};
}
}
}
}
}


namespace nbl
{
namespace this_example
{

// There's different ways to accumulate with on-line averaging:
// - one Sample every Frame: `avg + (sample-avg)/N`
// - variable Samples every Frame without skipping: `avg + (sampleSum-avg*sampleCount)/N` or `avg + (sampleAvg-avg)*sampleCount/N`
//   the second option has 1 MUL extra compared to regular accumulation, whereas first does 1 MUL extra but it requires cheap averaging
// - variable Samples every Frame with skipping: `avg + (sampleSum-avg*(sampleCount+skippedSamples))/N` equivalently `avg+(sampleSum-avg*(N-oldSamples))/N`
//	 pre-averaged variant is then `avg + (sampleAvg - avg*(N-oldSamples)/sampleCount)*sampleCount/N`

template<typename LoadStoreImageAccessor>// NBL_PRIMARY_REQUIRES(
//	hlsl::concepts::accessors::LoadableImage<LoadStoreImageAccessor,typename LoadStoreImageAccessor::scalar_type,LoadStoreImageAccessor::Dimension,LoadStoreImageAccessor::Components> &&
//	hlsl::concepts::accessors::StorableImage<LoadStoreImageAccessor,typename LoadStoreImageAccessor::scalar_type,LoadStoreImageAccessor::Dimension,LoadStoreImageAccessor::Components>
//)
struct Accumulator
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Dimension = LoadStoreImageAccessor::Dimension;
	using coded_type = typename LoadStoreImageAccessor::coded_type;

	// TODO: some check that `T` is same integral type and sign
	template<typename T, int ComponentOverride NBL_FUNC_REQUIRES(ComponentOverride<=LoadStoreImageAccessor::Components)
	void accumulate(const vector<uint16_t,Dimension> coord, const uint16_t layer, const vector<T,ComponentOverride> data, const T rcpNewSampleCount, const bool keepAccumulating)
	{
		coded_type val;

		if (keepAccumulating)
		{
			composed.get(val,coord,layer);
			NBL_UNROLL for (uint16_t i=0; i<ComponentOverride; i++)
				val[i] += (data[i] - val[i]) * rcpNewSampleCount;
			// don't threshold the store, most threads will store, just adding extra if-statement. Comeback to it when we have very uniform AoV views to optimize export there
			//if (hlsl::all(hlsl::truncate<delctype(val)>(hlsl::abs(delta) < LoadStoreImageAccessor::QuantizationThreshold)))
				// return;
		}
		else // clear path
		NBL_UNROLL for (uint16_t i=0; i<LoadStoreImageAccessor::Components; i++)
			val[i] = hlsl::select(i<ComponentOverride,data[i],T(0));

		composed.set(coord,layer,val);
	}

	LoadStoreImageAccessor composed;
};


// get it so that -1.0 maps to -511 (513 unsigned so 0.501466275) and 1.0 maps to 511 (0.4995112) and 0 maps to 0
template<typename T, int N>
vector<T,N> correctSNorm10WhenStoringToUnorm(const vector<T,N> input)
{
	using vec_t = vector<T,N>;
    const T factor = _static_cast<T>(0.499512);
	return hlsl::mix(input*factor+hlsl::promote<vec_t>(_static_cast<T>(0.999022)),input*factor,hlsl::promote<vec_t>(_static_cast<T>(0))<input);
}

using scramble_state_t = hlsl::Xoroshiro64Star;
using randgen_t = hlsl::examples::KeyedQuantizedSequence<scramble_state_t>;

// sample count incrementing function
struct SPixelSamplingInfo
{
	randgen_t randgen;
	float32_t rcpNewSampleCount;
	uint16_t newSampleCount;
	uint16_t firstSample;
};
SPixelSamplingInfo advanceSampleCount(const uint16_t3 coord, const uint16_t newSamplesThisPixel, const uint16_t dontClear, const uint16_t maxSamples)
{
	SPixelSamplingInfo retval;
	// 
    const uint16_t oldSampleCount = _static_cast<uint16_t>(gSampleCount[coord]);
	retval.firstSample = oldSampleCount*dontClear;
	// setup randgen
	{
		retval.randgen.pSampleBuffer = gScene.init.pSampleSequence;
		// TODO: experiment with storing every dimension scramble in the texture to not pollute the ray payload
        // TODO: experiment with truncating the sequence to a fixed number of dimensions, and using the scramble, sampleIndex and rank keys as xoroshiro64 seed and use simple RNG for last dimensions
		retval.randgen.rng = scramble_state_t::construct(gScrambleKey[uint16_t3(coord.xy & uint16_t(511), 0)]);
		retval.randgen.sequenceSamplesLog2 = gScene.init.sequenceSamplesLog2; // TODO: make this compile time constant - Spec Constant?
	}
	//
	retval.newSampleCount = hlsl::min(retval.firstSample+newSamplesThisPixel,maxSamples);
    if (retval.newSampleCount!=oldSampleCount) // whether this pays off depends on ratio of pixels with 0 spp in a dispatch
    	gSampleCount[coord] = retval.newSampleCount;
	// handle overflow properly
	retval.rcpNewSampleCount = 1.f/float32_t(retval.newSampleCount);
	return retval;
}

// raygen functions
struct SRay
{
    using ray_dir_info_type = typename hlsl::material_compiler3::backends::default_upt::BxDFConfig::ray_dir_info_type;

	float32_t3 origin;
    ray_dir_info_type direction;
};

struct SPrimaryRay
{
    SRay ray;
    float32_t tMin;
};

SPrimaryRay genPrimaryRay(const SSensorDynamics sensor, const float32_t2 pixelSizeNDC, const float32_t2 ndc, const float16_t2 xi)
{
    using namespace nbl::hlsl;

    // stochastic reconstruction filter
    const float16_t stddev = _static_cast<float16_t>(1.2);
    const float32_t3 adjNDC = float32_t3(path_tracing::GaussianFilter<float16_t>::create(stddev, stddev).sample(xi) * pixelSizeNDC + ndc, -1.f);

    SPrimaryRay retval;
    // unproject
    if (sensor.orthoCam)
    {
        const float32_t3 viewOrigin = float32_t3(hlsl::mul(sensor.ndcToRay,adjNDC),0.f);
        retval.ray.origin = hlsl::math::linalg::promoted_mul(sensor.invView,viewOrigin);
        retval.ray.direction.setDirection(float32_t3(0,0,-1));
        retval.tMin = sensor.nearClip;
    }
    else
    {
        retval.ray.origin = hlsl::transpose(sensor.invView)[3];
        float32_t3 viewDir;
        if (spirv::LaunchSizeKHR.z != 6u)
            viewDir = float32_t3(hlsl::mul(sensor.ndcToRay, adjNDC), -1.0);
        else
        {
            // TODO: handle cubemap cameras 
        }
        viewDir = hlsl::normalize(viewDir);
        retval.tMin = sensor.nearClip / hlsl::abs(viewDir.z);
        retval.ray.direction.setDirection(viewDir);
    }
    // rotate and scale with camera 
    retval.ray.direction = retval.ray.direction.transform(hlsl::math::linalg::truncate<3,3,3,4>(sensor.invView));
    // TODO: fix this later introduce `transformOrthonormal` and `transform`
    retval.ray.direction.direction = hlsl::normalize(retval.ray.direction.direction);
    return retval;
}

// variables that multiply together
struct SAOVThroughputs
{
    inline void clear(const float16_t weight)
    {
        albedo = hlsl::promote<float16_t3>(transparency = weight);
    }

    inline SAOVThroughputs operator-(const SAOVThroughputs factor)
    {
        SAOVThroughputs retval = this;
        retval.albedo -= factor.albedo;
        retval.transparency -= factor.transparency;
        return retval;
    }

    inline SAOVThroughputs operator*(const float16_t factor)
    {
        SAOVThroughputs retval = this;
        retval.albedo *= factor;
        retval.transparency *= factor;
        return retval;
    }
    inline SAOVThroughputs operator*(const SAOVThroughputs factor)
    {
        SAOVThroughputs retval = this;
        retval.albedo *= factor.albedo;
        retval.transparency *= factor.transparency;
        return retval;
    }

    inline SAOVThroughputs operator/(const float16_t factor)
    {
        SAOVThroughputs retval = this;
        retval.albedo /= factor;
        retval.transparency /= factor;
        return retval;
    }
    inline SAOVThroughputs operator/(const SAOVThroughputs factor)
    {
        SAOVThroughputs retval = this;
        retval.albedo /= factor.albedo;
        retval.transparency /= factor.transparency;
        return retval;
    }


    // RGB transparency of smooth reflections and refractions, used for modulating albedo and most AOVs
    float16_t3 albedo;
    // Motion is special because Real Time defines it as a mapping of where current pixel was last frame.
    // True motion output would require us to implement differentiable rendering and formulate motion as an integral of `Throughput dScreenPos/dTime` which is super tricky because:
    // - A turning mirror imparts motion on the reflection of a static object
    // - whats the motion vector for a disoccluded part of a reflection? How to even know about a disocclusion/our reflection's motion vector reprojecting badly?
    // - its not generally a function, the current pixel could be in multiple places at in the last frame (think about the flow of the reflection of your face in a concave spoon) 
    // - Non-differentiability, hard edges of triangles and Breps
    // - lighting imparts is own motion vectors, e.g. shadows move across static surfaces
    // - how to weigh contributions? luma of RGB effect? inidividually, etc.
    // TL;DR you can't just blend motions and get something useful (even less than normals), only directly tranmissive paths should be allowed to accumulate their motion vectors (easy to calculate)
    // as we pass through surfaces we need to know how much of the outgoing ray distribution is focused around the directly transmissive direction. This can modulate both our masking and motion vectors.
    // Albeit for smooth but refractive surfaces we could experiment with accepting transparent masking even though ray direction won't match in a simple Photoshop composting,
    // but would you rather have an opaque swimming pool, round glass vase, or water droplet OR composted with no refraction? But then we'd need a motion throughput and track some more metadata.
    float16_t transparency;
};

using spectral_t = float32_t3;


struct SArbitraryOutputValues
{
    inline void clear()
    {
        normal = albedo = float16_t3(0,0,0);
        // TODO: motion
    }

    inline SArbitraryOutputValues operator+(const SArbitraryOutputValues other)
    {
        SArbitraryOutputValues retval;
        retval.albedo = albedo+other.albedo;
        retval.normal = normal+other.normal;
        return retval;
    }

    inline SArbitraryOutputValues operator*(const float16_t factor)
    {
        SArbitraryOutputValues retval;
        retval.albedo = albedo*factor;
        retval.normal = normal*factor;
        return retval;
    }

    inline SArbitraryOutputValues operator*(const SAOVThroughputs throughput)
    {
        SArbitraryOutputValues retval;
        retval.albedo = albedo*throughput.albedo;
        retval.normal = normal*hlsl::dot(throughput.albedo,_static_cast<float16_t3>(hlsl::material_compiler3::backends::default_upt::LumaConversionCoeffs));
        return retval;
    }

    // AoVs are handled as "special emission", basically the contribution of albedo is same as the material illuminated in a White Furnace
    // so for transparent (anyhit) to impart its albedo or normal into the AoV it can add it same way it would add any color emission
    float16_t3 albedo;
    // One would think that normals can't be blended, but yes they can! Just make sure you weigh then using the Luma of the RGB aovThroughput.
    // Here's the problem with dealing with reflections & refractions, the reflection of a wall with an X- normal in the X+ window should have an apparent X+ normal.
    // This means that one would need to track the surfaces through which we reflect in a stack along the path, eg:
    // `originalNormal - 2 dot(originalNormal, reflectorNormal) * reflectorNormal == (Identity - 2 outerProductMatrix(reflectorNormal)) originalNormal`
    // To follow through 2 or more reflections we'd need to multiply these 3x3 matrices together along the ray like so
    // `(I - 2 n_0 n_0^T) (I - 2 n_1 n_1^T) = I + 4 n_0 (n_0^T n_1) n_1^T - 2 (n_0 n_0^T + n_1 n_1^T)`
    // Theoretically because every series of reflections is just one reflection and a rotation, it could be possible to store this in 3 floats, due to the properties of SO(3)
    // "The orthogonal group, consisting of all proper and improper rotations, is generated by reflections. Every proper rotation is the composition of two reflections, a special case of the Cartan–Dieudonné theorem."
    // I'm not sure how we could extend that for refractions but probably a similar form is possible  - virtual object corresponding under transmission to what's seen under refraction.
    // The question is.. is it worth it? Do er really need objects warped by in a labyrynth of wonky mirrors to have warped normals? Or a ceiling reflected in a choppy swimming pool to inherit the pool's wave normals ?
    // NO because this is an input to a denoiser to stop it blurring lighting across surfaces oriented in different directions! Doesn't matter what the reflection and refraction normals are as long as they're consistent.
    // For the example of a flat building wall reflected in a wavy but smooth reflector, that would actually be a massive self-own and leave behind a lot of noise!
    float16_t3 normal;
    // TODO: motion (RG vector to past location, B or BA as a measure of spread, e.g. spherical gaussian, direction and its variance, Polar Harmonics - Laplace on a Circle)
    //float16_t3or4 motion;
};

// only callable from closestHit
inline float32_t3 reconstructGeometricNormal()
{
    using namespace nbl::hlsl;

    // Do diffs in high precision, edges can be very long and dot products can easily overflow 64k max float16_t value and normalizing one extra time makes no sense
    const float32_t3 geometricNormal = hlsl::cross(
        spirv::HitTriangleVertexPositionsKHR[1]-spirv::HitTriangleVertexPositionsKHR[0],
        spirv::HitTriangleVertexPositionsKHR[2]-spirv::HitTriangleVertexPositionsKHR[0]
    );

    // Scales can be absolutely huge, we'd need special per-instance pre-scaled 3x3 matrices and also guarantee `geometricNormal` isn't huge
    // this would require a normalization before the matrix multiplication, making everything slower/
    const float32_t3x3 normalMatrix = hlsl::math::linalg::truncate<3,3,3,4>(hlsl::transpose(float32_t4x3(spirv::WorldToObjectKHR)));
    // normalization also needs to be done in full floats because length squared can easily be over 64k
    return hlsl::normalize(hlsl::mul(normalMatrix,geometricNormal));
}
inline float32_t3 reconstructGeometricNormal(NBL_REF_ARG(spirv::HitObjectEXT) hitObject)
{
    using namespace nbl::hlsl;

    const float32_t3 vertices[3] = spirv::hitObjectGetIntersectionTriangleVertexPositionsEXT(hitObject);

    // Do diffs in high precision, edges can be very long and dot products can easily overflow 64k max float16_t value and normalizing one extra time makes no sense
    const float32_t3 geometricNormal = hlsl::cross(vertices[1]-vertices[0],vertices[2]-vertices[0]);
    // Scales can be absolutely huge, we'd need special per-instance pre-scaled 3x3 matrices and also guarantee `geometricNormal` isn't huge
    // this would require a normalization before the matrix multiplication, making everything slower
  
    // This is Inverse Transpose of ObjectToWorld matrix
    // Note that SPIR-V gives tranposed matrices already vs our row-major feeding SSBO and BDA, as well as contrary to maths (an affine matrix should be 3x4 not 4x3)
    const float32_t3x3 normalMatrix = hlsl::math::linalg::truncate<3,3,4,3>(spirv::hitObjectGetWorldToObjectEXT(hitObject));
    // normalization also needs to be done in full floats because length squared can easily be over 64k
    return hlsl::normalize(hlsl::mul(normalMatrix,geometricNormal));
}


// This is not only used for Russian Roulette but also for culling low throughput paths early (adds bias but keeps the critical path of the path tracer - pun intended - manageable)
struct MaxContributionEstimator
{
    // TODO: apply inverse exposure so we're sensitive to screen output (previous beauty), but don't go overkill and apply toonemapped luma derivative based on current inverse tonemapping of color accumulation
    static inline MaxContributionEstimator create(const float16_t3 constantThroughputWeights)
    {
        MaxContributionEstimator retval;
        // essentially how much can we move the accumulation needle
        retval.throughputWeights = constantThroughputWeights;
        return retval;
    }

    // notCulled instead of culled because of NaN handling
    inline bool surviveRussianRoulette(NBL_REF_ARG(float32_t3) throughput, bool skipRussianRoulette, NBL_REF_ARG(float32_t) xi)
    {
        // recompute after previous hit
        const float16_t surviveProb = hlsl::dot(float16_t3(throughput),throughputWeights);
        // TODO: prevent "fireflies in AoVs" because AoV targets are not HDR - don't do RR if that will overshoot our albedo and normal contributions
        // skipRussianRoulette = skipRussianRoulette && ...;
        // cull really low throughput paths (adds bias)
        const float16_t RelativeLumaThroughputThreshold = hlsl::numeric_limits<float16_t>::min;
        if (surviveProb<RelativeLumaThroughputThreshold)
            return false;
        if (skipRussianRoulette)
            return true;
        // < instead of <= very important for handling zero probability, note that nextULP correction doesn't need to be applied because we use unclamped probability here
        const bool retval = xi<surviveProb;
        {
            const float16_t UnityFp16 = 1;
            // now apply the clamp
            const float32_t rcpSurvivalProb = max(UnityFp16/surviveProb,UnityFp16);
            // rescale rand
            xi *= rcpSurvivalProb;
            // apply to throughput
            throughput *= rcpSurvivalProb;
        }
        return retval;
    }

    // The idea is that the throughput weights scale HDR world-referenced throughput into a Probability value
    float16_t3 throughputWeights;
};

// accumulated color
using accum_t = float16_t3;

//
struct SEnvSample
{
    accum_t color;
    SArbitraryOutputValues aov;
};
// tmp stuff
const static float32_t sunConeHalfAngleCos = 0.99999;
const static float32_t3 sunDir = normalize(float32_t3(1,1,1));
const static accum_t skyColor = accum_t(0.5f, 0.5f, 1.f);
const static accum_t sunColor = accum_t(10000, 10000, 10000);
//
SEnvSample sampleEnv(const float32_t3 raydir)
{
    SEnvSample retval;
    // TODO: sample the envmap texture
    retval.color = skyColor;
    if (hlsl::dot(raydir,sunDir)>sunConeHalfAngleCos)
        retval.color = sunColor;
    // TODO: apply some tonemapping operator with exposure (first envmap's avg luma, then our own)
    retval.aov.albedo = hlsl::min(retval.color,float16_t3(1,1,1));
    retval.aov.normal = -hlsl::normalize(float16_t3(raydir));
    return retval;
}

}
}
 

// TODO: should this be here?
using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::this_example;
using namespace nbl::hlsl::path_tracing;