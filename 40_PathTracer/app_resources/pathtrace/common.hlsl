#include "renderer/shaders/pathtrace/common.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"

#include "nbl/examples/common/KeyedQuantizedSequence.hlsl"


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
	return hlsl::mix(input*T(0.499512)+hlsl::promote<vec_t,T>(0.999022),input*T(0.499512),hlsl::promote<vec_t,T>(0.f)<input);
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
SPixelSamplingInfo advanceSampleCount(const uint16_t3 coord, const uint16_t newSamplesThisPixel, const uint16_t dontClear)
{
	SPixelSamplingInfo retval;
	// 
	retval.firstSample = uint16_t(gSampleCount[coord])*dontClear;
	// setup randgen
	{
		retval.randgen.pSampleBuffer = gScene.init.pSampleSequence;
		// TODO: experiment with storing every dimension scramble in the texture to not pollute the ray payload
		retval.randgen.rng = scramble_state_t::construct(gScrambleKey[uint16_t3(coord.xy & uint16_t(511), 0)]);
		retval.randgen.sequenceSamplesLog2 = gScene.init.sequenceSamplesLog2; // TODO: make this compile time constant - Spec Constant?
	}
	//
	retval.newSampleCount = retval.firstSample+newSamplesThisPixel;
    if (newSamplesThisPixel!=0 || dontClear==0) // whether this pays off depends on ratio of pixels with 0 spp in a dispatch
    	gSampleCount[coord] = retval.newSampleCount;
	// handle overflow properly
	retval.rcpNewSampleCount = hlsl::select(retval.newSampleCount>retval.firstSample,1.f/float32_t(retval.newSampleCount),0.f);
	return retval;
}

// TODO: split into RayDir 
// raygen functions
struct SRay
{
	static SRay create(const SSensorDynamics sensor, const float32_t2 pixelSizeNDC, const float32_t2 ndc, const float16_t2 xi)
	{
		using namespace nbl::hlsl;
		using namespace nbl::hlsl::math::linalg;

        // stochastic reconstruction filter
		const float16_t stddev = float16_t(1.2);
        const float32_t3 adjNDC = float32_t3(path_tracing::GaussianFilter<float16_t>::create(stddev,stddev).sample(xi)*pixelSizeNDC+ndc,-1.f);
        // unproject
        const float32_t3 direction = hlsl::normalize(float32_t3(hlsl::mul(sensor.ndcToRay,adjNDC), -1.0));
        const float32_t3 origin = -float32_t3(direction.xy/direction.z,sensor.nearClip);
		// rotate with camera
		SRay retval;
		retval.origin = promoted_mul(sensor.invView,origin);
		retval.tMin = sensor.nearClip;
		retval.direction = hlsl::normalize(hlsl::mul(truncate<3,3,3,4>(sensor.invView),direction));
		retval.tMax = sensor.tMax;
		return retval;
	}

	float32_t3 origin;
	float32_t tMin;
	float32_t3 direction;
	float32_t tMax;
	// TODO: ray differentials or covariance
};

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

// TODO: use the CIE stuff
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR float16_t3 LumaConversionCoeffs = float16_t3(0.39,0.5,0.11);

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
        retval.normal = normal*hlsl::dot(throughput.albedo,LumaConversionCoeffs);
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

// accumulated color
using accum_t = float16_t3;

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
    inline bool notCulled(NBL_REF_ARG(float32_t3) throughput, bool skipRussianRoulette, NBL_REF_ARG(float32_t) xi)
    {
        // recompute after previous hit
        const float16_t surviveProb = hlsl::dot(float16_t3(throughput),throughputWeights);
        // TODO: prevent "fireflies in AoVs" because AoV targets are not HDR - don't do RR if that will overshoot our albedo and normal contributions
        // skipRussianRoulette = skipRussianRoulette && ...;
        // cull really low throughput paths (adds bias)
        const float16_t RelativeLumaThroughputThreshold = hlsl::numeric_limits<float16_t>::min;
        // < instead of <= very important for handling zero probability, note that nextULP correction doesn't need to be applied because we use unclamped probability here
        if (surviveProb>RelativeLumaThroughputThreshold && (skipRussianRoulette || xi<surviveProb))
        {
            const float16_t UnityFp16 = 1;
            // now apply the clamp
            const float32_t rcpSurvivalProb = max(UnityFp16/surviveProb,UnityFp16);
            // rescale rand
            xi *= rcpSurvivalProb;
            // apply to throughput
            throughput *= rcpSurvivalProb;
            return true;
        }
        return false;
    }

    // The idea is that the throughput weights scale HDR world-referenced throughput into a Probability value
    float16_t3 throughputWeights;
};

//
struct SEnvSample
{
    accum_t color;
    SArbitraryOutputValues aov;
};
SEnvSample sampleEnv(const float32_t3 raydir)
{
    SEnvSample retval;
    // TODO: sample the envmap texture
    retval.color = float16_t3(0.5f,0.5f,1.f);
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