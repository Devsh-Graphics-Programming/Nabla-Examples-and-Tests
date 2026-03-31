#include "renderer/shaders/pathtrace/common.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"

#include "nbl/examples/common/KeyedQuantizedSequence.hlsl"


namespace nbl
{
namespace this_example
{

// accumulators
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
	void accumulate(const vector<uint16_t,Dimension> coord, const uint16_t layer, const vector<T,ComponentOverride> data, const T rcpNewSampleCount)
	{
		coded_type val;

		if (rcpNewSampleCount<1.f)
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
	const uint16_t newSampleCount = retval.firstSample+newSamplesThisPixel;
	gSampleCount[coord] = newSampleCount;
	retval.rcpNewSampleCount = hlsl::select(newSampleCount>retval.firstSample,1.f/float32_t(newSampleCount),0.f);
	return retval;
}

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

}
}
 

// TODO: should this be here?
using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::this_example;
using namespace nbl::hlsl::path_tracing;