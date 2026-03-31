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
	// TODO: RWMC accumulator where we can skip samples
//	template<typename T>
//	void accumulate(const vector<uint16_t, Dims> coord, const uint16_t layer, const vector<T, Dims> val, const float rcpNewSampleCount)
//	{
//	}

	LoadStoreImageAccessor composed;
};

//
template<typename T, int N>
vector<T,N> correctSNorm10WhenStoringToUnorm(const vector<T,N> input)
{
	using vec_t = vector<T,N>;
	return hlsl::mix(input*T(0.499512)+hlsl::promote<vec_t,T>(0.999022),input*T(0.499512),hlsl::promote<vec_t,T>(0.f)<input);
}

// sample count incrementing function
struct SPixelSamplingInfo
{
	hlsl::examples::KeyedQuantizedSequence<hlsl::Xoroshiro64Star> randgen;
	float32_t rcpNewSampleCount;
	uint16_t firstSample;
};
SPixelSamplingInfo advanceSampleCount(const uint16_t3 coord, const uint16_t newSamplesThisPixel, const uint16_t dontClear)
{
	SPixelSamplingInfo retval;
	// 
	const uint32_t sampleCount = gSampleCount[coord];
	retval.firstSample = uint16_t(sampleCount)*dontClear;
	// setup randgen
	{
		retval.randgen.pSampleBuffer = gScene.init.pSampleSequence;
		// TODO: experiment with storing every dimension scramble in the texture to not pollute the ray payload
		retval.randgen.rng = hlsl::Xoroshiro64Star::construct(gScrambleKey[uint16_t3(coord.xy & uint16_t(511), 0)]);
		retval.randgen.sequenceSamplesLog2 = gScene.init.sequenceSamplesLog2; // TODO: make this compile time constant - Spec Constant?
	}
	//
	const uint16_t newSampleCount = retval.firstSample+newSamplesThisPixel;
	gSampleCount[coord] = newSampleCount;
	retval.rcpNewSampleCount = hlsl::select(newSampleCount>retval.firstSample,1.f/float32_t(newSampleCount),0.f);
	return retval;
}

// raygen functions
// ..
}
}
 

// TODO: should this be here?
using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::this_example;
using namespace nbl::hlsl::path_tracing;