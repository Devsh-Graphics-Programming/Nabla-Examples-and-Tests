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

	template<typename T, int ComponentOverride NBL_FUNC_REQUIRES(hlsl::is_same_v<typename LoadStoreImageAccessor::scalar_type,T> && ComponentOverride<=LoadStoreImageAccessor::Components)
	void accumulate(const vector<uint16_t,Dimension> coord, const uint16_t layer, const vector<T,ComponentOverride> data, const float rcpNewSampleCount)
	{
		coded_type val;

		if (rcpNewSampleCount<1.f)
		{
			composed.template get<T,Dimension>(val,coord,layer);
			NBL_UNROLL for (uint16_t i=0; i<ComponentOverride; i++)
				val[i] += (data[i] - val[i]) * rcpNewSampleCount;
			// don't threshold the store, most threads will store, just adding extra if-statement. Comeback to it when we have very uniform AoV views to optimize export there
			//if (hlsl::all(hlsl::truncate<delctype(val)>(hlsl::abs(delta) < LoadStoreImageAccessor::QuantizationThreshold)))
				// return;
		}
		else // clear path
		NBL_UNROLL for (uint16_t i=0; i<LoadStoreImageAccessor::Components; i++)
			val[i] = hlsl::select(i<ComponentOverride,data[i],T(0));

		composed.template set<T,Dimension>(coord,layer,val);
	}
	// TODO: RWMC accumulator where we can skip samples
//	template<typename T>
//	void accumulate(const vector<uint16_t, Dims> coord, const uint16_t layer, const vector<T, Dims> val, const float rcpNewSampleCount)
//	{
//	}

	LoadStoreImageAccessor composed;
};

// raygen functions
// ..
}
}
 

// TODO: should this be here?
using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::this_example;
using namespace nbl::hlsl::path_tracing;