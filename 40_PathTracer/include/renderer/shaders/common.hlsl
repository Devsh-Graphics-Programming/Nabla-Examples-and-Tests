#ifndef _NBL_THIS_EXAMPLE_COMMON_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_COMMON_HLSL_INCLUDED_


#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/concepts/accessors/loadable_image.hlsl"
#include "nbl/builtin/hlsl/concepts/accessors/storable_image.hlsl"
//
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
//
#include "nbl/builtin/hlsl/path_tracing/gaussian_filter.hlsl"


// TODO: move to type_traits?
namespace nbl
{
namespace hlsl
{
#ifdef __HLSL_VERSION
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct texture_traits;

template<typename T, int32_t N>
struct texture_traits<Texture2DArray<vector<T,N> > >
{
	NBL_CONSTEXPR_STATIC_INLINE int32_t Dimension = 2;
	using coded_type = vector<T,N>;
};
//special case
template<typename T> NBL_PARTIAL_REQ_TOP(is_scalar_v<T>)
struct texture_traits<Texture2DArray<T> NBL_PARTIAL_REQ_BOT(is_scalar_v<T>)>
{
	NBL_CONSTEXPR_STATIC_INLINE int32_t Dimension = 1;
	using coded_type = vector<T,1>;
};

template<typename T, int32_t N>
struct texture_traits<RWTexture2DArray<vector<T,N> > >
{
	NBL_CONSTEXPR_STATIC_INLINE int32_t Dimension = 2;
	using coded_type = vector<T,N>;
};
//special case
template<typename T> NBL_PARTIAL_REQ_TOP(is_scalar_v<T>)
struct texture_traits<RWTexture2DArray<T> NBL_PARTIAL_REQ_BOT(is_scalar_v<T>)>
{
	NBL_CONSTEXPR_STATIC_INLINE int32_t Dimension = 1;
	using coded_type = vector<T,1>;
};
#endif
}
}

//
#define DEFINE_TEXTURE_ACCESSOR(TEX_NAME) struct ImageAccessor_ ## TEX_NAME \
{ \
	using texture_traits_t = ::nbl::hlsl::texture_traits<decltype(TEX_NAME)>; \
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Dimension = texture_traits_t::Dimension; \
	using coded_type = typename texture_traits_t::coded_type; \
	using coded_type_traits = ::nbl::hlsl::vector_traits<coded_type>; \
	using scalar_type = typename coded_type_traits::scalar_type; \
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Components = coded_type_traits::Dimension; \
\
	template<typename T, int32_t Dims NBL_FUNC_REQUIRES(Dims==Dimension) \
	void get(NBL_REF_ARG(vector<T,Components>) value, const vector<uint16_t,Dims> coord, const uint16_t layer) \
	{ \
		value = TEX_NAME[vector<uint16_t,Dims+1>(coord,layer)]; \
	} \
	template<typename T, int32_t Dims NBL_FUNC_REQUIRES(Dims==Dimension) \
	void set(const vector<uint16_t,Dims> coord, const uint16_t layer, const vector<T,Components> value) \
	{ \
		TEX_NAME[vector<uint16_t,Dims+1>(coord,layer)] = value; \
	} \
}

//
#define MAX_PATH_DEPTH_LOG2 8

#endif  // _NBL_THIS_EXAMPLE_COMMON_HLSL_INCLUDED_
