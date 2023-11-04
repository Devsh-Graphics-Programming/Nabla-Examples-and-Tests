#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"

template<uint32_t kScanElementCount=1024*1024>
struct Output
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t ScanElementCount = kScanElementCount;

	uint32_t subgroupSize;
	uint32_t data[ScanElementCount];
};

// Thanks to our unified HLSL/C++ STD lib we're able to remove a whole load of code
template<typename T>
struct bit_and : nbl::hlsl::bit_and<T>
{
	using base_t = nbl::hlsl::bit_and<T>;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t BindingIndex = 0;
#ifndef __HLSL_VERSION
	static inline constexpr const char* name = "bit_and";
#endif
};
template<typename T>
struct bit_or : nbl::hlsl::bit_or<T>
{
	using base_t = nbl::hlsl::bit_or<T>;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t BindingIndex = 1;
#ifndef __HLSL_VERSION
	static inline constexpr const char* name = "bit_xor";
#endif
};
template<typename T>
struct bit_xor : nbl::hlsl::bit_xor<T>
{
	using base_t = nbl::hlsl::bit_xor<T>;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t BindingIndex = 2;
#ifndef __HLSL_VERSION
	static inline constexpr const char* name = "bit_or";
#endif
};
template<typename T>
struct plus : nbl::hlsl::plus<T>
{
	using base_t = nbl::hlsl::plus<T>;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t BindingIndex = 3;
#ifndef __HLSL_VERSION
	static inline constexpr const char* name = "plus";
#endif
};
template<typename T>
struct multiplies : nbl::hlsl::multiplies<T>
{
	using base_t = nbl::hlsl::multiplies<T>;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t BindingIndex = 4;
#ifndef __HLSL_VERSION
	static inline constexpr const char* name = "multiplies";
#endif
};
template<typename T>
struct minimum : nbl::hlsl::minimum<T>
{
	using base_t = nbl::hlsl::minimum<T>;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t BindingIndex = 5;
#ifndef __HLSL_VERSION
	static inline constexpr const char* name = "minimum";
#endif
};
template<typename T>
struct maximum : nbl::hlsl::maximum<T>
{
	using base_t = nbl::hlsl::maximum<T>;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t BindingIndex = 6;
#ifndef __HLSL_VERSION
	static inline constexpr const char* name = "maximum";
#endif
};

template<typename T>
struct ballot : nbl::hlsl::plus<T>
{
	using base_t = nbl::hlsl::plus<T>;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t BindingIndex = 7;
#ifndef __HLSL_VERSION
	static inline constexpr const char* name = "bitcount";
#endif
};

#include "nbl/builtin/hlsl/subgroup/basic.hlsl"