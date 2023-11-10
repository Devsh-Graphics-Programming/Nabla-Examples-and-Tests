// This is a magical header that provides most HLSL types and intrinsics in C++
//#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#ifdef __HLSL_VERSION
#define NBL_CONSTEXPR static const
#else
#define NBL_CONSTEXPR constexpr
#endif


NBL_CONSTEXPR uint32_t WorkgroupSize = 256;