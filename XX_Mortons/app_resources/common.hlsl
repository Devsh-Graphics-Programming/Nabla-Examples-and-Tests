#include "nbl/builtin/hlsl/math/morton.hlsl"

NBL_CONSTEXPR uint32_t bufferSize = 256;
using scalar_t = int32_t;
using unsigned_scalar_t = nbl::hlsl::make_unsigned_t<scalar_t>;

struct PushConstantData
{
	uint64_t deviceBufferAddress;
};